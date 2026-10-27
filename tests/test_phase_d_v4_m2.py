"""M2 deterministic runner, persistence, timing, resume, and CLI tests."""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import random
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

from gradient_phase_d_resnet_v4 import (
    HistoricalFeedbackController,
    _synthetic_fixture,
    build_models,
    build_optimizers,
    configure_determinism,
    evaluate,
    parse_args,
    resume_run,
    run_epoch,
    run_training,
    set_training_mode,
)
from phase_d_v4_io import (
    SCHEMA_VERSION,
    MethodId,
    NormalizationMode,
    RunConfig,
    SyntheticBehavior,
    TimingLedger,
    capture_rng_state,
    copy_checkpoint_verified,
    latest_committed_checkpoint,
    load_checkpoint,
    load_run_config,
    reconcile_outputs,
    save_checkpoint,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "phase_d_v4_reference.json"
PYTHON = Path(r"C:\Users\Anupam Dasgupta\AppData\Local\Programs\Python\Python313\python.exe")


class RandomizedTinyDataset(Dataset):
    """Produce tiny images with global-RNG augmentation to exercise replay seeding."""

    def __init__(self, size: int, seed: int) -> None:
        """Create fixed base tensors; item access later consumes Torch RNG."""

        generator = torch.Generator().manual_seed(seed)
        self.inputs = torch.randn(size, 3, 32, 32, generator=generator)
        self.labels = torch.randint(0, 10, (size,), generator=generator)

    def __len__(self) -> int:
        """Return the fixed number of synthetic examples without RNG changes."""

        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one image with deterministic-per-epoch random noise and its label."""

        noise = torch.rand((), dtype=self.inputs.dtype) * 0.01
        return self.inputs[index] + noise, self.labels[index]


class FailAfterDataset(Dataset):
    """Raise after a fixed number of sample reads to simulate an interrupted epoch."""

    def __init__(self, dataset: Dataset, successful_reads: int) -> None:
        """Wrap a dataset and set the number of item reads allowed before failure."""

        self.dataset = dataset
        self.successful_reads = successful_reads
        self.reads = 0

    def __len__(self) -> int:
        """Return the wrapped length without consuming an item."""

        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        """Return an item until the configured interruption point, then raise."""

        if self.reads >= self.successful_reads:
            raise RuntimeError("simulated mid-epoch interruption")
        self.reads += 1
        return self.dataset[index]


def _runner_config(
    run_directory: Path,
    *,
    method: MethodId = MethodId.SCHEDULED,
    warmup_epochs: int = 1,
    postwarm_epochs: int = 2,
    batch_size: int = 4,
) -> RunConfig:
    """Build one explicit tiny-run configuration with a unique output identity."""

    values = load_run_config(CONFIG_PATH).to_dict()
    identity = values["scientific"]["identity"]
    behavior = SyntheticBehavior.EXACT if method in {MethodId.EXACT, MethodId.EXACT_STANDARD_BN} else SyntheticBehavior.LEARNED
    normalization = NormalizationMode.STANDARD_BN if method == MethodId.EXACT_STANDARD_BN else NormalizationMode.FROZEN_HEAD_POSTWARM
    identity.update(
        method=method.value,
        algorithm_version="v3" if method in {
            MethodId.EXACT,
            MethodId.SCHEDULED,
            MethodId.RELATIVE_FEEDBACK,
            MethodId.ABSOLUTE_FEEDBACK,
        } else "v4",
        normalization=normalization.value,
        synthetic_behavior=behavior.value,
        mask_role="none",
        mask_sha256=None,
        retrospective=False,
        deployment_eligible=True,
        classifier_seed=2,
        mask_replicate=None,
    )
    values["scientific"]["model"].update(width=4, blocks_per_stage=1, boundary_shape=[None, 8, 16, 16])
    values["scientific"]["predictor"].update(input_channels=8, bottleneck_width=4)
    values["scientific"]["training"].update(
        warmup_epochs=warmup_epochs,
        postwarm_epochs=postwarm_epochs,
        total_epochs=warmup_epochs + postwarm_epochs,
        batch_size=batch_size,
        expected_training_batches_per_epoch=3,
        expected_final_batch_size=2,
    )
    values["execution"].update(workers=0, pin_memory=False)
    values["output"].update(
        root=str(run_directory),
        study_id="m2_test",
        run_id=run_directory.name,
    )
    return RunConfig(values)


def _identities() -> dict[str, str]:
    """Return stable fake materialized split hashes for isolated engineering data."""

    return {
        "training_indices_sha256": hashlib.sha256(b"m2-training-indices").hexdigest(),
        "validation_indices_sha256": hashlib.sha256(b"m2-validation-indices").hexdigest(),
    }


def _artifacts() -> dict[str, str]:
    """Return the real immutable v3 source identity used by test checkpoints."""

    return {"v3_source": sha256_file(PROJECT_ROOT / "gradient_phase_d_resnet_v3.py")}


def _calibration() -> dict[str, Any]:
    """Return a high-error history that deterministically grants quota four."""

    return {
        "relative_errors": [1.0e9] * 12,
        "q_abs_low": 1.0e9,
        "q_abs_high": 2.0e9,
        "median_abs": 1.5e9,
    }


def _assert_tensor_tree_equal(case: unittest.TestCase, left: Any, right: Any) -> None:
    """Recursively compare tensors and primitive containers exactly."""

    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        case.assertEqual(set(left), set(right))
        for key in left:
            _assert_tensor_tree_equal(case, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        case.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right):
            _assert_tensor_tree_equal(case, left_item, right_item)
    else:
        case.assertEqual(left, right)


def _scientific_epoch_rows(context: dict[str, Any]) -> list[dict[str, Any]]:
    """Strip elapsed/checkpoint fields and return comparable committed epoch metrics."""

    excluded = {
        "run_id", "training_seconds", "validation_seconds",
        "observation_time_seconds", "checkpoint_sha256",
    }
    return [
        {key: value for key, value in row.items() if key not in excluded}
        for row in context["state"].epoch_records
    ]


class TimingAndIsolationTests(unittest.TestCase):
    """Verify ledger formulas, validation isolation, and CLI command boundaries."""

    def test_timing_ledger_formulas_do_not_double_count(self) -> None:
        """Synthetic durations produce the exact planned timing definitions."""

        ledger = TimingLedger()
        values = {
            "setup": 1.0,
            "scale_estimation": 2.0,
            "warmup_training": 3.0,
            "warmup_validation": 4.0,
            "calibration": 5.0,
            "postwarm_training": 6.0,
            "postwarm_validation": 7.0,
            "checkpoint_log_io": 8.0,
            "finalization": 9.0,
            "diagnostics": 10.0,
        }
        for component, seconds in values.items():
            ledger.add(component, seconds)
        self.assertEqual(
            ledger.totals(),
            {
                "learning_seconds": 16.0,
                "evaluated_seconds": 27.0,
                "workflow_seconds": 55.0,
                "postwarm_seconds": 13.0,
                "postwarm_training_seconds": 6.0,
            },
        )

    def test_validation_restores_modes_buffers_and_rng(self) -> None:
        """Evaluation returns metrics while leaving production state and RNG unchanged."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            config = _runner_config(Path(temporary) / "run", method=MethodId.SCHEDULED)
            configure_determinism(config, "cpu")
            torch.manual_seed(77)
            model, predictor = build_models(config, "cpu")
            set_training_mode(model, predictor, "postwarm", NormalizationMode.FROZEN_HEAD_POSTWARM)
            validation = TensorDataset(torch.randn(5, 3, 32, 32), torch.arange(5) % 10)
            before_model = copy.deepcopy(model.state_dict())
            before_flags = [module.training for module in model.modules()]
            before_rng = capture_rng_state()
            metrics = evaluate(model, validation, config, "cpu")
            self.assertEqual(metrics["samples"], 5)
            _assert_tensor_tree_equal(self, before_model, model.state_dict())
            self.assertEqual(before_flags, [module.training for module in model.modules()])
            _assert_tensor_tree_equal(self, before_rng, capture_rng_state())

    def test_cli_exposes_engineering_and_explicit_scientific_modes(self) -> None:
        """The parser requires full inputs for scientific modes without launching them."""

        for command in ("audit", "smoke", "resume", "profile", "run", "diagnose"):
            arguments = [command]
            if command in {"smoke", "resume", "profile", "diagnose"}:
                arguments += ["--out", str(PROJECT_ROOT / "runs_phase_d_v4" / "parser-only")]
            if command == "run":
                arguments += [
                    "--config", str(CONFIG_PATH),
                    "--data-dir", str(PROJECT_ROOT / "data"),
                    "--split-file", str(PROJECT_ROOT / "parser-split.npz"),
                    "--out", str(PROJECT_ROOT / "runs_phase_d_v4" / "parser-only"),
                ]
            parsed = parse_args(arguments)
            self.assertEqual(parsed.command, command)
            self.assertFalse(hasattr(parsed, "unavailable_reason"))


class ResumeEquivalenceTests(unittest.TestCase):
    """Verify deterministic fresh/resumed state and quota continuity."""

    def test_uninterrupted_equals_warmup_interrupted_resume(self) -> None:
        """A warm-up boundary resume reproduces model, optimizer, metrics, data, and RNG."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            base = Path(temporary)
            train = RandomizedTinyDataset(10, 101)
            validation = TensorDataset(torch.randn(6, 3, 32, 32), torch.arange(6) % 10)
            full_dir, resumed_dir = base / "full", base / "resumed"
            full_config = _runner_config(full_dir)
            resumed_config = _runner_config(resumed_dir)
            full = run_training(
                full_config, train, validation, full_dir, "cpu",
                artifacts=_artifacts(), data_identities=_identities(),
            )
            interrupted = run_training(
                resumed_config, train, validation, resumed_dir, "cpu",
                artifacts=_artifacts(), data_identities=_identities(), stop_after_global_epoch=0,
            )
            checkpoint = interrupted["last_checkpoint"]
            with self.assertRaisesRegex(RuntimeError, "mid-epoch interruption"):
                run_epoch(
                    interrupted["model"],
                    interrupted["predictor"],
                    interrupted["optimizers"],
                    FailAfterDataset(train, 4),
                    resumed_config,
                    "cpu",
                    interrupted["state"],
                    interrupted["ledger"],
                    global_epoch=1,
                    controller=interrupted["controller"],
                )
            resumed = resume_run(
                resumed_config, train, validation, resumed_dir, "cpu",
                artifacts=_artifacts(), data_identities=_identities(), checkpoint_path=checkpoint,
            )
            _assert_tensor_tree_equal(self, full["model"].state_dict(), resumed["model"].state_dict())
            _assert_tensor_tree_equal(self, full["predictor"].state_dict(), resumed["predictor"].state_dict())
            for name in ("trunk", "head", "predictor"):
                _assert_tensor_tree_equal(
                    self,
                    full["optimizers"][name].state_dict(),
                    resumed["optimizers"][name].state_dict(),
                )
            self.assertEqual(_scientific_epoch_rows(full), _scientific_epoch_rows(resumed))
            self.assertEqual(full["state"].data_order_hashes, resumed["state"].data_order_hashes)
            self.assertEqual(full["state"].executed_exact_mask, resumed["state"].executed_exact_mask)
            full_payload = load_checkpoint(
                full["last_checkpoint"], journal_path=full_dir / "checkpoint_journal.json"
            )
            resumed_payload = load_checkpoint(
                resumed["last_checkpoint"], journal_path=resumed_dir / "checkpoint_journal.json"
            )
            _assert_tensor_tree_equal(self, full_payload["rng_state"], resumed_payload["rng_state"])
            self.assertEqual(len(resumed["state"].interruptions), 1)
            self.assertFalse(resumed["state"].interruptions[0]["mid_epoch_resume"])

    def test_feedback_quota_crosses_epoch_boundary_and_resumes(self) -> None:
        """Pending feedback quota persists through checkpoint and drives next epoch synthetic."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            run_dir = Path(temporary) / "feedback"
            full_dir = Path(temporary) / "feedback_full"
            config = _runner_config(
                run_dir,
                method=MethodId.RELATIVE_FEEDBACK,
                warmup_epochs=1,
                postwarm_epochs=2,
                batch_size=4,
            )
            train = RandomizedTinyDataset(4, 102)
            validation = TensorDataset(torch.randn(4, 3, 32, 32), torch.arange(4) % 10)
            full_config = _runner_config(
                full_dir,
                method=MethodId.RELATIVE_FEEDBACK,
                warmup_epochs=1,
                postwarm_epochs=2,
                batch_size=4,
            )
            full = run_training(
                full_config, train, validation, full_dir, "cpu",
                artifacts=_artifacts(), data_identities=_identities(), calibration=_calibration(),
            )
            partial = run_training(
                config, train, validation, run_dir, "cpu",
                artifacts=_artifacts(), data_identities=_identities(),
                calibration=_calibration(), stop_after_global_epoch=1,
            )
            self.assertEqual(partial["state"].pending_quota, 4)
            payload = load_checkpoint(
                partial["last_checkpoint"], journal_path=run_dir / "checkpoint_journal.json"
            )
            self.assertEqual(payload["pending_quota"], 4)
            required = {
                "schema_version", "scientific_fingerprint", "execution_fingerprint", "artifacts",
                "data_identities", "resolved_config", "seed", "method", "normalization",
                "stage", "completed_epoch", "next_epoch", "model", "predictor", "optimizers",
                "scale", "lr_global_epoch", "controller", "pending_quota", "mask_role",
                "source_mask_sha256", "executed_mask_sha256", "run_state", "rng_state",
                "timing_ledger", "environment", "run_complete", "checkpoint_sequence",
            }
            self.assertTrue(required.issubset(payload))
            resumed = resume_run(
                config, train, validation, run_dir, "cpu",
                artifacts=_artifacts(), data_identities=_identities(),
                calibration=_calibration(), checkpoint_path=partial["last_checkpoint"],
            )
            self.assertEqual(resumed["state"].executed_exact_mask, [True, False])
            self.assertEqual(resumed["state"].pending_quota, 3)
            self.assertEqual(resumed["state"].controller_events[0]["synthetic_quota"], 4)
            self.assertEqual(full["controller"].state_dict(), resumed["controller"].state_dict())
            _assert_tensor_tree_equal(self, full["model"].state_dict(), resumed["model"].state_dict())


class PersistenceFailureTests(unittest.TestCase):
    """Verify corruption, identity, commit, reconciliation, and remote-copy handling."""

    def _one_epoch_run(self, root: Path) -> tuple[RunConfig, dict[str, Any]]:
        """Create and return one committed tiny warm-up boundary under a test root."""

        run_dir = root / "run"
        config = _runner_config(run_dir, method=MethodId.SCHEDULED)
        train, validation, identities = _synthetic_fixture(seed=123)
        context = run_training(
            config, train, validation, run_dir, "cpu",
            artifacts=_artifacts(), data_identities=identities, stop_after_global_epoch=0,
        )
        return config, {"context": context, "train": train, "validation": validation, "identities": identities}

    def test_corruption_and_identity_mismatches_fail_before_restore(self) -> None:
        """Checksum, source, mask, and data identity mismatches are rejected loudly."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            config, fixture = self._one_epoch_run(root)
            checkpoint = fixture["context"]["last_checkpoint"]
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                load_checkpoint(checkpoint, expected={"data_identities": {"wrong": "f" * 64}})
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                load_checkpoint(checkpoint, expected={"scientific_fingerprint": "f" * 64})
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                load_checkpoint(checkpoint, expected={"source_mask_sha256": "f" * 64})
            with checkpoint.open("r+b") as handle:
                handle.seek(20)
                byte = handle.read(1)
                handle.seek(20)
                handle.write(bytes([byte[0] ^ 1]))
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_checkpoint(checkpoint)

    def test_partial_commit_is_ignored_and_missing_journal_is_flagged(self) -> None:
        """Uncommitted files are ignored while committed files expose missing I/O time."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            directory = Path(temporary)
            partial = directory / "checkpoints" / "checkpoint_000001.pt"
            payload = {"schema_version": SCHEMA_VERSION, "checkpoint_sequence": 1}
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                save_checkpoint(partial, payload, simulate_failure="after_checkpoint")
            with self.assertRaises(FileNotFoundError):
                latest_committed_checkpoint(partial.parent)

            committed = directory / "checkpoints" / "checkpoint_000002.pt"
            with self.assertRaisesRegex(RuntimeError, "simulated"):
                save_checkpoint(committed, {**payload, "checkpoint_sequence": 2}, simulate_failure="before_journal")
            loaded = load_checkpoint(committed, journal_path=directory / "missing_journal.json")
            self.assertTrue(loaded["resume_metadata"]["timing_incomplete"])

    def test_reconciliation_removes_duplicate_uncommitted_rows(self) -> None:
        """Checkpoint records replace duplicated CSV rows after an interrupted output commit."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            _config, fixture = self._one_epoch_run(root)
            run_dir = root / "run"
            checkpoint = fixture["context"]["last_checkpoint"]
            payload = load_checkpoint(
                checkpoint, journal_path=run_dir / "checkpoint_journal.json"
            )
            epochs_path = run_dir / "epochs.csv"
            original = epochs_path.read_text(encoding="utf-8")
            with epochs_path.open("a", encoding="utf-8") as handle:
                handle.write(original.splitlines()[-1] + "\n")
            reconcile_outputs(run_dir, payload)
            with epochs_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 1)

    def test_verified_remote_fixture_publishes_marker_last(self) -> None:
        """Local mock persistence copies immutable bytes and supports committed discovery."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            _config, fixture = self._one_epoch_run(root)
            checkpoint = fixture["context"]["last_checkpoint"]
            remote = root / "mock_drive"
            copied = copy_checkpoint_verified(checkpoint, remote)
            self.assertEqual(sha256_file(checkpoint), sha256_file(copied))
            self.assertTrue(copied.with_suffix(copied.suffix + ".commit.json").is_file())
            self.assertEqual(latest_committed_checkpoint(remote), copied)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class CudaSmokeTests(unittest.TestCase):
    """Exercise one committed tiny CUDA warm-up without scientific benchmarking."""

    def test_cuda_tiny_checkpoint_and_validation(self) -> None:
        """A one-epoch synthetic CUDA run commits a loadable checkpoint."""

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            run_dir = Path(temporary) / "cuda"
            config = _runner_config(run_dir, method=MethodId.EXACT)
            train, validation, identities = _synthetic_fixture(seed=222)
            context = run_training(
                config, train, validation, run_dir, "cuda",
                artifacts=_artifacts(), data_identities=identities, stop_after_global_epoch=0,
            )
            loaded = load_checkpoint(
                context["last_checkpoint"], journal_path=run_dir / "checkpoint_journal.json"
            )
            self.assertEqual(loaded["environment"]["device_type"], "cuda")
            self.assertFalse(loaded["resume_metadata"]["timing_incomplete"])


if __name__ == "__main__":
    unittest.main()
