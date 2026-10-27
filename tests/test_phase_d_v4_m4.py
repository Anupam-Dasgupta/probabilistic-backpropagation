"""Synthetic fixed-mask replay tests for Phase D v4 milestone M4."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset

from gradient_phase_d_resnet_v4 import (
    check_replay_equality,
    reference_config,
    resume_run,
    run_training,
)
from phase_d_v4_analysis import build_predictor_replay_table
from phase_d_v4_io import (
    METHOD_SPECS,
    MethodId,
    RunConfig,
    load_checkpoint,
    mask_identity_hash,
    sha256_file,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_SOURCE = PROJECT_ROOT / "gradient_phase_d_resnet_v3.py"


class ReplayDataset(Dataset):
    """Provide deterministic base tensors with augmentation from Torch's epoch RNG."""

    def __init__(self, size: int = 9, seed: int = 4401) -> None:
        """Create ``size`` synthetic images and labels from a private generator.

        Base tensors are retained in memory. Construction consumes only the
        private generator and has no file, network, or global-RNG side effects.
        """

        generator = torch.Generator().manual_seed(seed)
        self.inputs = torch.randn(size, 3, 32, 32, generator=generator)
        self.labels = torch.randint(0, 10, (size,), generator=generator)

    def __len__(self) -> int:
        """Return the number of synthetic observations without side effects."""

        return int(self.labels.shape[0])

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one randomly augmented image and label for the supplied index.

        The image receives scalar noise from the globally seeded per-epoch Torch
        RNG, intentionally exercising augmentation replay. No stored tensor is
        modified; the global RNG advances by one draw.
        """

        return self.inputs[index] + torch.rand((), dtype=self.inputs.dtype) * 0.01, self.labels[index]


def _identities() -> dict[str, str]:
    """Return fixed synthetic split identities without reading dataset content."""

    return {
        "training_indices_sha256": hashlib.sha256(b"m4-train").hexdigest(),
        "validation_indices_sha256": hashlib.sha256(b"m4-validation").hexdigest(),
    }


def _artifacts() -> dict[str, str]:
    """Return the immutable v3 source identity used by synthetic checkpoints."""

    return {"v3_source": sha256_file(PYTHON_SOURCE)}


def _calibration() -> dict[str, Any]:
    """Return a history that grants long quotas and crosses an epoch boundary."""

    return {
        "relative_errors": [1.0e9] * 12,
        "q_abs_low": 1.0e9,
        "q_abs_high": 2.0e9,
        "median_abs": 1.5e9,
    }


def _base_config(root: Path, run_id: str, method: MethodId) -> RunConfig:
    """Build a resolved tiny config for one donor or replay engineering run.

    Inputs select an output root/run ID and method. The returned config has one
    warm-up plus two post-warm-up epochs, width four, batch size four, seed zero,
    and method-correct identity fields. It does not validate or write files.
    """

    values = reference_config().to_dict()
    spec = METHOD_SPECS[method]
    behavior = sorted(item.value for item in spec["behaviors"])[0]
    values["scientific"]["identity"].update(
        classifier_seed=0,
        mask_replicate=None,
        method=method.value,
        algorithm_version=spec["version"],
        normalization=spec["normalization"].value,
        synthetic_behavior=behavior,
        mask_role=spec["mask_role"].value,
        mask_sha256=None,
        retrospective=spec["retrospective"],
        deployment_eligible=spec["deployment_eligible"],
    )
    values["scientific"]["model"].update(
        width=4, blocks_per_stage=1, boundary_shape=[None, 8, 16, 16]
    )
    values["scientific"]["predictor"].update(input_channels=8, bottleneck_width=4)
    values["scientific"]["training"].update(
        warmup_epochs=1,
        postwarm_epochs=2,
        total_epochs=3,
        batch_size=4,
        expected_training_batches_per_epoch=3,
        expected_final_batch_size=1,
    )
    values["execution"].update(workers=0, pin_memory=False)
    values["output"].update(root=str(root), study_id="m4_engineering", run_id=run_id)
    return RunConfig(values)


def _replay_config(
    root: Path,
    run_id: str,
    method: MethodId,
    donor_manifest: dict[str, Any],
) -> RunConfig:
    """Build a replay config bound to one verified donor export identity.

    Inputs are local output identity, replay method, and donor mask manifest.
    The returned scientific config includes the source hash/run/checkpoint/seed/
    method/fingerprint and mandatory retrospective labels; no file is changed.
    """

    config = _base_config(root, run_id, method).to_dict()
    export = donor_manifest["replay_export"]
    config["scientific"]["identity"]["mask_sha256"] = export["sha256"]
    config["scientific"]["replay_source"] = {
        "run_id": export["source_run_id"],
        "checkpoint_sha256": export["source_checkpoint_sha256"],
        "seed": export["source_seed"],
        "method": export["source_method"],
        "scientific_fingerprint": export["source_scientific_fingerprint"],
    }
    return RunConfig(config)


def _assert_tree_equal(case: unittest.TestCase, left: Any, right: Any) -> None:
    """Recursively compare tensors and nested optimizer/model state exactly.

    Inputs are a test case and two state trees. Nothing is returned or mutated;
    mismatch raises with bitwise tensor tolerances.
    """

    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        case.assertEqual(left.keys(), right.keys())
        for key in left:
            _assert_tree_equal(case, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        case.assertEqual(len(left), len(right))
        for a, b in zip(left, right, strict=True):
            _assert_tree_equal(case, a, b)
    else:
        case.assertEqual(left, right)


class ReplayFixture(unittest.TestCase):
    """Create one bounded donor and expose helpers for replay conditions."""

    temporary: tempfile.TemporaryDirectory[str]
    root: Path
    train: ReplayDataset
    validation: TensorDataset
    donor: dict[str, Any]
    donor_manifest: dict[str, Any]
    donor_mask_path: Path
    donor_manifest_path: Path

    @classmethod
    def setUpClass(cls) -> None:
        """Run one synthetic relative-feedback donor shared by M4 tests.

        The class-level local temporary directory receives a three-epoch tiny
        run. No real data, downloads, remote storage, or scientific experiment
        is used. State is removed by ``tearDownClass``.
        """

        cls.temporary = tempfile.TemporaryDirectory(dir=PROJECT_ROOT)
        cls.root = Path(cls.temporary.name)
        cls.train = ReplayDataset()
        generator = torch.Generator().manual_seed(4402)
        cls.validation = TensorDataset(
            torch.randn(5, 3, 32, 32, generator=generator),
            torch.randint(0, 10, (5,), generator=generator),
        )
        config = _base_config(cls.root, "donor", MethodId.RELATIVE_FEEDBACK)
        donor_dir = cls.root / "donor"
        cls.donor = run_training(
            config,
            cls.train,
            cls.validation,
            donor_dir,
            "cpu",
            artifacts=_artifacts(),
            data_identities=_identities(),
            calibration=_calibration(),
        )
        cls.donor_mask_path = donor_dir / "masks.npz"
        cls.donor_manifest_path = donor_dir / "mask_manifest.json"
        cls.donor_manifest = json.loads(cls.donor_manifest_path.read_text(encoding="utf-8"))

    @classmethod
    def tearDownClass(cls) -> None:
        """Remove the synthetic M4 fixture directory and return nothing."""

        cls.temporary.cleanup()

    def run_replay(
        self,
        method: MethodId,
        run_id: str,
        *,
        stop_after_global_epoch: int | None = None,
    ) -> tuple[RunConfig, dict[str, Any]]:
        """Execute one tiny replay using the shared donor artifact.

        Inputs select the method, unique local run ID, and optional committed
        boundary. The returned config/context include persisted checkpoints;
        classifier/optimizer/RNG state changes only within this new run.
        """

        config = _replay_config(self.root, run_id, method, self.donor_manifest)
        context = run_training(
            config,
            self.train,
            self.validation,
            self.root / run_id,
            "cpu",
            artifacts=_artifacts(),
            data_identities=_identities(),
            mask_path=self.donor_mask_path,
            mask_manifest_path=self.donor_manifest_path,
            stop_after_global_epoch=stop_after_global_epoch,
        )
        return config, context


class ReplayTrajectoryTests(ReplayFixture):
    """Verify donor equivalence, causal controls, masks, counts, and fast paths."""

    def test_learned_replay_reproduces_donor_and_all_masks(self) -> None:
        """Learned fixed replay exactly matches its donor without controller decisions."""

        _, replay = self.run_replay(MethodId.RELATIVE_REPLAY_LEARNED, "learned")
        report = check_replay_equality(self.donor, replay)
        self.assertTrue(all(report.values()))
        self.assertIsNone(replay["controller"])
        state = replay["state"]
        source_mask = self.donor["state"].executed_exact_mask
        self.assertEqual(state.replay_schedule_mask, source_mask)
        self.assertEqual(state.executed_exact_mask, source_mask)
        self.assertEqual(state.executed_head_mask, source_mask)
        self.assertEqual(state.mask_cursor, len(source_mask))
        self.assertEqual(
            (state.exact_trunk_backwards - 3) + state.synthetic_backwards,
            len(source_mask),
        )

    def test_zero_skip_exact_trunk_counts_and_data_order_reconcile(self) -> None:
        """All causal controls follow donor slots while reporting distinct operations."""

        contexts: dict[MethodId, dict[str, Any]] = {}
        for method in (
            MethodId.RELATIVE_REPLAY_ZERO_SHADOW,
            MethodId.RELATIVE_REPLAY_SKIP_SHADOW,
            MethodId.RELATIVE_REPLAY_EXACT_TRUNK,
        ):
            _, contexts[method] = self.run_replay(method, method.value)
        source_mask = self.donor["state"].executed_exact_mask
        total = len(source_mask)
        exact_count = sum(source_mask)
        warm_batches = 3
        hashes = self.donor["state"].data_order_hashes
        for context in contexts.values():
            self.assertEqual(context["state"].data_order_hashes, hashes)
            self.assertEqual(context["state"].executed_head_mask, source_mask)
            self.assertEqual(context["state"].head_optimizer_steps, warm_batches + exact_count)
            postwarm_rows = context["state"].epoch_records[1:]
            for epoch_index, row in enumerate(postwarm_rows):
                start = self.donor["state"].mask_epoch_offsets[epoch_index]
                end = self.donor["state"].mask_epoch_offsets[epoch_index + 1]
                epoch_exact = sum(source_mask[start:end])
                self.assertEqual(row["head_optimizer_steps"], epoch_exact)
        zero = contexts[MethodId.RELATIVE_REPLAY_ZERO_SHADOW]["state"]
        self.assertEqual(zero.executed_exact_mask, source_mask)
        self.assertEqual(zero.zero_backwards, total - exact_count)
        self.assertEqual((zero.exact_trunk_backwards - warm_batches) + zero.zero_backwards, total)
        self.assertEqual(zero.trunk_optimizer_steps, warm_batches + total)
        skip = contexts[MethodId.RELATIVE_REPLAY_SKIP_SHADOW]["state"]
        self.assertEqual(skip.executed_exact_mask, source_mask)
        self.assertEqual(skip.zero_backwards, 0)
        self.assertEqual(skip.synthetic_backwards, 0)
        self.assertEqual(skip.skipped_slots, total - exact_count)
        self.assertEqual((skip.exact_trunk_backwards - warm_batches) + skip.skipped_slots, total)
        self.assertEqual(skip.trunk_optimizer_steps, warm_batches + exact_count)
        zero_context = contexts[MethodId.RELATIVE_REPLAY_ZERO_SHADOW]
        skip_context = contexts[MethodId.RELATIVE_REPLAY_SKIP_SHADOW]
        self.assertFalse(
            all(
                torch.equal(left, right)
                for left, right in zip(
                    zero_context["model"].trunk.state_dict().values(),
                    skip_context["model"].trunk.state_dict().values(),
                    strict=True,
                )
            )
        )
        momentum_buffers = [
            value["momentum_buffer"]
            for value in zero_context["optimizers"]["trunk"].state.values()
            if "momentum_buffer" in value
        ]
        self.assertTrue(momentum_buffers)
        self.assertTrue(any(float(buffer.abs().sum().item()) > 0 for buffer in momentum_buffers))
        exact_trunk = contexts[MethodId.RELATIVE_REPLAY_EXACT_TRUNK]["state"]
        self.assertEqual(exact_trunk.executed_exact_mask, [True] * total)
        self.assertEqual(exact_trunk.exact_trunk_backwards, warm_batches + total)
        self.assertEqual(exact_trunk.exact_trunk_backwards - warm_batches, total)
        self.assertEqual(exact_trunk.predictor_fits, warm_batches)
        exact_dir = self.root / MethodId.RELATIVE_REPLAY_EXACT_TRUNK.value
        with np.load(exact_dir / "masks.npz", allow_pickle=False) as arrays:
            self.assertEqual(arrays["schedule"].tolist(), source_mask)
            self.assertEqual(arrays["actual_trunk_exact"].tolist(), [True] * total)
            self.assertEqual(arrays["head_update"].tolist(), source_mask)

    def test_shadow_and_fast_classifier_trajectories_match(self) -> None:
        """Removing predictor work leaves zero/skip classifier trajectories identical."""

        pairs = (
            (MethodId.RELATIVE_REPLAY_ZERO_SHADOW, MethodId.RELATIVE_REPLAY_ZERO_FAST),
            (MethodId.RELATIVE_REPLAY_SKIP_SHADOW, MethodId.RELATIVE_REPLAY_SKIP_FAST),
        )
        for shadow_method, fast_method in pairs:
            with self.subTest(fast=fast_method.value):
                _, shadow = self.run_replay(shadow_method, f"pair-{shadow_method.value}")
                _, fast = self.run_replay(fast_method, f"pair-{fast_method.value}")
                _assert_tree_equal(self, shadow["model"].state_dict(), fast["model"].state_dict())
                for key in ("trunk", "head"):
                    _assert_tree_equal(
                        self,
                        shadow["optimizers"][key].state_dict(),
                        fast["optimizers"][key].state_dict(),
                    )
                self.assertIsNotNone(shadow["predictor"])
                self.assertIsNone(fast["predictor"])
                self.assertEqual(
                    shadow["state"].data_order_hashes, fast["state"].data_order_hashes
                )
                self.assertEqual(
                    shadow["state"].executed_exact_mask, fast["state"].executed_exact_mask
                )
                self.assertEqual(
                    shadow["state"].executed_head_mask, fast["state"].executed_head_mask
                )

    def test_provenance_cost_records_and_paired_table_keep_controls_non_deployable(self) -> None:
        """Replay artifacts separate conditional/dependency costs and causal identities."""

        methods = (
            MethodId.RELATIVE_REPLAY_LEARNED,
            MethodId.RELATIVE_REPLAY_ZERO_SHADOW,
            MethodId.RELATIVE_REPLAY_SKIP_SHADOW,
            MethodId.RELATIVE_REPLAY_EXACT_TRUNK,
        )
        records: list[dict[str, Any]] = []
        for method in methods:
            config, context = self.run_replay(method, f"cost-{method.value}")
            provenance = json.loads(
                (self.root / f"cost-{method.value}" / "replay_provenance.json").read_text(encoding="utf-8")
            )
            self.assertTrue(provenance["retrospective"])
            self.assertFalse(provenance["deployment_eligible"])
            self.assertEqual(
                set(provenance["cost_records"]),
                {"conditional_execution", "dependency_inclusive"},
            )
            state = context["state"]
            records.append(
                {
                    "method": method.value,
                    "source_run_id": provenance["mask"]["source_run_id"],
                    "source_mask_sha256": provenance["mask"]["sha256"],
                    "retrospective": True,
                    "deployment_eligible": False,
                    "final_accuracy": state.epoch_records[-1]["validation_accuracy"],
                    "exact_trunk_backwards": state.exact_trunk_backwards,
                    "synthetic_backwards": state.synthetic_backwards,
                    "zero_backwards": state.zero_backwards,
                    "skipped_slots": state.skipped_slots,
                    "trunk_optimizer_steps": state.trunk_optimizer_steps,
                    "head_optimizer_steps": state.head_optimizer_steps,
                    "predictor_fits": state.predictor_fits,
                    "predictor_inferences": state.predictor_inferences,
                    "conditional_cost": provenance["cost_records"]["conditional_execution"],
                    "dependency_inclusive_cost": provenance["cost_records"]["dependency_inclusive"],
                }
            )
            identity = config.scientific_dict()["identity"]
            self.assertTrue(identity["retrospective"])
            self.assertFalse(identity["deployment_eligible"])
        table = build_predictor_replay_table(records)
        self.assertEqual(len(table), 4)


class ReplayFailureAndResumeTests(ReplayFixture):
    """Verify complete preflight rejection and checkpoint-boundary replay resume."""

    def test_malformed_masks_fail_before_output_or_parameter_initialization(self) -> None:
        """Length, boundary, hash, role, and source mismatches all fail pre-update."""

        base_config = _replay_config(
            self.root, "malformed-base", MethodId.RELATIVE_REPLAY_LEARNED, self.donor_manifest
        )
        with np.load(self.donor_mask_path, allow_pickle=False) as arrays:
            original = {name: arrays[name].copy() for name in arrays.files}
        cases: list[tuple[str, dict[str, np.ndarray], dict[str, Any], RunConfig]] = []

        changed = copy.deepcopy(original)
        changed["actual_trunk_exact"][0] = ~changed["actual_trunk_exact"][0]
        cases.append(("hash", changed, copy.deepcopy(self.donor_manifest), base_config))
        shortened = copy.deepcopy(original)
        shortened["actual_trunk_exact"] = shortened["actual_trunk_exact"][:-1]
        cases.append(("length", shortened, copy.deepcopy(self.donor_manifest), base_config))
        boundaries = copy.deepcopy(original)
        boundaries["epoch_offsets"] = np.asarray([0, 2, 6], dtype=np.int64)
        cases.append(("boundaries", boundaries, copy.deepcopy(self.donor_manifest), base_config))
        wrong_role_manifest = copy.deepcopy(self.donor_manifest)
        wrong_role_manifest["replay_export"]["compatible_roles"] = ["head_update"]
        cases.append(("role", copy.deepcopy(original), wrong_role_manifest, base_config))
        source_config_values = base_config.to_dict()
        source_config_values["scientific"]["replay_source"]["run_id"] = "wrong-donor"
        cases.append(("source", copy.deepcopy(original), copy.deepcopy(self.donor_manifest), RunConfig(source_config_values)))

        for name, arrays, manifest, config in cases:
            with self.subTest(case=name):
                fixture = self.root / f"bad-{name}"
                fixture.mkdir()
                mask_path = fixture / "mask.npz"
                np.savez_compressed(mask_path, **arrays)
                manifest_path = fixture / "mask.json"
                manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                output = self.root / f"bad-output-{name}"
                with self.assertRaises(ValueError):
                    run_training(
                        config,
                        self.train,
                        self.validation,
                        output,
                        "cpu",
                        artifacts=_artifacts(),
                        data_identities=_identities(),
                        mask_path=mask_path,
                        mask_manifest_path=manifest_path,
                    )
                self.assertFalse(output.exists())

    def test_replay_resume_matches_uninterrupted_run_at_epoch_boundary(self) -> None:
        """A replay resumed after its first post-warm epoch matches a clean run."""

        full_config, full = self.run_replay(
            MethodId.RELATIVE_REPLAY_ZERO_SHADOW, "resume-full"
        )
        partial_config, partial = self.run_replay(
            MethodId.RELATIVE_REPLAY_ZERO_SHADOW,
            "resume-partial",
            stop_after_global_epoch=1,
        )
        checkpoint = partial["last_checkpoint"]
        resumed = resume_run(
            partial_config,
            self.train,
            self.validation,
            self.root / "resume-partial",
            "cpu",
            artifacts=_artifacts(),
            data_identities=_identities(),
            mask_path=self.donor_mask_path,
            mask_manifest_path=self.donor_manifest_path,
            checkpoint_path=checkpoint,
        )
        _assert_tree_equal(self, full["model"].state_dict(), resumed["model"].state_dict())
        _assert_tree_equal(
            self, full["predictor"].state_dict(), resumed["predictor"].state_dict()
        )
        for key in ("trunk", "head", "predictor"):
            _assert_tree_equal(
                self,
                full["optimizers"][key].state_dict(),
                resumed["optimizers"][key].state_dict(),
            )
        self.assertEqual(full["state"].replay_schedule_mask, resumed["state"].replay_schedule_mask)
        self.assertEqual(full["state"].executed_exact_mask, resumed["state"].executed_exact_mask)
        self.assertEqual(full["state"].executed_head_mask, resumed["state"].executed_head_mask)
        self.assertEqual(full["state"].data_order_hashes, resumed["state"].data_order_hashes)
        loaded = load_checkpoint(
            resumed["last_checkpoint"],
            expected={"replay_schedule_sha256": self.donor_manifest["replay_export"]["sha256"]},
        )
        self.assertEqual(loaded["run_state"]["mask_cursor"], len(loaded["run_state"]["replay_schedule_mask"]))


class MaskHashTests(unittest.TestCase):
    """Check that mask identity includes both bytes and epoch partitioning."""

    def test_mask_hash_changes_with_bytes_or_boundaries(self) -> None:
        """Canonical mask hashes distinguish content and epoch offsets exactly."""

        first = mask_identity_hash([True, False, True, False], [0, 2, 4])
        self.assertNotEqual(first, mask_identity_hash([False, False, True, False], [0, 2, 4]))
        self.assertNotEqual(first, mask_identity_hash([True, False, True, False], [0, 1, 4]))


if __name__ == "__main__":
    unittest.main()
