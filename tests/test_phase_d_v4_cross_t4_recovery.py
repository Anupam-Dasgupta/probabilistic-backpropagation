"""Synthetic tests for the versioned compatible-T4 recovery policy."""

from __future__ import annotations

import copy
import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch

from analyze_phase_d_v4 import validate_run_directory, verify_resume_equivalence
import gradient_phase_d_resnet_v4 as training_module
import phase_d_v4_io as io_module
from gradient_phase_d_resnet_v4 import (
    _synthetic_fixture,
    probe_checkpoint_continuation,
    resume_run,
    run_training,
)
from phase_d_v4_io import (
    COMPATIBLE_T4_RESUME_POLICY,
    STRICT_RESUME_POLICY,
    capture_rng_state,
    execution_fingerprint,
    latest_committed_checkpoint,
    load_resume_attempts,
    load_continuation_evidence,
    record_resume_attempt,
    restore_or_verify_resume_snapshot,
    restore_resume_snapshot,
    resume_compatibility_report,
    sha256_file,
    verify_resume_snapshot_execution,
)
from tests.test_phase_d_v4_m2 import _artifacts, _identities, _runner_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _fake_t4_environment(uuid: str) -> dict[str, object]:
    """Return a complete deterministic T4 environment fixture for one UUID.

    The UUID is the only allocation-specific input. The returned mapping models
    metadata only; it performs no CUDA query, backend change, file write, or RNG
    operation and is not evidence about real T4 numerical behavior.
    """

    return {
        "python": "3.13.6",
        "platform": "Linux-6.6.0-colab-x86_64",
        "system": "Linux",
        "machine": "x86_64",
        "processor": "x86_64",
        "torch": "2.10.0+cu130",
        "torchvision": "0.25.0+cu130",
        "numpy": "2.3.4",
        "cuda_runtime": "13.0",
        "cudnn": 91002,
        "device_type": "cuda",
        "device_index": 0,
        "device_name": "NVIDIA T4",
        "device_uuid": uuid,
        "device_capability": [7, 5],
        "total_memory_bytes": 15835660288,
        "cuda_device_count": 1,
        "driver_version": "580.97",
        "driver_query": {
            "status": "ok", "source": "nvidia-smi",
            "version": "580.97", "detail": None,
        },
        "cublas_workspace_config": ":4096:8",
        "matmul_tf32": False,
        "cudnn_tf32": False,
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
    }


def _cpu_probe_report_as_mocked_cuda(
    config: object, checkpoint: str | Path, _device: object
) -> dict[str, object]:
    """Run the real copied-state probe on CPU and relabel only mocked metadata.

    Inputs mirror the runner probe callback. The returned real CPU engineering
    result has ``device_type`` changed to ``cuda`` solely because these tests
    model T4 environment metadata without possessing a T4. Checkpoint/RNG
    isolation and state comparisons still execute; no cross-GPU claim is made.
    """

    report = probe_checkpoint_continuation(config, checkpoint, "cpu")
    report["device_type"] = "cuda"
    return report


class CompatibilityPolicyTests(unittest.TestCase):
    """Check that only the enumerated compatible-T4 differences are admitted."""

    def test_strict_and_compatible_uuid_behavior(self) -> None:
        """Strict rejects a new UUID while the explicit T4 policy records it.

        Two metadata fixtures and one tiny resolved config are read only. The
        compatible report must preserve both hashes and identify UUID provenance;
        strict mode raises without changing config, files, RNG, or device state.
        """

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            config = _runner_config(Path(temporary) / "run", warmup_epochs=1, postwarm_epochs=1)
            saved = _fake_t4_environment("GPU-A")
            current = _fake_t4_environment("GPU-B")
            same = resume_compatibility_report(config, saved, config, saved, STRICT_RESUME_POLICY)
            self.assertFalse(same["cross_device"])
            with self.assertRaisesRegex(ValueError, "strict resume"):
                resume_compatibility_report(config, saved, config, current, STRICT_RESUME_POLICY)
            report = resume_compatibility_report(
                config, saved, config, current, COMPATIBLE_T4_RESUME_POLICY
            )
            self.assertTrue(report["cross_device"])
            self.assertEqual(report["provenance_only_changes"]["device_uuid"]["saved"], "GPU-A")
            self.assertEqual(report["saved_execution_fingerprint"], execution_fingerprint(config, saved))
            self.assertEqual(report["current_execution_fingerprint"], execution_fingerprint(config, current))

    def test_required_mismatches_and_legacy_evidence_are_rejected(self) -> None:
        """GPU model, software, backend, and missing legacy fields remain fatal.

        Each case changes a copied fixture only. The policy raises before any
        checkpoint/model operation and leaves all input mappings unchanged.
        """

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            config = _runner_config(Path(temporary) / "run", warmup_epochs=1, postwarm_epochs=1)
            saved = _fake_t4_environment("GPU-A")
            cases = {
                "device": ("device_name", "NVIDIA A100-SXM4-40GB"),
                "software": ("torch", "2.10.1+cu130"),
                "backend": ("cudnn_deterministic", False),
                "precision": ("matmul_tf32", True),
                "driver": ("driver_version", "581.01"),
            }
            for label, (field, value) in cases.items():
                with self.subTest(label=label):
                    current = _fake_t4_environment("GPU-B")
                    current[field] = value
                    with self.assertRaises(ValueError):
                        resume_compatibility_report(
                            config, saved, config, current, COMPATIBLE_T4_RESUME_POLICY
                        )
            missing_driver = _fake_t4_environment("GPU-B")
            missing_driver["driver_version"] = None
            missing_driver["driver_query"] = {
                "status": "unknown", "source": "nvidia-smi",
                "version": None, "detail": "invalid_or_ambiguous_output",
            }
            with self.assertRaisesRegex(ValueError, "driver evidence"):
                resume_compatibility_report(
                    config, saved, config, missing_driver,
                    COMPATIBLE_T4_RESUME_POLICY,
                )
            legacy = copy.deepcopy(saved)
            legacy.pop("device_capability")
            legacy_strict = resume_compatibility_report(
                config, legacy, config, legacy, STRICT_RESUME_POLICY
            )
            self.assertTrue(legacy_strict["compatible"])
            with self.assertRaisesRegex(ValueError, "legacy checkpoint"):
                resume_compatibility_report(
                    config, legacy, config, _fake_t4_environment("GPU-B"),
                    COMPATIBLE_T4_RESUME_POLICY,
                )

    def test_driver_query_reports_success_and_every_failure_mode(self) -> None:
        """The bounded NVIDIA query distinguishes real versions from failures.

        Mocked subprocess results cover a valid display-driver release, missing
        executable, nonzero exit, and placeholder output. The returned evidence
        is explicit and no actual driver, command, CUDA state, RNG, or file is
        changed.
        """

        success = io_module.subprocess.CompletedProcess([], 0, "580.97\n", "")
        with mock.patch.object(io_module.subprocess, "run", return_value=success):
            self.assertEqual(io_module._query_nvidia_driver_version(0)["version"], "580.97")
        with mock.patch.object(io_module.subprocess, "run", side_effect=FileNotFoundError):
            self.assertEqual(io_module._query_nvidia_driver_version(0)["status"], "unavailable")
        failed = io_module.subprocess.CompletedProcess([], 9, "", "failure")
        with mock.patch.object(io_module.subprocess, "run", return_value=failed):
            self.assertEqual(io_module._query_nvidia_driver_version(0)["status"], "failed")
        unknown = io_module.subprocess.CompletedProcess([], 0, "N/A\n", "")
        with mock.patch.object(io_module.subprocess, "run", return_value=unknown):
            self.assertEqual(io_module._query_nvidia_driver_version(0)["status"], "unknown")

    def test_notebook_defaults_strict_and_checks_before_restore(self) -> None:
        """The recovery cell exposes compatibility without weakening safe gates.

        Notebook JSON is parsed and each code cell is compiled but never run. The
        assertions require strict default selection, early compatibility checking,
        runner-owned continuation validation, explicit resume propagation, and unchanged
        real-data/full-pilot gates; no files or runtime state are modified.
        """

        notebook = json.loads((PROJECT_ROOT / "PHASE_D_V4_COLAB.ipynb").read_text(encoding="utf-8"))
        code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        for index, source in enumerate(code):
            compile(source, f"recovery_notebook_cell_{index}", "exec")
        recovery = next(source for source in code if "RESTORE_RUN_ID" in source)
        self.assertIn("RECOVERY_COMPATIBILITY = STRICT_RESUME_POLICY", recovery)
        self.assertIn("COMPATIBLE_T4_RESUME_POLICY", recovery)
        self.assertLess(
            recovery.index("verify_resume_snapshot_execution("),
            recovery.index("restore_or_verify_resume_snapshot("),
        )
        self.assertNotIn("probe_checkpoint_continuation(", recovery)
        dataset_stage = next(source for source in code if "AUTHORIZE_RECOVERY_DATA_SETUP" in source)
        continuation_stage = next(source for source in code if "AUTHORIZE_RECOVERY_CONTINUATION" in source)
        self.assertIn("AUTHORIZE_RECOVERY_DATA_SETUP = False", dataset_stage)
        self.assertIn("AUTHORIZE_RECOVERY_CONTINUATION = False", continuation_stage)
        self.assertIn("'--resume-compatibility',RECOVERY_COMPATIBILITY", continuation_stage)
        joined = "\n".join(code)
        self.assertIn("AUTHORIZE_REAL_DATA_PREFLIGHT = False", joined)
        self.assertIn("AUTHORIZE_FULL_PILOT = False", joined)


class CrossT4SyntheticRecoveryTests(unittest.TestCase):
    """Exercise repeated mocked allocation changes through the real runner."""

    def test_repeated_transitions_preserve_state_and_analysis_labels(self) -> None:
        """Two UUID transitions match uninterrupted synthetic scientific state.

        Tiny CPU tensors execute the production checkpoint/resume path while only
        environment metadata is mocked as T4 allocations. The test verifies the
        immutable origin, three ordered sessions, unknown gaps, epoch-boundary
        continuation, and analysis exclusion. It does not claim cross-T4 numeric
        equivalence and writes only beneath a temporary directory.
        """

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            train, validation, _ = _synthetic_fixture(seed=6610)
            full_dir = root / "full"
            resumed_dir = root / "resumed"
            full_config = _runner_config(full_dir, warmup_epochs=1, postwarm_epochs=2)
            resumed_config = _runner_config(resumed_dir, warmup_epochs=1, postwarm_epochs=2)
            allocation_a = _fake_t4_environment("GPU-A")
            allocation_b = _fake_t4_environment("GPU-B")
            allocation_c = _fake_t4_environment("GPU-C")
            with mock.patch.object(training_module, "environment_metadata", return_value=allocation_a):
                run_training(
                    full_config, train, validation, full_dir, "cpu",
                    artifacts=_artifacts(), data_identities=_identities(),
                )
                run_training(
                    resumed_config, train, validation, resumed_dir, "cpu",
                    artifacts=_artifacts(), data_identities=_identities(),
                    stop_after_global_epoch=0,
                )
            origin_manifest = json.loads((resumed_dir / "manifest.json").read_text(encoding="utf-8"))
            origin_fingerprint = origin_manifest["execution_fingerprint"]
            # Model an allocation that began recovery from checkpoint 1 and was
            # then lost before publishing any later epoch checkpoint.
            record_resume_attempt(
                resumed_dir,
                None,
                {
                    "schema_version": origin_manifest["schema_version"],
                    "policy_version": "phase_d_v4.resume_compatibility.v1",
                    "policy": COMPATIBLE_T4_RESUME_POLICY,
                    "started_from_checkpoint_sequence": 1,
                    "restart_from_global_epoch": 1,
                    "repeated_prior_attempt_ids": [],
                    "saved_execution_fingerprint": origin_fingerprint,
                    "current_execution_fingerprint": execution_fingerprint(
                        resumed_config, allocation_b
                    ),
                    "current_environment": allocation_b,
                    "cross_device": True,
                    "interruption_gap_seconds": None,
                    "recovery_preparation_seconds_before_attempt_commit": 0.01,
                },
            )
            with (
                mock.patch.object(training_module, "environment_metadata", return_value=allocation_b),
                mock.patch.object(
                    training_module, "probe_checkpoint_continuation",
                    side_effect=_cpu_probe_report_as_mocked_cuda,
                ),
            ):
                resume_run(
                    resumed_config, train, validation, resumed_dir, "cpu",
                    artifacts=_artifacts(), data_identities=_identities(),
                    stop_after_global_epoch=1,
                    resume_compatibility_policy=COMPATIBLE_T4_RESUME_POLICY,
                )
            with (
                mock.patch.object(training_module, "environment_metadata", return_value=allocation_c),
                mock.patch.object(
                    training_module, "probe_checkpoint_continuation",
                    side_effect=_cpu_probe_report_as_mocked_cuda,
                ),
            ):
                resume_run(
                    resumed_config, train, validation, resumed_dir, "cpu",
                    artifacts=_artifacts(), data_identities=_identities(),
                    resume_compatibility_policy=COMPATIBLE_T4_RESUME_POLICY,
                )
            self.assertTrue(verify_resume_equivalence(full_dir, resumed_dir)["equivalent"])
            manifest = json.loads((resumed_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["execution_fingerprint"], origin_fingerprint)
            self.assertEqual(len(manifest["execution_sessions"]), 3)
            self.assertEqual(
                [row["environment"]["device_uuid"] for row in manifest["execution_sessions"]],
                ["GPU-A", "GPU-B", "GPU-C"],
            )
            self.assertTrue(all(row["interruption_gap_seconds"] is None for row in manifest["execution_sessions"]))
            self.assertEqual(len(manifest["resume_attempts"]), 3)
            repeated = [row["repeated_prior_attempt_ids"] for row in manifest["resume_attempts"]]
            self.assertTrue(any(values for values in repeated), repeated)
            record = validate_run_directory(resumed_dir)
            self.assertTrue(record["accuracy_comparison_eligible"])
            self.assertFalse(record["clean_timing_comparison_eligible"])
            self.assertTrue(record["cross_device_resumed"])
            manifest["continuation_evidence"] = []
            (resumed_dir / "manifest.json").write_text(
                json.dumps(manifest, indent=2), encoding="utf-8"
            )
            missing = validate_run_directory(resumed_dir)
            self.assertFalse(missing["continuation_evidence_valid"])
            self.assertFalse(missing["accuracy_comparison_eligible"])

    def test_snapshot_precheck_and_copied_continuation_are_nonmutating(self) -> None:
        """Persisted metadata admits a compatible UUID and cloned update check.

        A one-epoch tiny run is persisted locally, then checked with a replacement
        UUID. The continuation probe preserves checkpoint bytes and all process RNG
        streams exactly. No production context, external storage, or dataset is used.
        """

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            run = root / "run"
            remote = root / "remote"
            config = _runner_config(run, warmup_epochs=1, postwarm_epochs=1)
            train, validation, _ = _synthetic_fixture(seed=6611)
            saved = _fake_t4_environment("GPU-A")
            current = _fake_t4_environment("GPU-B")
            with mock.patch.object(training_module, "environment_metadata", return_value=saved):
                run_training(
                    config, train, validation, run, "cpu",
                    artifacts=_artifacts(), data_identities=_identities(),
                    stop_after_global_epoch=0, persistence_directory=remote,
                )
            verified = verify_resume_snapshot_execution(
                remote,
                current_config=config,
                current_environment=current,
                policy=COMPATIBLE_T4_RESUME_POLICY,
            )
            self.assertTrue(verified["resume_compatibility"]["cross_device"])
            checkpoint = latest_committed_checkpoint(run / "checkpoints")
            before_hash = sha256_file(checkpoint)
            before_rng = capture_rng_state()
            report = probe_checkpoint_continuation(config, checkpoint, "cpu")
            after_rng = capture_rng_state()
            self.assertEqual(sha256_file(checkpoint), before_hash)
            self.assertTrue(training_module._state_trees_equal(before_rng, after_rng))
            self.assertFalse(report["cross_device_numerical_equivalence_proven"])
            with (
                mock.patch.object(training_module, "environment_metadata", return_value=current),
                mock.patch.object(
                    training_module, "probe_checkpoint_continuation",
                    side_effect=_cpu_probe_report_as_mocked_cuda,
                ),
            ):
                resume_run(
                    config, train, validation, run, "cpu",
                    artifacts=_artifacts(), data_identities=_identities(),
                    persistence_directory=remote,
                    resume_compatibility_policy=COMPATIBLE_T4_RESUME_POLICY,
                )
            evidence = load_continuation_evidence(run, remote)
            self.assertEqual(len(evidence), 1)
            self.assertEqual(evidence[0]["checkpoint_sha256"], before_hash)
            shutil.rmtree(run)
            restore_resume_snapshot(remote, run)
            restored_record = validate_run_directory(run)
            self.assertTrue(restored_record["continuation_evidence_valid"])
            self.assertTrue(restored_record["accuracy_comparison_eligible"])
            self.assertEqual(len(load_continuation_evidence(run, remote)), 1)

    def test_cli_cross_device_resume_stops_when_continuation_probe_fails(self) -> None:
        """The CLI cannot bypass a failing authoritative continuation check.

        A tiny synthetic CLI run commits one epoch under mocked allocation A.
        Allocation B then raises inside the copied-state probe. Resume propagates
        the failure before another checkpoint or evidence record is published;
        no real GPU metadata, network, dataset, or scientific work is used.
        """

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            run = Path(temporary) / "cli"
            with mock.patch.object(
                training_module, "environment_metadata",
                return_value=_fake_t4_environment("GPU-A"),
            ):
                training_module.main([
                    "smoke", "--device", "cpu", "--out", str(run),
                    "--stop-after-global-epoch", "0",
                ])
            checkpoint_before = latest_committed_checkpoint(run / "checkpoints")
            with (
                mock.patch.object(
                    training_module, "environment_metadata",
                    return_value=_fake_t4_environment("GPU-B"),
                ),
                mock.patch.object(
                    training_module, "probe_checkpoint_continuation",
                    side_effect=RuntimeError("synthetic continuation failure"),
                ),
            ):
                with self.assertRaisesRegex(RuntimeError, "continuation failure"):
                    training_module.main([
                        "resume", "--device", "cpu", "--out", str(run),
                        "--resume-compatibility", COMPATIBLE_T4_RESUME_POLICY,
                    ])
            self.assertEqual(latest_committed_checkpoint(run / "checkpoints"), checkpoint_before)
            self.assertEqual(load_continuation_evidence(run), [])

    def test_corrupt_persisted_attempt_history_is_rejected(self) -> None:
        """A damaged committed interruption record cannot be silently ignored.

        One tiny JSON attempt is written to temporary local/persistence paths,
        then only the persisted fixture is deliberately corrupted. Loading raises
        before resume; no model, dataset, network, RNG, or real artifact is used.
        """

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            run = root / "run"
            remote = root / "remote"
            record_resume_attempt(
                run,
                remote,
                {
                    "started_from_checkpoint_sequence": 1,
                    "restart_from_global_epoch": 1,
                    "policy_version": "phase_d_v4.resume_compatibility.v1",
                    "policy": COMPATIBLE_T4_RESUME_POLICY,
                    "saved_execution_fingerprint": "a" * 64,
                    "current_execution_fingerprint": "b" * 64,
                    "current_environment": _fake_t4_environment("GPU-B"),
                    "cross_device": True,
                    "interruption_gap_seconds": None,
                    "repeated_prior_attempt_ids": [],
                },
            )
            attempts = load_resume_attempts(run, remote)
            self.assertEqual([item["attempt_id"] for item in attempts], [0])
            (remote / "resume_attempts" / "attempt_000000.json").write_text(
                "{}", encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_resume_attempts(run, remote)


if __name__ == "__main__":
    unittest.main()
