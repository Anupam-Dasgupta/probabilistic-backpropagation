"""Synthetic pre-experiment analysis and integration tests for milestone M6."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path

import torch

from analyze_phase_d_v4 import (
    average_replica_records_within_seed,
    classify_costs,
    dependency_inclusive_cost,
    deployment_pareto,
    epoch_observation_times,
    hardware_drift_check,
    make_run_order_manifest,
    paired_accuracy_difference,
    paired_time_ratio,
    plot_accuracy_curves,
    reconcile_timing,
    target_attainment_rows,
    validate_run_directory,
)
from gradient_phase_d_resnet_v4 import (
    _diagnostic_fixture_state,
    _synthetic_fixture,
    build_condition_config,
    calibrate_controller,
    main,
    make_split_indices,
    probe_normalization_transition,
    resume_run,
    run_training,
)
from phase_d_v4_io import MethodId, RunConfig, TimingLedger, sha256_file
from tests.test_phase_d_v4_m2 import (
    _artifacts, _assert_tensor_tree_equal, _identities, _runner_config,
)


def _timing(components: dict[str, float] | None = None) -> dict[str, object]:
    """Build a reconciled fabricated timing ledger in seconds for analysis tests."""

    ledger = TimingLedger()
    for name, value in (components or {
        "setup": 1.0,
        "scale_estimation": 0.5,
        "warmup_training": 2.0,
        "warmup_validation": 0.5,
        "calibration": 0.25,
        "postwarm_training": 3.0,
        "postwarm_validation": 0.75,
        "checkpoint_log_io": 0.4,
        "diagnostics": 0.0,
        "finalization": 0.1,
    }).items():
        ledger.add(name, value)
    return ledger.to_dict()


def _record(method: str, seed: int, accuracy: float, seconds: float = 10.0) -> dict[str, object]:
    """Return one completed fabricated run record with a stable hardware identity."""

    timing = _timing(
        {
            "setup": 1.0,
            "warmup_training": 2.0,
            "scale_estimation": 0.0,
            "calibration": 0.0,
            "postwarm_training": seconds - 4.0,
            "warmup_validation": 0.5,
            "postwarm_validation": 0.5,
            "checkpoint_log_io": 0.0,
            "diagnostics": 0.0,
            "finalization": 0.0,
        }
    )
    return {
        "run_id": f"{method}-s{seed}",
        "seed": seed,
        "method": method,
        "normalization": "frozen_head_postwarm",
        "completed": True,
        "comparison_eligible": True,
        "resumed": False,
        "retrospective": False,
        "deployment_eligible": True,
        "hardware_key": "same-t4",
        "final_accuracy": accuracy,
        "timing": reconcile_timing(timing),
        "epochs": [{"global_epoch": 0}],
    }


class AnalysisMathematicsTests(unittest.TestCase):
    """Check units, pairing, intervals, attainment, costs, and Pareto eligibility."""

    def test_timing_reconciliation_and_endpoint_checkpoint_order(self) -> None:
        """Timing formulas match by hand and current checkpoint I/O is charged later."""

        timing = _timing()
        timing["checkpoint_io"] = [
            {"checkpoint_sequence": 1, "io_seconds": 0.2},
            {"checkpoint_sequence": 2, "io_seconds": 0.2},
        ]
        rows = [
            {
                "global_epoch": 0, "stage": "warmup", "checkpoint_sequence": 1,
                "training_seconds": 2.0, "validation_seconds": 0.5,
            },
            {
                "global_epoch": 1, "stage": "postwarm", "checkpoint_sequence": 2,
                "training_seconds": 3.0, "validation_seconds": 0.75,
            },
        ]
        reconciled = reconcile_timing(timing)
        self.assertAlmostEqual(reconciled["totals"]["learning_seconds"], 5.75)
        self.assertAlmostEqual(reconciled["totals"]["workflow_seconds"], 8.5)
        endpoints = epoch_observation_times(rows, timing)
        self.assertAlmostEqual(endpoints[0]["observation_time_seconds"], 4.0)
        self.assertAlmostEqual(endpoints[1]["observation_time_seconds"], 8.2)

    def test_accuracy_pp_student_t_and_insufficient_sample(self) -> None:
        """Accuracy effects use percentage points and preserve one-seed uncertainty."""

        records = []
        for seed, reference, candidate in ((0, 0.80, 0.81), (1, 0.82, 0.84), (2, 0.79, 0.82)):
            records.extend((_record("reference", seed, reference), _record("candidate", seed, candidate)))
        result = paired_accuracy_difference(records, "reference", "candidate")
        for observed, expected in zip(
            [row["difference_pp"] for row in result["per_seed"]],
            [1.0, 2.0, 3.0],
            strict=True,
        ):
            self.assertAlmostEqual(observed, expected)
        self.assertAlmostEqual(result["mean"], 2.0)
        self.assertAlmostEqual(result["sample_sd"], 1.0)
        half = 4.302653 / math.sqrt(3)
        self.assertAlmostEqual(result["ci_low"], 2.0 - half)
        one_seed = paired_accuracy_difference(records[:2], "reference", "candidate")
        self.assertEqual(one_seed["status"], "insufficient_sample")
        self.assertIsNone(one_seed["ci_low"])

    def test_time_log_ratio_pairing_and_rejections(self) -> None:
        """Time ratios use logs and reject duplicate seeds, resumes, and hardware drift."""

        records = []
        for seed in range(3):
            records.extend((_record("reference", seed, 0.8, 12.0), _record("candidate", seed, 0.8, 8.0)))
        result = paired_time_ratio(records, "reference", "candidate")
        self.assertAlmostEqual(result["geometric_mean_ratio"], 1.5)
        duplicate = records + [_record("candidate", 0, 0.8, 8.0)]
        with self.assertRaisesRegex(ValueError, "duplicate seed"):
            paired_time_ratio(duplicate, "reference", "candidate")
        mismatched = copy.deepcopy(records)
        mismatched[1]["hardware_key"] = "other-gpu"
        with self.assertRaisesRegex(ValueError, "mismatched hardware"):
            paired_time_ratio(mismatched, "reference", "candidate")
        resumed = copy.deepcopy(records)
        resumed[1]["resumed"] = True
        with self.assertRaisesRegex(ValueError, "resumed"):
            paired_time_ratio(resumed, "reference", "candidate")

    def test_random_masks_average_within_seed_before_pairing(self) -> None:
        """Two gap replicas become one seed-level observation, not two samples."""

        replicas = []
        for replica, accuracy in ((0, 0.80), (1, 0.84)):
            record = _record("relative_gap_permutation_epoch", 3, accuracy)
            record["run_id"] += f"-r{replica}"
            record["mask_replicate"] = replica
            record["epochs"] = [
                {
                    "global_epoch": 0,
                    "validation_accuracy": accuracy,
                    "validation_ce": 1.0,
                    "observation_time_seconds": 10.0 + replica,
                }
            ]
            replicas.append(record)
        averaged = average_replica_records_within_seed(replicas)
        self.assertEqual(len(averaged), 1)
        self.assertEqual(averaged[0]["replica_count"], 2)
        self.assertFalse(averaged[0]["replicas_are_independent_seeds"])
        self.assertAlmostEqual(averaged[0]["final_accuracy"], 0.82)

    def test_accuracy_plots_use_all_observed_epoch_and_time_points(self) -> None:
        """Fabricated endpoint records create both planned descriptive figures."""

        records = [
            {
                "run_id": "plot-run",
                "seed": 0,
                "method": "exact",
                "epochs": [
                    {"global_epoch": 0, "validation_accuracy": 0.50, "observation_time_seconds": 1.0},
                    {"global_epoch": 1, "validation_accuracy": 0.60, "observation_time_seconds": 2.0},
                ],
            }
        ]
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            paths = plot_accuracy_curves(records, temporary)
            self.assertEqual(len(paths), 2)
            self.assertTrue(all(path.is_file() and path.stat().st_size > 0 for path in paths))

    def test_targets_cost_classes_and_deployable_pareto(self) -> None:
        """Warm-up attainment and censoring coexist with strict cost eligibility."""

        record = _record("candidate", 0, 0.88)
        record["epochs"] = [
            {
                "stage": "warmup", "global_epoch": 0, "validation_accuracy": 0.86,
                "observation_time_seconds": 3.0, "checkpoint_sha256": "a" * 64,
            },
            {
                "stage": "postwarm", "global_epoch": 1, "validation_accuracy": 0.88,
                "observation_time_seconds": 7.0, "checkpoint_sha256": "b" * 64,
            },
        ]
        targets = target_attainment_rows([record], [0.85, 0.89])
        self.assertTrue(targets[0]["reached"])
        self.assertEqual(targets[0]["stage"], "warmup")
        self.assertFalse(targets[1]["reached"])
        self.assertIsNone(targets[1]["attainment_time_seconds"])
        retrospective = copy.deepcopy(record)
        retrospective.update(run_id="retro", retrospective=True, deployment_eligible=False)
        self.assertEqual(deployment_pareto([record, retrospective])[0]["run_id"], record["run_id"])
        costs = classify_costs(retrospective)
        self.assertFalse(costs["counts_are_flops"])
        self.assertIsNone(costs["replay_conditional"])
        source_timing = _timing()
        execution_timing = _timing()
        execution_timing["checkpoint_io"] = [
            {"checkpoint_sequence": 1, "io_seconds": 0.2},
            {"checkpoint_sequence": 2, "io_seconds": 0.2},
        ]
        inclusive = dependency_inclusive_cost(
            source_timing,
            execution_timing,
            shared_warmup_checkpoint_sequences=[1],
        )
        self.assertAlmostEqual(inclusive["shared_warmup_subtracted_seconds"], 3.2)
        self.assertAlmostEqual(inclusive["dependency_inclusive_workflow_seconds"], 13.8)


class OrderingAndIntegrationTests(unittest.TestCase):
    """Check predeclared ordering, drift flags, probe prerequisites, and run schemas."""

    def test_private_order_manifest_rotation_and_hardware_drift(self) -> None:
        """Ordering leaves global RNG alone and repeated blocks rotate saved order."""

        before = copy.deepcopy(__import__("numpy").random.get_state())
        first = make_run_order_manifest("pilot", ["donor"], ["a", "b", "c"], repetition=0)
        after = __import__("numpy").random.get_state()
        self.assertEqual(before[0], after[0])
        self.assertTrue((before[1] == after[1]).all())
        second = make_run_order_manifest(
            "pilot", ["donor"], ["a", "b", "c"], repetition=1,
            base_order=first["measured_block"],
        )
        self.assertEqual(second["measured_block"], first["measured_block"][1:] + first["measured_block"][:1])
        self.assertFalse(hardware_drift_check([10, 10, 10], [10.5, 10.5, 10.5])["flagged"])
        self.assertTrue(hardware_drift_check([10, 10, 10], [11.1, 11.1, 11.1])["flagged"])

    def test_split_indices_follow_private_torch_permutation(self) -> None:
        """Small split construction matches the frozen prefix rule without RNG leakage."""

        before = torch.get_rng_state().clone()
        training, validation = make_split_indices(11, 3, 71503)
        expected = torch.randperm(11, generator=torch.Generator().manual_seed(71503)).numpy()
        self.assertTrue((validation == expected[:3]).all())
        self.assertTrue((training == expected[3:]).all())
        torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)

    def test_normalization_uses_training_mode_trunk_boundary(self) -> None:
        """Transition probes execute their copied trunk with training-mode BatchNorm."""

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            state = _diagnostic_fixture_state(Path(temporary), torch.device("cpu"))
            config, model, predictor = state[:3]
            observed: list[bool] = []

            def hook(module: torch.nn.Module, _inputs: object, _output: object) -> None:
                """Record copied trunk mode without changing tensors or buffers."""

                observed.append(module.training)

            handle = model.trunk.register_forward_hook(hook)
            try:
                probe_normalization_transition(
                    model, predictor, state[4], state[5], state[8], config,
                    sample_source="deterministic_training",
                )
            finally:
                handle.remove()
            self.assertEqual(observed, [True])

    def test_deferred_calibration_isolated_and_scientific_rows_validate(self) -> None:
        """A tiny feedback run calibrates after warm-up and emits complete identities."""

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            config = _runner_config(
                root / "feedback", method=MethodId.RELATIVE_FEEDBACK,
                warmup_epochs=1, postwarm_epochs=1, batch_size=8,
            )
            values = config.to_dict()
            values["scientific"]["calibration_and_profile"]["calibration_batches"] = 3
            config = RunConfig(values)
            train, validation, _fixture_identities = _synthetic_fixture()
            context = run_training(
                config, train, validation, root / "feedback", "cpu",
                artifacts=_artifacts(), data_identities=_identities(),
            )
            self.assertIsNotNone(context["controller"])
            self.assertTrue((root / "feedback" / "calibration.json").is_file())
            validated = validate_run_directory(root / "feedback")
            self.assertEqual(validated["run_id"], "feedback")
            self.assertEqual(validated["seed"], 2)
            self.assertEqual(validated["method"], MethodId.RELATIVE_FEEDBACK.value)
            self.assertEqual(len(validated["epochs"]), 2)
            self.assertTrue(all(len(row["checkpoint_sha256"]) == 64 for row in validated["epochs"]))
            manifest_path = root / "feedback" / "manifest.json"
            original_manifest = manifest_path.read_text(encoding="utf-8")
            malformed = json.loads(original_manifest)
            malformed["schema_version"] = "unknown"
            manifest_path.write_text(json.dumps(malformed), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unknown run schema"):
                validate_run_directory(root / "feedback")
            manifest_path.write_text(original_manifest, encoding="utf-8")
            epochs_path = root / "feedback" / "epochs.csv"
            original_epochs = epochs_path.read_text(encoding="utf-8")
            epochs_path.write_text("\n".join(original_epochs.splitlines()[:-1]) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "missing epochs"):
                validate_run_directory(root / "feedback")
            epochs_path.write_text(original_epochs, encoding="utf-8")

    def test_deferred_calibration_resume_matches_uninterrupted(self) -> None:
        """Warm-up-boundary resume calibrates once and reproduces feedback state."""

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            train, validation, _fixture_identities = _synthetic_fixture(seed=1203)
            configs = []
            for name in ("full", "resumed"):
                base = _runner_config(
                    root / name, method=MethodId.RELATIVE_FEEDBACK,
                    warmup_epochs=1, postwarm_epochs=1, batch_size=8,
                ).to_dict()
                base["scientific"]["calibration_and_profile"]["calibration_batches"] = 3
                configs.append(RunConfig(base))
            full = run_training(
                configs[0], train, validation, root / "full", "cpu",
                artifacts=_artifacts(), data_identities=_identities(),
            )
            run_training(
                configs[1], train, validation, root / "resumed", "cpu",
                artifacts=_artifacts(), data_identities=_identities(),
                stop_after_global_epoch=0,
            )
            resumed = resume_run(
                configs[1], train, validation, root / "resumed", "cpu",
                artifacts=_artifacts(), data_identities=_identities(),
            )
            _assert_tensor_tree_equal(self, full["model"].state_dict(), resumed["model"].state_dict())
            _assert_tensor_tree_equal(
                self, full["predictor"].state_dict(), resumed["predictor"].state_dict()
            )
            self.assertEqual(full["controller"].state_dict(), resumed["controller"].state_dict())
            self.assertEqual(
                full["state"].data_order_hashes, resumed["state"].data_order_hashes
            )

    def test_config_builder_and_cli_do_not_launch_work(self) -> None:
        """Independent pilot configs resolve exactly and CLI config creation is local only."""

        config = build_condition_config(
            MethodId.EXACT_STANDARD_BN,
            run_id="pilot-s0-exact-standard",
            output_root="runs_phase_d_v4/pilot",
        )
        identity = config.values["scientific"]["identity"]
        self.assertEqual(identity["normalization"], "standard_bn")
        self.assertTrue(identity["deployment_eligible"])
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            destination = Path(temporary) / "config.json"
            self.assertEqual(
                main(
                    [
                        "make-config", "--method", "exact", "--run-id", "pilot-exact",
                        "--output-root", "runs_phase_d_v4/pilot", "--out-config", str(destination),
                    ]
                ),
                0,
            )
            self.assertTrue(destination.is_file())


if __name__ == "__main__":
    unittest.main()
