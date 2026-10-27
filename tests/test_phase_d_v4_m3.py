"""Synthetic correctness and isolation tests for Phase D v4 milestone M3."""

from __future__ import annotations

import copy
import csv
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any

import torch

import phase_d_v4_diagnostics as diagnostics
from gradient_phase_d_resnet_v4 import (
    _diagnostic_fixture_state,
    gradient_metrics,
    main,
    probe_boundary_gradients,
    probe_normalization_transition,
    probe_parameter_updates,
    profile_steps,
)
from phase_d_v4_analysis import (
    GRADIENT_DIAGNOSTIC_COLUMNS,
    NORMALIZATION_PROBE_COLUMNS,
    PROFILE_SAMPLE_COLUMNS,
    UPDATE_PROBE_COLUMNS,
    aggregate_per_example_rows,
    finite_summary,
    fixed_probe_indices,
    first_target_attainment,
)
from phase_d_v4_io import capture_rng_state


def _assert_tree_equal(case: unittest.TestCase, left: Any, right: Any) -> None:
    """Recursively assert exact equality for tensors and nested state containers.

    Inputs are a test case and two trees. Nothing is returned or mutated; a
    mismatch raises through unittest with tensors compared bitwise.
    """

    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        case.assertEqual(left.keys(), right.keys())
        for key in left:
            _assert_tree_equal(case, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        case.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right, strict=True):
            _assert_tree_equal(case, left_item, right_item)
    else:
        case.assertEqual(left, right)


def _snapshot_state(
    model: torch.nn.Module,
    predictor: torch.nn.Module,
    optimizers: dict[str, torch.optim.Optimizer | None],
) -> dict[str, Any]:
    """Capture production parameters, optimizers, modes, gradients, and RNG.

    Inputs are live production-like objects. The detached output is suitable for
    exact isolation assertions and the function changes no source state.
    """

    return {
        "model": copy.deepcopy(model.state_dict()),
        "predictor": copy.deepcopy(predictor.state_dict()),
        "optimizers": {
            key: copy.deepcopy(value.state_dict()) if value is not None else None
            for key, value in optimizers.items()
        },
        "model_modes": [module.training for module in model.modules()],
        "predictor_modes": [module.training for module in predictor.modules()],
        "model_gradients": [
            parameter.grad.detach().clone() if parameter.grad is not None else None
            for parameter in model.parameters()
        ],
        "predictor_gradients": [
            parameter.grad.detach().clone() if parameter.grad is not None else None
            for parameter in predictor.parameters()
        ],
        "rng": capture_rng_state(),
    }


class AnalyticGradientMetricTests(unittest.TestCase):
    """Check diagnostic mathematics and aggregation against known tensors."""

    def test_exact_zero_orthogonal_near_zero_and_invalid_cases(self) -> None:
        """Known vectors produce the specified direction and validity outcomes."""

        target = torch.tensor([3.0, 4.0])
        exact = gradient_metrics(target, target, 1e-8)
        self.assertAlmostEqual(exact["cosine"], 1.0)
        self.assertAlmostEqual(exact["norm_ratio"], 1.0)
        self.assertAlmostEqual(exact["projection"], 1.0)
        self.assertAlmostEqual(exact["relative_residual"], 0.0)

        zero = gradient_metrics(target, torch.zeros_like(target), 1e-8)
        self.assertIsNone(zero["cosine"])
        self.assertAlmostEqual(zero["unfloored_relative_residual"], 1.0)
        self.assertAlmostEqual(zero["norm_ratio"], 0.0)
        self.assertAlmostEqual(zero["projection"], 0.0)

        orthogonal = gradient_metrics(
            torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]), 1e-8
        )
        self.assertAlmostEqual(orthogonal["cosine"], 0.0)
        self.assertAlmostEqual(orthogonal["norm_ratio"], 1.0)
        self.assertAlmostEqual(orthogonal["projection"], 0.0)

        near_zero = gradient_metrics(torch.tensor([1e-12]), torch.tensor([1e-12]), 1e-8)
        self.assertTrue(near_zero["true_near_zero"])
        self.assertIsNone(near_zero["cosine"])
        invalid = gradient_metrics(torch.tensor([float("nan")]), torch.tensor([0.0]), 1e-8)
        self.assertTrue(invalid["invalid"])
        self.assertIsNone(invalid["relative_residual"])

    def test_per_example_and_batch_global_aggregation_are_not_conflated(self) -> None:
        """Global ratios use concatenated tensors while sample summaries weight samples."""

        target = torch.tensor([[10.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        estimate = torch.tensor([[0.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
        rows = diagnostics._gradient_rows(
            target, estimate, units="known", floor=1e-8, sample_ids=[10, 11, 12]
        )
        per_example = [row for row in rows if row["aggregation"] == "per_example"]
        global_row = next(row for row in rows if row["aggregation"] == "batch_global")
        summary = aggregate_per_example_rows(per_example)
        self.assertEqual(summary["sample_count"], 3)
        self.assertAlmostEqual(summary["relative_residual_mean"], 1.0 / 3.0)
        self.assertNotAlmostEqual(
            summary["relative_residual_mean"], global_row["relative_residual"]
        )
        hand_global = 10.0 / math.sqrt(102.0)
        self.assertAlmostEqual(global_row["relative_residual"], hand_global)

    def test_parameter_agreement_reports_actual_pre_and_postclip_vectors(self) -> None:
        """Tensorwise parameter metrics reflect unequal exact/injected clipping."""

        rows = diagnostics._parameter_gradient_metrics(
            [torch.tensor([3.0, 4.0])], [torch.tensor([3.0, 4.0])],
            exact_coefficient=0.5, injected_coefficient=0.25,
        )
        self.assertAlmostEqual(rows[0]["parameter_cosine"], 1.0)
        self.assertAlmostEqual(rows[0]["parameter_norm_ratio"], 1.0)
        self.assertAlmostEqual(rows[0]["parameter_relative_residual"], 0.0)
        self.assertAlmostEqual(rows[1]["parameter_cosine"], 1.0)
        self.assertAlmostEqual(rows[1]["parameter_norm_ratio"], 0.5)
        self.assertAlmostEqual(rows[1]["parameter_projection"], 0.5)

    def test_summaries_use_every_finite_timed_sample(self) -> None:
        """Profile summary statistics match a hand-calculated sample list."""

        summary = finite_summary([1.0, 9.0, 5.0, float("nan")])
        self.assertEqual(summary["count"], 4)
        self.assertEqual(summary["finite_count"], 3)
        self.assertEqual(summary["invalid_count"], 1)
        self.assertAlmostEqual(summary["mean"], 5.0)
        self.assertAlmostEqual(summary["median"], 5.0)

    def test_fixed_probe_ids_use_private_seed_and_planned_defaults(self) -> None:
        """Probe selection is deterministic, disjoint, and independent of global RNG."""

        before = capture_rng_state()
        first = fixed_probe_indices(200)
        second = fixed_probe_indices(200)
        after = capture_rng_state()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertTrue(all(len(batch) == 32 for batch in first))
        self.assertEqual(len({value for batch in first for value in batch}), 128)
        _assert_tree_equal(self, before, after)


class CopiedStateProbeTests(unittest.TestCase):
    """Prove diagnostic probes preserve all production-like state."""

    def test_boundary_update_and_normalization_probes_are_isolated(self) -> None:
        """Every copied-state probe leaves models, optimizers, gradients, and RNG exact."""

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            state = _diagnostic_fixture_state(Path(temporary), torch.device("cpu"))
            config, model, predictor, optimizers = state[:4]
            inputs, labels, fixed_inputs, fixed_labels, scale = state[4:9]
            input_state = tuple(
                tensor.detach().clone()
                for tensor in (inputs, labels, fixed_inputs, fixed_labels)
            )
            before = _snapshot_state(model, predictor, optimizers)
            boundary = probe_boundary_gradients(
                model, predictor, inputs, labels, scale, config,
                sample_ids=[101, 102, 103, 104],
            )
            updates = probe_parameter_updates(
                model, predictor, optimizers, inputs, labels, fixed_inputs,
                fixed_labels, scale, config,
            )
            normalization = probe_normalization_transition(
                model, predictor, inputs, labels, scale, config,
                sample_source="deterministic_training",
            )
            after = _snapshot_state(model, predictor, optimizers)
            _assert_tree_equal(self, before, after)
            _assert_tree_equal(
                self, input_state, (inputs, labels, fixed_inputs, fixed_labels)
            )

            self.assertEqual(len(boundary["gradient_rows"]), 10)
            self.assertEqual(
                {row["units"] for row in boundary["gradient_rows"]},
                {"raw_mean_loss_boundary_gradient", "normalized_target"},
            )
            normalized_global = next(
                row for row in boundary["gradient_rows"]
                if row["units"] == "normalized_target"
                and row["aggregation"] == "batch_global"
            )
            self.assertGreaterEqual(normalized_global["controller_denominator"], 0.05)
            self.assertIsInstance(normalized_global["controller_floor_active"], bool)
            raw_global = next(
                row for row in boundary["gradient_rows"]
                if row["units"] == "raw_mean_loss_boundary_gradient"
                and row["aggregation"] == "batch_global"
            )
            self.assertAlmostEqual(
                normalized_global["true_norm"],
                raw_global["true_norm"] * len(labels) / scale,
                places=5,
            )
            floor_active = probe_boundary_gradients(
                model, predictor, inputs, labels, 1.0e12, config,
                sample_ids=[101, 102, 103, 104],
            )
            floor_global = next(
                row for row in floor_active["gradient_rows"]
                if row["units"] == "normalized_target"
                and row["aggregation"] == "batch_global"
            )
            self.assertTrue(floor_global["controller_floor_active"])
            self.assertEqual(floor_global["controller_denominator"], 0.05)
            self.assertEqual({row["parameter_stage"] for row in boundary["parameter_rows"]}, {"preclip", "postclip"})
            self.assertEqual({row["alternative"] for row in updates}, {"exact", "learned", "zero", "skip"})
            self.assertTrue(all(row["head_update"] is False for row in updates))
            self.assertEqual(
                {row["reference"] for row in normalization},
                {"conventional_exact_standard_bn", "v3_exact_frozen_head"},
            )
            conventional = next(row for row in normalization if row["head_training"])
            frozen = next(row for row in normalization if not row["head_training"])
            self.assertTrue(conventional["bn_buffers_changed"])
            self.assertFalse(frozen["bn_buffers_changed"])

    def test_update_probe_preserves_momentum_decay_and_learning_rate(self) -> None:
        """Hypothetical alternatives use cloned configured SGD rather than plain SGD."""

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            state = _diagnostic_fixture_state(Path(temporary), torch.device("cpu"))
            config, model, predictor, optimizers = state[:4]
            rows = probe_parameter_updates(
                model, predictor, optimizers, *state[4:9], config
            )
            group = optimizers["trunk"].param_groups[0]
            for row in rows:
                self.assertEqual(row["trunk_lr"], group["lr"])
                self.assertEqual(row["momentum"], group["momentum"])
                self.assertEqual(row["weight_decay"], group["weight_decay"])
            skip = next(row for row in rows if row["alternative"] == "skip")
            self.assertIsNone(skip["model_clip_coefficient"])
            zero = next(row for row in rows if row["alternative"] == "zero")
            self.assertIsNotNone(zero["model_clip_coefficient"])
            self.assertNotEqual(zero["same_loss_change"], 0.0)

    def test_profile_samples_and_controller_overhead_are_separate_and_isolated(self) -> None:
        """Bounded profiling returns every sample and does not mutate source state."""

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            state = _diagnostic_fixture_state(Path(temporary), torch.device("cpu"))
            config, model, predictor, optimizers = state[:4]
            controller = state[9]
            before = _snapshot_state(model, predictor, optimizers)
            controller_before = copy.deepcopy(controller.state_dict())
            profiled = profile_steps(
                model, predictor, optimizers, controller, state[4], state[5],
                state[8], config, samples=2, settling=1,
            )
            after = _snapshot_state(model, predictor, optimizers)
            _assert_tree_equal(self, before, after)
            _assert_tree_equal(self, controller_before, controller.state_dict())
            self.assertEqual(len(profiled["samples"]), 12)
            self.assertEqual(
                {row["component"] for row in profiled["samples"]},
                {
                    "plain_exact", "exact_plus_predictor", "learned_synthetic",
                    "zero_synthetic_shadow", "skip_synthetic_shadow",
                    "controller_overhead",
                },
            )
            for component in profiled["summary"]["components"].values():
                self.assertEqual(component["count"], 2)
                self.assertEqual(component["finite_count"], 2)


class AnalysisAndOutputTests(unittest.TestCase):
    """Check endpoint semantics and planned M3 output schemas."""

    def test_attainment_in_warmup_and_unattained_target_are_preserved(self) -> None:
        """First attainment uses observed warm-up endpoints and censors no target."""

        records = [
            {"stage": "warmup", "global_epoch": 0, "validation_accuracy": 0.86, "observation_time_seconds": 5.0},
            {"stage": "postwarm", "global_epoch": 1, "validation_accuracy": 0.88, "observation_time_seconds": 9.0},
        ]
        results = first_target_attainment(records, [0.85, 0.89])
        self.assertTrue(results[0]["reached"])
        self.assertEqual(results[0]["stage"], "warmup")
        self.assertEqual(results[0]["attainment_time_seconds"], 5.0)
        self.assertFalse(results[1]["reached"])
        self.assertIsNone(results[1]["attainment_time_seconds"])
        self.assertEqual(results[1]["observation_horizon_seconds"], 9.0)

    def test_synthetic_cli_writes_exact_diagnostic_and_profile_schemas(self) -> None:
        """M3 commands create planned local files without invoking scientific run."""

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            root = Path(temporary)
            diagnose = root / "diagnose"
            profile = root / "profile"
            self.assertEqual(main(["diagnose", "--device", "cpu", "--out", str(diagnose)]), 0)
            for filename, columns in (
                ("gradient_diagnostics.csv", GRADIENT_DIAGNOSTIC_COLUMNS),
                ("update_probes.csv", UPDATE_PROBE_COLUMNS),
                ("normalization_probes.csv", NORMALIZATION_PROBE_COLUMNS),
            ):
                with (diagnose / filename).open(encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    self.assertEqual(tuple(reader.fieldnames or ()), columns)
                    self.assertTrue(list(reader))
            self.assertEqual(main(["profile", "--device", "cpu", "--out", str(profile)]), 0)
            with (profile / "profile_samples.csv").open(encoding="utf-8", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(tuple(reader.fieldnames or ()), PROFILE_SAMPLE_COLUMNS)
                self.assertEqual(len(list(reader)), 72)
            summary = json.loads((profile / "profile_summary.json").read_text(encoding="utf-8"))
            self.assertTrue(summary["engineering_only"])
            self.assertEqual(summary["components"]["controller_overhead"]["count"], 12)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
class CudaDiagnosticSmokeTests(unittest.TestCase):
    """Exercise a small copied-state diagnostic path on CUDA when available."""

    def test_cuda_boundary_probe_is_finite_and_isolated(self) -> None:
        """CUDA produces bounded metrics while preserving production-like state."""

        with tempfile.TemporaryDirectory(dir=Path.cwd()) as temporary:
            state = _diagnostic_fixture_state(Path(temporary), torch.device("cuda"))
            config, model, predictor, optimizers = state[:4]
            before = _snapshot_state(model, predictor, optimizers)
            result = probe_boundary_gradients(
                model, predictor, state[4], state[5], state[8], config,
                sample_ids=[0, 1, 2, 3],
            )
            after = _snapshot_state(model, predictor, optimizers)
            _assert_tree_equal(self, before, after)
            global_rows = [
                row for row in result["gradient_rows"] if row["aggregation"] == "batch_global"
            ]
            self.assertEqual(len(global_rows), 2)
            self.assertTrue(all(math.isfinite(row["residual_norm"]) for row in global_rows))


if __name__ == "__main__":
    unittest.main()
