"""M1 mathematical, optimizer, normalization, and v3 parity tests."""

from __future__ import annotations

import copy
import importlib
import math
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from gradient_phase_d_resnet_v4 import (
    GradientPredictor,
    HistoricalFeedbackController,
    SplitResNet,
    build_models,
    build_optimizers,
    estimate_initial_scale,
    exact_step,
    relative_error,
    scheduled_exact,
    set_epoch_lr,
    set_training_mode,
    synthetic_step,
)
from phase_d_v4_io import NormalizationMode, RunConfig, SyntheticBehavior, load_run_config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "phase_d_v4_reference.json"
CPU_RTOL = 1e-5
CPU_ATOL = 1e-6
CUDA_RTOL = 1e-4
CUDA_ATOL = 3e-5


def _toy_config(
    *,
    momentum: float = 0.9,
    weight_decay: float = 0.0005,
    clip: float = 5.0,
) -> RunConfig:
    """Return a validated tiny-model config while preserving scientific formulas."""

    values = load_run_config(CONFIG_PATH).to_dict()
    scientific = values["scientific"]
    scientific["model"].update(width=4, blocks_per_stage=1, boundary_shape=[None, 8, 16, 16])
    scientific["predictor"].update(input_channels=8, bottleneck_width=4)
    scientific["training"].update(
        warmup_epochs=1,
        postwarm_epochs=1,
        total_epochs=2,
        batch_size=8,
        expected_training_batches_per_epoch=2,
        expected_final_batch_size=3,
    )
    for name in ("trunk", "head"):
        scientific["optimizers"][name].update(momentum=momentum, weight_decay=weight_decay)
    scientific["gradient_contract"]["global_l2_clip"] = clip
    return RunConfig(values)


def _v3_args(config: RunConfig) -> SimpleNamespace:
    """Translate resolved toy values into the arguments consumed by v3 helpers."""

    scientific = config.values["scientific"]
    model = scientific["model"]
    predictor = scientific["predictor"]
    optimizers = scientific["optimizers"]
    gradient = scientific["gradient_contract"]
    controller = scientific["controller"]
    training = scientific["training"]
    return SimpleNamespace(
        width=model["width"],
        blocks=model["blocks_per_stage"],
        predictor_width=predictor["bottleneck_width"],
        lr=optimizers["trunk"]["lr_start"],
        lr_min=scientific["learning_rate"]["lr_min"],
        pred_lr=optimizers["predictor"]["lr"],
        weight_decay=optimizers["trunk"]["weight_decay"],
        cosine_weight=gradient["cosine_weight"],
        clip_grad=gradient["global_l2_clip"],
        warmup_epochs=training["warmup_epochs"],
        epochs=training["postwarm_epochs"],
        low_q=controller["low_quantile"],
        high_q=controller["high_quantile"],
        error_window=controller["history_window"],
        ema_beta=controller["ema_old_value_weight"],
        emergency_ratio=controller["emergency_multiplier"],
        max_synthetic=controller["maximum_quota"],
        smoke_test=False,
    )


def _v3_module() -> Any:
    """Import v3 in memory with bytecode writing disabled and without calling main."""

    sys.dont_write_bytecode = True
    return importlib.import_module("gradient_phase_d_resnet_v3")


def _assert_module_close(
    case: unittest.TestCase,
    left: nn.Module,
    right: nn.Module,
    *,
    rtol: float = 0.0,
    atol: float = 0.0,
) -> None:
    """Assert every parameter and buffer in two modules is numerically equal."""

    case.assertEqual(set(left.state_dict()), set(right.state_dict()))
    for name, tensor in left.state_dict().items():
        torch.testing.assert_close(tensor, right.state_dict()[name], rtol=rtol, atol=atol, msg=name)


def _assert_nested_close(case: unittest.TestCase, left: Any, right: Any) -> None:
    """Recursively compare optimizer states containing tensors and primitives."""

    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=CPU_RTOL, atol=CPU_ATOL)
    elif isinstance(left, dict):
        case.assertEqual(set(left), set(right))
        for key in left:
            _assert_nested_close(case, left[key], right[key])
    elif isinstance(left, (list, tuple)):
        case.assertEqual(len(left), len(right))
        for left_item, right_item in zip(left, right):
            _assert_nested_close(case, left_item, right_item)
    else:
        case.assertEqual(left, right)


def _bn_buffers(module: nn.Module) -> dict[str, torch.Tensor]:
    """Clone BatchNorm running buffers so mutation can be checked explicitly."""

    result: dict[str, torch.Tensor] = {}
    for name, child in module.named_modules():
        if isinstance(child, nn.modules.batchnorm._BatchNorm):
            result[f"{name}.running_mean"] = child.running_mean.detach().clone()
            result[f"{name}.running_var"] = child.running_var.detach().clone()
            result[f"{name}.num_batches_tracked"] = child.num_batches_tracked.detach().clone()
    return result


def _buffers_changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> bool:
    """Return whether at least one corresponding BatchNorm buffer changed."""

    return any(not torch.equal(before[name], after[name]) for name in before)


def _make_batch(batch_size: int, device: torch.device, seed: int = 100) -> tuple[torch.Tensor, torch.Tensor]:
    """Create deterministic synthetic CIFAR-shaped inputs and labels on a device."""

    generator = torch.Generator(device="cpu").manual_seed(seed + batch_size)
    inputs = torch.randn(batch_size, 3, 32, 32, generator=generator).to(device)
    labels = torch.randint(0, 10, (batch_size,), generator=generator).to(device)
    return inputs, labels


class GradientMathematicsTests(unittest.TestCase):
    """Verify VJP injection and batch-size scaling under both BN regimes."""

    def _check_injection(self, device: torch.device) -> tuple[float, float]:
        """Compare gradients on one device and return maximum absolute/relative errors."""

        tolerance = (CUDA_RTOL, CUDA_ATOL) if device.type == "cuda" else (CPU_RTOL, CPU_ATOL)
        maximum_absolute = 0.0
        maximum_relative = 0.0
        for mode in (NormalizationMode.FROZEN_HEAD_POSTWARM, NormalizationMode.STANDARD_BN):
            for batch_size in (1, 8, 32):
                with self.subTest(device=device.type, mode=mode.value, batch_size=batch_size):
                    torch.manual_seed(800 + batch_size)
                    direct = SplitResNet(4, 1).to(device)
                    injected = copy.deepcopy(direct)
                    set_training_mode(direct, None, "postwarm", mode)
                    set_training_mode(injected, None, "postwarm", mode)
                    inputs, labels = _make_batch(batch_size, device)

                    direct.zero_grad(set_to_none=True)
                    direct_boundary = direct.trunk(inputs)
                    direct_loss = F.cross_entropy(direct.head(direct_boundary), labels)
                    direct_loss.backward()
                    direct_gradients = [parameter.grad.detach().clone() for parameter in direct.trunk.parameters()]

                    injected.zero_grad(set_to_none=True)
                    injected_boundary = injected.trunk(inputs)
                    injected_loss = F.cross_entropy(injected.head(injected_boundary), labels)
                    boundary_gradient = torch.autograd.grad(injected_loss, injected_boundary)[0].detach()
                    injected_boundary.backward(boundary_gradient)
                    for expected, parameter in zip(direct_gradients, injected.trunk.parameters()):
                        difference = (parameter.grad - expected).abs()
                        maximum_absolute = max(maximum_absolute, float(difference.max().item()))
                        relative = difference / expected.abs().clamp_min(tolerance[1])
                        maximum_relative = max(maximum_relative, float(relative.max().item()))
                        torch.testing.assert_close(parameter.grad, expected, rtol=tolerance[0], atol=tolerance[1])
                    self.assertTrue(all(parameter.grad is None for parameter in injected.head.parameters()))
        return maximum_absolute, maximum_relative

    def test_cpu_true_gradient_injection(self) -> None:
        """Direct and injected trunk VJPs agree on CPU for sizes 1, 8, and 32."""

        maximum_absolute, maximum_relative = self._check_injection(torch.device("cpu"))
        print(f"CPU injection max_abs={maximum_absolute:.3e} max_relative={maximum_relative:.3e}")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_true_gradient_injection(self) -> None:
        """Direct and injected trunk VJPs agree on CUDA at the declared tolerance."""

        maximum_absolute, maximum_relative = self._check_injection(torch.device("cuda"))
        print(f"CUDA injection max_abs={maximum_absolute:.3e} max_relative={maximum_relative:.3e}")

    def test_scale_round_trip_including_last_batch(self) -> None:
        """The normalized target reconstructs mean-loss gradients for unequal batches."""

        for mode in (NormalizationMode.FROZEN_HEAD_POSTWARM, NormalizationMode.STANDARD_BN):
            for batch_size in (1, 8, 32):
                with self.subTest(mode=mode.value, batch_size=batch_size):
                    torch.manual_seed(900 + batch_size)
                    model = SplitResNet(4, 1)
                    set_training_mode(model, None, "postwarm", mode)
                    inputs, labels = _make_batch(batch_size, torch.device("cpu"), seed=200)
                    boundary = model.trunk(inputs)
                    loss = F.cross_entropy(model.head(boundary), labels)
                    mean_gradient = torch.autograd.grad(loss, boundary)[0].detach()
                    scale = 0.037
                    target = batch_size * mean_gradient / scale
                    reconstructed = scale * target / batch_size
                    torch.testing.assert_close(reconstructed, mean_gradient, rtol=CPU_RTOL, atol=CPU_ATOL)

    def test_initial_scale_restores_modes_and_bn_buffers(self) -> None:
        """Scale estimation is positive and leaves mixed training modes and BN state intact."""

        model = SplitResNet(4, 1)
        set_training_mode(model, None, "postwarm", NormalizationMode.FROZEN_HEAD_POSTWARM)
        flags = [module.training for module in model.modules()]
        buffers = _bn_buffers(model)
        inputs, labels = _make_batch(3, torch.device("cpu"), seed=300)
        scale = estimate_initial_scale(model, inputs, labels)
        self.assertGreaterEqual(scale, 1e-6)
        self.assertEqual(flags, [module.training for module in model.modules()])
        for name, value in buffers.items():
            torch.testing.assert_close(value, _bn_buffers(model)[name], rtol=0, atol=0)


class OptimizerAndModeTests(unittest.TestCase):
    """Verify momentum, skip, clipping, predictor isolation, and BatchNorm semantics."""

    def test_zero_uses_momentum_and_decay_while_skip_is_bitwise_unchanged(self) -> None:
        """Zero steps update SGD state/parameters, while skip leaves both untouched."""

        config = _toy_config()
        torch.manual_seed(1001)
        base_model, base_predictor = build_models(config, "cpu")
        base_optimizers = build_optimizers(base_model, base_predictor, config)
        set_training_mode(base_model, base_predictor, "postwarm", NormalizationMode.FROZEN_HEAD_POSTWARM)
        inputs, labels = _make_batch(4, torch.device("cpu"), seed=400)
        exact_step(base_model, base_predictor, base_optimizers, inputs, labels, 0.1, config, fit_predictor=False)

        zero_model, zero_predictor = copy.deepcopy(base_model), copy.deepcopy(base_predictor)
        skip_model, skip_predictor = copy.deepcopy(base_model), copy.deepcopy(base_predictor)
        zero_optimizers = build_optimizers(zero_model, zero_predictor, config)
        skip_optimizers = build_optimizers(skip_model, skip_predictor, config)
        for name in ("trunk", "head", "predictor"):
            if base_optimizers[name] is not None:
                zero_optimizers[name].load_state_dict(copy.deepcopy(base_optimizers[name].state_dict()))
                skip_optimizers[name].load_state_dict(copy.deepcopy(base_optimizers[name].state_dict()))

        old_parameters = [parameter.detach().clone() for parameter in zero_model.trunk.parameters()]
        old_buffers = [zero_optimizers["trunk"].state[parameter]["momentum_buffer"].detach().clone() for parameter in zero_model.trunk.parameters()]
        lr = zero_optimizers["trunk"].param_groups[0]["lr"]
        settings = config.values["scientific"]["optimizers"]["trunk"]
        synthetic_step(zero_model, zero_predictor, zero_optimizers, inputs, labels, 0.1, config, kind="zero")
        for parameter, old_parameter, old_buffer in zip(zero_model.trunk.parameters(), old_parameters, old_buffers):
            expected_buffer = settings["momentum"] * old_buffer + settings["weight_decay"] * old_parameter
            expected_parameter = old_parameter - lr * expected_buffer
            torch.testing.assert_close(
                zero_optimizers["trunk"].state[parameter]["momentum_buffer"],
                expected_buffer,
                rtol=CPU_RTOL,
                atol=CPU_ATOL,
            )
            torch.testing.assert_close(parameter, expected_parameter, rtol=CPU_RTOL, atol=CPU_ATOL)

        skip_parameters = [parameter.detach().clone() for parameter in skip_model.parameters()]
        skip_state = copy.deepcopy(skip_optimizers["trunk"].state_dict())
        result = synthetic_step(skip_model, skip_predictor, skip_optimizers, inputs, labels, 0.1, config, kind="skip")
        self.assertEqual(result.trunk_optimizer_steps, 0)
        for before, after in zip(skip_parameters, skip_model.parameters()):
            torch.testing.assert_close(before, after, rtol=0, atol=0)
        _assert_nested_close(self, skip_state, skip_optimizers["trunk"].state_dict())

    def test_zero_without_momentum_or_decay_changes_no_parameters(self) -> None:
        """A true zero step is parameter-neutral when SGD has no momentum or decay."""

        config = _toy_config(momentum=0.0, weight_decay=0.0)
        torch.manual_seed(1002)
        model, predictor = build_models(config, "cpu")
        optimizers = build_optimizers(model, predictor, config)
        set_training_mode(model, predictor, "postwarm", NormalizationMode.FROZEN_HEAD_POSTWARM)
        inputs, labels = _make_batch(4, torch.device("cpu"), seed=401)
        before = [parameter.detach().clone() for parameter in model.trunk.parameters()]
        result = synthetic_step(model, predictor, optimizers, inputs, labels, 0.1, config, kind="zero")
        self.assertEqual(result.zero_backwards, 1)
        for expected, parameter in zip(before, model.trunk.parameters()):
            torch.testing.assert_close(expected, parameter, rtol=0, atol=0)

    def test_synthetic_preserves_predictor_and_never_executes_head(self) -> None:
        """Learned injection leaves predictor state/gradients unchanged and bypasses head."""

        config = _toy_config()
        torch.manual_seed(1003)
        model, predictor = build_models(config, "cpu")
        assert predictor is not None
        optimizers = build_optimizers(model, predictor, config)
        set_training_mode(model, predictor, "postwarm", NormalizationMode.FROZEN_HEAD_POSTWARM)
        inputs, labels = _make_batch(4, torch.device("cpu"), seed=402)
        exact_step(model, predictor, optimizers, inputs, labels, 0.1, config, fit_predictor=True)
        predictor_parameters = [parameter.detach().clone() for parameter in predictor.parameters()]
        predictor_gradients = [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in predictor.parameters()]
        predictor_state = copy.deepcopy(optimizers["predictor"].state_dict())
        calls = []
        handle = model.head.register_forward_hook(lambda *_: calls.append(1))
        try:
            synthetic_step(model, predictor, optimizers, inputs, labels, 0.1, config, kind="learned")
        finally:
            handle.remove()
        self.assertEqual(calls, [])
        self.assertTrue(all(parameter.grad is None for parameter in model.head.parameters()))
        for expected, parameter, gradient in zip(predictor_parameters, predictor.parameters(), predictor_gradients):
            torch.testing.assert_close(expected, parameter, rtol=0, atol=0)
            if gradient is None:
                self.assertIsNone(parameter.grad)
            else:
                torch.testing.assert_close(gradient, parameter.grad, rtol=0, atol=0)
        _assert_nested_close(self, predictor_state, optimizers["predictor"].state_dict())

    def test_predictor_fit_cannot_change_classifier_update(self) -> None:
        """Detached predictor fitting leaves classifier parameters and SGD state identical."""

        config = _toy_config()
        torch.manual_seed(1004)
        fit_model, fit_predictor = build_models(config, "cpu")
        plain_model, plain_predictor = copy.deepcopy(fit_model), copy.deepcopy(fit_predictor)
        fit_optimizers = build_optimizers(fit_model, fit_predictor, config)
        plain_optimizers = build_optimizers(plain_model, plain_predictor, config)
        set_training_mode(fit_model, fit_predictor, "postwarm", NormalizationMode.FROZEN_HEAD_POSTWARM)
        set_training_mode(plain_model, plain_predictor, "postwarm", NormalizationMode.FROZEN_HEAD_POSTWARM)
        inputs, labels = _make_batch(5, torch.device("cpu"), seed=403)
        exact_step(fit_model, fit_predictor, fit_optimizers, inputs, labels, 0.1, config, fit_predictor=True)
        exact_step(plain_model, plain_predictor, plain_optimizers, inputs, labels, 0.1, config, fit_predictor=False)
        _assert_module_close(self, fit_model, plain_model, rtol=CPU_RTOL, atol=CPU_ATOL)
        for name in ("trunk", "head"):
            _assert_nested_close(self, fit_optimizers[name].state_dict(), plain_optimizers[name].state_dict())

    def test_clipping_groups_and_lr_endpoints(self) -> None:
        """Exact uses joint clipping, synthetic uses trunk clipping, and LR reaches endpoints."""

        config = _toy_config(clip=0.01)
        torch.manual_seed(1005)
        model, predictor = build_models(config, "cpu")
        optimizers = build_optimizers(model, predictor, config)
        set_training_mode(model, predictor, "postwarm", NormalizationMode.FROZEN_HEAD_POSTWARM)
        inputs, labels = _make_batch(6, torch.device("cpu"), seed=404)
        exact_result = exact_step(model, predictor, optimizers, inputs * 20, labels, 0.1, config, fit_predictor=False)
        self.assertLess(exact_result.metrics["model_clip_coefficient"], 1.0)
        joint_norm = torch.stack([parameter.grad.norm() for parameter in model.parameters() if parameter.grad is not None]).norm()
        self.assertLessEqual(float(joint_norm), 0.01001)
        synthetic_result = synthetic_step(model, predictor, optimizers, inputs, labels, 0.1, config, kind="learned")
        self.assertIn("trunk_clip_coefficient", synthetic_result.metrics)
        self.assertTrue(all(parameter.grad is None for parameter in model.head.parameters()))
        self.assertAlmostEqual(set_epoch_lr(optimizers, config, 0), 0.04, places=12)
        self.assertAlmostEqual(set_epoch_lr(optimizers, config, 1), 0.001, places=12)

    def test_batchnorm_modes_and_affine_training(self) -> None:
        """Warm-up, frozen-head, standard, synthetic, and eval modes mutate intended buffers."""

        config = _toy_config()
        inputs, labels = _make_batch(5, torch.device("cpu"), seed=405)

        torch.manual_seed(1006)
        warmup_model, warmup_predictor = build_models(config, "cpu")
        warmup_optimizers = build_optimizers(warmup_model, warmup_predictor, config)
        set_training_mode(warmup_model, warmup_predictor, "warmup", NormalizationMode.FROZEN_HEAD_POSTWARM)
        warmup_trunk_before = _bn_buffers(warmup_model.trunk)
        warmup_head_before = _bn_buffers(warmup_model.head)
        exact_step(
            warmup_model,
            warmup_predictor,
            warmup_optimizers,
            inputs,
            labels,
            0.1,
            config,
            fit_predictor=False,
        )
        self.assertTrue(_buffers_changed(warmup_trunk_before, _bn_buffers(warmup_model.trunk)))
        self.assertTrue(_buffers_changed(warmup_head_before, _bn_buffers(warmup_model.head)))

        for normalization, expect_head_change in (
            (NormalizationMode.FROZEN_HEAD_POSTWARM, False),
            (NormalizationMode.STANDARD_BN, True),
        ):
            with self.subTest(normalization=normalization.value):
                torch.manual_seed(1006)
                model, predictor = build_models(config, "cpu")
                optimizers = build_optimizers(model, predictor, config)
                set_training_mode(model, predictor, "postwarm", normalization)
                trunk_before, head_before = _bn_buffers(model.trunk), _bn_buffers(model.head)
                head_affine_before = [parameter.detach().clone() for module in model.head.modules() if isinstance(module, nn.BatchNorm2d) for parameter in module.parameters()]
                exact_step(model, predictor, optimizers, inputs, labels, 0.1, config, fit_predictor=False)
                self.assertTrue(_buffers_changed(trunk_before, _bn_buffers(model.trunk)))
                self.assertEqual(_buffers_changed(head_before, _bn_buffers(model.head)), expect_head_change)
                head_affine_after = [parameter for module in model.head.modules() if isinstance(module, nn.BatchNorm2d) for parameter in module.parameters()]
                self.assertTrue(all(parameter.requires_grad for parameter in head_affine_after))
                self.assertTrue(any(not torch.equal(before, after) for before, after in zip(head_affine_before, head_affine_after)))

        torch.manual_seed(1007)
        model, predictor = build_models(config, "cpu")
        optimizers = build_optimizers(model, predictor, config)
        set_training_mode(model, predictor, "postwarm", NormalizationMode.FROZEN_HEAD_POSTWARM)
        trunk_before, head_before = _bn_buffers(model.trunk), _bn_buffers(model.head)
        synthetic_step(model, predictor, optimizers, inputs, labels, 0.1, config, kind="skip")
        self.assertTrue(_buffers_changed(trunk_before, _bn_buffers(model.trunk)))
        self.assertFalse(_buffers_changed(head_before, _bn_buffers(model.head)))

        set_training_mode(model, predictor, "calibration", NormalizationMode.FROZEN_HEAD_POSTWARM)
        calibration_trunk_before = _bn_buffers(model.trunk)
        calibration_head_before = _bn_buffers(model.head)
        with torch.no_grad():
            boundary = model.trunk(inputs)
            model.head(boundary)
        self.assertTrue(_buffers_changed(calibration_trunk_before, _bn_buffers(model.trunk)))
        self.assertFalse(_buffers_changed(calibration_head_before, _bn_buffers(model.head)))
        set_training_mode(model, predictor, "evaluation", NormalizationMode.FROZEN_HEAD_POSTWARM)
        eval_before = _bn_buffers(model)
        with torch.no_grad():
            model(inputs)
        self.assertFalse(_buffers_changed(eval_before, _bn_buffers(model)))


class HistoricalParityTests(unittest.TestCase):
    """Compare M1 compatibility paths directly against imported v3 helpers."""

    def test_exact_and_synthetic_steps_match_v3(self) -> None:
        """A short exact/learned sequence matches v3 outputs and optimizer buffers."""

        v3 = _v3_module()
        config = _toy_config()
        args = _v3_args(config)
        torch.manual_seed(1100)
        model, predictor = build_models(config, "cpu")
        assert predictor is not None
        v3_model = v3.SplitResNet(4, 1)
        v3_predictor = v3.GradientPredictor(8, 4)
        v3_model.load_state_dict(copy.deepcopy(model.state_dict()))
        v3_predictor.load_state_dict(copy.deepcopy(predictor.state_dict()))
        optimizers = build_optimizers(model, predictor, config)
        v3_optimizers = v3.opts_for(v3_model, v3_predictor, args)
        set_training_mode(model, predictor, "postwarm", NormalizationMode.FROZEN_HEAD_POSTWARM)
        v3_model.postwarm_mode()
        v3_predictor.train()
        inputs, labels = _make_batch(5, torch.device("cpu"), seed=500)

        result = exact_step(model, predictor, optimizers, inputs, labels, 0.17, config, fit_predictor=True)
        v3_loss, v3_error = v3.exact_step(v3_model, v3_predictor, v3_optimizers, inputs, labels, 0.17, args, fit_predictor=True)
        self.assertAlmostEqual(result.metrics["classifier_loss"], v3_loss, places=7)
        self.assertAlmostEqual(result.metrics["error_abs"], v3_error, places=7)
        _assert_module_close(self, model, v3_model, rtol=CPU_RTOL, atol=CPU_ATOL)
        _assert_module_close(self, predictor, v3_predictor, rtol=CPU_RTOL, atol=CPU_ATOL)

        synthetic_step(
            model,
            predictor,
            optimizers,
            inputs,
            labels,
            0.17,
            config,
            kind="learned",
        )
        v3.synthetic_step(
            v3_model,
            v3_predictor,
            v3_optimizers,
            inputs,
            labels,
            0.17,
            args,
        )
        second_inputs, second_labels = _make_batch(3, torch.device("cpu"), seed=502)
        exact_step(
            model,
            predictor,
            optimizers,
            second_inputs,
            second_labels,
            0.17,
            config,
            fit_predictor=True,
        )
        v3.exact_step(
            v3_model,
            v3_predictor,
            v3_optimizers,
            second_inputs,
            second_labels,
            0.17,
            args,
            fit_predictor=True,
        )
        synthetic_step(
            model,
            predictor,
            optimizers,
            second_inputs,
            second_labels,
            0.17,
            config,
            kind="learned",
        )
        v3.synthetic_step(
            v3_model,
            v3_predictor,
            v3_optimizers,
            second_inputs,
            second_labels,
            0.17,
            args,
        )
        _assert_module_close(self, model, v3_model, rtol=CPU_RTOL, atol=CPU_ATOL)
        _assert_module_close(self, predictor, v3_predictor, rtol=0, atol=0)
        for name, v3_optimizer in zip(("trunk", "head", "predictor"), v3_optimizers):
            _assert_nested_close(self, optimizers[name].state_dict(), v3_optimizer.state_dict())

    def test_historical_controllers_and_schedule_match_v3(self) -> None:
        """Relative/absolute decisions and scheduled phase boundaries equal v3."""

        v3 = _v3_module()
        config = _toy_config()
        args = _v3_args(config)
        calibration = {
            "relative_errors": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1],
            "q_abs_low": 0.6,
            "q_abs_high": 1.0,
            "median_abs": 0.8,
        }
        observations = [(0.2, 0.55), (0.8, 0.75), (1.5, 1.2), (0.7, 0.65)]
        for kind in ("relative_feedback", "absolute_feedback"):
            with self.subTest(kind=kind):
                current = HistoricalFeedbackController(kind, config, calibration)
                historical = v3.Feedback(kind, args, calibration)
                for absolute, relative in observations:
                    current_result = current.choose(absolute, relative)
                    historical_result = historical.choose(absolute, relative)
                    self.assertEqual(current_result[0], historical_result[0])
                    for key in current_result[1]:
                        self.assertAlmostEqual(current_result[1][key], historical_result[1][key], places=12)
        for epoch in range(12):
            for step in (0, 1, 4, 5, 100, 1406, 1407):
                self.assertEqual(scheduled_exact(step, epoch, 12), v3.scheduled_exact(step, epoch, 12))

    def test_sparse_head_exact_step_matches_v3_joint_clipping(self) -> None:
        """Exact-trunk sparse-head behavior retains v3 joint clipping and skips head SGD."""

        v3 = _v3_module()
        config = _toy_config(clip=0.03)
        args = _v3_args(config)
        torch.manual_seed(1101)
        model, predictor = build_models(config, "cpu")
        assert predictor is not None
        v3_model = v3.SplitResNet(4, 1)
        v3_predictor = v3.GradientPredictor(8, 4)
        v3_model.load_state_dict(copy.deepcopy(model.state_dict()))
        v3_predictor.load_state_dict(copy.deepcopy(predictor.state_dict()))
        optimizers = build_optimizers(model, predictor, config)
        v3_optimizers = v3.opts_for(v3_model, v3_predictor, args)
        set_training_mode(model, predictor, "postwarm", NormalizationMode.FROZEN_HEAD_POSTWARM)
        v3_model.postwarm_mode()
        inputs, labels = _make_batch(7, torch.device("cpu"), seed=501)
        head_before = [parameter.detach().clone() for parameter in model.head.parameters()]
        result = exact_step(
            model,
            predictor,
            optimizers,
            inputs,
            labels,
            0.2,
            config,
            fit_predictor=False,
            head_update=False,
        )
        v3.exact_step(
            v3_model,
            None,
            v3_optimizers,
            inputs,
            labels,
            0.2,
            args,
            fit_predictor=False,
            head_update=False,
        )
        self.assertEqual(result.head_optimizer_steps, 0)
        for expected, parameter in zip(head_before, model.head.parameters()):
            torch.testing.assert_close(expected, parameter, rtol=0, atol=0)
        self.assertTrue(all(parameter.grad is None for parameter in model.head.parameters()))
        _assert_module_close(self, model, v3_model, rtol=CPU_RTOL, atol=CPU_ATOL)
        for name, v3_optimizer in zip(("trunk", "head"), v3_optimizers[:2]):
            _assert_nested_close(self, optimizers[name].state_dict(), v3_optimizer.state_dict())

    def test_existing_v3_small_audits_pass_in_memory(self) -> None:
        """The historical injection and controller audit helpers pass without file output."""

        v3 = _v3_module()
        v3.audit_injection(torch.device("cpu"))
        v3.audit_controller()

    def test_relative_error_matches_v3_with_floor_active_and_inactive(self) -> None:
        """Batch-global relative RMSE agrees with v3 on both sides of its floor."""

        v3 = _v3_module()
        for magnitude in (0.001, 0.5):
            target = torch.full((3, 2, 4, 4), magnitude)
            estimate = target * 0.25
            self.assertAlmostEqual(relative_error(estimate, target), v3.rel_error(estimate, target), places=12)


if __name__ == "__main__":
    unittest.main()
