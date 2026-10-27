"""Copied-state diagnostic probes for Phase D v4 milestone M3.

All public probes accept fixed tensors rather than loaders and restore RNG state.
They operate only on deep copies so production training state cannot be changed.
"""

from __future__ import annotations

import copy
import math
import time
from typing import Any, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from phase_d_v4_analysis import finite_summary
from phase_d_v4_io import RunConfig, capture_rng_state, restore_rng_state


def gradient_metrics(
    target: Tensor, estimate: Tensor, floor: float
) -> dict[str, float | bool | None]:
    """Measure direction, magnitude, residual, ratio, and projection for tensors.

    Inputs are same-shaped gradients in one declared unit and a positive norm
    floor in that unit. The output contains detached scalars and validity flags.
    Undefined cosine or unfloored ratios are null; no state is changed.
    """

    if target.shape != estimate.shape:
        raise ValueError("target and estimate must have identical shapes")
    if floor <= 0:
        raise ValueError("gradient diagnostic floor must be positive")
    target_flat = target.detach().reshape(-1).to(dtype=torch.float64)
    estimate_flat = estimate.detach().reshape(-1).to(dtype=torch.float64)
    invalid = not bool(torch.isfinite(target_flat).all() and torch.isfinite(estimate_flat).all())
    if invalid:
        return {
            "valid": False, "invalid": True, "true_near_zero": False,
            "predicted_near_zero": False, "cosine": None, "true_norm": None,
            "predicted_norm": None, "residual_norm": None, "relative_residual": None,
            "unfloored_relative_residual": None, "norm_ratio": None, "projection": None,
            "true_rms": None, "predicted_rms": None, "residual_rms": None,
        }
    true_norm = float(target_flat.norm().item())
    predicted_norm = float(estimate_flat.norm().item())
    residual = estimate_flat - target_flat
    residual_norm = float(residual.norm().item())
    dot = float(torch.dot(estimate_flat, target_flat).item())
    true_near_zero = true_norm <= floor
    predicted_near_zero = predicted_norm <= floor
    cosine = None
    if not true_near_zero and not predicted_near_zero:
        cosine = dot / (true_norm * predicted_norm)
    coordinate_count = max(1, target_flat.numel())
    return {
        "valid": True,
        "invalid": False,
        "true_near_zero": true_near_zero,
        "predicted_near_zero": predicted_near_zero,
        "cosine": cosine,
        "true_norm": true_norm,
        "predicted_norm": predicted_norm,
        "residual_norm": residual_norm,
        "relative_residual": residual_norm / max(true_norm, floor),
        "unfloored_relative_residual": residual_norm / true_norm if true_norm > 0 else None,
        "norm_ratio": predicted_norm / max(true_norm, floor),
        "projection": dot / max(true_norm * true_norm, floor * floor),
        "true_rms": true_norm / math.sqrt(coordinate_count),
        "predicted_rms": predicted_norm / math.sqrt(coordinate_count),
        "residual_rms": residual_norm / math.sqrt(coordinate_count),
    }


def _gradient_rows(
    target: Tensor,
    estimate: Tensor,
    *,
    units: str,
    floor: float,
    sample_ids: list[int],
) -> list[dict[str, Any]]:
    """Build separate per-example and batch-global rows in declared units.

    Inputs are ``[B,...]`` tensors and B fixed sample IDs. The output has one row
    per example plus one global row; global ratios are not averages of ratios.
    """

    if target.shape[0] != len(sample_ids):
        raise ValueError("sample_ids must match the gradient batch dimension")
    rows: list[dict[str, Any]] = []
    for index, sample_id in enumerate(sample_ids):
        row = gradient_metrics(target[index], estimate[index], floor)
        row.update(
            sample_id=int(sample_id), aggregation="per_example", units=units,
            count=1, diagnostic_floor=float(floor), controller_denominator=None,
            controller_floor_active=None,
        )
        rows.append(row)
    global_row = gradient_metrics(target, estimate, floor)
    global_row.update(
        sample_id=None, aggregation="batch_global", units=units,
        count=int(target.shape[0]), diagnostic_floor=float(floor),
        controller_denominator=None, controller_floor_active=None,
    )
    rows.append(global_row)
    return rows


def _parameter_gradient_metrics(
    exact_gradients: list[Tensor | None],
    injected_gradients: list[Tensor | None],
    *,
    exact_coefficient: float,
    injected_coefficient: float,
) -> list[dict[str, Any]]:
    """Compare aligned parameter gradients before and after actual clipping.

    Inputs are detached tensor lists and update-path clip coefficients. Returned
    dot products and norms are accumulated tensorwise without concatenation.
    """

    if len(exact_gradients) != len(injected_gradients):
        raise ValueError("parameter gradient lists must be aligned")

    def summarize(exact_scale: float, injected_scale: float, stage: str) -> dict[str, Any]:
        """Summarize one clipping stage from the enclosing aligned gradients.

        Inputs are exact/injected scalar multipliers and a stage label. The
        returned parameter-unit metrics are detached; source tensors are not
        concatenated or changed, and absent gradients must remain aligned.
        """

        dot = exact_sq = injected_sq = residual_sq = 0.0
        for exact, injected in zip(exact_gradients, injected_gradients, strict=True):
            if exact is None and injected is None:
                continue
            if exact is None or injected is None:
                raise ValueError("exact and injected paths allocated different gradients")
            exact64 = exact.detach().to(dtype=torch.float64) * exact_scale
            injected64 = injected.detach().to(dtype=torch.float64) * injected_scale
            dot += float((exact64 * injected64).sum().item())
            exact_sq += float(exact64.square().sum().item())
            injected_sq += float(injected64.square().sum().item())
            residual_sq += float((injected64 - exact64).square().sum().item())
        exact_norm = math.sqrt(exact_sq)
        injected_norm = math.sqrt(injected_sq)
        residual_norm = math.sqrt(residual_sq)
        floor = 1e-12
        return {
            "parameter_stage": stage,
            "parameter_cosine": dot / (exact_norm * injected_norm)
            if exact_norm > floor and injected_norm > floor else None,
            "parameter_exact_norm": exact_norm,
            "parameter_injected_norm": injected_norm,
            "parameter_residual_norm": residual_norm,
            "parameter_relative_residual": residual_norm / max(exact_norm, floor),
            "parameter_norm_ratio": injected_norm / max(exact_norm, floor),
            "parameter_projection": dot / max(exact_sq, floor * floor),
        }

    return [
        summarize(1.0, 1.0, "preclip"),
        summarize(exact_coefficient, injected_coefficient, "postclip"),
    ]


def probe_boundary_gradients(
    model: nn.Module,
    predictor: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    scale: float,
    config: RunConfig,
    *,
    sample_ids: list[int] | None = None,
) -> dict[str, Any]:
    """Probe boundary and trunk-parameter gradients on disposable copies.

    Inputs are production modules, a fixed image batch, labels, positive scale,
    and config. Output distinguishes raw mean-loss gradients from normalized
    targets, per-example from global metrics, controller-floor status, predictor
    objective terms, and pre/postclip parameter agreement. Production state and
    every RNG are restored exactly.
    """

    from gradient_phase_d_resnet_v4 import (
        _clip_coefficient, _gradient_norm, _scientific, predictor_loss_components,
        set_training_mode,
    )

    if scale <= 0:
        raise ValueError("scale must be positive")
    batch_size = int(labels.shape[0])
    ids = list(range(batch_size)) if sample_ids is None else [int(value) for value in sample_ids]
    if len(ids) != batch_size:
        raise ValueError("sample_ids must contain one identity per example")
    rng = capture_rng_state()
    try:
        exact_model = copy.deepcopy(model)
        injected_model = copy.deepcopy(model)
        exact_predictor = copy.deepcopy(predictor)
        injected_predictor = copy.deepcopy(predictor)
        normalization = config.scientific_dict()["identity"]["normalization"]
        set_training_mode(exact_model, exact_predictor, "postwarm", normalization)
        set_training_mode(injected_model, injected_predictor, "postwarm", normalization)

        exact_model.zero_grad(set_to_none=True)
        boundary = exact_model.trunk(inputs)
        boundary.retain_grad()
        loss = F.cross_entropy(exact_model.head(boundary), labels)
        loss.backward()
        assert boundary.grad is not None
        raw_target = boundary.grad.detach()
        normalized_target = raw_target * batch_size / scale
        with torch.no_grad():
            normalized_estimate = exact_predictor(boundary.detach(), labels)
        raw_estimate = normalized_estimate * scale / batch_size

        normalized_floor = 1e-8
        raw_floor = scale * normalized_floor / batch_size
        rows = _gradient_rows(
            raw_target, raw_estimate, units="raw_mean_loss_boundary_gradient",
            floor=raw_floor, sample_ids=ids,
        )
        rows.extend(_gradient_rows(
            normalized_target, normalized_estimate, units="normalized_target",
            floor=normalized_floor, sample_ids=ids,
        ))
        normalized_global = next(
            row for row in rows
            if row["units"] == "normalized_target" and row["aggregation"] == "batch_global"
        )
        target_rms = float(normalized_target.square().mean().sqrt().item())
        normalized_global["controller_denominator"] = max(target_rms, 0.05)
        normalized_global["controller_floor_active"] = target_rms < 0.05

        contract = _scientific(config)["gradient_contract"]
        cosine_weight = float(contract["cosine_weight"])
        exact_predictor.zero_grad(set_to_none=True)
        total, mse, cosine = predictor_loss_components(
            exact_predictor(boundary.detach(), labels), normalized_target,
            cosine_weight=cosine_weight,
            valid_norm_floor=float(contract["valid_target_flat_norm_floor"]),
            cosine_eps=float(contract["cosine_eps"]),
        )
        total.backward()
        clip_limit = float(contract["global_l2_clip"])
        predictor_clip = _clip_coefficient(
            _gradient_norm(list(exact_predictor.parameters())), clip_limit
        )
        loss_components = {
            "predictor_loss": float(total.detach().item()),
            "predictor_mse": float(mse.detach().item()),
            "predictor_cosine": float(cosine.detach().item()),
            "predictor_mse_contribution": float(mse.detach().item()),
            "predictor_cosine_contribution": cosine_weight * float(cosine.detach().item()),
            "predictor_clip_coefficient": predictor_clip,
        }

        exact_trunk_gradients = [
            parameter.grad.detach().clone() if parameter.grad is not None else None
            for parameter in exact_model.trunk.parameters()
        ]
        exact_clip = _clip_coefficient(_gradient_norm(list(exact_model.parameters())), clip_limit)
        injected_model.zero_grad(set_to_none=True)
        injected_boundary = injected_model.trunk(inputs)
        with torch.no_grad():
            injected = injected_predictor(injected_boundary.detach(), labels) * scale / batch_size
        injected_boundary.backward(injected)
        injected_trunk_gradients = [
            parameter.grad.detach().clone() if parameter.grad is not None else None
            for parameter in injected_model.trunk.parameters()
        ]
        injected_clip = _clip_coefficient(
            _gradient_norm(list(injected_model.trunk.parameters())), clip_limit
        )
        parameter_rows = _parameter_gradient_metrics(
            exact_trunk_gradients, injected_trunk_gradients,
            exact_coefficient=exact_clip, injected_coefficient=injected_clip,
        )
        for row in parameter_rows:
            row["exact_model_clip_coefficient"] = exact_clip
            row["injected_trunk_clip_coefficient"] = injected_clip
        return {
            "gradient_rows": rows,
            "parameter_rows": parameter_rows,
            "loss_components": loss_components,
        }
    finally:
        restore_rng_state(rng)


def _eval_loss(model: nn.Module, inputs: Tensor, labels: Tensor) -> float:
    """Return eval-mode mean CE while restoring every original module mode.

    Inputs are a model and fixed batch. The detached scalar is returned without
    changing parameters, buffers, gradients, modes, or RNG.
    """

    flags = {module: module.training for module in model.modules()}
    model.eval()
    with torch.no_grad():
        value = float(F.cross_entropy(model(inputs), labels).item())
    for module, flag in flags.items():
        module.training = flag
    return value


def probe_parameter_updates(
    model: nn.Module,
    predictor: nn.Module,
    optimizers: Mapping[str, torch.optim.Optimizer | None],
    inputs: Tensor,
    labels: Tensor,
    fixed_inputs: Tensor,
    fixed_labels: Tensor,
    scale: float,
    config: RunConfig,
) -> list[dict[str, Any]]:
    """Compare hypothetical exact, learned, zero, and skipped trunk updates.

    Each alternative deep-copies the full model, predictor, and optimizer state,
    retaining momentum, weight decay, LR, and AdamW state. Head updates are off
    for every alternative. Returned eval losses cover the same and a separate
    fixed batch; production state, buffers, modes, and RNG remain unchanged.
    """

    from gradient_phase_d_resnet_v4 import exact_step, set_training_mode, synthetic_step

    rng = capture_rng_state()
    rows: list[dict[str, Any]] = []
    try:
        for alternative in ("exact", "learned", "zero", "skip"):
            local_model, local_predictor, local_optimizers = copy.deepcopy(
                (model, predictor, dict(optimizers))
            )
            normalization = config.scientific_dict()["identity"]["normalization"]
            set_training_mode(local_model, local_predictor, "postwarm", normalization)
            same_before = _eval_loss(local_model, inputs, labels)
            fixed_before = _eval_loss(local_model, fixed_inputs, fixed_labels)
            if alternative == "exact":
                result = exact_step(
                    local_model, local_predictor, local_optimizers, inputs, labels,
                    scale, config, fit_predictor=False, head_update=False,
                )
            else:
                result = synthetic_step(
                    local_model, local_predictor, local_optimizers, inputs, labels,
                    scale, config, kind=alternative,
                )
            same_after = _eval_loss(local_model, inputs, labels)
            fixed_after = _eval_loss(local_model, fixed_inputs, fixed_labels)
            trunk_optimizer = local_optimizers["trunk"]
            if trunk_optimizer is None:
                raise ValueError("update probes require a trunk optimizer")
            trunk_group = trunk_optimizer.param_groups[0]
            clip = result.metrics.get(
                "model_clip_coefficient", result.metrics.get("trunk_clip_coefficient")
            )
            rows.append(
                {
                    "alternative": alternative,
                    "head_update": False,
                    "normalization": config.scientific_dict()["identity"]["normalization"],
                    "same_loss_before": same_before,
                    "same_loss_after": same_after,
                    "same_loss_change": same_after - same_before,
                    "fixed_loss_before": fixed_before,
                    "fixed_loss_after": fixed_after,
                    "fixed_loss_change": fixed_after - fixed_before,
                    "trunk_lr": float(trunk_group["lr"]),
                    "momentum": float(trunk_group.get("momentum", 0.0)),
                    "weight_decay": float(trunk_group.get("weight_decay", 0.0)),
                    "model_clip_coefficient": clip,
                    "status": "ok",
                }
            )
        return rows
    finally:
        restore_rng_state(rng)


def _bn_buffers(module: nn.Module) -> dict[str, Tensor]:
    """Clone BatchNorm running buffers for later mutation checks.

    The output maps qualified names to CPU copies and changes no module state.
    """

    return {
        name: value.detach().cpu().clone()
        for name, value in module.named_buffers()
        if "running_mean" in name or "running_var" in name or "num_batches_tracked" in name
    }


def _bn_change(before: Mapping[str, Tensor], after: Mapping[str, Tensor]) -> tuple[bool, float]:
    """Return whether BN buffers changed and their combined native-unit L2 change.

    Inputs must have identical keys. The function is read-only and rejects a
    structural mismatch instead of silently omitting a buffer.
    """

    if before.keys() != after.keys():
        raise ValueError("BatchNorm buffer snapshots have different keys")
    total = 0.0
    changed = False
    for name in before:
        difference = after[name].to(torch.float64) - before[name].to(torch.float64)
        total += float(difference.square().sum().item())
        changed = changed or not torch.equal(before[name], after[name])
    return changed, math.sqrt(total)


def probe_normalization_transition(
    model: nn.Module,
    predictor: nn.Module,
    inputs: Tensor,
    labels: Tensor,
    scale: float,
    config: RunConfig,
    *,
    sample_source: str,
) -> list[dict[str, Any]]:
    """Compare conventional train-mode and frozen-head exact targets on copies.

    Both references receive identical detached boundary features from a fixed
    batch labeled deterministic or augmented. Results contain normalized-target
    change, predictor error/clipping, and BN mutations. Production state and RNG
    remain unchanged; conventional exact is labeled separately from v3 exact.
    """

    from gradient_phase_d_resnet_v4 import (
        _clip_coefficient, _gradient_norm, _scientific, predictor_loss_components,
    )

    if sample_source not in {"deterministic_training", "augmented_training"}:
        raise ValueError("sample_source must disclose deterministic or augmented training data")
    if scale <= 0:
        raise ValueError("scale must be positive")
    rng = capture_rng_state()
    try:
        trunk_copy = copy.deepcopy(model.trunk)
        # The transition question concerns the boundary that the post-warm-up
        # training path actually supplies.  Trunk BatchNorm therefore remains
        # in training mode; only this disposable copy may update its buffers.
        trunk_copy.train()
        with torch.no_grad():
            shared_boundary = trunk_copy(inputs).detach()
            predicted = copy.deepcopy(predictor)(shared_boundary, labels).detach()
        contract = _scientific(config)["gradient_contract"]
        clip_limit = float(contract["global_l2_clip"])
        targets: dict[str, Tensor] = {}
        rows: list[dict[str, Any]] = []
        for reference, train_head in (
            ("conventional_exact_standard_bn", True),
            ("v3_exact_frozen_head", False),
        ):
            head = copy.deepcopy(model.head)
            head.train(train_head)
            before = _bn_buffers(head)
            boundary = shared_boundary.detach().clone().requires_grad_(True)
            loss = F.cross_entropy(head(boundary), labels)
            raw_target = torch.autograd.grad(loss, boundary)[0]
            target = raw_target * labels.shape[0] / scale
            after = _bn_buffers(head)
            changed, change_l2 = _bn_change(before, after)
            targets[reference] = target.detach()
            metrics = gradient_metrics(target, predicted, 1e-8)
            predictor_copy = copy.deepcopy(predictor)
            predictor_copy.zero_grad(set_to_none=True)
            total, _, _ = predictor_loss_components(
                predictor_copy(shared_boundary, labels), target.detach(),
                cosine_weight=float(contract["cosine_weight"]),
                valid_norm_floor=float(contract["valid_target_flat_norm_floor"]),
                cosine_eps=float(contract["cosine_eps"]),
            )
            total.backward()
            predictor_clip = _clip_coefficient(
                _gradient_norm(list(predictor_copy.parameters())), clip_limit
            )
            rows.append(
                {
                    "sample_source": sample_source,
                    "reference": reference,
                    "head_training": train_head,
                    "target_units": "normalized_target",
                    "target_rms": float(target.square().mean().sqrt().item()),
                    "predictor_residual_rms": metrics["residual_rms"],
                    "predictor_relative_residual": metrics["relative_residual"],
                    "predictor_clip_coefficient": predictor_clip,
                    "bn_buffers_changed": changed,
                    "bn_change_l2": change_l2,
                    "target_change_norm": None,
                    "target_change_relative": None,
                    "status": "ok",
                }
            )
        frozen = targets["v3_exact_frozen_head"]
        conventional = targets["conventional_exact_standard_bn"]
        change = float((conventional - frozen).norm().item())
        relative = change / max(float(frozen.norm().item()), 1e-8)
        for row in rows:
            row["target_change_norm"] = change
            row["target_change_relative"] = relative
        return rows
    finally:
        restore_rng_state(rng)


def _profile_elapsed_ms(function: Any, device: torch.device) -> float:
    """Measure one callable with CUDA events or synchronized CPU wall time.

    The callable may mutate only disposable captured state. CUDA synchronizes at
    the microbenchmark boundary. The output is milliseconds; errors propagate.
    """

    if device.type == "cuda":
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(device)
        start.record()
        function()
        end.record()
        torch.cuda.synchronize(device)
        return float(start.elapsed_time(end))
    start_time = time.perf_counter()
    function()
    return (time.perf_counter() - start_time) * 1000.0


def _profile_cpu_elapsed_ms(function: Any) -> float:
    """Measure one host-only callable with the CPU monotonic wall clock.

    The input callable performs controller or other CPU work and may mutate only
    disposable captured state. The returned duration is milliseconds. CUDA
    events are deliberately not used because they cannot measure host execution;
    no device synchronization or production state mutation occurs here.
    """

    started = time.perf_counter()
    function()
    return (time.perf_counter() - started) * 1000.0


def profile_steps(
    model: nn.Module,
    predictor: nn.Module,
    optimizers: Mapping[str, torch.optim.Optimizer | None],
    controller: Any,
    inputs: Tensor,
    labels: Tensor,
    scale: float,
    config: RunConfig,
    *,
    samples: int = 12,
    settling: int = 2,
) -> dict[str, Any]:
    """Profile copied kernels and controller overhead as individual samples.

    Clone construction is outside timed intervals. Defaults use two unreported
    settling iterations and twelve reported iterations per component. Summaries
    are recomputed from every reported sample. Production models, optimizer and
    controller states, buffers, modes, and RNG remain unchanged.
    """

    from gradient_phase_d_resnet_v4 import exact_step, set_training_mode, synthetic_step

    if samples <= 0 or settling < 0:
        raise ValueError("samples must be positive and settling nonnegative")
    device = inputs.device
    rng = capture_rng_state()
    observations: list[dict[str, Any]] = []
    components = (
        "plain_exact",
        "exact_plus_predictor",
        "learned_synthetic",
        "zero_synthetic_shadow",
        "skip_synthetic_shadow",
        "controller_overhead",
    )
    try:
        for component in components:
            for iteration in range(settling + samples):
                if component == "controller_overhead":
                    local_controller = copy.deepcopy(controller)

                    def operation() -> None:
                        """Execute one actual controller decision on copied state.

                        There are no inputs or return value. The captured copy's
                        history and counters change; production controller state
                        and RNG remain untouched.
                        """

                        local_controller.choose(0.1, 0.2)
                else:
                    local_model, local_predictor, local_optimizers = copy.deepcopy(
                        (model, predictor, dict(optimizers))
                    )
                    normalization = config.scientific_dict()["identity"]["normalization"]
                    set_training_mode(local_model, local_predictor, "postwarm", normalization)
                    if component in {"plain_exact", "exact_plus_predictor"}:

                        def operation() -> None:
                            """Execute one exact kernel on copied classifier state.

                            There are no inputs or return value. Captured copied
                            parameters, gradients, optimizer state, and BN buffers
                            change. Predictor fitting is enabled only for the
                            exact-plus-predictor profile component.
                            """

                            exact_step(
                                local_model, local_predictor, local_optimizers,
                                inputs, labels, scale, config,
                                fit_predictor=component == "exact_plus_predictor",
                            )
                    else:

                        def operation() -> None:
                            """Execute one selected synthetic kernel on copied state.

                            There are no inputs or return value. Captured copied
                            trunk parameters, optimizer state, gradients, and BN
                            buffers change as prescribed; production state remains
                            isolated. Shadow controls retain predictor inference.
                            """

                            behavior = {
                                "learned_synthetic": "learned",
                                "zero_synthetic_shadow": "zero",
                                "skip_synthetic_shadow": "skip",
                            }[component]
                            synthetic_step(
                                local_model, local_predictor, local_optimizers,
                                inputs, labels, scale, config, kind=behavior,
                                shadow_predictor=component.endswith("_shadow"),
                            )
                # Controller selection is ordinary Python/NumPy host work. CUDA
                # events would measure an unrelated empty device interval.
                latency = (
                    _profile_cpu_elapsed_ms(operation)
                    if component == "controller_overhead"
                    else _profile_elapsed_ms(operation, device)
                )
                if iteration >= settling:
                    observations.append(
                        {
                            "component": component,
                            "sample_index": iteration - settling,
                            "latency_ms": latency,
                            "device_type": device.type,
                            "settling_iterations": settling,
                            "status": "engineering_only",
                        }
                    )
        summary = {
            "engineering_only": True,
            "device_type": device.type,
            "settling_iterations": settling,
            "components": {
                component: finite_summary(
                    [row["latency_ms"] for row in observations if row["component"] == component]
                )
                for component in components
            },
        }
        return {"samples": observations, "summary": summary}
    finally:
        restore_rng_state(rng)


__all__ = [
    "gradient_metrics", "probe_boundary_gradients", "probe_normalization_transition",
    "probe_parameter_updates", "profile_steps",
]
