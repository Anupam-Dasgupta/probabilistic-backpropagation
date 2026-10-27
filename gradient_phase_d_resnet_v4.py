"""Phase D v4 mechanics and authorized pre-experiment execution through M6.

This module provides the frozen architecture and update rules from M1 plus the
deterministic runner, persistence, diagnostics, replay, M5 schedules, explicit
CIFAR-10 data/run entry points, and bounded M6 preflight integration. It never
launches work merely because it is imported or expands one command into a study.
"""

from __future__ import annotations

import collections
import argparse
import copy
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset

from phase_d_v4_io import (
    SCHEMA_VERSION,
    METHOD_SPECS,
    MaskMetadata,
    MaskRole,
    MethodId,
    NormalizationMode,
    PREDICTOR_REQUIRED_METHODS,
    REPLAY_METHODS,
    SOURCE_DERIVED_SCHEDULE_METHODS,
    FIXED_SCHEDULE_METHODS,
    MASK_EXECUTION_METHODS,
    RunConfig,
    RunStage,
    RunState,
    StepResult,
    SyntheticBehavior,
    TimingLedger,
    UpdateType,
    atomic_write_json,
    capture_rng_state,
    copy_checkpoint_verified,
    environment_metadata,
    execution_fingerprint,
    fingerprint,
    latest_committed_checkpoint,
    load_checkpoint,
    load_mask,
    mask_identity_hash,
    load_run_config,
    reconcile_outputs,
    restore_rng_state,
    run_state_from_dict,
    run_state_to_dict,
    save_checkpoint,
    sha256_file,
    validate_output_root,
    validate_config,
    backup_resume_snapshot,
    restore_resume_snapshot,
)
from phase_d_v4_diagnostics import (
    gradient_metrics,
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
    diagnostic_identity_fields,
    fixed_probe_indices,
    write_diagnostic_outputs,
    write_profile_outputs,
)


PROJECT_ROOT = Path(__file__).resolve().parent
REFERENCE_CONFIG_PATH = PROJECT_ROOT / "configs" / "phase_d_v4_reference.json"


def reference_config() -> RunConfig:
    """Load and validate the frozen M0 reference config without starting a run."""

    return load_run_config(REFERENCE_CONFIG_PATH)


def build_condition_config(
    method: MethodId | str,
    *,
    run_id: str,
    output_root: str,
    classifier_seed: int = 0,
    synthetic_behavior: SyntheticBehavior | str | None = None,
    mask_replicate: int | None = None,
    donor_mask_path: str | Path | None = None,
    donor_manifest_path: str | Path | None = None,
) -> RunConfig:
    """Resolve one frozen reference condition, optionally from a donor mask.

    Inputs select a registered method, run/output identity, classifier seed,
    optional fixed-schedule behavior/replica, and paired donor artifacts when the
    method is source-derived. The returned config is complete and validated.
    Donor bytes are read and hash-checked but never changed; no output is written,
    RNG consumed, data loaded, or training started. Missing/incompatible donor
    identities fail before a config is returned.
    """

    selected = MethodId(method)
    spec = METHOD_SPECS[selected]
    values = reference_config().to_dict()
    if not run_id or not output_root or classifier_seed < 0:
        raise ValueError("run_id/output_root must be nonempty and seed nonnegative")
    if synthetic_behavior is None:
        behavior = (
            SyntheticBehavior.EXACT
            if SyntheticBehavior.EXACT in spec["behaviors"]
            else SyntheticBehavior.LEARNED
        )
    else:
        behavior = SyntheticBehavior(synthetic_behavior)
    identity = values["scientific"]["identity"]
    identity.update(
        classifier_seed=classifier_seed,
        mask_replicate=mask_replicate,
        method=selected.value,
        algorithm_version=spec["version"],
        normalization=spec["normalization"].value,
        synthetic_behavior=behavior.value,
        mask_role=spec["mask_role"].value,
        mask_sha256=None,
        retrospective=spec["retrospective"],
        deployment_eligible=spec["deployment_eligible"],
    )
    values["output"].update(root=output_root, study_id="phase_d_v4_pilot_seed0", run_id=run_id)
    source_derived = selected in REPLAY_METHODS | SOURCE_DERIVED_SCHEDULE_METHODS
    if source_derived:
        if donor_mask_path is None or donor_manifest_path is None:
            raise ValueError("source-derived condition requires donor mask and manifest")
        manifest = json.loads(Path(donor_manifest_path).read_text(encoding="utf-8"))
        export = manifest.get("replay_export")
        if not isinstance(export, Mapping):
            raise ValueError("donor manifest lacks replay export")
        if int(export.get("source_seed", -1)) != classifier_seed:
            raise ValueError("donor classifier seed differs from requested condition")
        if export.get("source_method") != MethodId.RELATIVE_FEEDBACK.value:
            raise ValueError("pilot donor must be relative_feedback")
        donor_file = Path(donor_mask_path)
        if sha256_file(donor_file) != manifest.get("artifact_sha256"):
            raise ValueError("donor mask artifact checksum mismatch")
        with np.load(donor_file, allow_pickle=False) as arrays:
            donor_mask = [bool(item) for item in arrays[str(export["array"])].tolist()]
        offsets = tuple(int(item) for item in export["epoch_offsets"])
        donor_hash = mask_identity_hash(donor_mask, offsets)
        if donor_hash != export.get("sha256"):
            raise ValueError("donor mask content identity mismatch")
        source = {
            "run_id": export["source_run_id"],
            "checkpoint_sha256": export["source_checkpoint_sha256"],
            "seed": int(export["source_seed"]),
            "method": export["source_method"],
            "scientific_fingerprint": export["source_scientific_fingerprint"],
        }
        if selected in REPLAY_METHODS:
            identity["mask_sha256"] = donor_hash
            values["scientific"]["replay_source"] = source
        else:
            source["mask_sha256"] = donor_hash
            values["scientific"]["schedule_source"] = source
            epoch_lengths = [end - start for start, end in zip(offsets[:-1], offsets[1:], strict=True)]
            if selected == MethodId.RELATIVE_PERIODIC_TOTAL:
                constructed = make_periodic_mask(len(donor_mask), sum(donor_mask))
            elif selected == MethodId.RELATIVE_PERIODIC_EPOCH:
                constructed = make_epoch_matched_mask(donor_mask, offsets)
            else:
                if mask_replicate is None:
                    raise ValueError("gap permutation requires an explicit replica")
                constructed, _details = permute_epoch_gaps(
                    donor_mask, offsets,
                    classifier_seed=classifier_seed,
                    replicate=mask_replicate,
                )
            if len(constructed) != sum(epoch_lengths):
                raise ValueError("constructed schedule length differs from donor horizon")
            identity["mask_sha256"] = mask_identity_hash(constructed, offsets)
    elif donor_mask_path is not None or donor_manifest_path is not None:
        raise ValueError("donor-independent condition must not receive donor artifacts")
    config = RunConfig(values)
    validate_config(config)
    return config


class BasicBlock(nn.Module):
    """Apply a two-convolution residual block to an image feature tensor."""

    def __init__(self, n_in: int, n_out: int, stride: int = 1) -> None:
        """Create paths for ``[B,n_in,H,W]`` inputs and ``n_out`` outputs."""

        super().__init__()
        self.conv1 = nn.Conv2d(n_in, n_out, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(n_out)
        self.conv2 = nn.Conv2d(n_out, n_out, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(n_out)
        self.skip = (
            nn.Identity()
            if n_in == n_out and stride == 1
            else nn.Sequential(
                nn.Conv2d(n_in, n_out, 1, stride, bias=False),
                nn.BatchNorm2d(n_out),
            )
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return residual features, updating BatchNorm buffers only in train mode."""

        residual = self.conv2(F.relu(self.bn1(self.conv1(inputs))))
        return F.relu(self.bn2(residual) + self.skip(inputs))


def make_layer(n_in: int, n_out: int, blocks: int, stride: int) -> nn.Sequential:
    """Build one ResNet stage, consuming RNG only for parameter initialization."""

    if blocks <= 0:
        raise ValueError("blocks must be positive")
    pieces = [BasicBlock(n_in, n_out, stride)]
    pieces.extend(BasicBlock(n_out, n_out) for _ in range(blocks - 1))
    return nn.Sequential(*pieces)


class SplitResNet(nn.Module):
    """Split a CIFAR ResNet into an upstream trunk and classifier head."""

    def __init__(self, width: int = 64, blocks: int = 2, classes: int = 10) -> None:
        """Create the v3-compatible split model with newly initialized parameters."""

        super().__init__()
        self.width = width
        self.trunk = nn.Sequential(
            nn.Conv2d(3, width, 3, 1, 1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            make_layer(width, width, blocks, 1),
            make_layer(width, width * 2, blocks, 2),
        )
        self.head = nn.Sequential(
            make_layer(width * 2, width * 4, blocks, 2),
            make_layer(width * 4, width * 8, blocks, 2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(width * 8, classes),
        )

    def forward(self, inputs: Tensor) -> Tensor:
        """Return ``[B,classes]`` logits for ``[B,3,32,32]`` inputs."""

        return self.head(self.trunk(inputs))


class GradientPredictor(nn.Module):
    """Predict normalized boundary gradients from detached features and labels."""

    def __init__(self, channels: int, width: int = 32, classes: int = 10) -> None:
        """Create the v3 label embedding and spatial predictor layers."""

        super().__init__()
        self.label = nn.Embedding(classes, channels)
        self.net = nn.Sequential(
            nn.Conv2d(channels, width, 1),
            nn.ReLU(),
            nn.Conv2d(width, width, 3, padding=1, groups=width),
            nn.ReLU(),
            nn.Conv2d(width, channels, 1),
        )

    def forward(self, boundary: Tensor, labels: Tensor) -> Tensor:
        """Return ``[B,C,H,W]`` predictions without classifier graph attachment."""

        conditioned = boundary.detach() + self.label(labels)[:, :, None, None]
        return self.net(conditioned)


def _scientific(config: RunConfig) -> Mapping[str, Any]:
    """Return the resolved scientific section for internal read-only access."""

    return config.values["scientific"]


def build_models(
    config: RunConfig,
    device: torch.device | str,
    include_predictor: bool = True,
) -> tuple[SplitResNet, GradientPredictor | None]:
    """Build a fresh classifier and optional predictor on the requested device.

    Configuration supplies widths and class counts. Construction consumes the
    PyTorch initialization RNG but loads no data and performs no optimization.
    """

    validate_config(config)
    scientific = _scientific(config)
    model_values = scientific["model"]
    predictor_values = scientific["predictor"]
    model = SplitResNet(
        int(model_values["width"]),
        int(model_values["blocks_per_stage"]),
        int(model_values["classes"]),
    ).to(device)
    predictor = None
    if include_predictor:
        predictor = GradientPredictor(
            int(predictor_values["input_channels"]),
            int(predictor_values["bottleneck_width"]),
            int(predictor_values["label_embedding_classes"]),
        ).to(device)
    return model, predictor


def build_optimizers(
    model: SplitResNet,
    predictor: GradientPredictor | None,
    config: RunConfig,
) -> dict[str, torch.optim.Optimizer | None]:
    """Create separate model SGD optimizers and optional predictor AdamW.

    Optimizer states begin empty and reference supplied parameters. Future steps
    mutate parameters and buffers; construction itself performs no update.
    """

    values = _scientific(config)["optimizers"]
    model_optimizers: dict[str, torch.optim.Optimizer | None] = {}
    for name, parameters in (("trunk", model.trunk.parameters()), ("head", model.head.parameters())):
        settings = values[name]
        model_optimizers[name] = torch.optim.SGD(
            parameters,
            lr=float(settings["lr_start"]),
            momentum=float(settings["momentum"]),
            weight_decay=float(settings["weight_decay"]),
            dampening=float(settings["dampening"]),
            nesterov=bool(settings["nesterov"]),
        )
    model_optimizers["predictor"] = None
    if predictor is not None:
        settings = values["predictor"]
        model_optimizers["predictor"] = torch.optim.AdamW(
            predictor.parameters(),
            lr=float(settings["lr"]),
            weight_decay=float(settings["weight_decay"]),
            betas=tuple(float(value) for value in settings["betas"]),
            eps=float(settings["eps"]),
        )
    return model_optimizers


def set_epoch_lr(
    optimizers: Mapping[str, torch.optim.Optimizer | None],
    config: RunConfig,
    global_epoch: int,
) -> float:
    """Set model LRs from the v3 global epoch clock and return the value."""

    if global_epoch < 0:
        raise ValueError("global_epoch must be nonnegative")
    scientific = _scientific(config)
    training = scientific["training"]
    schedule = scientific["learning_rate"]
    horizon = int(training["warmup_epochs"]) + int(training["postwarm_epochs"])
    phase = min(global_epoch, max(0, horizon - 1)) / max(1, horizon - 1)
    value = float(schedule["lr_min"]) + 0.5 * (
        float(schedule["lr_start"]) - float(schedule["lr_min"])
    ) * (1.0 + math.cos(math.pi * phase))
    for name in ("trunk", "head"):
        optimizer = optimizers[name]
        if optimizer is None:
            raise ValueError(f"missing {name} optimizer")
        for group in optimizer.param_groups:
            group["lr"] = value
    return value


def set_training_mode(
    model: SplitResNet,
    predictor: GradientPredictor | None,
    phase: str,
    normalization: NormalizationMode | str,
) -> None:
    """Set warm-up, post-warm-up, calibration, or evaluation module modes.

    This changes training flags only, never affine trainability or buffers.
    Unsupported phases and undefined standard-BN calibration raise errors.
    """

    mode = NormalizationMode(normalization)
    if phase == "warmup":
        model.train()
        if predictor is not None:
            predictor.train()
    elif phase == "postwarm":
        if mode == NormalizationMode.FROZEN_HEAD_POSTWARM:
            model.trunk.train()
            model.head.eval()
        else:
            model.train()
        if predictor is not None:
            predictor.train()
    elif phase == "calibration":
        if mode != NormalizationMode.FROZEN_HEAD_POSTWARM:
            raise ValueError("calibration is defined only for frozen_head_postwarm")
        model.trunk.train()
        model.head.eval()
        if predictor is not None:
            predictor.eval()
    elif phase == "evaluation":
        model.eval()
        if predictor is not None:
            predictor.eval()
    else:
        raise ValueError(f"unsupported training phase: {phase}")


def estimate_initial_scale(model: SplitResNet, inputs: Tensor, labels: Tensor) -> float:
    """Return eval-mode summed-CE boundary-gradient RMS with a ``1e-6`` floor.

    Inputs have shapes ``[B,3,32,32]`` and ``[B]``. Original module modes are
    restored, and parameters, gradients, and BN buffers remain unchanged.
    """

    training_flags = {module: module.training for module in model.modules()}
    model.eval()
    with torch.no_grad():
        boundary = model.trunk(inputs)
    leaf = boundary.detach().requires_grad_(True)
    summed_loss = F.cross_entropy(model.head(leaf), labels, reduction="sum")
    target = torch.autograd.grad(summed_loss, leaf)[0]
    scale = max(float(target.square().mean().sqrt().item()), 1e-6)
    for module, was_training in training_flags.items():
        module.training = was_training
    return scale


def predictor_loss_components(
    estimate: Tensor,
    target: Tensor,
    cosine_weight: float = 0.2,
    valid_norm_floor: float = 1e-5,
    cosine_eps: float = 1e-8,
) -> tuple[Tensor, Tensor, Tensor]:
    """Return total, coordinate-MSE, and valid-example cosine losses.

    Estimate and target share ``[B,C,H,W]`` normalized-target units. This pure
    calculation changes no model or optimizer state.
    """

    mse = (estimate - target).square().mean()
    estimate_flat = estimate.flatten(1)
    target_flat = target.flatten(1)
    valid = target_flat.norm(dim=1) > valid_norm_floor
    if bool(valid.any()):
        cosine = (
            1.0
            - F.cosine_similarity(
                estimate_flat[valid], target_flat[valid], dim=1, eps=cosine_eps
            )
        ).mean()
    else:
        cosine = mse * 0.0
    return mse + cosine_weight * cosine, mse, cosine


def relative_error(estimate: Tensor, target: Tensor, denominator_floor: float = 0.05) -> float:
    """Return v3 batch-global normalized RMSE in normalized target units."""

    denominator = target.square().mean().sqrt().clamp_min(denominator_floor)
    return float(((estimate - target).square().mean().sqrt() / denominator).item())


def _gradient_norm(parameters: list[nn.Parameter]) -> Tensor:
    """Return the global L2 norm of allocated gradients without concatenation."""

    allocated = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not allocated:
        device = parameters[0].device if parameters else torch.device("cpu")
        return torch.zeros((), device=device)
    return torch.stack([gradient.detach().norm(2) for gradient in allocated]).norm(2)


def _clip_coefficient(preclip_norm: Tensor, limit: float) -> float:
    """Return PyTorch's effective global clip coefficient as a scalar."""

    return min(1.0, float(limit / (float(preclip_norm.item()) + 1e-6)))


def exact_step(
    model: SplitResNet,
    predictor: GradientPredictor | None,
    optimizers: Mapping[str, torch.optim.Optimizer | None],
    inputs: Tensor,
    labels: Tensor,
    scale: float,
    config: RunConfig,
    *,
    fit_predictor: bool,
    head_update: bool = True,
) -> StepResult:
    """Run one exact classifier backward and prescribed optimizer updates.

    Mean-CE targets use ``B*dL/dh/scale``. Predictor fitting uses a detached
    target and the historical second predictor evaluation. The function mutates
    module parameters, gradients, optimizer state, and train-mode BN buffers.
    """

    if scale <= 0:
        raise ValueError("scale must be positive")
    trunk_optimizer = optimizers["trunk"]
    head_optimizer = optimizers["head"]
    predictor_optimizer = optimizers["predictor"]
    if trunk_optimizer is None or head_optimizer is None:
        raise ValueError("exact_step requires trunk and head optimizers")
    if fit_predictor and (predictor is None or predictor_optimizer is None):
        raise ValueError("predictor fitting requires predictor and optimizer")

    trunk_optimizer.zero_grad(set_to_none=True)
    head_optimizer.zero_grad(set_to_none=True)
    boundary = model.trunk(inputs)
    if fit_predictor:
        boundary.retain_grad()
    loss = F.cross_entropy(model.head(boundary), labels)
    loss.backward()
    metrics: dict[str, float | int | bool | None] = {"classifier_loss": float(loss.item())}
    predictor_fits = 0
    predictor_inferences = 0
    gradient_values = _scientific(config)["gradient_contract"]
    clip_limit = float(gradient_values["global_l2_clip"])

    if fit_predictor:
        assert predictor is not None and predictor_optimizer is not None and boundary.grad is not None
        target = boundary.grad.detach() * labels.shape[0] / scale
        # Controller errors must precede this exact batch's predictor update.
        with torch.no_grad():
            prefit = predictor(boundary.detach(), labels)
            error_abs = float((prefit - target).square().mean().sqrt().item())
            error_rel = relative_error(prefit, target)
        predictor_optimizer.zero_grad(set_to_none=True)
        total, mse, cosine = predictor_loss_components(
            predictor(boundary.detach(), labels),
            target,
            cosine_weight=float(gradient_values["cosine_weight"]),
            valid_norm_floor=float(gradient_values["valid_target_flat_norm_floor"]),
            cosine_eps=float(gradient_values["cosine_eps"]),
        )
        total.backward()
        predictor_parameters = list(predictor.parameters())
        predictor_preclip = _gradient_norm(predictor_parameters)
        nn.utils.clip_grad_norm_(predictor_parameters, clip_limit)
        predictor_optimizer.step()
        predictor_fits = 1
        predictor_inferences = 2
        metrics.update(
            error_abs=error_abs,
            error_relative=error_rel,
            target_rms=float(target.square().mean().sqrt().item()),
            predictor_loss=float(total.detach().item()),
            predictor_mse=float(mse.detach().item()),
            predictor_cosine=float(cosine.detach().item()),
            predictor_preclip_norm=float(predictor_preclip.item()),
            predictor_clip_coefficient=_clip_coefficient(predictor_preclip, clip_limit),
        )

    model_parameters = list(model.parameters())
    model_preclip = _gradient_norm(model_parameters)
    # Preserve v3 joint clipping even when the head optimizer will not step.
    nn.utils.clip_grad_norm_(model_parameters, clip_limit)
    trunk_optimizer.step()
    if head_update:
        head_optimizer.step()
    else:
        head_optimizer.zero_grad(set_to_none=True)
    metrics.update(
        model_preclip_norm=float(model_preclip.item()),
        model_clip_coefficient=_clip_coefficient(model_preclip, clip_limit),
    )
    return StepResult(
        update_type=UpdateType.EXACT,
        exact_trunk_backwards=1,
        trunk_optimizer_steps=1,
        head_optimizer_steps=int(head_update),
        predictor_fits=predictor_fits,
        predictor_inferences=predictor_inferences,
        metrics=metrics,
    )


def synthetic_step(
    model: SplitResNet,
    predictor: GradientPredictor | None,
    optimizers: Mapping[str, torch.optim.Optimizer | None],
    inputs: Tensor,
    labels: Tensor,
    scale: float,
    config: RunConfig,
    *,
    kind: SyntheticBehavior | str,
    shadow_predictor: bool = False,
) -> StepResult:
    """Run one learned, zero-gradient, or truly skipped synthetic slot.

    The trunk forward always runs so its BN buffers see the batch; the head never
    runs. Learned/zero paths backward and step trunk SGD, while skip does neither.
    Shadow inference is discarded. Predictor parameters, gradients, and optimizer
    state are never changed by this function.
    """

    behavior = SyntheticBehavior(kind)
    if behavior == SyntheticBehavior.EXACT:
        raise ValueError("synthetic_step does not accept exact behavior")
    if scale <= 0:
        raise ValueError("scale must be positive")
    trunk_optimizer = optimizers["trunk"]
    head_optimizer = optimizers["head"]
    if trunk_optimizer is None or head_optimizer is None:
        raise ValueError("synthetic_step requires trunk and head optimizers")
    if (behavior == SyntheticBehavior.LEARNED or shadow_predictor) and predictor is None:
        raise ValueError("learned or shadow synthetic steps require a predictor")

    trunk_optimizer.zero_grad(set_to_none=True)
    # Remove stale head data gradients without stepping head momentum or decay.
    head_optimizer.zero_grad(set_to_none=True)
    predictor_inferences = 0
    if behavior == SyntheticBehavior.SKIP:
        with torch.no_grad():
            boundary = model.trunk(inputs)
            if shadow_predictor:
                assert predictor is not None
                predictor(boundary, labels)
                predictor_inferences = 1
        return StepResult(
            update_type=UpdateType.SKIP,
            skipped_slots=1,
            predictor_inferences=predictor_inferences,
        )

    boundary = model.trunk(inputs)
    if behavior == SyntheticBehavior.LEARNED:
        assert predictor is not None
        with torch.no_grad():
            injected = predictor(boundary.detach(), labels) * scale / labels.shape[0]
        predictor_inferences = 1
        update_type = UpdateType.LEARNED_SYNTHETIC
        synthetic_backwards, zero_backwards = 1, 0
    else:
        if shadow_predictor:
            assert predictor is not None
            with torch.no_grad():
                predictor(boundary.detach(), labels)
            predictor_inferences = 1
        # True zero gradients allocate buffers, keeping SGD momentum/decay active.
        injected = torch.zeros_like(boundary)
        update_type = UpdateType.ZERO_SYNTHETIC
        synthetic_backwards, zero_backwards = 0, 1
    boundary.backward(injected.detach())
    trunk_parameters = list(model.trunk.parameters())
    trunk_preclip = _gradient_norm(trunk_parameters)
    clip_limit = float(_scientific(config)["gradient_contract"]["global_l2_clip"])
    nn.utils.clip_grad_norm_(trunk_parameters, clip_limit)
    trunk_optimizer.step()
    return StepResult(
        update_type=update_type,
        synthetic_backwards=synthetic_backwards,
        zero_backwards=zero_backwards,
        trunk_optimizer_steps=1,
        predictor_inferences=predictor_inferences,
        metrics={
            "trunk_preclip_norm": float(trunk_preclip.item()),
            "trunk_clip_coefficient": _clip_coefficient(trunk_preclip, clip_limit),
        },
    )


class HistoricalFeedbackController:
    """Reproduce v3 relative or absolute feedback choices without model access."""

    def __init__(self, kind: str, config: RunConfig, calibration: Mapping[str, Any]) -> None:
        """Initialize ordered relative history and fixed absolute thresholds."""

        if kind not in {MethodId.RELATIVE_FEEDBACK.value, MethodId.ABSOLUTE_FEEDBACK.value}:
            raise ValueError("kind must be relative_feedback or absolute_feedback")
        values = _scientific(config)["controller"]
        self.kind = kind
        self.low_quantile = float(values["low_quantile"])
        self.high_quantile = float(values["high_quantile"])
        self.maximum_quota = int(values["maximum_quota"])
        self.ema_old_value_weight = float(values["ema_old_value_weight"])
        self.emergency_multiplier = float(values["emergency_multiplier"])
        self.history = collections.deque(
            calibration["relative_errors"][-int(values["history_window"]):],
            maxlen=int(values["history_window"]),
        )
        if not self.history:
            raise ValueError("relative calibration history must not be empty")
        self.absolute_low = float(calibration["q_abs_low"])
        self.absolute_high = float(calibration["q_abs_high"])
        self.absolute_median = float(calibration["median_abs"])
        self.ema: float | None = None
        self.choices: collections.Counter[int] = collections.Counter()

    def choose(self, error_abs: float, error_relative_value: float) -> tuple[int, dict[str, float | int]]:
        """Choose a future quota from pre-fit error, then mutate controller state."""

        if self.kind == MethodId.ABSOLUTE_FEEDBACK.value:
            self.ema = (
                error_abs
                if self.ema is None
                else self.ema_old_value_weight * self.ema
                + (1.0 - self.ema_old_value_weight) * error_abs
            )
            score = self.ema
            low, high = self.absolute_low, self.absolute_high
            if score > self.absolute_median * self.emergency_multiplier:
                quota = 0
            elif score <= low:
                quota = 4
            elif score <= high:
                quota = 2
            else:
                quota = 0
        else:
            history = np.asarray(self.history, dtype=np.float64)
            low = float(np.quantile(history, self.low_quantile, method="linear"))
            high = float(np.quantile(history, self.high_quantile, method="linear"))
            score = error_relative_value
            if score <= low:
                quota = 4
            elif score <= high:
                quota = 2
            else:
                quota = 0
            # Current exact-batch error must not influence its own thresholds.
            self.history.append(error_relative_value)
        quota = min(quota, self.maximum_quota)
        self.choices[quota] += 1
        return quota, {
            "error_abs": error_abs,
            "error_relative": error_relative_value,
            "controller_score": float(score),
            "threshold_low": low,
            "threshold_high": high,
            "synthetic_quota": quota,
        }

    def state_dict(self) -> dict[str, Any]:
        """Return ordered controller history, thresholds, EMA, and counts as primitives."""

        return {
            "kind": self.kind,
            "low_quantile": self.low_quantile,
            "high_quantile": self.high_quantile,
            "maximum_quota": self.maximum_quota,
            "ema_old_value_weight": self.ema_old_value_weight,
            "emergency_multiplier": self.emergency_multiplier,
            "history": list(self.history),
            "history_maxlen": self.history.maxlen,
            "absolute_low": self.absolute_low,
            "absolute_high": self.absolute_high,
            "absolute_median": self.absolute_median,
            "ema": self.ema,
            "choices": {str(key): int(value) for key, value in self.choices.items()},
        }

    def load_state_dict(self, values: Mapping[str, Any]) -> None:
        """Restore exact controller state, rejecting a mismatched controller kind."""

        if values["kind"] != self.kind:
            raise ValueError("controller kind mismatch")
        self.low_quantile = float(values["low_quantile"])
        self.high_quantile = float(values["high_quantile"])
        self.maximum_quota = int(values["maximum_quota"])
        self.ema_old_value_weight = float(values["ema_old_value_weight"])
        self.emergency_multiplier = float(values["emergency_multiplier"])
        self.history = collections.deque(
            [float(item) for item in values["history"]], maxlen=int(values["history_maxlen"])
        )
        self.absolute_low = float(values["absolute_low"])
        self.absolute_high = float(values["absolute_high"])
        self.absolute_median = float(values["absolute_median"])
        self.ema = None if values.get("ema") is None else float(values["ema"])
        self.choices = collections.Counter(
            {int(key): int(value) for key, value in values.get("choices", {}).items()}
        )


def scheduled_exact(global_postwarm_step: int, postwarm_epoch: int, postwarm_epochs: int) -> bool:
    """Return the unchanged v3 scheduled decision on the global post-warm clock."""

    if global_postwarm_step < 0 or postwarm_epoch < 0 or postwarm_epochs <= 0:
        raise ValueError("scheduled indices must be nonnegative and horizon positive")
    quartile = min(3, int(4 * postwarm_epoch / postwarm_epochs))
    ratio = (4, 3, 2, 1)[quartile]
    return global_postwarm_step % (ratio + 1) == 0


def historical_midpoint_periodic_mask(length: int, exact_count: int) -> list[bool]:
    """Reproduce v3's midpoint periodic construction without changing its math.

    Inputs are a positive mask length and exact count in ``[1,length]``. The
    returned boolean list places points at ``floor((j+0.5)*n/k)``. It consumes no
    RNG and changes no state; zero exact steps remain invalid exactly as in v3.
    """

    if length <= 0 or exact_count < 1 or exact_count > length:
        raise ValueError("historical midpoint mask requires 1 <= exact_count <= length")
    mask = [False] * length
    for index in range(exact_count):
        position = math.floor((index + 0.5) * length / exact_count)
        mask[position] = True
    if sum(mask) != exact_count:
        raise AssertionError("historical midpoint construction produced duplicate positions")
    return mask


def make_periodic_mask(length: int, exact_count: int) -> list[bool]:
    """Place a new periodic schedule's exact slots from the first position.

    Inputs satisfy ``0 <= exact_count <= length`` with nonnegative length. The
    returned mask sets ``floor(j*n/k)`` for ``j=0..k-1``; zero gives all false and
    full budget gives all true. The helper is deterministic, consumes no RNG,
    and is intentionally distinct from the historical midpoint construction.
    """

    if length < 0 or exact_count < 0 or exact_count > length:
        raise ValueError("periodic mask requires 0 <= exact_count <= length")
    mask = [False] * length
    if exact_count == 0:
        return mask
    for index in range(exact_count):
        mask[math.floor(index * length / exact_count)] = True
    if sum(mask) != exact_count:
        raise AssertionError("new periodic construction produced duplicate positions")
    return mask


def make_epoch_matched_mask(
    source_mask: list[bool], epoch_offsets: list[int] | tuple[int, ...]
) -> list[bool]:
    """Match each donor epoch's exact count with the new periodic construction.

    Inputs are a donor mask and validated epoch offsets. The returned fresh mask
    preserves every epoch length and exact count, but makes no claim to preserve
    burst lengths or within-epoch phase. Source data and all RNG remain unchanged.
    """

    offsets = [int(value) for value in epoch_offsets]
    if not offsets or offsets[0] != 0 or offsets[-1] != len(source_mask):
        raise ValueError("epoch offsets must span the complete source mask")
    if offsets != sorted(offsets):
        raise ValueError("epoch offsets must be ordered")
    result: list[bool] = []
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        result.extend(make_periodic_mask(end - start, sum(source_mask[start:end])))
    return result


def make_frontloaded_mask(
    epoch_lengths: list[int] | tuple[int, ...],
    *,
    phase_fractions: tuple[float, float, float, float] = (0.40, 0.35, 0.30, 0.30),
    phase_epochs: int = 3,
) -> list[bool]:
    """Build the predefined four-phase front-loaded schedule without donor data.

    Inputs are positive epoch lengths and the frozen four fractions/three-epoch
    phase width. Each epoch budget is ``floor(n*f+0.5)`` and uses the new periodic
    helper. The fresh output consumes no RNG and is approximately, not exactly,
    donor-budget-comparable. Invalid or altered phase dimensions raise.
    """

    if phase_epochs <= 0 or len(phase_fractions) != 4:
        raise ValueError("front-loaded construction requires four positive-width phases")
    if any(length <= 0 for length in epoch_lengths):
        raise ValueError("front-loaded epoch lengths must be positive")
    result: list[bool] = []
    for epoch, length in enumerate(epoch_lengths):
        fraction = phase_fractions[min(3, epoch // phase_epochs)]
        exact_count = math.floor(length * fraction + 0.5)
        result.extend(make_periodic_mask(length, exact_count))
    return result


def make_uniform_third_mask(epoch_lengths: list[int] | tuple[int, ...]) -> list[bool]:
    """Build the predefined donor-independent one-third periodic schedule.

    Inputs are positive epoch lengths. Each budget is ``floor(n/3+0.5)`` and the
    returned mask uses the new periodic helper independently per epoch. No RNG,
    donor outcome, or mutable state is accessed.
    """

    if any(length <= 0 for length in epoch_lengths):
        raise ValueError("uniform-third epoch lengths must be positive")
    result: list[bool] = []
    for length in epoch_lengths:
        result.extend(make_periodic_mask(length, math.floor(length / 3.0 + 0.5)))
    return result


def permute_epoch_gaps(
    source_mask: list[bool],
    epoch_offsets: list[int] | tuple[int, ...],
    *,
    classifier_seed: int,
    replicate: int,
) -> tuple[list[bool], dict[str, Any]]:
    """Permute each epoch's internal synthetic gaps with the private M5 RNG.

    Inputs are a donor mask/boundaries and nonnegative classifier seed/replica.
    The output is a fresh mask plus per-epoch seeds, original/permuted gaps,
    moved-gap fractions, and degeneracy labels. Leading/trailing runs, first/last
    exact positions, gap multisets, counts, lengths, and cross-epoch runs are
    preserved. NumPy's global RNG, augmentation RNG, and source mask are unchanged.
    """

    if classifier_seed < 0 or replicate < 0:
        raise ValueError("classifier_seed and replicate must be nonnegative")
    offsets = [int(value) for value in epoch_offsets]
    if not offsets or offsets[0] != 0 or offsets[-1] != len(source_mask):
        raise ValueError("epoch offsets must span the complete source mask")
    if offsets != sorted(offsets):
        raise ValueError("epoch offsets must be ordered")
    result: list[bool] = []
    records: list[dict[str, Any]] = []
    moved_total = gap_total = 0
    for epoch, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        epoch_mask = [bool(value) for value in source_mask[start:end]]
        exact_positions = [index for index, value in enumerate(epoch_mask) if value]
        seed = 900000 + 1000 * classifier_seed + 100 * replicate + epoch
        if len(exact_positions) < 2:
            result.extend(epoch_mask)
            records.append(
                {
                    "postwarm_epoch": epoch,
                    "seed": seed,
                    "randomizable": False,
                    "unchanged": True,
                    "reason": "fewer_than_two_exact_steps",
                    "original_gaps": [],
                    "permuted_gaps": [],
                    "moved_gap_fraction": 0.0,
                }
            )
            continue
        leading = exact_positions[0]
        trailing = len(epoch_mask) - 1 - exact_positions[-1]
        gaps = [
            right - left - 1
            for left, right in zip(exact_positions[:-1], exact_positions[1:], strict=True)
        ]
        generator = np.random.default_rng(seed)
        permuted = [int(value) for value in generator.permutation(gaps).tolist()]
        reconstructed = [False] * leading + [True]
        for gap in permuted:
            reconstructed.extend([False] * gap)
            reconstructed.append(True)
        reconstructed.extend([False] * trailing)
        if len(reconstructed) != len(epoch_mask):
            raise AssertionError("gap permutation changed epoch length")
        moved = sum(left != right for left, right in zip(gaps, permuted, strict=True))
        moved_total += moved
        gap_total += len(gaps)
        result.extend(reconstructed)
        records.append(
            {
                "postwarm_epoch": epoch,
                "seed": seed,
                "randomizable": True,
                "unchanged": reconstructed == epoch_mask,
                "reason": "permutation_sampled_once",
                "original_gaps": gaps,
                "permuted_gaps": permuted,
                "moved_gap_fraction": moved / len(gaps),
            }
        )
    return result, {
        "classifier_seed": classifier_seed,
        "replicate": replicate,
        "epoch_records": records,
        "moved_gap_fraction": moved_total / gap_total if gap_total else 0.0,
        "randomizable_gap_count": gap_total,
    }


def schedule_mask_diagnostics(
    mask: list[bool], epoch_offsets: list[int] | tuple[int, ...]
) -> dict[str, Any]:
    """Describe exact budgets, positions, and full-horizon synthetic run lengths.

    Inputs are one complete schedule and its epoch boundaries. The returned
    JSON-safe record contains full/per-epoch exact counts, first/last positions,
    and a histogram of positive false runs computed across epoch boundaries.
    It has no RNG or mutation side effects and rejects inconsistent boundaries.
    """

    offsets = [int(value) for value in epoch_offsets]
    if not offsets or offsets[0] != 0 or offsets[-1] != len(mask):
        raise ValueError("epoch offsets must span the complete schedule")
    epochs: list[dict[str, Any]] = []
    for epoch, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
        positions = [index for index, value in enumerate(mask[start:end]) if value]
        epochs.append(
            {
                "postwarm_epoch": epoch,
                "length": end - start,
                "exact_count": len(positions),
                "first_exact": positions[0] if positions else None,
                "last_exact": positions[-1] if positions else None,
                "leading_synthetic": positions[0] if positions else end - start,
                "trailing_synthetic": (end - start - 1 - positions[-1]) if positions else end - start,
            }
        )
    runs: list[int] = []
    current = 0
    for exact in mask:
        if exact:
            if current:
                runs.append(current)
                current = 0
        else:
            current += 1
    if current:
        runs.append(current)
    histogram: dict[str, int] = {}
    for length in runs:
        histogram[str(length)] = histogram.get(str(length), 0) + 1
    return {
        "length": len(mask),
        "exact_count": sum(mask),
        "epoch_exact_counts": [record["exact_count"] for record in epochs],
        "epochs": epochs,
        "synthetic_run_lengths": runs,
        "synthetic_run_length_histogram": histogram,
    }


class IndexedDataset(Dataset):
    """Attach stable source indices to samples without changing dataset behavior."""

    def __init__(self, dataset: Dataset) -> None:
        """Wrap a map-style dataset; item access may retain its transform side effects."""

        self.dataset = dataset

    def __len__(self) -> int:
        """Return the wrapped dataset length without consuming samples or RNG."""

        return len(self.dataset)

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor | int, int]:
        """Return input, label, and stable integer index for one wrapped sample."""

        inputs, label = self.dataset[index]
        return inputs, label, index


class PhaseTimer:
    """Measure one non-overlapping phase and synchronize CUDA only at boundaries."""

    def __init__(self, ledger: TimingLedger, device: torch.device | str) -> None:
        """Bind a timing ledger and device without starting a timed interval."""

        self.ledger = ledger
        self.device = torch.device(device)
        self.active_component: str | None = None
        self.started = 0.0

    def start(self, component: str) -> None:
        """Start one phase, failing if another phase is already active."""

        if self.active_component is not None:
            raise RuntimeError("timing phases may not overlap")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        self.active_component = component
        self.started = time.perf_counter()

    def stop(self) -> float:
        """Stop the active phase, add elapsed seconds, and return the duration."""

        if self.active_component is None:
            raise RuntimeError("no timing phase is active")
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed = time.perf_counter() - self.started
        self.ledger.add(self.active_component, elapsed)
        self.active_component = None
        return elapsed


def configure_determinism(config: RunConfig, device: torch.device | str) -> None:
    """Apply the resolved strict deterministic and TF32 backend contract.

    This mutates process-global PyTorch backend flags. Requested CUDA must exist;
    unsupported deterministic operations later fail rather than warning or falling
    back to a different algorithm.
    """

    selected = torch.device(device)
    execution = config.values["execution"]

    
    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; CPU fallback is disabled"
        )

    if selected.type == "cuda":
        workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")

        if workspace is None:
            if torch.cuda.is_initialized():
                raise RuntimeError(
                    "CUDA was initialized before CUBLAS_WORKSPACE_CONFIG "
                    "was set. Restart the Python process."
                )
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

        elif workspace not in {":4096:8", ":16:8"}:
            raise RuntimeError(
                "Unsupported CUBLAS_WORKSPACE_CONFIG for strict determinism"
            )
    torch.backends.cudnn.benchmark = bool(execution["cudnn_benchmark"])
    
    torch.backends.cudnn.deterministic = bool(execution["cudnn_deterministic"])
    torch.backends.cuda.matmul.allow_tf32 = bool(execution["matmul_tf32"])
    torch.backends.cudnn.allow_tf32 = bool(execution["cudnn_tf32"])
    mode = config.values["scientific"]["reproducibility"]["deterministic_algorithms"]
    if mode != "strict":
        raise ValueError("M2 controlled execution supports only strict deterministic algorithms")
    torch.use_deterministic_algorithms(True, warn_only=False)


def seed_initialization(config: RunConfig) -> int:
    """Seed Python, NumPy, and Torch for the configured classifier initialization."""

    classifier_seed = int(config.values["scientific"]["identity"]["classifier_seed"])
    seed = 18371 + classifier_seed
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def epoch_loader(
    dataset: Dataset,
    config: RunConfig,
    global_epoch: int,
    *,
    shuffle: bool,
) -> DataLoader:
    """Create the v3-seeded epoch loader and deterministically reseed augmentation.

    The loader yields inputs, labels, and stable indices. With workers zero,
    random transforms consume the explicitly reseeded global streams. Creating a
    training loader therefore mutates Python, NumPy, and Torch RNG states.
    """

    scientific = config.values["scientific"]
    classifier_seed = int(scientific["identity"]["classifier_seed"])
    augmentation_seed = 245600 + 100000 * classifier_seed + global_epoch
    if shuffle:
        random.seed(augmentation_seed)
        np.random.seed(augmentation_seed % (2**32 - 1))
        torch.manual_seed(augmentation_seed)
    generator = torch.Generator().manual_seed(augmentation_seed)
    return DataLoader(
        IndexedDataset(dataset),
        batch_size=int(scientific["training"]["batch_size"]),
        shuffle=shuffle,
        generator=generator,
        drop_last=bool(scientific["training"]["drop_last"]),
        num_workers=int(config.values["execution"]["workers"]),
        pin_memory=bool(config.values["execution"]["pin_memory"]),
        persistent_workers=False,
    )


def _index_array_sha256(indices: np.ndarray) -> str:
    """Hash one one-dimensional int64 split array with shape and byte identity.

    The input contains materialized dataset indices. The lowercase SHA-256 output
    is stable for identical values and ordering; the array and all RNG state are
    unchanged. Non-int64 or non-vector arrays fail to avoid ambiguous identities.
    """

    if indices.dtype != np.int64 or indices.ndim != 1:
        raise ValueError("split indices must be a one-dimensional int64 array")
    digest = hashlib.sha256()
    digest.update(b"phase_d_v4_indices_v1\0")
    digest.update(len(indices).to_bytes(8, byteorder="little", signed=False))
    digest.update(indices.tobytes(order="C"))
    return digest.hexdigest()


def make_split_indices(
    dataset_size: int, validation_size: int, split_seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Create deterministic train/validation index arrays with the frozen rule.

    Inputs are population size, validation prefix length, and Torch generator
    seed. Outputs are copied int64 training and validation arrays: validation is
    the permutation prefix and training the remainder. Only a private CPU Torch
    generator is consumed; invalid sizes fail and global RNG/file state is unchanged.
    """

    if dataset_size <= 0 or validation_size <= 0 or validation_size >= dataset_size:
        raise ValueError("split sizes must satisfy 0 < validation_size < dataset_size")
    if split_seed < 0:
        raise ValueError("split seed must be nonnegative")
    generator = torch.Generator().manual_seed(split_seed)
    permutation = torch.randperm(dataset_size, generator=generator).numpy().astype(np.int64, copy=False)
    return permutation[validation_size:].copy(), permutation[:validation_size].copy()


def prepare_cifar10_split(
    config: RunConfig,
    data_directory: str | Path,
    split_path: str | Path,
    *,
    download: bool = False,
) -> dict[str, str]:
    """Materialize the frozen CIFAR-10 train/validation split without training.

    Inputs are the resolved reference config, local dataset cache, fresh split
    artifact path, and an explicit download choice. Only CIFAR-10's official
    training partition is opened; the test partition is never loaded. The split
    NPZ is written atomically and its index hashes returned. Existing destinations
    are rejected, dataset length mismatches fail, and no model or run is created.
    """

    validate_config(config)
    from torchvision import datasets

    data_values = config.values["scientific"]["data"]
    setup_started = time.perf_counter()
    dataset = datasets.CIFAR10(
        root=str(Path(data_directory)), train=True, download=download, transform=None
    )
    expected_size = int(data_values["dataset_size"])
    if len(dataset) != expected_size:
        raise ValueError(f"CIFAR-10 training partition has {len(dataset)} items, expected {expected_size}")
    validation_size = int(data_values["validation_size"])
    training_indices, validation_indices = make_split_indices(
        expected_size, validation_size, int(data_values["split_seed"])
    )
    if len(training_indices) != int(data_values["training_size"]):
        raise ValueError("materialized training split size differs from resolved config")
    destination = Path(split_path)
    if destination.exists():
        raise FileExistsError("split artifact already exists; use a fresh explicit path")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            training_indices=training_indices,
            validation_indices=validation_indices,
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    identities = {
        "training_indices_sha256": _index_array_sha256(training_indices),
        "validation_indices_sha256": _index_array_sha256(validation_indices),
    }
    setup_seconds = time.perf_counter() - setup_started
    atomic_write_json(
        destination.with_suffix(destination.suffix + ".manifest.json"),
        {
            "schema_version": SCHEMA_VERSION,
            "dataset": "CIFAR10",
            "source_partition": "official_train_only",
            "split_seed": int(data_values["split_seed"]),
            "training_size": len(training_indices),
            "validation_size": len(validation_indices),
            **identities,
            "artifact_sha256": sha256_file(destination),
            "final_test_loaded": False,
            "download_requested": bool(download),
            "dataset_and_split_setup_seconds": setup_seconds,
            "excluded_from_steady_training_timing": True,
        },
    )
    return identities


def load_cifar10_datasets(
    config: RunConfig,
    data_directory: str | Path,
    split_path: str | Path,
) -> tuple[Dataset, Dataset, dict[str, str]]:
    """Load frozen transformed CIFAR-10 subsets from a verified split artifact.

    Inputs are a resolved config, existing local dataset cache, and split NPZ.
    Outputs are augmented training and deterministic validation subsets plus
    materialized index hashes. The official test partition is never opened and
    downloading is disabled. Split sizes, bounds, disjointness, manifest hashes,
    and config transform constants are verified before returning datasets.
    """

    validate_config(config)
    from torchvision import datasets, transforms

    data_values = config.values["scientific"]["data"]
    mean = tuple(float(value) for value in data_values["train_transform"][-1]["mean"])
    std = tuple(float(value) for value in data_values["train_transform"][-1]["std"])
    normalize = transforms.Normalize(mean, std)
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            normalize,
        ]
    )
    validation_transform = transforms.Compose([transforms.ToTensor(), normalize])
    augmented = datasets.CIFAR10(
        root=str(Path(data_directory)), train=True, download=False, transform=train_transform
    )
    validation_source = datasets.CIFAR10(
        root=str(Path(data_directory)), train=True, download=False, transform=validation_transform
    )
    if len(augmented) != int(data_values["dataset_size"]) or len(validation_source) != len(augmented):
        raise ValueError("cached CIFAR-10 training partition size differs from resolved config")
    split_file = Path(split_path)
    manifest = json.loads(
        split_file.with_suffix(split_file.suffix + ".manifest.json").read_text(encoding="utf-8")
    )
    if sha256_file(split_file) != manifest.get("artifact_sha256"):
        raise ValueError("split artifact checksum mismatch")
    with np.load(split_file, allow_pickle=False) as values:
        training_indices = np.asarray(values["training_indices"])
        validation_indices = np.asarray(values["validation_indices"])
    if (
        training_indices.dtype != np.int64
        or validation_indices.dtype != np.int64
        or training_indices.ndim != 1
        or validation_indices.ndim != 1
    ):
        raise ValueError("split artifact arrays must be one-dimensional int64 values")
    identities = {
        "training_indices_sha256": _index_array_sha256(training_indices),
        "validation_indices_sha256": _index_array_sha256(validation_indices),
    }
    if any(identities[key] != manifest.get(key) for key in identities):
        raise ValueError("split index identity differs from its manifest")
    if len(training_indices) != int(data_values["training_size"]) or len(validation_indices) != int(data_values["validation_size"]):
        raise ValueError("split sizes differ from resolved config")
    combined = np.concatenate((training_indices, validation_indices))
    if len(np.unique(combined)) != len(combined) or combined.min() < 0 or combined.max() >= len(augmented):
        raise ValueError("split indices are overlapping, duplicated, or out of bounds")
    return (
        Subset(augmented, training_indices.tolist()),
        Subset(validation_source, validation_indices.tolist()),
        identities,
    )


def _update_data_hash(digest: Any, inputs: Tensor, labels: Tensor, indices: Tensor) -> None:
    """Add actual CPU batch tensors and indices to a streaming SHA-256 digest."""

    for name, tensor in (("inputs", inputs), ("labels", labels), ("indices", indices)):
        contiguous = tensor.detach().cpu().contiguous()
        digest.update(name.encode("ascii") + b"\0")
        digest.update(str(contiguous.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(contiguous.shape)).encode("ascii") + b"\0")
        digest.update(contiguous.numpy().tobytes(order="C"))


@torch.no_grad()
def evaluate(
    model: SplitResNet,
    dataset: Dataset,
    config: RunConfig,
    device: torch.device | str,
) -> dict[str, float | int]:
    """Evaluate CE and accuracy without changing production modes, buffers, or RNG.

    Validation inputs come from a map-style dataset and are batched without
    shuffling. The returned accuracy is a fraction and CE is per sample. Model
    training flags and all captured RNG states are restored even on failure.
    """

    selected = torch.device(device)
    flags = {module: module.training for module in model.modules()}
    rng_state = capture_rng_state()
    try:
        model.eval()
        loader = epoch_loader(dataset, config, 0, shuffle=False)
        correct = 0
        count = 0
        summed_loss = 0.0
        for inputs, labels, _indices in loader:
            inputs = inputs.to(selected, non_blocking=True)
            labels = labels.to(selected, non_blocking=True)
            logits = model(inputs)
            summed_loss += float(F.cross_entropy(logits, labels, reduction="sum").item())
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            count += int(labels.shape[0])
        if count == 0:
            raise ValueError("validation dataset must not be empty")
        return {"accuracy": correct / count, "ce": summed_loss / count, "samples": count}
    finally:
        for module, was_training in flags.items():
            module.training = was_training
        restore_rng_state(rng_state)


def calibrate_controller(
    model: SplitResNet,
    predictor: GradientPredictor,
    dataset: Dataset,
    scale: float,
    config: RunConfig,
    device: torch.device | str,
) -> dict[str, Any]:
    """Measure frozen historical controller calibration on a disposable copy.

    Inputs are warm-up-final production modules, training dataset, positive scale,
    resolved config, and device. Up to the configured 48 v3-seeded training-stream
    batches are evaluated in post-warm mode on copied modules. The output contains
    pre-fit absolute/relative errors and fixed quantiles in normalized target
    units. Production weights, BatchNorm buffers, optimizer state, RNG, and loader
    state remain unchanged; fewer than three batches or nonfinite values fail.
    """

    if scale <= 0:
        raise ValueError("controller calibration requires a positive fixed scale")
    scientific = config.values["scientific"]
    requested = int(scientific["calibration_and_profile"]["calibration_batches"])
    calibration_epoch = int(
        scientific["calibration_and_profile"].get("calibration_loader_epoch", 911)
    )
    rng_state = capture_rng_state()
    try:
        model_copy = copy.deepcopy(model)
        predictor_copy = copy.deepcopy(predictor)
        normalization = NormalizationMode(scientific["identity"]["normalization"])
        set_training_mode(model_copy, predictor_copy, "postwarm", normalization)
        predictor_copy.eval()
        absolute_errors: list[float] = []
        relative_errors: list[float] = []
        for batch_index, (cpu_inputs, cpu_labels, _indices) in enumerate(
            epoch_loader(dataset, config, calibration_epoch, shuffle=True)
        ):
            if batch_index >= requested:
                break
            inputs = cpu_inputs.to(device, non_blocking=True)
            labels = cpu_labels.to(device, non_blocking=True)
            # Trunk BN updates occur only on this disposable copy, matching the
            # historical post-warm calibration stream without contaminating reuse.
            with torch.no_grad():
                boundary = model_copy.trunk(inputs)
            leaf = boundary.detach().requires_grad_(True)
            summed_loss = F.cross_entropy(model_copy.head(leaf), labels, reduction="sum")
            target = torch.autograd.grad(summed_loss, leaf)[0].detach() / scale
            with torch.no_grad():
                estimate = predictor_copy(boundary.detach(), labels)
                error_abs = float((estimate - target).square().mean().sqrt().item())
                error_relative = relative_error(
                    estimate,
                    target,
                    float(scientific["controller"]["relative_denominator_floor"]),
                )
            if not math.isfinite(error_abs) or not math.isfinite(error_relative):
                raise ValueError("controller calibration produced a nonfinite error")
            absolute_errors.append(error_abs)
            relative_errors.append(error_relative)
        if len(absolute_errors) < 3:
            raise RuntimeError("controller calibration requires at least three training batches")
        low = float(scientific["controller"]["low_quantile"])
        high = float(scientific["controller"]["high_quantile"])
        return {
            "schema_version": SCHEMA_VERSION,
            "source": "training_stream_disposable_warmup_copy",
            "loader_epoch_seed_component": calibration_epoch,
            "calibration_batches_requested": requested,
            "calibration_batches_observed": len(absolute_errors),
            "quantiles": [low, high],
            "absolute_errors": absolute_errors,
            "relative_errors": relative_errors,
            "q_abs_low": float(np.quantile(absolute_errors, low, method="linear")),
            "q_abs_high": float(np.quantile(absolute_errors, high, method="linear")),
            "median_abs": float(np.median(absolute_errors)),
            "median_rel": float(np.median(relative_errors)),
            "production_state_unchanged_by_construction": True,
        }
    finally:
        restore_rng_state(rng_state)


def _accumulate_step(state: RunState, result: StepResult) -> None:
    """Add detached step counters to mutable run state without interpreting metrics."""

    state.exact_trunk_backwards += result.exact_trunk_backwards
    state.synthetic_backwards += result.synthetic_backwards
    state.zero_backwards += result.zero_backwards
    state.skipped_slots += result.skipped_slots
    state.trunk_optimizer_steps += result.trunk_optimizer_steps
    state.head_optimizer_steps += result.head_optimizer_steps
    state.predictor_fits += result.predictor_fits
    state.predictor_inferences += result.predictor_inferences


def replay_slot_decision(method: MethodId | str, mask_entry: bool) -> dict[str, Any]:
    """Resolve one immutable replay-mask entry into independent update decisions.

    Inputs are an M4 replay method and one prescribed boolean slot. The output
    explicitly separates trunk exactness, head stepping, predictor fitting,
    synthetic behavior, and shadow inference. It has no side effects and rejects
    non-replay methods instead of making an online switching decision.
    """

    selected = MethodId(method)
    if selected not in REPLAY_METHODS:
        raise ValueError("replay_slot_decision accepts only M4 replay methods")
    exact_slot = bool(mask_entry)
    if selected == MethodId.RELATIVE_REPLAY_EXACT_TRUNK:
        return {
            "trunk_exact": True,
            "head_step": exact_slot,
            "predictor_fit": False,
            "synthetic_kind": None,
            "shadow_predictor": False,
        }
    shadow = selected in {
        MethodId.RELATIVE_REPLAY_ZERO_SHADOW,
        MethodId.RELATIVE_REPLAY_SKIP_SHADOW,
    }
    predictor_fit = exact_slot and selected not in {
        MethodId.RELATIVE_REPLAY_ZERO_FAST,
        MethodId.RELATIVE_REPLAY_SKIP_FAST,
    }
    if exact_slot:
        return {
            "trunk_exact": True,
            "head_step": True,
            "predictor_fit": predictor_fit,
            "synthetic_kind": None,
            "shadow_predictor": False,
        }
    behavior = {
        MethodId.RELATIVE_REPLAY_LEARNED: SyntheticBehavior.LEARNED,
        MethodId.RELATIVE_REPLAY_ZERO_SHADOW: SyntheticBehavior.ZERO,
        MethodId.RELATIVE_REPLAY_SKIP_SHADOW: SyntheticBehavior.SKIP,
        MethodId.RELATIVE_REPLAY_ZERO_FAST: SyntheticBehavior.ZERO,
        MethodId.RELATIVE_REPLAY_SKIP_FAST: SyntheticBehavior.SKIP,
    }[selected]
    return {
        "trunk_exact": False,
        "head_step": False,
        "predictor_fit": False,
        "synthetic_kind": behavior,
        "shadow_predictor": shadow,
    }


def schedule_slot_decision(config: RunConfig, mask_entry: bool) -> dict[str, Any]:
    """Resolve one M5 fixed or source-derived schedule entry into update actions.

    Inputs are a resolved M5 config and prescribed boolean slot. The returned
    decision uses the config's learned/zero/skip behavior, fits a predictor only
    when one exists on exact slots, and never consults feedback or future data.
    It has no side effects and rejects methods outside M5 schedule controls.
    """

    identity = config.values["scientific"]["identity"]
    method = MethodId(identity["method"])
    if method not in SOURCE_DERIVED_SCHEDULE_METHODS | FIXED_SCHEDULE_METHODS:
        raise ValueError("schedule_slot_decision accepts only M5 schedule methods")
    behavior = SyntheticBehavior(identity["synthetic_behavior"])
    if mask_entry:
        return {
            "trunk_exact": True,
            "head_step": True,
            "predictor_fit": behavior == SyntheticBehavior.LEARNED,
            "synthetic_kind": None,
            "shadow_predictor": False,
        }
    return {
        "trunk_exact": False,
        "head_step": False,
        "predictor_fit": False,
        "synthetic_kind": behavior,
        "shadow_predictor": False,
    }


def execute_slot(
    model: SplitResNet,
    predictor: GradientPredictor | None,
    optimizers: Mapping[str, torch.optim.Optimizer | None],
    inputs: Tensor,
    labels: Tensor,
    scale: float,
    config: RunConfig,
    decision: Mapping[str, Any],
) -> StepResult:
    """Execute one already-decided replay slot without consulting a controller.

    Inputs are live training state, a ``[B,3,32,32]`` batch, labels, target
    scale, config, and the explicit decision from :func:`replay_slot_decision`.
    The returned detached ``StepResult`` reports actual operations. Model,
    optimizer, predictor, gradients, and BN buffers change according to the
    chosen exact/zero/skip/learned path; inconsistent decisions fail first.
    """

    trunk_exact = bool(decision.get("trunk_exact"))
    head_step = bool(decision.get("head_step"))
    predictor_fit = bool(decision.get("predictor_fit"))
    synthetic_kind = decision.get("synthetic_kind")
    shadow_predictor = bool(decision.get("shadow_predictor"))
    if trunk_exact:
        if synthetic_kind is not None or shadow_predictor:
            raise ValueError("exact replay slot cannot declare a synthetic or shadow path")
        return exact_step(
            model, predictor, optimizers, inputs, labels, scale, config,
            fit_predictor=predictor_fit, head_update=head_step,
        )
    if head_step or predictor_fit or synthetic_kind is None:
        raise ValueError("synthetic replay slot cannot step head, fit predictor, or omit kind")
    return synthetic_step(
        model, predictor, optimizers, inputs, labels, scale, config,
        kind=SyntheticBehavior(synthetic_kind), shadow_predictor=shadow_predictor,
    )


def run_epoch(
    model: SplitResNet,
    predictor: GradientPredictor | None,
    optimizers: Mapping[str, torch.optim.Optimizer | None],
    dataset: Dataset,
    config: RunConfig,
    device: torch.device | str,
    state: RunState,
    ledger: TimingLedger,
    *,
    global_epoch: int,
    controller: HistoricalFeedbackController | None = None,
    fit_predictor_warmup: bool = True,
) -> dict[str, Any]:
    """Execute one deterministic training epoch and return counters plus data hash.

    Warm-up is all-exact. Post-warm-up supports historical methods plus M4 fixed
    replay masks. Replay reads the next prescribed entry and never asks a
    controller. The function mutates modules, optimizers, controller, RNG, and
    run counters; masks must already have passed pre-update validation.
    """

    scientific = config.values["scientific"]
    identity = scientific["identity"]
    method = MethodId(identity["method"])
    normalization = NormalizationMode(identity["normalization"])
    warmup_epochs = int(scientific["training"]["warmup_epochs"])
    postwarm_epochs = int(scientific["training"]["postwarm_epochs"])
    stage = RunStage.WARMUP if global_epoch < warmup_epochs else RunStage.POSTWARM
    state.stage = stage
    state.lr_global_epoch = global_epoch
    learning_rate = set_epoch_lr(optimizers, config, global_epoch)
    set_training_mode(model, predictor, "warmup" if stage == RunStage.WARMUP else "postwarm", normalization)
    timer = PhaseTimer(ledger, device)
    component = "warmup_training" if stage == RunStage.WARMUP else "postwarm_training"
    timer.start(component)
    scale_seconds = 0.0
    digest = hashlib.sha256()
    batches = samples = 0
    before = {
        name: getattr(state, name)
        for name in (
            "exact_trunk_backwards", "synthetic_backwards", "zero_backwards",
            "skipped_slots",
            "trunk_optimizer_steps", "head_optimizer_steps", "predictor_fits",
            "predictor_inferences",
        )
    }
    for epoch_batch, (cpu_inputs, cpu_labels, indices) in enumerate(
        epoch_loader(dataset, config, global_epoch, shuffle=True)
    ):
        _update_data_hash(digest, cpu_inputs, cpu_labels, indices)
        inputs = cpu_inputs.to(device, non_blocking=True)
        labels = cpu_labels.to(device, non_blocking=True)
        if state.scale is None:
            scale_started = time.perf_counter()
            state.scale = estimate_initial_scale(model, inputs, labels)
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize(device)
            scale_seconds += time.perf_counter() - scale_started

        if stage == RunStage.WARMUP:
            result = exact_step(
                model, predictor, optimizers, inputs, labels, state.scale, config,
                fit_predictor=fit_predictor_warmup,
            )
        elif method in {MethodId.EXACT, MethodId.EXACT_STANDARD_BN}:
            result = exact_step(
                model, predictor, optimizers, inputs, labels, state.scale, config,
                fit_predictor=False,
            )
        elif method == MethodId.SCHEDULED:
            if scheduled_exact(state.postwarm_step, global_epoch - warmup_epochs, postwarm_epochs):
                result = exact_step(
                    model, predictor, optimizers, inputs, labels, state.scale, config,
                    fit_predictor=True,
                )
            else:
                result = synthetic_step(
                    model, predictor, optimizers, inputs, labels, state.scale, config,
                    kind=SyntheticBehavior.LEARNED,
                )
        elif method in {MethodId.RELATIVE_FEEDBACK, MethodId.ABSOLUTE_FEEDBACK}:
            if controller is None:
                raise ValueError("feedback method requires controller state")
            if state.pending_quota == 0:
                result = exact_step(
                    model, predictor, optimizers, inputs, labels, state.scale, config,
                    fit_predictor=True,
                )
                state.pending_quota, details = controller.choose(
                    float(result.metrics["error_abs"]), float(result.metrics["error_relative"])
                )
                state.controller_events.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "run_id": config.values["output"]["run_id"],
                        "seed": int(identity["classifier_seed"]),
                        "method": method.value,
                        "global_epoch": global_epoch,
                        "epoch_batch": epoch_batch,
                        "global_step": state.global_step,
                        **details,
                        "checkpoint_sequence": state.checkpoint_sequence + 1,
                    }
                )
            else:
                result = synthetic_step(
                    model, predictor, optimizers, inputs, labels, state.scale, config,
                    kind=SyntheticBehavior.LEARNED,
                )
                state.pending_quota -= 1
        elif method in MASK_EXECUTION_METHODS:
            if controller is not None:
                raise ValueError("mask-driven methods must not have an online controller")
            if state.mask_cursor >= len(state.replay_schedule_mask):
                raise ValueError("schedule mask exhausted before the post-warm-up horizon")
            prescribed = state.replay_schedule_mask[state.mask_cursor]
            decision = (
                replay_slot_decision(method, prescribed)
                if method in REPLAY_METHODS else schedule_slot_decision(config, prescribed)
            )
            result = execute_slot(
                model, predictor, optimizers, inputs, labels, state.scale, config, decision
            )
        else:
            raise NotImplementedError(
                f"{method.value} execution belongs to M5 or later"
            )
        _accumulate_step(state, result)
        if stage == RunStage.POSTWARM:
            state.executed_exact_mask.append(result.update_type == UpdateType.EXACT)
            state.executed_head_mask.append(result.head_optimizer_steps == 1)
        state.global_step += 1
        if stage == RunStage.POSTWARM:
            state.postwarm_step += 1
            # Cursor tracks consumed prescribed slots immediately; delaying this
            # until epoch end would replay the first entry for every epoch batch.
            state.mask_cursor = state.postwarm_step
        batches += 1
        samples += int(labels.shape[0])
    elapsed = timer.stop()
    if scale_seconds:
        # Scale estimation occurred inside the first training interval, so move
        # that measured duration rather than counting it in both components.
        ledger.components[component] -= scale_seconds
        ledger.add("scale_estimation", scale_seconds)
        elapsed -= scale_seconds
    state.mask_cursor = state.postwarm_step
    if stage == RunStage.POSTWARM:
        state.mask_epoch_offsets.append(len(state.executed_exact_mask))
        if method in MASK_EXECUTION_METHODS:
            replay_epoch = global_epoch - warmup_epochs
            expected_boundary = state.replay_epoch_offsets[replay_epoch + 1]
            if state.mask_cursor != expected_boundary:
                raise ValueError("replay mask cursor and committed epoch boundary diverged")
    return {
        "stage": stage.value,
        "global_epoch": global_epoch,
        "stage_epoch": global_epoch if stage == RunStage.WARMUP else global_epoch - warmup_epochs,
        "lr": learning_rate,
        "samples_seen": samples,
        "batches_seen": batches,
        "training_seconds": elapsed,
        "data_order_sha256": digest.hexdigest(),
        **{name: getattr(state, name) - value for name, value in before.items()},
    }


def _cpu_tree(value: Any) -> Any:
    """Detach tensors to CPU recursively for graph-free checkpoint persistence."""

    if isinstance(value, Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _resolved_artifacts(artifacts: Mapping[str, str]) -> dict[str, str]:
    """Add current v4 source hashes to caller-supplied immutable artifact hashes."""

    resolved = dict(artifacts)
    resolved["phase_d_v4_io"] = sha256_file(PROJECT_ROOT / "phase_d_v4_io.py")
    resolved["gradient_phase_d_resnet_v4"] = sha256_file(Path(__file__))
    resolved["phase_d_v4_diagnostics"] = sha256_file(PROJECT_ROOT / "phase_d_v4_diagnostics.py")
    resolved["phase_d_v4_analysis"] = sha256_file(PROJECT_ROOT / "phase_d_v4_analysis.py")
    analysis_cli = PROJECT_ROOT / "analyze_phase_d_v4.py"
    if analysis_cli.is_file():
        resolved["analyze_phase_d_v4"] = sha256_file(analysis_cli)
    return resolved


def _mask_prefix_hash(mask: list[bool]) -> str:
    """Hash executed boolean mask bytes, including its explicit current length."""

    digest = hashlib.sha256()
    digest.update(len(mask).to_bytes(8, byteorder="little", signed=False))
    digest.update(bytes(int(item) for item in mask))
    return digest.hexdigest()


def _checkpoint_payload(
    config: RunConfig,
    model: SplitResNet,
    predictor: GradientPredictor | None,
    optimizers: Mapping[str, torch.optim.Optimizer | None],
    state: RunState,
    ledger: TimingLedger,
    controller: HistoricalFeedbackController | None,
    environment: Mapping[str, Any],
    scientific_hash: str,
    execution_hash: str,
    artifacts: Mapping[str, str],
    data_identities: Mapping[str, str],
    run_id: str,
    replay_mask_metadata: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build a complete graph-free epoch-boundary checkpoint payload in memory."""

    identity = config.values["scientific"]["identity"]
    return {
        "schema_version": SCHEMA_VERSION,
        "scientific_fingerprint": scientific_hash,
        "execution_fingerprint": execution_hash,
        "artifacts": dict(artifacts),
        "data_identities": dict(data_identities),
        "resolved_config": config.to_dict(),
        "run_id": run_id,
        "seed": int(identity["classifier_seed"]),
        "method": identity["method"],
        "normalization": identity["normalization"],
        "retrospective": bool(identity["retrospective"]),
        "deployment_eligible": bool(identity["deployment_eligible"]),
        "stage": state.stage.value,
        "completed_epoch": state.completed_epoch,
        "next_epoch": state.next_epoch,
        "checkpoint_sequence": state.checkpoint_sequence,
        "model": _cpu_tree(model.state_dict()),
        "predictor": None if predictor is None else _cpu_tree(predictor.state_dict()),
        "optimizers": {
            name: None if optimizer is None else _cpu_tree(optimizer.state_dict())
            for name, optimizer in optimizers.items()
        },
        "scale": state.scale,
        "lr_global_epoch": state.lr_global_epoch,
        "controller": None if controller is None else controller.state_dict(),
        "pending_quota": state.pending_quota,
        "mask_role": identity["mask_role"],
        "source_mask_sha256": (
            replay_mask_metadata.get("construction_details", {}).get("source_mask_sha256")
            if replay_mask_metadata is not None
            and isinstance(replay_mask_metadata.get("construction_details"), Mapping)
            and replay_mask_metadata.get("construction_details", {}).get("source_mask_sha256")
            else identity.get("mask_sha256")
        ),
        "executed_mask_sha256": _mask_prefix_hash(state.executed_exact_mask),
        "executed_head_mask_sha256": _mask_prefix_hash(state.executed_head_mask),
        "replay_schedule_sha256": (
            None if replay_mask_metadata is None else replay_mask_metadata["sha256"]
        ),
        "replay_mask_metadata": (
            None if replay_mask_metadata is None else dict(replay_mask_metadata)
        ),
        "run_state": run_state_to_dict(state),
        "rng_state": capture_rng_state(),
        "timing_ledger": ledger.to_dict(),
        "environment": dict(environment),
        "run_complete": state.completed,
    }


def _restore_optimizer_devices(optimizer: torch.optim.Optimizer | None) -> None:
    """Move loaded optimizer tensor state to each parameter's current device."""

    if optimizer is None:
        return
    for parameter, parameter_state in optimizer.state.items():
        for key, value in parameter_state.items():
            if isinstance(value, Tensor):
                parameter_state[key] = value.to(parameter.device)


def _validate_data_identities(data_identities: Mapping[str, str]) -> None:
    """Require nonempty SHA-256 identities for train/validation split content."""

    for key in ("training_indices_sha256", "validation_indices_sha256"):
        value = data_identities.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"missing valid data identity: {key}")
        int(value, 16)


def _make_controller(
    config: RunConfig, calibration: Mapping[str, Any] | None
) -> HistoricalFeedbackController | None:
    """Construct a historical feedback controller only for feedback methods."""

    method = MethodId(config.values["scientific"]["identity"]["method"])
    if method not in {MethodId.RELATIVE_FEEDBACK, MethodId.ABSOLUTE_FEEDBACK}:
        return None
    if calibration is None:
        raise ValueError("feedback execution requires training-stream calibration state")
    return HistoricalFeedbackController(method.value, config, calibration)


def _controller_from_checkpoint(
    config: RunConfig, saved_state: Mapping[str, Any] | None
) -> HistoricalFeedbackController | None:
    """Recreate a feedback controller directly from its complete saved state.

    Inputs are a resolved config and optional checkpoint controller mapping. A
    non-feedback method returns ``None`` and must not supply state. Feedback state
    seeds a temporary constructor and is then loaded exactly, preserving history,
    thresholds, EMA, and counts without recalibration or RNG side effects.
    """

    method = MethodId(config.values["scientific"]["identity"]["method"])
    if method not in {MethodId.RELATIVE_FEEDBACK, MethodId.ABSOLUTE_FEEDBACK}:
        if saved_state is not None:
            raise ValueError("non-feedback checkpoint unexpectedly contains controller state")
        return None
    if saved_state is None:
        return None
    calibration = {
        "relative_errors": list(saved_state["history"]),
        "q_abs_low": float(saved_state["absolute_low"]),
        "q_abs_high": float(saved_state["absolute_high"]),
        "median_abs": float(saved_state["absolute_median"]),
    }
    controller = HistoricalFeedbackController(method.value, config, calibration)
    controller.load_state_dict(saved_state)
    return controller


def _method_uses_predictor(config: RunConfig) -> bool:
    """Return whether one resolved method must allocate predictor state.

    The input config is read only. M4 shadow/exact-trunk and learned schedule
    conditions retain predictors; fixed zero/skip schedules omit them. The
    boolean output changes no modules, RNG, optimizer, or file state.
    """

    identity = config.values["scientific"]["identity"]
    method = MethodId(identity["method"])
    if method in FIXED_SCHEDULE_METHODS:
        return SyntheticBehavior(identity["synthetic_behavior"]) == SyntheticBehavior.LEARNED
    return method in PREDICTOR_REQUIRED_METHODS


def _expected_postwarm_offsets(dataset: Dataset, config: RunConfig) -> tuple[int, ...]:
    """Return deterministic post-warm-up batch boundaries for mask validation.

    Inputs are a nonempty map-style training dataset and resolved batch/horizon
    config. The output counts slots with the runner's non-dropping loader and has
    no RNG, loader, model, or file side effects.
    """

    batch_size = int(config.values["scientific"]["training"]["batch_size"])
    batches = math.ceil(len(dataset) / batch_size)
    if batches <= 0:
        raise ValueError("replay training dataset must not be empty")
    epochs = int(config.values["scientific"]["training"]["postwarm_epochs"])
    return tuple(epoch * batches for epoch in range(epochs + 1))


def _mask_metadata_dict(metadata: MaskMetadata) -> dict[str, Any]:
    """Convert validated replay metadata into a JSON/checkpoint-safe mapping.

    The input immutable record is not changed. The output preserves role, source
    identity, epoch boundaries, and source timing components without live objects.
    """

    return {
        "role": metadata.role.value,
        "length": metadata.length,
        "epoch_offsets": list(metadata.epoch_offsets),
        "sha256": metadata.sha256,
        "construction": metadata.construction,
        "construction_version": metadata.construction_version,
        "source_run_id": metadata.source_run_id,
        "source_checkpoint_sha256": metadata.source_checkpoint_sha256,
        "source_seed": metadata.source_seed,
        "source_method": metadata.source_method,
        "source_scientific_fingerprint": metadata.source_scientific_fingerprint,
        "source_timing_ledger": metadata.source_timing_ledger,
        "construction_details": metadata.construction_details,
    }


def _prepare_schedule_inputs(
    config: RunConfig,
    dataset: Dataset,
    mask_path: str | Path | None,
    mask_manifest_path: str | Path | None,
) -> tuple[list[bool], MaskMetadata] | None:
    """Validate or construct every M4/M5 schedule before model initialization.

    Inputs are config, dataset, and optional donor artifact paths. M4 replay uses
    the donor unchanged; source-derived M5 methods verify then transform it; fixed
    M5 methods construct from epoch lengths alone. The returned mask/metadata are
    immutable inputs to training. All incompatibilities raise before RNG,
    parameters, output directories, or training state change.
    """

    method = MethodId(config.values["scientific"]["identity"]["method"])
    identity = config.values["scientific"]["identity"]
    offsets = _expected_postwarm_offsets(dataset, config)
    epoch_lengths = [end - start for start, end in zip(offsets[:-1], offsets[1:], strict=True)]
    if method in FIXED_SCHEDULE_METHODS:
        if mask_path is not None or mask_manifest_path is not None:
            raise ValueError("fixed schedules must not receive donor mask artifacts")
        contract = config.values["scientific"]["schedule_contract"]
        if method == MethodId.FIXED_FRONTLOADED:
            fractions = tuple(float(value) for value in contract["fixed_frontloaded_epoch_phase_fractions"])
            schedule = make_frontloaded_mask(
                epoch_lengths,
                phase_fractions=fractions,  # type: ignore[arg-type]
                phase_epochs=int(contract["fixed_frontloaded_phase_epochs"]),
            )
            construction = "fixed_frontloaded_predefined_fractions"
            matching_scope = "approximately_budget_comparable_not_donor_matched"
        else:
            schedule = make_uniform_third_mask(epoch_lengths)
            construction = "fixed_uniform_one_third"
            matching_scope = "fixed_uniform_not_donor_matched"
        details = schedule_mask_diagnostics(schedule, offsets)
        details.update(matching_scope=matching_scope, source_dependent=False)
        metadata = MaskMetadata(
            role=MaskRole.SCHEDULE,
            length=len(schedule),
            epoch_offsets=offsets,
            sha256=mask_identity_hash(schedule, offsets),
            construction=construction,
            construction_version="v4_m5",
            construction_details=details,
        )
        return schedule, metadata
    if method not in REPLAY_METHODS and method not in SOURCE_DERIVED_SCHEDULE_METHODS:
        if mask_path is not None or mask_manifest_path is not None:
            raise ValueError("external masks are accepted only by source-derived methods")
        return None
    if mask_path is None or mask_manifest_path is None:
        raise ValueError("source-derived execution requires mask and manifest paths")
    source = (
        config.values["scientific"]["replay_source"]
        if method in REPLAY_METHODS
        else config.values["scientific"]["schedule_source"]
    )
    donor_mask, donor_metadata = load_mask(
        mask_path,
        mask_manifest_path,
        expected_sha256=(
            str(identity["mask_sha256"])
            if method in REPLAY_METHODS else str(source["mask_sha256"])
        ),
        expected_role=MaskRole.SCHEDULE if method in SOURCE_DERIVED_SCHEDULE_METHODS else MaskRole(identity["mask_role"]),
        expected_epoch_offsets=offsets,
        expected_source=source,
    )
    if method in REPLAY_METHODS:
        return donor_mask, donor_metadata

    source_diagnostics = schedule_mask_diagnostics(donor_mask, offsets)
    gap_details = None
    if method == MethodId.RELATIVE_PERIODIC_TOTAL:
        schedule = make_periodic_mask(len(donor_mask), sum(donor_mask))
        construction = "relative_total_exact_budget_periodic"
        matching_scope = "total_exact_budget_only_not_temporal_matching"
    elif method == MethodId.RELATIVE_PERIODIC_EPOCH:
        schedule = make_epoch_matched_mask(donor_mask, offsets)
        construction = "relative_per_epoch_exact_budget_periodic"
        matching_scope = "per_epoch_exact_budget_not_burst_or_phase_matching"
    else:
        schedule, gap_details = permute_epoch_gaps(
            donor_mask,
            offsets,
            classifier_seed=int(identity["classifier_seed"]),
            replicate=int(identity["mask_replicate"]),
        )
        construction = "relative_per_epoch_internal_gap_permutation"
        matching_scope = "epoch_budget_and_gap_multiset_not_complete_temporal_matching"
    schedule_hash = mask_identity_hash(schedule, offsets)
    if schedule_hash != identity["mask_sha256"]:
        raise ValueError("constructed schedule hash does not match resolved scientific identity")
    details = {
        "matching_scope": matching_scope,
        "source_dependent": True,
        "source_mask_sha256": donor_metadata.sha256,
        "source_diagnostics": source_diagnostics,
        "constructed_diagnostics": schedule_mask_diagnostics(schedule, offsets),
        "gap_permutation": gap_details,
    }
    metadata = MaskMetadata(
        role=MaskRole.SCHEDULE,
        length=len(schedule),
        epoch_offsets=offsets,
        sha256=schedule_hash,
        construction=construction,
        construction_version="v4_m5",
        source_run_id=donor_metadata.source_run_id,
        source_checkpoint_sha256=donor_metadata.source_checkpoint_sha256,
        source_seed=donor_metadata.source_seed,
        source_method=donor_metadata.source_method,
        source_scientific_fingerprint=donor_metadata.source_scientific_fingerprint,
        source_timing_ledger=donor_metadata.source_timing_ledger,
        construction_details=details,
    )
    return schedule, metadata


def _initialize_run(
    config: RunConfig,
    run_directory: Path,
    device: torch.device,
    artifacts: Mapping[str, str],
    data_identities: Mapping[str, str],
    calibration: Mapping[str, Any] | None,
    replay_inputs: tuple[list[bool], MaskMetadata] | None,
    persistence_directory: Path | None,
) -> dict[str, Any]:
    """Create fresh deterministic modules, state, identities, and manifest.

    Inputs are the resolved run contract, fresh local destination, device,
    immutable artifact/data identities, optional calibration/replay data, and an
    optional verified-copy destination. The returned mutable context owns all
    production state. Files are created locally; no training or remote network
    connection occurs here, and invalid identities fail before model updates.
    """

    validate_config(config)
    validate_output_root(run_directory, PROJECT_ROOT)
    _validate_data_identities(data_identities)
    if run_directory.exists() and any(run_directory.iterdir()):
        raise FileExistsError("fresh run directory must be absent or empty")
    run_directory.mkdir(parents=True, exist_ok=True)
    configure_determinism(config, device)
    setup_started = time.perf_counter()
    seed_initialization(config)
    method = MethodId(config.values["scientific"]["identity"]["method"])
    include_predictor = _method_uses_predictor(config)
    model, predictor = build_models(config, device, include_predictor=include_predictor)
    optimizers = build_optimizers(model, predictor, config)
    method = MethodId(config.values["scientific"]["identity"]["method"])
    controller = (
        None
        if calibration is None
        and method in {MethodId.RELATIVE_FEEDBACK, MethodId.ABSOLUTE_FEEDBACK}
        else _make_controller(config, calibration)
    )
    state = RunState()
    replay_metadata = None
    if replay_inputs is not None:
        replay_mask, metadata = replay_inputs
        state.replay_schedule_mask = list(replay_mask)
        state.replay_epoch_offsets = list(metadata.epoch_offsets)
        replay_metadata = _mask_metadata_dict(metadata)
    ledger = TimingLedger()
    for component in config.values["scientific"]["timing"]["ledger_components"]:
        ledger.components[str(component)] = 0.0
    environment = environment_metadata(device)
    resolved_artifacts = _resolved_artifacts(artifacts)
    scientific_hash = fingerprint(config, {**resolved_artifacts, **data_identities})
    execution_hash = execution_fingerprint(config, environment)
    ledger.add("setup", time.perf_counter() - setup_started)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": config.values["output"]["run_id"],
        "scientific_fingerprint": scientific_hash,
        "execution_fingerprint": execution_hash,
        "resolved_config": config.to_dict(),
        "artifacts": resolved_artifacts,
        "data_identities": dict(data_identities),
        "environment": environment,
        "resumed": False,
        "replay_provenance": replay_metadata,
        "verified_checkpoint_persistence": {
            "enabled": persistence_directory is not None,
            "destination": None if persistence_directory is None else str(persistence_directory),
            "copy_policy": "each_committed_epoch_checkpoint_marker_published_last",
        },
    }
    atomic_write_json(run_directory / "manifest.json", manifest)
    (run_directory / "errors.jsonl").write_text("", encoding="utf-8")
    return {
        "model": model,
        "predictor": predictor,
        "optimizers": optimizers,
        "controller": controller,
        "state": state,
        "ledger": ledger,
        "environment": environment,
        "artifacts": resolved_artifacts,
        "scientific_fingerprint": scientific_hash,
        "execution_fingerprint": execution_hash,
        "replay_mask_metadata": replay_metadata,
        "calibration": None if calibration is None else dict(calibration),
        "persistence_directory": persistence_directory,
    }


def _commit_epoch(
    context: dict[str, Any],
    config: RunConfig,
    run_directory: Path,
    epoch_result: Mapping[str, Any],
    validation_result: Mapping[str, Any],
    validation_seconds: float,
) -> Path:
    """Commit one validated epoch checkpoint and reconcile all derived outputs."""

    state: RunState = context["state"]
    ledger: TimingLedger = context["ledger"]
    state.checkpoint_sequence += 1
    state.completed_epoch = int(epoch_result["global_epoch"])
    state.next_epoch = state.completed_epoch + 1
    total_epochs = int(config.values["scientific"]["training"]["total_epochs"])
    state.completed = state.next_epoch >= total_epochs
    identity = config.values["scientific"]["identity"]
    record = {
        "schema_version": SCHEMA_VERSION,
        "run_id": config.values["output"]["run_id"],
        "seed": int(identity["classifier_seed"]),
        "method": identity["method"],
        **dict(epoch_result),
        "checkpoint_sequence": state.checkpoint_sequence,
        "checkpoint_sha256": "",
        "validation_accuracy": float(validation_result["accuracy"]),
        "validation_ce": float(validation_result["ce"]),
        "validation_seconds": validation_seconds,
        # This endpoint precedes its own checkpoint write, so time-to-target
        # never includes future persistence work. Prior committed I/O is present.
        "observation_time_seconds": ledger.totals()["workflow_seconds"],
    }
    state.epoch_records.append(record)
    state.data_order_hashes.append(
        {
            "global_epoch": state.completed_epoch,
            "sha256": epoch_result["data_order_sha256"],
        }
    )
    for target in config.values["scientific"]["evaluation"]["targets"]:
        key = str(target)
        if key not in state.target_attainment and float(validation_result["accuracy"]) >= float(target):
            state.target_attainment[key] = {
                "reached": True,
                "global_epoch": state.completed_epoch,
                "checkpoint_sequence": state.checkpoint_sequence,
            }
    io_started = time.perf_counter()
    payload = _checkpoint_payload(
        config,
        context["model"],
        context["predictor"],
        context["optimizers"],
        state,
        ledger,
        context["controller"],
        context["environment"],
        context["scientific_fingerprint"],
        context["execution_fingerprint"],
        context["artifacts"],
        context["data_identities"],
        config.values["output"]["run_id"],
        context.get("replay_mask_metadata"),
    )
    checkpoint_path = run_directory / "checkpoints" / f"checkpoint_{state.checkpoint_sequence:06d}.pt"
    journal_path = run_directory / "checkpoint_journal.json"
    commit = save_checkpoint(checkpoint_path, payload, journal_path=journal_path)
    state.epoch_records[-1]["checkpoint_sha256"] = commit["sha256"]
    output_payload = _checkpoint_payload(
        config,
        context["model"],
        context["predictor"],
        context["optimizers"],
        state,
        ledger,
        context["controller"],
        context["environment"],
        context["scientific_fingerprint"],
        context["execution_fingerprint"],
        context["artifacts"],
        context["data_identities"],
        config.values["output"]["run_id"],
        context.get("replay_mask_metadata"),
    )
    output_payload["resume_metadata"] = {
        "checkpoint_sha256": commit["sha256"],
        "timing_incomplete": False,
    }
    reconcile_outputs(run_directory, output_payload)
    persistence_copy = None
    if context.get("persistence_directory") is not None:
        # Copy only the immutable committed checkpoint. The helper verifies its
        # bytes and publishes the destination marker last, so partial copies are
        # never eligible for recovery.
        persistence_copy = copy_checkpoint_verified(
            checkpoint_path,
            Path(context["persistence_directory"]) / "checkpoints",
        )
    if state.completed and context["predictor"] is not None:
        predictor_path = run_directory / "final_predictor.pt"
        predictor_temporary = run_directory / "final_predictor.pt.tmp"
        with predictor_temporary.open("wb") as handle:
            torch.save(_cpu_tree(context["predictor"].state_dict()), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(predictor_temporary, predictor_path)
    total_io_seconds = time.perf_counter() - io_started
    ledger.add("checkpoint_log_io", total_io_seconds)
    ledger.checkpoint_io.append(
        {
            "checkpoint_sequence": state.checkpoint_sequence,
            "sha256": commit["sha256"],
            "io_seconds": total_io_seconds,
            "verified_persistence_copy": None if persistence_copy is None else str(persistence_copy),
        }
    )
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    for journal_record in journal["records"]:
        if int(journal_record["checkpoint_sequence"]) == state.checkpoint_sequence:
            journal_record["io_seconds"] = total_io_seconds
    atomic_write_json(journal_path, journal)
    atomic_write_json(run_directory / "timing.json", ledger.to_dict())
    if context.get("persistence_directory") is not None:
        backup_resume_snapshot(
            run_directory,
            context["persistence_directory"],
            checkpoint_path,
        )
    return checkpoint_path


def _continue_run(
    context: dict[str, Any],
    config: RunConfig,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    run_directory: Path,
    device: torch.device,
    *,
    stop_after_global_epoch: int | None,
) -> dict[str, Any]:
    """Continue from the next committed boundary through the requested toy horizon."""

    state: RunState = context["state"]
    ledger: TimingLedger = context["ledger"]
    total_epochs = int(config.values["scientific"]["training"]["total_epochs"])
    last_checkpoint: Path | None = None
    for global_epoch in range(state.next_epoch, total_epochs):
        warmup_epochs = int(config.values["scientific"]["training"]["warmup_epochs"])
        method = MethodId(config.values["scientific"]["identity"]["method"])
        if (
            global_epoch == warmup_epochs
            and method in {MethodId.RELATIVE_FEEDBACK, MethodId.ABSOLUTE_FEEDBACK}
            and context["controller"] is None
        ):
            if context["predictor"] is None or state.scale is None:
                raise ValueError("feedback calibration requires warm-up predictor and fixed scale")
            calibration_timer = PhaseTimer(ledger, device)
            calibration_timer.start("calibration")
            calibration = calibrate_controller(
                context["model"], context["predictor"], train_dataset,
                state.scale, config, device,
            )
            calibration_seconds = calibration_timer.stop()
            calibration["seconds"] = calibration_seconds
            context["calibration"] = calibration
            context["controller"] = _make_controller(config, calibration)
            atomic_write_json(run_directory / "calibration.json", calibration)
        epoch_result = run_epoch(
            context["model"],
            context["predictor"],
            context["optimizers"],
            train_dataset,
            config,
            device,
            state,
            ledger,
            global_epoch=global_epoch,
            controller=context["controller"],
            fit_predictor_warmup=context["predictor"] is not None,
        )
        component = "warmup_validation" if epoch_result["stage"] == RunStage.WARMUP.value else "postwarm_validation"
        timer = PhaseTimer(ledger, device)
        timer.start(component)
        validation = evaluate(context["model"], validation_dataset, config, device)
        validation_seconds = timer.stop()
        last_checkpoint = _commit_epoch(
            context, config, run_directory, epoch_result, validation, validation_seconds
        )
        if stop_after_global_epoch is not None and global_epoch >= stop_after_global_epoch:
            break
    context["last_checkpoint"] = last_checkpoint
    return context


def run_training(
    config: RunConfig,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    run_directory: str | Path,
    device: torch.device | str,
    *,
    artifacts: Mapping[str, str],
    data_identities: Mapping[str, str],
    calibration: Mapping[str, Any] | None = None,
    mask_path: str | Path | None = None,
    mask_manifest_path: str | Path | None = None,
    stop_after_global_epoch: int | None = None,
    persistence_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Start a fresh deterministic tiny run and checkpoint every completed epoch.

    Inputs include map-style datasets, immutable source/split identities, optional
    replay/calibration artifacts, an optional inclusive stop epoch, and an
    optional verified-copy directory. The returned context contains final model,
    optimizer, controller, counters, and timing state. Local outputs and optional
    checksum-verified checkpoint copies are created; this programmatic engineering
    runner is not itself authorization to load CIFAR or run a study.
    """

    validate_config(config)
    replay_inputs = _prepare_schedule_inputs(config, train_dataset, mask_path, mask_manifest_path)
    selected = torch.device(device)
    run_path = Path(run_directory).resolve()
    persistence_path = None if persistence_directory is None else Path(persistence_directory).resolve()
    context = _initialize_run(
        config, run_path, selected, artifacts, data_identities, calibration,
        replay_inputs, persistence_path,
    )
    context["data_identities"] = dict(data_identities)
    return _continue_run(
        context,
        config,
        train_dataset,
        validation_dataset,
        run_path,
        selected,
        stop_after_global_epoch=stop_after_global_epoch,
    )


def resume_run(
    config: RunConfig,
    train_dataset: Dataset,
    validation_dataset: Dataset,
    run_directory: str | Path,
    device: torch.device | str,
    *,
    artifacts: Mapping[str, str],
    data_identities: Mapping[str, str],
    calibration: Mapping[str, Any] | None = None,
    mask_path: str | Path | None = None,
    mask_manifest_path: str | Path | None = None,
    checkpoint_path: str | Path | None = None,
    stop_after_global_epoch: int | None = None,
    persistence_directory: str | Path | None = None,
) -> dict[str, Any]:
    """Restore a committed epoch boundary and continue without duplicating rows.

    Inputs identify the same datasets, identities, optional replay artifacts,
    checkpoint/stop boundary, and optional verified-copy directory as the fresh
    run. The returned context contains restored and continued production state.
    Scientific/execution/source/data/mask identities must match before modules or
    optimizers are mutated; timing gaps are flagged, outputs are reconciled, and
    an uncommitted interrupted epoch is intentionally repeated in full.
    """

    validate_config(config)
    _validate_data_identities(data_identities)
    replay_inputs = _prepare_schedule_inputs(config, train_dataset, mask_path, mask_manifest_path)
    selected = torch.device(device)
    configure_determinism(config, selected)
    run_path = Path(run_directory).resolve()
    checkpoint = Path(checkpoint_path) if checkpoint_path is not None else latest_committed_checkpoint(run_path / "checkpoints")
    environment = environment_metadata(selected)
    resolved_artifacts = _resolved_artifacts(artifacts)
    scientific_hash = fingerprint(config, {**resolved_artifacts, **data_identities})
    execution_hash = execution_fingerprint(config, environment)
    expected_schedule_hash = (
        replay_inputs[1].sha256
        if replay_inputs is not None else config.values["scientific"]["identity"].get("mask_sha256")
    )
    expected_source_mask_hash = config.values["scientific"]["identity"].get("mask_sha256")
    if replay_inputs is not None and isinstance(replay_inputs[1].construction_details, Mapping):
        expected_source_mask_hash = replay_inputs[1].construction_details.get(
            "source_mask_sha256", expected_source_mask_hash
        )
    expected = {
        "scientific_fingerprint": scientific_hash,
        "execution_fingerprint": execution_hash,
        "run_id": config.values["output"]["run_id"],
        "seed": int(config.values["scientific"]["identity"]["classifier_seed"]),
        "method": config.values["scientific"]["identity"]["method"],
        "normalization": config.values["scientific"]["identity"]["normalization"],
        "artifacts": resolved_artifacts,
        "data_identities": dict(data_identities),
        "source_mask_sha256": expected_source_mask_hash,
        "replay_schedule_sha256": expected_schedule_hash,
    }
    payload = load_checkpoint(
        checkpoint,
        expected=expected,
        journal_path=run_path / "checkpoint_journal.json",
    )
    method = MethodId(config.values["scientific"]["identity"]["method"])
    include_predictor = _method_uses_predictor(config)
    model, predictor = build_models(config, selected, include_predictor=include_predictor)
    model.load_state_dict(payload["model"])
    if predictor is not None:
        if payload["predictor"] is None:
            raise ValueError("predictor-bearing run has no predictor checkpoint state")
        predictor.load_state_dict(payload["predictor"])
    optimizers = build_optimizers(model, predictor, config)
    for name, optimizer in optimizers.items():
        saved = payload["optimizers"][name]
        if optimizer is None:
            if saved is not None:
                raise ValueError(f"unexpected saved optimizer state for {name}")
        else:
            optimizer.load_state_dict(saved)
            _restore_optimizer_devices(optimizer)
    controller = _controller_from_checkpoint(config, payload["controller"])
    method = MethodId(config.values["scientific"]["identity"]["method"])
    warmup_epochs = int(config.values["scientific"]["training"]["warmup_epochs"])
    if (
        method in {MethodId.RELATIVE_FEEDBACK, MethodId.ABSOLUTE_FEEDBACK}
        and int(payload["next_epoch"]) > warmup_epochs
        and controller is None
    ):
        raise ValueError("post-warm feedback checkpoint lacks controller state")
    state = run_state_from_dict(payload["run_state"])

    # The checkpoint cannot contain its own SHA-256 because that hash
    # is only known after serialization. Restore it from the verified
    # checkpoint commit metadata before continuing training.
    current_sequence = int(payload["checkpoint_sequence"])
    current_hash = payload["resume_metadata"]["checkpoint_sha256"]

    matching_records = [
        row for row in state.epoch_records
        if int(row["checkpoint_sequence"]) == current_sequence
    ]
    if len(matching_records) != 1:
        raise ValueError(
            "Expected exactly one epoch record for resumed checkpoint"
        )

    recorded_hash = matching_records[0].get("checkpoint_sha256", "")
    if recorded_hash not in ("", current_hash):
        raise ValueError(
            "Epoch record checkpoint hash conflicts with verified checkpoint"
        )

    matching_records[0]["checkpoint_sha256"] = current_hash

    # An older missing hash must never be silently accepted.
    for row in state.epoch_records:
        digest = row.get("checkpoint_sha256", "")
        if not isinstance(digest, str) or len(digest) != 64:
            raise ValueError(
                "Earlier epoch has missing checkpoint identity"
            )
        try:
            int(digest, 16)
        except ValueError as exc:
            raise ValueError(
                "Earlier epoch has invalid checkpoint identity"
            ) from exc

    if state.pending_quota != int(payload["pending_quota"]):
        raise ValueError("pending quota mismatch inside checkpoint")
    if _mask_prefix_hash(state.executed_exact_mask) != payload["executed_mask_sha256"]:
        raise ValueError("executed mask content mismatch")
    if _mask_prefix_hash(state.executed_head_mask) != payload.get("executed_head_mask_sha256"):
        raise ValueError("executed head-update mask content mismatch")
    replay_metadata = None
    if replay_inputs is not None:
        replay_mask, metadata = replay_inputs
        replay_metadata = _mask_metadata_dict(metadata)
        if state.replay_schedule_mask != replay_mask:
            raise ValueError("checkpoint replay schedule differs from verified source mask")
        if state.replay_epoch_offsets != list(metadata.epoch_offsets):
            raise ValueError("checkpoint replay epoch boundaries differ from verified source mask")
        if payload.get("replay_mask_metadata") != replay_metadata:
            raise ValueError("checkpoint replay provenance differs from verified source mask")
        if state.mask_cursor != state.postwarm_step or state.mask_cursor > len(replay_mask):
            raise ValueError("checkpoint replay cursor is inconsistent")
    ledger = TimingLedger.from_dict(payload["timing_ledger"])
    timing_record = payload["resume_metadata"].get("checkpoint_io_record")
    if timing_record is None:
        ledger.incomplete.append(f"missing checkpoint I/O timing for sequence {state.checkpoint_sequence}")
    elif not any(
        int(item["checkpoint_sequence"]) == state.checkpoint_sequence
        for item in ledger.checkpoint_io
    ):
        ledger.add("checkpoint_log_io", float(timing_record["io_seconds"]))
        ledger.checkpoint_io.append(dict(timing_record))
    restore_rng_state(payload["rng_state"])
    state.interruptions.append(
        {
            "resume_checkpoint_sequence": state.checkpoint_sequence,
            "restart_from_global_epoch": state.next_epoch,
            "mid_epoch_resume": False,
        }
    )
    context = {
        "model": model,
        "predictor": predictor,
        "optimizers": optimizers,
        "controller": controller,
        "state": state,
        "ledger": ledger,
        "environment": environment,
        "artifacts": resolved_artifacts,
        "data_identities": dict(data_identities),
        "scientific_fingerprint": scientific_hash,
        "execution_fingerprint": execution_hash,
        "replay_mask_metadata": replay_metadata,
        "calibration": None if calibration is None else dict(calibration),
        "persistence_directory": (
            None if persistence_directory is None else Path(persistence_directory).resolve()
        ),
    }
    manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
    manifest["resumed"] = True
    manifest["resume_checkpoint_sequence"] = state.checkpoint_sequence
    if persistence_directory is not None:
        manifest["verified_checkpoint_persistence"] = {
            "enabled": True,
            "destination": str(Path(persistence_directory).resolve()),
            "copy_policy": "each_committed_epoch_checkpoint_marker_published_last",
        }
    atomic_write_json(run_path / "manifest.json", manifest)
    reconcile_outputs(run_path, payload)
    return _continue_run(
        context,
        config,
        train_dataset,
        validation_dataset,
        run_path,
        selected,
        stop_after_global_epoch=stop_after_global_epoch,
    )


def _state_trees_equal(left: Any, right: Any) -> bool:
    """Compare nested tensor state exactly without modifying either input.

    Inputs may contain tensors, mappings, sequences, or scalar values. The
    boolean output requires bitwise tensor equality and identical structure;
    live computation graphs and RNG are neither traversed nor changed.
    """

    if isinstance(left, Tensor):
        return isinstance(right, Tensor) and torch.equal(left, right)
    if isinstance(left, Mapping):
        return isinstance(right, Mapping) and left.keys() == right.keys() and all(
            _state_trees_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            _state_trees_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def check_replay_equality(
    source_context: Mapping[str, Any], replay_context: Mapping[str, Any]
) -> dict[str, bool]:
    """Verify that learned replay exactly reproduces its relative-feedback donor.

    Inputs are completed runner contexts. The output reports exact equality of
    classifier, predictor, optimizer, prescribed/actual masks, and data-order
    hashes. Any false field raises ``ValueError`` so a source curve cannot stand
    in for replay accidentally. Contexts, modules, and optimizer state are read only.
    """

    source_state: RunState = source_context["state"]
    replay_state: RunState = replay_context["state"]
    report = {
        "model": _state_trees_equal(
            source_context["model"].state_dict(), replay_context["model"].state_dict()
        ),
        "predictor": _state_trees_equal(
            source_context["predictor"].state_dict(), replay_context["predictor"].state_dict()
        ),
        "optimizers": _state_trees_equal(
            {key: value.state_dict() if value is not None else None for key, value in source_context["optimizers"].items()},
            {key: value.state_dict() if value is not None else None for key, value in replay_context["optimizers"].items()},
        ),
        "source_mask_replayed": source_state.executed_exact_mask
        == replay_state.replay_schedule_mask,
        "actual_trunk_exact_mask": source_state.executed_exact_mask
        == replay_state.executed_exact_mask,
        "head_update_mask": source_state.executed_head_mask
        == replay_state.executed_head_mask,
        "data_order": source_state.data_order_hashes == replay_state.data_order_hashes,
    }
    failed = [name for name, equal in report.items() if not equal]
    if failed:
        raise ValueError(f"learned replay differs from donor: {failed}")
    return report


def _synthetic_fixture(seed: int = 991) -> tuple[TensorDataset, TensorDataset, dict[str, str]]:
    """Create tiny deterministic train/validation tensors and content identities."""

    generator = torch.Generator().manual_seed(seed)
    inputs = torch.randn(23, 3, 32, 32, generator=generator)
    labels = torch.randint(0, 10, (23,), generator=generator)
    train = TensorDataset(inputs[:17], labels[:17])
    validation = TensorDataset(inputs[17:], labels[17:])
    identities = {
        "training_indices_sha256": hashlib.sha256(bytes(range(17))).hexdigest(),
        "validation_indices_sha256": hashlib.sha256(bytes(range(17, 23))).hexdigest(),
    }
    return train, validation, identities


def _smoke_config(output_root: str, run_id: str) -> RunConfig:
    """Return a tiny scheduled-run config derived explicitly from reference values."""

    values = reference_config().to_dict()
    identity = values["scientific"]["identity"]
    identity.update(
        method=MethodId.SCHEDULED.value,
        algorithm_version="v3",
        normalization=NormalizationMode.FROZEN_HEAD_POSTWARM.value,
        synthetic_behavior=SyntheticBehavior.LEARNED.value,
        mask_role=MaskRole.NONE.value,
        mask_sha256=None,
        retrospective=False,
        deployment_eligible=True,
    )
    values["scientific"]["model"].update(width=4, blocks_per_stage=1, boundary_shape=[None, 8, 16, 16])
    values["scientific"]["predictor"].update(input_channels=8, bottleneck_width=4)
    values["scientific"]["training"].update(
        warmup_epochs=1,
        postwarm_epochs=1,
        total_epochs=2,
        batch_size=8,
        expected_training_batches_per_epoch=3,
        expected_final_batch_size=1,
    )
    values["execution"].update(workers=0, pin_memory=False)
    values["output"].update(root=output_root, study_id="m2_engineering", run_id=run_id)
    return RunConfig(values)


def _diagnostic_fixture_state(
    output_root: Path, device: torch.device
) -> tuple[RunConfig, SplitResNet, GradientPredictor, dict[str, torch.optim.Optimizer | None], Tensor, Tensor, Tensor, Tensor, float, HistoricalFeedbackController]:
    """Build a bounded synthetic post-warm-up state for M3 CLI engineering checks.

    Inputs are a local output path and selected device. The returned config,
    modules, optimizers, two fixed batches, scale, and controller are freshly
    initialized and receive one exact warm-up update to populate optimizer state.
    Global RNG and these new objects change; no files, datasets, or production
    experiment state are read or written.
    """

    config = _smoke_config(str(output_root), output_root.name)
    configure_determinism(config, device)
    seed_initialization(config)
    model, predictor = build_models(config, device)
    if predictor is None:
        raise RuntimeError("synthetic diagnostic configuration requires a predictor")
    optimizers = build_optimizers(model, predictor, config)
    train, _, _ = _synthetic_fixture()
    train_inputs, train_labels = train.tensors
    # The engineering fixture explicitly uses smaller batches, but retains the
    # protocol's private probe-selection seed and never consumes augmentation RNG.
    probe_ids = fixed_probe_indices(len(train), batch_size=4, batches=2)
    inputs = train_inputs[probe_ids[0]].to(device)
    labels = train_labels[probe_ids[0]].to(device)
    fixed_inputs = train_inputs[probe_ids[1]].to(device)
    fixed_labels = train_labels[probe_ids[1]].to(device)
    scale = estimate_initial_scale(model, inputs, labels)
    normalization = config.scientific_dict()["identity"]["normalization"]
    set_training_mode(model, predictor, "warmup", normalization)
    exact_step(
        model, predictor, optimizers, inputs, labels, scale, config,
        fit_predictor=True,
    )
    set_training_mode(model, predictor, "postwarm", normalization)
    controller = HistoricalFeedbackController(
        MethodId.RELATIVE_FEEDBACK.value,
        config,
        {
            "relative_errors": [0.2, 0.4, 0.6, 0.8],
            "q_abs_low": 0.2,
            "q_abs_high": 0.6,
            "median_abs": 0.4,
        },
    )
    return (
        config, model, predictor, optimizers, inputs, labels, fixed_inputs,
        fixed_labels, scale, controller,
    )


def _complete_row(columns: tuple[str, ...], values: Mapping[str, Any]) -> dict[str, Any]:
    """Fill a fixed output schema with nulls and reject unknown diagnostic keys.

    Inputs are an ordered column tuple and partial values. The fresh output has
    every column exactly once; unknown keys raise before any file is written.
    """

    unknown = set(values).difference(columns)
    if unknown:
        raise ValueError(f"diagnostic row contains unknown columns: {sorted(unknown)}")
    return {column: values.get(column) for column in columns}


def _run_synthetic_diagnostics(output: Path, device: torch.device) -> None:
    """Run one bounded copied-state diagnostic pass and commit planned CSVs.

    Inputs select a local output directory and CPU/CUDA device. Three M3 CSVs
    are atomically written. Only synthetic tensors are used; the temporary local
    engineering state may change, while each probe proves copied-state isolation.
    """

    (
        config, model, predictor, optimizers, inputs, labels, fixed_inputs,
        fixed_labels, scale, _controller,
    ) = _diagnostic_fixture_state(output, device)
    run_id = output.name
    identity = diagnostic_identity_fields(run_id, "synthetic_postwarm", "fixed_batch_0")
    sample_ids = fixed_probe_indices(17, batch_size=4, batches=2)[0]
    boundary = probe_boundary_gradients(
        model, predictor, inputs, labels, scale, config, sample_ids=sample_ids
    )
    gradient_rows: list[dict[str, Any]] = []
    for measured in boundary["gradient_rows"]:
        values = {**identity, **measured, **boundary["loss_components"]}
        gradient_rows.append(_complete_row(GRADIENT_DIAGNOSTIC_COLUMNS, values))
    for measured in boundary["parameter_rows"]:
        values = {
            **identity, **measured, "aggregation": "parameter_global",
            "units": "parameter_gradient", "count": 1, "valid": True,
        }
        gradient_rows.append(_complete_row(GRADIENT_DIAGNOSTIC_COLUMNS, values))

    update_rows = [
        _complete_row(UPDATE_PROBE_COLUMNS, {**identity, **row})
        for row in probe_parameter_updates(
            model, predictor, optimizers, inputs, labels, fixed_inputs,
            fixed_labels, scale, config,
        )
    ]
    normalization_rows = [
        _complete_row(NORMALIZATION_PROBE_COLUMNS, {**identity, **row})
        for row in probe_normalization_transition(
            model, predictor, inputs, labels, scale, config,
            sample_source="deterministic_training",
        )
    ]
    write_diagnostic_outputs(
        output,
        gradient_rows=gradient_rows,
        update_rows=update_rows,
        normalization_rows=normalization_rows,
    )


def _run_synthetic_profile(output: Path, device: torch.device) -> None:
    """Run the bounded M3 microprofile and commit samples plus summaries.

    Inputs select local output and device. The profile uses synthetic fixed data,
    default settling/sample counts, and writes no scientific run artifact or
    remote state. Reported timings are labeled engineering-only.
    """

    (
        config, model, predictor, optimizers, inputs, labels, _fixed_inputs,
        _fixed_labels, scale, controller,
    ) = _diagnostic_fixture_state(output, device)
    profiled = profile_steps(
        model, predictor, optimizers, controller, inputs, labels, scale, config
    )
    sample_rows = [
        _complete_row(
            PROFILE_SAMPLE_COLUMNS,
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": output.name,
                "profile_id": "synthetic_postwarm",
                **row,
            },
        )
        for row in profiled["samples"]
    ]
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": output.name,
        "profile_id": "synthetic_postwarm",
        **profiled["summary"],
    }
    write_profile_outputs(output, sample_rows, summary)


def _scientific_artifacts() -> dict[str, str]:
    """Return governing source identities required by every scientific run.

    The output hashes v3, the research specification, and implementation plan.
    Files are read only; datasets, models, network resources, and output paths are
    untouched. Missing governing files fail before a run directory is created.
    """

    return {
        "v3_source": sha256_file(PROJECT_ROOT / "gradient_phase_d_resnet_v3.py"),
        "research_spec": sha256_file(PROJECT_ROOT / "RESEARCH_SPEC.md"),
        "implementation_plan": sha256_file(PROJECT_ROOT / "IMPLEMENTATION_PLAN.md"),
    }


def _run_scientific_command(args: argparse.Namespace, *, resume: bool) -> None:
    """Execute one explicitly configured CIFAR-10 run or epoch-boundary resume.

    CLI inputs identify a resolved config, existing local training cache, verified
    split artifact, fresh/compatible output, optional donor mask, and CUDA device.
    The official test set is never loaded and downloading is disabled. The chosen
    run function performs deterministic training and persistence; no other method,
    retry, analysis, or study expansion is launched automatically.
    """

    config = load_run_config(args.config)
    _require_scientific_authorization(
        args,
        full_pilot=(
            resume
            or args.stop_after_global_epoch is None
            or int(args.stop_after_global_epoch)
            >= int(config.values["scientific"]["training"]["warmup_epochs"])
        ),
    )
    declared = (
        Path(config.values["output"]["root"])
        / str(config.values["output"]["run_id"])
    ).resolve()
    if Path(args.out).resolve() != declared:
        raise ValueError("--out must equal the resolved config output.root/run_id")
    train, validation, identities = load_cifar10_datasets(
        config, args.data_dir, args.split_file
    )
    keyword = {
        "artifacts": _scientific_artifacts(),
        "data_identities": identities,
        "mask_path": args.mask,
        "mask_manifest_path": args.mask_manifest,
        "persistence_directory": args.persistence_dir,
    }
    if resume:
        resume_run(
            config, train, validation, args.out, args.device,
            checkpoint_path=args.checkpoint,
            stop_after_global_epoch=args.stop_after_global_epoch,
            **keyword,
        )
    else:
        run_training(
            config, train, validation, args.out, args.device,
            stop_after_global_epoch=args.stop_after_global_epoch,
            **keyword,
        )


def _run_scientific_preflight(args: argparse.Namespace) -> None:
    """Profile a verified warm-up checkpoint on bounded real-width copied state.

    Inputs identify a predictor-bearing resolved config, existing CIFAR training
    cache/split, committed warm-up checkpoint, CUDA device, and fresh output.
    Exactly the configured settling and timed batch counts are profiled on copied
    state, including plain exact, exact-plus-predictor, learned synthetic, and CPU
    controller overhead. The checkpoint and production-like loaded state are not
    mutated; no training continuation, download, retry, or analysis is launched.
    """

    _require_scientific_authorization(args, full_pilot=False)
    config = load_run_config(args.config)
    train, _validation, identities = load_cifar10_datasets(
        config, args.data_dir, args.split_file
    )
    device = torch.device(args.device)
    configure_determinism(config, device)
    environment = environment_metadata(device)
    artifacts = _resolved_artifacts(_scientific_artifacts())
    expected = {
        "scientific_fingerprint": fingerprint(config, {**artifacts, **identities}),
        "execution_fingerprint": execution_fingerprint(config, environment),
        "run_id": config.values["output"]["run_id"],
        "seed": int(config.values["scientific"]["identity"]["classifier_seed"]),
        "method": config.values["scientific"]["identity"]["method"],
        "normalization": config.values["scientific"]["identity"]["normalization"],
        "artifacts": artifacts,
        "data_identities": identities,
    }
    checkpoint = Path(args.checkpoint)
    journal = checkpoint.parent.parent / "checkpoint_journal.json"
    payload = load_checkpoint(checkpoint, expected=expected, journal_path=journal)
    model, predictor = build_models(config, device, include_predictor=True)
    if predictor is None or payload.get("predictor") is None:
        raise ValueError("scientific preflight requires a predictor-bearing checkpoint")
    model.load_state_dict(payload["model"])
    predictor.load_state_dict(payload["predictor"])
    optimizers = build_optimizers(model, predictor, config)
    for name, optimizer in optimizers.items():
        saved = payload["optimizers"].get(name)
        if optimizer is None or saved is None:
            raise ValueError("scientific preflight requires complete optimizer state")
        optimizer.load_state_dict(saved)
        _restore_optimizer_devices(optimizer)
    scale = float(payload["scale"])
    calibration = calibrate_controller(model, predictor, train, scale, config, device)
    controller = HistoricalFeedbackController(
        MethodId.RELATIVE_FEEDBACK.value, config, calibration
    )
    warmup_epochs = int(config.values["scientific"]["training"]["warmup_epochs"])
    cpu_inputs, cpu_labels, _indices = next(
        iter(epoch_loader(train, config, warmup_epochs, shuffle=True))
    )
    inputs = cpu_inputs.to(device, non_blocking=True)
    labels = cpu_labels.to(device, non_blocking=True)
    set_training_mode(
        model, predictor, "postwarm",
        config.values["scientific"]["identity"]["normalization"],
    )
    profiled = profile_steps(
        model,
        predictor,
        optimizers,
        controller,
        inputs,
        labels,
        scale,
        config,
        samples=int(config.values["scientific"]["calibration_and_profile"]["profile_timed_batches"]),
        settling=int(config.values["scientific"]["calibration_and_profile"]["profile_settling_batches"]),
    )
    output = Path(args.out)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("preflight output must be absent or empty")
    sample_rows = [
        _complete_row(
            PROFILE_SAMPLE_COLUMNS,
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": config.values["output"]["run_id"],
                "seed": int(config.values["scientific"]["identity"]["classifier_seed"]),
                "method": config.values["scientific"]["identity"]["method"],
                "checkpoint_sha256": payload["resume_metadata"]["checkpoint_sha256"],
                "profile_id": "scientific_preflight_warmup_boundary",
                **row,
            },
        )
        for row in profiled["samples"]
    ]
    checkpoint_size = checkpoint.stat().st_size
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_id": config.values["output"]["run_id"],
        "profile_id": "scientific_preflight_warmup_boundary",
        "scientific_fingerprint": payload["scientific_fingerprint"],
        "checkpoint_sha256": payload["resume_metadata"]["checkpoint_sha256"],
        "checkpoint_size_bytes": checkpoint_size,
        "environment": environment,
        "engineering_preflight_not_scientific_result": True,
        **profiled["summary"],
    }
    write_profile_outputs(output, sample_rows, summary)


def _require_scientific_authorization(
    args: argparse.Namespace, *, full_pilot: bool
) -> None:
    """Stop real-data or full-pilot execution unless explicit CLI gates are set.

    The input is a parsed command namespace and whether the requested action can
    enter post-warm-up pilot execution. Nothing is returned or mutated. Missing
    ``--authorize-real-data`` always raises before dataset access; a full pilot
    additionally requires ``--authorize-full-pilot``. Both flags default false.
    """

    if not bool(getattr(args, "authorize_real_data", False)):
        raise PermissionError(
            "real-data execution is disabled; rerun only after approval with "
            "--authorize-real-data"
        )
    if full_pilot and not bool(getattr(args, "authorize_full_pilot", False)):
        raise PermissionError(
            "full-pilot execution is disabled; rerun only after approval with "
            "--authorize-full-pilot"
        )


def parse_args(arguments: list[str] | None = None) -> argparse.Namespace:
    """Parse explicit engineering or M6 run commands without starting work."""

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="validate config and environment only")
    audit.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    smoke = subparsers.add_parser("smoke", help="run a tiny synthetic M2 workflow")
    smoke.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    smoke.add_argument("--out", required=True)
    smoke.add_argument("--stop-after-global-epoch", type=int)
    smoke.add_argument("--persistence-dir")
    resume = subparsers.add_parser(
        "resume", help="resume a synthetic workflow or one explicit scientific run"
    )
    resume.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    resume.add_argument("--out", required=True)
    resume.add_argument("--checkpoint")
    resume.add_argument("--config")
    resume.add_argument("--data-dir")
    resume.add_argument("--split-file")
    resume.add_argument("--mask")
    resume.add_argument("--mask-manifest")
    resume.add_argument("--stop-after-global-epoch", type=int)
    resume.add_argument("--persistence-dir")
    resume.add_argument("--authorize-real-data", action="store_true")
    resume.add_argument("--authorize-full-pilot", action="store_true")
    profile = subparsers.add_parser("profile", help="run the bounded synthetic M3 step profile")
    profile.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    profile.add_argument("--out", required=True)
    diagnose = subparsers.add_parser("diagnose", help="run copied-state synthetic M3 probes")
    diagnose.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    diagnose.add_argument("--out", required=True)
    prepare_data = subparsers.add_parser(
        "prepare-data", help="materialize the frozen train/validation split only"
    )
    prepare_data.add_argument("--config", required=True)
    prepare_data.add_argument("--data-dir", required=True)
    prepare_data.add_argument("--split-file", required=True)
    prepare_data.add_argument("--download", action="store_true")
    prepare_data.add_argument("--authorize-real-data", action="store_true")
    make_config = subparsers.add_parser(
        "make-config", help="write one resolved frozen scientific condition config"
    )
    make_config.add_argument("--method", choices=tuple(item.value for item in MethodId), required=True)
    make_config.add_argument("--run-id", required=True)
    make_config.add_argument("--output-root", required=True)
    make_config.add_argument("--out-config", required=True)
    make_config.add_argument("--classifier-seed", type=int, default=0)
    make_config.add_argument("--synthetic-behavior", choices=tuple(item.value for item in SyntheticBehavior))
    make_config.add_argument("--mask-replicate", type=int)
    make_config.add_argument("--donor-mask")
    make_config.add_argument("--donor-manifest")
    run = subparsers.add_parser("run", help="run exactly one resolved CIFAR-10 condition")
    run.add_argument("--device", choices=("cuda",), default="cuda")
    run.add_argument("--config", required=True)
    run.add_argument("--data-dir", required=True)
    run.add_argument("--split-file", required=True)
    run.add_argument("--out", required=True)
    run.add_argument("--mask")
    run.add_argument("--mask-manifest")
    run.add_argument("--stop-after-global-epoch", type=int)
    run.add_argument("--persistence-dir")
    run.add_argument("--authorize-real-data", action="store_true")
    run.add_argument("--authorize-full-pilot", action="store_true")
    preflight = subparsers.add_parser(
        "preflight", help="profile a committed warm-up checkpoint on bounded copied state"
    )
    preflight.add_argument("--device", choices=("cuda",), default="cuda")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--data-dir", required=True)
    preflight.add_argument("--split-file", required=True)
    preflight.add_argument("--checkpoint", required=True)
    preflight.add_argument("--out", required=True)
    preflight.add_argument("--authorize-real-data", action="store_true")
    return parser.parse_args(arguments)


def main(arguments: list[str] | None = None) -> int:
    """Execute one explicit CLI command and never cascade into another mode."""

    args = parse_args(arguments)
    if args.command == "audit":
        config = reference_config()
        configure_determinism(config, args.device)
        print(json.dumps(environment_metadata(args.device), indent=2, sort_keys=True))
        return 0
    if args.command == "prepare-data":
        _require_scientific_authorization(args, full_pilot=False)
        config = load_run_config(args.config)
        identities = prepare_cifar10_split(
            config, args.data_dir, args.split_file, download=args.download
        )
        print(json.dumps(identities, indent=2, sort_keys=True))
        return 0
    if args.command == "make-config":
        config = build_condition_config(
            args.method,
            run_id=args.run_id,
            output_root=args.output_root,
            classifier_seed=args.classifier_seed,
            synthetic_behavior=args.synthetic_behavior,
            mask_replicate=args.mask_replicate,
            donor_mask_path=args.donor_mask,
            donor_manifest_path=args.donor_manifest,
        )
        destination = Path(args.out_config)
        if destination.exists():
            raise FileExistsError("resolved config destination already exists")
        atomic_write_json(destination, config.to_dict())
        print(f"Resolved config written: {destination.resolve()}")
        return 0
    if args.command == "run":
        _run_scientific_command(args, resume=False)
        print(f"Scientific run completed: {Path(args.out).resolve()}")
        return 0
    if args.command == "preflight":
        _run_scientific_preflight(args)
        print(f"Scientific preflight completed: {Path(args.out).resolve()}")
        return 0
    output = Path(args.out).resolve()
    if args.command == "diagnose":
        _run_synthetic_diagnostics(output, torch.device(args.device))
        print(f"Synthetic diagnose completed: {output}")
        return 0
    if args.command == "profile":
        _run_synthetic_profile(output, torch.device(args.device))
        print(f"Synthetic profile completed: {output}")
        return 0
    if args.command == "resume" and args.config is not None:
        required = (args.data_dir, args.split_file)
        if any(value is None for value in required):
            raise ValueError("scientific resume requires --data-dir and --split-file")
        _run_scientific_command(args, resume=True)
        print(f"Scientific resume completed: {output}")
        return 0
    if args.command == "resume" and any(
        value is not None
        for value in (
            args.data_dir, args.split_file, args.mask, args.mask_manifest,
        )
    ):
        raise ValueError("synthetic resume must not receive scientific data or mask arguments")
    config = _smoke_config(str(output), output.name)
    train, validation, identities = _synthetic_fixture()
    artifacts = {"v3_source": sha256_file(PROJECT_ROOT / "gradient_phase_d_resnet_v3.py")}
    if args.command == "smoke":
        run_training(
            config,
            train,
            validation,
            output,
            args.device,
            artifacts=artifacts,
            data_identities=identities,
            stop_after_global_epoch=args.stop_after_global_epoch,
            persistence_directory=args.persistence_dir,
        )
    elif args.command == "resume":
        resume_run(
            config,
            train,
            validation,
            output,
            args.device,
            artifacts=artifacts,
            data_identities=identities,
            checkpoint_path=args.checkpoint,
            persistence_directory=args.persistence_dir,
        )
    print(f"Synthetic {args.command} completed: {output}")
    return 0


__all__ = [
    "BasicBlock", "GradientPredictor", "HistoricalFeedbackController", "SplitResNet",
    "build_models", "build_optimizers", "estimate_initial_scale", "exact_step",
    "predictor_loss_components", "reference_config", "relative_error", "scheduled_exact",
    "configure_determinism", "epoch_loader", "evaluate", "resume_run", "run_epoch",
    "run_training", "set_epoch_lr", "set_training_mode", "synthetic_step",
    "build_condition_config", "calibrate_controller", "load_cifar10_datasets",
    "make_split_indices", "prepare_cifar10_split",
    "gradient_metrics", "probe_boundary_gradients", "probe_parameter_updates",
    "probe_normalization_transition", "profile_steps", "replay_slot_decision",
    "execute_slot", "check_replay_equality",
    "historical_midpoint_periodic_mask", "make_periodic_mask",
    "make_epoch_matched_mask", "make_frontloaded_mask", "make_uniform_third_mask",
    "permute_epoch_gaps", "schedule_mask_diagnostics", "schedule_slot_decision",
]


if __name__ == "__main__":
    raise SystemExit(main())
