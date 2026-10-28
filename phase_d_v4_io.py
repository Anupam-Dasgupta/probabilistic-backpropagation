"""Identity and schema helpers for the Phase D v4 implementation.

This M0 module deliberately uses only the Python standard library and does not
import the training module.  It defines stable configuration records and
fingerprints without loading data or starting scientific work.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import json
import math
import os
import platform
import random
import re
import shutil
import subprocess
import time
from importlib import metadata as importlib_metadata
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch


SCHEMA_VERSION = "phase_d_v4.m0.1"
REFERENCE_CONFIG_RELATIVE_PATH = Path("configs") / "phase_d_v4_reference.json"
REFERENCE_MANIFEST_RELATIVE_PATH = Path("configs") / "phase_d_v4_reference_manifest.json"


class MethodId(str, Enum):
    """Name every historical and planned condition without changing its meaning."""

    EXACT = "exact"
    SCHEDULED = "scheduled"
    RELATIVE_FEEDBACK = "relative_feedback"
    ABSOLUTE_FEEDBACK = "absolute_feedback"
    MATCHED_PERIODIC = "matched_periodic"
    EXACT_HEAD_MASK = "exact_head_mask"
    EXACT_STANDARD_BN = "exact_standard_bn"
    RELATIVE_REPLAY_LEARNED = "relative_replay_learned"
    RELATIVE_REPLAY_ZERO_SHADOW = "relative_replay_zero_shadow"
    RELATIVE_REPLAY_SKIP_SHADOW = "relative_replay_skip_shadow"
    RELATIVE_REPLAY_EXACT_TRUNK = "relative_replay_exact_trunk"
    RELATIVE_REPLAY_ZERO_FAST = "relative_replay_zero_fast"
    RELATIVE_REPLAY_SKIP_FAST = "relative_replay_skip_fast"
    RELATIVE_PERIODIC_TOTAL = "relative_periodic_total"
    RELATIVE_PERIODIC_EPOCH = "relative_periodic_epoch"
    RELATIVE_GAP_PERMUTATION_EPOCH = "relative_gap_permutation_epoch"
    FIXED_FRONTLOADED = "fixed_frontloaded"
    FIXED_UNIFORM_THIRD = "fixed_uniform_third"


class NormalizationMode(str, Enum):
    """Identify whether downstream BatchNorm buffers freeze after warm-up."""

    FROZEN_HEAD_POSTWARM = "frozen_head_postwarm"
    STANDARD_BN = "standard_bn"


class SyntheticBehavior(str, Enum):
    """Distinguish learned, zero-gradient, skipped, and all-exact slots."""

    EXACT = "exact"
    LEARNED = "learned"
    ZERO = "zero"
    SKIP = "skip"


class MaskRole(str, Enum):
    """Keep schedule, trunk-exact, and head-update masks scientifically distinct."""

    NONE = "none"
    SCHEDULE = "schedule"
    TRUNK_EXACT = "trunk_exact"
    HEAD_UPDATE = "head_update"


class UpdateType(str, Enum):
    """Label the one update path executed for a minibatch."""

    EXACT = "exact"
    LEARNED_SYNTHETIC = "learned_synthetic"
    ZERO_SYNTHETIC = "zero_synthetic"
    SKIP = "skip"


class RunStage(str, Enum):
    """Identify the committed training boundary represented by run state."""

    INITIALIZED = "initialized"
    WARMUP = "warmup"
    CALIBRATION = "calibration"
    POSTWARM = "postwarm"
    COMPLETE = "complete"
    FAILED = "failed"


HISTORICAL_METHODS = frozenset(
    {
        MethodId.EXACT,
        MethodId.SCHEDULED,
        MethodId.RELATIVE_FEEDBACK,
        MethodId.ABSOLUTE_FEEDBACK,
        MethodId.MATCHED_PERIODIC,
        MethodId.EXACT_HEAD_MASK,
    }
)

MASK_REQUIRED_METHODS = frozenset(
    {
        MethodId.MATCHED_PERIODIC,
        MethodId.EXACT_HEAD_MASK,
        MethodId.RELATIVE_REPLAY_LEARNED,
        MethodId.RELATIVE_REPLAY_ZERO_SHADOW,
        MethodId.RELATIVE_REPLAY_SKIP_SHADOW,
        MethodId.RELATIVE_REPLAY_EXACT_TRUNK,
        MethodId.RELATIVE_REPLAY_ZERO_FAST,
        MethodId.RELATIVE_REPLAY_SKIP_FAST,
        MethodId.RELATIVE_PERIODIC_TOTAL,
        MethodId.RELATIVE_PERIODIC_EPOCH,
        MethodId.RELATIVE_GAP_PERMUTATION_EPOCH,
    }
)

PREDICTOR_REQUIRED_METHODS = frozenset(
    {
        MethodId.SCHEDULED,
        MethodId.RELATIVE_FEEDBACK,
        MethodId.ABSOLUTE_FEEDBACK,
        MethodId.MATCHED_PERIODIC,
        MethodId.RELATIVE_REPLAY_LEARNED,
        MethodId.RELATIVE_REPLAY_ZERO_SHADOW,
        MethodId.RELATIVE_REPLAY_SKIP_SHADOW,
        MethodId.RELATIVE_REPLAY_EXACT_TRUNK,
        MethodId.RELATIVE_PERIODIC_TOTAL,
        MethodId.RELATIVE_PERIODIC_EPOCH,
        MethodId.RELATIVE_GAP_PERMUTATION_EPOCH,
    }
)

REPLAY_METHODS = frozenset(
    {
        MethodId.RELATIVE_REPLAY_LEARNED,
        MethodId.RELATIVE_REPLAY_ZERO_SHADOW,
        MethodId.RELATIVE_REPLAY_SKIP_SHADOW,
        MethodId.RELATIVE_REPLAY_EXACT_TRUNK,
        MethodId.RELATIVE_REPLAY_ZERO_FAST,
        MethodId.RELATIVE_REPLAY_SKIP_FAST,
    }
)

SOURCE_DERIVED_SCHEDULE_METHODS = frozenset(
    {
        MethodId.RELATIVE_PERIODIC_TOTAL,
        MethodId.RELATIVE_PERIODIC_EPOCH,
        MethodId.RELATIVE_GAP_PERMUTATION_EPOCH,
    }
)

FIXED_SCHEDULE_METHODS = frozenset(
    {MethodId.FIXED_FRONTLOADED, MethodId.FIXED_UNIFORM_THIRD}
)

MASK_EXECUTION_METHODS = REPLAY_METHODS | SOURCE_DERIVED_SCHEDULE_METHODS | FIXED_SCHEDULE_METHODS

METHOD_SPECS: Mapping[MethodId, Mapping[str, Any]] = {
    MethodId.EXACT: {"version": "v3", "normalization": NormalizationMode.FROZEN_HEAD_POSTWARM, "behaviors": {SyntheticBehavior.EXACT}, "mask_role": MaskRole.NONE, "mask_required": False, "retrospective": False, "deployment_eligible": True},
    MethodId.SCHEDULED: {"version": "v3", "normalization": NormalizationMode.FROZEN_HEAD_POSTWARM, "behaviors": {SyntheticBehavior.LEARNED}, "mask_role": MaskRole.NONE, "mask_required": False, "retrospective": False, "deployment_eligible": True},
    MethodId.RELATIVE_FEEDBACK: {"version": "v3", "normalization": NormalizationMode.FROZEN_HEAD_POSTWARM, "behaviors": {SyntheticBehavior.LEARNED}, "mask_role": MaskRole.NONE, "mask_required": False, "retrospective": False, "deployment_eligible": True},
    MethodId.ABSOLUTE_FEEDBACK: {"version": "v3", "normalization": NormalizationMode.FROZEN_HEAD_POSTWARM, "behaviors": {SyntheticBehavior.LEARNED}, "mask_role": MaskRole.NONE, "mask_required": False, "retrospective": False, "deployment_eligible": True},
    MethodId.MATCHED_PERIODIC: {"version": "v3", "normalization": NormalizationMode.FROZEN_HEAD_POSTWARM, "behaviors": {SyntheticBehavior.LEARNED}, "mask_role": MaskRole.SCHEDULE, "mask_required": True, "retrospective": True, "deployment_eligible": False},
    MethodId.EXACT_HEAD_MASK: {"version": "v3", "normalization": NormalizationMode.FROZEN_HEAD_POSTWARM, "behaviors": {SyntheticBehavior.EXACT}, "mask_role": MaskRole.HEAD_UPDATE, "mask_required": True, "retrospective": True, "deployment_eligible": False},
    MethodId.EXACT_STANDARD_BN: {"version": "v4", "normalization": NormalizationMode.STANDARD_BN, "behaviors": {SyntheticBehavior.EXACT}, "mask_role": MaskRole.NONE, "mask_required": False, "retrospective": False, "deployment_eligible": True},
    MethodId.RELATIVE_REPLAY_LEARNED: {"version": "v4", "normalization": NormalizationMode.FROZEN_HEAD_POSTWARM, "behaviors": {SyntheticBehavior.LEARNED}, "mask_role": MaskRole.SCHEDULE, "mask_required": True, "retrospective": True, "deployment_eligible": False},
    MethodId.RELATIVE_REPLAY_ZERO_SHADOW: {"version": "v4", "normalization": NormalizationMode.FROZEN_HEAD_POSTWARM, "behaviors": {SyntheticBehavior.ZERO}, "mask_role": MaskRole.SCHEDULE, "mask_required": True, "retrospective": True, "deployment_eligible": False},
    MethodId.RELATIVE_REPLAY_SKIP_SHADOW: {"version": "v4", "normalization": NormalizationMode.FROZEN_HEAD_POSTWARM, "behaviors": {SyntheticBehavior.SKIP}, "mask_role": MaskRole.SCHEDULE, "mask_required": True, "retrospective": True, "deployment_eligible": False},
    MethodId.RELATIVE_REPLAY_EXACT_TRUNK: {"version": "v4", "normalization": NormalizationMode.FROZEN_HEAD_POSTWARM, "behaviors": {SyntheticBehavior.EXACT}, "mask_role": MaskRole.HEAD_UPDATE, "mask_required": True, "retrospective": True, "deployment_eligible": False},
    MethodId.RELATIVE_REPLAY_ZERO_FAST: {"version": "v4", "normalization": NormalizationMode.FROZEN_HEAD_POSTWARM, "behaviors": {SyntheticBehavior.ZERO}, "mask_role": MaskRole.SCHEDULE, "mask_required": True, "retrospective": True, "deployment_eligible": False},
    MethodId.RELATIVE_REPLAY_SKIP_FAST: {"version": "v4", "normalization": NormalizationMode.FROZEN_HEAD_POSTWARM, "behaviors": {SyntheticBehavior.SKIP}, "mask_role": MaskRole.SCHEDULE, "mask_required": True, "retrospective": True, "deployment_eligible": False},
    MethodId.RELATIVE_PERIODIC_TOTAL: {"version": "v4", "normalization": NormalizationMode.FROZEN_HEAD_POSTWARM, "behaviors": {SyntheticBehavior.LEARNED}, "mask_role": MaskRole.SCHEDULE, "mask_required": True, "retrospective": True, "deployment_eligible": False},
    MethodId.RELATIVE_PERIODIC_EPOCH: {"version": "v4", "normalization": NormalizationMode.FROZEN_HEAD_POSTWARM, "behaviors": {SyntheticBehavior.LEARNED}, "mask_role": MaskRole.SCHEDULE, "mask_required": True, "retrospective": True, "deployment_eligible": False},
    MethodId.RELATIVE_GAP_PERMUTATION_EPOCH: {"version": "v4", "normalization": NormalizationMode.FROZEN_HEAD_POSTWARM, "behaviors": {SyntheticBehavior.LEARNED}, "mask_role": MaskRole.SCHEDULE, "mask_required": True, "retrospective": True, "deployment_eligible": False},
    MethodId.FIXED_FRONTLOADED: {"version": "v4", "normalization": NormalizationMode.FROZEN_HEAD_POSTWARM, "behaviors": {SyntheticBehavior.LEARNED, SyntheticBehavior.ZERO, SyntheticBehavior.SKIP}, "mask_role": MaskRole.NONE, "mask_required": False, "retrospective": False, "deployment_eligible": True},
    MethodId.FIXED_UNIFORM_THIRD: {"version": "v4", "normalization": NormalizationMode.FROZEN_HEAD_POSTWARM, "behaviors": {SyntheticBehavior.LEARNED, SyntheticBehavior.ZERO, SyntheticBehavior.SKIP}, "mask_role": MaskRole.NONE, "mask_required": False, "retrospective": False, "deployment_eligible": True},
}


@dataclass(frozen=True)
class RunConfig:
    """Hold a complete resolved run configuration as JSON-compatible values.

    ``values`` contains scientific, execution, and output sections.  The input
    is copied, the output is a deep copy, and no files or random states change.
    Invalid values are reported by :func:`validate_config`.
    """

    values: Mapping[str, Any]

    def __post_init__(self) -> None:
        """Detach configuration values from caller mutation without other effects."""

        object.__setattr__(self, "values", _json_copy(self.values))

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "RunConfig":
        """Build a detached config from a mapping and return the new record."""

        return cls(values=values)

    def to_dict(self) -> dict[str, Any]:
        """Return a deep JSON-compatible copy without exposing internal state."""

        return _json_copy(self.values)

    def scientific_dict(self) -> dict[str, Any]:
        """Return only identity-bearing protocol values, excluding execution paths."""

        return _json_copy(_required_mapping(self.values, "scientific"))


@dataclass
class RunState:
    """Track resumable counters without storing model tensors or computation graphs.

    Counts use minibatches or optimizer steps as named.  This mutable record has
    no side effects beyond caller-requested field updates.
    """

    stage: RunStage = RunStage.INITIALIZED
    completed_epoch: int = -1
    next_epoch: int = 0
    global_step: int = 0
    postwarm_step: int = 0
    pending_quota: int = 0
    exact_trunk_backwards: int = 0
    synthetic_backwards: int = 0
    zero_backwards: int = 0
    skipped_slots: int = 0
    trunk_optimizer_steps: int = 0
    head_optimizer_steps: int = 0
    predictor_fits: int = 0
    predictor_inferences: int = 0
    scale: float | None = None
    lr_global_epoch: int = -1
    mask_cursor: int = 0
    executed_exact_mask: list[bool] = field(default_factory=list)
    executed_head_mask: list[bool] = field(default_factory=list)
    replay_schedule_mask: list[bool] = field(default_factory=list)
    replay_epoch_offsets: list[int] = field(default_factory=lambda: [0])
    mask_epoch_offsets: list[int] = field(default_factory=lambda: [0])
    checkpoint_sequence: int = 0
    epoch_records: list[dict[str, Any]] = field(default_factory=list)
    controller_events: list[dict[str, Any]] = field(default_factory=list)
    data_order_hashes: list[dict[str, Any]] = field(default_factory=list)
    target_attainment: dict[str, Any] = field(default_factory=dict)
    interruptions: list[dict[str, Any]] = field(default_factory=list)
    completed: bool = False


@dataclass
class TimingLedger:
    """Accumulate non-overlapping workflow durations in seconds.

    Component values are elapsed seconds and may be updated after each phase.
    ``incomplete`` records missing journal intervals rather than inventing time.
    The record mutates only when callers add durations or flags.
    """

    components: dict[str, float] = field(default_factory=dict)
    checkpoint_io: list[dict[str, Any]] = field(default_factory=list)
    incomplete: list[str] = field(default_factory=list)
    sessions: list[dict[str, Any]] = field(default_factory=list)

    def begin_session(
        self,
        environment: Mapping[str, Any],
        execution_fingerprint_value: str,
        *,
        policy: str,
        cross_device: bool,
        checkpoint_sequence: int,
        resume_attempt_id: int | None = None,
    ) -> None:
        """Start one execution-session timing bucket without adding elapsed time.

        Inputs identify the environment, execution hash, recovery policy, whether
        the physical GPU changed, and committed checkpoint used as the boundary.
        Nothing is returned. A new mutable session record is appended; the prior
        record and all cumulative components remain unchanged. Interruption gap is
        explicitly unknown because a checkpoint contains no trustworthy wall-clock
        end time for a disconnected runtime.
        """

        self.sessions.append(
            {
                "session_index": len(self.sessions),
                "environment": _json_copy(environment),
                "execution_fingerprint": str(execution_fingerprint_value),
                "resume_policy": str(policy),
                "cross_device": bool(cross_device),
                "started_from_checkpoint_sequence": int(checkpoint_sequence),
                "resume_attempt_id": (
                    None if resume_attempt_id is None else int(resume_attempt_id)
                ),
                "interruption_gap_seconds": None,
                "components": {},
            }
        )

    def add(self, component: str, seconds: float) -> None:
        """Add a finite nonnegative duration to one named ledger component."""

        if not component:
            raise ValueError("timing component must be nonempty")
        if not isinstance(seconds, (int, float)) or not np.isfinite(seconds) or seconds < 0:
            raise ValueError("timing duration must be finite and nonnegative")
        self.components[component] = self.components.get(component, 0.0) + float(seconds)
        if self.sessions:
            values = self.sessions[-1]["components"]
            values[component] = values.get(component, 0.0) + float(seconds)

    def totals(self) -> dict[str, float]:
        """Return planned learning, evaluated, workflow, and post-warm totals."""

        get = lambda name: self.components.get(name, 0.0)
        learning = (
            get("warmup_training")
            + get("scale_estimation")
            + get("calibration")
            + get("postwarm_training")
        )
        evaluated = learning + get("warmup_validation") + get("postwarm_validation")
        workflow = (
            get("setup")
            + evaluated
            + get("checkpoint_log_io")
            + get("finalization")
            + get("diagnostics")
        )
        return {
            "learning_seconds": learning,
            "evaluated_seconds": evaluated,
            "workflow_seconds": workflow,
            "postwarm_seconds": get("postwarm_training") + get("postwarm_validation"),
            "postwarm_training_seconds": get("postwarm_training"),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible timing data without changing the ledger."""

        return {
            "components": copy.deepcopy(self.components),
            "checkpoint_io": copy.deepcopy(self.checkpoint_io),
            "incomplete": list(self.incomplete),
            "sessions": copy.deepcopy(self.sessions),
            "totals": self.totals(),
        }

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "TimingLedger":
        """Restore timing data from primitive checkpoint values and return it."""

        ledger = cls(
            components={key: float(value) for key, value in values.get("components", {}).items()},
            checkpoint_io=copy.deepcopy(values.get("checkpoint_io", [])),
            incomplete=list(values.get("incomplete", [])),
            sessions=copy.deepcopy(values.get("sessions", [])),
        )
        for component, seconds in ledger.components.items():
            if not component or seconds < 0 or not np.isfinite(seconds):
                raise ValueError("invalid timing ledger component")
        for index, session in enumerate(ledger.sessions):
            if int(session.get("session_index", -1)) != index:
                raise ValueError("invalid timing session ordering")
            for seconds in session.get("components", {}).values():
                if float(seconds) < 0 or not np.isfinite(float(seconds)):
                    raise ValueError("invalid timing session component")
        return ledger


@dataclass(frozen=True)
class StepResult:
    """Return detached step metrics and explicit update counts to the runner.

    Metrics must be scalar JSON values in their documented units.  Counts are
    nonnegative integers.  Construction raises ``ValueError`` for retained or
    otherwise non-serializable values and has no optimizer or RNG side effects.
    """

    update_type: UpdateType
    exact_trunk_backwards: int = 0
    synthetic_backwards: int = 0
    zero_backwards: int = 0
    skipped_slots: int = 0
    trunk_optimizer_steps: int = 0
    head_optimizer_steps: int = 0
    predictor_fits: int = 0
    predictor_inferences: int = 0
    metrics: Mapping[str, float | int | bool | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject negative counts and non-scalar metrics without changing state."""

        count_names = (
            "exact_trunk_backwards",
            "synthetic_backwards",
            "zero_backwards",
            "skipped_slots",
            "trunk_optimizer_steps",
            "head_optimizer_steps",
            "predictor_fits",
            "predictor_inferences",
        )
        for name in count_names:
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        clean_metrics = dict(self.metrics)
        if any(not isinstance(value, (float, int, bool, type(None))) for value in clean_metrics.values()):
            raise ValueError("StepResult metrics must be detached scalar JSON values")
        object.__setattr__(self, "metrics", clean_metrics)


@dataclass(frozen=True)
class MaskMetadata:
    """Describe immutable mask bytes and their source without conflating mask roles.

    Lengths and offsets count post-warm-up minibatch slots.  The record does not
    read or write mask data; malformed lengths, hashes, or offsets raise errors.
    """

    role: MaskRole
    length: int
    epoch_offsets: tuple[int, ...]
    sha256: str
    construction: str
    construction_version: str
    source_run_id: str | None = None
    source_checkpoint_sha256: str | None = None
    source_seed: int | None = None
    source_method: str | None = None
    source_scientific_fingerprint: str | None = None
    source_timing_ledger: Mapping[str, Any] | None = None
    construction_details: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        """Validate mask boundaries and content identifiers without file access."""

        if self.length < 0:
            raise ValueError("mask length must be nonnegative")
        if not self.epoch_offsets or self.epoch_offsets[0] != 0:
            raise ValueError("epoch offsets must start at zero")
        if tuple(sorted(self.epoch_offsets)) != self.epoch_offsets:
            raise ValueError("epoch offsets must be ordered")
        if self.epoch_offsets[-1] != self.length:
            raise ValueError("final epoch offset must equal mask length")
        _validate_sha256(self.sha256, "mask sha256")
        if self.source_checkpoint_sha256 is not None:
            _validate_sha256(self.source_checkpoint_sha256, "source checkpoint sha256")
        if self.source_scientific_fingerprint is not None:
            _validate_sha256(self.source_scientific_fingerprint, "source scientific fingerprint")
        if self.source_timing_ledger is not None:
            object.__setattr__(self, "source_timing_ledger", _json_copy(self.source_timing_ledger))
        if self.construction_details is not None:
            object.__setattr__(self, "construction_details", _json_copy(self.construction_details))


def _json_copy(value: Any) -> Any:
    """Copy JSON-compatible values and fail clearly on unsupported objects."""

    try:
        return json.loads(json.dumps(value, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("configuration values must be finite JSON data") from exc


def _required_mapping(values: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    """Return a required mapping section or raise a descriptive validation error."""

    value = values.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _required_number(values: Mapping[str, Any], key: str, positive: bool = False) -> float:
    """Return a finite numeric field and optionally require it to be positive."""

    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{key} must be numeric")
    number = float(value)
    if not (number == number and abs(number) != float("inf")):
        raise ValueError(f"{key} must be finite")
    if positive and number <= 0:
        raise ValueError(f"{key} must be positive")
    return number


def _validate_sha256(value: str, label: str) -> None:
    """Accept a lowercase or uppercase 64-digit SHA-256 string or raise an error."""

    if not isinstance(value, str) or len(value) != 64:
        raise ValueError(f"{label} must contain 64 hexadecimal digits")
    try:
        int(value, 16)
    except ValueError as exc:
        raise ValueError(f"{label} must contain 64 hexadecimal digits") from exc


def load_run_config(path: str | os.PathLike[str]) -> RunConfig:
    """Read a UTF-8 JSON configuration and return a validated detached record."""

    source = Path(path)
    config = RunConfig.from_dict(json.loads(source.read_text(encoding="utf-8")))
    validate_config(config)
    return config


def canonical_json(value: Any) -> str:
    """Serialize finite JSON values deterministically without modifying inputs."""

    clean = _json_copy(value)
    return json.dumps(clean, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def fingerprint(config: RunConfig, artifacts: Mapping[str, str]) -> str:
    """Hash scientific config plus resolved artifact hashes and return lowercase hex.

    Artifact keys are logical identities rather than timestamps or paths.  Every
    artifact value must be a SHA-256 digest.  Execution hardware, output paths,
    and timing policy are intentionally excluded from this scientific identity.
    """

    clean_artifacts = dict(artifacts)
    for name, digest in clean_artifacts.items():
        if not isinstance(name, str) or not name:
            raise ValueError("artifact names must be nonempty strings")
        _validate_sha256(digest, f"artifact {name}")
    payload = {"scientific": config.scientific_dict(), "artifacts": clean_artifacts}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def sha256_file(path: str | os.PathLike[str], chunk_size: int = 1024 * 1024) -> str:
    """Hash a regular file in bounded chunks and return lowercase SHA-256 hex."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_tree_hash(root: str | os.PathLike[str]) -> tuple[str, list[dict[str, Any]]]:
    """Hash every file identity under a directory and return a stable tree digest.

    Entries contain relative POSIX paths, byte sizes, and content hashes.  The
    directory is read only; symlinks and non-files are excluded.
    """

    base = Path(root).resolve()
    if not base.is_dir():
        raise ValueError(f"artifact root is not a directory: {base}")
    entries: list[dict[str, Any]] = []
    for path in sorted((item for item in base.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        entries.append(
            {
                "path": path.relative_to(base).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return hashlib.sha256(canonical_json(entries).encode("utf-8")).hexdigest(), entries


def split_identity_hash(split: Mapping[str, Any]) -> str:
    """Hash the declared dataset-partition contract without loading the dataset."""

    return hashlib.sha256(canonical_json(split).encode("utf-8")).hexdigest()


def mask_identity_hash(mask: Sequence[bool], epoch_offsets: Sequence[int]) -> str:
    """Hash boolean mask bytes together with their explicit epoch boundaries.

    Inputs are a one-dimensional schedule and ordered slot offsets. The returned
    SHA-256 is platform-independent and distinguishes identical bytes under
    different epoch partitions. No files or caller containers are changed.
    """

    values = [bool(item) for item in mask]
    offsets = [int(item) for item in epoch_offsets]
    if not offsets or offsets[0] != 0 or offsets[-1] != len(values):
        raise ValueError("mask epoch offsets must start at zero and end at mask length")
    if offsets != sorted(offsets):
        raise ValueError("mask epoch offsets must be ordered")
    digest = hashlib.sha256(b"phase_d_v4_mask_v1")
    digest.update(len(values).to_bytes(8, "little", signed=False))
    digest.update(bytes(int(item) for item in values))
    digest.update(len(offsets).to_bytes(8, "little", signed=False))
    for offset in offsets:
        digest.update(offset.to_bytes(8, "little", signed=False))
    return digest.hexdigest()


def load_mask(
    mask_path: str | os.PathLike[str],
    manifest_path: str | os.PathLike[str],
    *,
    expected_sha256: str,
    expected_role: MaskRole | str,
    expected_epoch_offsets: Sequence[int],
    expected_source: Mapping[str, Any],
) -> tuple[list[bool], MaskMetadata]:
    """Load and fully verify a donor replay mask before model updates begin.

    Inputs identify an NPZ artifact, its JSON manifest, expected mask hash/role,
    epoch boundaries, and source run/checkpoint/seed/method/fingerprint. The
    output is detached boolean data plus validated metadata. Files are read only;
    missing arrays, malformed boundaries, role conflicts, hashes, or provenance
    raise ``ValueError`` before callers initialize or update a model.
    """

    role = MaskRole(expected_role)
    _validate_sha256(expected_sha256, "expected replay mask sha256")
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    export = manifest.get("replay_export")
    if not isinstance(export, Mapping):
        raise ValueError("mask manifest has no replay_export record")
    compatible = export.get("compatible_roles")
    if not isinstance(compatible, list) or role.value not in compatible:
        raise ValueError(f"donor mask is not compatible with role={role.value}")
    source_checks = {
        "run_id": export.get("source_run_id"),
        "checkpoint_sha256": export.get("source_checkpoint_sha256"),
        "seed": export.get("source_seed"),
        "method": export.get("source_method"),
        "scientific_fingerprint": export.get("source_scientific_fingerprint"),
    }
    for key, observed in source_checks.items():
        if observed != expected_source.get(key):
            raise ValueError(f"replay mask source {key} mismatch")
    array_name = export.get("array")
    if array_name != "actual_trunk_exact":
        raise ValueError("replay export must use the donor's actual trunk-exact mask")
    with np.load(Path(mask_path), allow_pickle=False) as arrays:
        if array_name not in arrays or "epoch_offsets" not in arrays:
            raise ValueError("mask artifact lacks required arrays")
        raw_mask = arrays[array_name]
        raw_offsets = arrays["epoch_offsets"]
        if raw_mask.ndim != 1 or raw_mask.dtype != np.bool_:
            raise ValueError("replay mask must be a one-dimensional boolean array")
        if raw_offsets.ndim != 1 or not np.issubdtype(raw_offsets.dtype, np.integer):
            raise ValueError("mask epoch offsets must be a one-dimensional integer array")
        mask = [bool(item) for item in raw_mask.tolist()]
        offsets = tuple(int(item) for item in raw_offsets.tolist())
    expected_offsets = tuple(int(item) for item in expected_epoch_offsets)
    if offsets != expected_offsets:
        raise ValueError("replay mask epoch boundaries do not match the requested run")
    observed_hash = mask_identity_hash(mask, offsets)
    if observed_hash != expected_sha256 or export.get("sha256") != expected_sha256:
        raise ValueError("replay mask content hash mismatch")
    if int(export.get("length", -1)) != len(mask):
        raise ValueError("replay mask declared length mismatch")
    metadata = MaskMetadata(
        role=role,
        length=len(mask),
        epoch_offsets=offsets,
        sha256=observed_hash,
        construction=str(export.get("construction")),
        construction_version=str(export.get("construction_version")),
        source_run_id=str(export["source_run_id"]),
        source_checkpoint_sha256=str(export["source_checkpoint_sha256"]),
        source_seed=int(export["source_seed"]),
        source_method=str(export["source_method"]),
        source_scientific_fingerprint=str(export["source_scientific_fingerprint"]),
        source_timing_ledger=export.get("source_timing_ledger"),
    )
    return mask, metadata


def validate_config(config: RunConfig) -> None:
    """Validate M0 identities and scientific bounds before any training can start.

    Inputs are a fully resolved :class:`RunConfig`; nothing is returned and no
    state changes.  Missing fields, invalid numerical bounds, or unsupported
    method/version/normalization combinations raise ``ValueError``.
    """

    root = config.values
    if root.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    scientific = _required_mapping(root, "scientific")
    identity = _required_mapping(scientific, "identity")
    data = _required_mapping(scientific, "data")
    training = _required_mapping(scientific, "training")
    controller = _required_mapping(scientific, "controller")
    execution = _required_mapping(root, "execution")
    output = _required_mapping(root, "output")

    try:
        method = MethodId(identity.get("method"))
        normalization = NormalizationMode(identity.get("normalization"))
        behavior = SyntheticBehavior(identity.get("synthetic_behavior"))
        mask_role = MaskRole(identity.get("mask_role"))
    except ValueError as exc:
        raise ValueError("unsupported method, normalization, behavior, or mask role") from exc

    spec = METHOD_SPECS[method]
    if identity.get("algorithm_version") != spec["version"]:
        raise ValueError(f"{method.value} requires algorithm_version={spec['version']}")
    if normalization != spec["normalization"]:
        raise ValueError(f"{method.value} requires normalization={spec['normalization'].value}")
    if behavior not in spec["behaviors"]:
        allowed = sorted(item.value for item in spec["behaviors"])
        raise ValueError(f"{method.value} requires synthetic_behavior in {allowed}")
    if mask_role != spec["mask_role"]:
        raise ValueError(f"{method.value} requires mask_role={spec['mask_role'].value}")
    if identity.get("retrospective") is not spec["retrospective"]:
        raise ValueError(f"{method.value} requires retrospective={spec['retrospective']}")
    if identity.get("deployment_eligible") is not spec["deployment_eligible"]:
        raise ValueError(f"{method.value} requires deployment_eligible={spec['deployment_eligible']}")

    classifier_seed = identity.get("classifier_seed")
    if not isinstance(classifier_seed, int) or isinstance(classifier_seed, bool) or classifier_seed < 0:
        raise ValueError("classifier_seed must be a nonnegative integer")
    mask_replicate = identity.get("mask_replicate")
    if method == MethodId.RELATIVE_GAP_PERMUTATION_EPOCH:
        if not isinstance(mask_replicate, int) or isinstance(mask_replicate, bool) or mask_replicate < 0:
            raise ValueError("relative_gap_permutation_epoch requires a nonnegative mask_replicate")
    elif mask_replicate is not None:
        raise ValueError(f"mask_replicate is not applicable to {method.value}")

    mask_sha = identity.get("mask_sha256")
    if spec["mask_required"] and mask_sha is None:
        raise ValueError(f"{method.value} requires an immutable mask identity")
    if not spec["mask_required"] and mask_sha is not None:
        raise ValueError(f"{method.value} must not declare an external mask identity")
    if mask_sha is not None:
        _validate_sha256(mask_sha, "mask_sha256")
    if method in REPLAY_METHODS:
        source = _required_mapping(scientific, "replay_source")
        if not isinstance(source.get("run_id"), str) or not source["run_id"]:
            raise ValueError("replay source requires a nonempty run_id")
        _validate_sha256(str(source.get("checkpoint_sha256")), "replay source checkpoint")
        _validate_sha256(str(source.get("scientific_fingerprint")), "replay source fingerprint")
        if source.get("method") != MethodId.RELATIVE_FEEDBACK.value:
            raise ValueError("predictor replay source method must be relative_feedback")
        source_seed = source.get("seed")
        if source_seed != classifier_seed:
            raise ValueError("replay source seed must match classifier_seed")
        if identity.get("retrospective") is not True or identity.get("deployment_eligible") is not False:
            raise ValueError("source-derived replay must be retrospective and deployment-ineligible")
    if method in SOURCE_DERIVED_SCHEDULE_METHODS:
        source = _required_mapping(scientific, "schedule_source")
        if not isinstance(source.get("run_id"), str) or not source["run_id"]:
            raise ValueError("source-derived schedule requires a nonempty run_id")
        _validate_sha256(str(source.get("mask_sha256")), "schedule source mask")
        _validate_sha256(str(source.get("checkpoint_sha256")), "schedule source checkpoint")
        _validate_sha256(str(source.get("scientific_fingerprint")), "schedule source fingerprint")
        if source.get("method") != MethodId.RELATIVE_FEEDBACK.value:
            raise ValueError("controller schedule source method must be relative_feedback")
        if source.get("seed") != classifier_seed:
            raise ValueError("schedule source seed must match classifier_seed")
        if identity.get("retrospective") is not True or identity.get("deployment_eligible") is not False:
            raise ValueError("source-derived schedules must be retrospective and deployment-ineligible")
    if method in FIXED_SCHEDULE_METHODS and scientific.get("schedule_source") is not None:
        raise ValueError("fixed schedules must not declare a donor source")

    warmup = _required_number(training, "warmup_epochs", positive=True)
    postwarm = _required_number(training, "postwarm_epochs", positive=True)
    if int(warmup) != warmup or int(postwarm) != postwarm:
        raise ValueError("epoch counts must be positive integers")
    total = _required_number(training, "total_epochs", positive=True)
    if int(total) != total or total != warmup + postwarm:
        raise ValueError("total_epochs must equal warmup_epochs plus postwarm_epochs")
    for key in ("batch_size", "history_window", "maximum_quota"):
        section = training if key == "batch_size" else controller
        value = _required_number(section, key, positive=True)
        if int(value) != value:
            raise ValueError(f"{key} must be a positive integer")

    low = _required_number(controller, "low_quantile")
    high = _required_number(controller, "high_quantile")
    if not 0.0 < low < high < 1.0:
        raise ValueError("controller quantiles must satisfy 0 < low < high < 1")
    ema = _required_number(controller, "ema_old_value_weight")
    if not 0.0 <= ema <= 1.0:
        raise ValueError("EMA old-value weight must be within [0, 1]")
    _required_number(controller, "emergency_multiplier", positive=True)

    if data.get("final_test_loaded") is not False:
        raise ValueError("routine v4 configuration must not load the final test set")
    if execution.get("precision") != "fp32" or execution.get("amp") is not False:
        raise ValueError("the frozen reference requires FP32 with AMP disabled")
    if execution.get("gradient_accumulation_steps") != 1:
        raise ValueError("gradient accumulation is unsupported in the frozen reference")
    if not isinstance(output.get("root"), str) or not output["root"]:
        raise ValueError("output.root must be a nonempty path string")


def validate_output_root(output_root: str | os.PathLike[str], project_root: str | os.PathLike[str]) -> Path:
    """Return a safe resolved output directory or reject protected historical paths.

    The function performs read-only path checks.  It rejects the repository root,
    existing files, and any location inside ``archive``, ``data``, or the v3
    results, as well as the governing source/specification files.
    """

    project = Path(project_root).resolve()
    candidate = (project / output_root).resolve() if not Path(output_root).is_absolute() else Path(output_root).resolve()
    protected_directories = tuple((project / name).resolve() for name in ("archive", "data", "cifar_d_resnet_v3"))
    protected_files = tuple(
        (project / name).resolve()
        for name in ("AGENTS.md", "IMPLEMENTATION_PLAN.md", "RESEARCH_SPEC.md", "gradient_phase_d_resnet_v3.py")
    )
    if candidate == project or candidate in protected_files or candidate.is_file():
        raise ValueError(f"refusing unsafe output root: {candidate}")
    for protected in protected_directories:
        try:
            candidate.relative_to(protected)
        except ValueError:
            continue
        raise ValueError(f"refusing output root inside protected directory: {candidate}")
    return candidate


def validate_issue_register(issue_register: Mapping[str, Sequence[str]]) -> None:
    """Ensure every specified S1-S15 and A1-A18 issue has an owner milestone."""

    expected = {f"S{index}" for index in range(1, 16)} | {f"A{index}" for index in range(1, 19)}
    if set(issue_register) != expected:
        missing = sorted(expected - set(issue_register))
        extra = sorted(set(issue_register) - expected)
        raise ValueError(f"issue register mismatch; missing={missing}, extra={extra}")
    for issue, owners in issue_register.items():
        if not owners or any(owner not in {f"M{index}" for index in range(8)} for owner in owners):
            raise ValueError(f"{issue} must name one or more M0-M7 owners")


EPOCH_COLUMNS = (
    "schema_version",
    "run_id",
    "seed",
    "method",
    "stage",
    "global_epoch",
    "stage_epoch",
    "checkpoint_sequence",
    "checkpoint_sha256",
    "lr",
    "samples_seen",
    "batches_seen",
    "exact_trunk_backwards",
    "synthetic_backwards",
    "zero_backwards",
    "skipped_slots",
    "trunk_optimizer_steps",
    "head_optimizer_steps",
    "predictor_fits",
    "predictor_inferences",
    "validation_accuracy",
    "validation_ce",
    "training_seconds",
    "validation_seconds",
    "observation_time_seconds",
    "data_order_sha256",
)

CONTROLLER_EVENT_COLUMNS = (
    "schema_version",
    "run_id",
    "seed",
    "method",
    "global_epoch",
    "epoch_batch",
    "global_step",
    "error_abs",
    "error_relative",
    "controller_score",
    "threshold_low",
    "threshold_high",
    "synthetic_quota",
    "checkpoint_sequence",
)

RESUME_COMPATIBILITY_POLICY_VERSION = "phase_d_v4.resume_compatibility.v1"
STRICT_RESUME_POLICY = "strict_execution"
COMPATIBLE_T4_RESUME_POLICY = "compatible_nvidia_t4"
RESUME_COMPATIBILITY_POLICIES = (STRICT_RESUME_POLICY, COMPATIBLE_T4_RESUME_POLICY)


def execution_fingerprint(config: RunConfig, environment: Mapping[str, Any]) -> str:
    """Hash execution settings and resolved environment without paths or timestamps."""

    payload = {"execution": config.to_dict()["execution"], "environment": _json_copy(environment)}
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def _query_nvidia_driver_version(device_index: int) -> dict[str, Any]:
    """Read the installed NVIDIA display-driver version with a bounded query.

    The input is a nonnegative CUDA device index. The returned mapping always has
    ``status``, ``source``, ``version``, and ``detail`` fields. ``nvidia-smi`` is
    invoked once with a five-second timeout; its display-driver release is kept
    distinct from PyTorch's CUDA build/runtime version. Missing executables,
    timeouts, nonzero exits, empty output, and placeholder values are reported
    explicitly rather than converted to a version. No driver, backend, RNG, or
    file state is changed.
    """

    if device_index < 0:
        raise ValueError("CUDA device index must be nonnegative")
    command = [
        "nvidia-smi",
        f"--id={device_index}",
        "--query-gpu=driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
    except FileNotFoundError:
        return {"status": "unavailable", "source": "nvidia-smi", "version": None, "detail": "executable_not_found"}
    except subprocess.TimeoutExpired:
        return {"status": "failed", "source": "nvidia-smi", "version": None, "detail": "timeout"}
    except OSError as exc:
        return {"status": "failed", "source": "nvidia-smi", "version": None, "detail": f"os_error:{type(exc).__name__}"}
    if completed.returncode != 0:
        return {
            "status": "failed",
            "source": "nvidia-smi",
            "version": None,
            "detail": f"exit_code:{completed.returncode}",
        }
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    version = lines[0] if len(lines) == 1 else None
    if version is None or re.fullmatch(r"\d+(?:\.\d+)+", version) is None:
        return {"status": "unknown", "source": "nvidia-smi", "version": None, "detail": "invalid_or_ambiguous_output"}
    return {"status": "ok", "source": "nvidia-smi", "version": version, "detail": None}


def resume_compatibility_report(
    saved_config: RunConfig,
    saved_environment: Mapping[str, Any],
    current_config: RunConfig,
    current_environment: Mapping[str, Any],
    policy: str = STRICT_RESUME_POLICY,
) -> dict[str, Any]:
    """Validate one saved execution against an explicitly selected resume policy.

    Inputs are the saved/current resolved configurations and complete environment
    records plus a versioned policy name. The returned JSON-like report records
    required comparisons, provenance-only changes, both execution fingerprints,
    and whether the physical GPU changed. It does not mutate files, devices, RNG,
    or backend flags. Strict mode requires identical execution identity.
    Compatible-T4 mode permits only UUID, device index, detailed Linux platform
    string, and CPU-description changes; missing legacy evidence or any required
    hardware/software/backend mismatch raises before state restoration.
    """

    if policy not in RESUME_COMPATIBILITY_POLICIES:
        raise ValueError(f"unsupported resume compatibility policy: {policy}")
    saved_execution = saved_config.to_dict()["execution"]
    current_execution = current_config.to_dict()["execution"]
    if canonical_json(saved_execution) != canonical_json(current_execution):
        raise ValueError("resolved execution configuration differs from checkpoint")
    saved_fingerprint = execution_fingerprint(saved_config, saved_environment)
    current_fingerprint = execution_fingerprint(current_config, current_environment)
    if policy == STRICT_RESUME_POLICY:
        if saved_fingerprint != current_fingerprint:
            raise ValueError(
                "strict resume execution fingerprint differs; the physical GPU UUID "
                "and complete saved execution environment must match"
            )
        return {
            "policy_version": RESUME_COMPATIBILITY_POLICY_VERSION,
            "policy": policy,
            "compatible": True,
            "cross_device": False,
            "saved_execution_fingerprint": saved_fingerprint,
            "current_execution_fingerprint": current_fingerprint,
            "required_fields": sorted(saved_environment),
            "provenance_only_changes": {},
        }

    provenance_only = {"device_uuid", "device_index", "platform", "processor"}
    required = {
        "python", "system", "machine", "torch", "torchvision", "numpy",
        "cuda_runtime", "cudnn", "device_type", "device_name",
        "device_capability", "total_memory_bytes", "driver_version", "driver_query",
        "cuda_device_count", "cublas_workspace_config", "matmul_tf32",
        "cudnn_tf32", "cudnn_benchmark", "cudnn_deterministic",
        "deterministic_algorithms",
    }
    missing = sorted(
        key for key in required | {"device_uuid"}
        if key not in saved_environment or key not in current_environment
    )
    if missing:
        raise ValueError(
            "legacy checkpoint lacks fields required by compatible-T4 policy: "
            + ", ".join(missing)
        )
    for label, environment in (("saved", saved_environment), ("current", current_environment)):
        query = environment.get("driver_query")
        version = environment.get("driver_version")
        if (
            not isinstance(query, Mapping)
            or query.get("status") != "ok"
            or query.get("source") != "nvidia-smi"
            or not isinstance(version, str)
            or re.fullmatch(r"\d+(?:\.\d+)+", version) is None
            or query.get("version") != version
        ):
            raise ValueError(f"compatible-T4 recovery lacks valid {label} NVIDIA driver evidence")
    allowed_t4_names = {"NVIDIA T4", "Tesla T4"}
    if (
        saved_environment["device_type"] != "cuda"
        or current_environment["device_type"] != "cuda"
        or saved_environment["device_name"] not in allowed_t4_names
        or current_environment["device_name"] not in allowed_t4_names
        or list(saved_environment["device_capability"]) != [7, 5]
        or list(current_environment["device_capability"]) != [7, 5]
    ):
        raise ValueError("compatible-T4 recovery requires NVIDIA T4 compute capability 7.5")
    mismatches = {
        key: {"saved": saved_environment.get(key), "current": current_environment.get(key)}
        for key in sorted(required - {"device_name"})
        if canonical_json(saved_environment.get(key))
        != canonical_json(current_environment.get(key))
    }
    if mismatches:
        raise ValueError(
            "compatible-T4 recovery has required environment mismatches: "
            + ", ".join(mismatches)
        )
    changes = {
        key: {"saved": saved_environment.get(key), "current": current_environment.get(key)}
        for key in sorted(provenance_only)
        if canonical_json(saved_environment.get(key))
        != canonical_json(current_environment.get(key))
    }
    saved_uuid = str(saved_environment["device_uuid"])
    current_uuid = str(current_environment["device_uuid"])
    if not saved_uuid or not current_uuid:
        raise ValueError("compatible-T4 recovery requires nonempty saved/current GPU UUIDs")
    return {
        "policy_version": RESUME_COMPATIBILITY_POLICY_VERSION,
        "policy": policy,
        "compatible": True,
        "cross_device": saved_uuid != current_uuid,
        "saved_execution_fingerprint": saved_fingerprint,
        "current_execution_fingerprint": current_fingerprint,
        "required_fields": sorted(required),
        "provenance_only_changes": changes,
    }


def environment_metadata(device: torch.device | str) -> dict[str, Any]:
    """Return reproducibility metadata without changing hardware or backend flags."""

    selected = torch.device(device)
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "system": platform.system(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch": str(torch.__version__),
        "torchvision": importlib_metadata.version("torchvision"),
        "numpy": np.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_type": selected.type,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "matmul_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_tf32": bool(torch.backends.cudnn.allow_tf32),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
    }
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = selected.index if selected.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        metadata.update(
            device_index=index,
            device_name=properties.name,
            device_uuid=str(getattr(properties, "uuid", "")),
            device_capability=[int(properties.major), int(properties.minor)],
            total_memory_bytes=int(properties.total_memory),
            cuda_device_count=int(torch.cuda.device_count()),
        )
        driver_query = _query_nvidia_driver_version(index)
        metadata["driver_query"] = driver_query
        metadata["driver_version"] = driver_query["version"]
    else:
        metadata["driver_query"] = {
            "status": "not_applicable", "source": "nvidia-smi",
            "version": None, "detail": "non_cuda_device",
        }
        metadata["driver_version"] = None
    return metadata


def capture_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, CPU Torch, and all CUDA RNG states as safe primitives."""

    numpy_state = np.random.get_state()
    return {
        "python": _tuples_to_lists(random.getstate()),
        "numpy": {
            "algorithm": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": int(numpy_state[2]),
            "has_gauss": int(numpy_state[3]),
            "cached_gaussian": float(numpy_state[4]),
        },
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone().cpu() for state in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore every captured RNG stream, mutating process-global generator states."""

    random.setstate(_lists_to_tuples(state["python"]))
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            numpy_state["algorithm"],
            numpy_state["keys"].cpu().numpy().astype(np.uint32, copy=False),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state.get("torch_cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        if len(state["torch_cuda"]) != torch.cuda.device_count():
            raise RuntimeError("CUDA device count differs from checkpoint RNG state")
        torch.cuda.set_rng_state_all([item.cpu() for item in state["torch_cuda"]])


def _tuples_to_lists(value: Any) -> Any:
    """Convert nested tuples to primitive lists for weights-only checkpoint loading."""

    if isinstance(value, tuple):
        return [_tuples_to_lists(item) for item in value]
    if isinstance(value, list):
        return [_tuples_to_lists(item) for item in value]
    return value


def _lists_to_tuples(value: Any) -> Any:
    """Convert nested lists back to tuples for Python's random-state API."""

    if isinstance(value, list):
        return tuple(_lists_to_tuples(item) for item in value)
    return value


def run_state_to_dict(state: RunState) -> dict[str, Any]:
    """Serialize mutable run counters and committed records to primitive values."""

    return {
        "stage": state.stage.value,
        "completed_epoch": state.completed_epoch,
        "next_epoch": state.next_epoch,
        "global_step": state.global_step,
        "postwarm_step": state.postwarm_step,
        "pending_quota": state.pending_quota,
        "exact_trunk_backwards": state.exact_trunk_backwards,
        "synthetic_backwards": state.synthetic_backwards,
        "zero_backwards": state.zero_backwards,
        "skipped_slots": state.skipped_slots,
        "trunk_optimizer_steps": state.trunk_optimizer_steps,
        "head_optimizer_steps": state.head_optimizer_steps,
        "predictor_fits": state.predictor_fits,
        "predictor_inferences": state.predictor_inferences,
        "scale": state.scale,
        "lr_global_epoch": state.lr_global_epoch,
        "mask_cursor": state.mask_cursor,
        "executed_exact_mask": list(state.executed_exact_mask),
        "executed_head_mask": list(state.executed_head_mask),
        "replay_schedule_mask": list(state.replay_schedule_mask),
        "replay_epoch_offsets": list(state.replay_epoch_offsets),
        "mask_epoch_offsets": list(state.mask_epoch_offsets),
        "checkpoint_sequence": state.checkpoint_sequence,
        "epoch_records": copy.deepcopy(state.epoch_records),
        "controller_events": copy.deepcopy(state.controller_events),
        "data_order_hashes": copy.deepcopy(state.data_order_hashes),
        "target_attainment": copy.deepcopy(state.target_attainment),
        "interruptions": copy.deepcopy(state.interruptions),
        "completed": state.completed,
    }


def run_state_from_dict(values: Mapping[str, Any]) -> RunState:
    """Restore a :class:`RunState` from validated primitive checkpoint values."""

    return RunState(
        stage=RunStage(values["stage"]),
        completed_epoch=int(values["completed_epoch"]),
        next_epoch=int(values["next_epoch"]),
        global_step=int(values["global_step"]),
        postwarm_step=int(values["postwarm_step"]),
        pending_quota=int(values["pending_quota"]),
        exact_trunk_backwards=int(values["exact_trunk_backwards"]),
        synthetic_backwards=int(values["synthetic_backwards"]),
        zero_backwards=int(values["zero_backwards"]),
        skipped_slots=int(values.get("skipped_slots", 0)),
        trunk_optimizer_steps=int(values["trunk_optimizer_steps"]),
        head_optimizer_steps=int(values["head_optimizer_steps"]),
        predictor_fits=int(values["predictor_fits"]),
        predictor_inferences=int(values["predictor_inferences"]),
        scale=None if values.get("scale") is None else float(values["scale"]),
        lr_global_epoch=int(values.get("lr_global_epoch", -1)),
        mask_cursor=int(values.get("mask_cursor", 0)),
        executed_exact_mask=[bool(item) for item in values.get("executed_exact_mask", [])],
        executed_head_mask=[bool(item) for item in values.get("executed_head_mask", [])],
        replay_schedule_mask=[bool(item) for item in values.get("replay_schedule_mask", [])],
        replay_epoch_offsets=[int(item) for item in values.get("replay_epoch_offsets", [0])],
        mask_epoch_offsets=[int(item) for item in values.get("mask_epoch_offsets", [0])],
        checkpoint_sequence=int(values.get("checkpoint_sequence", 0)),
        epoch_records=copy.deepcopy(values.get("epoch_records", [])),
        controller_events=copy.deepcopy(values.get("controller_events", [])),
        data_order_hashes=copy.deepcopy(values.get("data_order_hashes", [])),
        target_attainment=copy.deepcopy(values.get("target_attainment", {})),
        interruptions=copy.deepcopy(values.get("interruptions", [])),
        completed=bool(values.get("completed", False)),
    )


def atomic_write_json(path: str | os.PathLike[str], values: Mapping[str, Any]) -> None:
    """Atomically write finite JSON in one directory and fsync before replacement."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    encoded = (json.dumps(_json_copy(values), indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)


def _read_checkpoint_journal(path: Path) -> list[dict[str, Any]]:
    """Read checkpoint timing records from either current or empty journal state."""

    if not path.exists():
        return []
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(loaded, dict):
        loaded = loaded.get("records", [])
    if not isinstance(loaded, list):
        raise ValueError("checkpoint journal must contain records")
    return loaded


def save_checkpoint(
    path: str | os.PathLike[str],
    payload: Mapping[str, Any],
    *,
    journal_path: str | os.PathLike[str] | None = None,
    simulate_failure: str | None = None,
) -> dict[str, Any]:
    """Commit one immutable checkpoint, checksum metadata, then timing journal.

    ``payload`` must contain a positive ``checkpoint_sequence`` and safe values
    accepted by ``torch.load(weights_only=True)``. The destination must be new.
    Failures can be injected after file replacement or before journal append for
    tests; such artifacts remain visibly partial rather than being reused.
    """

    destination = Path(path)
    if destination.exists() or destination.with_suffix(destination.suffix + ".commit.json").exists():
        raise FileExistsError(f"checkpoint already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    sequence = int(payload["checkpoint_sequence"])
    temporary = destination.with_name(destination.name + ".tmp")
    started = time.perf_counter()
    with temporary.open("wb") as handle:
        torch.save(dict(payload), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    if simulate_failure == "after_checkpoint":
        raise RuntimeError("simulated failure after checkpoint replacement")
    digest = sha256_file(destination)
    commit_path = destination.with_suffix(destination.suffix + ".commit.json")
    atomic_write_json(
        commit_path,
        {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_sequence": sequence,
            "checkpoint_file": destination.name,
            "sha256": digest,
            "size_bytes": destination.stat().st_size,
        },
    )
    elapsed = time.perf_counter() - started
    if simulate_failure == "before_journal":
        raise RuntimeError("simulated failure before checkpoint timing journal")
    if journal_path is not None:
        journal = Path(journal_path)
        records = _read_checkpoint_journal(journal)
        records = [item for item in records if int(item["checkpoint_sequence"]) != sequence]
        records.append(
            {
                "checkpoint_sequence": sequence,
                "checkpoint_file": destination.name,
                "sha256": digest,
                "io_seconds": elapsed,
            }
        )
        records.sort(key=lambda item: int(item["checkpoint_sequence"]))
        atomic_write_json(journal, {"records": records})
    return {"sha256": digest, "size_bytes": destination.stat().st_size, "io_seconds": elapsed}


def load_checkpoint(
    path: str | os.PathLike[str],
    *,
    expected: Mapping[str, Any] | None = None,
    journal_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Verify checksum and identities, then weights-only load one checkpoint.

    Expected key/value pairs are compared exactly before caller state changes.
    Missing commit metadata, corruption, schema mismatch, or identity mismatch
    raises an error. A missing timing journal entry is explicitly flagged.
    """

    source = Path(path)
    commit_path = source.with_suffix(source.suffix + ".commit.json")
    if not source.is_file() or not commit_path.is_file():
        raise ValueError("checkpoint or commit metadata is missing")
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    digest = sha256_file(source)
    if digest != commit.get("sha256") or source.stat().st_size != int(commit.get("size_bytes", -1)):
        raise ValueError("checkpoint checksum or size mismatch")
    payload = torch.load(source, map_location="cpu", weights_only=True)
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("checkpoint schema mismatch")
    if int(payload.get("checkpoint_sequence", -1)) != int(commit["checkpoint_sequence"]):
        raise ValueError("checkpoint sequence metadata mismatch")
    for key, value in (expected or {}).items():
        if payload.get(key) != value:
            raise ValueError(f"checkpoint identity mismatch for {key}")
    timing_incomplete = False
    timing_record: dict[str, Any] | None = None
    if journal_path is not None:
        records = _read_checkpoint_journal(Path(journal_path))
        timing_record = next(
            (
                item for item in records
                if int(item["checkpoint_sequence"]) == int(payload["checkpoint_sequence"])
                and item.get("sha256") == digest
            ),
            None,
        )
        timing_incomplete = timing_record is None
    payload["resume_metadata"] = {
        "checkpoint_sha256": digest,
        "timing_incomplete": timing_incomplete,
        "checkpoint_io_record": timing_record,
    }
    return payload


def latest_committed_checkpoint(checkpoint_directory: str | os.PathLike[str]) -> Path:
    """Return the highest-sequence checksum-committed checkpoint in a directory."""

    directory = Path(checkpoint_directory)
    candidates: list[tuple[int, Path]] = []
    for commit_path in directory.glob("*.pt.commit.json"):
        try:
            commit = json.loads(commit_path.read_text(encoding="utf-8"))
            checkpoint = directory / commit["checkpoint_file"]
            if checkpoint.is_file() and sha256_file(checkpoint) == commit["sha256"]:
                candidates.append((int(commit["checkpoint_sequence"]), checkpoint))
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError):
            continue
    if not candidates:
        raise FileNotFoundError("no valid committed checkpoint found")
    return max(candidates, key=lambda item: item[0])[1]


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    """Atomically write rows with one stable schema and no extra columns."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in columns})
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _unique_records(rows: Sequence[Mapping[str, Any]], keys: Sequence[str]) -> list[dict[str, Any]]:
    """Return sorted unique records or fail when one logical key is duplicated."""

    seen: set[tuple[Any, ...]] = set()
    result: list[dict[str, Any]] = []
    for row in rows:
        identity = tuple(row[key] for key in keys)
        if identity in seen:
            raise ValueError(f"duplicate committed record: {identity}")
        seen.add(identity)
        result.append(dict(row))
    return result


def reconcile_outputs(run_directory: str | os.PathLike[str], payload: Mapping[str, Any]) -> None:
    """Rebuild committed CSV/JSON outputs idempotently from checkpoint records.

    Existing uncommitted or duplicated rows are discarded because the checkpoint
    is authoritative. Files are atomically replaced; malformed committed records
    fail rather than being silently deduplicated.
    """

    run_directory = Path(run_directory)
    state = payload["run_state"]
    epoch_rows = copy.deepcopy(state.get("epoch_records", []))
    current_digest = payload.get("resume_metadata", {}).get("checkpoint_sha256", "")
    for row in epoch_rows:
        if int(row.get("checkpoint_sequence", -1)) == int(payload["checkpoint_sequence"]):
            row["checkpoint_sha256"] = current_digest
    epoch_rows = _unique_records(epoch_rows, ("run_id", "global_epoch"))
    event_rows = _unique_records(
        state.get("controller_events", []), ("run_id", "global_epoch", "epoch_batch")
    )
    _write_csv_atomic(run_directory / "epochs.csv", epoch_rows, EPOCH_COLUMNS)
    _write_csv_atomic(run_directory / "controller_events.csv", event_rows, CONTROLLER_EVENT_COLUMNS)
    atomic_write_json(run_directory / "timing.json", payload["timing_ledger"])
    atomic_write_json(
        run_directory / "summary.json",
        {
            "schema_version": SCHEMA_VERSION,
            "run_id": payload["run_id"],
            "seed": payload["seed"],
            "method": payload["method"],
            "completed": state.get("completed", False),
            "completed_epoch": state.get("completed_epoch", -1),
            "checkpoint_sequence": payload["checkpoint_sequence"],
            "timing_incomplete": payload.get("resume_metadata", {}).get("timing_incomplete", False),
        },
    )
    mask_path = run_directory / "masks.npz"
    mask_temporary = run_directory / "masks.npz.tmp"
    actual_trunk = [bool(item) for item in state.get("executed_exact_mask", [])]
    actual_head = [bool(item) for item in state.get("executed_head_mask", [])]
    actual_offsets = [int(item) for item in state.get("mask_epoch_offsets", [0])]
    schedule = [bool(item) for item in state.get("replay_schedule_mask", [])]
    replay_metadata = payload.get("replay_mask_metadata")
    schedule_offsets = (
        [int(item) for item in replay_metadata["epoch_offsets"]]
        if isinstance(replay_metadata, Mapping) else [0]
    )
    with mask_temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            actual_trunk_exact=np.asarray(actual_trunk, dtype=np.bool_),
            head_update=np.asarray(actual_head, dtype=np.bool_),
            schedule=np.asarray(schedule, dtype=np.bool_),
            epoch_offsets=np.asarray(actual_offsets, dtype=np.int64),
            schedule_epoch_offsets=np.asarray(schedule_offsets, dtype=np.int64),
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(mask_temporary, mask_path)
    actual_hash = mask_identity_hash(actual_trunk, actual_offsets)
    head_hash = mask_identity_hash(actual_head, actual_offsets)
    schedule_hash = mask_identity_hash(schedule, schedule_offsets) if schedule else None
    replay_export = None
    if payload.get("method") == MethodId.RELATIVE_FEEDBACK.value:
        replay_export = {
            "array": "actual_trunk_exact",
            "compatible_roles": [MaskRole.SCHEDULE.value, MaskRole.HEAD_UPDATE.value],
            "length": len(actual_trunk),
            "epoch_offsets": actual_offsets,
            "sha256": actual_hash,
            "construction": "relative_feedback_recorded_actual_trunk_exact",
            "construction_version": "v4_m4",
            "source_run_id": payload["run_id"],
            "source_checkpoint_sha256": current_digest,
            "source_seed": payload["seed"],
            "source_method": payload["method"],
            "source_scientific_fingerprint": payload["scientific_fingerprint"],
            "source_timing_ledger": payload["timing_ledger"],
        }
    mask_manifest = {
        "schema_version": SCHEMA_VERSION,
        "declared_role": payload.get("mask_role", "none"),
        "artifact_sha256": sha256_file(mask_path),
        "actual_trunk_exact": {
            "length": len(actual_trunk), "epoch_offsets": actual_offsets, "sha256": actual_hash,
        },
        "head_update": {
            "length": len(actual_head), "epoch_offsets": actual_offsets, "sha256": head_hash,
        },
        "schedule": {
            "length": len(schedule), "epoch_offsets": schedule_offsets, "sha256": schedule_hash,
        },
        "source_mask_sha256": payload.get("source_mask_sha256"),
        "consumed_cursor": state.get("mask_cursor", 0),
        "replay_export": replay_export,
        "construction_details": (
            replay_metadata.get("construction_details")
            if isinstance(replay_metadata, Mapping) else None
        ),
    }
    atomic_write_json(run_directory / "mask_manifest.json", mask_manifest)
    if isinstance(replay_metadata, Mapping):
        source_dependent = replay_metadata.get("source_run_id") is not None
        execution_cost_key = (
            "conditional_execution"
            if payload["method"] in {item.value for item in REPLAY_METHODS}
            else "execution"
        )
        cost_records: dict[str, Any] = {
            execution_cost_key: {
                "meaning": (
                    "execution_given_an_already_available_source_mask"
                    if source_dependent else "standalone_fixed_schedule_execution"
                ),
                "timing_ledger": payload["timing_ledger"],
            }
        }
        if source_dependent:
            cost_records["dependency_inclusive"] = {
                "meaning": "source_dependency_plus_schedule_execution_without_independent_deployability",
                "source_timing_ledger": replay_metadata.get("source_timing_ledger"),
                "schedule_timing_ledger": payload["timing_ledger"],
                "shared_warmup_counting": "count_once_in_later_M6_analysis",
                "status": "components_recorded_no_M5_speed_claim",
            }
        provenance_name = (
            "replay_provenance.json"
            if payload["method"] in {item.value for item in REPLAY_METHODS}
            else "schedule_provenance.json"
        )
        atomic_write_json(
            run_directory / provenance_name,
            {
                "schema_version": SCHEMA_VERSION,
                "method": payload["method"],
                "retrospective": bool(payload["retrospective"]),
                "deployment_eligible": bool(payload["deployment_eligible"]),
                "mask": dict(replay_metadata),
                "cost_records": cost_records,
            },
        )


def copy_checkpoint_verified(
    checkpoint: str | os.PathLike[str], remote_directory: str | os.PathLike[str]
) -> Path:
    """Copy a committed checkpoint to a directory and publish its marker last.

    This local-filesystem implementation models Drive persistence without network
    access. The immutable destination is verified byte-for-byte before its commit
    marker is atomically written. Existing mismatched files are rejected.
    """

    source = Path(checkpoint)
    source_commit = source.with_suffix(source.suffix + ".commit.json")
    if not source_commit.is_file():
        raise ValueError("source checkpoint is not committed")
    metadata = json.loads(source_commit.read_text(encoding="utf-8"))
    if sha256_file(source) != metadata["sha256"]:
        raise ValueError("source checkpoint checksum mismatch")
    destination_directory = Path(remote_directory)
    destination_directory.mkdir(parents=True, exist_ok=True)
    destination = destination_directory / source.name
    if destination.exists():
        if sha256_file(destination) != metadata["sha256"]:
            raise FileExistsError("remote checkpoint name exists with different content")
    else:
        staging = destination.with_name(destination.name + ".partial")
        shutil.copyfile(source, staging)
        if sha256_file(staging) != metadata["sha256"]:
            raise IOError("remote staging checksum mismatch")
        os.replace(staging, destination)
    atomic_write_json(destination.with_suffix(destination.suffix + ".commit.json"), metadata)
    return destination


def copy_file_verified(
    source_file: str | os.PathLike[str], destination_file: str | os.PathLike[str]
) -> Path:
    """Copy one immutable support artifact and publish checksum metadata last.

    Inputs are an existing regular source file and an exact destination path.
    The returned destination has identical bytes and a sibling ``.commit.json``
    marker containing SHA-256 and size. Parent directories are created; an
    existing mismatched destination is rejected, partial staging is ignored by
    recovery, and no source, RNG, model, or network state is changed.
    """

    source = Path(source_file)
    if not source.is_file():
        raise FileNotFoundError(f"support artifact is missing: {source}")
    digest = sha256_file(source)
    size = source.stat().st_size
    destination = Path(destination_file)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256_file(destination) != digest or destination.stat().st_size != size:
            raise FileExistsError("support artifact destination has different content")
    else:
        staging = destination.with_name(destination.name + ".partial")
        shutil.copyfile(source, staging)
        if sha256_file(staging) != digest or staging.stat().st_size != size:
            raise IOError("support artifact staging checksum mismatch")
        os.replace(staging, destination)
    atomic_write_json(
        destination.with_suffix(destination.suffix + ".commit.json"),
        {"sha256": digest, "size_bytes": size, "source_name": source.name},
    )
    return destination


def load_resume_attempts(
    run_directory: str | os.PathLike[str],
    persistence_directory: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Load every checksum-committed resume-attempt record in order.

    Inputs identify the local run and optional verified persistence directory.
    The returned records merge identical attempt IDs, prefer neither location,
    and reject corrupt/conflicting committed copies. Missing directories and
    uncommitted partial files contribute nothing. No files, RNG, or training
    state are changed.
    """

    roots = [Path(run_directory) / "resume_attempts"]
    if persistence_directory is not None:
        roots.append(Path(persistence_directory) / "resume_attempts")
    by_id: dict[int, dict[str, Any]] = {}
    for root in roots:
        for marker_path in sorted(root.glob("attempt_*.json.commit.json")):
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            record_path = marker_path.with_name(marker_path.name.removesuffix(".commit.json"))
            if (
                not record_path.is_file()
                or sha256_file(record_path) != marker.get("sha256")
                or record_path.stat().st_size != int(marker.get("size_bytes", -1))
            ):
                raise ValueError("resume-attempt record checksum mismatch")
            record = json.loads(record_path.read_text(encoding="utf-8"))
            attempt_id = int(record.get("attempt_id", -1))
            if attempt_id < 0:
                raise ValueError("resume-attempt record lacks a valid identity")
            if attempt_id in by_id and canonical_json(by_id[attempt_id]) != canonical_json(record):
                raise ValueError("conflicting local and persisted resume-attempt records")
            by_id[attempt_id] = record
    return [by_id[key] for key in sorted(by_id)]


def record_resume_attempt(
    run_directory: str | os.PathLike[str],
    persistence_directory: str | os.PathLike[str] | None,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Append one immutable resume-attempt record before continuation starts.

    Inputs identify local/optional persistence locations and a JSON-like record.
    The returned record gains the next monotonic attempt ID. It is atomically
    written locally and, when enabled, copied with checksum marker to persistence.
    Existing attempt files are never overwritten, so a later recovery can detect
    repeated work from the same committed epoch. Failures raise before training;
    no model, optimizer, controller, loader, or RNG state is changed.
    """

    existing = load_resume_attempts(run_directory, persistence_directory)
    values = _json_copy(record)
    values["attempt_id"] = 0 if not existing else int(existing[-1]["attempt_id"]) + 1
    local = Path(run_directory) / "resume_attempts" / f"attempt_{values['attempt_id']:06d}.json"
    if local.exists():
        raise FileExistsError("resume-attempt identity already exists")
    atomic_write_json(local, values)
    local_marker = {
        "sha256": sha256_file(local),
        "size_bytes": local.stat().st_size,
        "source_name": local.name,
    }
    atomic_write_json(local.with_suffix(local.suffix + ".commit.json"), local_marker)
    if persistence_directory is not None:
        copy_file_verified(
            local,
            Path(persistence_directory) / "resume_attempts" / local.name,
        )
    return values


def load_continuation_evidence(
    run_directory: str | os.PathLike[str],
    persistence_directory: str | os.PathLike[str] | None = None,
) -> list[dict[str, Any]]:
    """Load checksum-committed cross-device continuation evidence in order.

    Inputs identify a local run and optional persistence directory. The returned
    records are merged by resume-attempt ID and require identical local/remote
    content, valid checksums, and a successful copied-state status. Corrupt,
    conflicting, or malformed committed evidence raises before training. Missing
    directories and partial files are ignored; no state or files are modified.
    """

    roots = [Path(run_directory) / "continuation_checks"]
    if persistence_directory is not None:
        roots.append(Path(persistence_directory) / "continuation_checks")
    by_id: dict[int, dict[str, Any]] = {}
    for root in roots:
        for marker_path in sorted(root.glob("attempt_*.json.commit.json")):
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            record_path = marker_path.with_name(marker_path.name.removesuffix(".commit.json"))
            if (
                not record_path.is_file()
                or sha256_file(record_path) != marker.get("sha256")
                or record_path.stat().st_size != int(marker.get("size_bytes", -1))
            ):
                raise ValueError("continuation evidence checksum mismatch")
            record = json.loads(record_path.read_text(encoding="utf-8"))
            attempt_id = int(record.get("resume_attempt_id", -1))
            if (
                attempt_id < 0
                or record.get("status") != "copied_state_deterministic_continuation_passed"
                or record.get("production_state_mutated") is not False
            ):
                raise ValueError("continuation evidence identity or status is invalid")
            if attempt_id in by_id and canonical_json(by_id[attempt_id]) != canonical_json(record):
                raise ValueError("conflicting local and persisted continuation evidence")
            by_id[attempt_id] = record
    return [by_id[key] for key in sorted(by_id)]


def record_continuation_evidence(
    run_directory: str | os.PathLike[str],
    persistence_directory: str | os.PathLike[str] | None,
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    """Persist one immutable successful continuation check for a resume attempt.

    Inputs identify local/optional persistence locations and evidence already
    bound to a checkpoint, environment, and attempt. The returned primitive
    record is written atomically with checksum markers locally and remotely.
    Existing evidence is never overwritten; invalid status or IDs raise before
    writing. This changes only evidence files and has no training/RNG side effect.
    """

    values = _json_copy(evidence)
    attempt_id = int(values.get("resume_attempt_id", -1))
    if (
        attempt_id < 0
        or values.get("status") != "copied_state_deterministic_continuation_passed"
        or values.get("production_state_mutated") is not False
    ):
        raise ValueError("only successful bound continuation evidence may be persisted")
    existing = load_continuation_evidence(run_directory, persistence_directory)
    if any(int(item["resume_attempt_id"]) == attempt_id for item in existing):
        raise FileExistsError("continuation evidence already exists for resume attempt")
    local = Path(run_directory) / "continuation_checks" / f"attempt_{attempt_id:06d}.json"
    atomic_write_json(local, values)
    marker = {
        "sha256": sha256_file(local),
        "size_bytes": local.stat().st_size,
        "source_name": local.name,
    }
    atomic_write_json(local.with_suffix(local.suffix + ".commit.json"), marker)
    if persistence_directory is not None:
        copy_file_verified(
            local,
            Path(persistence_directory) / "continuation_checks" / local.name,
        )
    return values


def backup_study_support(
    source_file: str | os.PathLike[str],
    study_root: str | os.PathLike[str],
    persistence_root: str | os.PathLike[str],
) -> Path:
    """Back up one study-local support file under its verified relative path.

    Inputs are an existing file, the local study root that must contain it, and
    one persistence root. The returned immutable copy is placed below
    ``_study_support`` with a checksum marker. Paths outside the study fail,
    existing mismatches are rejected, and no source file is modified.
    """

    source = Path(source_file).resolve()
    root = Path(study_root).resolve()
    try:
        relative = source.relative_to(root)
    except ValueError as exc:
        raise ValueError("support artifact must be inside the study root") from exc
    return copy_file_verified(
        source, Path(persistence_root) / "_study_support" / relative
    )


def restore_study_support(
    destination_file: str | os.PathLike[str],
    study_root: str | os.PathLike[str],
    persistence_root: str | os.PathLike[str],
) -> Path:
    """Restore one checksum-verified study support file without overwriting it.

    Inputs identify the desired study-local path, study root, and persistence
    root. The returned local file is copied through a temporary path after both
    source and marker validation. Existing identical files are reused; existing
    differences, path escape, missing markers, or corruption raise. No training,
    RNG, device, or unrelated historical data is touched.
    """

    destination = Path(destination_file).resolve()
    root = Path(study_root).resolve()
    try:
        relative = destination.relative_to(root)
    except ValueError as exc:
        raise ValueError("support destination must be inside the study root") from exc
    source = Path(persistence_root) / "_study_support" / relative
    marker = source.with_suffix(source.suffix + ".commit.json")
    if not source.is_file() or not marker.is_file():
        raise FileNotFoundError(f"verified support artifact is missing: {source}")
    metadata = json.loads(marker.read_text(encoding="utf-8"))
    if (
        sha256_file(source) != metadata.get("sha256")
        or source.stat().st_size != int(metadata.get("size_bytes", -1))
    ):
        raise ValueError(f"support artifact checksum mismatch: {source}")
    if destination.exists():
        if (
            sha256_file(destination) != metadata["sha256"]
            or destination.stat().st_size != int(metadata["size_bytes"])
        ):
            raise FileExistsError(f"existing support artifact differs: {destination}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.with_name(destination.name + ".restore.partial")
    shutil.copyfile(source, staging)
    if (
        sha256_file(staging) != metadata["sha256"]
        or staging.stat().st_size != int(metadata["size_bytes"])
    ):
        staging.unlink(missing_ok=True)
        raise IOError("restored support artifact failed checksum validation")
    os.replace(staging, destination)
    return destination


def backup_resume_snapshot(
    run_directory: str | os.PathLike[str],
    persistence_directory: str | os.PathLike[str],
    checkpoint: str | os.PathLike[str],
) -> Path:
    """Back up one committed epoch's complete recovery metadata.

    The checkpoint is copied and verified first. Every support file is copied
    to an epoch-specific snapshot, and snapshot.ready.json is written last.
    An interrupted snapshot cannot replace a previously valid recovery point.
    """

    run = Path(run_directory)
    remote = Path(persistence_directory)
    source_checkpoint = Path(checkpoint)

    remote_checkpoint = copy_checkpoint_verified(
        source_checkpoint, remote / "checkpoints"
    )

    commit = json.loads(
        source_checkpoint.with_suffix(
            source_checkpoint.suffix + ".commit.json"
        ).read_text(encoding="utf-8")
    )
    sequence = int(commit["checkpoint_sequence"])

    snapshot = remote / "snapshots" / f"checkpoint_{sequence:06d}"
    ready_path = snapshot / "snapshot.ready.json"

    if ready_path.exists():
        raise FileExistsError(
            f"Recovery snapshot already committed: {ready_path}"
        )

    required = (
        "manifest.json",
        "checkpoint_journal.json",
        "timing.json",
        "epochs.csv",
        "controller_events.csv",
        "summary.json",
        "masks.npz",
        "mask_manifest.json",
        "errors.jsonl",
    )
    optional = (
        "calibration.json",
        "replay_provenance.json",
        "schedule_provenance.json",
        "final_predictor.pt",
    )

    files = {}

    for name in required + optional:
        source = run / name

        if not source.exists():
            if name in required:
                raise FileNotFoundError(
                    f"Required recovery file missing: {source}"
                )
            continue

        copied = copy_file_verified(source, snapshot / name)
        files[name] = {
            "sha256": sha256_file(copied),
            "size_bytes": copied.stat().st_size,
        }

    ready = {
        "schema_version": SCHEMA_VERSION,
        "run_id": run.name,
        "checkpoint_sequence": sequence,
        "checkpoint_file": remote_checkpoint.name,
        "checkpoint_sha256": commit["sha256"],
        "files": files,
        "timing_record_required": True,
    }

    atomic_write_json(ready_path, ready)
    return ready_path


def commit_resume_snapshot_timing(
    ready_path: str | os.PathLike[str], io_seconds: float
) -> Path:
    """Publish the measured snapshot duration after the snapshot commits.

    Inputs are a committed ``snapshot.ready.json`` path and the finite,
    nonnegative checkpoint/log/persistence duration in seconds measured by the
    caller. The returned timing-record path is written atomically and receives a
    checksum marker last. Recovery rejects snapshots without this record, so a
    crash cannot make an incompletely timed workflow look comparison-eligible.
    The small journal publication itself follows the plan's post-commit timing
    convention and is not recursively charged to its own duration.
    """

    ready_file = Path(ready_path)
    if not ready_file.is_file():
        raise FileNotFoundError("snapshot ready marker is missing")
    seconds = float(io_seconds)
    if not math.isfinite(seconds) or seconds < 0.0:
        raise ValueError("snapshot I/O duration must be finite and nonnegative")
    ready = json.loads(ready_file.read_text(encoding="utf-8"))
    record_path = ready_file.parent / "snapshot.timing.json"
    atomic_write_json(
        record_path,
        {
            "schema_version": SCHEMA_VERSION,
            "checkpoint_sequence": int(ready["checkpoint_sequence"]),
            "checkpoint_sha256": ready["checkpoint_sha256"],
            "checkpoint_log_io_seconds": seconds,
        },
    )
    marker = {
        "sha256": sha256_file(record_path),
        "size_bytes": record_path.stat().st_size,
    }
    atomic_write_json(
        record_path.with_suffix(record_path.suffix + ".commit.json"), marker
    )
    return record_path


def _verified_snapshot_timing(ready_path: Path, ready: Mapping[str, Any]) -> dict[str, Any]:
    """Read one checksum-verified snapshot timing record without changing files.

    Inputs are a snapshot ready path and its parsed identity. The returned
    mapping contains the matching sequence, checkpoint hash, and elapsed seconds.
    Missing, corrupt, nonfinite, or mismatched records raise so callers can skip
    that recovery point rather than treating its timing as complete.
    """

    record_path = ready_path.parent / "snapshot.timing.json"
    marker_path = record_path.with_suffix(record_path.suffix + ".commit.json")
    if not record_path.is_file() or not marker_path.is_file():
        raise ValueError("snapshot timing record is not committed")
    marker = json.loads(marker_path.read_text(encoding="utf-8"))
    if (
        sha256_file(record_path) != marker.get("sha256")
        or record_path.stat().st_size != int(marker.get("size_bytes", -1))
    ):
        raise ValueError("snapshot timing record checksum mismatch")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    seconds = float(record.get("checkpoint_log_io_seconds", -1.0))
    if (
        record.get("schema_version") != SCHEMA_VERSION
        or int(record.get("checkpoint_sequence", -1)) != int(ready["checkpoint_sequence"])
        or record.get("checkpoint_sha256") != ready["checkpoint_sha256"]
        or not math.isfinite(seconds)
        or seconds < 0.0
    ):
        raise ValueError("snapshot timing identity is invalid")
    return record


def verify_resume_snapshot_execution(
    persistence_directory: str | os.PathLike[str],
    expected_execution_fingerprint: str | None = None,
    *,
    current_config: RunConfig | None = None,
    current_environment: Mapping[str, Any] | None = None,
    policy: str = STRICT_RESUME_POLICY,
) -> dict[str, Any]:
    """Fail early when a persisted run can resume under the selected policy.

    Inputs are one run's persistence directory and the fingerprint resolved from
    an exact fingerprint, or current config/environment with a versioned policy.
    The newest fully timed snapshot manifest is verified and returned with its
    compatibility report. A corrupt identity or disallowed difference raises
    before dependent recovery artifacts are copied. The legacy positional form
    remains strict and cannot enable cross-GPU recovery.
    """

    remote = Path(persistence_directory)
    for ready_path in sorted(
        (remote / "snapshots").glob("checkpoint_*/snapshot.ready.json"),
        reverse=True,
    ):
        try:
            ready = json.loads(ready_path.read_text(encoding="utf-8"))
            try:
                _verified_snapshot_timing(ready_path, ready)
            except ValueError:
                continue
            manifest_path = ready_path.parent / "manifest.json"
            marker_path = manifest_path.with_suffix(manifest_path.suffix + ".commit.json")
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            if (
                sha256_file(manifest_path) != marker.get("sha256")
                or manifest_path.stat().st_size != int(marker.get("size_bytes", -1))
            ):
                continue
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            observed = manifest.get("execution_fingerprint")
            if current_config is None or current_environment is None:
                if expected_execution_fingerprint is None:
                    raise ValueError("strict snapshot verification requires an execution fingerprint")
                if observed != expected_execution_fingerprint:
                    raise ValueError(
                        "strict resume execution fingerprint differs; the physical GPU UUID "
                        "and complete saved execution environment must match"
                    )
                return manifest
            saved_values = manifest.get("resolved_config")
            saved_environment = manifest.get("current_environment", manifest.get("environment"))
            if not isinstance(saved_values, Mapping) or not isinstance(saved_environment, Mapping):
                raise ValueError("snapshot lacks saved config/environment compatibility evidence")
            saved_config = RunConfig(copy.deepcopy(saved_values))
            report = resume_compatibility_report(
                saved_config, saved_environment, current_config, current_environment, policy
            )
            recorded_current = manifest.get("current_execution_fingerprint", observed)
            if report["saved_execution_fingerprint"] != recorded_current:
                raise ValueError("snapshot saved execution fingerprint does not match its environment")
            manifest = dict(manifest)
            manifest["resume_compatibility"] = report
            return manifest
        except FileNotFoundError:
            continue
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
    raise RuntimeError("No verified compatible recovery snapshot found")


def restore_resume_snapshot(
    persistence_directory: str | os.PathLike[str],
    run_directory: str | os.PathLike[str],
) -> Path:
    """Restore the newest completely verified recovery snapshot.

    Only snapshots with a ready marker and matching file/checkpoint hashes are
    eligible. Restores into a fresh local directory. Never overwrites a run.
    """

    remote = Path(persistence_directory)
    local = Path(run_directory)

    if local.exists():
        raise FileExistsError(
            f"Recovery destination must not exist: {local}"
        )

    candidates = sorted(
        (remote / "snapshots").glob(
            "checkpoint_*/snapshot.ready.json"
        ),
        reverse=True,
    )

    selected = None

    for ready_path in candidates:
        try:
            ready = json.loads(ready_path.read_text(encoding="utf-8"))

            if (
                ready["schema_version"] != SCHEMA_VERSION
                or ready["run_id"] != local.name
            ):
                continue

            checkpoint = remote / "checkpoints" / ready["checkpoint_file"]
            checkpoint_commit = checkpoint.with_suffix(
                checkpoint.suffix + ".commit.json"
            )

            if not checkpoint.is_file() or not checkpoint_commit.is_file():
                continue

            commit = json.loads(
                checkpoint_commit.read_text(encoding="utf-8")
            )

            if (
                sha256_file(checkpoint) != ready["checkpoint_sha256"]
                or commit["sha256"] != ready["checkpoint_sha256"]
                or int(commit["checkpoint_sequence"])
                != int(ready["checkpoint_sequence"])
                or checkpoint.stat().st_size != int(commit["size_bytes"])
            ):
                continue

            timing_record = _verified_snapshot_timing(ready_path, ready)

            for name, identity in ready["files"].items():
                source = ready_path.parent / name
                marker = source.with_suffix(
                    source.suffix + ".commit.json"
                )

                if not source.is_file() or not marker.is_file():
                    raise ValueError("Missing recovery support file or marker")

                file_commit = json.loads(
                    marker.read_text(encoding="utf-8")
                )

                if (
                    sha256_file(source) != identity["sha256"]
                    or source.stat().st_size != identity["size_bytes"]
                    or file_commit["sha256"] != identity["sha256"]
                    or file_commit["size_bytes"] != identity["size_bytes"]
                ):
                    raise ValueError("Recovery support checksum mismatch")

            selected = (ready, ready_path.parent, checkpoint, timing_record)
            break

        except (OSError, ValueError, KeyError, TypeError):
            continue

    if selected is None:
        raise RuntimeError("No completely verified recovery snapshot found")

    ready, snapshot, checkpoint, timing_record = selected
    staging = local.with_name(f".{local.name}.restore.partial")

    if staging.exists():
        raise FileExistsError(
            f"Remove abandoned recovery staging manually: {staging}"
        )

    staging.mkdir(parents=True)

    try:
        checkpoint_directory = staging / "checkpoints"
        checkpoint_directory.mkdir()

        # Restore the plan's resumable/probe retention set, not merely the latest
        # boundary. This keeps the warm-up transition and post-warm probe points
        # available after an ephemeral-runtime loss without copying other runs.
        manifest = json.loads((snapshot / "manifest.json").read_text(encoding="utf-8"))
        training = manifest["resolved_config"]["scientific"]["training"]
        warmup_epochs = int(training["warmup_epochs"])
        total_epochs = int(training["total_epochs"])
        latest_sequence = int(ready["checkpoint_sequence"])
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        retained_sequences = {1, latest_sequence, max(1, latest_sequence - 1), warmup_epochs}
        retained_sequences.update(
            warmup_epochs + offset
            for offset in (1, 6, 12)
            if warmup_epochs + offset <= total_epochs
        )
        retained_sequences.update(
            int(record["checkpoint_sequence"])
            for record in payload["run_state"].get("target_attainment", {}).values()
            if record.get("reached") is True
        )
        retained_sequences = {
            sequence for sequence in retained_sequences
            if 1 <= sequence <= latest_sequence
        }

        for sequence in sorted(retained_sequences):
            source = remote / "checkpoints" / f"checkpoint_{sequence:06d}.pt"
            source_commit = source.with_suffix(source.suffix + ".commit.json")
            if not source.is_file() or not source_commit.is_file():
                raise ValueError(f"required retained checkpoint is missing: {sequence}")
            commit = json.loads(source_commit.read_text(encoding="utf-8"))
            if (
                int(commit.get("checkpoint_sequence", -1)) != sequence
                or sha256_file(source) != commit.get("sha256")
                or source.stat().st_size != int(commit.get("size_bytes", -1))
            ):
                raise ValueError(f"required retained checkpoint is corrupt: {sequence}")
            destination = checkpoint_directory / source.name
            shutil.copyfile(source, destination)
            shutil.copyfile(
                source_commit,
                destination.with_suffix(destination.suffix + ".commit.json"),
            )

        for name, identity in ready["files"].items():
            destination = staging / name
            shutil.copyfile(snapshot / name, destination)

            if sha256_file(destination) != identity["sha256"]:
                raise IOError(f"Restored file failed checksum: {name}")

        # Snapshot support files were copied before their own persistence cost
        # was known. The committed timing sidecar supplies that one missing
        # checkpoint/log I/O record exactly once during recovery.
        sequence = int(ready["checkpoint_sequence"])
        seconds = float(timing_record["checkpoint_log_io_seconds"])
        journal_path = staging / "checkpoint_journal.json"
        journal = json.loads(journal_path.read_text(encoding="utf-8"))
        matching = [
            row for row in journal["records"]
            if int(row["checkpoint_sequence"]) == sequence
        ]
        if len(matching) != 1:
            raise ValueError("restored checkpoint journal lacks one current record")
        matching[0]["io_seconds"] = seconds
        atomic_write_json(journal_path, journal)

        timing_path = staging / "timing.json"
        ledger = TimingLedger.from_dict(
            json.loads(timing_path.read_text(encoding="utf-8"))
        )
        if any(int(row["checkpoint_sequence"]) == sequence for row in ledger.checkpoint_io):
            raise ValueError("restored timing already contains current snapshot I/O")
        ledger.add("checkpoint_log_io", seconds)
        ledger.checkpoint_io.append(
            {
                "checkpoint_sequence": sequence,
                "sha256": ready["checkpoint_sha256"],
                "io_seconds": seconds,
                "verified_persistence_copy": str(checkpoint),
            }
        )
        atomic_write_json(timing_path, ledger.to_dict())

        local.parent.mkdir(parents=True, exist_ok=True)
        os.replace(staging, local)

    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    return local


def restore_or_verify_resume_snapshot(
    persistence_directory: str | os.PathLike[str],
    run_directory: str | os.PathLike[str],
) -> Path:
    """Restore a run once or verify that an earlier staged restore is identical.

    Inputs identify one persistence directory and intended local run directory.
    If the destination is absent, the newest complete snapshot is restored and
    returned. If it exists, the helper verifies the newest timed snapshot, every
    snapshot file hash, and the local latest committed checkpoint sequence/hash,
    then reuses it without writing. Missing, partial, advanced, or differing local
    state raises; valid files are never overwritten and no training/RNG occurs.
    """

    remote = Path(persistence_directory)
    local = Path(run_directory)
    if not local.exists():
        return restore_resume_snapshot(remote, local)
    if not local.is_dir():
        raise ValueError("staged recovery destination is not a directory")
    candidates = sorted(
        (remote / "snapshots").glob("checkpoint_*/snapshot.ready.json"), reverse=True
    )
    selected: tuple[Path, dict[str, Any], dict[str, Any]] | None = None
    for ready_path in candidates:
        try:
            ready = json.loads(ready_path.read_text(encoding="utf-8"))
            timing_record = _verified_snapshot_timing(ready_path, ready)
            selected = (ready_path, ready, timing_record)
            break
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            continue
    if selected is None:
        raise RuntimeError("No completely verified recovery snapshot found")
    ready_path, ready, timing_record = selected
    for name, identity in ready.get("files", {}).items():
        path = local / name
        if name == "timing.json":
            # Recovery legitimately adds the committed snapshot-copy duration to
            # the pre-I/O ledger, so its bytes differ from the snapshot source.
            # Validate the transformed ledger structurally instead of pretending
            # that the recovery accounting did not occur.
            TimingLedger.from_dict(json.loads(path.read_text(encoding="utf-8")))
            continue
        if name == "checkpoint_journal.json":
            journal = json.loads(path.read_text(encoding="utf-8"))
            matching = [
                row for row in journal.get("records", [])
                if int(row.get("checkpoint_sequence", -1))
                == int(ready["checkpoint_sequence"])
            ]
            if (
                len(matching) != 1
                or float(matching[0].get("io_seconds", -1.0))
                != float(timing_record["checkpoint_log_io_seconds"])
            ):
                raise ValueError("existing staged recovery journal timing differs")
            continue
        if (
            not path.is_file()
            or sha256_file(path) != identity.get("sha256")
            or path.stat().st_size != int(identity.get("size_bytes", -1))
        ):
            raise ValueError(f"existing staged recovery file differs: {name}")
    checkpoint = latest_committed_checkpoint(local / "checkpoints")
    commit = json.loads(
        checkpoint.with_suffix(checkpoint.suffix + ".commit.json").read_text(encoding="utf-8")
    )
    if (
        int(commit.get("checkpoint_sequence", -1)) != int(ready["checkpoint_sequence"])
        or commit.get("sha256") != ready.get("checkpoint_sha256")
        or sha256_file(checkpoint) != ready.get("checkpoint_sha256")
    ):
        raise ValueError("existing staged recovery checkpoint differs from verified snapshot")
    return local

