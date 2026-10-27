"""Small analysis and output helpers for Phase D v4 diagnostic engineering.

The helpers in this module summarize already-computed diagnostic observations.
They do not train models, select scientific conditions, or read evaluation data.
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np

from phase_d_v4_io import SCHEMA_VERSION, atomic_write_json


GRADIENT_DIAGNOSTIC_COLUMNS = (
    "schema_version", "run_id", "seed", "method", "checkpoint", "probe_id", "sample_id",
    "aggregation", "units", "count", "valid", "invalid", "true_near_zero",
    "predicted_near_zero", "cosine", "true_norm", "predicted_norm",
    "residual_norm", "relative_residual", "unfloored_relative_residual",
    "norm_ratio", "projection", "true_rms", "predicted_rms", "residual_rms",
    "diagnostic_floor", "controller_denominator", "controller_floor_active",
    "predictor_loss", "predictor_mse", "predictor_cosine", "predictor_mse_contribution",
    "predictor_cosine_contribution", "predictor_clip_coefficient",
    "exact_model_clip_coefficient", "injected_trunk_clip_coefficient",
    "parameter_stage", "parameter_cosine", "parameter_exact_norm",
    "parameter_injected_norm", "parameter_residual_norm",
    "parameter_relative_residual", "parameter_norm_ratio", "parameter_projection",
)

UPDATE_PROBE_COLUMNS = (
    "schema_version", "run_id", "seed", "method", "checkpoint", "probe_id", "alternative",
    "head_update", "normalization", "same_loss_before", "same_loss_after",
    "same_loss_change", "fixed_loss_before", "fixed_loss_after",
    "fixed_loss_change", "trunk_lr", "momentum", "weight_decay",
    "model_clip_coefficient", "status",
)

NORMALIZATION_PROBE_COLUMNS = (
    "schema_version", "run_id", "seed", "method", "checkpoint", "probe_id", "sample_source",
    "reference", "head_training", "target_units", "target_rms",
    "predictor_residual_rms", "predictor_relative_residual",
    "predictor_clip_coefficient", "bn_buffers_changed", "bn_change_l2",
    "target_change_norm", "target_change_relative", "status",
)

PROFILE_SAMPLE_COLUMNS = (
    "schema_version", "run_id", "seed", "method", "checkpoint_sha256",
    "profile_id", "component", "sample_index",
    "latency_ms", "device_type", "settling_iterations", "status",
)


def finite_summary(values: Sequence[float]) -> dict[str, float | int | None]:
    """Summarize finite scalar samples without silently dropping invalid data.

    Inputs are scalar observations in one declared unit. The returned mapping
    contains total/finite/invalid counts plus mean, median, and linear 10th/90th
    percentiles. Empty finite input produces null statistics and changes no state.
    """

    finite = [float(value) for value in values if math.isfinite(float(value))]
    result: dict[str, float | int | None] = {
        "count": len(values),
        "finite_count": len(finite),
        "invalid_count": len(values) - len(finite),
        "mean": None,
        "median": None,
        "p10": None,
        "p90": None,
    }
    if finite:
        result.update(
            mean=float(sum(finite) / len(finite)),
            median=float(median(finite)),
            p10=float(np.percentile(finite, 10)),
            p90=float(np.percentile(finite, 90)),
        )
    return result


def aggregate_per_example_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-example gradient rows with each sample weighted once.

    Inputs must be rows whose ``aggregation`` is ``per_example``. The output
    reports sample counts and mean/median/p10/p90 for each defined metric;
    undefined cosines remain counted as invalid rather than becoming zero.
    No input row is modified.
    """

    metrics = (
        "cosine", "true_norm", "predicted_norm", "residual_norm",
        "relative_residual", "unfloored_relative_residual", "norm_ratio", "projection",
    )
    result: dict[str, Any] = {"sample_count": len(rows)}
    for metric in metrics:
        values = [float(row[metric]) for row in rows if row.get(metric) is not None]
        summary = finite_summary(values)
        result[f"{metric}_defined_count"] = len(values)
        result[f"{metric}_mean"] = summary["mean"]
        result[f"{metric}_median"] = summary["median"]
        result[f"{metric}_p10"] = summary["p10"]
        result[f"{metric}_p90"] = summary["p90"]
    result["invalid_count"] = sum(bool(row.get("invalid")) for row in rows)
    result["true_near_zero_count"] = sum(bool(row.get("true_near_zero")) for row in rows)
    return result


def first_target_attainment(
    epoch_records: Sequence[Mapping[str, Any]], targets: Sequence[float]
) -> list[dict[str, Any]]:
    """Find the first observed validation endpoint meeting each fixed target.

    Records use accuracy fractions and must provide ``validation_accuracy`` plus
    an endpoint time in ``observation_time_seconds``. Warm-up records are eligible.
    The output preserves unattained targets with null attainment fields and the
    final observed horizon. Inputs are not reordered or mutated.
    """

    observations = [row for row in epoch_records if row.get("validation_accuracy") is not None]
    horizon = (
        float(observations[-1]["observation_time_seconds"])
        if observations and observations[-1].get("observation_time_seconds") is not None
        else None
    )
    results: list[dict[str, Any]] = []
    for target in targets:
        if not 0.0 <= float(target) <= 1.0:
            raise ValueError("targets must be accuracy fractions in [0, 1]")
        attained = next(
            (row for row in observations if float(row["validation_accuracy"]) >= float(target)),
            None,
        )
        results.append(
            {
                "target": float(target),
                "reached": attained is not None,
                "attainment_time_seconds": (
                    float(attained["observation_time_seconds"]) if attained is not None else None
                ),
                "stage": attained.get("stage") if attained is not None else None,
                "global_epoch": attained.get("global_epoch") if attained is not None else None,
                "validation_accuracy": (
                    float(attained["validation_accuracy"]) if attained is not None else None
                ),
                "observation_horizon_seconds": horizon,
            }
        )
    return results


def fixed_probe_indices(
    training_size: int,
    *,
    batch_size: int = 32,
    batches: int = 4,
    seed: int = 88001,
) -> list[list[int]]:
    """Select fixed training probe IDs with a private deterministic generator.

    Inputs specify the training population and number/size of disjoint batches;
    defaults implement the M3 four-by-32 contract with seed 88001. The returned
    integer lists do not consume NumPy's global RNG. Insufficient populations or
    nonpositive dimensions raise before any loader or training state is touched.
    """

    if training_size <= 0 or batch_size <= 0 or batches <= 0:
        raise ValueError("training_size, batch_size, and batches must be positive")
    count = batch_size * batches
    if count > training_size:
        raise ValueError("training population is too small for disjoint fixed probe batches")
    generator = np.random.default_rng(seed)
    selected = generator.permutation(training_size)[:count].tolist()
    return [
        [int(value) for value in selected[offset : offset + batch_size]]
        for offset in range(0, count, batch_size)
    ]


def build_predictor_replay_table(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build a paired M4 predictor-control table from completed run records.

    Inputs are one source-derived record per replay method with method, source
    mask/source run identity, counters, final accuracy, and cost labels. The
    returned rows preserve every condition and reject mixed sources, duplicate
    methods, or deployable labels. It performs no statistics or speed claim and
    does not mutate records.
    """

    required_methods = {
        "relative_replay_learned",
        "relative_replay_zero_shadow",
        "relative_replay_skip_shadow",
        "relative_replay_exact_trunk",
    }
    by_method: dict[str, Mapping[str, Any]] = {}
    source_keys: set[tuple[Any, Any]] = set()
    for record in records:
        method = str(record.get("method"))
        if method in by_method:
            raise ValueError(f"duplicate replay record for {method}")
        if record.get("retrospective") is not True or record.get("deployment_eligible") is not False:
            raise ValueError("source-derived replay records must be retrospective and nondeployable")
        by_method[method] = record
        source_keys.add((record.get("source_run_id"), record.get("source_mask_sha256")))
    if not required_methods.issubset(by_method):
        raise ValueError("paired replay table lacks a required causal control")
    if len(source_keys) != 1:
        raise ValueError("paired replay records must share one donor and mask")
    columns = (
        "method", "source_run_id", "source_mask_sha256", "retrospective",
        "deployment_eligible", "final_accuracy", "exact_trunk_backwards",
        "synthetic_backwards", "zero_backwards", "trunk_optimizer_steps",
        "skipped_slots",
        "head_optimizer_steps", "predictor_fits", "predictor_inferences",
        "conditional_cost", "dependency_inclusive_cost",
    )
    return [
        {column: by_method[method].get(column) for column in columns}
        for method in sorted(by_method)
    ]


def build_controller_schedule_table(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Build M5 budget/spacing rows without overstating matching properties.

    Inputs are completed schedule records containing identity, provenance, exact
    budgets, run-length histogram, and construction scope. The returned rows keep
    source-derived methods retrospective, fixed methods donor-independent, and
    preserve the explicit matching-scope label. Invalid eligibility or claims of
    complete temporal matching raise; no statistics or speed claims are produced.
    """

    source_methods = {
        "relative_periodic_total",
        "relative_periodic_epoch",
        "relative_gap_permutation_epoch",
    }
    fixed_methods = {"fixed_frontloaded", "fixed_uniform_third"}
    rows: list[dict[str, Any]] = []
    for record in records:
        method = str(record.get("method"))
        scope = str(record.get("matching_scope"))
        if "complete_temporal" in scope and "not_complete_temporal" not in scope:
            raise ValueError("M5 controls must not claim complete temporal matching")
        if method in source_methods:
            if record.get("retrospective") is not True or record.get("deployment_eligible") is not False:
                raise ValueError("source-derived controller controls must be retrospective")
            if not record.get("source_run_id") or not record.get("source_mask_sha256"):
                raise ValueError("source-derived controller control lacks donor provenance")
        elif method in fixed_methods:
            if record.get("retrospective") is not False or record.get("deployment_eligible") is not True:
                raise ValueError("fixed schedules must retain deployable identity labels")
            if record.get("source_run_id") is not None:
                raise ValueError("fixed schedules must not depend on donor outcomes")
        else:
            raise ValueError(f"unsupported M5 controller schedule: {method}")
        rows.append(dict(record))
    return rows


def average_mask_replicas_within_seed(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Average random-mask replicas within each classifier seed and method.

    Inputs are per-replica engineering records with classifier seed, method,
    replica, final accuracy, exact count, and moved-gap fraction. The output has
    one row per seed/method/source with replica identities and arithmetic means.
    Replicas are never counted as independent seeds; duplicates and mixed donor
    masks raise. No confidence intervals or M6 inference are computed.
    """

    groups: dict[tuple[Any, Any, Any], list[Mapping[str, Any]]] = {}
    for record in records:
        key = (record.get("classifier_seed"), record.get("method"), record.get("source_run_id"))
        groups.setdefault(key, []).append(record)
    output: list[dict[str, Any]] = []
    for (seed, method, source_run_id), group in sorted(groups.items(), key=lambda item: str(item[0])):
        replicas = [int(record["replicate"]) for record in group]
        if len(replicas) != len(set(replicas)):
            raise ValueError("duplicate mask replica within classifier seed")
        source_hashes = {record.get("source_mask_sha256") for record in group}
        if len(source_hashes) != 1:
            raise ValueError("mask replicas within a seed must share one donor mask")
        output.append(
            {
                "classifier_seed": seed,
                "method": method,
                "source_run_id": source_run_id,
                "source_mask_sha256": next(iter(source_hashes)),
                "replica_count": len(group),
                "replicates": sorted(replicas),
                "mean_final_accuracy": sum(float(record["final_accuracy"]) for record in group) / len(group),
                "mean_exact_count": sum(float(record["exact_count"]) for record in group) / len(group),
                "mean_moved_gap_fraction": sum(float(record["moved_gap_fraction"]) for record in group) / len(group),
                "replicas_are_independent_seeds": False,
            }
        )
    return output


def _write_csv_atomic(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    """Write a fixed-schema CSV by replacing its temporary sibling atomically.

    Inputs are a destination, JSON-like rows, and an ordered column dictionary.
    The destination is created or replaced; extra row keys are rejected by the
    CSV writer, and a failed write leaves no committed partial destination.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def write_diagnostic_outputs(
    directory: str | os.PathLike[str],
    *,
    gradient_rows: Sequence[Mapping[str, Any]],
    update_rows: Sequence[Mapping[str, Any]],
    normalization_rows: Sequence[Mapping[str, Any]],
) -> None:
    """Commit the three planned diagnostic CSV files under one local directory.

    Inputs must already follow the published schemas. The function returns
    nothing and atomically replaces each local CSV; it does not contact remote
    storage or change model state.
    """

    root = Path(directory)
    _write_csv_atomic(root / "gradient_diagnostics.csv", gradient_rows, GRADIENT_DIAGNOSTIC_COLUMNS)
    _write_csv_atomic(root / "update_probes.csv", update_rows, UPDATE_PROBE_COLUMNS)
    _write_csv_atomic(root / "normalization_probes.csv", normalization_rows, NORMALIZATION_PROBE_COLUMNS)


def write_profile_outputs(
    directory: str | os.PathLike[str],
    samples: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any],
) -> None:
    """Commit individual profile samples and their all-sample summary locally.

    Latencies are milliseconds and settling iterations are not rows. The
    function returns nothing, atomically replaces the two files, and performs
    no model computation or remote persistence.
    """

    root = Path(directory)
    _write_csv_atomic(root / "profile_samples.csv", samples, PROFILE_SAMPLE_COLUMNS)
    atomic_write_json(root / "profile_summary.json", dict(summary))


def diagnostic_identity_fields(
    run_id: str,
    checkpoint: str,
    probe_id: str,
    *,
    seed: int | None = None,
    method: str | None = None,
) -> dict[str, Any]:
    """Return common schema and identity fields for one diagnostic observation.

    Inputs are stable run/checkpoint/probe identifiers plus optional classifier
    seed and method. The output is a fresh mapping and has no side effects.
    """

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "seed": seed,
        "method": method,
        "checkpoint": checkpoint,
        "probe_id": probe_id,
    }


__all__ = [
    "GRADIENT_DIAGNOSTIC_COLUMNS", "NORMALIZATION_PROBE_COLUMNS",
    "PROFILE_SAMPLE_COLUMNS", "UPDATE_PROBE_COLUMNS", "aggregate_per_example_rows",
    "diagnostic_identity_fields", "finite_summary", "first_target_attainment",
    "fixed_probe_indices", "build_predictor_replay_table", "build_controller_schedule_table",
    "average_mask_replicas_within_seed", "write_diagnostic_outputs",
    "write_profile_outputs",
]
