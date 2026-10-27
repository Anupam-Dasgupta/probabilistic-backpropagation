"""Validate and analyze Phase D v4 results without training or tuning models.

This M6 module reads committed run artifacts, enforces comparison eligibility,
computes seed-paired accuracy and time summaries, builds target/Pareto tables,
and optionally draws descriptive curves. It never loads CIFAR-10, changes a
checkpoint, selects a stopping rule, or launches an experiment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import median
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from phase_d_v4_analysis import first_target_attainment
from phase_d_v4_io import (
    EPOCH_COLUMNS,
    SCHEMA_VERSION,
    atomic_write_json,
    canonical_json,
    latest_committed_checkpoint,
    load_checkpoint,
)


T_CRITICAL_95 = {3: 4.302653, 5: 2.776445}
T_CRITICAL_97_5 = {5: 3.495406}
REQUIRED_RUN_FILES = (
    "manifest.json", "epochs.csv", "summary.json", "timing.json",
    "masks.npz", "mask_manifest.json", "errors.jsonl",
)


def _read_json(path: Path) -> dict[str, Any]:
    """Read one JSON object and return it without modifying the source file.

    The input is a local path. The output is a dictionary; missing files,
    malformed JSON, or non-object roots raise and no filesystem state changes.
    """

    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Read one UTF-8 CSV and return its header plus detached string rows.

    The input file must have a header. Empty or duplicate column names fail;
    returned mappings preserve file order and reading has no side effects.
    """

    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = list(reader.fieldnames or [])
        if not columns or len(columns) != len(set(columns)) or any(not item for item in columns):
            raise ValueError(f"CSV has an invalid header: {path}")
        return columns, [dict(row) for row in reader]


def _as_float(value: Any, label: str) -> float:
    """Convert one finite scalar to float or raise with its scientific label.

    Blank, null, NaN, infinity, and nonnumeric inputs fail. The returned value is
    detached from the input and this helper has no side effects.
    """

    if value is None or value == "":
        raise ValueError(f"{label} is unavailable")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def _as_int(value: Any, label: str) -> int:
    """Convert one exact integer field to int or report a malformed identity.

    Boolean, fractional, blank, and nonnumeric inputs fail. The returned integer
    is detached from the source value and this helper changes no state.
    """

    if isinstance(value, bool) or value is None or value == "":
        raise ValueError(f"{label} must be an integer")
    number = float(value)
    if not math.isfinite(number) or int(number) != number:
        raise ValueError(f"{label} must be an integer")
    return int(number)


def reconcile_timing(timing: Mapping[str, Any], *, tolerance_seconds: float = 1e-9) -> dict[str, Any]:
    """Recompute the non-overlapping M6 timing ledger and verify saved totals.

    Input durations are seconds. The output distinguishes learning, evaluated,
    workflow, post-warm-up, and post-warm-up training totals plus completeness.
    Negative/nonfinite values, hidden residuals beyond tolerance, or malformed
    ledgers raise; the supplied mapping is never changed.
    """

    components = timing.get("components")
    if not isinstance(components, Mapping):
        raise ValueError("timing ledger lacks components")
    clean: dict[str, float] = {}
    for name, value in components.items():
        seconds = _as_float(value, f"timing component {name}")
        if seconds < 0:
            raise ValueError(f"timing component {name} must be nonnegative")
        clean[str(name)] = seconds
    get = lambda name: clean.get(name, 0.0)
    learning = get("warmup_training") + get("scale_estimation") + get("calibration") + get("postwarm_training")
    evaluated = learning + get("warmup_validation") + get("postwarm_validation")
    workflow = get("setup") + evaluated + get("checkpoint_log_io") + get("finalization") + get("diagnostics")
    computed = {
        "learning_seconds": learning,
        "evaluated_seconds": evaluated,
        "workflow_seconds": workflow,
        "postwarm_seconds": get("postwarm_training") + get("postwarm_validation"),
        "postwarm_training_seconds": get("postwarm_training"),
    }
    saved = timing.get("totals")
    if not isinstance(saved, Mapping):
        raise ValueError("timing ledger lacks saved totals")
    residuals: dict[str, float] = {}
    for name, expected in computed.items():
        observed = _as_float(saved.get(name), f"saved timing total {name}")
        residuals[name] = observed - expected
        if abs(observed - expected) > tolerance_seconds:
            raise ValueError(f"timing total {name} has unexplained residual {observed - expected}")
    incomplete = timing.get("incomplete", [])
    if not isinstance(incomplete, list):
        raise ValueError("timing incomplete flags must be a list")
    return {
        "components": clean,
        "totals": computed,
        "residuals_seconds": residuals,
        "complete": len(incomplete) == 0,
        "incomplete_reasons": list(incomplete),
    }


def epoch_observation_times(
    epoch_rows: Sequence[Mapping[str, Any]], timing: Mapping[str, Any]
) -> list[dict[str, Any]]:
    """Attach cumulative endpoint times without charging future checkpoint I/O.

    Inputs are committed epoch rows and a timing ledger. Returned rows are fresh
    mappings with endpoint seconds from workflow start. Scale is charged before
    the first validation, calibration before the first post-warm validation, and
    each checkpoint write only before later endpoints. Missing epochs fail.
    """

    reconciled = reconcile_timing(timing)
    ordered = sorted(epoch_rows, key=lambda row: _as_int(row.get("global_epoch"), "global_epoch"))
    epochs = [_as_int(row.get("global_epoch"), "global_epoch") for row in ordered]
    if epochs != list(range(len(ordered))):
        raise ValueError("epoch rows must be complete and contiguous from zero")
    io_by_sequence: dict[int, float] = {}
    for item in timing.get("checkpoint_io", []):
        sequence = _as_int(item.get("checkpoint_sequence"), "checkpoint I/O sequence")
        if sequence in io_by_sequence:
            raise ValueError("duplicate checkpoint I/O timing sequence")
        io_by_sequence[sequence] = _as_float(item.get("io_seconds"), "checkpoint I/O seconds")
    elapsed = reconciled["components"].get("setup", 0.0)
    output: list[dict[str, Any]] = []
    calibration_charged = False
    for index, row in enumerate(ordered):
        if index == 0:
            elapsed += reconciled["components"].get("scale_estimation", 0.0)
        if row.get("stage") == "postwarm" and not calibration_charged:
            elapsed += reconciled["components"].get("calibration", 0.0)
            calibration_charged = True
        elapsed += _as_float(row.get("training_seconds"), "epoch training seconds")
        elapsed += _as_float(row.get("validation_seconds"), "epoch validation seconds")
        enriched = dict(row)
        enriched["observation_time_seconds"] = elapsed
        if row.get("observation_time_seconds") not in (None, ""):
            saved_endpoint = _as_float(
                row.get("observation_time_seconds"), "saved observation endpoint"
            )
            if abs(saved_endpoint - elapsed) > 1e-9:
                raise ValueError("saved observation endpoint does not reconcile with timing ledger")
        output.append(enriched)
        sequence = _as_int(row.get("checkpoint_sequence"), "checkpoint_sequence")
        elapsed += io_by_sequence.get(sequence, 0.0)
    return output


def hardware_identity(environment: Mapping[str, Any], execution: Mapping[str, Any]) -> str:
    """Return a canonical hardware/software identity for controlled timing pairs.

    Inputs are saved environment and execution mappings. The returned JSON string
    covers device, runtime, precision, loader, TF32, and determinism fields; it is
    stable across dictionary order and causes no device query or state change.
    """

    environment_keys = (
        "python", "torch", "torchvision", "numpy", "cuda_runtime", "cudnn",
        "device_type", "device_name", "device_uuid", "total_memory_bytes",
    )
    execution_keys = (
        "precision", "amp", "workers", "pin_memory", "matmul_tf32",
        "cudnn_tf32", "cudnn_benchmark", "cudnn_deterministic",
    )
    return canonical_json(
        {
            "environment": {key: environment.get(key) for key in environment_keys},
            "execution": {key: execution.get(key) for key in execution_keys},
        }
    )


def validate_run_directory(
    run_directory: str | Path, *, require_complete: bool = True
) -> dict[str, Any]:
    """Validate one committed scientific run and return an analysis-ready record.

    The directory is read only. Required schemas, run/seed/method/checkpoint
    identities, contiguous epochs, accuracy fractions, timing reconciliation, and
    eligibility labels are checked. Partial runs are accepted only when requested
    and remain ineligible for completed comparisons.
    """

    root = Path(run_directory)
    missing = [name for name in REQUIRED_RUN_FILES if not (root / name).exists()]
    if missing:
        raise ValueError(f"run is missing required artifacts: {missing}")
    manifest = _read_json(root / "manifest.json")
    summary = _read_json(root / "summary.json")
    timing = _read_json(root / "timing.json")
    if manifest.get("schema_version") != SCHEMA_VERSION or summary.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unknown run schema")
    completed = summary.get("completed") is True
    if require_complete and not completed:
        raise ValueError("run is partial and cannot enter a completed comparison")
    columns, rows = _read_csv(root / "epochs.csv")
    if columns != list(EPOCH_COLUMNS):
        raise ValueError("epochs.csv does not match the published schema")
    config = manifest.get("resolved_config")
    if not isinstance(config, Mapping):
        raise ValueError("manifest lacks resolved config")
    identity = config.get("scientific", {}).get("identity", {})
    run_id = str(manifest.get("run_id"))
    seed = _as_int(identity.get("classifier_seed"), "classifier seed")
    method = str(identity.get("method"))
    if (
        summary.get("run_id") != run_id
        or _as_int(summary.get("seed"), "summary seed") != seed
        or summary.get("method") != method
    ):
        raise ValueError("summary run/seed/method identity mismatch")
    for row in rows:
        if row.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("epoch row has unknown schema")
        if row.get("run_id") != run_id or _as_int(row.get("seed"), "epoch seed") != seed or row.get("method") != method:
            raise ValueError("epoch row run/seed/method identity mismatch")
        checkpoint = str(row.get("checkpoint_sha256", ""))
        if len(checkpoint) != 64:
            raise ValueError("epoch row lacks a checkpoint identity")
        int(checkpoint, 16)
        accuracy = _as_float(row.get("validation_accuracy"), "validation accuracy")
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError("validation accuracy must be a fraction in [0,1]")
    enriched = epoch_observation_times(rows, timing)
    if completed:
        expected = _as_int(config.get("scientific", {}).get("training", {}).get("total_epochs"), "total epochs")
        if len(enriched) != expected:
            raise ValueError("completed run is missing epochs")
    reconciled = reconcile_timing(timing)
    environment = manifest.get("environment")
    if not isinstance(environment, Mapping):
        raise ValueError("manifest lacks environment identity")
    errors = (root / "errors.jsonl").read_text(encoding="utf-8").strip()
    return {
        "schema_version": SCHEMA_VERSION,
        "path": str(root.resolve()),
        "run_id": run_id,
        "seed": seed,
        "method": method,
        "normalization": identity.get("normalization"),
        "scientific_fingerprint": manifest.get("scientific_fingerprint"),
        "execution_fingerprint": manifest.get("execution_fingerprint"),
        "completed": completed,
        "resumed": bool(manifest.get("resumed")),
        "retrospective": bool(identity.get("retrospective")),
        "deployment_eligible": bool(identity.get("deployment_eligible")),
        "mask_replicate": identity.get("mask_replicate"),
        "epochs": enriched,
        "final_accuracy": _as_float(enriched[-1]["validation_accuracy"], "final accuracy") if enriched else None,
        "timing": reconciled,
        "environment": dict(environment),
        "hardware_key": hardware_identity(environment, config.get("execution", {})),
        "error_log_empty": not bool(errors),
        "comparison_eligible": completed and reconciled["complete"] and not bool(errors),
    }


def _paired_values(
    records: Sequence[Mapping[str, Any]], reference_method: str, candidate_method: str
) -> list[tuple[int, Mapping[str, Any], Mapping[str, Any]]]:
    """Return complete one-to-one seed pairs or reject pseudoreplication.

    Inputs contain validated run summaries. The output is sorted by classifier
    seed and contains one record for each requested method. Duplicate or unequal
    seeds, incomplete/error runs, and mismatched horizons fail instead of silently
    shortening the comparison. Input records are not modified.
    """

    grouped: dict[str, dict[int, Mapping[str, Any]]] = {reference_method: {}, candidate_method: {}}
    for record in records:
        method = str(record.get("method"))
        if method not in grouped:
            continue
        seed = _as_int(record.get("seed"), "paired seed")
        if seed in grouped[method]:
            raise ValueError(f"duplicate seed {seed} for {method}; average replicas within seed first")
        if record.get("comparison_eligible") is not True:
            raise ValueError(f"incomplete or invalid run cannot be paired: {method} seed {seed}")
        grouped[method][seed] = record
    if not grouped[reference_method] or set(grouped[reference_method]) != set(grouped[candidate_method]):
        raise ValueError("paired methods must have identical nonempty classifier-seed sets")
    pairs = []
    for seed in sorted(grouped[reference_method]):
        reference = grouped[reference_method][seed]
        candidate = grouped[candidate_method][seed]
        if len(reference.get("epochs", [])) != len(candidate.get("epochs", [])):
            raise ValueError("paired runs have mismatched horizons")
        pairs.append((seed, reference, candidate))
    return pairs


def average_replica_records_within_seed(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Average repeated random masks within a classifier seed before inference.

    Inputs are validated run records. Unique method/seed rows pass through. A
    repeated group is accepted only for gap-permutation records with distinct,
    explicit replica IDs, common hardware/horizon, and comparison eligibility.
    Final accuracy, epoch accuracies/times, and reconciled timing totals are
    averaged arithmetically into one seed-level row. Replicas never increase the
    classifier-seed sample count; source records remain unchanged.
    """

    groups: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for record in records:
        key = (_as_int(record.get("seed"), "replica seed"), str(record.get("method")))
        groups.setdefault(key, []).append(record)
    output: list[dict[str, Any]] = []
    for (_seed, method), group in groups.items():
        if len(group) == 1:
            output.append(dict(group[0]))
            continue
        if method != "relative_gap_permutation_epoch":
            raise ValueError("duplicate non-random method within classifier seed")
        replicas = [record.get("mask_replicate") for record in group]
        if any(replica is None for replica in replicas) or len(replicas) != len(set(replicas)):
            raise ValueError("random-mask replicas must have distinct explicit identities")
        if any(record.get("comparison_eligible") is not True for record in group):
            raise ValueError("incomplete random-mask replica cannot enter seed average")
        if len({record.get("hardware_key") for record in group}) != 1:
            raise ValueError("random-mask replicas have mismatched hardware identity")
        horizon = len(group[0].get("epochs", []))
        if any(len(record.get("epochs", [])) != horizon for record in group):
            raise ValueError("random-mask replicas have mismatched horizons")
        averaged = dict(group[0])
        averaged["run_id"] = "+".join(sorted(str(record.get("run_id")) for record in group))
        averaged["replica_run_ids"] = sorted(str(record.get("run_id")) for record in group)
        averaged["replica_scientific_fingerprints"] = sorted(
            str(record.get("scientific_fingerprint")) for record in group
        )
        averaged["scientific_fingerprint"] = None
        averaged["mask_replicate"] = None
        averaged["replica_count"] = len(group)
        averaged["replica_ids"] = sorted(int(replica) for replica in replicas)
        averaged["replicas_are_independent_seeds"] = False
        averaged["final_accuracy"] = sum(float(record["final_accuracy"]) for record in group) / len(group)
        averaged["epochs"] = []
        for epoch_index in range(horizon):
            row = dict(group[0]["epochs"][epoch_index])
            for field in ("validation_accuracy", "validation_ce", "observation_time_seconds"):
                row[field] = sum(float(record["epochs"][epoch_index][field]) for record in group) / len(group)
            averaged["epochs"].append(row)
        timing = dict(group[0]["timing"])
        totals = dict(timing["totals"])
        for name in totals:
            totals[name] = sum(float(record["timing"]["totals"][name]) for record in group) / len(group)
        timing["totals"] = totals
        averaged["timing"] = timing
        output.append(averaged)
    return output


def _paired_interval(values: Sequence[float], *, family_adjusted: bool = False) -> dict[str, Any]:
    """Summarize independent classifier-seed differences with a Student-t CI.

    One value returns ``insufficient_sample`` with no SD or interval. Supported
    inferential sample counts are three and five at 95%; the planned 97.5%
    family interval supports five only. Other counts raise rather than using an
    undocumented approximation. Values must be finite and inputs are read only.
    """

    clean = [_as_float(value, "paired effect") for value in values]
    mean = sum(clean) / len(clean) if clean else None
    if len(clean) < 2:
        return {
            "status": "insufficient_sample",
            "n_classifier_seeds": len(clean),
            "mean": mean,
            "sample_sd": None,
            "ci_low": None,
            "ci_high": None,
            "confidence_level": None,
        }
    table = T_CRITICAL_97_5 if family_adjusted else T_CRITICAL_95
    if len(clean) not in table:
        raise ValueError(f"no predeclared t critical value for n={len(clean)}")
    sample_sd = math.sqrt(sum((value - mean) ** 2 for value in clean) / (len(clean) - 1))
    half_width = table[len(clean)] * sample_sd / math.sqrt(len(clean))
    return {
        "status": "ok_assumption_dependent_student_t",
        "n_classifier_seeds": len(clean),
        "mean": mean,
        "sample_sd": sample_sd,
        "ci_low": mean - half_width,
        "ci_high": mean + half_width,
        "confidence_level": 0.975 if family_adjusted else 0.95,
    }


def paired_accuracy_difference(
    records: Sequence[Mapping[str, Any]],
    reference_method: str,
    candidate_method: str,
    *,
    family_adjusted: bool = False,
) -> dict[str, Any]:
    """Compute final-accuracy candidate-minus-reference effects in percentage points.

    Inputs are validated completed records and two method identifiers. Every seed
    is retained, fractions are multiplied by 100 exactly once, and replicas must
    already be averaged within seed. The output includes individual effects and
    an explicit insufficient-sample status; no superiority conclusion is drawn.
    """

    pairs = _paired_values(records, reference_method, candidate_method)
    effects = [
        100.0 * (
            _as_float(candidate.get("final_accuracy"), "candidate accuracy")
            - _as_float(reference.get("final_accuracy"), "reference accuracy")
        )
        for _seed, reference, candidate in pairs
    ]
    reference_normalizations = {str(reference.get("normalization")) for _seed, reference, _candidate in pairs}
    candidate_normalizations = {str(candidate.get("normalization")) for _seed, _reference, candidate in pairs}
    if len(reference_normalizations) != 1 or len(candidate_normalizations) != 1:
        raise ValueError("each paired method must retain one explicit normalization identity")
    return {
        "schema_version": SCHEMA_VERSION,
        "reference_method": reference_method,
        "candidate_method": candidate_method,
        "reference_normalization": next(iter(reference_normalizations)),
        "candidate_normalization": next(iter(candidate_normalizations)),
        "horizon_epochs": len(pairs[0][1].get("epochs", [])),
        "unit": "percentage_points",
        "per_seed": [
            {"seed": seed, "difference_pp": effect}
            for (seed, _reference, _candidate), effect in zip(pairs, effects, strict=True)
        ],
        **_paired_interval(effects, family_adjusted=family_adjusted),
    }


def paired_time_ratio(
    records: Sequence[Mapping[str, Any]],
    reference_method: str,
    candidate_method: str,
    *,
    total: str = "workflow_seconds",
) -> dict[str, Any]:
    """Compute paired reference/candidate time ratios through log-space inference.

    Inputs are validated clean, uninterrupted runs on exactly matching hardware.
    ``total`` selects a reconciled seconds field. The output reports every ratio,
    geometric mean, and exponentiated CI; one seed is explicitly insufficient.
    Resumed, nonpositive, or hardware-mismatched pairs fail.
    """

    pairs = _paired_values(records, reference_method, candidate_method)
    logs: list[float] = []
    rows: list[dict[str, Any]] = []
    for seed, reference, candidate in pairs:
        if reference.get("resumed") or candidate.get("resumed"):
            raise ValueError("resumed runs cannot support clean efficiency comparisons")
        if reference.get("hardware_key") != candidate.get("hardware_key"):
            raise ValueError("timed pair has mismatched hardware/software identity")
        reference_seconds = _as_float(reference["timing"]["totals"].get(total), f"reference {total}")
        candidate_seconds = _as_float(candidate["timing"]["totals"].get(total), f"candidate {total}")
        if reference_seconds <= 0 or candidate_seconds <= 0:
            raise ValueError("timed comparisons require positive elapsed seconds")
        ratio = reference_seconds / candidate_seconds
        logs.append(math.log(ratio))
        rows.append(
            {
                "seed": seed,
                "reference_seconds": reference_seconds,
                "candidate_seconds": candidate_seconds,
                "ratio": ratio,
            }
        )
    interval = _paired_interval(logs)
    reference_normalizations = {str(reference.get("normalization")) for _seed, reference, _candidate in pairs}
    candidate_normalizations = {str(candidate.get("normalization")) for _seed, _reference, candidate in pairs}
    if len(reference_normalizations) != 1 or len(candidate_normalizations) != 1:
        raise ValueError("each timed method must retain one explicit normalization identity")
    return {
        "schema_version": SCHEMA_VERSION,
        "reference_method": reference_method,
        "candidate_method": candidate_method,
        "reference_normalization": next(iter(reference_normalizations)),
        "candidate_normalization": next(iter(candidate_normalizations)),
        "horizon_epochs": len(pairs[0][1].get("epochs", [])),
        "time_total": total,
        "unit": "reference_time_divided_by_candidate_time",
        "per_seed": rows,
        "status": interval["status"],
        "n_classifier_seeds": interval["n_classifier_seeds"],
        "geometric_mean_ratio": None if interval["mean"] is None else math.exp(interval["mean"]),
        "log_sample_sd": interval["sample_sd"],
        "ratio_ci_low": None if interval["ci_low"] is None else math.exp(interval["ci_low"]),
        "ratio_ci_high": None if interval["ci_high"] is None else math.exp(interval["ci_high"]),
        "confidence_level": interval["confidence_level"],
    }


def target_attainment_rows(
    records: Sequence[Mapping[str, Any]], targets: Sequence[float]
) -> list[dict[str, Any]]:
    """Build identified first-attainment rows including warm-up and censoring.

    Inputs are validated records and fixed accuracy fractions. Returned rows have
    run/seed/method/checkpoint identity, first observed endpoint, and observation
    horizon for unattained targets. There is no interpolation or substitution of
    final duration for a failed target, and source records remain unchanged.
    """

    output: list[dict[str, Any]] = []
    for record in records:
        for result in first_target_attainment(record.get("epochs", []), targets):
            checkpoint = None
            if result["reached"]:
                epoch = int(result["global_epoch"])
                checkpoint = record["epochs"][epoch]["checkpoint_sha256"]
            output.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "run_id": record.get("run_id"),
                    "seed": record.get("seed"),
                    "method": record.get("method"),
                    "checkpoint_sha256": checkpoint,
                    **result,
                }
            )
    return output


def classify_costs(
    record: Mapping[str, Any], provenance: Mapping[str, Any] | None = None
) -> dict[str, Any]:
    """Separate measured, attributed, conditional, and dependency-inclusive costs.

    Input is one validated record and optional replay/schedule provenance. The
    returned labels never convert update counts into computation or elapsed time.
    Retrospective conditional execution stays nondeployable. Missing dependency
    totals are reported as unavailable instead of inferred from counts.
    """

    measured = dict(record["timing"]["totals"])
    costs: dict[str, Any] = {
        "run_id": record.get("run_id"),
        "method": record.get("method"),
        "measured_continuous": measured,
        "attributed_standalone": measured,
        "replay_conditional": None,
        "dependency_inclusive": None,
        "retrospective": bool(record.get("retrospective")),
        "deployment_eligible": bool(record.get("deployment_eligible")),
        "counts_are_flops": False,
    }
    if provenance:
        cost_records = provenance.get("cost_records", {})
        conditional = cost_records.get("conditional_execution")
        if isinstance(conditional, Mapping):
            costs["replay_conditional"] = conditional.get("timing_ledger", {}).get("totals")
        dependency = cost_records.get("dependency_inclusive")
        if isinstance(dependency, Mapping):
            source = dependency.get("source_timing_ledger", {})
            execution = dependency.get("schedule_timing_ledger", {})
            if isinstance(source, Mapping) and isinstance(execution, Mapping):
                source_totals = source.get("totals")
                execution_totals = execution.get("totals")
                if isinstance(source_totals, Mapping) and isinstance(execution_totals, Mapping):
                    costs["dependency_inclusive"] = {
                        "status": "components_available_shared_warmup_requires_unique_phase_sum",
                        "source": dict(source_totals),
                        "execution": dict(execution_totals),
                    }
    return costs


def dependency_inclusive_cost(
    source_timing: Mapping[str, Any],
    execution_timing: Mapping[str, Any],
    *,
    shared_warmup_checkpoint_sequences: Sequence[int],
) -> dict[str, Any]:
    """Sum donor and conditional execution costs while counting shared warm-up once.

    Inputs are complete source/execution timing ledgers plus execution checkpoint
    sequences belonging to the reusable warm-up. Seconds are computed from
    reconciled measured components: both workflows are summed, then execution's
    warm-up training, scale, validation, and named warm-up checkpoint I/O are
    removed once. Setup, donor calibration/post-warm work, replay post-warm work,
    finalization, and other I/O remain charged. Missing sequences or incomplete
    ledgers fail; no update count is interpreted as computation.
    """

    source = reconcile_timing(source_timing)
    execution = reconcile_timing(execution_timing)
    if not source["complete"] or not execution["complete"]:
        raise ValueError("dependency-inclusive cost requires complete timing ledgers")
    requested = {int(value) for value in shared_warmup_checkpoint_sequences}
    if not requested:
        raise ValueError("shared warm-up checkpoint sequences must be explicit")
    io_records = execution_timing.get("checkpoint_io", [])
    io_by_sequence = {
        _as_int(item.get("checkpoint_sequence"), "checkpoint sequence"):
        _as_float(item.get("io_seconds"), "checkpoint I/O seconds")
        for item in io_records
    }
    if not requested.issubset(io_by_sequence):
        raise ValueError("execution timing lacks a declared warm-up checkpoint I/O record")
    components = execution["components"]
    shared_seconds = (
        components.get("warmup_training", 0.0)
        + components.get("scale_estimation", 0.0)
        + components.get("warmup_validation", 0.0)
        + sum(io_by_sequence[sequence] for sequence in requested)
    )
    total = (
        source["totals"]["workflow_seconds"]
        + execution["totals"]["workflow_seconds"]
        - shared_seconds
    )
    if total < 0:
        raise ValueError("dependency-inclusive cost became negative")
    return {
        "status": "measured_components_unique_shared_warmup_counted_once",
        "dependency_inclusive_workflow_seconds": total,
        "source_workflow_seconds": source["totals"]["workflow_seconds"],
        "conditional_execution_workflow_seconds": execution["totals"]["workflow_seconds"],
        "shared_warmup_subtracted_seconds": shared_seconds,
        "counts_are_flops": False,
    }


def deployment_pareto(
    records: Sequence[Mapping[str, Any]], *, time_total: str = "workflow_seconds"
) -> list[dict[str, Any]]:
    """Return nondominated deployable accuracy-time points only.

    Inputs are validated run summaries. Retrospective, conditional-mask, partial,
    resumed, or timing-incomplete records are excluded. A point is dominated when
    another is at least as accurate and no slower with one strict improvement.
    The returned rows make no speed or causal conclusion.
    """

    candidates = []
    for record in records:
        if not record.get("deployment_eligible") or record.get("retrospective"):
            continue
        if not record.get("comparison_eligible") or record.get("resumed"):
            continue
        candidates.append(
            {
                "run_id": record.get("run_id"),
                "seed": record.get("seed"),
                "method": record.get("method"),
                "normalization": record.get("normalization"),
                "accuracy": _as_float(record.get("final_accuracy"), "Pareto accuracy"),
                "time_seconds": _as_float(record["timing"]["totals"].get(time_total), "Pareto time"),
                "time_total": time_total,
            }
        )
    return [
        point
        for point in candidates
        if not any(
            other is not point
            and other["accuracy"] >= point["accuracy"]
            and other["time_seconds"] <= point["time_seconds"]
            and (other["accuracy"] > point["accuracy"] or other["time_seconds"] < point["time_seconds"])
            for other in candidates
        )
    ]


def hardware_drift_check(
    pre_latencies_ms: Sequence[float], post_latencies_ms: Sequence[float]
) -> dict[str, Any]:
    """Compare identical pre/post exact profiles and flag median drift above 10%.

    Inputs are finite positive latency samples in milliseconds. The output gives
    both medians, post/pre ratio, absolute fractional drift, and the fixed flag.
    Empty/nonpositive samples fail; no slow run is discarded or modified.
    """

    pre = [_as_float(value, "pre-block latency") for value in pre_latencies_ms]
    post = [_as_float(value, "post-block latency") for value in post_latencies_ms]
    if not pre or not post or any(value <= 0 for value in pre + post):
        raise ValueError("hardware drift profiles require positive samples")
    pre_median = float(median(pre))
    post_median = float(median(post))
    ratio = post_median / pre_median
    drift = abs(ratio - 1.0)
    return {
        "pre_median_ms": pre_median,
        "post_median_ms": post_median,
        "post_pre_ratio": ratio,
        "absolute_fractional_drift": drift,
        "threshold_fraction": 0.10,
        "flagged": drift > 0.10,
        "action": "retain_and_repeat_entire_block_if_authorized" if drift > 0.10 else "accepted",
    }


def make_run_order_manifest(
    study_id: str,
    donor_runs: Sequence[str],
    measured_runs: Sequence[str],
    *,
    repetition: int = 0,
    base_order: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Create the donor-first deterministic randomized timing-block order.

    Inputs name unique donor and measured run IDs. Repetition zero uses a private
    NumPy generator seeded 73001. Later repetitions require the saved base order
    and apply a cyclic rotation, preventing outcome-dependent reordering. The
    output is a manifest mapping; global RNG and files are unchanged.
    """

    if not study_id or repetition < 0:
        raise ValueError("study_id must be nonempty and repetition nonnegative")
    donors = list(donor_runs)
    measured = list(measured_runs)
    if len(donors + measured) != len(set(donors + measured)):
        raise ValueError("run-order identities must be unique")
    if repetition == 0:
        if base_order is not None:
            raise ValueError("initial randomized block must not receive a base order")
        generator = np.random.default_rng(73001)
        ordered = [measured[index] for index in generator.permutation(len(measured)).tolist()]
        construction = "private_rng_randomized"
    else:
        if base_order is None or set(base_order) != set(measured) or len(base_order) != len(measured):
            raise ValueError("repeated block requires the complete saved base order")
        shift = repetition % max(1, len(measured))
        ordered = list(base_order[shift:]) + list(base_order[:shift])
        construction = "cyclic_rotation_of_saved_base_order"
    return {
        "schema_version": SCHEMA_VERSION,
        "study_id": study_id,
        "repetition": repetition,
        "ordering_seed_namespace": 73001 + repetition,
        "construction": construction,
        "donor_phase": donors,
        "pre_block_profile": "plain_exact_identical_profile",
        "measured_block": ordered,
        "post_block_profile": "plain_exact_identical_profile",
        "hardware_drift_threshold_fraction": 0.10,
        "automatic_retry": False,
    }


def verify_feedback_mask_equality(
    donor_directory: str | Path, measured_directory: str | Path
) -> dict[str, Any]:
    """Verify that donor and measured feedback runs produced the identical mask.

    Inputs are two completed relative-feedback run directories in the same
    environment. The function validates both runs, compares exported exact-mask
    hash/boundaries plus per-epoch data-order hashes, and returns an identity
    report. Any mismatch raises before mask-derived comparisons may proceed; files
    and checkpoints remain unchanged.
    """

    donor = validate_run_directory(donor_directory)
    measured = validate_run_directory(measured_directory)
    if donor["method"] != "relative_feedback" or measured["method"] != "relative_feedback":
        raise ValueError("feedback mask equality requires two relative_feedback runs")
    if donor["seed"] != measured["seed"] or donor["hardware_key"] != measured["hardware_key"]:
        raise ValueError("feedback mask equality requires matching seed and environment")
    donor_manifest = _read_json(Path(donor_directory) / "mask_manifest.json")
    measured_manifest = _read_json(Path(measured_directory) / "mask_manifest.json")
    donor_export = donor_manifest.get("replay_export", {})
    measured_export = measured_manifest.get("replay_export", {})
    mask_equal = (
        donor_export.get("sha256") == measured_export.get("sha256")
        and donor_export.get("epoch_offsets") == measured_export.get("epoch_offsets")
        and donor_export.get("length") == measured_export.get("length")
    )
    data_equal = [row["data_order_sha256"] for row in donor["epochs"]] == [
        row["data_order_sha256"] for row in measured["epochs"]
    ]
    if not mask_equal or not data_equal:
        raise ValueError("measured feedback trajectory differs from donor mask or data stream")
    return {
        "seed": donor["seed"],
        "donor_run_id": donor["run_id"],
        "measured_run_id": measured["run_id"],
        "mask_sha256": donor_export.get("sha256"),
        "mask_equal": True,
        "data_order_equal": True,
    }


def _state_tree_equal(left: Any, right: Any) -> bool:
    """Compare checkpoint state trees exactly without changing either value.

    Inputs may be tensors, mappings, lists, tuples, or scalar primitives. The
    boolean output requires identical structure and bitwise-equal tensors. This
    helper performs CPU comparisons only and does not modify files, RNG, or state.
    """

    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, Mapping):
        return isinstance(right, Mapping) and left.keys() == right.keys() and all(
            _state_tree_equal(left[key], right[key]) for key in left
        )
    if isinstance(left, (list, tuple)):
        return isinstance(right, type(left)) and len(left) == len(right) and all(
            _state_tree_equal(a, b) for a, b in zip(left, right, strict=True)
        )
    return left == right


def verify_replay_trajectory_equality(
    donor_directory: str | Path, replay_directory: str | Path
) -> dict[str, Any]:
    """Prove a learned replay reproduced its completed feedback donor trajectory.

    Inputs are completed donor and ``relative_replay_learned`` directories. The
    returned report covers exact model, predictor, optimizer, schedule/head-mask,
    counter, and data-order equality at the final committed boundary. Committed
    checkpoints are checksum-verified and loaded safely on CPU; mismatched roles,
    seeds, hardware, or trajectory state raise without changing either run.
    """

    donor = validate_run_directory(donor_directory)
    replay = validate_run_directory(replay_directory)
    if donor["method"] != "relative_feedback":
        raise ValueError("replay equivalence donor must be relative_feedback")
    if replay["method"] != "relative_replay_learned":
        raise ValueError("replay equivalence candidate must be relative_replay_learned")
    if donor["seed"] != replay["seed"] or donor["hardware_key"] != replay["hardware_key"]:
        raise ValueError("replay equivalence requires matching seed and environment")
    donor_payload = load_checkpoint(
        latest_committed_checkpoint(Path(donor_directory) / "checkpoints")
    )
    replay_payload = load_checkpoint(
        latest_committed_checkpoint(Path(replay_directory) / "checkpoints")
    )
    donor_state = donor_payload["run_state"]
    replay_state = replay_payload["run_state"]
    fields = {
        "model": _state_tree_equal(donor_payload["model"], replay_payload["model"]),
        "predictor": _state_tree_equal(donor_payload["predictor"], replay_payload["predictor"]),
        "optimizers": _state_tree_equal(donor_payload["optimizers"], replay_payload["optimizers"]),
        "executed_exact_mask": donor_state["executed_exact_mask"] == replay_state["executed_exact_mask"],
        "executed_head_mask": donor_state["executed_head_mask"] == replay_state["executed_head_mask"],
        "counters": donor_state["counters"] == replay_state["counters"],
        "data_order_hashes": donor_state["data_order_hashes"] == replay_state["data_order_hashes"],
    }
    if not all(fields.values()):
        failed = sorted(name for name, equal in fields.items() if not equal)
        raise ValueError(f"learned replay trajectory mismatch: {failed}")
    return {
        "donor_run_id": donor["run_id"],
        "replay_run_id": replay["run_id"],
        "seed": donor["seed"],
        "equivalent": True,
        "checks": fields,
    }


def verify_resume_equivalence(
    uninterrupted_directory: str | Path, resumed_directory: str | Path
) -> dict[str, Any]:
    """Prove an interrupted/resumed smoke ended in the uninterrupted state.

    Inputs are two completed same-method run directories. The returned report
    checks model, predictor, optimizer, controller, masks, counters, RNG, and
    data-order state while deliberately excluding elapsed timing, run identity,
    and the resumed run's interruption record. Checkpoints are verified and read
    on CPU; any mismatch raises and neither run is modified.
    """

    uninterrupted = validate_run_directory(uninterrupted_directory)
    resumed = validate_run_directory(resumed_directory)
    if uninterrupted["method"] != resumed["method"] or uninterrupted["seed"] != resumed["seed"]:
        raise ValueError("resume equivalence requires matching method and classifier seed")
    left = load_checkpoint(
        latest_committed_checkpoint(Path(uninterrupted_directory) / "checkpoints")
    )
    right = load_checkpoint(
        latest_committed_checkpoint(Path(resumed_directory) / "checkpoints")
    )

    # Compare actual serialized training-state fields.
    # Timing, checkpoint content hashes, run-specific records and
    # interruption history are deliberately excluded.
    state_fields = (
        "stage",
        "completed_epoch",
        "next_epoch",
        "completed",
        "global_step",
        "postwarm_step",
        "pending_quota",
        "exact_trunk_backwards",
        "synthetic_backwards",
        "zero_backwards",
        "skipped_slots",
        "trunk_optimizer_steps",
        "head_optimizer_steps",
        "predictor_fits",
        "predictor_inferences",
        "scale",
        "lr_global_epoch",
        "mask_cursor",
        "executed_exact_mask",
        "executed_head_mask",
        "replay_schedule_mask",
        "replay_epoch_offsets",
        "mask_epoch_offsets",
        "checkpoint_sequence",
        "data_order_hashes",
        "target_attainment",
    )

    fields = {
        "model": _state_tree_equal(left["model"], right["model"]),
        "predictor": _state_tree_equal(left["predictor"], right["predictor"]),
        "optimizers": _state_tree_equal(left["optimizers"], right["optimizers"]),
        "controller": _state_tree_equal(left["controller"], right["controller"]),
        "rng_state": _state_tree_equal(left["rng_state"], right["rng_state"]),
        "run_state": all(
            _state_tree_equal(left["run_state"][name], right["run_state"][name])
            for name in state_fields
        ),
    }
    if not all(fields.values()):
        failed = sorted(name for name, equal in fields.items() if not equal)
        raise ValueError(f"resumed trajectory mismatch: {failed}")
    return {
        "uninterrupted_run_id": uninterrupted["run_id"],
        "resumed_run_id": resumed["run_id"],
        "equivalent": True,
        "timing_excluded": True,
        "checks": fields,
    }


def plot_accuracy_curves(
    records: Sequence[Mapping[str, Any]], output_directory: str | Path
) -> list[Path]:
    """Write descriptive accuracy-versus-epoch and time plots from all endpoints.

    Inputs are validated records and an analysis directory. Accuracy is plotted
    as percent from the first warm-up evaluation; time uses observed endpoint
    seconds without interpolation. Two PNG paths are returned. Matplotlib is
    optional and a clear import error is raised if unavailable.
    """

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required only for optional plots") from exc
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / "accuracy_vs_epoch.png", output / "accuracy_vs_time.png"]
    for x_key, x_label, path in (
        ("global_epoch", "Global epoch endpoint", paths[0]),
        ("observation_time_seconds", "Observed workflow time (seconds)", paths[1]),
    ):
        figure, axis = plt.subplots(figsize=(8, 5))
        for record in records:
            epochs = record.get("epochs", [])
            axis.plot(
                [float(row[x_key]) for row in epochs],
                [100.0 * float(row["validation_accuracy"]) for row in epochs],
                marker="o",
                label=f"{record.get('method')} seed {record.get('seed')}",
            )
        axis.set_xlabel(x_label)
        axis.set_ylabel("Validation accuracy (%)")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="small")
        figure.tight_layout()
        figure.savefig(path, dpi=150)
        plt.close(figure)
    return paths


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Atomically write same-schema analysis rows to one UTF-8 CSV file.

    Inputs are a destination and nonempty mappings with identical keys. Nothing
    is returned. Parent directories are created; schema mismatch raises before
    replacement and no training, checkpoint, or source artifact is touched.
    """

    if not rows:
        raise ValueError("analysis CSV requires at least one row")
    columns = list(rows[0])
    if any(list(row) != columns for row in rows):
        raise ValueError("analysis CSV rows must share one ordered schema")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    temporary.replace(path)


def analyze_study(
    run_directories: Sequence[str | Path], output_directory: str | Path
) -> dict[str, Any]:
    """Validate completed runs and write the planned descriptive M6 artifacts.

    Inputs are explicit run directories and a fresh/empty analysis directory.
    Outputs include validation summaries, target attainment, predefined paired
    accuracy tables when both methods are present, deployable Pareto rows, cost
    classes, and accuracy curves. All input runs are read only; no conclusions,
    tuning, retries, or missing methods are synthesized.
    """

    output = Path(output_directory)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("analysis output must be absent or empty")
    output.mkdir(parents=True, exist_ok=True)
    raw_records = [validate_run_directory(path) for path in run_directories]
    records = average_replica_records_within_seed(raw_records)
    identities = [(row["run_id"], row["seed"], row["method"]) for row in records]
    if len(identities) != len(set(identities)):
        raise ValueError("study input repeats a run/seed/method identity")
    targets = target_attainment_rows(records, [0.85, 0.87, 0.89])
    _write_csv(output / "target_attainment.csv", targets)
    method_names = {str(record["method"]) for record in records}
    contrasts = (
        ("relative_replay_zero_shadow", "relative_feedback"),
        ("relative_replay_skip_shadow", "relative_feedback"),
        ("relative_periodic_epoch", "relative_feedback"),
        ("relative_gap_permutation_epoch", "relative_feedback"),
    )
    paired = []
    for reference, candidate in contrasts:
        if {reference, candidate}.issubset(method_names):
            paired.append(paired_accuracy_difference(records, reference, candidate))
    atomic_write_json(output / "paired_comparisons.json", {"comparisons": paired})
    if paired:
        _write_csv(
            output / "paired_comparisons.csv",
            [
                {
                    "schema_version": item["schema_version"],
                    "reference_method": item["reference_method"],
                    "candidate_method": item["candidate_method"],
                    "reference_normalization": item["reference_normalization"],
                    "candidate_normalization": item["candidate_normalization"],
                    "horizon_epochs": item["horizon_epochs"],
                    "unit": item["unit"],
                    "status": item["status"],
                    "n_classifier_seeds": item["n_classifier_seeds"],
                    "mean_difference_pp": item["mean"],
                    "sample_sd_pp": item["sample_sd"],
                    "ci_low_pp": item["ci_low"],
                    "ci_high_pp": item["ci_high"],
                    "confidence_level": item["confidence_level"],
                    "per_seed_json": json.dumps(item["per_seed"], separators=(",", ":")),
                }
                for item in paired
            ],
        )
    pareto = deployment_pareto(records)
    atomic_write_json(output / "deployment_pareto.json", {"points": pareto})
    costs = [classify_costs(record) for record in records]
    atomic_write_json(output / "cost_classes.json", {"records": costs})
    atomic_write_json(
        output / "validated_runs.json",
        {
            "schema_version": SCHEMA_VERSION,
            "runs": [
                {
                    key: record[key]
                    for key in (
                        "run_id", "seed", "method", "normalization", "completed",
                        "retrospective", "deployment_eligible", "resumed",
                        "scientific_fingerprint", "execution_fingerprint",
                    )
                }
                for record in records
            ],
        },
    )
    plots = plot_accuracy_curves(records, output)
    return {
        "run_count": len(records),
        "target_row_count": len(targets),
        "paired_comparison_count": len(paired),
        "pareto_point_count": len(pareto),
        "plots": [str(path) for path in plots],
    }


def parse_args(arguments: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse read-only M6 analysis commands and return the selected namespace.

    Parsing does not read results, create output, import plotting libraries, or
    launch training, downloads, remote work, or follow-up analysis.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="validate committed run schemas")
    validate.add_argument("runs", nargs="+")
    validate.add_argument("--allow-partial", action="store_true")
    summarize = subparsers.add_parser("summarize", help="write planned read-only M6 tables and plots")
    summarize.add_argument("--out", required=True)
    summarize.add_argument("runs", nargs="+")
    compare_masks = subparsers.add_parser(
        "compare-feedback-masks", help="verify donor and fresh measured feedback masks"
    )
    compare_masks.add_argument("donor")
    compare_masks.add_argument("measured")
    compare_replay = subparsers.add_parser(
        "compare-replay", help="verify learned replay and donor final trajectory state"
    )
    compare_replay.add_argument("donor")
    compare_replay.add_argument("replay")
    compare_resume = subparsers.add_parser(
        "compare-resume", help="verify resumed and uninterrupted final scientific state"
    )
    compare_resume.add_argument("uninterrupted")
    compare_resume.add_argument("resumed")
    return parser.parse_args(arguments)


def main(arguments: Sequence[str] | None = None) -> int:
    """Execute one explicit read-only analysis command and return its exit code.

    Inputs are optional command-line strings. A compact validation report is
    printed; no training, dataset access, remote I/O, tuning, or retry occurs.
    """

    args = parse_args(arguments)
    if args.command == "validate":
        records = [
            validate_run_directory(path, require_complete=not args.allow_partial)
            for path in args.runs
        ]
        print(
            json.dumps(
                [
                    {
                        key: row[key]
                        for key in ("run_id", "seed", "method", "completed", "comparison_eligible")
                    }
                    for row in records
                ],
                indent=2,
            )
        )
        return 0
    if args.command == "summarize":
        result = analyze_study(args.runs, args.out)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    if args.command == "compare-feedback-masks":
        print(json.dumps(verify_feedback_mask_equality(args.donor, args.measured), indent=2, sort_keys=True))
        return 0
    if args.command == "compare-replay":
        print(json.dumps(verify_replay_trajectory_equality(args.donor, args.replay), indent=2, sort_keys=True))
        return 0
    if args.command == "compare-resume":
        print(json.dumps(verify_resume_equivalence(args.uninterrupted, args.resumed), indent=2, sort_keys=True))
        return 0
    raise ValueError(f"unsupported analysis command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "analyze_study", "average_replica_records_within_seed", "classify_costs",
    "dependency_inclusive_cost", "deployment_pareto", "epoch_observation_times",
    "hardware_drift_check", "hardware_identity", "make_run_order_manifest",
    "paired_accuracy_difference", "paired_time_ratio", "plot_accuracy_curves",
    "reconcile_timing", "target_attainment_rows", "validate_run_directory",
    "verify_feedback_mask_equality",
]
