"""Explore predictor quality on frozen v4 checkpoints without resuming training.

This separately authorized follow-up preserves the historical implementation and
IMPLEMENTATION_PLAN.md. It is exploratory, not a new pilot or an ADA-GP replica.
Only predictor parameters are fitted, on a small training-only cache. Completed
checkpoint reports survive interruptions; an unfinished checkpoint is repeated.
No dataset downloads, validation selection, or classifier training are performed.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import random
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Subset

from gradient_phase_d_resnet_v4 import (
    GradientPredictor, _resolved_artifacts, _restore_optimizer_devices,
    _scientific_artifacts, build_models, build_optimizers, configure_determinism,
    load_cifar10_datasets, predictor_loss_components, set_training_mode,
)
from phase_d_v4_analysis import finite_summary, fixed_probe_indices
from phase_d_v4_diagnostics import (
    gradient_metrics, probe_boundary_gradients, probe_normalization_transition,
    probe_parameter_updates,
)
from phase_d_v4_io import (
    RunConfig, atomic_write_json, capture_rng_state, environment_metadata,
    fingerprint, load_checkpoint, restore_rng_state, sha256_file,
    validate_config, validate_output_root,
)


SPEC = {
    "schema": "predictor_investigation_v1",
    "status": "exploratory_not_confirmatory",
    "source_run_id": "seed0_relative_feedback_measured",
    "checkpoint_sequences": [4, 5, 10, 16],
    "selection_seed": 88001,
    "augmentation_seed": 88003,
    "fit_seed": 88002,
    "batch_size": 32,
    "diagnostic_batches": 4,
    "fit_batches": 8,
    "heldout_batches": 4,
    "fit_updates": 256,
    "wide_channels": 64,
    "variants": ["saved_unfitted", "historical_fresh", "global64_fresh"],
    "objective_and_optimizer": "unchanged_reference_predictor_loss_and_AdamW",
    "normalization": "fixed_classifier_trunk_train_head_eval_reset_buffers_per_batch",
    "sample_source": "three_disjoint_subsets_of_official_training_split",
    "heldout_usage": "terminal_reporting_only_no_selection_or_early_stopping",
    "comparison_budget": "equal_updates_and_examples_not_equal_compute",
    "warmup_probe_lr": "saved_optimizer_lr_no_hypothetical_schedule_advance",
    "classifier_training": False,
    "clean_efficiency_claims": False,
}


class GlobalPredictor(nn.Module):
    """Add image-wide feature context to a wider copy of the local predictor."""

    def __init__(self, channels: int, width: int, classes: int) -> None:
        """Create fresh local and pooled-context layers; consumes initialization RNG."""
        super().__init__()
        self.label = nn.Embedding(classes, channels)
        self.local = nn.Conv2d(channels, width, 1)
        self.context = nn.Conv2d(channels, width, 1, bias=False)
        self.spatial = nn.Conv2d(width, width, 3, padding=1, groups=width)
        self.output = nn.Conv2d(width, channels, 1)

    def forward(self, boundary: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Map [B,C,H,W] features and [B] labels to detached-input gradient estimates.

        The output retains predictor autograd, but never the classifier graph.
        Pooling lets each location depend on the image-wide feature mean.
        """
        features = boundary.detach() + self.label(labels)[:, :, None, None]
        hidden = F.relu(self.local(features) + self.context(features.mean((2, 3), keepdim=True)))
        return self.output(F.relu(self.spatial(hidden)))


def select_groups(size: int, spec: dict) -> dict[str, list[list[int]]]:
    """Choose disjoint training-subset positions with one private permutation.

    Returns diagnostic, fitting, and held-out batch lists. The first four batches
    exactly preserve the plan's diagnostic selector. No global RNG is consumed.
    """
    counts = [spec[k] for k in ("diagnostic_batches", "fit_batches", "heldout_batches")]
    batches = fixed_probe_indices(size, batch_size=spec["batch_size"],
                                  batches=sum(counts), seed=spec["selection_seed"])
    a, b, _ = counts
    return {"diagnostic": batches[:a], "fit": batches[a:a+b], "heldout": batches[a+b:]}


def image_batches(dataset, groups: list[list[int]], device: torch.device, seed: int) -> list:
    """Materialize fixed image/label batches while restoring caller RNG afterward.

    Dataset positions must be training-only. Augmentation, when present, is
    reproducible under seed. Returns [B,3,H,W] images and [B] labels on device.
    """
    rng = capture_rng_state()
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        result = []
        for ids in groups:
            samples = [dataset[i] for i in ids]
            result.append((torch.stack([s[0] for s in samples]).to(device),
                           torch.tensor([s[1] for s in samples], device=device)))
        return result
    finally:
        restore_rng_state(rng)


def restore_models(payload: dict, device: torch.device):
    """Reconstruct independent model, predictor, and full optimizers from CPU state.

    Returns config and live copies on device. Saved parameters and SGD/AdamW
    moments are preserved; the payload is never modified. Missing state raises.
    """
    rng = capture_rng_state()
    try:
        config = RunConfig(copy.deepcopy(payload["resolved_config"]))
        model, predictor = build_models(config, device)
        if predictor is None or payload["predictor"] is None:
            raise ValueError("predictor checkpoint is required")
        model.load_state_dict(payload["model"], strict=True)
        predictor.load_state_dict(payload["predictor"], strict=True)
        optimizers = build_optimizers(model, predictor, config)
        for name, optimizer in optimizers.items():
            if optimizer is not None:
                optimizer.load_state_dict(copy.deepcopy(payload["optimizers"][name]))
                _restore_optimizer_devices(optimizer)
        set_training_mode(model, predictor, "postwarm", config.scientific_dict()["identity"]["normalization"])
        return config, model, predictor, optimizers
    finally:
        restore_rng_state(rng)


def cache_targets(model, batches: list, scale: float) -> list:
    """Cache normalized gradients for a fixed classifier, never update it.

    Returns CPU triples of [B,C,H,W] features/targets and [B] labels. A disposable
    trunk uses training BN and the head uses frozen BN. Every batch starts with
    identical saved buffers. Targets are B*d(mean CE)/dh/scale, without graphs.
    """
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be finite and positive")
    rng = capture_rng_state()
    try:
        local = copy.deepcopy(model)
        local.requires_grad_(False)
        local.trunk.train()
        local.head.eval()
        buffers = {name: value.detach().clone() for name, value in local.named_buffers()}
        result = []
        for inputs, labels in batches:
            with torch.no_grad():
                for name, buffer in local.named_buffers():
                    buffer.copy_(buffers[name])
                boundary = local.trunk(inputs)
            leaf = boundary.detach().requires_grad_(True)
            loss = F.cross_entropy(local.head(leaf), labels)
            target = torch.autograd.grad(loss, leaf)[0] * len(labels) / scale
            if not bool(torch.isfinite(boundary).all() and torch.isfinite(target).all()):
                raise ValueError("non-finite frozen target cache")
            result.append((boundary.detach().cpu(), labels.detach().cpu(), target.detach().cpu()))
        return result
    finally:
        restore_rng_state(rng)


def cache_identity(cache: list) -> str:
    """Hash tensor shapes, dtypes, and bytes to identify exactly which targets were fitted."""
    digest = hashlib.sha256()
    for batch in cache:
        for tensor in batch:
            value = tensor.detach().cpu().contiguous()
            digest.update(str((tuple(value.shape), value.dtype)).encode())
            digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def prediction_report(predictor, cache: list, device: torch.device) -> dict:
    """Report sample metrics and a separately pooled residual on fixed cached targets.

    Inputs use normalized-target units. A null predictor is the zero baseline.
    No updates occur. Undefined direction/norm metrics retain explicit null flags.
    """
    rows, sums = [], {"target_sq": 0.0, "error_sq": 0.0}
    with torch.no_grad():
        for boundary, labels, target in cache:
            prediction = torch.zeros_like(target) if predictor is None else predictor(
                boundary.to(device), labels.to(device)).cpu()
            for truth, estimate in zip(target, prediction, strict=True):
                rows.append(gradient_metrics(truth, estimate, 1e-8))
            sums["target_sq"] += float(target.double().square().sum())
            sums["error_sq"] += float((prediction.double()-target.double()).square().sum())
    return {
        "units": "normalized_target", "per_example": rows,
        "sample_summaries": {key: {**finite_summary([r[key] for r in rows if r[key] is not None]),
                                   "undefined_count": sum(r[key] is None for r in rows)}
                             for key in ("cosine", "norm_ratio", "projection", "relative_residual")},
        "pooled_relative_residual": math.sqrt(sums["error_sq"]/sums["target_sq"])
        if sums["target_sq"] > 0 and math.isfinite(sums["error_sq"]) else None,
        "invalid_samples": sum(r["invalid"] for r in rows),
        "near_zero_targets": sum(r["true_near_zero"] for r in rows),
    }


def fit_predictor(predictor, cache: list, config: RunConfig, spec: dict, device) -> list:
    """Fit only predictor parameters for the fixed update budget; return training telemetry.

    All candidates use the same cached-batch order, reference AdamW, loss, and
    clipping. This mutates the supplied fresh predictor, never the cache/model.
    Held-out examples cannot enter this function. Non-finite loss/gradients fail.
    """
    settings = config.scientific_dict()["optimizers"]["predictor"]
    contract = config.scientific_dict()["gradient_contract"]
    optimizer = torch.optim.AdamW(predictor.parameters(), lr=settings["lr"],
                                 weight_decay=settings["weight_decay"], betas=tuple(settings["betas"]),
                                 eps=settings["eps"])
    generator = np.random.default_rng(spec["fit_seed"])
    order, telemetry = [], []
    predictor.train()
    for update in range(spec["fit_updates"]):
        if not order:
            order = generator.permutation(len(cache)).tolist()
        batch_id = order.pop(0)
        boundary, labels, target = [v.to(device) for v in cache[batch_id]]
        optimizer.zero_grad(set_to_none=True)
        total, mse, cosine = predictor_loss_components(
            predictor(boundary, labels), target,
            cosine_weight=contract["cosine_weight"],
            valid_norm_floor=contract["valid_target_flat_norm_floor"], cosine_eps=contract["cosine_eps"])
        if not bool(torch.isfinite(total)):
            raise ValueError("non-finite predictor fitting loss")
        total.backward()
        norm = nn.utils.clip_grad_norm_(predictor.parameters(), contract["global_l2_clip"], error_if_nonfinite=True)
        optimizer.step()
        telemetry.append({"update": update+1, "batch_id": batch_id, "mse": float(mse.detach()),
                          "weighted_cosine": contract["cosine_weight"]*float(cosine.detach()),
                          "clip_coefficient": min(1.0, contract["global_l2_clip"]/(float(norm)+1e-6))})
        if (update+1) % 64 == 0:
            print(f'  Predictor fitting: {update+1}/{spec["fit_updates"]} updates', flush=True)
    return telemetry


def capacity_comparison(model, saved_predictor, fit_batches, heldout_batches,
                        scale: float, config: RunConfig, spec: dict, device) -> dict:
    """Compare the saved predictor and two fresh fits against identical frozen targets.

    Original model/predictor/RNG are preserved. Both new architectures receive
    equal updates/examples, not equal FLOPs. Reports terminal held-out metrics
    without model selection; elapsed seconds are diagnostic overhead only.
    """
    rng = capture_rng_state()
    try:
        train_cache = cache_targets(model, fit_batches, scale)
        heldout_cache = cache_targets(model, heldout_batches, scale)
        cache_hashes = {"fit": cache_identity(train_cache), "heldout": cache_identity(heldout_cache)}
        results = {"zero": {"fit": prediction_report(None, train_cache, device),
                            "heldout": prediction_report(None, heldout_cache, device)},
                   "saved_unfitted": {"fit": prediction_report(saved_predictor, train_cache, device),
                                       "heldout": prediction_report(saved_predictor, heldout_cache, device)}}
        cfg = config.scientific_dict()["predictor"]
        for name in ("historical_fresh", "global64_fresh"):
            print(f'  Capacity comparison: {name}', flush=True)
            torch.manual_seed(spec["fit_seed"])
            channels, classes = cfg["input_channels"], cfg["label_embedding_classes"]
            candidate = (GradientPredictor(channels, cfg["bottleneck_width"], classes)
                         if name == "historical_fresh" else GlobalPredictor(channels, spec["wide_channels"], classes)).to(device)
            before = prediction_report(candidate, train_cache, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            telemetry = fit_predictor(candidate, train_cache, config, spec, device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            results[name] = {"parameters": sum(p.numel() for p in candidate.parameters()),
                             "fit_seconds_diagnostic_only": time.perf_counter()-start,
                             "initial_fit": before, "training_telemetry": telemetry,
                             "fit": prediction_report(candidate, train_cache, device),
                             "heldout": prediction_report(candidate, heldout_cache, device)}
            del candidate
        return {"cache_hashes": cache_hashes, "results": results,
                "not_a_classifier_accuracy_or_efficiency_experiment": True,
                "capacity_and_context_change_together": True}
    finally:
        restore_rng_state(rng)


def diagnostic_report(model, predictor, optimizers, batches, augmented_batches,
                      ids, scale: float, config: RunConfig) -> dict:
    """Run the plan's isolated gradient, update, and BN probes on fixed training batches.

    Returns all raw rows; no result selection. Update alternatives retain full
    saved optimizer state/LR and use the next disjoint probe batch for evaluation.
    """
    gradient, updates, transitions = [], [], []
    for i, (inputs, labels) in enumerate(batches):
        print(f'  Diagnostic batch {i+1}/{len(batches)}', flush=True)
        gradient.append(probe_boundary_gradients(model, predictor, inputs, labels, scale, config, sample_ids=ids[i]))
        fixed_inputs, fixed_labels = batches[(i+1) % len(batches)]
        updates.append(probe_parameter_updates(model, predictor, optimizers, inputs, labels,
                                               fixed_inputs, fixed_labels, scale, config))
        for source, pair in (("deterministic_training", (inputs, labels)),
                             ("augmented_training", augmented_batches[i])):
            transitions.append({"batch": i, "rows": probe_normalization_transition(
                model, predictor, *pair, scale, config, sample_source=source)})
    return {"gradient": gradient, "updates": updates, "normalization": transitions}


def load_verified_source(path: Path, sequence: int) -> dict:
    """Verify a committed pilot checkpoint, its source/config identities, and boundary.

    Reads CPU tensors only. Rejects a different run, missing optimizer/predictor,
    edited source, invalid scale, or wrong epoch. Does not enforce the old GPU:
    these are explicitly labeled diagnostics, not production continuation.
    """
    payload = load_checkpoint(path, expected={"run_id": SPEC["source_run_id"], "checkpoint_sequence": sequence})
    config = RunConfig(payload["resolved_config"])
    validate_config(config)
    if payload["artifacts"] != _resolved_artifacts(_scientific_artifacts()):
        raise ValueError("source hashes differ from the pilot; use its pinned repository version")
    if fingerprint(config, {**payload["artifacts"], **payload["data_identities"]}) != payload["scientific_fingerprint"]:
        raise ValueError("checkpoint scientific fingerprint mismatch")
    if payload["completed_epoch"] != sequence-1 or payload["next_epoch"] != sequence:
        raise ValueError("checkpoint is not the requested committed epoch boundary")
    if payload["method"] != "relative_feedback" or payload["normalization"] != "frozen_head_postwarm":
        raise ValueError("this investigation requires the pilot feedback condition")
    if payload["predictor"] is None or any(payload["optimizers"].get(k) is None for k in ("trunk", "head", "predictor")):
        raise ValueError("full model/predictor/optimizer checkpoints are required")
    if not math.isfinite(payload["scale"]) or payload["scale"] <= 0:
        raise ValueError("invalid saved gradient scale")
    return payload


def write_record(path: Path, record: dict) -> None:
    """Atomically save a new report and its checksum marker; refuses overwrites."""
    if path.exists() or path.with_suffix(path.suffix+'.commit.json').exists():
        raise FileExistsError(f"report already exists: {path}")
    atomic_write_json(path, record)
    atomic_write_json(path.with_suffix(path.suffix+'.commit.json'),
                      {"sha256": sha256_file(path), "size_bytes": path.stat().st_size})


def read_record(path: Path) -> dict:
    """Read one committed report or fail on missing markers, size/hash corruption."""
    marker = json.loads(path.with_suffix(path.suffix+'.commit.json').read_text())
    if sha256_file(path) != marker["sha256"] or path.stat().st_size != marker["size_bytes"]:
        raise ValueError(f"corrupt investigation report: {path}")
    return json.loads(path.read_text())


def preserve_uncommitted_record(path: Path) -> None:
    """Keep a report interrupted before its commit marker, freeing its original name.

    A lone commit marker is corruption and raises. Committed pairs are left for
    checksum verification by read_record; no committed data is changed.
    """
    marker = path.with_suffix(path.suffix+'.commit.json')
    if marker.exists() and not path.exists():
        raise ValueError(f"commit marker has no report: {path}")
    if path.exists() and not marker.exists():
        path.rename(path.with_name(f'{path.stem}.uncommitted-{uuid.uuid4().hex}{path.suffix}'))


def run_study(args) -> None:
    """Run or recover this bounded study, writing only its separate output folder.

    Requires explicit authorization and existing full checkpoints/data/split.
    Completed reports are verified/skipped; uncommitted reports are preserved
    under unique names. Interruptions redo only the unfinished checkpoint.
    """
    if not args.authorize_predictor_study:
        raise PermissionError("pass --authorize-predictor-study to run the bounded follow-up")
    project = Path(__file__).resolve().parent
    out = validate_output_root(args.out, project)
    source_dir, split = Path(args.checkpoint_dir).resolve(), Path(args.split_file).resolve()
    data = Path(args.data_dir).resolve()
    for protected in (source_dir.parent, split.parent, data):
        if out == protected or out.is_relative_to(protected) or protected.is_relative_to(out):
            raise ValueError("investigation output must be separate from source runs, split, and dataset")
    paths = {n: source_dir/f'checkpoint_{n:06d}.pt' for n in SPEC["checkpoint_sequences"]}
    for path in paths.values():
        if not path.is_file() or not path.with_suffix('.pt.commit.json').is_file():
            raise FileNotFoundError(f"missing full checkpoint or commit marker: {path}")
    # Load and verify every source before any fitting, so a missing late boundary
    # cannot silently shorten the declared study.
    hashes, first = {}, None
    for sequence, path in paths.items():
        payload = load_verified_source(path, sequence)
        hashes[str(sequence)] = payload["resume_metadata"]["checkpoint_sha256"]
        if first is None:
            first = payload
        elif payload["scientific_fingerprint"] != first["scientific_fingerprint"]:
            raise ValueError("checkpoint series mixes scientific conditions")
    config = RunConfig(first["resolved_config"])
    device = torch.device(args.device)
    configure_determinism(config, device)
    augmented, validation, identities = load_cifar10_datasets(config, data, split)
    if any(first["data_identities"].get(k) != value for k, value in identities.items()):
        raise ValueError("split differs from the saved checkpoint")
    # Reuse only deterministic transforms; validation examples are never read.
    deterministic = Subset(validation.dataset, augmented.indices)
    del validation
    groups = select_groups(len(deterministic), SPEC)
    original_ids = {k: [[int(augmented.indices[i]) for i in batch] for batch in batches]
                    for k, batches in groups.items()}
    identity = {"spec": SPEC, "checkpoint_hashes": hashes, "split_sha256": sha256_file(split),
                "script_sha256": sha256_file(__file__), "source_artifacts": first["artifacts"],
                "official_training_indices": original_ids}
    identity_sha256 = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    out.mkdir(parents=True, exist_ok=True)
    manifest = out/'investigation.json'
    preserve_uncommitted_record(manifest)
    if manifest.exists():
        if read_record(manifest) != identity:
            raise ValueError("existing investigation identity differs; use a new output folder")
    else:
        if any(out.glob('checkpoint_*.json')):
            raise ValueError("checkpoint reports exist without an investigation manifest; preserve and investigate")
        write_record(manifest, identity)
    del first, payload
    for sequence, path in paths.items():
        report_path = out/f'checkpoint_{sequence:06d}.json'
        preserve_uncommitted_record(report_path)
        if report_path.exists() and report_path.with_suffix('.json.commit.json').exists():
            record = read_record(report_path)
            if (record['source_sha256'] != hashes[str(sequence)] or record['status'] != 'complete'
                    or record.get('investigation_sha256') != identity_sha256):
                raise ValueError("existing checkpoint report identity/status mismatch")
            print(f'Skip completed investigation: checkpoint {sequence}', flush=True)
            continue
        attempt = out/'attempts'/f'{uuid.uuid4().hex}.json'
        atomic_write_json(attempt, {"checkpoint": sequence, "status": "started"})
        started = time.perf_counter()
        try:
            print(f'Checkpoint {sequence}: isolated diagnostics and bounded predictor fits', flush=True)
            payload = load_verified_source(path, sequence)
            config, model, predictor, optimizers = restore_models(payload, device)
            batches = {k: image_batches(deterministic, v, device, SPEC['augmentation_seed']) for k,v in groups.items()}
            aug_batches = image_batches(augmented, groups['diagnostic'], device, SPEC['augmentation_seed'])
            diagnostics = diagnostic_report(model, predictor, optimizers, batches['diagnostic'], aug_batches,
                                            original_ids['diagnostic'], payload['scale'], config)
            capacity = capacity_comparison(model, predictor, batches['fit'], batches['heldout'],
                                           payload['scale'], config, SPEC, device)
            record = {"status": "complete", "source_sha256": hashes[str(sequence)],
                      "investigation_sha256": identity_sha256,
                      "source_environment": payload['environment'],
                      "analysis_environment": environment_metadata(device),
                      "elapsed_seconds_diagnostic_overhead": time.perf_counter()-started,
                      "diagnostics": diagnostics, "capacity": capacity}
            if sha256_file(path) != hashes[str(sequence)]:
                raise RuntimeError("source checkpoint changed during diagnosis")
            write_record(report_path, record)
            atomic_write_json(attempt, {"checkpoint": sequence, "status": "complete"})
            del payload, model, predictor, optimizers, batches, aug_batches
            print(f'Saved checkpoint {sequence} report: {report_path}', flush=True)
        except BaseException:
            atomic_write_json(attempt, {"checkpoint": sequence, "status": "interrupted_or_failed",
                                       "traceback": traceback.format_exc()})
            raise
    print('Investigation complete. No classifier training or clean speed claim was performed.', flush=True)


def main(argv=None) -> int:
    """Describe the protocol or run an explicitly authorized study from CLI paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--authorize-predictor-study', action='store_true')
    parser.add_argument('--checkpoint-dir')
    parser.add_argument('--data-dir')
    parser.add_argument('--split-file')
    parser.add_argument('--out')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps(SPEC, indent=2))
        return 0
    if any(getattr(args, key) is None for key in ('checkpoint_dir','data_dir','split_file','out')):
        parser.error('--checkpoint-dir, --data-dir, --split-file, and --out are required')
    run_study(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
