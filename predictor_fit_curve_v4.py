"""Measure bounded predictor learning curves on frozen pilot checkpoints.

This separately authorized exploratory follow-up changes only the fitting budget
and cache size relative to predictor_investigation_v4.py. It uses no downloads,
classifier training, validation selection, early stopping, or optimizer tuning.
Completed fits are immutable. Restarting repeats only unfinished fits, while
retaining their earlier partial curves as evidence.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import Subset

import predictor_investigation_v4 as prior


SPEC = {
    'schema': 'predictor_fit_curve_v1',
    'status': 'exploratory_not_confirmatory',
    'checkpoint_sequences': [5, 16],
    'variants': ['historical_fresh', 'global64_fresh'],
    'fit_cache_batches': [1, 8],
    'updates': 4096,
    'report_updates': [0, 64, 256, 1024, 4096],
    'selection': 'same_disjoint_training_ids_as_prior_investigation',
    'single_batch': 'first_batch_of_prior_fitting_cache',
    'heldout_usage': 'fixed_reporting_only_no_selection_or_early_stopping',
    'optimizer_loss_clipping': 'unchanged_pilot_predictor_contract',
    'comparison_budget': 'equal_updates_not_equal_compute',
    'classifier_training': False,
    'clean_efficiency_claims': False,
}


def digest(values: dict) -> str:
    """Identify JSON-compatible study values without depending on dictionary order."""
    return hashlib.sha256(json.dumps(values, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def fit_curve(predictor, train_cache: list, heldout_cache: list, config,
              device: torch.device, spec: dict, on_report=None) -> dict:
    """Fit one fresh predictor and report fixed learning-curve points.

    CPU cache triples contain boundary features/targets [B,C,H,W] and labels [B].
    Only predictor parameters change. One AdamW instance persists across every
    reporting point. Held-out values never determine gradients, stopping, or
    selection. Caller RNG is restored; nonfinite updates fail. The optional
    callback receives partial JSON reports for interruption evidence.
    """
    rng = prior.capture_rng_state()
    try:
        settings = config.scientific_dict()['optimizers']['predictor']
        contract = config.scientific_dict()['gradient_contract']
        optimizer = torch.optim.AdamW(predictor.parameters(), lr=settings['lr'],
                                     weight_decay=settings['weight_decay'], betas=tuple(settings['betas']),
                                     eps=settings['eps'])
        generator = np.random.default_rng(prior.SPEC['fit_seed'])
        predictor.train()
        order, curve, block = [], [], []
        started = time.perf_counter()
        for update in range(spec['updates'] + 1):
            if update in spec['report_updates']:
                point = {'update': update,
                         'fit': prior.prediction_report(predictor, train_cache, device),
                         'heldout': prior.prediction_report(predictor, heldout_cache, device),
                         'training_since_previous_report': block}
                curve.append(point)
                block = []
                if on_report is not None:
                    on_report({'curve': curve, 'completed_updates': update})
                print(f"  {update}/{spec['updates']}: fit residual "
                      f"{point['fit']['pooled_relative_residual']:.6f}, held-out "
                      f"{point['heldout']['pooled_relative_residual']:.6f}", flush=True)
            if update == spec['updates']:
                break
            if not order:
                order = generator.permutation(len(train_cache)).tolist()
            batch_id = order.pop(0)
            boundary, labels, target = [v.to(device) for v in train_cache[batch_id]]
            optimizer.zero_grad(set_to_none=True)
            total, mse, cosine = prior.predictor_loss_components(
                predictor(boundary, labels), target, cosine_weight=contract['cosine_weight'],
                valid_norm_floor=contract['valid_target_flat_norm_floor'], cosine_eps=contract['cosine_eps'])
            if not bool(torch.isfinite(total)):
                raise ValueError('non-finite predictor fitting loss')
            total.backward()
            norm = nn.utils.clip_grad_norm_(predictor.parameters(), contract['global_l2_clip'],
                                           error_if_nonfinite=True)
            optimizer.step()
            block.append({'update': update + 1, 'batch_id': batch_id, 'mse': float(mse.detach()),
                          'weighted_cosine': contract['cosine_weight'] * float(cosine.detach()),
                          'clip_coefficient': min(1.0, contract['global_l2_clip'] / (float(norm) + 1e-6))})
        return {'curve': curve, 'parameters': sum(p.numel() for p in predictor.parameters()),
                'elapsed_seconds_including_reporting': time.perf_counter() - started,
                'fit_cache_sha256': prior.cache_identity(train_cache),
                'heldout_cache_sha256': prior.cache_identity(heldout_cache)}
    finally:
        prior.restore_rng_state(rng)


def run_fit(path: Path, identity: dict, config, train_cache: list, heldout_cache: list,
            device: torch.device, variant: str, spec: dict) -> dict:
    """Run one isolated fit or verify its completed report; retain failed attempt curves.

    The identity binds source, samples, scripts and fit condition. Only new files
    under path.parent are written. An interruption repeats this fit from the same
    initialization; it never resets AdamW midway through a continuing fit.
    """
    identity_hash = digest(identity)
    prior.preserve_uncommitted_record(path)
    if path.exists():
        record = prior.read_record(path)
        if record.get('identity_sha256') != identity_hash or record.get('status') != 'complete':
            raise ValueError(f'incompatible completed fit: {path}')
        print(f'Skip completed fit: {path.stem}', flush=True)
        return record
    attempt = path.parent/'attempts'/f'{path.stem}-{uuid.uuid4().hex}.json'
    environment = prior.environment_metadata(device)

    def save_progress(progress):
        """Persist this attempt's latest curve; earlier attempts retain distinct files."""
        prior.atomic_write_json(attempt, {'status': 'running', 'identity_sha256': identity_hash,
                                         'analysis_environment': environment, **progress})

    rng = prior.capture_rng_state()
    try:
        torch.manual_seed(prior.SPEC['fit_seed'])
        cfg = config.scientific_dict()['predictor']
        channels, classes = cfg['input_channels'], cfg['label_embedding_classes']
        if variant == 'historical_fresh':
            candidate = prior.GradientPredictor(channels, cfg['bottleneck_width'], classes)
        elif variant == 'global64_fresh':
            candidate = prior.GlobalPredictor(channels, prior.SPEC['wide_channels'], classes)
        else:
            raise ValueError(f'unknown predictor variant: {variant}')
        print(f'Fit: {path.stem}', flush=True)
        results = fit_curve(candidate.to(device), train_cache, heldout_cache, config, device, spec, save_progress)
        record = {'status': 'complete', 'identity_sha256': identity_hash, 'identity': identity,
                  'analysis_environment': environment, **results}
        prior.write_record(path, record)
        prior.atomic_write_json(attempt, {'status': 'complete', 'identity_sha256': identity_hash})
        return record
    except BaseException:
        evidence = json.loads(attempt.read_text()) if attempt.exists() else {}
        prior.atomic_write_json(attempt, {**evidence, 'status': 'interrupted_or_failed',
                                         'identity_sha256': identity_hash, 'traceback': traceback.format_exc()})
        raise
    finally:
        prior.restore_rng_state(rng)


def run_study(args) -> None:
    """Verify the prior study and frozen inputs, then run eight independent bounded fits.

    Requires explicit authorization and existing CIFAR data/split/checkpoints.
    No source or prior report is written. Separate output permits resuming on a
    different allocation; each completed fit records its own environment.
    """
    if not args.authorize_fit_study:
        raise PermissionError('pass --authorize-fit-study to run this follow-up')
    project = Path(__file__).resolve().parent
    out = prior.validate_output_root(args.out, project)
    previous = Path(args.previous_study).resolve()
    source = Path(args.checkpoint_dir).resolve()
    split, data = Path(args.split_file).resolve(), Path(args.data_dir).resolve()
    for protected in (previous, source.parent, split.parent, data):
        if out == protected or out.is_relative_to(protected) or protected.is_relative_to(out):
            raise ValueError('output must be separate from source data, runs, and prior investigation')
    previous_manifest = prior.read_record(previous/'investigation.json')
    if previous_manifest['script_sha256'] != prior.sha256_file(prior.__file__):
        raise ValueError('prior investigation script changed; use the original file')
    if previous_manifest['spec'] != prior.SPEC or previous_manifest['split_sha256'] != prior.sha256_file(split):
        raise ValueError('prior study specification or split differs')
    first = None
    for sequence in SPEC['checkpoint_sequences']:
        payload = prior.load_verified_source(source/f'checkpoint_{sequence:06d}.pt', sequence)
        if payload['resume_metadata']['checkpoint_sha256'] != previous_manifest['checkpoint_hashes'][str(sequence)]:
            raise ValueError('checkpoint differs from the prior investigation')
        report = prior.read_record(previous/f'checkpoint_{sequence:06d}.json')
        if (report.get('investigation_sha256') != digest(previous_manifest)
                or report.get('status') != 'complete'
                or report.get('source_sha256') != previous_manifest['checkpoint_hashes'][str(sequence)]):
            raise ValueError('prior checkpoint report is incompatible')
        if first is None:
            first = payload
        elif first['scientific_fingerprint'] != payload['scientific_fingerprint']:
            raise ValueError('checkpoint conditions differ')
    device = torch.device(args.device)
    config = prior.RunConfig(first['resolved_config'])
    prior.configure_determinism(config, device)
    augmented, validation, identities = prior.load_cifar10_datasets(config, data, split)
    if identities != first['data_identities']:
        raise ValueError('split identities differ from the checkpoint')
    deterministic = Subset(validation.dataset, augmented.indices)
    groups = prior.select_groups(len(deterministic), prior.SPEC)
    selected = {k: [[int(augmented.indices[i]) for i in b] for b in v] for k, v in groups.items()}
    if selected != previous_manifest['official_training_indices']:
        raise ValueError('selected training examples differ from the prior investigation')
    identity = {'spec': SPEC, 'previous_investigation_sha256': digest(previous_manifest),
                'script_sha256': prior.sha256_file(__file__), 'prior_script_sha256': prior.sha256_file(prior.__file__),
                'checkpoint_hashes': previous_manifest['checkpoint_hashes'], 'official_training_indices': selected}
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out/'fit_study.json'
    prior.preserve_uncommitted_record(manifest_path)
    if manifest_path.exists():
        if prior.read_record(manifest_path) != identity:
            raise ValueError('output contains a different study; use a new folder')
    else:
        if any(out.glob('epoch_*.json')):
            raise ValueError('existing fit reports lack their study manifest')
        prior.write_record(manifest_path, identity)
    del first, payload, validation
    for sequence in SPEC['checkpoint_sequences']:
        payload = prior.load_verified_source(source/f'checkpoint_{sequence:06d}.pt', sequence)
        if payload['resume_metadata']['checkpoint_sha256'] != identity['checkpoint_hashes'][str(sequence)]:
            raise ValueError('source checkpoint changed during the study')
        config, model, saved, optimizers = prior.restore_models(payload, device)
        fit_batches = prior.image_batches(deterministic, groups['fit'], device, prior.SPEC['augmentation_seed'])
        heldout_batches = prior.image_batches(deterministic, groups['heldout'], device, prior.SPEC['augmentation_seed'])
        train_cache = prior.cache_targets(model, fit_batches, payload['scale'])
        heldout_cache = prior.cache_targets(model, heldout_batches, payload['scale'])
        del model, saved, optimizers, fit_batches, heldout_batches
        for batch_count in SPEC['fit_cache_batches']:
            for variant in SPEC['variants']:
                fit_identity = {'study_sha256': digest(identity), 'checkpoint': sequence,
                                'fit_batches': batch_count, 'variant': variant}
                path = out/f'epoch_{sequence:02d}_{variant}_batches{batch_count}.json'
                run_fit(path, fit_identity, config, train_cache[:batch_count], heldout_cache, device, variant, SPEC)
        del payload, train_cache, heldout_cache
    print(f'All eight bounded fits completed. Reports: {out}', flush=True)


def main(argv=None) -> int:
    """Run the authorized bounded study from explicit paths or describe its budget."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--authorize-fit-study', action='store_true')
    for name in ('previous-study', 'checkpoint-dir', 'data-dir', 'split-file', 'out'):
        parser.add_argument('--'+name)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps(SPEC, indent=2))
        return 0
    if any(getattr(args, name) is None for name in ('previous_study','checkpoint_dir','data_dir','split_file','out')):
        parser.error('provide --previous-study, --checkpoint-dir, --data-dir, --split-file and --out')
    run_study(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
