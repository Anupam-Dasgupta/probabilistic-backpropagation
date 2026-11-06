"""Run the separately approved six-condition, two-epoch predictor screen.

This exploratory continuation starts at the measured seed-0 epoch-5 checkpoint.
It preserves the original 16-epoch LR clock and all historical source files.
One command executes or recovers the whole screen. No downloads, split creation,
controller tuning, test-set access, or automatic extension are performed.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import copy
import hashlib
import json
import math
from pathlib import Path
import time
import uuid
import zipfile

import numpy as np
import torch

import predictor_investigation_v4 as prior
import predictor_utility_v4 as utility
from gradient_phase_d_resnet_v4 import (
    _assert_state_trees_close, _cpu_tree, _update_data_hash, epoch_loader, evaluate, exact_step,
    make_uniform_third_mask, set_epoch_lr, synthetic_step,
)


SPEC = {
    'schema': 'predictor_online_screen_v1',
    'status': 'exploratory_development_not_confirmatory',
    'source_run_id': 'seed0_relative_feedback_measured',
    'source_epoch': 5,
    'end_epoch': 7,
    'lr_horizon': 16,
    'batch_size': 32,
    'training_examples': 45000,
    'validation_examples': 5000,
    'conditions': ['saved', 'local32_refit', 'global32_refit',
                   'zero_fast', 'skip_fast', 'exact_trunk_matched_head'],
    'fit_examples': 4096,
    'fit_updates': 256,
    'fit_save_every': 64,
    'fit_seed': 88102,
    'augmentation_seed': 88103,
    'order_seed': 88201,
    'schedule': 'existing_uniform_third_per_epoch_floor_n_over_3_plus_half',
    'head_updates': 'same_uniform_third_slots_in_all_conditions',
    'refitting': 'once_before_continuation_then_one_fit_per_exact_scheduled_slot',
    'controls': 'zero_and_skip_have_no_predictor_or_shadow_overhead',
    'normalization': 'frozen_head_postwarm',
    'checkpointing': 'initial_state_and_each_completed_epoch',
    'evaluation': 'original_5000_validation_baseline_and_each_completed_epoch',
    'claim_scope': 'conditional_two_epoch_screen_no_standalone_speed_or_convergence_claim',
}

COUNTERS = ('exact_trunk_backwards', 'synthetic_backwards', 'zero_backwards',
            'skipped_slots', 'trunk_optimizer_steps', 'head_optimizer_steps',
            'predictor_fits', 'predictor_inferences')


def synchronize(device) -> None:
    """Finish queued device work before reading a clock; returns nothing."""
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


def commit_json(path: Path, value: dict) -> None:
    """Publish a new JSON record and checksum, preserving interrupted prior writes.

    Inputs are a destination and JSON-safe mapping. Writes only this destination;
    committed files are immutable and an orphan commit marker is an error.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    prior.preserve_uncommitted_record(path)
    prior.write_record(path, value)


@contextmanager
def measured_phase(out, phase, condition, identity, environment, device):
    """Record completed wall time and durable evidence of interrupted work.

    Yields nothing. Synchronizes CUDA at boundaries and writes start/end records.
    Times include work and checkpoint I/O inside the block. Missing end records
    mean unknown interrupted cost, never zero cost. Exceptions are recorded and
    re-raised; this does not retry or change scientific work.
    """
    synchronize(device)
    directory = out / 'costs' / uuid.uuid4().hex
    started = time.perf_counter()
    commit_json(directory / 'start.json', {
        'identity': identity, 'phase': phase, 'condition': condition,
        'environment': environment, 'status': 'started',
    })
    status, error = 'complete', None
    try:
        yield
        synchronize(device)
    except BaseException as exc:
        status, error = 'failed', type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        commit_json(directory / 'end.json', {
            'identity': identity, 'status': status, 'error': error,
            'seconds_including_phase_io': time.perf_counter() - started,
        })


def cost_report(out: Path, identity: str) -> dict:
    """Read attempt clocks without double-counting phases inside whole invocations.

    Returns measured seconds and unknown-cost flags; no files change. Preparation
    is counted even if repeated. Unfinished attempts retain unknown duration and
    exclude clean efficiency claims. Completed invocation time includes setup,
    while phase sums are explanatory subsets, not extra time to add to it.
    """
    phases, invocations, incomplete = {}, [], []
    for start_path in sorted((out / 'costs').glob('*/start.json')):
        if not start_path.with_suffix('.json.commit.json').exists():
            incomplete.append({'phase': 'uncommitted_phase_start', 'condition': 'unknown'})
            continue
        start = prior.read_record(start_path)
        if start['identity'] != identity:
            raise ValueError('cost record belongs to another study')
        end_path = start_path.with_name('end.json')
        if not end_path.with_suffix('.json.commit.json').exists():
            incomplete.append({'phase': start['phase'], 'condition': start['condition']})
            continue
        end = prior.read_record(end_path)
        if end['identity'] != identity:
            raise ValueError('cost completion identity differs')
        seconds = end['seconds_including_phase_io']
        if not math.isfinite(seconds) or seconds < 0:
            raise ValueError('invalid elapsed time')
        key = start['condition'] + ':' + start['phase']
        phases[key] = phases.get(key, 0.) + seconds
    for path in sorted((out / 'attempts').glob('*/start.json')):
        if not path.with_suffix('.json.commit.json').exists():
            incomplete.append({'phase': 'uncommitted_invocation_start', 'condition': 'shared'})
            continue
        record = prior.read_record(path)
        if record['identity'] != identity:
            raise ValueError('invocation belongs to another study')
        end = path.with_name('end.json')
        if end.with_suffix('.json.commit.json').exists():
            result = prior.read_record(end)
            if result['identity'] != identity:
                raise ValueError('invocation completion identity differs')
            invocations.append(result)
        else:
            incomplete.append({'phase': 'whole_invocation', 'condition': 'shared'})
    return {
        'completed_or_failed_invocation_seconds': sum(r['elapsed_seconds'] for r in invocations),
        'phase_seconds_including_io': phases,
        'unknown_interrupted_work': incomplete,
        'complete_workflow_cost_known': not incomplete,
        'interruption_gap_seconds': None,
        'clean_efficiency_claim_allowed': False,
        'source_training_cost': 'inherited_dependency_not_measured_here',
        'shared_target_cost': 'executed_once; attribute_full_build_cost_to_each_refit_method_when_compared_standalone',
        'accounting': 'phase totals are subsets; do not add them to invocation totals',
    }


def expected_counts(condition: str, mask: list[bool]) -> dict:
    """Return exact per-epoch operation counts for the named condition and mask."""
    if condition not in SPEC['conditions']:
        raise ValueError('unknown screen condition')
    exact, other = sum(mask), len(mask) - sum(mask)
    learned = condition in ('saved', 'local32_refit', 'global32_refit')
    return {
        'exact_trunk_backwards': len(mask) if condition == 'exact_trunk_matched_head' else exact,
        'synthetic_backwards': other if learned else 0,
        'zero_backwards': other if condition == 'zero_fast' else 0,
        'skipped_slots': other if condition == 'skip_fast' else 0,
        'trunk_optimizer_steps': exact if condition == 'skip_fast' else len(mask),
        'head_optimizer_steps': exact,
        'predictor_fits': exact if learned else 0,
        'predictor_inferences': 2 * exact + other if learned else 0,
    }


def train_epoch(model, predictor, optimizers, dataset, config, device, scale,
                condition: str, global_epoch: int) -> dict:
    """Advance one condition by a full epoch on the shared fixed data/mask stream.

    Inputs include independent model/optimizer state and a zero-based epoch.
    Returns counters, masks and input-byte hash. Mutates training state and RNG.
    Uses original clipping, normalization, loss scaling and 16-epoch LR clock;
    last batches use their actual size. Nonfinite metrics or count mismatches fail.
    """
    loader = epoch_loader(dataset, config, global_epoch, shuffle=True)
    mask = make_uniform_third_mask([len(loader)])
    desired = expected_counts(condition, mask)
    lr = set_epoch_lr(optimizers, config, global_epoch)
    prior.set_training_mode(model, predictor, 'postwarm', SPEC['normalization'])
    counts, data_hash, samples = dict.fromkeys(COUNTERS, 0), hashlib.sha256(), 0
    for slot, (x, y, indices) in enumerate(loader):
        _update_data_hash(data_hash, x, y, indices)
        x, y = x.to(device), y.to(device)
        if mask[slot] or condition == 'exact_trunk_matched_head':
            result = exact_step(model, predictor, optimizers, x, y, scale, config,
                                fit_predictor=predictor is not None,
                                head_update=mask[slot])
        else:
            kind = {'zero_fast': 'zero', 'skip_fast': 'skip'}.get(condition, 'learned')
            result = synthetic_step(model, predictor, optimizers, x, y, scale,
                                    config, kind=kind, shadow_predictor=False)
        if any(isinstance(v, (int, float)) and not math.isfinite(v)
               for v in result.metrics.values() if v is not None):
            raise ValueError('nonfinite training metric; no epoch is committed')
        for name in COUNTERS:
            counts[name] += getattr(result, name)
        samples += len(y)
    if counts != desired or samples != len(dataset):
        raise ValueError('executed operation counts disagree with fixed schedule')
    return {'epoch': global_epoch + 1, 'lr': lr, 'samples': samples,
            'batches': len(mask), 'head_mask': mask, 'mask_sha256': utility.digest(mask),
            'data_order_sha256': data_hash.hexdigest(), 'counters': counts}


def checkpoint_state(model, predictor, optimizers, epoch, rows, identity,
                     environment, condition) -> dict:
    """Copy all continuation state to CPU for a safe immutable epoch checkpoint.

    Returns weights-only-loadable tensors/primitives including optimizer moments,
    BN buffers and RNG. Reads live state without updates; copying can synchronize
    CUDA. Module modes are deterministically restored before each training epoch.
    """
    return {'identity': identity, 'condition': condition, 'epoch': epoch,
            'model': _cpu_tree(model.state_dict()),
            'predictor': _cpu_tree(predictor.state_dict()) if predictor is not None else None,
            'optimizers': {k: _cpu_tree(v.state_dict()) if v is not None else None
                           for k, v in optimizers.items()},
            'rng': prior.capture_rng_state(), 'rows': copy.deepcopy(rows),
            'environment': environment}


def latest_online(directory: Path, identity: str, environment: dict):
    """Verify the contiguous epoch chain and return the latest CPU state, or None.

    Uncommitted checkpoint files are preserved under distinct names. Corrupt
    committed files, missing intermediate epochs, or incompatible environments
    fail instead of rolling back. Replacement T4 UUIDs are allowed, with the
    same software/numerical settings, but are never clean timing evidence.
    """
    latest = None
    gap = False
    for epoch in range(SPEC['source_epoch'], SPEC['end_epoch'] + 1):
        path = directory / f'epoch_{epoch:06d}.pt'
        prior.preserve_uncommitted_record(path)
        if not path.exists():
            gap = True
            continue
        if gap:
            raise ValueError('online checkpoint chain has a missing epoch')
        state = utility.read_tensors(path, identity)
        if state['epoch'] != epoch or len(state['rows']) != epoch - SPEC['source_epoch']:
            raise ValueError('checkpoint epoch/row count mismatch')
        if utility.comparison_environment(state['environment']) != utility.comparison_environment(environment):
            raise ValueError('online checkpoint environment changed beyond allocation identity')
        latest = state
    return latest


def construct_condition(payload, condition, device):
    """Create independent models and optimizers from the shared source checkpoint.

    Returns config/model/predictor/optimizers. Fresh predictors use the fixed
    initialization seed; controls omit predictor work. Source tensors remain
    unchanged and no files or training examples are accessed.
    """
    config, model, predictor, optimizers = prior.restore_models(payload, device)
    if condition.endswith('_refit'):
        predictor = utility.make_candidate(condition.removesuffix('_refit'), config,
                                           SPEC['fit_seed'], device)
        optimizers['predictor'] = utility.optimizer_for(predictor, config)
    elif condition != 'saved':
        predictor, optimizers['predictor'] = None, None
    restore_optimizer_layout(optimizers)
    return config, model, predictor, optimizers


def restore_optimizer_layout(optimizers) -> None:
    """Keep ordinary AdamW step counters on CPU after loading GPU moments.

    Accepts the optimizer mapping and returns nothing. The historical generic
    device helper moves every tensor to CUDA, including AdamW's scalar step;
    noncapturable, nonfused AdamW normally keeps that counter on CPU. Correct
    its storage without changing values, hyperparameters or parameter moments.
    """
    for optimizer in optimizers.values():
        if isinstance(optimizer, torch.optim.AdamW):
            for group in optimizer.param_groups:
                if not group.get('capturable', False) and not group.get('fused', False):
                    for parameter in group['params']:
                        state = optimizer.state.get(parameter, {})
                        if isinstance(state.get('step'), torch.Tensor):
                            state['step'] = state['step'].cpu()


def restore_online(state, model, predictor, optimizers) -> None:
    """Load saved model/optimizer/RNG state into an independent live condition.

    Takes a verified CPU snapshot. Returns nothing; mutates supplied objects and
    process RNG. Predictor-presence mismatches and state-shape errors fail.
    """
    model.load_state_dict(state['model'], strict=True)
    if (predictor is None) != (state['predictor'] is None):
        raise ValueError('saved predictor presence differs')
    if predictor is not None:
        predictor.load_state_dict(state['predictor'], strict=True)
    for name, optimizer in optimizers.items():
        saved = state['optimizers'][name]
        if (optimizer is None) != (saved is None):
            raise ValueError('saved optimizer presence differs')
        if optimizer is not None:
            optimizer.load_state_dict(saved)
            prior._restore_optimizer_devices(optimizer)
    restore_optimizer_layout(optimizers)
    prior.restore_rng_state(state['rng'])


def continuation_probe(model, predictor, optimizers, config, scale, device) -> dict:
    """Check two copied synthetic continuations while preserving the live state.

    Accepts any screen predictor architecture, including the global variant.
    Uses a private two-image batch and the existing M1 device tolerances. Returns
    evidence of same-device repeatability, not proof of cross-GPU equivalence.
    No production state is updated; temporary copies and RNG are discarded.
    """
    rng = prior.capture_rng_state()
    try:
        results = []
        for _ in range(2):
            prior.restore_rng_state(rng)
            local, pred, opts = copy.deepcopy((model, predictor, optimizers))
            prior.set_training_mode(local, pred, 'postwarm', SPEC['normalization'])
            generator = torch.Generator().manual_seed(48151623)
            x = torch.randn(2, 3, 32, 32, generator=generator).to(device)
            y = torch.tensor([1, 7], device=device)
            exact_step(local, pred, opts, x, y, scale, config,
                       fit_predictor=pred is not None, head_update=True)
            snapshot = checkpoint_state(local, pred, opts, 0, [], 'probe', {}, 'probe')
            results.append({k: snapshot[k] for k in ('model', 'predictor', 'optimizers')})
            del local, pred, opts, snapshot
        rtol, atol = (1e-4, 3e-5) if torch.device(device).type == 'cuda' else (1e-5, 1e-6)
        _assert_state_trees_close(results[0], results[1], rtol=rtol, atol=atol)
        return {'status': 'passed', 'rtol': rtol, 'atol': atol,
                'cross_device_equivalence_proven': False}
    finally:
        prior.restore_rng_state(rng)


def prepare_cache(out, payload, train, groups, device, identity, environment):
    """Load or construct the shared frozen training-target cache exactly once.

    Returns CPU feature/label/gradient triples. Uses only the fixed fitting sample
    positions and augmented views, never validation data. Writes a checksum-bound
    cache and phase costs. A crash before commit requires rebuilding that cache.
    """
    path = out / 'preparation' / 'targets.pt'
    path.parent.mkdir(parents=True, exist_ok=True)
    key = utility.digest({'study': identity, 'kind': 'targets'})
    prior.preserve_uncommitted_record(path)
    phase = 'target_cache_load' if path.exists() else 'target_cache_build'
    with measured_phase(out, phase, 'shared', identity, environment, device):
        if path.exists():
            cache = utility.read_tensors(path, key)['cache']
        else:
            print('Building and saving the one-time epoch-5 target cache.', flush=True)
            _, model, predictor, opts = prior.restore_models(payload, device)
            cache = utility.cached_batches(model, train, groups, payload['scale'],
                                            SPEC['augmentation_seed'], device)
            utility.save_fit(path, {'identity': key, 'cache': cache,
                                   'target_batches': len(groups), 'environment': environment})
            del model, predictor, opts
    if len(cache) != len(groups):
        raise ValueError('cache batch count differs')
    return cache


def verify_paired_row(out, condition, row, identity) -> None:
    """Reject changed data order or head slots against any already completed peer.

    Inputs are the new epoch row and study identity. Reads reports only; returns
    nothing. The full matrix is checked again before the study summary is saved.
    """
    for name in SPEC['conditions']:
        if name == condition:
            continue
        key = utility.digest({'study': identity, 'condition': name})
        report = utility.checked_record(out / name / 'report.json', key)
        if report is not None:
            other = next(r for r in report['epochs'] if r['epoch'] == row['epoch'])
            for field in ('data_order_sha256', 'head_mask', 'samples', 'lr'):
                if row[field] != other[field]:
                    raise ValueError('paired stream differs: ' + field)


def run_condition(payload, train, validation, out, condition, device, identity,
                  environment, cache_factory, baseline):
    """Execute or recover one two-epoch branch and return its immutable report.

    Independent full optimizer state is saved before training and after each
    epoch. Fresh fitting resumes every 64 updates; interrupted online epochs
    restart from their preceding boundary. No source state is changed.
    """
    key = utility.digest({'study': identity, 'condition': condition})
    directory = out / condition
    directory.mkdir(parents=True, exist_ok=True)
    state = latest_online(directory, key, environment)
    report_path = directory / 'report.json'
    report = utility.checked_record(report_path, key)
    if report is not None:
        if state is None or state['epoch'] != SPEC['end_epoch'] or report['epochs'] != state['rows']:
            raise ValueError('completed report disagrees with its checkpoints')
        print('Skip complete:', condition, flush=True)
        return report
    config, model, predictor, optimizers = construct_condition(payload, condition, device)
    if state is None:
        if condition.endswith('_refit'):
            cache = cache_factory()
            fit_key = utility.digest({'condition': key, 'cache': prior.cache_identity(cache)})
            fit_spec = {'fit_seed': SPEC['fit_seed'], 'updates': SPEC['fit_updates'],
                        'save_every': SPEC['fit_save_every']}
            with measured_phase(out, 'initial_predictor_fitting', condition, identity, environment, device):
                fit_dir = directory / 'fit'
                fit_dir.mkdir(parents=True, exist_ok=True)
                optimizers['predictor'], _ = utility.fit_candidate(
                    predictor, cache, config, fit_spec, device, fit_dir, fit_key)
                restore_optimizer_layout(optimizers)
        with measured_phase(out, 'initial_checkpoint', condition, identity, environment, device):
            state = checkpoint_state(model, predictor, optimizers, SPEC['source_epoch'],
                                     [], key, environment, condition)
            utility.save_fit(directory / f"epoch_{SPEC['source_epoch']:06d}.pt", state)
    else:
        if state['condition'] != condition:
            raise ValueError('checkpoint condition differs')
        with measured_phase(out, 'restore', condition, identity, environment, device):
            restore_online(state, model, predictor, optimizers)
        if state['environment'].get('device_uuid') != environment.get('device_uuid'):
            if utility.comparison_environment(environment).get('device_name') != 'NVIDIA T4':
                raise ValueError('replacement allocations are supported only for compatible T4s')
            with measured_phase(out, 'replacement_T4_probe', condition, identity, environment, device):
                result = continuation_probe(model, predictor, optimizers, config, payload['scale'], device)
                commit_json(directory / 'recovery' / (uuid.uuid4().hex + '.json'), {
                    'identity': key, 'checkpoint_epoch': state['epoch'],
                    'old_environment': state['environment'], 'new_environment': environment,
                    'checkpoint_sha256': prior.sha256_file(directory / f"epoch_{state['epoch']:06d}.pt"),
                    **result})
    rows = state['rows']
    start_epoch = state['epoch']
    del state
    for global_epoch in range(start_epoch, SPEC['end_epoch']):
        with measured_phase(out, 'online_epoch', condition, identity, environment, device):
            row = train_epoch(model, predictor, optimizers, train, config, device,
                              payload['scale'], condition, global_epoch)
        with measured_phase(out, 'validation', condition, identity, environment, device):
            row['validation'] = evaluate(model, validation, config, device)
        if not all(math.isfinite(row['validation'][k]) for k in ('ce', 'accuracy')):
            raise ValueError('nonfinite validation')
        verify_paired_row(out, condition, row, identity)
        rows.append(row)
        with measured_phase(out, 'epoch_checkpoint', condition, identity, environment, device):
            utility.save_fit(directory / f'epoch_{global_epoch+1:06d}.pt', checkpoint_state(
                model, predictor, optimizers, global_epoch + 1, rows, key, environment, condition))
        print(f"{condition}: committed epoch {global_epoch+1}; validation accuracy "
              f"{100*row['validation']['accuracy']:.2f}%", flush=True)
    report = {'identity': key, 'study_identity': identity, 'status': 'complete',
              'condition': condition, 'baseline': baseline, 'epochs': rows,
              'initial_fit_updates': SPEC['fit_updates'] if condition.endswith('_refit') else 0,
              'predictor_parameters': sum(p.numel() for p in predictor.parameters()) if predictor is not None else 0,
              'counter_scope': 'committed_training_only; interrupted/repeated_operation_counts_not_reconstructed',
              'environment_at_report': environment, 'claim_scope': SPEC['claim_scope']}
    commit_json(report_path, report)
    return report


def validate_inputs(args):
    """Verify source, existing dataset/split and protected paths before creating output.

    Returns resolved paths, CPU source state, config, datasets and device. No
    downloads or scientific updates occur. Source epoch, horizon and dataset
    sizes must match the approved screen; old config is never rewritten.
    """
    root = Path(__file__).resolve().parent
    out = prior.validate_output_root(args.out, root)
    source, data, split = (Path(p).resolve() for p in (args.checkpoint_dir, args.data_dir, args.split_file))
    for protected in (source.parent, data, split.parent):
        if out == protected or out.is_relative_to(protected) or protected.is_relative_to(out):
            raise ValueError('output must be separate from source, split and dataset')
    payload = prior.load_verified_source(source / f"checkpoint_{SPEC['source_epoch']:06d}.pt", SPEC['source_epoch'])
    config, device = prior.RunConfig(payload['resolved_config']), torch.device(args.device)
    training = config.scientific_dict()['training']
    if (training['warmup_epochs'] + training['postwarm_epochs'] != SPEC['lr_horizon']
            or training['batch_size'] != SPEC['batch_size'] or training['drop_last']):
        raise ValueError('source horizon/batch contract differs from approved screen')
    prior.configure_determinism(config, device)
    train, validation, identities = prior.load_cifar10_datasets(config, data, split)
    if identities != payload['data_identities']:
        raise ValueError('original split identity differs')
    if len(train) != SPEC['training_examples'] or len(validation) != SPEC['validation_examples']:
        raise ValueError('dataset sizes differ from approved screen')
    return root, out, split, payload, config, train, validation, device


def run_study(args) -> None:
    """Run all six approved branches with one command, or resume committed work.

    Real data requires explicit authorization. Writes only the new output tree;
    no historical source/plan/checkpoint is changed. Caches and epoch snapshots
    should live on Drive. Software changes fail rather than silently restarting.
    """
    if not args.authorize_online_screen:
        raise PermissionError('pass --authorize-online-screen for this bounded study')
    invoked = time.perf_counter()
    root, out, split, payload, config, train, validation, device = validate_inputs(args)
    environment = prior.environment_metadata(device)
    groups = utility.select_samples(len(train), utility.SPEC)['fit']
    if sum(map(len, groups)) != SPEC['fit_examples']:
        raise ValueError('fitting selection differs from the approved 4096-example rule')
    order = np.random.default_rng(SPEC['order_seed']).permutation(SPEC['conditions']).tolist()
    identity_data = {
        'spec': SPEC, 'source_sha256': payload['resume_metadata']['checkpoint_sha256'],
        'historical_artifacts': payload['artifacts'], 'split_sha256': prior.sha256_file(split),
        'scripts': {p.name: prior.sha256_file(p) for p in
                    (Path(__file__), Path(prior.__file__), Path(utility.__file__))},
        'comparison_environment': utility.comparison_environment(environment),
        'fit_training_positions': groups,
        'fit_official_training_ids': [[int(train.indices[i]) for i in b] for b in groups],
        'condition_order': order,
    }
    identity = utility.digest(identity_data)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / 'online_screen.json'
    if utility.checked_record(manifest_path, identity) is None:
        if any(not p.name.startswith('online_screen.uncommitted-') for p in out.iterdir()):
            raise ValueError('nonempty output has no compatible study manifest')
        commit_json(manifest_path, {'identity': identity, 'status': 'complete', **identity_data})
    attempt = out / 'attempts' / uuid.uuid4().hex
    commit_json(attempt / 'start.json', {'identity': identity, 'status': 'started',
                                       'environment': environment})
    status, error = 'complete', None
    try:
        baseline_path = out / 'baseline.json'
        baseline = utility.checked_record(baseline_path, identity)
        if baseline is None:
            with measured_phase(out, 'baseline_validation', 'shared', identity, environment, device):
                _, model, pred, opts = prior.restore_models(payload, device)
                baseline = {'identity': identity, 'status': 'complete',
                            'epoch': SPEC['source_epoch'],
                            'validation': evaluate(model, validation, config, device)}
                commit_json(baseline_path, baseline)
                del model, pred, opts
        cache = None

        def cache_factory():
            """Lazily load the one shared CPU cache; returns training triples, never evaluation data."""
            nonlocal cache
            if cache is None:
                cache = prepare_cache(out, payload, train, groups, device, identity, environment)
            return cache

        reports = []
        for number, condition in enumerate(order, 1):
            print(f'[{number}/{len(order)}] {condition}', flush=True)
            reports.append(run_condition(payload, train, validation, out, condition, device,
                                         identity, environment, cache_factory, baseline))
        for report in reports:
            for row in report['epochs']:
                verify_paired_row(out, report['condition'], row, identity)
        summary = {'identity': identity, 'status': 'complete', 'baseline': baseline,
                   'conditions': reports, 'claim_scope': SPEC['claim_scope'],
                   'source_run_id': SPEC['source_run_id'],
                   'note': 'exact_trunk_matched_head is not conventional full-head exact BP'}
        old = utility.checked_record(out / 'study_report.json', identity)
        if old is None:
            commit_json(out / 'study_report.json', summary)
        elif old != summary:
            raise ValueError('completed study summary disagrees with immutable reports')
    except BaseException as exc:
        status, error = 'failed', type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        synchronize(device)
        commit_json(attempt / 'end.json', {'identity': identity, 'status': status,
                    'error': error, 'elapsed_seconds': time.perf_counter() - invoked})
        commit_json(attempt / 'cost_summary.json', cost_report(out, identity))
    print('Online screen complete. Results and recovery state:', out, flush=True)


def describe() -> dict:
    """Return the locked screen and operation budget without touching data or models."""
    batches = math.ceil(SPEC['training_examples'] / SPEC['batch_size'])
    epochs = SPEC['end_epoch'] - SPEC['source_epoch']
    mask = make_uniform_third_mask([batches])
    return {'spec': SPEC, 'continuation_epochs_per_condition': epochs,
            'total_classifier_epochs': epochs * len(SPEC['conditions']),
            'total_classifier_batches': epochs * batches * len(SPEC['conditions']),
            'initial_predictor_updates': 2 * SPEC['fit_updates'],
            'one_time_target_batches': SPEC['fit_examples'] // SPEC['batch_size'],
            'per_epoch_operations': {c: expected_counts(c, mask) for c in SPEC['conditions']},
            'estimated_GPU_hours': None,
            'target_cache_GiB_approx': 1.0,
            'storage_note': 'additional full model/optimizer states for 3 boundaries x 6 branches',
            'authorization': 'implementation and synthetic tests only; user launches Colab explicitly'}


def export_reports(out: Path, destination: Path) -> None:
    """Export committed JSON evidence without multi-GB tensors; never overwrite files.

    Inputs are a completed study directory and a new ZIP path. Checks every JSON
    checksum before writing. Tensor markers alone cannot verify omitted tensors.
    Reports retain paths relative to the study for independent local analysis.
    """
    prior.read_record(out / 'study_report.json')
    paths = sorted(out.rglob('*.json'))
    for path in paths:
        if not path.name.endswith('.commit.json'):
            if not path.with_suffix('.json.commit.json').exists():
                continue
            prior.read_record(path)
    with zipfile.ZipFile(destination, 'x', zipfile.ZIP_DEFLATED) as bundle:
        for path in paths:
            if path.name.endswith('.commit.json') or path.with_suffix('.json.commit.json').exists():
                bundle.write(path, path.relative_to(out))


def main(argv=None) -> int:
    """Describe, explicitly execute/recover, or export the fixed screen; return an exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--authorize-online-screen', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    for name in ('checkpoint-dir', 'data-dir', 'split-file', 'out', 'export-reports'):
        parser.add_argument('--' + name)
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps(describe(), indent=2))
        return 0
    if args.export_reports:
        if not args.out:
            parser.error('--export-reports requires --out')
        export_reports(Path(args.out), Path(args.export_reports))
        return 0
    if any(getattr(args, name) is None for name in ('checkpoint_dir', 'data_dir', 'split_file', 'out')):
        parser.error('provide checkpoint, data, saved split and separate output paths')
    run_study(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
