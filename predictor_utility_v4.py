"""Prepare and run a separate, bounded predictor-utility diagnostic.

Historical sources and studies remain immutable. This follow-up measures fresh
gradient generalization and copied-state update utility, not classifier accuracy
or deployable speed. Real-data execution requires an explicit CLI flag. Completed
fits and optimizer checkpoints are checksum verified and reused after interruption.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import time
import uuid

import numpy as np
import torch
from torch.utils.data import Subset

import predictor_investigation_v4 as prior
from gradient_phase_d_resnet_v4 import exact_step, synthetic_step
from phase_d_v4_diagnostics import _eval_loss


SPEC = {
    "schema": "predictor_utility_v1",
    "status": "exploratory_development_only",
    "checkpoint_sequences": [5, 16],
    "batch_size": 32,
    "fit_examples": [256, 4096],
    "evaluation_examples": 512,
    "episodes": 8,
    "future_steps": 4,
    "updates": 256,
    "save_every": 64,
    "selection_seed": 88101,
    "fit_seed": 88102,
    "augmentation_seed": 88103,
    "variants": ["local32", "local64", "global32", "global64"],
    "fit_transform": "one_fixed_augmented_view_per_example",
    "evaluation_transform": "deterministic",
    "probe_transform": "augmented_anchor_and_future_deterministic_evaluation",
    "budget": "equal_updates_and_examples_processed_not_equal_FLOPs",
    "objective": "unchanged_reference_predictor_loss_optimizer_and_clipping",
    "excluded_samples": "all_512_positions_from_prior_investigation",
    "evaluation_scope": "unseen_by_fresh_fit_but_previously_seen_by_source_classifier",
    "controller_probe": "prefit_score_then_exact_anchor_then_copied_future_branches",
    "statistical_unit": "one_classifier_seed_no_inferential_intervals",
    "scheduled_baseline_label": "ADA-GP-inspired_schedule_not_ADA-GP_reproduction",
    "total_compute_savings": None,
    "practical_accuracy_margin_pp": None,
    "claim_status": "no_online_controller_accuracy_convergence_or_efficiency_claim",
}


def digest(value: dict) -> str:
    """Return a stable SHA-256 for JSON identities; no files or state are changed."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def comparison_environment(environment: dict) -> dict:
    """Keep hardware/software comparison settings while allowing a replacement GPU UUID.

    Full environment records remain in snapshots/reports. Only physical allocation
    and detailed host descriptions are omitted from identity; device model, memory,
    libraries, driver version and precision/determinism settings must still match.
    """
    omitted = {'device_uuid', 'device_index', 'platform', 'processor', 'driver_query'}
    result = {k: v for k,v in environment.items() if k not in omitted}
    if result.get('device_name') in ('Tesla T4', 'NVIDIA T4'):
        result['device_name'] = 'NVIDIA T4'
    return result


def select_samples(size: int, spec: dict) -> dict:
    """Select disjoint training positions, excluding every previously inspected group.

    Returns nested fitting prefixes, fresh evaluation and independent episodes of
    anchor/future/evaluation batches. Uses a private RNG; too few examples fail.
    These are fresh predictor-development observations, not a new classifier test.
    """
    old = prior.select_groups(size, prior.SPEC)
    excluded = {i for batches in old.values() for batch in batches for i in batch}
    b = spec['batch_size']
    counts = spec['fit_examples']
    if (not counts or any(n <= 0 or n % b for n in counts)
            or counts != sorted(set(counts)) or spec['evaluation_examples'] % b
            or spec['evaluation_examples'] <= 0 or spec['episodes'] < 2
            or spec['future_steps'] < 1):
        raise ValueError('invalid sample sizes or episode budget')
    needed = max(counts) + spec['evaluation_examples'] + spec['episodes'] * (spec['future_steps'] + 2) * b
    pool = np.asarray([i for i in range(size) if i not in excluded])
    if needed > len(pool):
        raise ValueError('insufficient training examples after excluding old development samples')
    chosen = np.random.default_rng(spec['selection_seed']).permutation(pool)[:needed].tolist()
    batches = [chosen[i:i+b] for i in range(0, needed, b)]
    nfit, neval = max(counts)//b, spec['evaluation_examples']//b
    probes = batches[nfit+neval:]
    width = spec['future_steps'] + 2
    return {'fit': batches[:nfit], 'evaluation': batches[nfit:nfit+neval],
            'episodes': [probes[i:i+width] for i in range(0, len(probes), width)],
            'excluded': sorted(excluded)}


def make_candidate(name: str, config, seed: int, device):
    """Create a local/global width-32/64 candidate without consuming caller RNG.

    Returns a trainable predictor. Width and context form a factorial comparison;
    changing either is explicit. Unknown variants fail before model construction.
    """
    if name not in ('local32', 'local64', 'global32', 'global64'):
        raise ValueError('unknown predictor variant')
    rng = prior.capture_rng_state()
    try:
        torch.manual_seed(seed)
        cfg = config.scientific_dict()['predictor']
        cls = prior.GlobalPredictor if name.startswith('global') else prior.GradientPredictor
        return cls(cfg['input_channels'], int(name[-2:]), cfg['label_embedding_classes']).to(device)
    finally:
        prior.restore_rng_state(rng)


def optimizer_for(predictor, config):
    """Return fresh reference AdamW bound to this predictor; no updates are performed."""
    s = config.scientific_dict()['optimizers']['predictor']
    return torch.optim.AdamW(predictor.parameters(), lr=s['lr'], weight_decay=s['weight_decay'],
                            betas=tuple(s['betas']), eps=s['eps'])


def save_fit(path: Path, state: dict) -> None:
    """Publish an immutable optimizer snapshot and checksum marker, preserving partials.

    Writes only the supplied new path. A crash before the marker leaves evidence
    which the next call moves aside; committed snapshots are never overwritten.
    """
    prior.preserve_uncommitted_record(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    prior.atomic_write_json(path.with_suffix('.pt.commit.json'),
                            {'sha256': prior.sha256_file(path), 'size_bytes': path.stat().st_size})


def latest_fit(directory: Path, identity: str):
    """Verify all committed fit snapshots and return the latest CPU state or None.

    Uses weights-only loading. Corrupt, orphaned, or incompatible committed state
    raises; it is never silently replaced by an earlier checkpoint.
    """
    latest = None
    for marker in sorted(directory.glob('fit_*.pt.commit.json')):
        path = marker.with_name(marker.name.removesuffix('.commit.json'))
        state = read_tensors(path, identity)
        if path.name != f"fit_{state['update']:06d}.pt":
            raise ValueError('incompatible fitting state')
        latest = state
    return latest


def read_tensors(path: Path, identity: str) -> dict:
    """Read a checksum-committed tensor cache/snapshot on CPU, rejecting corruption.

    Only weights-only deserialization is permitted. Identity includes the source,
    sample selection, code and comparison environment; files remain unchanged.
    """
    marker = path.with_suffix(path.suffix+'.commit.json')
    record = json.loads(marker.read_text())
    if (not path.is_file() or path.stat().st_size != record['size_bytes']
            or prior.sha256_file(path) != record['sha256']):
        raise ValueError(f'corrupt committed tensors: {path}')
    state = torch.load(path, map_location='cpu', weights_only=True)
    if state['identity'] != identity:
        raise ValueError('incompatible tensor identity')
    return state


def fit_candidate(candidate, cache, config, spec, device, directory: Path, identity: str):
    """Fit only on training triples and resume the exact AdamW/order state if saved.

    Cache tensors are CPU [B,C,H,W], [B], [B,C,H,W]. Returns AdamW and telemetry;
    only candidate parameters and new snapshot files change. No evaluation data
    enters fitting. Nonfinite values fail, and caller RNG is restored on failure.
    """
    rng = prior.capture_rng_state()
    try:
        optimizer = optimizer_for(candidate, config)
        saved = latest_fit(directory, identity)
        generator = np.random.default_rng(spec['fit_seed'])
        order, telemetry, start = [], [], 0
        if saved is not None:
            if comparison_environment(saved['environment']) != comparison_environment(prior.environment_metadata(device)):
                raise ValueError('fitting environment changed beyond allocation identity')
            candidate.load_state_dict(saved['predictor'])
            optimizer.load_state_dict(saved['optimizer'])
            prior._restore_optimizer_devices(optimizer)
            generator.bit_generator.state = saved['order_rng']
            order, telemetry, start = saved['remaining_order'], saved['telemetry'], saved['update']
            if start > spec['updates'] or len(telemetry) != start:
                raise ValueError('invalid completed-update count')
        if not cache or spec['updates'] < 1 or spec['save_every'] < 1:
            raise ValueError('fitting needs nonempty data and positive update/save budgets')
        candidate.train()
        c = config.scientific_dict()['gradient_contract']
        for update in range(start, spec['updates']):
            if not order:
                order = generator.permutation(len(cache)).tolist()
            batch_id = order.pop(0)
            h, y, target = [t.to(device) for t in cache[batch_id]]
            optimizer.zero_grad(set_to_none=True)
            loss, mse, cosine = prior.predictor_loss_components(candidate(h, y), target,
                cosine_weight=c['cosine_weight'], valid_norm_floor=c['valid_target_flat_norm_floor'],
                cosine_eps=c['cosine_eps'])
            if not bool(torch.isfinite(loss)):
                raise ValueError('nonfinite predictor objective')
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(candidate.parameters(), c['global_l2_clip'],
                                                  error_if_nonfinite=True)
            optimizer.step()
            telemetry.append({'update': update+1, 'batch': batch_id, 'mse': float(mse.detach()),
                              'weighted_cosine': c['cosine_weight']*float(cosine.detach()),
                              'clip_coefficient': min(1., c['global_l2_clip']/(float(norm)+1e-6))})
            if (update+1) % spec['save_every'] == 0 or update+1 == spec['updates']:
                save_fit(directory/f'fit_{update+1:06d}.pt', {
                    'identity': identity, 'update': update+1, 'predictor': candidate.state_dict(),
                    'optimizer': optimizer.state_dict(), 'order_rng': generator.bit_generator.state,
                    'remaining_order': order, 'telemetry': telemetry,
                    'environment': prior.environment_metadata(device)})
                print(f'  committed fit update {update+1}/{spec["updates"]}', flush=True)
        return optimizer, telemetry
    finally:
        prior.restore_rng_state(rng)


def future_utility(model, predictor, optimizers, episode, scale, config) -> dict:
    """Test whether today's pre-fit score anticipates useful future updates on copies.

    Episode contains one anchor, future batches, and a separate evaluation batch.
    First execute the real exact anchor with predictor fitting and head update.
    Then branch into exact/learned/zero/skip trunk updates, with no head updates,
    preserving saved SGD momentum/decay and candidate AdamW state. Positive
    zero-minus-learned evaluation CE means learned helped over zero. No source
    state changes; CE is a local training-data diagnostic, not test accuracy.
    """
    if len(episode) < 3:
        raise ValueError('an episode needs anchor, future, and evaluation batches')
    rng = prior.capture_rng_state()
    try:
        base, pred, opts = copy.deepcopy((model, predictor, optimizers))
        prior.set_training_mode(base, pred, 'postwarm', config.scientific_dict()['identity']['normalization'])
        anchor = exact_step(base, pred, opts, *episode[0], scale, config, fit_predictor=True)
        before = _eval_loss(base, *episode[-1])
        if not all(math.isfinite(v) for v in (before, anchor.metrics['error_relative'],
                                              anchor.metrics['error_abs'])):
            raise ValueError('nonfinite anchor score or evaluation loss')
        rows = []
        for kind in ('exact', 'learned', 'zero', 'skip'):
            local, local_pred, local_opts = copy.deepcopy((base, pred, opts))
            for step, batch in enumerate(episode[1:-1], 1):
                if kind == 'exact':
                    exact_step(local, local_pred, local_opts, *batch, scale, config,
                               fit_predictor=False, head_update=False)
                else:
                    synthetic_step(local, local_pred, local_opts, *batch, scale, config, kind=kind)
                after = _eval_loss(local, *episode[-1])
                if not math.isfinite(after):
                    raise ValueError('nonfinite future-utility loss')
                rows.append({'alternative': kind, 'future_steps': step,
                             'evaluation_ce': after, 'evaluation_ce_change': after-before})
            del local, local_pred, local_opts
        benefits = []
        for step in range(1, len(episode)-1):
            losses = {r['alternative']: r['evaluation_ce'] for r in rows if r['future_steps'] == step}
            benefits.append({'future_steps': step,
                             'zero_minus_learned_ce': losses['zero']-losses['learned'],
                             'skip_minus_learned_ce': losses['skip']-losses['learned'],
                             'learned_minus_exact_ce': losses['learned']-losses['exact']})
        return {'anchor_prefit_relative_error': anchor.metrics['error_relative'],
                'anchor_prefit_absolute_error': anchor.metrics['error_abs'],
                'anchor_target_rms': anchor.metrics['target_rms'],
                'evaluation_ce_after_anchor': before, 'rows': rows, 'benefits': benefits,
                'scope': 'independent_copied_episodes_not_online_controller_evaluation'}
    finally:
        prior.restore_rng_state(rng)


def checked_record(path: Path, identity: str):
    """Return an existing compatible committed report, or None; preserve partial files."""
    prior.preserve_uncommitted_record(path)
    if not path.exists():
        return None
    record = prior.read_record(path)
    if record.get('identity') != identity or record.get('status') != 'complete':
        raise ValueError(f'incompatible report: {path}')
    return record


def cached_batches(model, dataset, groups, scale, seed, device):
    """Build CPU targets one image batch at a time to bound accelerator memory.

    Inputs are training-only dataset positions. Returns frozen cache triples;
    model and caller RNG are preserved by the reused state-isolated helpers.
    """
    def batches():
        """Yield one materialized device batch at a time; preserve augmentation RNG."""
        for i, ids in enumerate(groups):
            yield prior.image_batches(dataset, [ids], device, seed+i)[0]
    return prior.cache_targets(model, batches(), scale)


def summarize_episodes(episodes: list) -> list[dict]:
    """Summarize local benefit and score association, without seed-level inference.

    Positive benefit means lower learned CE than the comparator. A negative
    error/benefit correlation is consistent with lower error preceding greater
    benefit. Constant scores/benefits yield null, never a fabricated correlation.
    """
    scores = np.asarray([e['anchor_prefit_relative_error'] for e in episodes], dtype=float)
    summaries = []
    for step in range(1, len(episodes[0]['benefits'])+1):
        for comparator in ('zero', 'skip'):
            field = comparator+'_minus_learned_ce'
            values = np.asarray([e['benefits'][step-1][field] for e in episodes], dtype=float)
            correlation = None
            if len(scores) > 1 and np.std(scores) > 0 and np.std(values) > 0:
                correlation = float(np.corrcoef(scores, values)[0, 1])
            summaries.append({'future_steps': step, 'comparator': comparator,
                              'mean_ce_benefit': float(values.mean()),
                              'positive_benefit_episodes': int((values > 0).sum()),
                              'episodes': len(episodes), 'error_benefit_pearson': correlation,
                              'status': 'descriptive_within_one_checkpoint_no_confidence_interval'})
    return summaries


def run_study(args) -> None:
    """Run the explicit diagnostic on verified local inputs; never download data.

    Writes only a new study folder, preferably directly on mounted Drive. Reruns
    verify and skip complete conditions/episodes and resume saved fitting state.
    All timing is diagnostic overhead; environments are recorded per artifact.
    """
    if not args.authorize_utility_study:
        raise PermissionError('real-data diagnostics require --authorize-utility-study')
    root = Path(__file__).resolve().parent
    out = prior.validate_output_root(args.out, root)
    source, split, data = [Path(p).resolve() for p in (args.checkpoint_dir, args.split_file, args.data_dir)]
    for protected in (source.parent, split.parent, data):
        if out == protected or out.is_relative_to(protected) or protected.is_relative_to(out):
            raise ValueError('output must be separate from data, split and source study')
    hashes, first = {}, None
    for seq in SPEC['checkpoint_sequences']:
        payload = prior.load_verified_source(source/f'checkpoint_{seq:06d}.pt', seq)
        hashes[str(seq)] = payload['resume_metadata']['checkpoint_sha256']
        if first is not None and payload['scientific_fingerprint'] != first['scientific_fingerprint']:
            raise ValueError('source checkpoints mix conditions')
        if first is None:
            first = payload
    config, device = prior.RunConfig(first['resolved_config']), torch.device(args.device)
    prior.configure_determinism(config, device)
    augmented, validation, identities = prior.load_cifar10_datasets(config, data, split)
    if identities != first['data_identities']:
        raise ValueError('source data split differs')
    deterministic = Subset(validation.dataset, augmented.indices)
    groups = select_samples(len(augmented), SPEC)
    selected = {i for batches in (groups['fit'], groups['evaluation'], *groups['episodes'])
                for batch in batches for i in batch}
    official = {str(i): int(augmented.indices[i]) for i in sorted(selected)}
    identity = {'spec': SPEC, 'source_hashes': hashes, 'split_sha256': prior.sha256_file(split),
                'script_sha256': prior.sha256_file(__file__), 'prior_sha256': prior.sha256_file(prior.__file__),
                'comparison_environment': comparison_environment(prior.environment_metadata(device)),
                'groups': groups, 'official_training_ids': official}
    key = digest(identity)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out/'utility_study.json'
    if checked_record(manifest, key) is None:
        if any(not p.name.startswith('utility_study.uncommitted-') for p in out.iterdir()):
            raise ValueError('nonempty output lacks a compatible study manifest')
        prior.write_record(manifest, {'status': 'complete', 'identity': key, **identity})
    del first, payload, validation
    for seq in SPEC['checkpoint_sequences']:
        conditions = [('saved', 0)] + [(v, n) for n in SPEC['fit_examples'] for v in SPEC['variants']]
        def condition_identity(variant, count):
            """Bind a report to its condition so swapped report files are rejected."""
            return digest({'study': key, 'checkpoint': seq, 'variant': variant, 'examples': count})
        if all(checked_record(out/f'epoch{seq}_{v}_n{n}'/'report.json', condition_identity(v,n))
               is not None for v,n in conditions):
            print(f'Skip complete checkpoint {seq}', flush=True)
            continue
        payload = prior.load_verified_source(source/f'checkpoint_{seq:06d}.pt', seq)
        if payload['resume_metadata']['checkpoint_sha256'] != hashes[str(seq)]:
            raise ValueError('source changed during study')
        config, model, saved, optimizers = prior.restore_models(payload, device)
        cache_path = out/f'epoch{seq}_targets.pt'
        cache_key = digest({'study': key, 'checkpoint': seq, 'kind': 'frozen_targets'})
        prior.preserve_uncommitted_record(cache_path)
        if cache_path.exists():
            targets = read_tensors(cache_path, cache_key)
            cache, evaluation = targets['fit'], targets['evaluation']
            del targets
        else:
            cache = cached_batches(model, augmented, groups['fit'], payload['scale'], SPEC['augmentation_seed'], device)
            evaluation = cached_batches(model, deterministic, groups['evaluation'], payload['scale'], SPEC['augmentation_seed'], device)
            save_fit(cache_path, {'identity': cache_key, 'fit': cache, 'evaluation': evaluation,
                                  'environment': prior.environment_metadata(device)})
        for variant, n in conditions:
            directory = out/f'epoch{seq}_{variant}_n{n}'
            condition_key = condition_identity(variant, n)
            if checked_record(directory/'report.json', condition_key) is not None:
                continue
            directory.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            candidate = saved if variant == 'saved' else make_candidate(variant, config, SPEC['fit_seed'], device)
            fit_cache = cache[:n//SPEC['batch_size']]
            fit_identity = digest({'study': key, 'checkpoint': seq, 'variant': variant, 'examples': n,
                                   'cache': prior.cache_identity(fit_cache)})
            if variant == 'saved':
                candidate_optimizer, telemetry = optimizers['predictor'], []
            else:
                candidate_optimizer, telemetry = fit_candidate(candidate, fit_cache, config, SPEC, device,
                                                               directory, fit_identity)
            local_opts = {**optimizers, 'predictor': candidate_optimizer}
            episodes = []
            for i, batches in enumerate(groups['episodes']):
                episode_path = directory/f'episode_{i:03d}.json'
                record = checked_record(episode_path, fit_identity)
                if record is None:
                    inputs = prior.image_batches(augmented, batches[:-1], device, SPEC['augmentation_seed']+i)
                    inputs += prior.image_batches(deterministic, batches[-1:], device, SPEC['augmentation_seed'])
                    result = future_utility(model, candidate, local_opts, inputs, payload['scale'], config)
                    record = {'status': 'complete', 'identity': fit_identity,
                              'environment': prior.environment_metadata(device), **result}
                    prior.write_record(episode_path, record)
                    del inputs
                episodes.append(record)
            report = {'status': 'complete', 'identity': condition_key, 'study_identity': key,
                      'fit_identity': fit_identity,
                      'variant': variant, 'fit_examples': n, 'checkpoint': seq,
                      'parameters': sum(p.numel() for p in candidate.parameters()),
                      'environment': prior.environment_metadata(device), 'training_telemetry': telemetry,
                      'fitting_passes': SPEC['updates']*SPEC['batch_size']/n if n else 0,
                      'evaluation': prior.prediction_report(candidate, evaluation, device),
                      'zero_evaluation': prior.prediction_report(None, evaluation, device),
                      'episodes': episodes, 'utility_summary': summarize_episodes(episodes),
                      'this_invocation_seconds_diagnostic_only': time.perf_counter()-started,
                      'claim_status': SPEC['claim_status']}
            prior.write_record(directory/'report.json', report)
            print(f'Completed checkpoint {seq}: {variant}, {n} fit examples', flush=True)
            del candidate, candidate_optimizer, local_opts
        del payload, model, saved, optimizers, cache, evaluation


def main(argv=None) -> int:
    """Print the exact proposal or run it only with explicit paths and authorization."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--authorize-utility-study', action='store_true')
    for name in ('checkpoint-dir', 'data-dir', 'split-file', 'out'):
        parser.add_argument('--'+name)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps({'spec': SPEC, 'fresh_fits': 16, 'saved_predictor_conditions': 2,
                          'classifier_training_trajectories': 0, 'measured_GPU_hours': None}, indent=2))
        return 0
    if any(getattr(args, k) is None for k in ('checkpoint_dir', 'data_dir', 'split_file', 'out')):
        parser.error('provide checkpoint, dataset, split and new output paths')
    run_study(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
