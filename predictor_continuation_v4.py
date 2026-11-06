"""Continue the approved six-condition screen from epoch 7 through epoch 16.

The completed two-epoch screen stays read-only. This separate study restores
its full branch states, performs no initial refitting, and records new costs.
It reuses the original training implementation without changing its SPEC.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
import uuid

import predictor_online_screen_v4 as screen

prior, utility = screen.prior, screen.utility
SPEC = {
    'schema': 'predictor_continuation_v1',
    'source_epoch': 7,
    'end_epoch': 16,
    'lr_horizon': 16,
    'conditions': list(screen.SPEC['conditions']),
    'condition_order': 'retain_completed_screen_order',
    'initial_refitting': False,
    'checkpointing': 'import_epoch_7_then_each_completed_epoch',
    'claim_scope': 'exploratory_extension_selected_after_two_epoch_screen_not_convergence_proof',
}


def check_rows(rows, condition, epoch):
    """Check recorded epochs, fixed masks and operation counts; return nothing.

    Rows include inherited epochs 6-7 and any new epochs through the boundary.
    This reads metadata only; malformed or nonfinite records raise ValueError.
    """
    if [r['epoch'] for r in rows] != list(range(screen.SPEC['source_epoch'] + 1, epoch + 1)):
        raise ValueError('missing or duplicated epoch rows')
    batches = math.ceil(screen.SPEC['training_examples'] / screen.SPEC['batch_size'])
    mask = screen.make_uniform_third_mask([batches])
    for row in rows:
        if (row['head_mask'] != mask or row['mask_sha256'] != utility.digest(mask)
                or row['batches'] != batches
                or row['samples'] != screen.SPEC['training_examples']
                or row['counters'] != screen.expected_counts(condition, mask)
                or row['validation']['samples'] != screen.SPEC['validation_examples']):
            raise ValueError('recorded schedule, counts or sample sizes differ')
        if not (0 <= row['validation']['accuracy'] <= 1
                and math.isfinite(row['validation']['ce'])
                and math.isfinite(row['lr'])):
            raise ValueError('invalid validation or learning-rate record')


def paired_rows(reports):
    """Require identical input streams, LR and head slots across reports; return nothing."""
    if not reports:
        return
    reference = reports[0]['epochs']
    for report in reports[1:]:
        if len(report['epochs']) != len(reference):
            raise ValueError('paired reports have different epoch counts')
        for row, other in zip(report['epochs'], reference):
            for key in ('epoch', 'data_order_sha256', 'head_mask', 'samples', 'lr'):
                if row[key] != other[key]:
                    raise ValueError('paired continuation differs: ' + key)


def load_screen(directory, payload, split, environment):
    """Verify all six source branches before any continuation training.

    Reads committed JSON and CPU epoch-7 tensors without changing source files.
    Returns the manifest, summary, checkpoint hashes and inherited cost evidence.
    A report-only ZIP is insufficient; missing/corrupt tensors or changed code,
    dataset identity, protocol or numerical environment fail before updates.
    """
    manifest = prior.read_record(directory / 'online_screen.json')
    identity = manifest['identity']
    body = {k: v for k, v in manifest.items() if k not in ('identity', 'status')}
    if (manifest['status'] != 'complete' or utility.digest(body) != identity
            or manifest['spec'] != screen.SPEC):
        raise ValueError('source screen identity or specification differs')
    for name, expected in manifest['scripts'].items():
        if prior.sha256_file(Path(screen.__file__).parent / name) != expected:
            raise ValueError('screen dependency changed: ' + name)
    if (manifest['source_sha256'] != payload['resume_metadata']['checkpoint_sha256']
            or manifest['historical_artifacts'] != payload['artifacts']
            or manifest['split_sha256'] != prior.sha256_file(split)):
        raise ValueError('screen source checkpoint, artifacts or split differs')
    if manifest['comparison_environment'] != utility.comparison_environment(environment):
        raise ValueError('screen environment differs beyond compatible allocation identity')
    summary = prior.read_record(directory / 'study_report.json')
    baseline = prior.read_record(directory / 'baseline.json')
    if (summary['identity'] != identity or summary['status'] != 'complete'
            or summary['baseline'] != baseline or baseline['identity'] != identity):
        raise ValueError('source screen is incomplete or has inconsistent baseline')
    names = [r['condition'] for r in summary['conditions']]
    if names != manifest['condition_order'] or sorted(names) != sorted(SPEC['conditions']):
        raise ValueError('all six completed source conditions are required')
    hashes = {}
    for report in summary['conditions']:
        name = report['condition']
        key = utility.digest({'study': identity, 'condition': name})
        if (report != prior.read_record(directory / name / 'report.json')
                or report['identity'] != key or report['study_identity'] != identity
                or report['status'] != 'complete' or report['baseline'] != baseline):
            raise ValueError('source branch report differs')
        check_rows(report['epochs'], name, SPEC['source_epoch'])
        path = directory / name / f"epoch_{SPEC['source_epoch']:06d}.pt"
        state = utility.read_tensors(path, key)
        if (state['epoch'] != SPEC['source_epoch'] or state['condition'] != name
                or state['rows'] != report['epochs']
                or utility.comparison_environment(state['environment']) != manifest['comparison_environment']):
            raise ValueError('source checkpoint and branch report disagree')
        hashes[name] = prior.sha256_file(path)
        del state
    paired_rows(summary['conditions'])
    # Retain the raw small records so inherited costs can be independently audited.
    ledger = {}
    for folder in ('costs', 'attempts'):
        for path in sorted((directory / folder).rglob('*.json')):
            if not path.name.endswith('.commit.json') and path.with_suffix('.json.commit.json').exists():
                ledger[path.relative_to(directory).as_posix()] = prior.read_record(path)
    evidence = {'manifest': manifest, 'summary': summary, 'cost_records': ledger,
                'cost_summary': screen.cost_report(directory, identity),
                'epoch7_checkpoint_sha256': hashes}
    return manifest, summary, hashes, evidence


def latest_state(directory, key, condition, environment, inherited_rows):
    """Load the latest contiguous continuation checkpoint, preserving partial writes.

    Returns CPU state or None. Reads every saved boundary from 7 onward and
    rejects corruption, missing middle epochs, altered history or environments.
    Only uncommitted files in the new output directory may be renamed.
    """
    latest, gap = None, False
    for epoch in range(SPEC['source_epoch'], SPEC['end_epoch'] + 1):
        path = directory / f'epoch_{epoch:06d}.pt'
        prior.preserve_uncommitted_record(path)
        if not path.exists():
            gap = True
            continue
        if gap:
            raise ValueError('continuation checkpoint chain has a missing epoch')
        state = utility.read_tensors(path, key)
        if (state['epoch'] != epoch or state['condition'] != condition
                or state['rows'][:len(inherited_rows)] != inherited_rows
                or utility.comparison_environment(state['environment']) != utility.comparison_environment(environment)):
            raise ValueError('continuation checkpoint identity or inherited history differs')
        check_rows(state['rows'], condition, epoch)
        latest = state
    return latest


def continue_condition(payload, train, validation, source, out, source_report,
                       source_hash, identity, environment, device):
    """Restore one branch at its saved boundary and run only its remaining epochs.

    Returns a complete report containing epochs 6-16. Restores all weights,
    optimizer moments, BN buffers and RNG; no initial fit or target cache is used.
    Writes immutable checkpoints, validation and timing records only under out.
    """
    name = source_report['condition']
    key = utility.digest({'study': identity, 'condition': name})
    directory = out / name
    directory.mkdir(parents=True, exist_ok=True)
    state = latest_state(directory, key, name, environment, source_report['epochs'])
    report_path = directory / 'report.json'
    completed = utility.checked_record(report_path, key)
    if completed is not None:
        if state is None or state['epoch'] != SPEC['end_epoch'] or completed['epochs'] != state['rows']:
            raise ValueError('completed continuation report disagrees with checkpoint')
        print('Skip complete:', name, flush=True)
        return completed
    with screen.measured_phase(out, 'restore', name, identity, environment, device):
        config, model, predictor, optimizers = screen.construct_condition(payload, name, device)
        if state is None:
            path = source / name / f"epoch_{SPEC['source_epoch']:06d}.pt"
            if prior.sha256_file(path) != source_hash:
                raise ValueError('source changed after preflight validation')
            state = utility.read_tensors(path, source_report['identity'])
            # New identity describes provenance; all numerical state stays identical.
            state['identity'] = key
            utility.save_fit(directory / path.name, state)
        screen.restore_online(state, model, predictor, optimizers)
    if state['environment'].get('device_uuid') != environment.get('device_uuid'):
        if utility.comparison_environment(environment).get('device_name') != 'NVIDIA T4':
            raise ValueError('replacement allocations require compatible T4s')
        with screen.measured_phase(out, 'replacement_T4_probe', name, identity, environment, device):
            result = screen.continuation_probe(model, predictor, optimizers, config, payload['scale'], device)
            screen.commit_json(directory / 'recovery' / (uuid.uuid4().hex + '.json'), {
                'identity': key, 'epoch': state['epoch'], 'old_environment': state['environment'],
                'new_environment': environment, **result})
    rows, start_epoch = state['rows'], state['epoch']
    del state
    for global_epoch in range(start_epoch, SPEC['end_epoch']):
        with screen.measured_phase(out, 'online_epoch', name, identity, environment, device):
            row = screen.train_epoch(model, predictor, optimizers, train, config, device,
                                     payload['scale'], name, global_epoch)
        with screen.measured_phase(out, 'validation', name, identity, environment, device):
            row['validation'] = screen.evaluate(model, validation, config, device)
        rows.append(row)
        check_rows(rows, name, global_epoch + 1)
        # Compare the current epoch with any peer that has already finished.
        for peer in SPEC['conditions']:
            peer_key = utility.digest({'study': identity, 'condition': peer})
            other = utility.checked_record(out / peer / 'report.json', peer_key)
            if other is not None:
                paired_rows([{'epochs': rows}, {'epochs': other['epochs'][:len(rows)]}])
        with screen.measured_phase(out, 'epoch_checkpoint', name, identity, environment, device):
            utility.save_fit(directory / f'epoch_{global_epoch+1:06d}.pt', screen.checkpoint_state(
                model, predictor, optimizers, global_epoch + 1, rows, key, environment, name))
        print(f"{name}: committed epoch {global_epoch+1}/16; validation accuracy "
              f"{100*row['validation']['accuracy']:.2f}%", flush=True)
    report = {**source_report, 'identity': key, 'study_identity': identity,
              'source_report_identity': source_report['identity'],
              'source_epoch7_sha256': source_hash, 'epochs': rows,
              'new_initial_fit_updates': 0, 'environment_at_report': environment,
              'claim_scope': SPEC['claim_scope']}
    screen.commit_json(report_path, report)
    return report


def combined_costs(out, identity, inherited):
    """Combine screen and continuation clocks once, preserving unknown-cost flags.

    Returns a JSON-safe report; reads records only. Phase totals are subsets of
    invocation time. Epochs 1-5 remain an unmeasured inherited dependency.
    """
    current = screen.cost_report(out, identity)
    phases = dict(inherited['phase_seconds_including_io'])
    for name, seconds in current['phase_seconds_including_io'].items():
        phases[name] = phases.get(name, 0.) + seconds
    return {'screen': inherited, 'continuation': current,
            'combined_recorded_invocation_seconds': inherited['completed_or_failed_invocation_seconds'] + current['completed_or_failed_invocation_seconds'],
            'combined_phase_seconds_including_io': phases,
            'complete_workflow_cost_known': inherited['complete_workflow_cost_known'] and current['complete_workflow_cost_known'],
            'clean_efficiency_claim_allowed': False,
            'scope': 'screen_preparation_and_epochs_6_to_16; source_epochs_1_to_5_not_measured',
            'accounting': 'add invocation totals once; never add phase totals to them'}


def run_study(args):
    """Validate, execute or recover all six separately authorized continuations.

    Real CIFAR access requires the explicit CLI gate. Source results stay read-only;
    writes occur only in a separate output tree. No downloads or split rebuilding.
    The same command skips completed branches and resumes incomplete ones.
    """
    if not args.authorize_continuation:
        raise PermissionError('pass --authorize-continuation for epochs 8-16')
    invoked = time.perf_counter()
    source, destination = Path(args.screen_dir).resolve(), Path(args.out).resolve()
    if source == destination or source.is_relative_to(destination) or destination.is_relative_to(source):
        raise ValueError('continuation output must be separate from the completed screen')
    root, out, split, payload, config, train, validation, device = screen.validate_inputs(args)
    environment = prior.environment_metadata(device)
    manifest, summary, hashes, evidence = load_screen(source, payload, split, environment)
    body = {'spec': SPEC, 'source_study_identity': manifest['identity'],
            'source_evidence_sha256': utility.digest(evidence),
            'epoch7_checkpoint_sha256': hashes,
            'continuation_script_sha256': prior.sha256_file(Path(__file__)),
            'comparison_environment': utility.comparison_environment(environment),
            'condition_order': manifest['condition_order']}
    identity = utility.digest(body)
    out.mkdir(parents=True, exist_ok=True)
    path = out / 'continuation.json'
    if utility.checked_record(path, identity) is None:
        if any(not p.name.startswith('continuation.uncommitted-') for p in out.iterdir()):
            raise ValueError('nonempty output lacks compatible continuation identity')
        screen.commit_json(path, {'identity': identity, 'status': 'complete', **body})
    evidence_path = out / 'inherited_screen.json'
    if utility.checked_record(evidence_path, identity) is None:
        screen.commit_json(evidence_path, {'identity': identity, 'status': 'complete', 'evidence': evidence})
    attempt = out / 'attempts' / uuid.uuid4().hex
    screen.commit_json(attempt / 'start.json', {'identity': identity, 'status': 'started', 'environment': environment})
    status, error = 'complete', None
    try:
        reports = []
        for number, source_report in enumerate(summary['conditions'], 1):
            print(f"[{number}/6] {source_report['condition']}", flush=True)
            reports.append(continue_condition(payload, train, validation, source, out, source_report,
                           hashes[source_report['condition']], identity, environment, device))
        paired_rows(reports)
        result = {'identity': identity, 'status': 'complete', 'baseline': summary['baseline'],
                  'conditions': reports, 'claim_scope': SPEC['claim_scope'],
                  'source_study_identity': manifest['identity'],
                  'note': 'matched-head exact is not conventional full BP; inherited rows 6-7, new rows 8-16'}
        completed = utility.checked_record(out / 'study_report.json', identity)
        if completed is None:
            screen.commit_json(out / 'study_report.json', result)
        elif completed != result:
            raise ValueError('immutable continuation summary differs')
    except BaseException as exc:
        status, error = 'failed', type(exc).__name__ + ': ' + str(exc)
        raise
    finally:
        screen.synchronize(device)
        screen.commit_json(attempt / 'end.json', {'identity': identity, 'status': status,
                           'error': error, 'elapsed_seconds': time.perf_counter() - invoked})
        screen.commit_json(attempt / 'cost_summary.json', combined_costs(out, identity, evidence['cost_summary']))
    print('Continuation complete: all six conditions reached epoch 16.', flush=True)


def main(argv=None):
    """Describe, explicitly run/recover, or export the continuation; return exit code zero."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--describe', action='store_true')
    parser.add_argument('--authorize-continuation', action='store_true')
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    for name in ('screen-dir', 'checkpoint-dir', 'data-dir', 'split-file', 'out', 'export-reports'):
        parser.add_argument('--' + name)
    args = parser.parse_args(argv)
    if args.describe:
        print(json.dumps({'spec': SPEC, 'additional_epochs_per_condition': 9,
                          'total_additional_classifier_epochs': 54,
                          'additional_initial_fit_updates': 0}, indent=2))
    elif args.export_reports:
        if not args.out:
            parser.error('--export-reports requires --out')
        screen.export_reports(Path(args.out), Path(args.export_reports))
    else:
        if any(getattr(args, n) is None for n in ('screen_dir', 'checkpoint_dir', 'data_dir', 'split_file', 'out')):
            parser.error('provide screen, pilot checkpoint, dataset, saved split and separate output paths')
        run_study(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
