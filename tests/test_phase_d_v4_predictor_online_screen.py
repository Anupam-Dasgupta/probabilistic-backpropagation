"""Synthetic tests for the separately approved resumable online screen."""

import argparse
import ast
import contextlib
import copy
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch.utils.data import Subset, TensorDataset

import predictor_online_screen_v4 as screen
from phase_d_v4_io import RunConfig, fingerprint, save_checkpoint
from tests.test_phase_d_v4_predictor_investigation import source_payload, ROOT
from tests.test_phase_d_v4_m3 import _assert_tree_equal, _snapshot_state


def fixture():
    """Return an epoch-5 tiny source and fixed train/validation tensors, without files."""
    payload = source_payload(5)
    payload['resolved_config']['scientific']['training'].update(
        warmup_epochs=4, postwarm_epochs=12, total_epochs=16, batch_size=4, drop_last=False)
    payload['scientific_fingerprint'] = fingerprint(
        RunConfig(payload['resolved_config']), {**payload['artifacts'], **payload['data_identities']})
    generator = torch.Generator().manual_seed(991)
    data = TensorDataset(torch.randn(20, 3, 8, 8, generator=generator), torch.arange(20) % 10)
    return payload, Subset(data, list(range(12))), Subset(data, list(range(12, 20)))


def small_spec():
    """Return only reduced engineering sizes; the real scientific SPEC stays unchanged."""
    values = copy.deepcopy(screen.SPEC)
    values.update(training_examples=12, validation_examples=8, batch_size=4,
                  fit_examples=8, fit_updates=4, fit_save_every=2)
    return values


class OnlineScreenTests(unittest.TestCase):
    """Exercise scheduling, paired updates, isolation, recovery and report integrity."""

    @classmethod
    def setUpClass(cls):
        """Bound CPU test overhead and remember the caller's thread count."""
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        """Restore the process thread setting; returns nothing."""
        torch.set_num_threads(cls.threads)

    def prepare(self, directory):
        """Create isolated synthetic checkpoint/split inputs and return CLI args plus data."""
        payload, train, validation = fixture()
        source = directory / 'source' / 'checkpoints'
        source.mkdir(parents=True)
        save_checkpoint(source / 'checkpoint_000005.pt', payload)
        split = directory / 'split' / 'split.npz'
        split.parent.mkdir()
        split.write_bytes(b'synthetic split identity')
        args = argparse.Namespace(authorize_online_screen=True, checkpoint_dir=str(source),
                                  data_dir=str(directory / 'data'), split_file=str(split),
                                  out=str(directory / 'output'), device='cpu')
        return args, payload, train, validation

    def run_fixture(self, args, payload, train, validation):
        """Execute real runner logic on small tensors by replacing only data access/sizes."""
        with patch.object(screen, 'SPEC', small_spec()), patch.object(
                screen.prior, 'load_cifar10_datasets', return_value=(train, validation, payload['data_identities'])), patch.object(
                screen.utility, 'select_samples', return_value={'fit': [list(range(4)), list(range(4, 8))]}), contextlib.redirect_stdout(io.StringIO()):
            screen.run_study(args)

    def test_description_is_exact_bounded_protocol_without_data(self):
        """The public description fixes six two-epoch branches without loading CIFAR."""
        with patch.object(screen.prior, 'load_cifar10_datasets', side_effect=AssertionError('data accessed')):
            result = screen.describe()
        self.assertEqual(result['total_classifier_epochs'], 12)
        self.assertEqual(result['total_classifier_batches'], 16884)
        self.assertEqual(result['initial_predictor_updates'], 512)
        self.assertEqual(result['one_time_target_batches'], 128)
        self.assertEqual(result['spec']['lr_horizon'], 16)
        self.assertEqual(result['per_epoch_operations']['saved']['head_optimizer_steps'], 469)

    def test_authorization_precedes_all_data_access(self):
        """Without the explicit gate even paths and checkpoints are not read."""
        with patch.object(screen, 'validate_inputs', side_effect=AssertionError('read inputs')):
            with self.assertRaises(PermissionError):
                screen.run_study(argparse.Namespace(authorize_online_screen=False))

    def test_actual_six_condition_counts_and_source_isolation(self):
        """Real updates preserve source state and match head slots, input hashes and LR."""
        payload, train, _ = fixture()
        before = copy.deepcopy(payload)
        rows = []
        for condition in screen.SPEC['conditions']:
            cfg, model, pred, opts = screen.construct_condition(payload, condition, torch.device('cpu'))
            row = screen.train_epoch(model, pred, opts, train, cfg, 'cpu', payload['scale'], condition, 5)
            self.assertEqual(row['counters'], screen.expected_counts(condition, row['head_mask']))
            self.assertEqual(row['head_mask'], [True, False, False])
            self.assertEqual(row['counters']['head_optimizer_steps'], 1)
            self.assertEqual(row['counters']['predictor_fits'], int(condition in ('saved','local32_refit','global32_refit')))
            rows.append(row)
        self.assertEqual(len({r['data_order_sha256'] for r in rows}), 1)
        self.assertEqual(len({r['lr'] for r in rows}), 1)
        self.assertGreater(rows[0]['lr'], .02)  # Original horizon, not a two-epoch cosine.
        _assert_tree_equal(self, payload, before)

    def test_last_batch_and_manual_exact_control(self):
        """An incomplete batch uses actual labels; exact control matches explicit reference calls."""
        payload, train, _ = fixture()
        train = Subset(train, list(range(10)))
        cfg, model, pred, opts = screen.construct_condition(payload, 'exact_trunk_matched_head', 'cpu')
        manual, manual_opts = copy.deepcopy((model, opts))
        row = screen.train_epoch(model, pred, opts, train, cfg, 'cpu', payload['scale'], 'exact_trunk_matched_head', 5)
        screen.set_epoch_lr(manual_opts, cfg, 5)
        screen.prior.set_training_mode(manual, None, 'postwarm', screen.SPEC['normalization'])
        for i, (x, y, _) in enumerate(screen.epoch_loader(train, cfg, 5, shuffle=True)):
            screen.exact_step(manual, None, manual_opts, x, y, payload['scale'], cfg,
                              fit_predictor=False, head_update=(i == 0))
        _assert_tree_equal(self, model.state_dict(), manual.state_dict())
        _assert_tree_equal(self, opts['trunk'].state_dict(), manual_opts['trunk'].state_dict())
        self.assertEqual(row['samples'], 10)
        self.assertEqual(row['counters']['exact_trunk_backwards'], 3)

    def test_end_to_end_and_completed_rerun_do_not_retrain(self):
        """All six complete with paired streams, 256-style fit state, exports and immutable rerun."""
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            args, payload, train, val = self.prepare(Path(temporary))
            self.run_fixture(args, payload, train, val)
            out = Path(args.out)
            reports = [screen.prior.read_record(p) for p in out.glob('*/report.json')]
            self.assertEqual(len(reports), 6)
            self.assertTrue(all([r['epoch'] for r in p['epochs']] == [6, 7] for p in reports))
            self.assertEqual(sum(p['initial_fit_updates'] for p in reports), 8)
            before = {p: screen.prior.sha256_file(p) for p in out.glob('*/epoch_*.pt')}
            with patch.object(screen, 'train_epoch', side_effect=AssertionError('retrained')), patch.object(
                    screen, 'prepare_cache', side_effect=AssertionError('rebuilt cache')):
                self.run_fixture(args, payload, train, val)
            self.assertEqual(before, {p: screen.prior.sha256_file(p) for p in before})
            screen.export_reports(out, Path(temporary) / 'reports.zip')
            self.assertTrue((Path(temporary) / 'reports.zip').is_file())
            with self.assertRaises(FileExistsError):
                screen.export_reports(out, Path(temporary) / 'reports.zip')

    def test_interrupted_online_epoch_matches_uninterrupted(self):
        """A partially executed epoch repeats from its last boundary with identical final states."""
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            temp = Path(temporary)
            args, payload, train, val = self.prepare(temp)
            self.run_fixture(args, payload, train, val)
            clean = Path(args.out)
            args.out = str(temp / 'interrupted')
            real_epoch, real_synthetic = screen.train_epoch, screen.synthetic_step
            tripped = []

            def interrupted_epoch(*values, **kwargs):
                """Interrupt saved-predictor epoch 7 after one actual synthetic optimizer update."""
                if values[-2:] == ('saved', 6) and not tripped:
                    def interrupt_step(*a, **k):
                        """Execute the update before simulating a lost process."""
                        real_synthetic(*a, **k)
                        tripped.append(True)
                        raise KeyboardInterrupt('simulated interruption')
                    with patch.object(screen, 'synthetic_step', side_effect=interrupt_step):
                        return real_epoch(*values, **kwargs)
                return real_epoch(*values, **kwargs)

            with patch.object(screen, 'train_epoch', side_effect=interrupted_epoch):
                with self.assertRaises(KeyboardInterrupt):
                    self.run_fixture(args, payload, train, val)
            self.assertEqual(len(tripped), 1)
            with patch.object(screen.utility, 'cached_batches', side_effect=AssertionError('recomputed targets')):
                self.run_fixture(args, payload, train, val)
            for name in screen.SPEC['conditions']:
                a = torch.load(clean / name / 'epoch_000007.pt', weights_only=True)
                b = torch.load(Path(args.out) / name / 'epoch_000007.pt', weights_only=True)
                for field in ('model', 'predictor', 'optimizers', 'rng', 'rows'):
                    _assert_tree_equal(self, a[field], b[field])

    def test_corrupt_snapshot_and_environment_change_fail(self):
        """Committed corruption and numerical-environment changes cannot trigger fresh training."""
        payload, _, _ = fixture()
        cfg, model, pred, opts = screen.construct_condition(payload, 'saved', 'cpu')
        env = screen.prior.environment_metadata('cpu')
        state = screen.checkpoint_state(model, pred, opts, 5, [], 'key', env, 'saved')
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            path = Path(temporary) / 'epoch_000005.pt'
            screen.utility.save_fit(path, state)
            bad = {**env, 'torch': 'different'}
            with self.assertRaises(ValueError):
                screen.latest_online(path.parent, 'key', bad)
            with path.open('ab') as stream:
                stream.write(b'corruption')
            with self.assertRaises(ValueError):
                screen.latest_online(path.parent, 'key', env)

    def test_interrupted_initial_fit_reuses_cache_and_optimizer(self):
        """After a committed fit block, resume AdamW rather than restarting preparation."""
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            temp = Path(temporary)
            args, payload, train, val = self.prepare(temp)
            self.run_fixture(args, payload, train, val)
            clean = Path(args.out)
            args.out = str(temp / 'interrupted_fit')
            real_save, tripped = screen.utility.save_fit, []

            def interrupt_after_fit(path, state):
                """Commit the first local32 fitting block, then simulate process interruption."""
                real_save(path, state)
                if path.name == 'fit_000002.pt' and 'local32_refit' in path.parts and not tripped:
                    tripped.append(True)
                    raise KeyboardInterrupt('simulated fit interruption')

            with patch.object(screen.utility, 'save_fit', side_effect=interrupt_after_fit):
                with self.assertRaises(KeyboardInterrupt):
                    self.run_fixture(args, payload, train, val)
            self.assertEqual(tripped, [True])
            with patch.object(screen.utility, 'cached_batches', side_effect=AssertionError('recomputed targets')):
                self.run_fixture(args, payload, train, val)
            for name in ('local32_refit', 'global32_refit'):
                a = torch.load(clean / name / 'epoch_000007.pt', weights_only=True)
                b = torch.load(Path(args.out) / name / 'epoch_000007.pt', weights_only=True)
                for field in ('model', 'predictor', 'optimizers', 'rows'):
                    _assert_tree_equal(self, a[field], b[field])

    def test_uncommitted_snapshot_is_preserved_and_gap_is_rejected(self):
        """Interrupted writes are retained, while an absent committed middle epoch fails."""
        payload, _, _ = fixture()
        _, model, pred, opts = screen.construct_condition(payload, 'saved', 'cpu')
        env = screen.prior.environment_metadata('cpu')
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            directory = Path(temporary)
            (directory / 'epoch_000005.pt').write_bytes(b'partial')
            self.assertIsNone(screen.latest_online(directory, 'key', env))
            self.assertTrue(list(directory.glob('*.uncommitted-*.pt')))
            state = screen.checkpoint_state(model, pred, opts, 6, [{}], 'key', env, 'saved')
            screen.utility.save_fit(directory / 'epoch_000006.pt', state)
            with self.assertRaises(ValueError):
                screen.latest_online(directory, 'key', env)

    def test_cost_unknown_interruption_is_not_zero(self):
        """Missing completion records flag unknown costs and never claim a complete runtime."""
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            out = Path(temporary)
            with screen.measured_phase(out, 'fit', 'local32_refit', 'key', {}, 'cpu'):
                pass
            screen.commit_json(out / 'costs/lost/start.json', {
                'identity': 'key', 'phase': 'online_epoch', 'condition': 'saved'})
            report = screen.cost_report(out, 'key')
            self.assertGreater(report['phase_seconds_including_io']['local32_refit:fit'], 0)
            self.assertFalse(report['complete_workflow_cost_known'])
            self.assertEqual(len(report['unknown_interrupted_work']), 1)

    def test_new_source_or_changed_spec_cannot_reuse_results(self):
        """A scientific-identity mismatch fails before fitting or online updates."""
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            args, payload, train, val = self.prepare(Path(temporary))
            self.run_fixture(args, payload, train, val)
            changed = small_spec()
            changed['fit_seed'] += 1
            with patch.object(screen, 'SPEC', changed), patch.object(
                    screen.prior, 'load_cifar10_datasets', return_value=(train, val, payload['data_identities'])), patch.object(
                    screen.utility, 'select_samples', return_value={'fit': [list(range(4)), list(range(4, 8))]}), patch.object(
                    screen, 'train_epoch', side_effect=AssertionError('trained after mismatch')):
                with self.assertRaises(ValueError):
                    screen.run_study(args)

    def test_copied_continuation_probe_preserves_full_state(self):
        """A global predictor supports the recovery audit without changing live state or RNG."""
        payload, _, _ = fixture()
        cfg, model, pred, opts = screen.construct_condition(payload, 'global32_refit', 'cpu')
        before = _snapshot_state(model, pred, opts)
        result = screen.continuation_probe(model, pred, opts, cfg, payload['scale'], 'cpu')
        _assert_tree_equal(self, before, _snapshot_state(model, pred, opts))
        self.assertEqual(result['status'], 'passed')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_epoch_and_snapshot_restore(self):
        """Real CUDA updates and reloads preserve the paired mask and optimizer state."""
        payload, train, _ = fixture()
        cfg, model, pred, opts = screen.construct_condition(payload, 'global32_refit', 'cuda')
        screen.prior.configure_determinism(cfg, 'cuda')
        row = screen.train_epoch(model, pred, opts, train, cfg, 'cuda', payload['scale'], 'global32_refit', 5)
        state = screen.checkpoint_state(model, pred, opts, 6, [row], 'key', {}, 'global32_refit')
        _, other, other_pred, other_opts = screen.construct_condition(payload, 'global32_refit', 'cuda')
        screen.restore_online(state, other, other_pred, other_opts)
        _assert_tree_equal(self, model.state_dict(), other.state_dict())
        _assert_tree_equal(self, opts['predictor'].state_dict(), other_opts['predictor'].state_dict())

    def test_every_authored_function_has_docstring(self):
        """Keep plain-language function documentation in the new runner and its tests."""
        for path in (Path(screen.__file__), Path(__file__)):
            tree = ast.parse(path.read_text(encoding='utf-8'))
            missing = [n.name for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and not ast.get_docstring(n)]
            self.assertEqual(missing, [], path.name)


if __name__ == '__main__':
    unittest.main()
