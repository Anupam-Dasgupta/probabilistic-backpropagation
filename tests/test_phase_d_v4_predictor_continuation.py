"""Synthetic checks for the separately authorized epoch-7 to epoch-16 extension."""

import argparse
import ast
import contextlib
import copy
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

import predictor_continuation_v4 as continuation
from tests import test_phase_d_v4_predictor_online_screen as fixtures
from tests.test_phase_d_v4_m3 import _assert_tree_equal
from tests.test_phase_d_v4_predictor_investigation import ROOT

screen = continuation.screen
small_spec = fixtures.small_spec


class ContinuationTests(unittest.TestCase):
    """Check branch restoration, unchanged history, original LR and restart equivalence."""

    @classmethod
    def setUpClass(cls):
        """Use one CPU thread for small synthetic models; remember the prior setting."""
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        """Restore the caller's CPU thread setting; returns nothing."""
        torch.set_num_threads(cls.threads)

    def prepare(self, root, device='cpu'):
        """Build a complete tiny source screen and return continuation inputs/data."""
        helper = fixtures.OnlineScreenTests()
        args, payload, train, validation = helper.prepare(root)
        args.device = device
        helper.run_fixture(args, payload, train, validation)
        values = vars(args).copy()
        values.update(authorize_continuation=True, screen_dir=args.out, out=str(root / 'continuation'))
        return argparse.Namespace(**values), payload, train, validation

    def run_fixture(self, args, payload, train, validation):
        """Run the real continuation with only data construction and sizes replaced."""
        with patch.object(screen, 'SPEC', small_spec()), patch.object(
                screen.prior, 'load_cifar10_datasets', return_value=(train, validation, payload['data_identities'])), contextlib.redirect_stdout(io.StringIO()):
            continuation.run_study(args)

    def hashes(self, root):
        """Read relative file hashes to prove a source tree was not modified."""
        return {p.relative_to(root).as_posix(): screen.prior.sha256_file(p)
                for p in root.rglob('*') if p.is_file()}

    def test_authorization_and_separate_output_precede_data(self):
        """No authorization or overlapping output can access data or train."""
        with patch.object(screen, 'validate_inputs', side_effect=AssertionError('inputs accessed')):
            with self.assertRaises(PermissionError):
                continuation.run_study(argparse.Namespace(authorize_continuation=False))
            with self.assertRaises(ValueError):
                continuation.run_study(argparse.Namespace(authorize_continuation=True, screen_dir=ROOT, out=ROOT))

    def test_all_six_match_direct_continuation_preserve_sources_and_skip_complete(self):
        """Restored states through epoch 16 equal direct updates, without refits or source writes."""
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            args, payload, train, validation = self.prepare(Path(temporary))
            source, out = Path(args.screen_dir), Path(args.out)
            before = self.hashes(source)
            with (patch.object(screen.utility, 'fit_candidate', side_effect=AssertionError('refit')),
                  patch.object(screen, 'prepare_cache', side_effect=AssertionError('target cache'))):
                self.run_fixture(args, payload, train, validation)
            self.assertEqual(before, self.hashes(source))
            report = screen.prior.read_record(out / 'study_report.json')
            for item in report['conditions']:
                name = item['condition']
                inherited = screen.prior.read_record(source / name / 'report.json')
                state = screen.utility.read_tensors(source / name / 'epoch_000007.pt', inherited['identity'])
                cfg, model, predictor, optimizers = screen.construct_condition(payload, name, 'cpu')
                screen.restore_online(state, model, predictor, optimizers)
                rows = copy.deepcopy(state['rows'])
                for epoch in range(7, 16):
                    row = screen.train_epoch(model, predictor, optimizers, train, cfg, 'cpu', payload['scale'], name, epoch)
                    row['validation'] = screen.evaluate(model, validation, cfg, 'cpu')
                    rows.append(row)
                direct = screen.checkpoint_state(model, predictor, optimizers, 16, rows, 'direct', {}, name)
                saved = screen.utility.read_tensors(out / name / 'epoch_000016.pt', item['identity'])
                for field in ('model', 'predictor', 'optimizers', 'rng', 'rows'):
                    _assert_tree_equal(self, direct[field], saved[field])
                self.assertEqual(item['epochs'][:2], inherited['epochs'])
                self.assertEqual(item['new_initial_fit_updates'], 0)
                self.assertEqual(len(item['epochs']), 11)
                self.assertGreater(item['epochs'][2]['lr'], item['epochs'][-1]['lr'])
            stable = self.hashes(out)
            with patch.object(screen, 'train_epoch', side_effect=AssertionError('retrained')):
                self.run_fixture(args, payload, train, validation)
            for path, digest in stable.items():
                self.assertEqual(screen.prior.sha256_file(out / path), digest)
            bundle = Path(temporary) / 'reports.zip'
            screen.export_reports(out, bundle)
            self.assertTrue(bundle.is_file())
            with self.assertRaises(FileExistsError):
                screen.export_reports(out, bundle)

    def test_interrupted_epoch_recovers_identical_state_and_full_history(self):
        """An interrupted epoch repeats from its boundary, matching clean weights and RNG."""
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            args, payload, train, validation = self.prepare(Path(temporary))
            self.run_fixture(args, payload, train, validation)
            clean = Path(args.out)
            args.out = str(Path(temporary) / 'interrupted')
            original, tripped = screen.train_epoch, []

            def interrupt(*values, **kwargs):
                """Perform real epoch-9 updates once, then simulate loss before commit."""
                result = original(*values, **kwargs)
                if values[-1] == 8 and not tripped:
                    tripped.append(True)
                    raise KeyboardInterrupt('simulated interruption')
                return result

            with patch.object(screen, 'train_epoch', side_effect=interrupt):
                with self.assertRaises(KeyboardInterrupt):
                    self.run_fixture(args, payload, train, validation)
            self.assertEqual(tripped, [True])
            self.run_fixture(args, payload, train, validation)
            for name in continuation.SPEC['conditions']:
                a = torch.load(clean / name / 'epoch_000016.pt', weights_only=True)
                b = torch.load(Path(args.out) / name / 'epoch_000016.pt', weights_only=True)
                for field in ('model', 'predictor', 'optimizers', 'rng', 'rows'):
                    _assert_tree_equal(self, a[field], b[field])

    def test_corrupt_or_missing_source_branch_fails_before_training(self):
        """A late branch's corrupt epoch-7 state cannot permit earlier branches to train."""
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            args, payload, train, validation = self.prepare(Path(temporary))
            path = Path(args.screen_dir) / 'exact_trunk_matched_head/epoch_000007.pt'
            with path.open('ab') as stream:
                stream.write(b'corruption')
            with patch.object(screen, 'train_epoch', side_effect=AssertionError('trained')):
                with self.assertRaises(ValueError):
                    self.run_fixture(args, payload, train, validation)
            self.assertFalse(Path(args.out).exists())

    def test_missing_middle_checkpoint_and_changed_environment_are_rejected(self):
        """Recovery cannot silently roll back a gap or use incompatible numerical settings."""
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            args, payload, train, validation = self.prepare(Path(temporary))
            self.run_fixture(args, payload, train, validation)
            directory = Path(args.out) / 'saved'
            report = screen.prior.read_record(directory / 'report.json')
            environment = report['environment_at_report']
            inherited = report['epochs'][:2]
            with patch.object(screen, 'SPEC', small_spec()):
                with self.assertRaises(ValueError):
                    continuation.latest_state(directory, report['identity'], 'saved', {**environment, 'torch':'different'}, inherited)
                path = directory / 'epoch_000008.pt'
                path.rename(path.with_suffix('.held'))
                path.with_suffix('.pt.commit.json').rename(path.with_suffix('.held.commit.json'))
                with self.assertRaises(ValueError):
                    continuation.latest_state(directory, report['identity'], 'saved', environment, inherited)

    def test_inherited_costs_are_added_once_and_unknown_costs_remain_unknown(self):
        """Preparation cost is retained once; incomplete attempts cannot become free work."""
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            out = Path(temporary)
            screen.commit_json(out / 'attempts/lost/start.json', {'identity':'key','status':'started'})
            inherited = {'phase_seconds_including_io':{'shared:target_cache_build':15.},
                         'completed_or_failed_invocation_seconds':750.,'complete_workflow_cost_known':True}
            result = continuation.combined_costs(out, 'key', inherited)
            self.assertEqual(result['combined_recorded_invocation_seconds'],750.)
            self.assertEqual(result['combined_phase_seconds_including_io']['shared:target_cache_build'],15.)
            self.assertFalse(result['complete_workflow_cost_known'])
            self.assertFalse(result['clean_efficiency_claim_allowed'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_continuation_restores_all_branch_optimizers(self):
        """Actual CUDA branches complete epoch 16 with saved optimizer states and no refits."""
        with tempfile.TemporaryDirectory(dir=ROOT) as temporary:
            args, payload, train, validation = self.prepare(Path(temporary), 'cuda')
            with patch.object(screen.utility, 'fit_candidate', side_effect=AssertionError('refit')):
                self.run_fixture(args, payload, train, validation)
            report = screen.prior.read_record(Path(args.out) / 'study_report.json')
            self.assertEqual(len(report['conditions']),6)
            self.assertTrue(all(c['epochs'][-1]['epoch']==16 for c in report['conditions']))

    def test_description_and_docstrings(self):
        """Description needs no files and every new function has explanatory documentation."""
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(continuation.main(['--describe']),0)
        self.assertIn('54',output.getvalue())
        for path in (Path(continuation.__file__), Path(__file__)):
            nodes = ast.walk(ast.parse(path.read_text()))
            missing = [n.name for n in nodes if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and not ast.get_docstring(n)]
            self.assertEqual(missing,[])


if __name__ == '__main__':
    unittest.main()
