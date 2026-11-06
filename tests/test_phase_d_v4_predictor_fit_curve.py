"""Synthetic checks for the separately authorized predictor fitting curves."""

import argparse
import contextlib
import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import torch
from torch.utils.data import Subset, TensorDataset

import predictor_fit_curve_v4 as study
from phase_d_v4_io import save_checkpoint
from tests.test_phase_d_v4_predictor_investigation import fixture, small_spec, source_payload, ROOT
from tests.test_phase_d_v4_m3 import _assert_tree_equal


def small_curve_spec():
    """Return a tiny synthetic-only follow-up budget without modifying production defaults."""
    spec = copy.deepcopy(study.SPEC)
    spec.update(checkpoint_sequences=[5], fit_cache_batches=[1, 2], updates=6, report_updates=[0, 2, 6])
    return spec


def synthetic_cache(device=torch.device('cpu')):
    """Return two fixed-shape CPU target batches and a fresh tiny predictor on device."""
    torch.manual_seed(921)
    cache = [(torch.randn(2,8,4,4), torch.tensor([1,2]), torch.randn(2,8,4,4)) for _ in range(2)]
    return cache, study.prior.GradientPredictor(8,4,10).to(device)


class FitCurveTests(unittest.TestCase):
    """Verify optimizer continuity, held-out isolation, integrity and bounded recovery."""

    @classmethod
    def setUpClass(cls):
        """Keep tiny CPU fixtures efficient and remember the caller's thread setting."""
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        """Restore CPU thread count after all engineering fixtures complete."""
        torch.set_num_threads(cls.threads)

    def test_reporting_preserves_uninterrupted_optimizer_trajectory(self):
        """Curve evaluation must not reset AdamW or alter the reference update sequence."""
        cfg = fixture()[0]
        cache, candidate = synthetic_cache()
        reference = copy.deepcopy(candidate)
        before_rng = study.prior.capture_rng_state()
        spec = small_curve_spec()
        with contextlib.redirect_stdout(io.StringIO()):
            result = study.fit_curve(candidate, cache, cache, cfg, torch.device('cpu'), spec)
        _assert_tree_equal(self, before_rng, study.prior.capture_rng_state())
        old_spec = small_spec()
        old_spec['fit_updates'] = spec['updates']
        telemetry = study.prior.fit_predictor(reference, cache, cfg, old_spec, torch.device('cpu'))
        _assert_tree_equal(self, reference.state_dict(), candidate.state_dict())
        actual = [row for point in result['curve'] for row in point['training_since_previous_report']]
        self.assertEqual(actual, telemetry)
        self.assertEqual([p['update'] for p in result['curve']], spec['report_updates'])

    def test_heldout_changes_do_not_change_fitted_parameters(self):
        """Held-out targets affect reporting alone, never fitted weights or training telemetry."""
        cfg = fixture()[0]
        cache, first = synthetic_cache()
        second = copy.deepcopy(first)
        changed = [(x,y,g*4) for x,y,g in cache]
        with contextlib.redirect_stdout(io.StringIO()):
            a = study.fit_curve(first, cache, cache, cfg, torch.device('cpu'), small_curve_spec())
            b = study.fit_curve(second, cache, changed, cfg, torch.device('cpu'), small_curve_spec())
        _assert_tree_equal(self, first.state_dict(), second.state_dict())
        for left, right in zip(a['curve'], b['curve'], strict=True):
            self.assertEqual(left['fit'], right['fit'])
            self.assertEqual(left['training_since_previous_report'], right['training_since_previous_report'])
        self.assertNotEqual(a['heldout_cache_sha256'], b['heldout_cache_sha256'])

    def test_partial_fit_is_preserved_completed_fit_skipped(self):
        """An interrupted fit retains its curve; rerun repeats it once then verifies/skips it."""
        cfg = fixture()[0]
        cache, _ = synthetic_cache()
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp, contextlib.redirect_stdout(io.StringIO()):
            path = Path(tmp)/'epoch_05_historical_fresh_batches1.json'
            identity = {'fixture': True}
            real_fit = study.fit_curve

            def interrupted(*args):
                """Persist an initial curve then simulate a kernel interruption before commitment."""
                callback = args[-1]
                callback({'curve':[{'update':0}], 'completed_updates':0})
                raise KeyboardInterrupt('synthetic disconnect')

            with patch.object(study, 'fit_curve', side_effect=interrupted):
                with self.assertRaises(KeyboardInterrupt):
                    study.run_fit(path,identity,cfg,cache,cache,torch.device('cpu'),'historical_fresh',small_curve_spec())
            self.assertFalse(path.exists())
            evidence = [json.loads(p.read_text()) for p in (Path(tmp)/'attempts').glob('*.json')]
            self.assertEqual(evidence[0]['status'], 'interrupted_or_failed')
            self.assertEqual(evidence[0]['curve'], [{'update':0}])
            with patch.object(study, 'fit_curve', wraps=real_fit) as fitting:
                study.run_fit(path,identity,cfg,cache,cache,torch.device('cpu'),'historical_fresh',small_curve_spec())
                self.assertEqual(fitting.call_count,1)
            with patch.object(study, 'fit_curve') as fitting:
                study.run_fit(path,identity,cfg,cache,cache,torch.device('cpu'),'historical_fresh',small_curve_spec())
                fitting.assert_not_called()
                with self.assertRaisesRegex(ValueError, 'incompatible'):
                    study.run_fit(path,{'fixture':False},cfg,cache,cache,torch.device('cpu'),'historical_fresh',small_curve_spec())
            self.assertEqual(len(list((Path(tmp)/'attempts').glob('*.json'))),2)

    def test_unauthorized_study_does_not_read_paths(self):
        """The default command cannot read data or begin fitting without authorization."""
        with self.assertRaises(PermissionError):
            study.run_study(argparse.Namespace(authorize_fit_study=False))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_curves_and_reference_updates_match(self):
        """Fixed reporting points preserve the actual reference GPU optimizer trajectory."""
        cfg = fixture()[0]
        device = torch.device('cuda')
        study.prior.configure_determinism(cfg, device)
        cache, candidate = synthetic_cache(device)
        reference = copy.deepcopy(candidate)
        with contextlib.redirect_stdout(io.StringIO()):
            report = study.fit_curve(candidate,cache,cache,cfg,device,small_curve_spec())
        ref_spec = small_spec()
        ref_spec['fit_updates'] = 6
        study.prior.fit_predictor(reference,cache,cfg,ref_spec,device)
        _assert_tree_equal(self, reference.state_dict(),candidate.state_dict())
        self.assertEqual(report['curve'][-1]['heldout']['invalid_samples'],0)

    def test_real_synthetic_study_and_idempotent_recovery(self):
        """Exercise real source checks, cache construction, both models, and saved reports.

        Only CIFAR loading and budgets are replaced with synthetic fixtures.
        Repeating the study must leave all completed report bytes unchanged.
        """
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp, contextlib.redirect_stdout(io.StringIO()):
            root = Path(tmp)
            source = root/'source'/'checkpoints'
            payload = source_payload(5)
            checkpoint = source/'checkpoint_000005.pt'
            save_checkpoint(checkpoint,payload)
            split = root/'support'/'split.npz'
            split.parent.mkdir()
            split.write_bytes(b'fixture')
            dataset = TensorDataset(torch.randn(16,3,32,32),torch.arange(16)%10)
            train, validation = Subset(dataset,list(range(12))),Subset(dataset,list(range(12,16)))
            old_spec = small_spec()
            groups = study.prior.select_groups(12,old_spec)
            old_manifest = {'spec': old_spec, 'script_sha256':study.prior.sha256_file(study.prior.__file__),
                            'split_sha256':study.prior.sha256_file(split),
                            'checkpoint_hashes':{'5':study.prior.sha256_file(checkpoint)},
                            'official_training_indices':groups}
            previous = root/'previous'
            study.prior.write_record(previous/'investigation.json',old_manifest)
            study.prior.write_record(previous/'checkpoint_000005.json',
                                     {'status':'complete','investigation_sha256':study.digest(old_manifest),
                                      'source_sha256':old_manifest['checkpoint_hashes']['5']})
            args = argparse.Namespace(authorize_fit_study=True,previous_study=previous,checkpoint_dir=source,
                                      data_dir=root/'data',split_file=split,out=root/'output',device='cpu')
            with patch.object(study,'SPEC',small_curve_spec()), patch.object(study.prior,'SPEC',old_spec), \
                    patch.object(study.prior,'load_cifar10_datasets',return_value=(train,validation,payload['data_identities'])):
                study.run_study(args)
                paths = list(args.out.glob('epoch_*.json'))
                reports = [p for p in paths if not p.name.endswith('.commit.json')]
                self.assertEqual(len(reports),4)
                hashes = {p:study.prior.sha256_file(p) for p in paths}
                with patch.object(study,'fit_curve') as fitting:
                    study.run_study(args)
                    fitting.assert_not_called()
                self.assertEqual(hashes,{p:study.prior.sha256_file(p) for p in paths})
            for p in reports:
                report = study.prior.read_record(p)
                self.assertEqual([row['update'] for row in report['curve']], [0,2,6])


if __name__ == '__main__':
    unittest.main()
