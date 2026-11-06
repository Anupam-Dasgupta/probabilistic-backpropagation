"""Synthetic verification of fresh sampling, optimizer recovery and future utility."""

import argparse
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

import predictor_utility_v4 as study
from tests.test_phase_d_v4_predictor_investigation import fixture, source_payload, ROOT
from tests.test_phase_d_v4_m3 import _snapshot_state, _assert_tree_equal
from phase_d_v4_io import save_checkpoint


def tiny_spec():
    """Return small engineering budgets; no real-data protocol values are mutated."""
    spec = copy.deepcopy(study.SPEC)
    spec.update(batch_size=4, fit_examples=[4, 8], evaluation_examples=4,
                episodes=2, future_steps=1, updates=4, save_every=2,
                checkpoint_sequences=[5], variants=['local32'])
    return spec


class UtilityTests(unittest.TestCase):
    """Exercise real numeric paths and simulate interruption without scientific data."""

    @classmethod
    def setUpClass(cls):
        """Limit synthetic CPU overhead and remember the original thread count."""
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        """Restore the caller's thread setting after the engineering suite."""
        torch.set_num_threads(cls.threads)

    def test_fresh_disjoint_nested_selection_and_private_rng(self):
        """Exclude all prior samples; fitting prefixes and every probe group remain disjoint."""
        before = study.prior.capture_rng_state()
        groups = study.select_samples(45000, study.SPEC)
        _assert_tree_equal(self, before, study.prior.capture_rng_state())
        flat = [i for batches in (groups['fit'], groups['evaluation'], *groups['episodes'])
                for batch in batches for i in batch]
        self.assertEqual(len(flat), len(set(flat)))
        self.assertFalse(set(flat).intersection(groups['excluded']))
        self.assertEqual(len(groups['excluded']), 512)
        self.assertEqual(len(groups['fit'])*32, 4096)
        self.assertEqual(len(groups['evaluation'])*32, 512)
        self.assertEqual(groups, study.select_samples(45000, study.SPEC))
        with self.assertRaises(ValueError):
            study.select_samples(600, study.SPEC)

    def test_candidates_isolate_width_and_context(self):
        """The four explicit variants expose both width comparisons at each context setting."""
        cfg = fixture()[0]
        before = study.prior.capture_rng_state()
        candidates = {n: study.make_candidate(n, cfg, 10, 'cpu') for n in study.SPEC['variants']}
        _assert_tree_equal(self, before, study.prior.capture_rng_state())
        self.assertEqual(candidates['local32'].net[0].out_channels, 32)
        self.assertEqual(candidates['local64'].net[0].out_channels, 64)
        self.assertEqual(candidates['global32'].local.out_channels, 32)
        self.assertEqual(candidates['global64'].local.out_channels, 64)
        with self.assertRaises(ValueError):
            study.make_candidate('typo', cfg, 10, 'cpu')

    def test_actual_interruption_resumes_identical_adamw_and_order(self):
        """A stop after a committed snapshot must reproduce uninterrupted weights and moments."""
        cfg, model, _, _, x, y, fx, fy, scale, _ = fixture()
        cache = study.prior.cache_targets(model, [(x,y),(fx,fy)], scale)
        spec = tiny_spec()
        a = study.make_candidate('local32', cfg, 123, 'cpu')
        b = copy.deepcopy(a)
        with tempfile.TemporaryDirectory(dir=ROOT) as temp, contextlib.redirect_stdout(io.StringIO()):
            clean, interrupted = Path(temp)/'clean', Path(temp)/'interrupted'
            opt_a, rows_a = study.fit_candidate(a, cache, cfg, spec, 'cpu', clean, 'identity')
            real_save = study.save_fit

            def stop_after_save(path, state):
                """Commit the first snapshot then mimic a runtime interruption."""
                real_save(path, state)
                raise KeyboardInterrupt('synthetic disconnect')

            with patch.object(study, 'save_fit', side_effect=stop_after_save):
                with self.assertRaises(KeyboardInterrupt):
                    study.fit_candidate(b, cache, cfg, spec, 'cpu', interrupted, 'identity')
            # A new process would construct a new candidate before restoring it.
            b = study.make_candidate('local32', cfg, 999, 'cpu')
            opt_b, rows_b = study.fit_candidate(b, cache, cfg, spec, 'cpu', interrupted, 'identity')
            _assert_tree_equal(self, a.state_dict(), b.state_dict())
            _assert_tree_equal(self, opt_a.state_dict(), opt_b.state_dict())
            self.assertEqual(rows_a, rows_b)
            with self.assertRaises(ValueError):
                study.latest_fit(interrupted, 'wrong identity')
            newest = interrupted/'fit_000004.pt'
            newest.write_bytes(newest.read_bytes()+b'corruption')
            with self.assertRaises(ValueError):
                study.latest_fit(interrupted, 'identity')

    def test_future_probe_preserves_every_source_state(self):
        """Copied multi-step branches preserve model, predictor, optimizers, buffers and RNG."""
        cfg, model, pred, opts, x, y, fx, fy, scale, _ = fixture()
        before = _snapshot_state(model, pred, opts)
        result = study.future_utility(model, pred, opts, [(x,y),(fx,fy),(x,y),(fx,fy)], scale, cfg)
        _assert_tree_equal(self, before, _snapshot_state(model,pred,opts))
        self.assertEqual(len(result['rows']), 8)
        for benefit in result['benefits']:
            losses = {r['alternative']: r['evaluation_ce'] for r in result['rows']
                      if r['future_steps'] == benefit['future_steps']}
            self.assertAlmostEqual(benefit['zero_minus_learned_ce'], losses['zero']-losses['learned'])

    def test_one_future_step_matches_independent_existing_probe(self):
        """After the real anchor, future comparisons agree with the independently tested M3 probe."""
        cfg, model, pred, opts, x, y, fx, fy, scale, _ = fixture()
        result = study.future_utility(model,pred,opts,[(x,y),(fx,fy),(x,y)],scale,cfg)
        local, lp, lo = copy.deepcopy((model,pred,opts))
        study.prior.set_training_mode(local,lp,'postwarm',cfg.scientific_dict()['identity']['normalization'])
        anchor = study.exact_step(local,lp,lo,x,y,scale,cfg,fit_predictor=True)
        expected = study.prior.probe_parameter_updates(local,lp,lo,fx,fy,x,y,scale,cfg)
        self.assertEqual(result['anchor_prefit_relative_error'], anchor.metrics['error_relative'])
        for a,b in zip(result['rows'], expected, strict=True):
            self.assertEqual(a['alternative'], b['alternative'])
            self.assertEqual(a['evaluation_ce'], b['fixed_loss_after'])

    def test_fresh_evaluation_does_not_change_anchor_or_future_states(self):
        """Evaluation labels affect evaluated CE only, never the score or source state."""
        cfg, model, pred, opts, x, y, fx, fy, scale, _ = fixture()
        a = study.future_utility(model,pred,opts,[(x,y),(fx,fy),(x,y)],scale,cfg)
        b = study.future_utility(model,pred,opts,[(x,y),(fx,fy),(x,(y+1)%10)],scale,cfg)
        self.assertEqual(a['anchor_prefit_relative_error'], b['anchor_prefit_relative_error'])
        self.assertNotEqual(a['evaluation_ce_after_anchor'], b['evaluation_ce_after_anchor'])

    def test_association_sign_and_constant_score(self):
        """Analytic episodes distinguish useful inverse association from undefined correlation."""
        episodes = [{'anchor_prefit_relative_error': x,
                     'benefits': [{'zero_minus_learned_ce': -x, 'skip_minus_learned_ce': 2.}]}
                    for x in [1.,2.,3.]]
        rows = study.summarize_episodes(episodes)
        self.assertAlmostEqual(rows[0]['error_benefit_pearson'], -1.)
        self.assertIsNone(rows[1]['error_benefit_pearson'])
        self.assertEqual(rows[1]['positive_benefit_episodes'], 3)

    def test_no_authorization_means_no_data_access(self):
        """The CLI runner rejects execution before reading datasets or writing outputs."""
        with patch.object(study.prior, 'load_cifar10_datasets') as load:
            with self.assertRaises(PermissionError):
                study.run_study(argparse.Namespace(authorize_utility_study=False))
            load.assert_not_called()

    def test_replacement_uuid_allowed_but_numerical_environment_locked(self):
        """A new physical allocation can resume diagnostics; changed precision cannot mix in."""
        old = {'device_name': 'Tesla T4', 'device_uuid': 'old', 'torch': 'version', 'matmul_tf32': False}
        new = {**old, 'device_name': 'NVIDIA T4', 'device_uuid': 'new'}
        self.assertEqual(study.comparison_environment(old), study.comparison_environment(new))
        self.assertNotEqual(study.comparison_environment(old),
                            study.comparison_environment({**new, 'matmul_tf32': True}))

    def test_end_to_end_synthetic_study_reuses_completed_work(self):
        """Real fixtures exercise source checks, fitting, reports, resume and identity rejection."""
        payload = source_payload(5)
        data = TensorDataset(torch.randn(600,3,8,8), torch.arange(600)%10)
        with tempfile.TemporaryDirectory(dir=ROOT) as temp, contextlib.redirect_stdout(io.StringIO()):
            temp = Path(temp)
            source = temp/'source'/'checkpoints'
            source.mkdir(parents=True)
            save_checkpoint(source/'checkpoint_000005.pt', payload)
            split = temp/'split'/'split.json'
            split.parent.mkdir()
            split.write_text('{}')
            args = argparse.Namespace(authorize_utility_study=True, checkpoint_dir=str(source),
                                      data_dir=str(temp/'data'), split_file=str(split),
                                      out=str(temp/'output'), device='cpu')
            datasets = (Subset(data,list(range(600))), Subset(data,[]), payload['data_identities'])
            with patch.object(study,'SPEC',tiny_spec()), patch.object(study.prior,'load_cifar10_datasets',return_value=datasets):
                study.run_study(args)
                reports = list((temp/'output').glob('epoch*/report.json'))
                self.assertEqual(len(reports), 3)
                hashes = {str(p): study.prior.sha256_file(p) for p in reports}
                with patch.object(study,'cached_batches',side_effect=AssertionError('completed study rebuilt cache')):
                    study.run_study(args)
                self.assertEqual(hashes,{str(p): study.prior.sha256_file(p) for p in reports})
                record = study.prior.read_record(reports[0])
                self.assertIn('utility_summary', record)
                self.assertEqual(len(record['episodes']), 2)
                # Mimic a disconnect after probes, before the final report commit.
                reports[0].unlink()
                reports[0].with_suffix('.json.commit.json').unlink()
                with patch.object(study,'cached_batches',side_effect=AssertionError('recovery rebuilt targets')):
                    study.run_study(args)
                self.assertTrue(reports[0].is_file())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_future_probe_and_fitting(self):
        """Run small real CUDA fitting and copied branches with preserved source state."""
        from gradient_phase_d_resnet_v4 import _diagnostic_fixture_state
        device = torch.device('cuda')
        cfg, model, pred, opts, x, y, fx, fy, scale, _ = _diagnostic_fixture_state(
            ROOT/'runs_phase_d_v4'/'utility_cuda_fixture', device)
        before = _snapshot_state(model,pred,opts)
        cache = study.prior.cache_targets(model,[(x,y)],scale)
        candidate = study.make_candidate('global32',cfg,123,device)
        with tempfile.TemporaryDirectory(dir=ROOT) as temp, contextlib.redirect_stdout(io.StringIO()):
            optimizer, _ = study.fit_candidate(candidate,cache,cfg,tiny_spec(),device,Path(temp),'cuda')
            result = study.future_utility(model,candidate,{**opts,'predictor':optimizer},
                                         [(x,y),(fx,fy),(x,y)],scale,cfg)
        _assert_tree_equal(self,before,_snapshot_state(model,pred,opts))
        self.assertEqual(len(result['rows']), 4)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_full_width_batch32_cuda_shapes_and_copied_updates(self):
        """Verify the real model and batch shape with random tensors, never CIFAR data."""
        from gradient_phase_d_resnet_v4 import reference_config
        rng = study.prior.capture_rng_state()
        try:
            device = torch.device('cuda')
            cfg = reference_config()
            study.prior.configure_determinism(cfg,device)
            model, pred = study.prior.build_models(cfg,device)
            opts = study.prior.build_optimizers(model,pred,cfg)
            x, y = torch.randn(32,3,32,32,device=device), torch.arange(32,device=device)%10
            cache = study.prior.cache_targets(model,[(x,y)],1.)
            self.assertEqual(tuple(cache[0][0].shape),(32,128,16,16))
            for name in study.SPEC['variants']:
                candidate = study.make_candidate(name,cfg,123,device)
                with torch.no_grad():
                    estimate = candidate(cache[0][0].to(device),y)
                self.assertEqual(tuple(estimate.shape),(32,128,16,16))
                self.assertTrue(bool(torch.isfinite(estimate).all()))
                del candidate, estimate
            result = study.future_utility(model,pred,opts,[(x,y),(x,y),(x,y)],1.,cfg)
            self.assertEqual(len(result['rows']),4)
        finally:
            study.prior.restore_rng_state(rng)


if __name__ == '__main__':
    unittest.main()
