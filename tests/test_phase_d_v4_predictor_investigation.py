"""Bounded synthetic tests for the separate frozen-checkpoint investigation."""

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

import predictor_investigation_v4 as study
from gradient_phase_d_resnet_v4 import _diagnostic_fixture_state
from phase_d_v4_io import RunConfig, SCHEMA_VERSION, fingerprint, save_checkpoint, sha256_file
from phase_d_v4_analysis import fixed_probe_indices
from tests.test_phase_d_v4_m3 import _snapshot_state, _assert_tree_equal


ROOT = Path(__file__).resolve().parents[1]


def fixture():
    """Return a tiny synthetic classifier with populated SGD/AdamW state, without files."""
    return _diagnostic_fixture_state(ROOT/'runs_phase_d_v4'/'investigation_test', torch.device('cpu'))


def small_spec():
    """Return a reduced engineering-only budget; production SPEC is never modified."""
    spec = copy.deepcopy(study.SPEC)
    spec.update(batch_size=2, diagnostic_batches=2, fit_batches=2, heldout_batches=2,
                fit_updates=4, wide_channels=8, checkpoint_sequences=[4, 5])
    return spec


def source_payload(sequence=4):
    """Build a complete tiny source payload with actual code identities for integrity tests."""
    config, model, predictor, optimizers, _, _, _, _, scale, _ = fixture()
    values = config.to_dict()
    values['scientific']['identity'].update(method='relative_feedback', classifier_seed=0)
    values['output']['run_id'] = study.SPEC['source_run_id']
    config = RunConfig(values)
    artifacts = study._resolved_artifacts(study._scientific_artifacts())
    identities = {'training_indices_sha256': 'a'*64, 'validation_indices_sha256': 'b'*64}
    return {'schema_version': SCHEMA_VERSION, 'run_id': study.SPEC['source_run_id'],
            'checkpoint_sequence': sequence, 'completed_epoch': sequence-1, 'next_epoch': sequence,
            'resolved_config': config.to_dict(), 'artifacts': artifacts, 'data_identities': identities,
            'scientific_fingerprint': fingerprint(config, {**artifacts, **identities}),
            'method': 'relative_feedback', 'normalization': 'frozen_head_postwarm',
            'model': copy.deepcopy(model.state_dict()), 'predictor': copy.deepcopy(predictor.state_dict()),
            'optimizers': {k: copy.deepcopy(v.state_dict()) for k,v in optimizers.items()},
            'scale': scale, 'environment': {'device_type': 'cpu', 'fixture': True}}


class InvestigationTests(unittest.TestCase):
    """Check isolation, gradient units, fitting separation, and persistence failures."""

    @classmethod
    def setUpClass(cls):
        """Bound CPU engineering-test thread overhead; restore it after the class."""
        cls.threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        """Restore the caller's CPU thread setting without changing experiment files."""
        torch.set_num_threads(cls.threads)

    def test_selection_disjoint_and_preserves_plan_probes(self):
        """Plan probe IDs remain unchanged and fitting/held-out sets never overlap."""
        groups = study.select_groups(45000, study.SPEC)
        self.assertEqual(groups['diagnostic'], fixed_probe_indices(45000))
        sets = [set(sum(v, [])) for v in groups.values()]
        self.assertEqual([len(s) for s in sets], [128, 256, 128])
        self.assertTrue(all(not a.intersection(b) for i,a in enumerate(sets) for b in sets[i+1:]))

    def test_targets_use_exact_scale_and_preserve_classifier(self):
        """Cached targets equal independent autograd and cannot change source/RNG state."""
        cfg, model, pred, opts, x, y, fx, fy, scale, _ = fixture()
        before = _snapshot_state(model, pred, opts)
        cache = study.cache_targets(model, [(x,y),(fx,fy)], scale)
        _assert_tree_equal(self, before, _snapshot_state(model, pred, opts))
        local = copy.deepcopy(model)
        local.trunk.train()
        local.head.eval()
        h = local.trunk(x).detach().requires_grad_(True)
        g = torch.autograd.grad(torch.nn.functional.cross_entropy(local.head(h), y), h)[0]
        torch.testing.assert_close(cache[0][2], g*len(y)/scale, rtol=0, atol=0)
        self.assertFalse(any(t.requires_grad for batch in cache for t in batch))

    def test_global_context_can_affect_distant_positions(self):
        """A distant pixel affects the global predictor but not the local 3x3 predictor."""
        local = study.GradientPredictor(2, 2, 2)
        global_model = study.GlobalPredictor(2, 2, 2)
        with torch.no_grad():
            for p in global_model.parameters():
                p.fill_(0.1)
        x = torch.ones(1,2,8,8, requires_grad=True)
        changed = x.detach().clone()
        changed[:,:,7,7] += 1
        labels = torch.zeros(1,dtype=torch.long)
        self.assertTrue(torch.equal(local(x,labels)[:,:,0,0], local(changed,labels)[:,:,0,0]))
        self.assertFalse(torch.equal(global_model(x,labels)[:,:,0,0], global_model(changed,labels)[:,:,0,0]))
        global_model(x,labels).sum().backward()
        self.assertIsNone(x.grad)
        self.assertIsNotNone(global_model.context.weight.grad)

    def test_zero_metrics_and_real_predictor_learning(self):
        """Zero has residual one on nonzero targets; fitting reduces a representable error."""
        cfg, _, _, _, _, _, _, _, _, _ = fixture()
        pred = study.GradientPredictor(8,4,10)
        x, y = torch.randn(2,8,4,4), torch.tensor([0,1])
        zeros = [(x,y,torch.zeros_like(x))]
        ones = [(x,y,torch.ones_like(x))]
        device = torch.device('cpu')
        zero = study.prediction_report(None, ones, device)
        self.assertEqual(zero['pooled_relative_residual'], 1.0)
        self.assertEqual(zero['sample_summaries']['cosine']['undefined_count'], 2)
        before = float(pred(x,y).square().mean().detach())
        spec = small_spec(); spec['fit_updates'] = 64
        with contextlib.redirect_stdout(io.StringIO()):
            telemetry = study.fit_predictor(pred, zeros, cfg, spec, device)
        self.assertLess(float(pred(x,y).square().mean().detach()), before)
        self.assertEqual(len(telemetry), 64)

    def test_capacity_isolation_and_no_heldout_training(self):
        """Changing held-out targets cannot change fitting telemetry or training predictions."""
        cfg, model, pred, opts, x, y, fx, fy, scale, _ = fixture()
        before = _snapshot_state(model, pred, opts)
        with contextlib.redirect_stdout(io.StringIO()):
            a = study.capacity_comparison(model,pred,[(x,y)],[(fx,fy)],scale,cfg,small_spec(),torch.device('cpu'))
            b = study.capacity_comparison(model,pred,[(x,y)],[(fx,fy.roll(1))],scale,cfg,small_spec(),torch.device('cpu'))
        _assert_tree_equal(self, before, _snapshot_state(model, pred, opts))
        for name in ('historical_fresh','global64_fresh'):
            self.assertEqual(a['results'][name]['training_telemetry'], b['results'][name]['training_telemetry'])
            self.assertEqual(a['results'][name]['fit'], b['results'][name]['fit'])
        self.assertNotEqual(a['cache_hashes']['heldout'], b['cache_hashes']['heldout'])
        self.assertEqual(a['results']['historical_fresh']['training_telemetry'][0]['batch_id'],
                         a['results']['global64_fresh']['training_telemetry'][0]['batch_id'])

    def test_diagnostics_preserve_full_source_state(self):
        """All probes preserve classifier/predictor/optimizer/modes/gradients/RNG exactly."""
        cfg, model, pred, opts, x, y, fx, fy, scale, _ = fixture()
        batches = [(x,y),(fx,fy)]
        before = _snapshot_state(model,pred,opts)
        with contextlib.redirect_stdout(io.StringIO()):
            report = study.diagnostic_report(model,pred,opts,batches,batches,[[0,1,2,3],[4,5,6,7]],scale,cfg)
        _assert_tree_equal(self,before,_snapshot_state(model,pred,opts))
        self.assertEqual(len(report['gradient']),2)
        self.assertEqual({r['alternative'] for r in report['updates'][0]}, {'exact','learned','zero','skip'})

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_cached_fitting_and_isolation(self):
        """Exercise CPU target caching and GPU fits without changing source GPU state."""
        device = torch.device('cuda')
        cfg, model, pred, opts, x, y, fx, fy, scale, _ = _diagnostic_fixture_state(
            ROOT/'runs_phase_d_v4'/'investigation_cuda_test', device)
        before = _snapshot_state(model,pred,opts)
        with contextlib.redirect_stdout(io.StringIO()):
            report = study.capacity_comparison(model,pred,[(x,y)],[(fx,fy)],scale,cfg,small_spec(),device)
        _assert_tree_equal(self,before,_snapshot_state(model,pred,opts))
        for name in ('historical_fresh','global64_fresh'):
            self.assertIsNotNone(report['results'][name]['heldout']['pooled_relative_residual'])

    def test_checkpoint_integrity_and_optimizer_restoration(self):
        """Real commit checking rejects edited data and restores all optimizer moments."""
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp)/'checkpoint_000004.pt'
            payload = source_payload()
            save_checkpoint(path,payload)
            loaded = study.load_verified_source(path,4)
            cfg, model, pred, opts = study.restore_models(loaded,torch.device('cpu'))
            _assert_tree_equal(self, payload['model'],model.state_dict())
            _assert_tree_equal(self, payload['predictor'],pred.state_dict())
            _assert_tree_equal(self, payload['optimizers'],{k:v.state_dict() for k,v in opts.items()})
            with path.open('ab') as f:
                f.write(b'corrupt')
            with self.assertRaisesRegex(ValueError,'checksum'):
                study.load_verified_source(path,4)

    def test_committed_output_never_overwritten_and_corruption_rejected(self):
        """A valid record is immutable and a changed record cannot be reused."""
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp)/'report.json'
            study.write_record(path,{'ok':True})
            with self.assertRaises(FileExistsError):
                study.write_record(path,{'ok':False})
            self.assertEqual(study.read_record(path),{'ok':True})
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'corrupt'):
                study.read_record(path)

    def test_authorization_precedes_all_io(self):
        """The default CLI cannot start scientific work or inspect source paths."""
        args = argparse.Namespace(authorize_predictor_study=False)
        with self.assertRaises(PermissionError):
            study.run_study(args)

    def test_partial_record_preserved_and_orphan_marker_rejected(self):
        """An interrupted write is preserved; a missing committed report is never replaced."""
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            path = Path(tmp)/'investigation.json'
            path.write_bytes(b'{partial')
            study.preserve_uncommitted_record(path)
            self.assertFalse(path.exists())
            copies = list(Path(tmp).glob('investigation.uncommitted-*.json'))
            self.assertEqual(len(copies), 1)
            self.assertEqual(copies[0].read_bytes(), b'{partial')
            study.write_record(path, {'identity': True})
            study.preserve_uncommitted_record(path)
            self.assertEqual(study.read_record(path), {'identity': True})
            path.unlink()
            with self.assertRaisesRegex(ValueError, 'commit marker has no report'):
                study.preserve_uncommitted_record(path)

    def test_full_orchestration_recovery_and_source_preservation(self):
        """A failed second boundary preserves the first report and retries only unfinished work."""
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            source = root/'source'/'checkpoints'
            for seq in (4,5):
                save_checkpoint(source/f'checkpoint_{seq:06d}.pt',source_payload(seq))
            source_hashes = {p:sha256_file(p) for p in source.iterdir()}
            split = root/'support'/'split.npz'; split.parent.mkdir(); split.write_bytes(b'toy')
            dataset = TensorDataset(torch.randn(16,3,32,32),torch.arange(16)%10)
            train, validation = Subset(dataset,list(range(12))),Subset(dataset,list(range(12,16)))
            identities = source_payload()['data_identities']
            args = argparse.Namespace(authorize_predictor_study=True,checkpoint_dir=source,
                                      split_file=split,data_dir=root/'data',out=root/'output',device='cpu')
            spec = small_spec()
            with patch.object(study,'SPEC',spec), patch.object(study,'load_cifar10_datasets',return_value=(train,validation,identities)), contextlib.redirect_stdout(io.StringIO()):
                with patch.object(study,'diagnostic_report',side_effect=[{'first':True},RuntimeError('interrupted')]), patch.object(study,'capacity_comparison',return_value={'fixture':True}):
                    with self.assertRaisesRegex(RuntimeError,'interrupted'):
                        study.run_study(args)
                first = args.out/'checkpoint_000004.json'
                digest = sha256_file(first)
                with patch.object(study,'diagnostic_report',return_value={}) as probe, patch.object(study,'capacity_comparison',return_value={}):
                    study.run_study(args)
                    self.assertEqual(probe.call_count,1)
                with patch.object(study,'diagnostic_report') as probe, patch.object(study,'capacity_comparison') as fitting:
                    study.run_study(args)
                    probe.assert_not_called(); fitting.assert_not_called()
                self.assertEqual(sha256_file(first),digest)
                # A valid file checksum alone must not admit another study's report.
                wrong = study.read_record(first)
                wrong['investigation_sha256'] = '0'*64
                first.write_text(json.dumps(wrong))
                study.atomic_write_json(first.with_suffix('.json.commit.json'),
                                        {'sha256':sha256_file(first),'size_bytes':first.stat().st_size})
                with self.assertRaisesRegex(ValueError,'identity/status mismatch'):
                    study.run_study(args)
                manifest = args.out/'investigation.json'
                manifest.unlink()
                manifest.with_suffix('.json.commit.json').unlink()
                with self.assertRaisesRegex(ValueError,'without an investigation manifest'):
                    study.run_study(args)
            for p,digest in source_hashes.items():
                self.assertEqual(sha256_file(p),digest)
            attempts=[json.loads(p.read_text())['status'] for p in (args.out/'attempts').glob('*.json')]
            self.assertIn('interrupted_or_failed',attempts)

    def test_real_synthetic_end_to_end_report(self):
        """Run real probes/fits and committed JSON output with only CIFAR loading replaced.

        The tiny data are engineering fixtures. No scientific configuration or
        source checkpoint is changed; the temporary study is removed afterward.
        """
        with tempfile.TemporaryDirectory(dir=ROOT) as tmp:
            root = Path(tmp)
            source = root/'source'/'checkpoints'
            payload = source_payload(4)
            save_checkpoint(source/'checkpoint_000004.pt', payload)
            split = root/'support'/'split.npz'
            split.parent.mkdir()
            split.write_bytes(b'toy split')
            dataset = TensorDataset(torch.randn(16,3,32,32),torch.arange(16)%10)
            train = Subset(dataset,list(range(12)))
            validation = Subset(dataset,list(range(12,16)))
            args = argparse.Namespace(authorize_predictor_study=True,checkpoint_dir=source,
                                      split_file=split,data_dir=root/'data',out=root/'output',device='cpu')
            spec = small_spec()
            spec['checkpoint_sequences'] = [4]
            with patch.object(study,'SPEC',spec), patch.object(study,'load_cifar10_datasets',
                    return_value=(train,validation,payload['data_identities'])), contextlib.redirect_stdout(io.StringIO()):
                study.run_study(args)
            report = study.read_record(args.out/'checkpoint_000004.json')
            self.assertEqual(report['status'],'complete')
            self.assertEqual(len(report['diagnostics']['gradient']),2)
            self.assertEqual(len(report['diagnostics']['updates']),2)
            self.assertEqual(len(report['investigation_sha256']),64)
            for name in ('historical_fresh','global64_fresh'):
                candidate = report['capacity']['results'][name]
                self.assertEqual(len(candidate['training_telemetry']),4)
                self.assertIsNotNone(candidate['heldout']['pooled_relative_residual'])


if __name__ == '__main__':
    unittest.main()
