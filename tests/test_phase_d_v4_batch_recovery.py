"""Exercise the actual notebook batch orchestration on tiny persisted runs."""

import copy
import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest import mock

import torch
from torch.utils.data import TensorDataset

from gradient_phase_d_resnet_v4 import run_training, resume_run
from phase_d_v4_io import (
    MethodId, STRICT_RESUME_POLICY, COMPATIBLE_T4_RESUME_POLICY,
    atomic_write_json, backup_study_support, load_run_config,
    latest_committed_checkpoint, load_checkpoint, sha256_file,
)
from tests.test_phase_d_v4_m4 import (
    ReplayDataset, _base_config, _replay_config, _artifacts, _identities, _calibration,
)


PROJECT = Path(__file__).resolve().parents[1]


def notebook_namespace(study, remote):
    """Load the real notebook helper cell into an isolated test namespace.

    Inputs are temporary local and persistence directories. Returns functions
    with those paths bound; no notebook training or recovery cell is executed.
    """
    notebook = json.loads((PROJECT / 'PHASE_D_V4_COLAB.ipynb').read_text(encoding='utf-8'))
    namespace = {'Path': Path, 'json': json}
    exec(''.join(notebook['cells'][2]['source']), namespace)
    namespace.update(
        STUDY=study, CONFIGS=study/'configs', DATA=study/'data',
        SPLIT=study/'split_indices.npz', PERSISTENCE_ROOT=remote,
        DONOR_RUN=study/'seed0_relative_feedback_donor',
        DONOR_MASK=study/'seed0_relative_feedback_donor/masks.npz',
        DONOR_MANIFEST=study/'seed0_relative_feedback_donor/mask_manifest.json',
    )
    return namespace


class BatchRecoveryTests(unittest.TestCase):
    """Verify batch restore/resume with real synthetic checkpoints, never CIFAR."""

    def setUp(self):
        """Create small completed dependencies plus complete/partial/unstarted runs.

        Uses CPU tensors and local temporary directories only. Persists real
        snapshots, configs, split stand-ins and profile stand-ins for recovery.
        """
        self.temporary = tempfile.TemporaryDirectory(dir=PROJECT)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.study = self.root/'study'
        self.remote = self.root/'drive'
        self.namespace = notebook_namespace(self.study, self.remote)
        self.train = ReplayDataset()
        generator = torch.Generator().manual_seed(3301)
        self.validation = TensorDataset(
            torch.randn(5, 3, 32, 32, generator=generator),
            torch.randint(0, 10, (5,), generator=generator),
        )
        self.ids = ['done', 'interrupted', 'next']
        donor_id = 'seed0_relative_feedback_donor'
        replay_id = 'seed0_relative_replay_learned_equivalence'
        self.pilot = {
            'dependency_phase': [
                {'run_id': donor_id, 'method': 'relative_feedback'},
                {'run_id': replay_id, 'method': 'relative_replay_learned'},
            ],
            'measured_conditions': [
                {'run_id': name, 'method': 'exact', 'depends_on': []}
                for name in self.ids
            ],
            'run_order': {'measured_block': self.ids},
        }
        self.namespace['pilot'] = copy.deepcopy(self.pilot)
        donor = _base_config(self.study, donor_id, MethodId.RELATIVE_FEEDBACK)
        self.write_config(donor_id, donor)
        run_training(
            donor, self.train, self.validation, self.study/donor_id, 'cpu',
            artifacts=_artifacts(), data_identities=_identities(), calibration=_calibration(),
            persistence_directory=self.remote/donor_id,
        )
        donor_manifest = json.loads((self.study/donor_id/'mask_manifest.json').read_text())
        replay = _replay_config(self.study, replay_id, MethodId.RELATIVE_REPLAY_LEARNED, donor_manifest)
        self.write_config(replay_id, replay)
        run_training(
            replay, self.train, self.validation, self.study/replay_id, 'cpu',
            artifacts=_artifacts(), data_identities=_identities(),
            mask_path=self.namespace['DONOR_MASK'], mask_manifest_path=self.namespace['DONOR_MANIFEST'],
            persistence_directory=self.remote/replay_id,
        )
        for name in self.ids:
            config = _base_config(self.study, name, MethodId.EXACT)
            self.write_config(name, config)
            if name != 'next':
                run_training(
                    config, self.train, self.validation, self.study/name, 'cpu',
                    artifacts=_artifacts(), data_identities=_identities(),
                    persistence_directory=self.remote/name,
                    stop_after_global_epoch=0 if name == 'interrupted' else None,
                )
        supports = {
            'run_order_manifest.json': json.dumps(self.pilot),
            'split_indices.npz': 'synthetic split stand-in',
            'split_indices.npz.manifest.json': json.dumps(_identities()),
            'pre_block_profile/profile_summary.json': '{}',
            'pre_block_profile/profile_samples.csv': 'component,latency_ms\nplain_exact,1\n',
        }
        for relative, content in supports.items():
            destination = self.study/relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content)
            backup_study_support(destination, self.study, self.remote)
        self.commands = []
        self.namespace['run_checked'] = self.synthetic_command

    def write_config(self, name, config):
        """Save and checksum one synthetic config; return nothing."""
        destination = self.study/'configs'/f'{name}.json'
        atomic_write_json(destination, config.to_dict())
        backup_study_support(destination, self.study, self.remote)

    def synthetic_command(self, arguments):
        """Execute only tiny run/resume operations instead of scientific subprocesses.

        Records arguments, uses generated fixtures and returns no value. All
        real checkpoint, mask and persistence code is retained; errors propagate.
        """
        args = list(map(str, arguments))
        self.commands.append(args)
        config = load_run_config(args[args.index('--config') + 1])
        destination = Path(args[args.index('--out') + 1])
        kwargs = {
            'artifacts': _artifacts(), 'data_identities': _identities(),
            'persistence_directory': Path(args[args.index('--persistence-dir') + 1]),
        }
        if args[1] == 'resume':
            kwargs['resume_compatibility_policy'] = args[args.index('--resume-compatibility') + 1]
            kwargs['checkpoint_path'] = Path(args[args.index('--checkpoint') + 1])
            resume_run(config, self.train, self.validation, destination, 'cpu', **kwargs)
        else:
            self.assertEqual(args[1], 'run')
            run_training(config, self.train, self.validation, destination, 'cpu', **kwargs)

    def lose_runtime(self):
        """Delete only this fixture's local study after checking its absolute scope.

        Simulates Colab losing all local files while preserving fake Drive.
        Returns nothing; never touches repository sources or historical results.
        """
        self.assertTrue(self.study.resolve().is_relative_to(self.root.resolve()))
        shutil.rmtree(self.study)
        self.namespace = notebook_namespace(self.study, self.remote)
        self.namespace['run_checked'] = self.synthetic_command

    def restore(self):
        """Restore the actual batch via notebook helpers without console noise."""
        with redirect_stdout(io.StringIO()):
            return self.namespace['restore_measured_batch'](device='cpu')

    def test_lost_runtime_restores_whole_batch_and_continues_in_order(self):
        """Completed models stay unchanged; only partial and new runs execute.

        Repeats restoration before continuation and again after a second reset.
        Uses real snapshots; a second completed-batch invocation must train none.
        """
        before = sha256_file(latest_committed_checkpoint(self.study/'done/checkpoints'))
        self.lose_runtime()
        statuses = self.restore()
        self.assertEqual([status for _, status in statuses], ['complete', 'partial', 'not_started'])
        self.restore()
        with redirect_stdout(io.StringIO()):
            self.namespace['run_measured_batch'](authorized=True, device='cpu')
        self.assertEqual([args[1] for args in self.commands], ['resume', 'run'])
        self.assertEqual([Path(args[args.index('--out')+1]).name for args in self.commands], ['interrupted', 'next'])
        self.assertEqual(before, sha256_file(latest_committed_checkpoint(self.study/'done/checkpoints')))
        with redirect_stdout(io.StringIO()):
            self.namespace['run_measured_batch'](authorized=True, device='cpu')
        self.assertEqual(len(self.commands), 2)
        self.lose_runtime()
        self.assertTrue(all(status == 'complete' for _, status in self.restore()))
        with redirect_stdout(io.StringIO()):
            self.namespace['run_measured_batch'](authorized=True, device='cpu')
        self.assertEqual(len(self.commands), 2)

    def test_missing_local_state_cannot_restart_completed_runs(self):
        """A fresh runtime cannot enter the loop until Drive runs are restored."""
        self.lose_runtime()
        self.namespace['pilot'] = self.pilot
        with self.assertRaisesRegex(RuntimeError, 'whole-batch restoration'):
            self.namespace['run_measured_batch'](authorized=True, device='cpu')
        self.assertEqual(self.commands, [])

    def test_corrupt_config_and_snapshot_stop_before_training(self):
        """Corrupt support/checkpoints fail restoration rather than starting anew."""
        config = self.remote/'_study_support/configs/next.json'
        original = config.read_bytes()
        config.write_bytes(b'{}')
        self.lose_runtime()
        with self.assertRaisesRegex(ValueError, 'checksum'):
            self.restore()
        config.write_bytes(original)
        checkpoint = self.remote/'interrupted/checkpoints/checkpoint_000001.pt'
        checkpoint.write_bytes(b'corrupted')
        with self.assertRaisesRegex(RuntimeError, 'verified recovery snapshot'):
            self.restore()
        self.assertEqual(self.commands, [])

    def test_failed_command_stops_batch_and_blocks_automatic_retry(self):
        """A run failure starts no later condition and survives another reset."""
        self.namespace['run_checked'] = mock.Mock(side_effect=RuntimeError('test command failed'))
        with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, 'test command failed'):
            self.namespace['run_measured_batch'](authorized=True, device='cpu')
        self.assertEqual(self.namespace['run_checked'].call_count, 1)
        self.lose_runtime()
        with self.assertRaisesRegex(RuntimeError, 'Recorded failure'):
            self.restore()
        self.assertEqual(self.commands, [])

    def test_authorization_and_zero_checkpoint_loss_fail_closed(self):
        """No authorization or a lost first epoch cannot silently start training."""
        with self.assertRaises(PermissionError):
            self.namespace['run_measured_batch'](device='cpu')
        marker = self.remote/'_batch_status/next.started.json'
        atomic_write_json(marker, {'run_id': 'next'})
        with self.assertRaisesRegex(RuntimeError, 'no recoverable checkpoint'):
            self.namespace['verify_batch_progress']()
        self.assertEqual(self.commands, [])

    def test_compatibility_policy_is_forwarded_to_resume(self):
        """Replacement-T4 opt-in reaches resume unchanged; fresh runs stay fresh.

        Only command execution is mocked, because CPU cannot claim T4 evidence.
        Existing separate cross-T4 tests cover the real compatibility checks.
        """
        progress = self.namespace['verify_batch_progress']()
        complete_after = mock.Mock(return_value='complete')
        with mock.patch.dict(self.namespace, {
            'verify_batch_progress': mock.Mock(return_value=progress),
            'batch_run_status': complete_after,
            'run_checked': mock.Mock(),
        }), redirect_stdout(io.StringIO()):
            self.namespace['run_measured_batch'](
                authorized=True, policy=COMPATIBLE_T4_RESUME_POLICY, device='cpu',
            )
            calls = self.namespace['run_checked'].call_args_list
        args = calls[0].args[0]
        self.assertEqual(args[args.index('--resume-compatibility')+1], COMPATIBLE_T4_RESUME_POLICY)
        self.assertNotIn('--resume-compatibility', calls[1].args[0])

    def test_replay_comparison_uses_real_flat_counters_and_rejects_differences(self):
        """Real donor/replay pass; a single altered flat count fails exact equality."""
        self.assertTrue(self.namespace['verify_batch_replay']()['equivalent'])
        actual_load = self.namespace['load_checkpoint']

        def altered(path, **kwargs):
            """Alter only a loaded replay count in memory; never modify a checkpoint."""
            payload = actual_load(path, **kwargs)
            if 'relative_replay_learned_equivalence' in str(path):
                payload['run_state']['predictor_fits'] += 1
            return payload

        with mock.patch.dict(self.namespace, {'load_checkpoint': altered}):
            with self.assertRaisesRegex(ValueError, 'counters'):
                self.namespace['verify_batch_replay']()

    def test_three_existing_recovery_cells_continue_the_entire_batch(self):
        """Execute the actual restore/data/continue cells after losing local files.

        Only GPU choice, Colab paths and CIFAR cache lookup are adapted to the
        tiny CPU fixture. Production restore/resume and notebook control flow
        remain real. No network, Colab or scientific data is accessed.
        """
        self.lose_runtime()
        notebook = json.loads((PROJECT/'PHASE_D_V4_COLAB.ipynb').read_text(encoding='utf-8'))
        restore = ''.join(notebook['cells'][14]['source']).replace(
            "STUDY = Path('/content/runs_phase_d_v4/phase_d_v4_pilot_seed0')",
            f'STUDY = Path({self.study.as_posix()!r})',
        )
        real_restore = self.namespace['restore_measured_batch']
        real_run = self.namespace['run_measured_batch']

        def restore_cpu(**kwargs):
            """Select CPU for this synthetic restore while preserving the policy."""
            return real_restore(device='cpu', **kwargs)

        def run_cpu(**kwargs):
            """Select CPU for synthetic continuation while preserving authorization."""
            return real_run(device='cpu', **kwargs)

        self.namespace.update(restore_measured_batch=restore_cpu, run_measured_batch=run_cpu)
        with redirect_stdout(io.StringIO()):
            exec(restore, self.namespace)
        self.assertNotIn('RESTORE_RUN_ID', self.namespace)
        self.assertEqual([status for _, status in self.namespace['recovered_progress']],
                         ['complete', 'partial', 'not_started'])
        dataset = ''.join(notebook['cells'][15]['source'])
        continuation = ''.join(notebook['cells'][16]['source'])
        with mock.patch('torchvision.datasets.CIFAR10', return_value=object()), redirect_stdout(io.StringIO()):
            exec(dataset, self.namespace)
            with self.assertRaises(PermissionError):
                exec(continuation, self.namespace)
            exec(continuation.replace('AUTHORIZE_RECOVERY_CONTINUATION = False',
                                      'AUTHORIZE_RECOVERY_CONTINUATION = True'), self.namespace)
        self.assertEqual([args[1] for args in self.commands], ['resume', 'run'])

    def test_incompatible_environment_stops_before_run_restoration(self):
        """Rejected allocation compatibility cannot copy runs or start updates."""
        self.lose_runtime()
        with mock.patch('phase_d_v4_io.verify_resume_snapshot_execution',
                        side_effect=ValueError('incompatible replacement runtime')):
            with mock.patch('phase_d_v4_io.restore_or_verify_resume_snapshot') as restore:
                with self.assertRaisesRegex(ValueError, 'incompatible replacement'):
                    self.restore()
                restore.assert_not_called()
        self.assertEqual(self.commands, [])


if __name__ == '__main__':
    unittest.main()
