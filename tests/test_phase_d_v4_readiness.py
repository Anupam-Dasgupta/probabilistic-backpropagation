"""Final local Colab-readiness tests for persistence, recovery, and notebook gates."""

from __future__ import annotations

import json
import shutil
import tempfile
import time
import unittest
import zipfile
from pathlib import Path
from unittest import mock

from analyze_phase_d_v4 import validate_run_directory, verify_resume_equivalence
from gradient_phase_d_resnet_v4 import _synthetic_fixture, resume_run, run_training
from phase_d_v4_io import (
    backup_study_support,
    latest_committed_checkpoint,
    load_checkpoint,
    restore_resume_snapshot,
    restore_or_verify_resume_snapshot,
    restore_study_support,
    sha256_file,
    verify_resume_snapshot_execution,
)
from tests.test_phase_d_v4_m2 import _artifacts, _identities, _runner_config
import gradient_phase_d_resnet_v4 as training_module


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class NotebookReadinessTests(unittest.TestCase):
    """Check notebook ordering and executable safety without running its cells."""

    def test_notebook_compiles_and_keeps_default_disabled_gates(self) -> None:
        """Every code cell parses and both scientific authorization gates default false.

        The test reads the notebook only. It compiles code strings without
        executing them, then checks subprocess failure propagation, early cuBLAS
        setup, archive restoration, and the two default-disabled boundaries.
        """

        notebook = json.loads((PROJECT_ROOT / "PHASE_D_V4_COLAB.ipynb").read_text(encoding="utf-8"))
        code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        for index, source in enumerate(code):
            compile(source, f"notebook_cell_{index}", "exec")
        joined = "\n".join(code)
        self.assertIn('os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"', code[0])
        self.assertLess(code[0].index("CUBLAS_WORKSPACE_CONFIG"), code[0].index("import numpy as np, torch"))
        self.assertIn("check=True", joined)
        self.assertIn("AUTHORIZE_REAL_DATA_PREFLIGHT = False", joined)
        self.assertIn("AUTHORIZE_FULL_PILOT = False", joined)
        self.assertIn("AUTHORIZE_RECOVERY_DATA_SETUP = False", joined)
        self.assertIn("reference_results_archive", joined)
        self.assertNotIn("!python", joined)

    def test_config_backups_follow_successful_creation(self) -> None:
        """Donor and plain configs are preserved only after make-config succeeds.

        The notebook is read as JSON and no cell is executed. Ordering failures
        raise through assertions and no files, RNG, devices, or subprocesses are
        changed.
        """

        notebook = json.loads((PROJECT_ROOT / "PHASE_D_V4_COLAB.ipynb").read_text(encoding="utf-8"))
        source = "".join(notebook["cells"][7]["source"])
        donor_create = source.index("'make-config', '--method', 'relative_feedback'")
        donor_backup = source.index("preserve_support(DONOR_CONFIG)")
        plain_create = source.index("'make-config', '--method', 'exact'")
        plain_backup = source.index("preserve_support(PLAIN_CONFIG)")
        self.assertLess(donor_create, donor_backup)
        self.assertLess(plain_create, plain_backup)

    def test_recovery_dataset_and_continuation_cells_are_staged_and_rerunnable(self) -> None:
        """Recovery can pause at each executable authorization boundary.

        The actual notebook dataset and continuation cell sources execute with
        temporary paths plus mocked CIFAR/cache and subprocess operations. A
        default run stops, an authorized dataset reconstruction verifies split
        identities, a later rerun reuses the cache, and continuation remains
        independently disabled until edited. No download, training, CUDA, or
        external persistence operation occurs.
        """

        notebook = json.loads((PROJECT_ROOT / "PHASE_D_V4_COLAB.ipynb").read_text(encoding="utf-8"))
        code = ["".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code"]
        dataset_source = next(source for source in code if "AUTHORIZE_RECOVERY_DATA_SETUP" in source)
        continuation_source = next(source for source in code if "AUTHORIZE_RECOVERY_CONTINUATION" in source)
        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            run = root / "run"
            study = root / "study"
            split = study / "split_indices.npz"
            config = study / "config.json"
            run.mkdir(parents=True)
            split.parent.mkdir(parents=True, exist_ok=True)
            split.write_bytes(b"split")
            config.write_text("{}", encoding="utf-8")
            split_manifest = split.with_suffix(split.suffix + ".manifest.json")
            identities = {"training_indices_sha256": "a" * 64, "validation_indices_sha256": "b" * 64}
            split_manifest.write_text(json.dumps(identities), encoding="utf-8")
            commands: list[list[object]] = []

            def fake_command(arguments: list[object]) -> None:
                """Record a mocked notebook command and materialize split evidence.

                The input command is stored for assertions and nothing is
                returned. For prepare-data only, temporary split/manifest files
                are created under the test directory; no process or network runs.
                """

                commands.append(arguments)
                if "prepare-data" in arguments:
                    destination = Path(arguments[arguments.index("--split-file") + 1])
                    destination.write_bytes(b"temporary split")
                    destination.with_suffix(destination.suffix + ".manifest.json").write_text(
                        json.dumps(identities), encoding="utf-8"
                    )

            namespace = {
                "Path": Path, "json": json, "RUN_DIR": run, "DATA": root / "data",
                "SPLIT": split, "RUN_CONFIG": config, "STUDY": study,
                "run_checked": fake_command,
            }
            with mock.patch("torchvision.datasets.CIFAR10", side_effect=RuntimeError("missing")):
                with self.assertRaisesRegex(RuntimeError, "default-disabled"):
                    exec(dataset_source, namespace)
            authorized = dataset_source.replace(
                "AUTHORIZE_RECOVERY_DATA_SETUP = False",
                "AUTHORIZE_RECOVERY_DATA_SETUP = True",
            )
            with mock.patch("torchvision.datasets.CIFAR10", side_effect=RuntimeError("missing")):
                exec(authorized, namespace)
            self.assertEqual(sum("prepare-data" in command for command in commands), 1)
            with mock.patch("torchvision.datasets.CIFAR10", return_value=object()):
                exec(dataset_source, namespace)
            self.assertEqual(sum("prepare-data" in command for command in commands), 1)
            continuation_namespace = {
                **namespace,
                "RECOVERY_COMPATIBILITY": "strict_execution",
                "RESTORE_RUN_ID": "run",
                "persistence_arguments": lambda _run_id: [],
                "method": "exact",
                "source_methods": set(),
            }
            with self.assertRaisesRegex(PermissionError, "default-disabled"):
                exec(continuation_source, continuation_namespace)
            enabled = continuation_source.replace(
                "AUTHORIZE_RECOVERY_CONTINUATION = False",
                "AUTHORIZE_RECOVERY_CONTINUATION = True",
            )
            with mock.patch("torchvision.datasets.CIFAR10", return_value=object()):
                exec(enabled, continuation_namespace)
            self.assertTrue(any("resume" in command for command in commands))


class PersistenceReadinessTests(unittest.TestCase):
    """Exercise complete timing and lost-runtime recovery with tiny local fixtures."""

    def test_snapshot_timing_and_retained_warmup_restore_are_complete(self) -> None:
        """Persisted epoch work is timed and recovery restores warm-up plus latest state.

        A two-epoch synthetic run writes to a temporary persistence directory.
        The local runtime is then simulated as lost, the run is restored, and its
        timing/schema eligibility and final checkpoint tensors are compared. No
        external data, network, CUDA requirement, or scientific result is used.
        """

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            run = root / "active" / "timed-run"
            remote = root / "remote" / "timed-run"
            config = _runner_config(run, warmup_epochs=1, postwarm_epochs=1)
            train, validation, _ = _synthetic_fixture(seed=5401)
            actual_backup = training_module.backup_resume_snapshot

            def delayed_backup(*args: object, **kwargs: object) -> Path:
                """Add a known delay inside snapshot work for timing coverage.

                Inputs and returned path are forwarded unchanged to the real
                backup helper. The only extra side effect is a 0.02-second sleep;
                persistence files and failures retain their normal semantics.
                """

                time.sleep(0.02)
                return actual_backup(*args, **kwargs)

            with mock.patch.object(
                training_module, "backup_resume_snapshot", side_effect=delayed_backup
            ):
                context = run_training(
                    config, train, validation, run, "cpu",
                    artifacts=_artifacts(), data_identities=_identities(),
                    persistence_directory=remote,
                )
            local_timing = json.loads((run / "timing.json").read_text(encoding="utf-8"))
            self.assertEqual(len(local_timing["checkpoint_io"]), 2)
            self.assertTrue(all(row["io_seconds"] >= 0.02 for row in local_timing["checkpoint_io"]))
            for sequence in (1, 2):
                snapshot = remote / "snapshots" / f"checkpoint_{sequence:06d}"
                self.assertTrue((snapshot / "snapshot.ready.json").is_file())
                self.assertTrue((snapshot / "snapshot.timing.json.commit.json").is_file())

            execution_id = json.loads((run / "manifest.json").read_text(encoding="utf-8"))[
                "execution_fingerprint"
            ]
            self.assertEqual(
                verify_resume_snapshot_execution(remote, execution_id)["execution_fingerprint"],
                execution_id,
            )
            with self.assertRaisesRegex(ValueError, "physical GPU UUID"):
                verify_resume_snapshot_execution(remote, "replacement-device")

            final_before = load_checkpoint(latest_committed_checkpoint(run / "checkpoints"))
            shutil.rmtree(run)
            restored = restore_or_verify_resume_snapshot(remote, run)
            self.assertEqual(restored, run)
            self.assertEqual(restore_or_verify_resume_snapshot(remote, run), run)
            self.assertTrue((run / "checkpoints" / "checkpoint_000001.pt").is_file())
            final_after = load_checkpoint(latest_committed_checkpoint(run / "checkpoints"))
            self.assertEqual(final_before["run_state"], final_after["run_state"])
            record = validate_run_directory(run)
            self.assertTrue(record["timing"]["complete"])
            self.assertTrue(record["comparison_eligible"])
            self.assertEqual(
                context["state"].data_order_hashes,
                final_after["run_state"]["data_order_hashes"],
            )

    def test_lost_runtime_restore_and_resume_matches_uninterrupted(self) -> None:
        """A restored warm-up boundary resumes to the uninterrupted scientific state.

        Inputs are generated tensors and two same-seed tiny configs with distinct
        run IDs. The partial local directory is deleted inside a temporary root,
        restored from verified files, and resumed. Exact scientific state must
        match; elapsed timing and run identity are intentionally excluded.
        """

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            train, validation, _ = _synthetic_fixture(seed=5402)
            full_dir = root / "full"
            resumed_dir = root / "resumed"
            remote = root / "remote" / "resumed"
            full_config = _runner_config(full_dir, warmup_epochs=1, postwarm_epochs=1)
            resumed_config = _runner_config(resumed_dir, warmup_epochs=1, postwarm_epochs=1)
            run_training(
                full_config, train, validation, full_dir, "cpu",
                artifacts=_artifacts(), data_identities=_identities(),
            )
            run_training(
                resumed_config, train, validation, resumed_dir, "cpu",
                artifacts=_artifacts(), data_identities=_identities(),
                stop_after_global_epoch=0, persistence_directory=remote,
            )
            shutil.rmtree(resumed_dir)
            restore_resume_snapshot(remote, resumed_dir)
            resume_run(
                resumed_config, train, validation, resumed_dir, "cpu",
                artifacts=_artifacts(), data_identities=_identities(),
                persistence_directory=remote,
            )
            report = verify_resume_equivalence(full_dir, resumed_dir)
            self.assertTrue(report["equivalent"])

    def test_support_bundle_restores_configs_split_masks_order_and_profiles(self) -> None:
        """Study support files survive a simulated lost runtime with exact bytes.

        Tiny stand-ins for the required config, split, donor mask/manifest,
        run-order manifest, and pre-block profile records are backed up beneath a
        temporary study only. Restoration verifies every checksum and never
        executes a profile or training command.
        """

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            study = root / "study"
            remote = root / "remote"
            files = {
                "configs/donor.json": b"{}",
                "split_indices.npz": b"split",
                "split_indices.npz.manifest.json": b"{}",
                "donor/masks.npz": b"mask",
                "donor/mask_manifest.json": b"{}",
                "run_order_manifest.json": b"{\"order\":[]}",
                "pre_block_profile/profile_samples.csv": b"component,latency_ms\nplain_exact,1\n",
                "pre_block_profile/profile_summary.json": b"{}",
            }
            for relative, content in files.items():
                path = study / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                backup_study_support(path, study, remote)
            shutil.rmtree(study)
            for relative, content in files.items():
                restored = restore_study_support(study / relative, study, remote)
                self.assertEqual(restored.read_bytes(), content)

    def test_corrupt_support_and_incomplete_snapshot_are_rejected(self) -> None:
        """Checksum corruption and snapshots missing committed timing cannot restore.

        All files live under a temporary directory. Corruption is deliberate and
        bounded to copied fixtures; failures raise before any destination run or
        support artifact is accepted.
        """

        with tempfile.TemporaryDirectory(dir=PROJECT_ROOT) as temporary:
            root = Path(temporary)
            study = root / "study"
            source = study / "config.json"
            source.parent.mkdir(parents=True)
            source.write_text("{}", encoding="utf-8")
            remote = root / "remote"
            copied = backup_study_support(source, study, remote)
            copied.write_text("corrupt", encoding="utf-8")
            source.unlink()
            with self.assertRaisesRegex(ValueError, "checksum"):
                restore_study_support(source, study, remote)

            run = root / "active" / "partial"
            persistence = root / "persist" / "partial"
            config = _runner_config(run, warmup_epochs=1, postwarm_epochs=1)
            train, validation, _ = _synthetic_fixture(seed=5403)
            run_training(
                config, train, validation, run, "cpu",
                artifacts=_artifacts(), data_identities=_identities(),
                stop_after_global_epoch=0, persistence_directory=persistence,
            )
            timing_marker = (
                persistence / "snapshots" / "checkpoint_000001"
                / "snapshot.timing.json.commit.json"
            )
            timing_marker.unlink()
            with self.assertRaisesRegex(RuntimeError, "No completely verified"):
                restore_resume_snapshot(persistence, root / "restored" / "partial")
            with self.assertRaisesRegex(RuntimeError, "No verified compatible"):
                verify_resume_snapshot_execution(persistence, "different")


class HistoricalArchiveSetupTests(unittest.TestCase):
    """Verify the offline historical-results transfer artifact without extraction."""

    def test_results_archive_identity_and_layout(self) -> None:
        """The supplied archive matches its recorded hash and single safe root.

        The local archive is read only in bounded hash chunks and through ZIP
        metadata; it is not extracted or modified. Missing, corrupt, unsafe, or
        differently rooted archives fail before a fresh Colab setup could test.
        """

        manifest = json.loads(
            (PROJECT_ROOT / "configs" / "phase_d_v4_reference_manifest.json").read_text(encoding="utf-8")
        )
        identity = manifest["reference_results_archive"]
        archive = PROJECT_ROOT / identity["path"]
        self.assertEqual(archive.stat().st_size, identity["size_bytes"])
        self.assertEqual(sha256_file(archive), identity["sha256"])
        with zipfile.ZipFile(archive) as bundle:
            names = bundle.namelist()
        self.assertTrue(names)
        self.assertTrue(all(name.startswith(identity["required_top_level"]) for name in names))
        self.assertTrue(all(".." not in Path(name).parts and not Path(name).is_absolute() for name in names))


if __name__ == "__main__":
    unittest.main()
