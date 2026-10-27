"""M0 identity, validation, and preservation tests for Phase D v4."""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from phase_d_v4_io import (
    METHOD_SPECS,
    REPLAY_METHODS,
    SOURCE_DERIVED_SCHEDULE_METHODS,
    SCHEMA_VERSION,
    MaskMetadata,
    MaskRole,
    MethodId,
    RunConfig,
    StepResult,
    SyntheticBehavior,
    UpdateType,
    artifact_tree_hash,
    canonical_json,
    fingerprint,
    load_run_config,
    sha256_file,
    split_identity_hash,
    validate_config,
    validate_issue_register,
    validate_output_root,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "phase_d_v4_reference.json"
MANIFEST_PATH = PROJECT_ROOT / "configs" / "phase_d_v4_reference_manifest.json"


def _reference_values() -> dict:
    """Return a fresh validated reference mapping without changing files."""

    return load_run_config(CONFIG_PATH).to_dict()


def _set_nested(values: dict, path: tuple[str, ...], replacement: object) -> None:
    """Replace one nested dictionary leaf in a mutable test fixture in place."""

    cursor = values
    for key in path[:-1]:
        cursor = cursor[key]
    cursor[path[-1]] = replacement


def _add_replay_source(values: dict) -> None:
    """Attach a complete synthetic donor identity to a replay config fixture.

    The input mapping is mutated in place and nothing is returned. Hashes are
    deterministic placeholders used only for pre-training schema validation.
    """

    values["scientific"]["replay_source"] = {
        "run_id": "synthetic-relative-donor",
        "checkpoint_sha256": "a" * 64,
        "seed": values["scientific"]["identity"]["classifier_seed"],
        "method": MethodId.RELATIVE_FEEDBACK.value,
        "scientific_fingerprint": "b" * 64,
    }


def _add_schedule_source(values: dict) -> None:
    """Attach a complete synthetic donor identity to an M5 config fixture.

    The input mapping is mutated in place and nothing is returned. Hashes are
    schema-only placeholders independent of any scientific result.
    """

    values["scientific"]["schedule_source"] = {
        "run_id": "synthetic-relative-donor",
        "mask_sha256": "c" * 64,
        "checkpoint_sha256": "a" * 64,
        "seed": values["scientific"]["identity"]["classifier_seed"],
        "method": MethodId.RELATIVE_FEEDBACK.value,
        "scientific_fingerprint": "b" * 64,
    }


def _leaf_paths(value: object, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    """List dictionary leaf paths used to audit fingerprint coverage."""

    if isinstance(value, dict):
        paths: list[tuple[str, ...]] = []
        for key, child in value.items():
            paths.extend(_leaf_paths(child, prefix + (key,)))
        return paths
    return [prefix]


def _different_json_value(value: object) -> object:
    """Return a deterministic unequal JSON value for identity sensitivity tests."""

    if value is None:
        return "identity-test-value"
    if isinstance(value, bool):
        return not value
    if isinstance(value, (int, float)):
        return value + 1
    if isinstance(value, str):
        return value + "__changed"
    if isinstance(value, list):
        return value + ["__changed"]
    raise TypeError(f"unsupported test leaf: {type(value)!r}")


class ConfigIdentityTests(unittest.TestCase):
    """Verify resolved config identity and validation behavior."""

    def test_config_round_trip_preserves_every_value(self) -> None:
        """A JSON round trip returns the exact resolved configuration."""

        config = load_run_config(CONFIG_PATH)
        restored = RunConfig.from_dict(json.loads(json.dumps(config.to_dict())))
        validate_config(restored)
        self.assertEqual(config.to_dict(), restored.to_dict())

    def test_canonical_hash_ignores_dictionary_order(self) -> None:
        """Reordering dictionaries does not change canonical scientific identity."""

        values = _reference_values()
        reversed_values = dict(reversed(list(values.items())))
        artifacts = {"v3_source": "a" * 64, "v3_results": "b" * 64}
        self.assertEqual(fingerprint(RunConfig(values), artifacts), fingerprint(RunConfig(reversed_values), dict(reversed(list(artifacts.items())))))

    def test_every_scientific_leaf_changes_identity(self) -> None:
        """Changing any resolved scientific leaf changes the canonical fingerprint."""

        values = _reference_values()
        artifacts = {"v3_source": "a" * 64}
        original = fingerprint(RunConfig(values), artifacts)
        for path in _leaf_paths(values["scientific"], ("scientific",)):
            with self.subTest(path=".".join(path)):
                changed = copy.deepcopy(values)
                cursor: object = values
                for key in path:
                    cursor = cursor[key]  # type: ignore[index]
                _set_nested(changed, path, _different_json_value(cursor))
                self.assertNotEqual(original, fingerprint(RunConfig(changed), artifacts))

    def test_execution_and_output_values_do_not_change_scientific_identity(self) -> None:
        """Hardware and output paths remain outside the scientific fingerprint."""

        values = _reference_values()
        artifacts = {"v3_source": "a" * 64}
        original = fingerprint(RunConfig(values), artifacts)
        values["execution"]["device"] = "cpu"
        values["output"]["run_id"] = "another-run"
        self.assertEqual(original, fingerprint(RunConfig(values), artifacts))

    def test_classifier_seed_and_mask_replicate_change_identity(self) -> None:
        """Classifier seed and applicable mask replica both alter scientific identity."""

        artifacts = {"v3_source": "a" * 64}
        values = _reference_values()
        original = fingerprint(RunConfig(values), artifacts)
        values["scientific"]["identity"]["classifier_seed"] = 1
        self.assertNotEqual(original, fingerprint(RunConfig(values), artifacts))

        values = _reference_values()
        identity = values["scientific"]["identity"]
        spec = METHOD_SPECS[MethodId.RELATIVE_GAP_PERMUTATION_EPOCH]
        identity.update(
            method=MethodId.RELATIVE_GAP_PERMUTATION_EPOCH.value,
            algorithm_version=spec["version"],
            normalization=spec["normalization"].value,
            synthetic_behavior=SyntheticBehavior.LEARNED.value,
            mask_role=spec["mask_role"].value,
            mask_sha256="f" * 64,
            retrospective=True,
            deployment_eligible=False,
            mask_replicate=0,
        )
        first = fingerprint(RunConfig(values), artifacts)
        identity["mask_replicate"] = 1
        self.assertNotEqual(first, fingerprint(RunConfig(values), artifacts))

    def test_invalid_core_values_fail_before_training(self) -> None:
        """Invalid horizons, windows, quantiles, EMA, and combinations are rejected."""

        cases = [
            (("scientific", "training", "warmup_epochs"), 0),
            (("scientific", "training", "batch_size"), -1),
            (("scientific", "controller", "history_window"), 0),
            (("scientific", "controller", "low_quantile"), 0.8),
            (("scientific", "controller", "ema_old_value_weight"), 1.1),
            (("scientific", "identity", "normalization"), "standard_bn"),
            (("scientific", "identity", "algorithm_version"), "v4"),
            (("scientific", "data", "final_test_loaded"), True),
            (("execution", "precision"), "fp16"),
        ]
        for path, replacement in cases:
            with self.subTest(path=".".join(path)):
                values = _reference_values()
                _set_nested(values, path, replacement)
                with self.assertRaises(ValueError):
                    validate_config(RunConfig(values))

    def test_mask_dependent_method_requires_mask_hash(self) -> None:
        """A replay method cannot validate without an immutable source mask identity."""

        values = _reference_values()
        identity = values["scientific"]["identity"]
        identity.update(
            method="relative_replay_learned",
            algorithm_version="v4",
            synthetic_behavior="learned",
            mask_role="schedule",
            retrospective=True,
            deployment_eligible=False,
        )
        with self.assertRaisesRegex(ValueError, "requires an immutable mask"):
            validate_config(RunConfig(values))
        identity["mask_sha256"] = "c" * 64
        _add_replay_source(values)
        validate_config(RunConfig(values))

    def test_every_registered_method_has_one_valid_identity(self) -> None:
        """Every historical and planned method resolves to an explicit valid identity."""

        for method, spec in METHOD_SPECS.items():
            with self.subTest(method=method.value):
                values = _reference_values()
                identity = values["scientific"]["identity"]
                identity.update(
                    method=method.value,
                    algorithm_version=spec["version"],
                    normalization=spec["normalization"].value,
                    synthetic_behavior=sorted(item.value for item in spec["behaviors"])[0],
                    mask_role=spec["mask_role"].value,
                    mask_sha256="e" * 64 if spec["mask_required"] else None,
                    retrospective=spec["retrospective"],
                    deployment_eligible=spec["deployment_eligible"],
                    mask_replicate=0 if method == MethodId.RELATIVE_GAP_PERMUTATION_EPOCH else None,
                )
                if method in REPLAY_METHODS:
                    _add_replay_source(values)
                if method in SOURCE_DERIVED_SCHEDULE_METHODS:
                    _add_schedule_source(values)
                validate_config(RunConfig(values))

    def test_retrospective_controls_reject_deployment_labels(self) -> None:
        """Source-derived matched and replay controls reject misleading eligibility."""

        methods = (MethodId.MATCHED_PERIODIC, MethodId.EXACT_HEAD_MASK, MethodId.RELATIVE_REPLAY_LEARNED)
        for method in methods:
            spec = METHOD_SPECS[method]
            for field, wrong_value in (("retrospective", False), ("deployment_eligible", True)):
                with self.subTest(method=method.value, field=field):
                    values = _reference_values()
                    identity = values["scientific"]["identity"]
                    identity.update(
                        method=method.value,
                        algorithm_version=spec["version"],
                        normalization=spec["normalization"].value,
                        synthetic_behavior=sorted(item.value for item in spec["behaviors"])[0],
                        mask_role=spec["mask_role"].value,
                        mask_sha256="e" * 64,
                        retrospective=spec["retrospective"],
                        deployment_eligible=spec["deployment_eligible"],
                    )
                    if method in REPLAY_METHODS:
                        _add_replay_source(values)
                    identity[field] = wrong_value
                    with self.assertRaises(ValueError):
                        validate_config(RunConfig(values))


class RecordTests(unittest.TestCase):
    """Verify M0 records preserve scientific distinctions and scalar outputs."""

    def test_step_result_accepts_only_detached_scalar_metrics(self) -> None:
        """Step outputs reject negative counters and non-scalar metric objects."""

        result = StepResult(UpdateType.EXACT, exact_trunk_backwards=1, metrics={"loss": 1.25})
        self.assertEqual(result.exact_trunk_backwards, 1)
        with self.assertRaises(ValueError):
            StepResult(UpdateType.EXACT, trunk_optimizer_steps=-1)
        with self.assertRaises(ValueError):
            StepResult(UpdateType.EXACT, metrics={"retained": object()})

    def test_mask_metadata_keeps_roles_and_boundaries_explicit(self) -> None:
        """Mask metadata accepts valid epoch boundaries and rejects malformed ones."""

        metadata = MaskMetadata(MaskRole.HEAD_UPDATE, 5, (0, 2, 5), "d" * 64, "fixture", "1")
        self.assertEqual(metadata.role, MaskRole.HEAD_UPDATE)
        with self.assertRaises(ValueError):
            MaskMetadata(MaskRole.SCHEDULE, 5, (0, 2, 4), "d" * 64, "fixture", "1")


class PreservationTests(unittest.TestCase):
    """Verify output protection and the immutable historical reference manifest."""

    def test_rejects_historical_and_governing_output_roots(self) -> None:
        """Protected directories and source/specification files cannot be outputs."""

        rejected = [
            ".",
            "archive",
            "archive/new",
            "data/new",
            "cifar_d_resnet_v3/new",
            "AGENTS.md",
            "IMPLEMENTATION_PLAN.md",
            "RESEARCH_SPEC.md",
            "gradient_phase_d_resnet_v3.py",
        ]
        for path in rejected:
            with self.subTest(path=path), self.assertRaises(ValueError):
                validate_output_root(path, PROJECT_ROOT)
        accepted = validate_output_root("runs_phase_d_v4/study/run", PROJECT_ROOT)
        self.assertTrue(str(accepted).endswith(str(Path("runs_phase_d_v4/study/run"))))

    def test_governing_source_hashes_match_manifest(self) -> None:
        """Governing instructions, source, and specification retain audited bytes."""

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        for relative, expected in manifest["governing_artifacts"].items():
            with self.subTest(path=relative):
                path = PROJECT_ROOT / relative
                self.assertEqual(path.stat().st_size, expected["size_bytes"])
                self.assertEqual(sha256_file(path), expected["sha256"])

    def test_reference_results_tree_matches_manifest(self) -> None:
        """All saved v3 result bytes retain the audited aggregate tree identity."""

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        expected = manifest["reference_results"]
        actual_hash, entries = artifact_tree_hash(PROJECT_ROOT / expected["path"])
        self.assertEqual(actual_hash, expected["tree_sha256"])
        self.assertEqual(len(entries), expected["file_count"])
        self.assertEqual(sum(entry["size_bytes"] for entry in entries), expected["total_size_bytes"])

    def test_split_contract_and_config_hashes_match_manifest(self) -> None:
        """The declared partition contract and resolved config retain their identities."""

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        config = _reference_values()
        data = config["scientific"]["data"]
        keys = (
            "dataset", "source_partition", "dataset_size", "split_seed", "split_algorithm",
            "validation_selector", "training_selector", "training_size", "validation_size",
        )
        split = {key: data[key] for key in keys}
        self.assertEqual(split_identity_hash(split), manifest["split_identity"]["sha256"])
        self.assertEqual(sha256_file(CONFIG_PATH), manifest["reference_config"]["sha256"])
        artifacts = {
            "v3_source": manifest["governing_artifacts"]["gradient_phase_d_resnet_v3.py"]["sha256"],
            "v3_results": manifest["reference_results"]["tree_sha256"],
            "split_contract": manifest["split_identity"]["sha256"],
        }
        self.assertEqual(
            fingerprint(RunConfig(config), artifacts),
            manifest["scientific_fingerprint"]["sha256"],
        )

    def test_issue_register_assigns_every_plan_issue(self) -> None:
        """Every S1-S15 and A1-A18 plan issue names an owner milestone."""

        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        validate_issue_register(manifest["issue_register_owners"])

    def test_import_and_validation_create_no_output(self) -> None:
        """Identity validation does not create a run directory or other artifact."""

        candidate = PROJECT_ROOT / "runs_phase_d_v4" / "__m0_validation_must_not_create__"
        self.assertFalse(candidate.exists(), "reserved no-write test path already exists")
        validate_config(load_run_config(CONFIG_PATH))
        validate_output_root(candidate, PROJECT_ROOT)
        self.assertFalse(candidate.exists())

    def test_schema_constant_matches_reference_files(self) -> None:
        """Code, config, and manifest advertise one exact schema version."""

        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.assertEqual(config["schema_version"], SCHEMA_VERSION)
        self.assertEqual(manifest["schema_version"], SCHEMA_VERSION)


if __name__ == "__main__":
    unittest.main()
