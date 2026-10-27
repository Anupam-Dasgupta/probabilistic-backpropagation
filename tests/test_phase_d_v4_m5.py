"""Synthetic schedule construction and runner tests for milestone M5."""

from __future__ import annotations

import copy
import json
import math
import unittest
from pathlib import Path
from typing import Any

import numpy as np
import torch

from gradient_phase_d_resnet_v4 import (
    historical_midpoint_periodic_mask,
    make_epoch_matched_mask,
    make_frontloaded_mask,
    make_periodic_mask,
    make_uniform_third_mask,
    mask_identity_hash,
    permute_epoch_gaps,
    resume_run,
    run_training,
    schedule_mask_diagnostics,
    scheduled_exact,
)
from phase_d_v4_analysis import (
    average_mask_replicas_within_seed,
    build_controller_schedule_table,
)
from phase_d_v4_io import METHOD_SPECS, MethodId, RunConfig, capture_rng_state
from tests.test_phase_d_v4_m4 import (
    ReplayFixture,
    _artifacts,
    _assert_tree_equal,
    _base_config,
    _identities,
)


def _epoch_shape(mask: list[bool]) -> dict[str, Any]:
    """Return invariant spacing features for one epoch mask.

    The input is read only. Output includes length/count, first/last positions,
    leading/trailing false lengths, and the sorted internal gap multiset.
    """

    exact = [index for index, value in enumerate(mask) if value]
    gaps = [right - left - 1 for left, right in zip(exact[:-1], exact[1:], strict=True)]
    return {
        "length": len(mask),
        "count": len(exact),
        "first": exact[0] if exact else None,
        "last": exact[-1] if exact else None,
        "leading": exact[0] if exact else len(mask),
        "trailing": len(mask) - 1 - exact[-1] if exact else len(mask),
        "gaps": sorted(gaps),
    }


def _m5_config(
    root: Path,
    run_id: str,
    method: MethodId,
    donor_manifest: dict[str, Any] | None = None,
    donor_mask: list[bool] | None = None,
    offsets: list[int] | None = None,
    *,
    replicate: int = 0,
) -> RunConfig:
    """Build a tiny fixed or donor-derived M5 configuration.

    Inputs identify the method/output and, for relative controls, the donor
    manifest/mask/boundaries plus replica. The returned config contains the
    constructed schedule hash and complete source provenance. No file or RNG
    state changes; missing donor inputs fail for source-derived methods.
    """

    values = _base_config(root, run_id, method).to_dict()
    source_methods = {
        MethodId.RELATIVE_PERIODIC_TOTAL,
        MethodId.RELATIVE_PERIODIC_EPOCH,
        MethodId.RELATIVE_GAP_PERMUTATION_EPOCH,
    }
    if method not in source_methods:
        return RunConfig(values)
    if donor_manifest is None or donor_mask is None or offsets is None:
        raise ValueError("source-derived M5 config requires donor inputs")
    if method == MethodId.RELATIVE_PERIODIC_TOTAL:
        schedule = make_periodic_mask(len(donor_mask), sum(donor_mask))
    elif method == MethodId.RELATIVE_PERIODIC_EPOCH:
        schedule = make_epoch_matched_mask(donor_mask, offsets)
    else:
        schedule, _ = permute_epoch_gaps(
            donor_mask, offsets, classifier_seed=0, replicate=replicate
        )
        values["scientific"]["identity"]["mask_replicate"] = replicate
    export = donor_manifest["replay_export"]
    values["scientific"]["identity"]["mask_sha256"] = mask_identity_hash(schedule, offsets)
    values["scientific"]["schedule_source"] = {
        "run_id": export["source_run_id"],
        "mask_sha256": export["sha256"],
        "checkpoint_sha256": export["source_checkpoint_sha256"],
        "seed": export["source_seed"],
        "method": export["source_method"],
        "scientific_fingerprint": export["source_scientific_fingerprint"],
    }
    return RunConfig(values)


class ScheduleConstructionTests(unittest.TestCase):
    """Verify periodic, fixed, and gap-permutation algorithms analytically."""

    def test_new_periodic_all_small_budgets_and_historical_midpoint(self) -> None:
        """Every small ``n,k`` has exact count and the two periodic rules stay distinct."""

        for length in range(0, 13):
            for exact_count in range(length + 1):
                with self.subTest(length=length, exact_count=exact_count):
                    mask = make_periodic_mask(length, exact_count)
                    expected = (
                        {math.floor(index * length / exact_count) for index in range(exact_count)}
                        if exact_count else set()
                    )
                    self.assertEqual({index for index, value in enumerate(mask) if value}, expected)
                    self.assertEqual(sum(mask), exact_count)
                    if exact_count:
                        self.assertTrue(mask[0])
                    if exact_count == length:
                        self.assertTrue(all(mask))
        historical = historical_midpoint_periodic_mask(10, 3)
        self.assertEqual(
            [index for index, value in enumerate(historical) if value], [1, 5, 8]
        )
        self.assertNotEqual(historical, make_periodic_mask(10, 3))
        with self.assertRaises(ValueError):
            historical_midpoint_periodic_mask(10, 0)

    def test_epoch_total_frontloaded_and_uniform_budget_semantics(self) -> None:
        """Matched and fixed constructors preserve exactly their declared budgets."""

        source = [False, True, False, False, False, True, True, False, False]
        offsets = [0, 4, 9]
        epoch = make_epoch_matched_mask(source, offsets)
        self.assertEqual(sum(epoch[:4]), sum(source[:4]))
        self.assertEqual(sum(epoch[4:]), sum(source[4:]))
        total = make_periodic_mask(len(source), sum(source))
        self.assertEqual(sum(total), sum(source))
        self.assertNotEqual([sum(total[:4]), sum(total[4:])], [sum(source[:4]), sum(source[4:])])

        frontloaded = make_frontloaded_mask([10] * 12)
        diagnostics = schedule_mask_diagnostics(frontloaded, [10 * index for index in range(13)])
        self.assertEqual(diagnostics["epoch_exact_counts"], [4, 4, 4, 4, 4, 4, 3, 3, 3, 3, 3, 3])
        uniform = make_uniform_third_mask([1, 2, 3, 4, 5])
        uniform_diagnostics = schedule_mask_diagnostics(uniform, [0, 1, 3, 6, 10, 15])
        self.assertEqual(uniform_diagnostics["epoch_exact_counts"], [0, 1, 1, 1, 2])

    def test_gap_permutation_preserves_every_epoch_and_cross_boundary_invariant(self) -> None:
        """Permutation preserves declared spacing invariants and source bytes."""

        source = [
            False, True, False, False, True, False, True, False,
            False, False, True, False, False, True, False, True, False,
            False, False, False,
        ]
        offsets = [0, 8, 17, 20]
        source_before = list(source)
        permuted, details = permute_epoch_gaps(
            source, offsets, classifier_seed=3, replicate=2
        )
        self.assertEqual(source, source_before)
        self.assertEqual(len(permuted), len(source))
        for start, end in zip(offsets[:-1], offsets[1:], strict=True):
            self.assertEqual(_epoch_shape(source[start:end]), _epoch_shape(permuted[start:end]))
        for boundary in offsets[1:-1]:
            left_source = _epoch_shape(source[offsets[offsets.index(boundary)-1]:boundary])["trailing"]
            right_source = _epoch_shape(source[boundary:offsets[offsets.index(boundary)+1]])["leading"]
            left_new = _epoch_shape(permuted[offsets[offsets.index(boundary)-1]:boundary])["trailing"]
            right_new = _epoch_shape(permuted[boundary:offsets[offsets.index(boundary)+1]])["leading"]
            self.assertEqual(left_source + right_source, left_new + right_new)
        self.assertEqual(details["replicate"], 2)
        self.assertTrue(all(
            record["seed"] == 900000 + 3000 + 200 + record["postwarm_epoch"]
            for record in details["epoch_records"]
        ))

    def test_gap_permutation_degenerate_epochs_rng_independence_and_reproducibility(self) -> None:
        """Zero/one/equal-gap epochs are labeled and private RNG never leaks."""

        source = [False, False, False, True, False, False, True, False, True, False, True]
        offsets = [0, 3, 6, 11]
        rng_before = capture_rng_state()
        first, details_first = permute_epoch_gaps(source, offsets, classifier_seed=1, replicate=0)
        second, details_second = permute_epoch_gaps(source, offsets, classifier_seed=1, replicate=0)
        rng_after = capture_rng_state()
        _assert_tree_equal(self, rng_before, rng_after)
        self.assertEqual(first, second)
        self.assertEqual(details_first, details_second)
        self.assertFalse(details_first["epoch_records"][0]["randomizable"])
        self.assertFalse(details_first["epoch_records"][1]["randomizable"])
        equal_gap_record = details_first["epoch_records"][2]
        self.assertTrue(equal_gap_record["randomizable"])
        self.assertTrue(equal_gap_record["unchanged"])
        self.assertEqual(equal_gap_record["reason"], "permutation_sampled_once")

    def test_analysis_labels_and_replica_averaging_stay_within_seed(self) -> None:
        """Controller tables preserve eligibility and replicas do not inflate seeds."""

        table = build_controller_schedule_table(
            [
                {
                    "method": "relative_periodic_epoch", "retrospective": True,
                    "deployment_eligible": False, "source_run_id": "donor",
                    "source_mask_sha256": "a" * 64,
                    "matching_scope": "per_epoch_exact_budget_not_burst_or_phase_matching",
                },
                {
                    "method": "fixed_uniform_third", "retrospective": False,
                    "deployment_eligible": True, "source_run_id": None,
                    "source_mask_sha256": None,
                    "matching_scope": "fixed_uniform_not_donor_matched",
                },
            ]
        )
        self.assertEqual(len(table), 2)
        records = [
            {
                "classifier_seed": 4, "method": "relative_gap_permutation_epoch",
                "source_run_id": "donor", "source_mask_sha256": "a" * 64,
                "replicate": replicate, "final_accuracy": 0.7 + 0.01 * replicate,
                "exact_count": 5, "moved_gap_fraction": 0.2 * replicate,
            }
            for replicate in (0, 1, 2)
        ]
        averaged = average_mask_replicas_within_seed(records)
        self.assertEqual(len(averaged), 1)
        self.assertEqual(averaged[0]["replica_count"], 3)
        self.assertEqual(averaged[0]["replicates"], [0, 1, 2])
        self.assertFalse(averaged[0]["replicas_are_independent_seeds"])
        self.assertAlmostEqual(averaged[0]["mean_final_accuracy"], 0.71)


class ScheduleRunnerTests(ReplayFixture):
    """Exercise all M5 conditions, provenance, rejection, and resume."""

    @classmethod
    def setUpClass(cls) -> None:
        """Create the shared synthetic donor and load its immutable mask arrays."""

        super().setUpClass()
        with np.load(cls.donor_mask_path, allow_pickle=False) as arrays:
            cls.source_mask = [bool(value) for value in arrays["actual_trunk_exact"].tolist()]
            cls.offsets = [int(value) for value in arrays["epoch_offsets"].tolist()]

    def run_schedule(
        self,
        method: MethodId,
        run_id: str,
        *,
        replicate: int = 0,
        stop_after_global_epoch: int | None = None,
    ) -> tuple[RunConfig, dict[str, Any]]:
        """Run one tiny M5 method with donor paths only when scientifically required.

        Inputs select condition, output, replica, and optional stop boundary. The
        returned config/context are local synthetic artifacts. No real dataset,
        remote service, or scientific execution is used.
        """

        config = _m5_config(
            self.root, run_id, method, self.donor_manifest, self.source_mask,
            self.offsets, replicate=replicate,
        )
        source_derived = method in {
            MethodId.RELATIVE_PERIODIC_TOTAL,
            MethodId.RELATIVE_PERIODIC_EPOCH,
            MethodId.RELATIVE_GAP_PERMUTATION_EPOCH,
        }
        context = run_training(
            config,
            self.train,
            self.validation,
            self.root / run_id,
            "cpu",
            artifacts=_artifacts(),
            data_identities=_identities(),
            mask_path=self.donor_mask_path if source_derived else None,
            mask_manifest_path=self.donor_manifest_path if source_derived else None,
            stop_after_global_epoch=stop_after_global_epoch,
        )
        return config, context

    def test_all_m5_runs_match_declared_budgets_and_provenance(self) -> None:
        """Short runs persist every schedule's exact budget, spacing, and labels."""

        methods = (
            MethodId.RELATIVE_PERIODIC_TOTAL,
            MethodId.RELATIVE_PERIODIC_EPOCH,
            MethodId.FIXED_FRONTLOADED,
            MethodId.FIXED_UNIFORM_THIRD,
            MethodId.RELATIVE_GAP_PERMUTATION_EPOCH,
        )
        contexts: dict[MethodId, dict[str, Any]] = {}
        for method in methods:
            _, contexts[method] = self.run_schedule(method, f"m5-{method.value}")
        total = contexts[MethodId.RELATIVE_PERIODIC_TOTAL]["state"].replay_schedule_mask
        epoch = contexts[MethodId.RELATIVE_PERIODIC_EPOCH]["state"].replay_schedule_mask
        gap = contexts[MethodId.RELATIVE_GAP_PERMUTATION_EPOCH]["state"].replay_schedule_mask
        self.assertEqual(sum(total), sum(self.source_mask))
        for start, end in zip(self.offsets[:-1], self.offsets[1:], strict=True):
            self.assertEqual(sum(epoch[start:end]), sum(self.source_mask[start:end]))
            self.assertEqual(_epoch_shape(gap[start:end]), _epoch_shape(self.source_mask[start:end]))
        for method, context in contexts.items():
            state = context["state"]
            self.assertEqual(state.executed_exact_mask, state.replay_schedule_mask)
            self.assertEqual(state.executed_head_mask, state.replay_schedule_mask)
            self.assertEqual(state.mask_cursor, len(state.replay_schedule_mask))
            provenance = json.loads(
                (self.root / f"m5-{method.value}" / "schedule_provenance.json").read_text(encoding="utf-8")
            )
            diagnostics = provenance["mask"]["construction_details"]
            self.assertEqual(
                diagnostics.get("constructed_diagnostics", diagnostics)["exact_count"],
                sum(state.replay_schedule_mask),
            )
            if method in {MethodId.FIXED_FRONTLOADED, MethodId.FIXED_UNIFORM_THIRD}:
                self.assertFalse(provenance["retrospective"])
                self.assertTrue(provenance["deployment_eligible"])
                self.assertNotIn("dependency_inclusive", provenance["cost_records"])
                self.assertIsNone(provenance["mask"]["source_run_id"])
            else:
                self.assertTrue(provenance["retrospective"])
                self.assertFalse(provenance["deployment_eligible"])
                self.assertIn("dependency_inclusive", provenance["cost_records"])
        self.assertEqual(self.source_mask, self.donor["state"].executed_exact_mask)

    def test_incompatible_hash_and_fixed_donor_fail_before_output(self) -> None:
        """Constructed-hash mismatches and donor access by fixed methods fail pre-update."""

        values = _m5_config(
            self.root, "bad-hash", MethodId.RELATIVE_PERIODIC_EPOCH,
            self.donor_manifest, self.source_mask, self.offsets,
        ).to_dict()
        values["scientific"]["identity"]["mask_sha256"] = "f" * 64
        bad_config = RunConfig(values)
        bad_output = self.root / "bad-hash"
        with self.assertRaisesRegex(ValueError, "constructed schedule hash"):
            run_training(
                bad_config, self.train, self.validation, bad_output, "cpu",
                artifacts=_artifacts(), data_identities=_identities(),
                mask_path=self.donor_mask_path, mask_manifest_path=self.donor_manifest_path,
            )
        self.assertFalse(bad_output.exists())

        fixed = _m5_config(self.root, "bad-fixed", MethodId.FIXED_UNIFORM_THIRD)
        fixed_output = self.root / "bad-fixed"
        with self.assertRaisesRegex(ValueError, "must not receive donor"):
            run_training(
                fixed, self.train, self.validation, fixed_output, "cpu",
                artifacts=_artifacts(), data_identities=_identities(),
                mask_path=self.donor_mask_path, mask_manifest_path=self.donor_manifest_path,
            )
        self.assertFalse(fixed_output.exists())

    def test_gap_schedule_resume_matches_uninterrupted_and_history_stays_global(self) -> None:
        """Gap control resumes exactly while historical clocks retain prior semantics."""

        full_config, full = self.run_schedule(
            MethodId.RELATIVE_GAP_PERMUTATION_EPOCH, "gap-full", replicate=1
        )
        partial_config, partial = self.run_schedule(
            MethodId.RELATIVE_GAP_PERMUTATION_EPOCH,
            "gap-partial",
            replicate=1,
            stop_after_global_epoch=1,
        )
        resumed = resume_run(
            partial_config,
            self.train,
            self.validation,
            self.root / "gap-partial",
            "cpu",
            artifacts=_artifacts(),
            data_identities=_identities(),
            mask_path=self.donor_mask_path,
            mask_manifest_path=self.donor_manifest_path,
            checkpoint_path=partial["last_checkpoint"],
        )
        _assert_tree_equal(self, full["model"].state_dict(), resumed["model"].state_dict())
        _assert_tree_equal(self, full["predictor"].state_dict(), resumed["predictor"].state_dict())
        self.assertEqual(full["state"].replay_schedule_mask, resumed["state"].replay_schedule_mask)
        self.assertEqual(full["state"].data_order_hashes, resumed["state"].data_order_hashes)
        self.assertEqual(self.donor["state"].executed_exact_mask, [True, False, False, False, False, True])
        expected = [
            (step % ((4, 3, 2, 1)[min(3, int(4 * epoch / 12))] + 1)) == 0
            for epoch, step in ((0, 0), (0, 5), (3, 5), (6, 9), (11, 12))
        ]
        observed = [scheduled_exact(step, epoch, 12) for epoch, step in ((0, 0), (0, 5), (3, 5), (6, 9), (11, 12))]
        self.assertEqual(observed, expected)


if __name__ == "__main__":
    unittest.main()
