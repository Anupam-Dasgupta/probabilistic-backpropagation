# Phase D v4: pre-experiment M6 Colab and local workflow

This guide covers the implementation through the authorized pre-experiment
portions of M6. It supports configuration auditing, synthetic smoke/resume,
copied-state diagnostics, bounded profiling, explicit one-condition CIFAR-10
execution, resume, and read-only analysis. It does not record a Colab execution
or authorize the five-seed study, optional ablations, retries, sweeps, or M7.

## Verified environment

The implementation was tested on Windows with:

- Python 3.13.6
- PyTorch 2.10.0+cu130
- torchvision 0.25.0+cu130
- NumPy 2.3.3
- NVIDIA RTX 3060 Laptop GPU with CUDA available

These are the tested compatible PyTorch/torchvision versions. On Colab, first
inspect the preinstalled GPU stack instead of blindly reinstalling it:

```python
import os, sys
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
import torch, torchvision, numpy
print(sys.version)
print(torch.__version__, torchvision.__version__, numpy.__version__)
print(torch.cuda.is_available(), torch.version.cuda)
print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no CUDA")
```

If the runtime does not already provide a mutually compatible PyTorch and
torchvision pair, choose a pair from the official PyTorch installation matrix.
Do not replace a working Colab CUDA stack merely to reproduce the local `cu130`
build tag.

## Tests

From the repository root, use the supported launcher for the complete engineering
suite:

```bash
python -B run_phase_d_v4_tests.py
```

The launcher sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before any test imports
PyTorch or initializes CUDA. Strict deterministic algorithms remain enabled.
Launching discovery directly after another module has initialized CUDA is not a
supported equivalent and can produce cuBLAS determinism initialization errors.

Training tests use generated tensors and temporary local directories. The
preservation/readiness tests also read (but never modify or extract locally) the
existing historical results tree/archive. They do not load CIFAR-10, download
data, contact Drive, or launch a scientific run.

## Audit and synthetic smoke

Audit configuration and the current CUDA environment without creating a run:

```bash
python -B gradient_phase_d_resnet_v4.py audit --device cuda
```

Run the bounded synthetic workflow into a fresh directory:

```bash
python -B gradient_phase_d_resnet_v4.py smoke \
  --device cuda \
  --out runs_phase_d_v4/m2_engineering/colab_smoke
```

The smoke command uses 23 generated images, a width-4 one-block model, one
warm-up epoch, and one post-warm-up epoch. Its accuracy and timing are engineering
artifacts only and must not be reported as scientific evidence.

Resume a stopped synthetic workflow from its latest committed epoch boundary:

```bash
python -B gradient_phase_d_resnet_v4.py resume \
  --device cuda \
  --out runs_phase_d_v4/m2_engineering/colab_smoke
```

An explicit checkpoint can be supplied with `--checkpoint`. Mid-epoch state is
not resumed: an interrupted epoch restarts from the preceding committed boundary.

## Persistence behavior

Every epoch checkpoint is written to the local run directory through a temporary
file, flushed, atomically replaced, checksummed, and given a commit marker. CSV
and JSON outputs are rebuilt from committed checkpoint records on resume, so
uncommitted or duplicated rows are not retained.

The code includes a checksum-verified directory-copy helper used by local tests
to model Drive persistence. A completed immutable checkpoint is copied first,
verified, and its completion marker is published last. Actual Google Drive
mounting and transfer were not tested in M2. On Colab, keep the active run on the
local filesystem and do not treat a partial Drive file as resumable.

## Synthetic diagnostics and profiling

Run copied-state probes on generated tensors:

```bash
python -B gradient_phase_d_resnet_v4.py diagnose \
  --device cuda \
  --out runs_phase_d_v4/m3_engineering/colab_diagnose
```

This writes `gradient_diagnostics.csv`, `update_probes.csv`, and
`normalization_probes.csv`. Rows explicitly distinguish per-example from
batch-global aggregation and raw mean-loss-gradient units from normalized target
units. The normalization probe labels conventional train-mode BatchNorm exact BP
separately from the v3 frozen-head exact reference.

Run the bounded microprofile:

```bash
python -B gradient_phase_d_resnet_v4.py profile \
  --device cuda \
  --out runs_phase_d_v4/m3_engineering/colab_profile
```

This writes every timed observation to `profile_samples.csv` and summaries
recomputed from all observations to `profile_summary.json`. Two settling samples
are unreported; twelve samples per component are reported. Controller decision
overhead is a separate component from the exact-plus-predictor kernel. These
synthetic latencies and losses are engineering checks, not evidence of speed or
accuracy on CIFAR-10.

## Output meanings

A synthetic M2 run contains:

- `manifest.json`: resolved configuration, source/data identities, environment,
  and scientific/execution fingerprints.
- `epochs.csv`: one committed row per validation boundary, including warm-up.
- `controller_events.csv`: pre-fit feedback decisions when applicable.
- `masks.npz` and `mask_manifest.json`: executed post-warm exact mask and hashes.
- `timing.json`: non-overlapping timing components and completeness flags.
- `summary.json`: completion and checkpoint status.
- `checkpoint_journal.json`: post-commit checkpoint I/O durations.
- `checkpoints/`: immutable epoch-boundary checkpoints and commit metadata.
- `errors.jsonl`: reserved structured failure log; empty when no failure occurred.
- `final_predictor.pt`: final predictor state for predictor-bearing completed runs.

Accuracy is stored as a fraction, wall time in seconds, latency in milliseconds,
and memory in MiB. Missing values use blank/null plus status rather than zero.

## Scientific commands prepared in M6

`prepare-data`, `make-config`, `run`, `resume`, and `preflight` are now wired for
the frozen CIFAR-10 protocol. Dataset downloading occurs only with an explicit
`prepare-data --download`. Real-data commands also require the default-disabled
`--authorize-real-data` executable gate. A run that can enter post-warm-up work,
and every scientific resume, additionally requires the default-disabled
`--authorize-full-pilot` gate. These flags are execution interlocks, not a
substitute for human approval. `run` executes exactly one resolved condition and
never launches a matrix, retry, profile, or analysis. The final test partition
is never opened. Scientific resume requires the same config, split, environment,
and donor-mask identities. No command silently falls back to `smoke`.

## M4 fixed-mask replay controls

The runner now supports these programmatic fixed-mask conditions:

- `relative_replay_learned`
- `relative_replay_zero_shadow`
- `relative_replay_skip_shadow`
- `relative_replay_exact_trunk`
- `relative_replay_zero_fast`
- `relative_replay_skip_fast`

They are exposed only as one-condition commands behind the scientific execution
guards. A caller must provide a completed relative-feedback donor's `masks.npz` and
`mask_manifest.json`, plus a resolved replay config containing the exact source
run, checkpoint, seed, method, scientific fingerprint, and mask hash. Validation
of bytes, length, epoch boundaries, role, and provenance completes before model
initialization or output creation.

Replay outputs keep three arrays distinct: `schedule`, `actual_trunk_exact`, and
`head_update`. `replay_provenance.json` labels every donor-derived condition
retrospective and independently deployment-ineligible. It preserves separate
conditional-execution and dependency-inclusive component records; M4 does not
turn either record into a speed claim. Fast variants omit predictor work but do
not become deployable merely by being faster conditional on an observed donor
mask.

The engineering tests generate only a tiny synthetic donor and bounded replay
runs. Real-data replay, causal timing tables, and scientific conclusions require
later explicit authorization.

## M5 controller-usefulness schedules

The runner now implements five additional mask-driven conditions:

- `relative_periodic_total`: donor total exact count only; not epoch or temporal matching.
- `relative_periodic_epoch`: donor exact count per epoch; not burst or phase matching.
- `relative_gap_permutation_epoch`: donor epoch budgets, endpoints, prefixes,
  suffixes, and internal gap multiset; not complete temporal matching.
- `fixed_frontloaded`: predefined 0.40/0.35/0.30/0.30 three-epoch phases.
- `fixed_uniform_third`: predefined one-third exact budget per epoch.

The new periodic helper starts at slot zero and places exact steps at
`floor(j*n/k)`. The v3 midpoint construction remains a separate historical
helper. Gap permutation uses only the private seed formula recorded in the
resolved configuration; unchanged or unrandomizable epochs are retained and
reported without redrawing.

Each M5 run writes `schedule_provenance.json` and distinct schedule,
actual-trunk-exact, and head-update arrays. Provenance includes full/per-epoch
budgets, first/last exact positions, full-horizon synthetic-run distributions,
construction details, matching-scope labels, random seeds, and moved-gap
fractions. Donor-derived schedules remain retrospective and independently
deployment-ineligible. Fixed schedules reject donor artifacts.

M5 tests use a small synthetic donor only. M6 adds explicit execution wiring but
does not turn synthetic evidence into a scientific result.

## Ordered Colab execution

The runnable cells are in `PHASE_D_V4_COLAB.ipynb`. Every subprocess uses
`subprocess.run(..., check=True)`, so a failed setup, test, smoke, config, run,
resume, profile, or analysis command prevents dependent work. A fresh **Run all**
stops in cell 6 at a default-false executable guard before dataset download.
Run cells 1–5 only for environment/source verification and synthetic smoke.
Cells 6–9 require explicit real-data-preflight approval; cells 11–17 require a
separate full-pilot approval. The essential sequence is repeated below.

### Fresh-clone files required before tests

A Git clone intentionally excludes `cifar_d_resnet_v3/`. Before cells 3–4,
upload the existing `cifar_d_resnet_v3.zip` into the repository root, alongside
`AGENTS.md`. Its required SHA-256 is
`469dbc42283afe1ace2c8fe29dbdebcee3b22ff6092358ec503b2cdb2920e127`, size
998,800,703 bytes, and its only top-level extraction root is
`cifar_d_resnet_v3/`. The notebook verifies the archive and safe member layout,
extracts it only when the results directory is absent, then verifies the
extracted 44-file tree against `configs/phase_d_v4_reference_manifest.json`
before running the full preservation test. Do not upload it under `/content/data`
or alter/repack it. No dataset, `archive/` directory, or historical source copy
is needed for environment checks and synthetic smoke beyond the repository files
and this results archive.

### 1. Verify the T4 and source identities

Use a fresh GPU runtime. Before comparing timings, require
`torch.cuda.is_available()` and an NVIDIA T4 device name. Record Python, PyTorch,
torchvision, NumPy, CUDA/cuDNN, GPU identity, and the TF32/determinism flags. Do
not reinstall a working Colab PyTorch/torchvision pair.

Verify protected hashes from `configs/phase_d_v4_reference_manifest.json`, and
record hashes for every v4 Python module, the resolved reference config, and
`configs/phase_d_v4_pilot_seed0_manifest.json`.

### 2. Run tests and the required synthetic resume smoke

```bash
python -B run_phase_d_v4_tests.py
python -B gradient_phase_d_resnet_v4.py audit --device cuda
python -B gradient_phase_d_resnet_v4.py smoke --device cuda \
  --out /content/phase_d_v4_smoke --stop-after-global-epoch 0
python -B gradient_phase_d_resnet_v4.py resume --device cuda \
  --out /content/phase_d_v4_smoke
```

Smoke accuracy and timing are engineering checks only.

### 3. Prepare the official-train-only split

This is the only command that may download, because the flag is explicit:

```bash
python -B gradient_phase_d_resnet_v4.py prepare-data \
  --authorize-real-data \
  --config configs/phase_d_v4_reference.json \
  --data-dir /content/data \
  --split-file /content/runs_phase_d_v4/phase_d_v4_pilot_seed0/split_indices.npz \
  --download
```

The split uses `torch.randperm` seed 71503, with 5,000 validation and 45,000
training indices. Its NPZ and manifest contain separate content hashes.

### 4. Create a predictor warm-up boundary and bounded preflight

```bash
python -B gradient_phase_d_resnet_v4.py make-config \
  --method relative_feedback \
  --run-id seed0_relative_feedback_donor \
  --output-root /content/runs_phase_d_v4/phase_d_v4_pilot_seed0 \
  --out-config /content/runs_phase_d_v4/phase_d_v4_pilot_seed0/configs/seed0_relative_feedback_donor.json

python -B gradient_phase_d_resnet_v4.py run --device cuda \
  --authorize-real-data \
  --config /content/runs_phase_d_v4/phase_d_v4_pilot_seed0/configs/seed0_relative_feedback_donor.json \
  --data-dir /content/data \
  --split-file /content/runs_phase_d_v4/phase_d_v4_pilot_seed0/split_indices.npz \
  --out /content/runs_phase_d_v4/phase_d_v4_pilot_seed0/seed0_relative_feedback_donor \
  --stop-after-global-epoch 3

python -B gradient_phase_d_resnet_v4.py make-config \
  --method exact \
  --run-id seed0_exact_plain_preflight \
  --output-root /content/runs_phase_d_v4/phase_d_v4_pilot_seed0 \
  --out-config /content/runs_phase_d_v4/phase_d_v4_pilot_seed0/configs/seed0_exact_plain_preflight.json

python -B gradient_phase_d_resnet_v4.py run --device cuda \
  --authorize-real-data \
  --config /content/runs_phase_d_v4/phase_d_v4_pilot_seed0/configs/seed0_exact_plain_preflight.json \
  --data-dir /content/data \
  --split-file /content/runs_phase_d_v4/phase_d_v4_pilot_seed0/split_indices.npz \
  --out /content/runs_phase_d_v4/phase_d_v4_pilot_seed0/seed0_exact_plain_preflight \
  --stop-after-global-epoch 3

python -B gradient_phase_d_resnet_v4.py preflight --device cuda \
  --authorize-real-data \
  --config /content/runs_phase_d_v4/phase_d_v4_pilot_seed0/configs/seed0_relative_feedback_donor.json \
  --data-dir /content/data \
  --split-file /content/runs_phase_d_v4/phase_d_v4_pilot_seed0/split_indices.npz \
  --checkpoint /content/runs_phase_d_v4/phase_d_v4_pilot_seed0/seed0_relative_feedback_donor/checkpoints/checkpoint_000004.pt \
  --out /content/runs_phase_d_v4/phase_d_v4_pilot_seed0/preflight
```

The predictor and plain partial runs provide separate four-epoch warm-up timing.
Together they perform **eight complete passes over the 45,000-example training
split** before the full-pilot approval boundary;
the plain preflight run is never counted as a completed scientific trajectory.
The real-width preflight uses copied predictor warm-up state, two settling samples, and
twelve reported samples per component. It includes plain exact,
exact-plus-predictor, learned synthetic, zero-shadow, skip-shadow, and separately
CPU-timed controller overhead. It does not continue training.

### 5. Display the exact pilot and stop for approval

Display `configs/phase_d_v4_pilot_seed0_manifest.json`. It declares seed 0,
4+12 epochs, batch size 32, one gap replica, all dependencies, and:

- 1 donor-generation trajectory;
- 1 learned-replay equivalence trajectory, not an independent condition;
- 11 randomized measured trajectories: the nine-condition core plus required
  total-periodic and fixed-frontloaded comparisons;
- **13 actual full trajectories** and one independent classifier seed.

There is also one bounded plain-warm-up-only preflight trajectory. It is resource
overhead, not a completed accuracy condition or an independent sample.

Use `12 * 1407` post-warm slots and the measured component medians to report
low/high **scenarios**, not guaranteed runtime bounds, until the donor determines
the actual slot mix. Separate the actual study-resource estimate from attributed
standalone method costs. Attribute
the measured plain warm-up to the two exact references and predictor warm-up to
the eleven predictor-bearing full trajectories. Estimate storage
from the measured checkpoint size and the runner's actual retention behavior.
The prepared runner currently retains all epoch checkpoints, so the notebook
counts 13*16 full-run checkpoints plus four checkpoints from the extra plain
warm-up. It separately lists unestimated configs, split artifacts, masks, logs,
final predictors, plots, checkpoint growth, and Drive overhead. Runtime scenarios
omit or approximate dataset setup, validation, calibration, checkpoint/log/Drive
I/O, finalization, hardware drift, thermal effects, and mask mix. Never call
update counts FLOPs.

Stop here and present the T4 identity, smoke result, manifest, component medians,
runtime range, storage estimate, and exclusions. Do not complete the donor or
launch the measured block before approval.

### 6. Post-approval execution only

After approval, finish the donor with scientific `resume` using both
`--authorize-real-data` and `--authorize-full-pilot`. Resolve each config
with `make-config`; source-derived methods require both donor `masks.npz` and
`mask_manifest.json`, gap permutation also requires `--mask-replicate 0`, and
fixed frontloaded uses `--synthetic-behavior learned` without donor artifacts.

Run learned replay equivalence and require `analyze_phase_d_v4.py compare-replay`
to prove final classifier, predictor, optimizer, masks, counters, and data-order
state. Then execute the eleven measured trajectories one at a time; the notebook
derives the next unfinished run from the saved order rather than accepting an
arbitrary default. After the fresh measured feedback run, require:

```bash
python -B analyze_phase_d_v4.py compare-feedback-masks \
  .../seed0_relative_feedback_donor \
  .../seed0_relative_feedback_measured
```

A mismatch is a stop condition. Do not substitute a new mask after observing
outcomes.

### 7. Persistence and recovery

Keep active training under `/content`. Mount Drive only when the user chooses to
persist. Set the notebook's `PERSISTENCE_ROOT`; it passes `--persistence-dir` so
each locally committed epoch checkpoint is copied, SHA-256 verified, and given
its `.commit.json` marker last. Copy time remains in checkpoint/log I/O. Ignore
`.partial`, missing-marker, or mismatched remote files. This workflow is tested
only with local directory fixtures; Drive mounting, latency, disconnection, and
recovery remain Colab-only checks.

On recovery, restore the saved run-order manifest, resolved configs, split and
manifest, donor mask and manifest where applicable, and the marker-verified
checkpoint retention set to the original local layout. The retained set includes
the latest and preceding boundaries, warm-up-final boundary, available post-warm
probe boundaries 1/6/12, earliest committed boundary, and first-attainment
boundaries. The pre-block profile CSV/summary is restored as a prior measurement;
its absence is a hard failure and never triggers silent re-profiling. Resume with
both execution flags and the same
`--persistence-dir` from that epoch boundary. Record repeated incomplete epochs and downtime; resumed
runs may contribute accuracy evidence after continuation checks but are not clean
uninterrupted timing benchmarks. File restoration itself does not download data;
cell 14 performs only verified restoration, cell 15 owns the separate
`AUTHORIZE_RECOVERY_DATA_SETUP=False` gate, and cell 16 owns the independent
`AUTHORIZE_RECOVERY_CONTINUATION=False` gate. Each cell can be run later without
repeating or deleting a valid earlier stage.


### Verified epoch recovery snapshots

When `--persistence-dir` is enabled, every completed epoch writes:

- A checksum-verified immutable checkpoint and commit marker.
- An epoch-specific snapshot of run metadata and derived outputs.
- A `snapshot.ready.json` marker published only after all support files are verified.
- A checksum-committed `snapshot.timing.json` record. Recovery rejects snapshots
  lacking it and uses it to add snapshot work exactly once to checkpoint/log I/O.

The recovery workflow selects the newest fully verified snapshot and restores
it into a fresh local run directory. Rerunning restoration reuses the directory
only after every snapshot file and the latest committed checkpoint are verified
against the same newest remote snapshot; it never overwrites existing local state.

Immutable resolved configs, dataset split artifacts, donor masks, the saved
run-order manifest, and resource/pre-block profile records are backed up
separately under `_study_support`. Support copies are checksum-verified and
restore refuses existing differences.

The notebook distinguishes three cases:

1. Fresh run: execute `run`.
2. Existing incomplete local run: execute `resume`.
3. New Colab session: restore support files and a verified snapshot, then
   explicitly authorize `resume`.

Strict recovery remains the notebook default and requires the complete original
execution fingerprint, including the physical GPU UUID. For a Colab VM replacement,
set `RECOVERY_COMPATIBILITY = COMPATIBLE_T4_RESUME_POLICY` explicitly in the
recovery cell. The versioned policy accepts only a different physical UUID/index
and detailed VM platform/CPU description when both devices are NVIDIA T4s with
compute capability 7.5 and equal reported memory, and the Python/PyTorch/
torchvision/NumPy/CUDA/cuDNN/driver, OS family/architecture, device count,
precision, TF32, cuBLAS workspace, deterministic backend, resolved execution
configuration, source, data, mask, optimizer, and controller identities remain
compatible. Missing legacy evidence, another GPU model, or any required mismatch
stops before the run/donor snapshot is restored.

The display-driver release is queried separately from the CUDA runtime/build by
a bounded `nvidia-smi --query-gpu=driver_version` call. The environment records
the query source, status, version, and failure detail. Compatible-T4 recovery
requires valid `status=ok` evidence and an exact driver-version match; missing,
placeholder, failed, or legacy-absent evidence is rejected.

The original environment and execution fingerprint remain immutable. Recovery
appends the replacement environment, current execution fingerprint, policy
version, and transition to the session history. It records work by session,
reports recovery overhead separately, and leaves the interruption gap null
because browser/VM downtime cannot be reconstructed reliably. Cross-device
resumed trajectories may enter labeled accuracy analysis after the copied-state
continuation check. They are not new seeds and are excluded from clean timing,
speed, Pareto, or end-to-end efficiency comparisons; no cross-device timing
normalization is performed.

Before continuation begins, each resume writes an immutable, checksum-marked
`resume_attempts/attempt_*.json` record locally and, when persistence is enabled,
to that run's persistence directory. If a VM disappears before the next epoch
checkpoint, the next recovery loads these records, identifies another attempt
from the same committed sequence, and records the repeated incomplete epoch.
Corrupt or conflicting committed attempt records stop recovery; partial files are
not accepted. Preparation time recorded before the attempt commit and successful
session recovery overhead remain separate from scientific training time.

The authoritative `resume` runner—not the notebook—executes one fixed synthetic exact update twice from
independent copies of the restored production checkpoint using the M1 device
tolerances. It restores process RNG and never mutates the checkpoint or live
training state. The result is bound to the checkpoint hash, environment,
execution fingerprint, and resume-attempt ID, then checksum-persisted locally
and to Drive before production updates. Analysis rejects cross-device accuracy
eligibility when this evidence is missing or invalid. This verifies deterministic load/optimizer continuation on the
replacement allocation; it does **not** prove numerical equivalence between two
physical T4s or validate scientific accuracy. The two-allocation Colab check is:

1. On allocation A, complete at least one epoch, verify the Drive snapshot and
   its timing marker, and record the environment/session in the manifest.
2. Use Colab **Disconnect and delete runtime** so allocation A is gone. A browser
   disconnect alone leaves Python/GPU state alive; a Python runtime restart may
   retain the VM/GPU and should normally use strict recovery.
3. On allocation B, run environment/source checks, mount the same Drive, select
   the explicit compatible-T4 policy, and run restoration cell 14. It must display
   a different UUID, an allowed transition report, and verified checkpoint/support
   hashes. No dataset or continuation runs in this cell.
4. After separate dataset authorization, set the gate in cell 15 and reconstruct
   or verify the cache. Stop again. After separate full-pilot authorization, set
   the continuation gate in cell 16 and run it once. The runner performs and
   persists the continuation check before production updates. Do not auto-retry. Inspect the manifest for two
   execution sessions, an unknown interruption gap, and the cross-device label.

To simulate disconnect locally without claiming cross-T4 evidence, run a tiny
synthetic job with persistence into a temporary directory, delete only its local
run directory, restore the verified snapshot into a fresh path, and resume under
strict mode. Compatibility unit tests may alter fixture UUID strings, but those
tests prove policy behavior only, not hardware numerical equivalence.


### 8. Read-only analysis

```bash
python -B analyze_phase_d_v4.py validate RUN_DIRECTORY [...]
python -B analyze_phase_d_v4.py summarize \
  --out /content/runs_phase_d_v4/phase_d_v4_pilot_seed0/analysis \
  RUN_DIRECTORY [...]
```

The analysis writes target attainment with explicit censoring, paired tables,
cost classes, deployable-only Pareto points, and accuracy-vs-epoch/time plots.
With one seed, inferential output is explicitly `insufficient_sample`.

## Colab limitations not yet verified

No Colab or Drive command has been executed while preparing these artifacts.
T4 deterministic-operation support, dependency compatibility, real CIFAR-10
throughput, persistence, drift, resource estimates, and scientific accuracy are
unverified. M6 remains incomplete until approved real results are validated and
analyzed.
