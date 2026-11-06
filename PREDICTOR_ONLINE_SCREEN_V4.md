# Two-epoch online predictor screen

This is the separately approved follow-up to the predictor-utility results. It
does not change `IMPLEMENTATION_PLAN.md`, the original notebook, v3/v4 sources,
prior study scripts, checkpoints, or results. No scientific run was launched while
implementing it. The runner is `predictor_online_screen_v4.py`.

## Locked scope

- Start all conditions from the measured seed-0 epoch-5 checkpoint, including
  classifier weights, BatchNorm buffers, SGD momentum and gradient scale.
- Continue through epochs **6 and 7 only**. Keep the original **16-epoch LR clock**;
  shortening the observation window does not shorten the LR schedule.
- Use the existing donor-independent uniform-third mask: each epoch has
  `floor(n_batches/3 + 0.5)` exact slots at `floor(j*n_batches/k)` positions.
- Compare six distinct conditions:
  - `saved`: original predictor and AdamW state, ordinary online fitting.
  - `local32_refit`: fresh local32, one initial 256-update fit on 4,096 examples.
  - `global32_refit`: fresh global32 with the same fitting examples/update budget.
  - `zero_fast`: real zero gradients with momentum/decay; no predictor work.
  - `skip_fast`: no trunk optimizer step on synthetic slots; no predictor work.
  - `exact_trunk_matched_head`: exact trunk gradients on every batch, but head
    updates only on the same uniform-third slots as every other condition.
- Fresh predictors retain their fitted AdamW state and then receive ordinary
  single-update fitting on scheduled exact steps. There is no repeated offline
  refit, new controller, threshold search, LR reset or head compensation.
- All conditions have identical epoch data order, augmentation and head-update
  masks. Historical clipping and frozen downstream BatchNorm are retained.
- Evaluate the original 5,000-example validation split at the shared starting
  state and after each completed epoch. Do not use it for fitting or stopping.
- The initial fit uses the same 4,096 training positions and fixed augmented views
  as the preceding utility study. Results remain exploratory development evidence.

The exact-trunk control is **not conventional full-head exact BP**. The study is
conditional on the historical measured trajectory and its inherited optimizer
state; it is not six new independent classifier seeds or a standalone training
benchmark. No official CIFAR test examples are used.

## Budget and cost accounting

There are six two-epoch continuations: 12 classifier epochs total, not six full
16-epoch runs. At 45,000 training examples and batch size 32 this is 16,884 online
classifier batches. Each epoch has 469 scheduled exact/head-update slots.
The two refits add 512 predictor updates, plus one shared 128-batch frozen-target
construction. Validation, source loading, persistence and recovery also cost work.

The target cache is approximately 1 GiB and is saved on Drive. Additional storage
holds full classifier/optimizer snapshots at epochs 5, 6 and 7 for all six branches
and the small predictor-fitting snapshots. T4 runtime has not been measured.
`--describe` displays the fixed operation budget without accessing CIFAR.

`costs/` records phase wall time; `attempts/` records complete invocation time.
Invocation time includes source/data setup. Phase times are subsets and must not
be added to invocation time. Interrupted work with no completion record remains
explicitly unknown. Failed/repeated completed phases remain in accounting.
Shared target construction is executed once in the study, but its full build cost
must be attributed to each refit method when estimating standalone method cost.
Historical source-training cost is a dependency, not free algorithmic work.
Operation counters describe committed training; operation counts for interrupted
or repeated work are not reconstructed. Recorded phase time still includes known
repeated attempts, and unknown durations stay explicitly unknown.
Recorded times are exploratory; the runner makes no clean speedup claim.

## Colab: files and setup

Keep the successful T4 environment and pinned historical source from the previous
investigation. Mount Drive using the existing setup. Do not run the original
pilot, donor, warm-up, dataset-download or split-generation cells again.

Upload only these new files:

1. `predictor_online_screen_v4.py` into the Colab repository root.
2. `tests/test_phase_d_v4_predictor_online_screen.py` into its `tests/` directory.

Existing dependencies must remain present: `predictor_investigation_v4.py`,
`predictor_utility_v4.py`, original v4 modules/configuration/governing files, the
previous investigation test and M3 test helpers, and `tests/__init__.py`.
Do not edit the historical source to resolve a checkpoint identity mismatch.

Run this setup cell. These are the existing Drive paths; adjust a path only if
the files were deliberately stored elsewhere. Only epoch 5 is needed.

```python
from pathlib import Path
import os, sys, subprocess

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
ROOT = Path('/content/probabilistic backpropagation')
PILOT = Path('/content/drive/MyDrive/phase_d_v4_pilot_seed0')
CHECKPOINT_DIR = PILOT / 'seed0_relative_feedback_measured/checkpoints'
DATA = PILOT / '_dataset_cache'
SPLIT = PILOT / '_study_support/split_indices.npz'
OUT = Path('/content/drive/MyDrive/phase_d_v4_predictor_online_screen_seed0_v1')

assert Path('/content/drive/MyDrive').is_dir(), 'Mount Drive first.'
assert ROOT.is_dir(), f'Repository missing: {ROOT}'
os.chdir(ROOT)
required = [
    ROOT/'predictor_online_screen_v4.py',
    ROOT/'predictor_investigation_v4.py',
    ROOT/'predictor_utility_v4.py',
    ROOT/'tests/__init__.py',
    ROOT/'tests/test_phase_d_v4_predictor_online_screen.py',
    CHECKPOINT_DIR/'checkpoint_000005.pt',
    CHECKPOINT_DIR/'checkpoint_000005.pt.commit.json',
    SPLIT, SPLIT.with_suffix('.npz.manifest.json'),
]
missing = [str(p) for p in required if not p.is_file()]
assert not missing, 'Missing files:\n' + '\n'.join(missing)
assert (DATA/'cifar-10-batches-py').is_dir(), 'Extracted CIFAR cache is missing.'
print('Inputs exist. Output:', OUT)
```

Run the engineering tests and description. They do not train on CIFAR:

```python
subprocess.run([
    sys.executable, '-B', '-m', 'unittest',
    'tests.test_phase_d_v4_predictor_online_screen', '-v',
], cwd=ROOT, check=True)
subprocess.run([
    sys.executable, '-B', 'predictor_online_screen_v4.py', '--describe',
], cwd=ROOT, check=True)
```

After those pass, this is the **single main execution/recovery cell**:

```python
subprocess.run([
    sys.executable, '-u', '-B', 'predictor_online_screen_v4.py',
    '--authorize-online-screen', '--device', 'cuda',
    '--checkpoint-dir', str(CHECKPOINT_DIR),
    '--data-dir', str(DATA), '--split-file', str(SPLIT), '--out', str(OUT),
], cwd=ROOT, check=True)
```

The command checks source/data identities, saves a fixed condition order and
executes all six branches. It prints each completed epoch and validation accuracy.
The original classifier, predictor and optimizer files are read only. No internet
or dataset download is used by this runner.

## Interruption and recovery

- In a still-running runtime, do not start a second copy of the cell.
- After losing a runtime, recreate the same environment, mount Drive and restore
  the scripts. Rerun the setup cell, then the **same main cell with the same OUT**.
- Committed conditions are verified and skipped. Initial predictor fitting resumes
  its last 64-update checkpoint. Online training resumes the last completed epoch;
  an interrupted epoch repeats from its beginning, preserving partial evidence.
- A committed target cache is loaded rather than recomputed. If interruption
  happened before its first successful commit, it must be built again.
- Do not delete caches, checkpoints or markers. Do not change code or dependencies
  midway through this study. Hash corruption or incompatible state causes a stop.
- A replacement T4 UUID is allowed only with the same recorded model, memory,
  software, driver and numerical settings. Before continuing a saved online branch
  on a replacement allocation, the runner checks two isolated synthetic updates.
  This checks repeatability there, not equality between physical GPUs. Recovery
  evidence and its cost are retained; no clean cross-allocation timing is claimed.

## Download results

After the final `Online screen complete` message, export only the small JSON
reports. Keep the full output directory on Drive for recovery. The ZIP name is
unique so an earlier downloaded report is never overwritten.

```python
import uuid
from google.colab import files

ZIP = Path('/content') / f'predictor_online_screen_reports_{uuid.uuid4().hex[:8]}.zip'
subprocess.run([
    sys.executable, '-B', 'predictor_online_screen_v4.py',
    '--out', str(OUT), '--export-reports', str(ZIP),
], cwd=ROOT, check=True)
files.download(str(ZIP))
```

Upload that ZIP to the project root for analysis. `study_report.json` contains
all six paired validation trajectories and operation counts. Phase/invocation
records retain preparation, fitting, evaluation, I/O and interruption evidence.
Only a separately authorized follow-up may extend the screen beyond epoch 7.
