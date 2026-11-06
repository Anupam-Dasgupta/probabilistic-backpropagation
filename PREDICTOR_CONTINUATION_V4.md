# Continue all six branches through epoch 16

This is the separately authorized continuation of the completed two-epoch
screen. The old screen, source code, notebook and implementation plan stay
unchanged. All six branches start from their own epoch-7 checkpoint and continue
through epochs 8-16 using the original 16-epoch learning-rate schedule, data
order, uniform-third head mask, normalization and optimizer state. There is no
new initial predictor fit or target-cache construction. Ordinary online predictor
fitting continues on exact scheduled steps. This is exploratory follow-up chosen
after inspecting the short screen, not a convergence or confirmatory claim.

The additional budget is 54 classifier epochs (nine per branch), or 75,978
training batches, plus validation, persistence and possible recovery probes.
Allow roughly 6 GiB of additional free Drive space: the uploaded screen's
checkpoint sizes project to about 5 GiB for the 60 retained boundaries, plus
metadata, temporary writes and any interruption evidence. This is an estimate.
Outputs retain all epoch-6-to-16 rows and the original preparation/cost records.
Source epochs 1-5 remain an inherited training dependency, not free work.
No new accuracy-target thresholds are selected automatically.

## Files and prerequisites

Commit and push these three new files to `main`:

- `predictor_continuation_v4.py`
- `tests/test_phase_d_v4_predictor_continuation.py`
- `PREDICTOR_CONTINUATION_V4.md`

Use the same working T4 dependency environment. Run the existing starting cell
to mount Drive and restore the original screen's pinned repository. Do not run
the old screen's training cell again. The cell below fetches only the two new
Python files from a separately pinned GitHub commit; it does not update any
historical training source. The source screen must be the full Drive directory,
including all six `epoch_000007.pt` files and their commit markers. The downloaded
report-only ZIP cannot resume training.

## Single execution/recovery cell

Run this after the existing environment/starting cell. It checks the files,
runs bounded synthetic tests and starts all six continuations. There are no
dependency installs, dataset downloads or split regeneration.

```python
from pathlib import Path
import hashlib, json, os, subprocess, sys

os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
ROOT = Path('/content/probabilistic backpropagation')
DRIVE = Path('/content/drive/MyDrive')
PILOT = DRIVE / 'phase_d_v4_pilot_seed0'
SOURCE = DRIVE / 'phase_d_v4_predictor_online_screen_seed0_v1'
OUT = DRIVE / 'phase_d_v4_predictor_continuation_seed0_v1'
PIN = DRIVE / 'phase_d_v4_uploads/predictor_continuation_seed0_v1_commit.txt'
assert (ROOT / '.git').is_dir(), 'Run your existing GitHub starting cell first.'
assert DRIVE.is_dir(), 'Mount Drive first.'
os.chdir(ROOT)

def git_text(*arguments):
    """Run a read/fetch Git command and return text; failures stop this cell."""
    return subprocess.check_output(['git', '-C', str(ROOT), *arguments], text=True).strip()

remote = git_text('remote', 'get-url', 'origin').removesuffix('.git')
assert remote == 'https://github.com/Anupam-Dasgupta/probabilistic-backpropagation'
if PIN.is_file():
    commit = PIN.read_text().strip()
    git_text('fetch', 'origin', commit)
else:
    assert not OUT.exists(), 'Existing continuation has no code pin; do not select new code.'
    git_text('fetch', 'origin', 'main')
    commit = git_text('rev-parse', 'FETCH_HEAD')

paths = ['predictor_continuation_v4.py', 'tests/test_phase_d_v4_predictor_continuation.py']
contents = {name: subprocess.check_output(
    ['git', '-C', str(ROOT), 'show', f'{commit}:{name}']) for name in paths}
for name, content in contents.items():
    destination = ROOT / name
    if destination.exists():
        assert destination.read_bytes() == content, f'Local file differs from pinned code: {name}'
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
if not PIN.exists():
    PIN.parent.mkdir(parents=True, exist_ok=True)
    PIN.write_text(commit + '\n')
print('Continuation code pinned:', commit)

# Restore only the exact historical document bytes already required by the pilot.
manifest = json.loads((ROOT / 'configs/phase_d_v4_reference_manifest.json').read_text())
spec_path = ROOT / 'RESEARCH_SPEC.md'
expected = manifest['governing_artifacts']['RESEARCH_SPEC.md']['sha256']
original = spec_path.read_bytes()
if hashlib.sha256(original).hexdigest() != expected:
    restored = original.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')
    assert hashlib.sha256(restored).hexdigest() == expected, 'Research specification content differs.'
    spec_path.write_bytes(restored)

subprocess.run([sys.executable, '-B', '-m', 'unittest',
    'tests.test_phase_d_v4_predictor_continuation', '-v'], cwd=ROOT, check=True)

subprocess.run([
    sys.executable, '-u', '-B', 'predictor_continuation_v4.py',
    '--authorize-continuation', '--device', 'cuda',
    '--screen-dir', str(SOURCE),
    '--checkpoint-dir', str(PILOT / 'seed0_relative_feedback_measured/checkpoints'),
    '--data-dir', str(PILOT / '_dataset_cache'),
    '--split-file', str(PILOT / '_study_support/split_indices.npz'),
    '--out', str(OUT),
], cwd=ROOT, check=True)
```

After a runtime reset, recreate the same environment and run the existing starting
cell, then rerun this same cell. Completed branches are checked and skipped;
unfinished branches resume from their last completed epoch. An interrupted epoch
repeats. A replacement T4 must match the recorded software/numerical settings and
pass the existing isolated continuation probe. No second copy may run concurrently.

Keep the source screen and new output folder on Drive. Do not edit `end_epoch`
in the original runner, change the code pins, or delete checkpoint/commit files.
Timing remains labeled as exploratory and records recovery overhead separately.

## Download the results

After `Continuation complete: all six conditions reached epoch 16.`, run:

```python
import uuid
from google.colab import files

ZIP = Path('/content') / f'predictor_continuation_reports_{uuid.uuid4().hex[:8]}.zip'
subprocess.run([
    sys.executable, '-B', 'predictor_continuation_v4.py',
    '--out', str(OUT), '--export-reports', str(ZIP),
], cwd=ROOT, check=True)
files.download(str(ZIP))
```

Put the downloaded ZIP in the project root. It includes full reported accuracy
and loss trajectories, operation counts, inherited screen evidence and separate
screen/continuation/combined cost accounting. It omits model tensors; retain the
full Drive output for recovery and any separately authorized follow-up.
