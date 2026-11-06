# Phase D — Error-feedback synthetic-gradient switching (software pilot)

This script performs **actual** synthetic-gradient updates. It does not just rank
or evaluate saved gradient vectors. It deliberately omits backward computation
through a small downstream MLP head on synthetic batches, and substitutes a
predicted gradient at the upstream activation boundary.

## Install / environment

Windows CMD, Python 3.13 with your CUDA-enabled PyTorch installation:

```cmd
py -3.13 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Install any **missing** additional packages in that interpreter:

```cmd
py -3.13 -m pip install numpy matplotlib torchvision
```

Do **not** replace your working CUDA-enabled `torch` just to install these.

## Run

Put `gradient_phase_d.py` in your existing `probabilistic backpropagation` project
folder; it is standalone and **does not require the R1/R1+ checkpoint folders**.

Smoke test on synthetic random tensors (no scientific meaning):

```cmd
py -3.13 gradient_phase_d.py --smoke-test --device cpu --out phase_d_smoke
```

Full exploratory run (3 seeds, 20 total training epochs per regime):

```cmd
py -3.13 gradient_phase_d.py --device cuda --deterministic --seeds 0 1 2 --out phase_d_study
```

If deterministic CUDA raises a cuBLAS error, try a new CMD window and:

```cmd
set CUBLAS_WORKSPACE_CONFIG=:4096:8
py -3.13 gradient_phase_d.py --device cuda --deterministic --seeds 0 1 2 --out phase_d_study
```

Single-seed faster *exploratory* run:

```cmd
py -3.13 gradient_phase_d.py --device cuda --seeds 0 --out phase_d_pilot
```

The script caches the common predictor-trained warmup at `seed_<n>_warmup.pt`.
Its cache checks key architecture/training settings. Use `--no-cache` to force
retraining when necessary. Use a **new `--out`** for a materially different
experiment to avoid mixing results.

## The four methods

1. `exact`: ordinary full backpropagation on every minibatch; it **does not**
   train or call the predictor after the common training checkpoint.
2. `scheduled`: predetermined ratio of synthetic to exact minibatches, roughly
   4:1, 3:1, 2:1, then 1:1 over successive quarters of post-warmup training.
3. `feedback`: calculate relative gradient-estimation RMS error on exact
   minibatches, track its exponential moving average, and allow 4/2/0
   subsequent synthetic minibatches for low/medium/high measured error.
4. `matched_periodic`: automatically run *after* feedback with the **same total
   number of post-warmup exact minibatches**, distributed evenly throughout
   training. This is the key compute-budget-control baseline.

All runs start from **identical** classifier weights, synthetic predictor
weights, optimizer states, and minibatch orders for each random seed. A normal
BP-only warmup is timed independently so the exact baseline is not charged for
predictor preparation it doesn't use. The classifier states resulting from
ordinary vs predictor-trained warmup are compared numerically.

`--no-matched` disables the fourth method. Use `--modes exact scheduled feedback`
to run only a subset, but note that the matched periodic baseline requires
feedback to run first.

## What a real synthetic batch does

- Compute upstream activation `h = trunk(x)` with autograd enabled.
- Compute a small head forward pass for logging CE, **under `torch.no_grad()`**.
- Compute predicted per-example `dCE/dh` under `torch.no_grad()`.
- Run `h.backward(estimated_gradient / batch_size)`, updating **only trunk**
  parameters; **no head backward or head optimizer step** is performed.

In an exact minibatch, standard mean-CE backprop trains trunk and head. The
predictor is updated from true per-example gradients in *scheduled*, *feedback*,
and *matched_periodic* modes. In exact baseline, the predictor is disabled.

The code independently checks that injecting exact `dL/dh` reproduces trunk
parameter gradients and that injected backward does not compute head gradients.

## Error feedback formula

Let `g` be the exact **per-example** activation gradients and `ghat` the
predicted per-example activation gradients. The scalar error is

```
e = RMS(g - ghat) / max(RMS(g), 0.05 * initial_gradient_scale)
s <- (1 - alpha) * s + alpha * e
```

Defaults: `alpha=0.2`, error below `0.7` -> 4 synthetic minibatches; below
`1.1` -> 2; at least `1.1` -> 0. These are **exploratory**, not calibrated
or optimized values. For controlled tuning use the training/validation data,
then freeze the controller thresholds before any final held-out evaluation.

## Outputs

In the output directory:

- `config.json` — full settings, GPU info, gradient audit.
- `epochs.csv` — per-epoch validation accuracy, CE, exact/synthetic counts,
  measured errors, cumulative elapsed training+validation time.
- `summary.csv` — one row per seed/method, final accuracy, total exact BP
  fraction, training time, warmup-adjusted costs, time to target accuracy.
- `summary_aggregate.csv` — mean/SD across completed seeds by method.
- `paired_vs_baselines.csv` — within-seed feedback minus exact/scheduled/
  matched periodic differences.
- `accuracy.png`, `time_by_epoch.png` — exploratory summary plots.
- `seed_<n>_warmup.pt`, `seed_<n>_<method>_final.pt` — checkpoints.

`--target-accuracy 0.88` changes the accuracy target for time-to-target reporting.
If a regime never reaches it, that entry is blank (do not interpret it as 0).

## Essential limitations

- **This is not the official ADA-GP implementation.** Scheduled switches are a
  simplified software analogue of its published training strategy.
- This first study uses Fashion-MNIST and a small MLP. The saved computation
  through its head is very small; on GPU, extra predictor and Python overhead
  may outweigh savings. A *negative* speed result is scientifically valid.
- On synthetic steps, the downstream head is held fixed. Fewer head updates can
  affect convergence independently of gradient estimation error; quantify it.
- The 50,000/10,000 split is development train/validation. The official test
  dataset is **not used**. Real benchmark claims need a locked final test and
  stronger CNN/ResNet architecture comparisons, 5+ seeds, and matched budgets.
- Validation is performed once per epoch; reported time-to-target accuracy
  has epoch-level resolution. `total_train_seconds` excludes validation; the
  wall metric includes validation, predictor training and each method's proper
  warmup, but excludes initial data download and file I/O.
- The warmup itself is a common research setup, not an external free pretrained
  predictor. Its cost is included in the synthetic methods' totals.

**Decision criterion:** At matched exact BP counts, does feedback give higher
validation accuracy or better time-to-target than periodic switching? Also,
does it actually beat full BP in training cost for a predeclared accuracy target?
Don't declare a speedup based only on skipped backward passes.
