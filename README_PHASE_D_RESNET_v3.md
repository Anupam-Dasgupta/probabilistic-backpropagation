# Phase D — CIFAR-10 ResNet error-feedback switching

**What this is:** a *standalone* controlled research prototype for measuring whether feedback-controlled synthetic gradients improve training over periodic synthetic-gradient switching. **It is not an exact reproduction of ADA-GP.** It does not use any Fashion-MNIST R1/R1+ or Phase D v1/v2 files.

## Setup (Windows CMD, your CUDA-enabled Python 3.13)

Put `gradient_phase_d_resnet.py` in your project folder. You do **not** need your old experiment folders. Use the existing CUDA-enabled Python installation. Verify dependencies first:

```cmd
py -3.13 -c "import torch, torchvision, numpy; print(torch.__version__, torch.cuda.is_available())"
```

Expected CUDA flag: `True`. To install missing `numpy` or `matplotlib` without unnecessarily replacing your CUDA-enabled Torch:

```cmd
py -3.13 -m pip install numpy matplotlib
```

If `torchvision` is missing, install a version matching your currently installed Torch from https://pytorch.org/get-started/locally/ . Do not blindly replace CUDA-enabled Torch with a CPU-only build.

## Smoke test (fast, synthetic data)

```cmd
py -3.13 gradient_phase_d_resnet.py --smoke-test --mode full --device cpu --out d_resnet_smoke
```

Smoke testing checks exact gradient injection, controller thresholds, periodic-mask counts, and executes all six training regimes on fake images. **Smoke-test accuracy and timing are scientifically meaningless.**

## The go/no-go GPU cost test (run first)

```cmd
py -3.13 gradient_phase_d_resnet.py --device cuda --mode preflight --seeds 0 --out cifar_d_resnet
```

This downloads CIFAR-10 if needed, trains the classifier and predictor together for a common warm-up, and creates a separate exact-BP-only warm-up timing control. It then profiles on-GPU synchronized full training steps with **identical warm-up weights**:

- `exact_plain`: full ResNet exact gradients, no predictor updates;
- `exact_plus_predictor`: exact gradients plus predictor training;
- `synthetic`: trunk gradient injected from predictor, no head forward/backward.

Key output: `cifar_d_resnet/feasibility.csv` (also `seed_0_feasibility.csv`). Look at `estimated_speedup_ratio` at 20%, 35%, and 50% exact-gradient budgets. Also see the predictor training overhead, savings per synthetic batch, modeled break-even exact fraction, and `seed_0_profile.csv` with GPU memory where available.

**Important:** The preflight is a *cost-only* feasibility model. It excludes predictor warm-up and calibration, data loading, validation, drift, and differences in convergence. It does not establish an end-to-end speedup. If all modeled speedups are <= 1, do **not** run the expensive full study without first modifying the architecture or predictor.

With an RTX 3060 and **6 GB VRAM**, the default batch size is 32. If memory fails, rerun with `--batch-size 16` and a **different output folder** so caching cannot mix incompatible configurations. Use `nvidia-smi -l 1` in another CMD window to monitor GPU memory.

## Full exploratory study

Once preflight indicates the possibility of savings:

```cmd
py -3.13 gradient_phase_d_resnet.py --device cuda --mode full --deterministic --seeds 0 1 2 --out cifar_d_resnet
```

**Use the same options for preflight and full** (notably `--deterministic`, `--batch-size`, `--width`, `--blocks`, `--epochs`), otherwise the warm-up signature changes and the warm-up must be recomputed. For exact checkpoint reuse, use this same deterministic flag **in the preflight command too**, if intended:

```cmd
py -3.13 gradient_phase_d_resnet.py --device cuda --mode preflight --deterministic --seeds 0 --out cifar_d_resnet
py -3.13 gradient_phase_d_resnet.py --device cuda --mode full --deterministic --seeds 0 1 2 --out cifar_d_resnet
```

Results are cached **per finished method and seed**; re-running the same command reuses them. Use a new output folder when changing experimental parameters. `--overwrite` explicitly recalculates and replaces cached runs.

The default architecture is **CIFAR-style ResNet-18** (`--width 64 --blocks 2`, standard two residual blocks per stage, with CIFAR 3×3 stem). The split is **after layer 2** (activation: batch × 128 × 16 × 16). The predictor is a small class-conditioned convolutional network. To do a cheaper exploratory *smaller ResNet*, use `--width 32` with a new output folder. This is *not* a standard-width ResNet-18.

The default study uses a 45k/5k split of CIFAR-10's training data. It **does not access the official 10k-image test set**. Data augmentation and minibatch order are matched across methods via controlled RNG. The classifier receives identical initialization and warm-up optimizer states. Each seed is independent. The full training default is 4 warm-up epochs + 12 postwarm epochs, with a cosine learning-rate schedule. Consider a second, longer run with validation-selected hyperparameters if initial results are promising; do not interpret a short run as full convergence.

## Comparisons (six methods)

| Method | Exact trunk backward? | Predictor update? | Head update? |
|---|---|---|---|
| `exact` | Every batch | Never | Every batch |
| `scheduled` | Scheduled batches | On exact batches | On exact batches |
| `relative_feedback` | Chosen from rolling *relative* error quantiles | On exact batches | On exact batches |
| `absolute_feedback` | Chosen from *fixed calibration* absolute normalized-error thresholds | On exact batches | On exact batches |
| `matched_periodic` | Same **total count** of exact batches as absolute-feedback, spaced evenly | On exact batches | On exact batches |
| `exact_head_mask` | **Every** batch | Never | Only at absolute-feedback's exact positions |

All synthetic batches really **skip the head forward and backward computation**. The classifier trunk still receives its synthetic activation-gradient VJP. Sparse head updates are controlled separately by `exact_head_mask`.

### Controller details

Calibration is done at the end of warm-up using training minibatches without changing the classifier/predictor weights. On each exact batch, an observed gradient error is measured **before** fitting the predictor to that batch.

- **Rolling-relative controller:** compares relative error to quantiles of previous exact-batch errors. This can track local ranking but mask absolute deterioration.
- **Fixed-absolute controller:** compares an EMA of absolute normalized RMSE with the 33rd/67th calibration quantiles, never recomputing them after warm-up. More global deterioration can trigger extra exact backpropagation. Quotas are 0, 2, or 4 synthetic steps. The current batch used to calculate error is always exact.

Both controllers are heuristics; defaults are not theoretically optimal. `seed_*_absolute_feedback_controller.csv` logs actual decisions and thresholds. If fewer than two quotas appear in substantive runs, the controller did not meaningfully adapt.

### BatchNorm caveat

**The head's BatchNorm running statistics are frozen after warm-up for every method**, including exact BP. Without this, the synthetic path (which skips head forward) would update BN running statistics less often, introducing a major confound. Affine BN weights can still train on exact batches. The trunk BatchNorm layers continue to update normally. This controlled setting is not directly equivalent to a conventional full ResNet-18 training recipe. A subsequent experiment should test alternative normalization strategies.

## Outputs

- `summary.csv`, `summary_aggregate.csv`: final/mean accuracy, exact fraction, wall time (warm-up + calibration + training + validation), time-to-target when reached.
- `paired_absolute_vs_baselines.csv`: within-seed accuracy and runtime differences.
- `epochs.csv`: validation accuracy and cumulative wall time over epochs.
- `accuracy_vs_epoch.png`, `accuracy_vs_walltime.png`: optional plots if matplotlib is installed.
- `seed_*_absolute_feedback_controller.csv`, `seed_*_relative_feedback_controller.csv`: all controller decisions.
- `seed_*_feasibility.csv`, `seed_*_profile.csv`: synchronized preflight GPU costs and modeled break-even.
- `seed_*_*_result.pt`: finished method checkpoints for resuming without repeating completed regimes.

## Research interpretation

A useful incremental research contribution would require the absolute-feedback controller to **beat matched-periodic scheduling at matched exact-gradient budget** *and* improve net accuracy-versus-wall-clock versus strongly tuned baselines. Include all predictor warm-up, calibration, predictor updates, and evaluation in accounting. The head-mask exact control diagnoses whether scarce downstream updates are the main problem.

Even if the GPU cost preflight looks good, long-term convergence can fail. If it does, report that rather than treating skipped backward steps as a speedup. Initial 3-seed results are exploratory, not publication-strength proof; repeat with more seeds and independent architecture/dataset validation if promising.
