# Phase D v2 — Calibrated feedback switching + head-update controls

This is a **new version of Phase D**: it actually trains with synthetic gradients. It is a controlled Fashion-MNIST/software pilot, **not an exact ADA-GP implementation** and not evidence of a training speedup until the results show one.

## Files

- `gradient_phase_d_v2.py`: new experiment and reports.
- `gradient_phase_d.py`: the **original Phase D script**, included in the ZIP because v2 imports its model, gradient-injection audit, training step implementation, loaders, and warm-up.
- `requirements_phase_d_v2.txt`: Python dependencies.

Keep **both Python files in the same folder**.

## Run in Windows CMD using your CUDA-enabled Python 3.13

```cmd
py -3.13 -m pip install numpy matplotlib torchvision
py -3.13 -c "import torch; print(torch.cuda.is_available(), torch.__version__)"
py -3.13 gradient_phase_d_v2.py --smoke-test --device cpu --out phase_d_v2_smoke
```

If `torch.cuda.is_available()` is `True`, run the study:

```cmd
py -3.13 gradient_phase_d_v2.py --device cuda --deterministic --seeds 0 1 2 --out phase_d_v2
```

**Save substantial repeated warm-up work** by reusing the original `phase_d_study` folder, if it still contains `seed_0_warmup.pt`, `seed_1_warmup.pt`, and `seed_2_warmup.pt`:

```cmd
py -3.13 gradient_phase_d_v2.py --device cuda --deterministic --warmup-root phase_d_study --seeds 0 1 2 --out phase_d_v2
```

If your original Phase D folder is named differently, use that path after `--warmup-root`. The script checks architecture, optimizer, warm-up, and deterministic settings for compatibility, and refuses incompatible caches.

No R1/R1+ checkpoints are required. This is a continuation of Phase D, not a new R1+ training run.

## What's changed from Phase D v1

1. **Calibrated error controller:** Measures the relative RMS error of predicted activation gradients on a separate set of training minibatches at the frozen warm-up checkpoint. The calibration does **not update** the classifier or predictor and its wall time is included for feedback.
2. **Actual feedback decisions:** At each exact-backprop batch, compare the just-observed gradient error against **quantiles of previous exact-gradient errors**, kept in a rolling window. Choose a subsequent quota of **0, 2, or 4 synthetic batches**. After recording the decision, add the new error to the history. Also apply a conservative guard if the rolling median or current error is much larger than its initial calibration value.
3. **Matched periodic baseline:** Choose a regular schedule with exactly the **same realized number of exact-backprop batches** as feedback. This is an offline experimental control; it uses feedback's realized budget, not its instantaneous information.
4. **Head-update controls:** Each minibatch gets **exact full backpropagation through the head and trunk**, but the head optimizer steps only when it would have stepped under (a) feedback or (b) matched periodic selection. The trunk updates on **every minibatch**. This isolates how much accuracy loss comes from fewer head updates, as distinct from incorrect synthetic trunk gradients.
5. **GPU timing:** The primary measurements synchronize CUDA only at epoch boundaries, measuring real training runs, including data loading and optimizer overhead. A separate microbenchmark divides time among trunk forward/backward, head forward/backward, predictor inference and training, and optimizer steps. It deliberately synchronizes for diagnostic component measurements; it does **not** run inside or inflate full training-time measurements.

## Methods

| Method | Trunk gradient every batch | Head optimizer step | Predictor |
|---|---|---|---|
| `exact` | Exact | Every batch | None during training |
| `scheduled` | Mix exact/synthetic by a predetermined ratio | Exact batches only | Updated on exact batches |
| `feedback` | Mix exact/synthetic by rolling prediction error | Exact batches only | Updated on exact batches |
| `matched_periodic` | Mix exact/synthetic, matched exact count | Exact batches only | Updated on exact batches |
| `exact_head_feedback_mask` | **Exact** on every batch | Only at feedback's exact-batch positions | Not updated |
| `exact_head_periodic_mask` | **Exact** on every batch | Only at periodic's exact-batch positions | Not updated |

All methods use the same post-warm-up initialization, minibatch order, and model optimizer states. The full exact-BP control is charged a **predictor-free warm-up cost**; synthetic methods are charged the warm-up in which the predictor is trained.

## Main output files

- `summary.csv`: per-seed results, accuracy, exact-gradient fraction, head-update fraction, complete training/validation time (including warm-up and feedback calibration).
- `summary_aggregate.csv`: means and standard deviations across seeds.
- `paired_vs_feedback.csv`: per-seed paired differences against feedback.
- `epochs.csv`, `accuracy.png`, `time.png`, `head_updates.png`: complete trajectories.
- `seed_0_controller_events.csv` etc.: error, rolling thresholds, percentile, chosen quota, and emergency guard for **every actual feedback decision**.
- `seed_0_masks.npz` etc.: exact-batch and matched-periodic masks used to reproduce the head-update controls.
- `seed_0_calibration.json` etc.: initial error distribution and calibration time.
- `profile_summary.csv` and `profile_components.csv`: separate diagnostic microbenchmark; **do not interpret component sums as actual training times** because the microbenchmark splits autograd and synchronizes after every component.
- `seed_*_*_final.pt`: trained final checkpoints.

## What constitutes evidence?

1. **Controller changed states** (at least two of 0,2,4 quotas observed in each substantive run). If not, it did not test adaptive scheduling; check the event CSV.
2. **Same compute budget:** Feedback should beat `matched_periodic` at matched exact-gradient count, preferably consistently across independent seeds.
3. **Head confound:** `exact_head_feedback_mask` tells us how much degradation was caused solely by sparse head updates. If this control matches exact BP but feedback fails, blame inaccurate synthetic trunk gradients more strongly.
4. **Net speed:** Feedback should improve actual time-to-accuracy or accuracy-versus-wall-clock, **including** warm-up and calibration. Reducing the number of head backward passes alone is not a speedup.
5. **Robustness:** The original 3 seeds and small Fashion-MNIST MLP are an exploratory stage; positive results require more seeds and more demanding networks. The Fashion-MNIST validation split is development data, not an untouched final test set.

## Optional parameters

- `--epochs 24`: run 24 post-warm-up epochs (default 16).
- `--calibration-batches 96`: number of exact gradient probes at the warm-up checkpoint (default 64).
- `--error-window 128`: history size for quantile adaptation (default 96).
- `--drift-guard-ratio 2.0`: trigger conservative exact BP when rolling error median doubles relative to initial calibration.
- `--emergency-ratio 3.0`: force exact BP when an observed error exceeds three times the initial median.
- `--skip-profile`: omit separate profiling to reduce diagnostic overhead (does **not** affect reported training times).
- `--profile-batches 16`: diagnostic batch count per method.

**Avoid tuning controller thresholds on the same validation results used to make a research claim.** For this pass treat all the defaults as pre-specified, then decide on future experiments using development data and reserve a final confirmation set.
