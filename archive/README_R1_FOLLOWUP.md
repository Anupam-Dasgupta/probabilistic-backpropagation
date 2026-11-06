# R1+ Robust Gradient Prediction Follow-Up

This is the **next diagnostic experiment after R1**, not an implementation of faster backpropagation. It works with the original classifier checkpoints and **does not retrain the Fashion-MNIST classifiers**.

## Files

Keep these together in your project directory:

- `gradient_r1_followup.py` — this new study
- `gradient_robust_r1.py` — original R1 code, imported by this study
- `r1_study/` — your **extracted** original R1 results, containing `config.json`, `split_indices.npz`, and `seed_0/`, `seed_1/`, `seed_2/` with `classifier_e10.pt`, etc.

If your original folder is named `r1_study - Copy`, use `--r1-root "r1_study - Copy"` instead of `--r1-root r1_study`. A ZIP is not sufficient: extract it first.

## Windows CMD commands

Check CUDA in the Python 3.13 installation you already have:

```cmd
py -3.13 -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Smoke test (constructs a tiny synthetic classifier and tests the entire pipeline; **not scientific evidence**):

```cmd
py -3.13 gradient_r1_followup.py --smoke-test --device cpu --out smoke_r1_followup
```

Full run on your RTX 3060 GPU (3 seeds, classifier checkpoints 10/20/40/60; **two** gradient predictors trained up to 120 epochs at each checkpoint):

```cmd
py -3.13 gradient_r1_followup.py --r1-root r1_study --device cuda --deterministic --out r1_followup
```

Or run the smaller pilot first in a **separate** output directory:

```cmd
py -3.13 gradient_r1_followup.py --r1-root r1_study --device cuda --seeds 0 --checkpoints 10,20 --pred-epochs 40 --pred-snapshots 10,20,30,40 --out r1_followup_pilot
```

To use 150 predictor epochs instead of 120:

```cmd
py -3.13 gradient_r1_followup.py --r1-root r1_study --device cuda --pred-epochs 150 --pred-snapshots 25,50,75,100,125,150 --out r1_followup_150
```

Do not use the same output directory for different configurations. You may **rerun the exact same command** after interruption. Completed model fits are reused, and predictor training resumes from its last **snapshot** checkpoint (not necessarily the exact interrupted minibatch).

Required packages in Python 3.13: `torch`, `torchvision`, `numpy`. `matplotlib` is optional for plots. No separate CUDA toolkit is needed for an appropriate CUDA-enabled PyTorch build.

## What it does

- Uses saved classifier checkpoints only; never re-trains the original 60-epoch classifiers.
- Reconstructs the **original** 40k train / 10k validation / 10k development split, then internally divides the original training split into **32k for predictor/magnitude training** and **8k disjoint examples for risk-model training**. It verifies the original `split_indices.npz`.
- Trains **cosine+MSE** and **Gaussian NLL** gradient predictors separately, evaluating checkpoints at training epochs 20, 40, 60, 80, 100, 120. Selects by validation cosine or validation normalized RMSE respectively.
- Tests a raw cosine predictor, training-fitted **single global magnitude factor**, and a small learned **gradient-norm regressor** trained on the 32k predictor-training samples, with its own validation selection. Also evaluates Gaussian and class-mean baselines.
- Evaluates actual **first-layer parameter-gradient proxies** using the ReLU Jacobian and exact input norms (no need to materialize 128x784 tensors per sample).
- Evaluates non-mutating **virtual single-step SGD updates** on development minibatches. This measures immediate local loss change; it is **not** a test of multi-step training convergence or a real optimizer schedule.
- Fits an independent risk model on 8k samples, targeting a **fixed** proxy event: cosine < 0.8 on valid gradients, OR relative L2 error > 2. Uses only features available from the forward pass and synthetic predictor; true gradients never enter its prediction features. Selects risk-model epoch by validation Brier score.
- Evaluates all source predictors and risk estimators **without retraining** on every *later* classifier checkpoint for the same seed, measuring staleness.
- Compares exact top-k selection budgets 10%, 20%, 30% against disagreement, Gaussian sigma, low predicted norm, classifier confidence, and random selection.
- Reports means and sample standard deviations across independent seeds, plus plots if matplotlib is available.

## Outputs

- `metrics.csv` — per-seed, per-source/target checkpoint gradient metrics for each predictor and magnitude correction.
- `summary_by_checkpoint.csv` — cross-seed means and SDs for cosine, normalized RMSE, parameter-gradient proxy error, and near-zero rates.
- `risk_ranking.csv` and `risk_summary.csv` — gradient-error risk AUROC, prevalence, and drift summaries.
- `risk_selection.csv` — unsafe recall, risk remaining in the synthetic group, and observed exact fractions at **matched** selection budgets.
- `virtual_sgd_probe.csv` — immediate loss before/after hypothetical **trunk-only SGD** updates (not real training acceleration).
- `cosine_by_checkpoint.png`, `normalized_rmse_by_checkpoint.png`, `weight_gradient_rmse_by_checkpoint.png`, `risk_auc_under_drift.png` (optional).
- Per-seed, per-checkpoint `.pt` predictor/risk/magnitude models, selected epochs, training curves, scalar correction metadata.

## Limits and interpretation

1. **Not a fully locked confirmatory study:** your original R1 development set has already been inspected. The official Fashion-MNIST test split remains untouched.
2. **No actual synthetic-gradient updates are performed.** This script does not establish training speedup, convergence, or reliability of online gradient selection.
3. **Model capacity is still the same small MLP.** Generalization to a CNN or ResNet must be tested separately.
4. The risk event is a **proxy** (direction/magnitude deviation), not a directly harmful optimizer update. AUC may be undefined if all examples have the same label (possible in the smoke test).
5. A classifier frozen at a later checkpoint is not the same as a fresh gradient predictor fitted at that checkpoint; this study measures **both** fresh fits and frozen-predictor drift.
6. The follow-up's 32k/8k internal subdivision means results need not match old R1 predictor metrics trained on all 40k training examples.
7. `--deterministic` may reduce GPU throughput. Use matched settings when comparing methods.

The next stage after interpreting the results is a real minibatch/block-level synthetic-gradient training experiment with measured GPU time and matched compute budgets.
