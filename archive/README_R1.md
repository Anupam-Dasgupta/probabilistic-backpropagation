# Robust synthetic-gradient study — R0/R1

This is the **first rigorously controlled rerun**, not the Phase D acceleration experiment.
It implements:

- **R0 correctness audit**: compares batch and per-example intermediate activation gradients, verifies upstream vector-Jacobian propagation, and checks one derivative with finite differences.
- **R1 classifier checkpoints**: trains the 128-wide Fashion-MNIST MLP to 60 epochs with learning-rate reductions; saves checkpoints at epochs 5/10/20/40/60.
- **R1 independent gradient-prediction study**: at each frozen classifier checkpoint, computes exact *unaveraged per-example* intermediate gradients; trains MSE, cosine+MSE, Gaussian-NLL, and class-mean residual predictors; evaluates zero/global/class means.
- **Independent selection and reporting**: fixed 40,000 train / 10,000 validation / 10,000 development examples from the official Fashion-MNIST *training* split. Predictor training duration is chosen **only on validation**; development results are used for analysis. The official 10,000-image test split is not loaded.
- **Multiple classifier seeds**: default 0,1,2; predictor models independently initialized. Checkpoint-by-seed tables and paired differences from class-mean baseline are saved.

## Setup — Windows CMD

```cmd
py -3.14 -m pip install numpy matplotlib torch torchvision
```

If you want GPU acceleration, first ensure CUDA-enabled PyTorch is installed for **Python 3.14**. Follow the selector at https://pytorch.org/get-started/locally/ . Check:

```cmd
nvidia-smi
py -3.14 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
```

The second command must print `True` for `torch.cuda.is_available()` before `--device cuda` will work. Python automatically picks GPU with `--device auto` only if CUDA is actually available.

## Run

Place `gradient_robust_r1.py` in your project folder.

Check the entire pipeline quickly with **synthetic data** (only a technical test):

```cmd
py -3.14 gradient_robust_r1.py --smoke-test --device cpu
```

Quick, reduced **real Fashion-MNIST** pilot (not robust evidence):

```cmd
py -3.14 gradient_robust_r1.py --device cuda --out r1_quick --seeds 0 --class-epochs 10 --checkpoints 5,10 --predictor-epochs 12 --predictor-snapshots 6,12
```

Full intended multi-seed study (computationally substantial):

```cmd
py -3.14 gradient_robust_r1.py --device cuda --deterministic --out r1_study
```

Can split into two processes so you can inspect classifier learning curves first:

```cmd
py -3.14 gradient_robust_r1.py --device cuda --deterministic --out r1_study --mode classifiers
py -3.14 gradient_robust_r1.py --device cuda --deterministic --out r1_study --mode predictors
```

The class checkpoints are reused when every checkpoint exists; completed predictor methods are skipped if their output metric CSV exists. Each output directory is tied to its original configuration. **Change `--out`** for different settings. Do not manually edit the result CSVs and then resume.

> Running on CPU with the full default matrix may take a very long time. If CUDA is unavailable, begin with the smoke test and small pilot instead.

## Output structure

- `config.json` — experiment parameters and Python/PyTorch versions.
- `gradient_audit.json` — exact-gradient sanity tests.
- `split_indices.npz` — fixed sample assignments.
- `seed_*/classifier_e*.pt` — frozen classifier snapshots.
- `seed_*/classifier_history.csv` — training and validation loss/accuracy.
- `seed_*/checkpoint_*/gradient_stats.json` — train-only gradient scaling and norm threshold.
- `seed_*/checkpoint_*/baselines.csv` — zero/global/class-mean metrics.
- `seed_*/checkpoint_*/<method>_metrics.csv` — validation snapshots and **one** validation-selected development measurement per predictor.
- `seed_*/checkpoint_*/<method>_selected.pt` — corresponding predictor weights, training scale, and class means.
- `all_metrics.csv` — pooled per-run values (not an independent-sample CI).
- `summary_dev.csv` — mean and sample SD across independent classifier seeds per checkpoint/method.
- `paired_vs_class_mean.csv` — method minus class mean on the **same classifier seed**.
- `cosine_by_checkpoint.png`, `rmse_by_checkpoint.png` — development performance across classifier checkpoints (requires matplotlib).

Use `--save-predictions` to add per-example diagnostic `.npz` files for the selected models.

## Main metrics and caveats

- `cosine_mean`: mean direction cosine for gradients with norm above the **checkpoint-specific, train-only** norm threshold. Look at `direction_valid_fraction` to ensure near-zero gradients aren't silently excluded.
- `normalized_rmse`: coordinate RMSE divided by the RMS of the **true gradients**, to enable comparisons across training checkpoints.
- `relative_l2_median`: residual L2 divided by clamped true-gradient L2; can be unstable near zero, so do not use by itself.
- `nonpositive_cosine_fraction`: fraction of valid intermediate gradients whose predictions are nonpositively aligned. **This is not parameter-update descent risk.**
- `cosine_correct` / `cosine_incorrect`: conditional cosine means by classifier outcome.

Each classifier checkpoint gets **fresh predictors** trained on a fixed network. This isolates intrinsic checkpoint difficulty. It does **not** yet test predictor staleness/drift (R2), other model architectures (R3), synthetic-gradient updates (R4), or end-to-end speedup. Those stages must be implemented and evaluated separately before making the claim that this method accelerates training.

A full 60-epoch/5-checkpoint/3-seed/4-predictor/50-epoch sweep is fairly expensive. Review the one-seed pilot first to make sure training is stable; then run the locked multi-seed study. Development data should not drive repeated hyperparameter search indefinitely; final claims require a separate held-out official test evaluation.
