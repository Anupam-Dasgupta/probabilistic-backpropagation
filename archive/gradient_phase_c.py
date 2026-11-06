"""Phase C: Can inexpensive features predict directional synthetic-gradient failures?

REQUIRES gradient_phase_b.py in the same directory. Runs a fresh, leakage-controlled
experiment rather than reusing Phase B checkpoints (which saw all 50k predictor images).

Splits Fashion-MNIST TRAIN (60k) into four disjoint parts:
  40k classifier/gradient-predictor fitting | 8k directional-risk fitting |
   6k calibration | 6k untouched final test.
No use of Fashion-MNIST test set. Train splitting is deterministic.

Trains a frozen-snapshot cosine+MSE gradient predictor (primary), an isotropic
Gaussian predictor (a diagnostic baseline), and a small binary risk classifier
whose target is the worst 20% of cosine alignments on risk-training examples.
The risk model sees *only cheap forward-pass features*, never exact gradients.

Evaluates directional-risk detection, low-gradient-norm confounding, budgeted
exact-gradient selection (observational only), calibration and robustness to
classifier parameter drift (predictors and risk estimator remain frozen).

IMPORTANT: This computes exact test gradients for OFFLINE scoring. It performs
NO synthetic optimizer updates, demonstrates NO speedup, and does NOT provide
statistical guarantees. Selection fractions are diagnostics, not saved compute.

Use:
  py -3.14 gradient_phase_c.py --device cuda
  py -3.14 gradient_phase_c.py --device cpu --smoke-test
  py -3.14 gradient_phase_c.py --device cuda --pred-epochs 20 --risk-epochs 20
Requires torch, torchvision, numpy; matplotlib is optional.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset

try:
    import gradient_phase_b as b
except ImportError as exc:
    raise SystemExit("Put gradient_phase_b.py in the same folder as gradient_phase_c.py") from exc


class RiskNet(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 64), nn.ReLU(),
                                 nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase C: directional-risk estimation and drift")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="./phase_c_output")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--class-epochs", type=int, default=6)
    p.add_argument("--pred-epochs", type=int, default=12)
    p.add_argument("--risk-epochs", type=int, default=15)
    p.add_argument("--drift-epochs", type=int, default=2)
    p.add_argument("--model-lr", type=float, default=1e-3)
    p.add_argument("--predictor-lr", type=float, default=1e-3)
    p.add_argument("--risk-lr", type=float, default=1e-3)
    p.add_argument("--drift-lr", type=float, default=2e-4)
    p.add_argument("--cosine-weight", type=float, default=0.1)
    p.add_argument("--bad-quantile", type=float, default=0.20,
                   help="Risk threshold = bottom fraction of training cosine scores")
    p.add_argument("--direction-floor-quantile", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke-test", action="store_true")
    p.add_argument("--skip-plots", action="store_true")
    args = p.parse_args()
    for key in ("batch_size", "hidden", "class_epochs", "pred_epochs", "risk_epochs"):
        if getattr(args, key) <= 0:
            p.error(f"--{key.replace('_', '-')} must be positive")
    if args.drift_epochs < 0 or not 0 < args.bad_quantile < 1 or not 0 <= args.direction_floor_quantile < 1:
        p.error("Invalid drift epochs, quantile, or direction floor quantile")
    if min(args.model_lr, args.predictor_lr, args.risk_lr, args.drift_lr) <= 0:
        p.error("Learning rates must be positive")
    if args.smoke_test:
        args.class_epochs = min(args.class_epochs, 1)
        args.pred_epochs = min(args.pred_epochs, 2)
        args.risk_epochs = min(args.risk_epochs, 3)
        args.drift_epochs = min(args.drift_epochs, 1)
    return args


def create_splits(args: argparse.Namespace, device: torch.device):
    if args.smoke_test:
        rng = torch.Generator().manual_seed(args.seed)
        ds = TensorDataset(torch.randn(768, 1, 28, 28, generator=rng),
                           torch.randint(0, 10, (768,), generator=rng))
    else:
        from torchvision import datasets, transforms
        ds = datasets.FashionMNIST(
            root=args.data_dir, train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,))]))
    n = len(ds)
    n_model, n_risk, n_cal = int(n * 2 / 3), int(n * 2 / 15), int(n * 1 / 10)
    permutation = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed)).tolist()
    subsets = {
        "model": Subset(ds, permutation[:n_model]),
        "risk": Subset(ds, permutation[n_model:n_model+n_risk]),
        "cal": Subset(ds, permutation[n_model+n_risk:n_model+n_risk+n_cal]),
        "test": Subset(ds, permutation[n_model+n_risk+n_cal:]),
    }
    print("Independent splits:", {k: len(v) for k, v in subsets.items()}, flush=True)
    pin = device.type == "cuda"
    def loader(name: str, shuffle: bool = False):
        return DataLoader(subsets[name], batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=pin,
                          generator=torch.Generator().manual_seed(args.seed + 7))
    return subsets, loader


def collect(model, loader, device, label):
    return b.collect_frozen_gradients(model, loader, device, label)[:3]


def mean_gradients(y: torch.Tensor, g: torch.Tensor, num_classes: int = 10):
    sums = torch.zeros(num_classes, g.shape[1], dtype=g.dtype)
    counts = torch.bincount(y, minlength=num_classes)
    sums.index_add_(0, y, g)
    means = sums / counts.clamp(min=1).unsqueeze(-1)
    means[counts == 0] = g.mean(dim=0)
    return means


@torch.inference_mode()
def cheap_features(classifier, h_cpu, y_cpu, primary, means, device, batch_size):
    """Everything used here is available without backward propagation.

    Returned features use h, label, head logits, predicted gradient and class
    mean. Class-mean gradient is fitted ONLY on the model-training split.
    """
    all_logits = []
    for (h,) in DataLoader(TensorDataset(h_cpu), batch_size=batch_size):
        all_logits.append(classifier.head(h.to(device)).cpu().numpy())
    logits = np.concatenate(all_logits).astype(np.float32)
    y = y_cpu.numpy()
    logits_shift = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits_shift)
    probs /= probs.sum(axis=1, keepdims=True)
    p_true = probs[np.arange(len(y)), y]
    p_max = probs.max(axis=1)
    entropy = -(probs * np.log(np.maximum(probs, 1e-12))).sum(axis=1)
    m = means[y].astype(np.float32)
    primary_norm = np.linalg.norm(primary, axis=1)
    means_norm = np.linalg.norm(m, axis=1)
    similarity = np.sum(primary * m, axis=1) / np.maximum(primary_norm * means_norm, 1e-10)
    scalars = np.column_stack([
        np.log1p(primary_norm), np.log1p(means_norm),
        similarity, 1 - similarity, -np.log(np.maximum(p_true, 1e-12)),
        1 - p_true, p_max, entropy,
        logits.max(axis=1) - logits.min(axis=1),
    ]).astype(np.float32)
    onehot = np.eye(10, dtype=np.float32)[y]
    features = np.column_stack([h_cpu.numpy(), onehot, scalars]).astype(np.float32)
    # Baseline scores (higher = more suspected directional error).
    baselines = {
        "small_pred_norm": -primary_norm,
        "large_pred_norm": primary_norm,
        "low_confidence": 1 - p_true,
        "high_entropy": entropy,
        "class_mean_disagreement": 1 - similarity,
    }
    return features, baselines


def per_example_cosine(true: np.ndarray, pred: np.ndarray):
    true_norm, pred_norm = np.linalg.norm(true, axis=1), np.linalg.norm(pred, axis=1)
    cosine = np.sum(true * pred, axis=1) / np.maximum(true_norm * pred_norm, 1e-12)
    return np.clip(cosine, -1.0, 1.0), true_norm


def normalize(train: np.ndarray, others: list[np.ndarray]):
    mean, std = train.mean(axis=0), train.std(axis=0)
    std = np.maximum(std, 1e-3)
    return [(x - mean) / std for x in [train, *others]], mean, std


def train_risk_network(train_features, train_labels, device, args):
    model = RiskNet(train_features.shape[1]).to(device)
    x = torch.from_numpy(train_features.astype(np.float32))
    y = torch.from_numpy(train_labels.astype(np.float32))
    loader = DataLoader(TensorDataset(x, y), batch_size=args.batch_size, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed + 201))
    pos = max(1, int(y.sum().item()))
    neg = max(1, len(y) - pos)
    pos_weight = torch.tensor(neg / pos, device=device, dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=args.risk_lr)
    for epoch in range(1, args.risk_epochs+1):
        model.train()
        total = 0.
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = F.binary_cross_entropy_with_logits(logits, yb, pos_weight=pos_weight)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(yb)
        if epoch in {1, args.risk_epochs} or epoch % max(1, args.risk_epochs//4) == 0:
            print(f"  risk epoch {epoch:2d}/{args.risk_epochs}: BCE={total/len(x):.4f}", flush=True)
    model.eval()
    return model


@torch.inference_mode()
def score_risk(model, x, device, batch_size):
    scores = []
    for (xb,) in DataLoader(TensorDataset(torch.from_numpy(x.astype(np.float32))),
                            batch_size=batch_size):
        scores.append(model(xb.to(device)).cpu().numpy())
    return np.concatenate(scores)


def sigmoid(x):
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def calibrate_logistic(logits: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Positive-slope Platt scaling on calibration data alone.

    Unlike temperature-only scaling, the offset corrects for the deliberately
    class-weighted binary classification loss used in the risk model.
    """
    z = torch.from_numpy(logits.astype(np.float32))
    target = torch.from_numpy(y.astype(np.float32))
    log_slope = torch.nn.Parameter(torch.tensor(0.))
    intercept = torch.nn.Parameter(torch.tensor(0.))
    optimizer = torch.optim.LBFGS([log_slope, intercept], lr=0.5,
                                   max_iter=70, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        slope = log_slope.clamp(-5, 5).exp()
        calibrated = slope * z + intercept
        loss = F.binary_cross_entropy_with_logits(calibrated, target)
        loss = loss + 1e-5 * (log_slope.square() + intercept.square())
        loss.backward()
        return loss
    optimizer.step(closure)
    return float(log_slope.detach().clamp(-5, 5).exp()), float(intercept.detach())


def auc_rank(labels: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(labels, dtype=bool)
    pos, neg = int(y.sum()), int((~y).sum())
    if not pos or not neg:
        return float("nan")
    ranks = b.average_ranks(scores)
    return float((ranks[y].sum() - pos*(pos-1)/2.) / (pos*neg))


def average_precision(labels, scores):
    y = np.asarray(labels, dtype=bool)
    n_pos = int(y.sum())
    if not n_pos:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    sorted_y = y[order]
    precision = np.cumsum(sorted_y) / np.arange(1, len(y)+1)
    return float(precision[sorted_y].sum() / n_pos)


def calibration_ece(prob, labels, bins=10):
    ece = 0.
    for edges in zip(np.linspace(0,1,bins+1)[:-1], np.linspace(0,1,bins+1)[1:]):
        lo, hi = edges
        mask = (prob >= lo) & ((prob < hi) if hi < 1 else (prob <= hi))
        if mask.any():
            ece += float(mask.mean() * abs(prob[mask].mean() - labels[mask].mean()))
    return ece


def stratified_spearman(a, b_val, norm, mask, n_bins=10):
    """Within true-gradient-norm deciles, weighted rank correlation.

    Diagnostic only: true norms are NEVER used as input features or gating.
    """
    a, b_val, norm = a[mask], b_val[mask], norm[mask]
    boundaries = np.quantile(norm, np.linspace(0, 1, n_bins+1))
    vals, weights = [], []
    for i in range(n_bins):
        idx = ((norm >= boundaries[i]) &
               (norm <= boundaries[i+1] if i == n_bins-1 else norm < boundaries[i+1]))
        if idx.sum() >= 5:
            corr = b.spearman(a[idx], b_val[idx])
            if np.isfinite(corr):
                vals.append(corr)
                weights.append(int(idx.sum()))
    return float(np.average(vals, weights=weights)) if vals else float("nan")


def threshold_at_budget(score, budget):
    # Learned on calibration features/scores only, not test labels or scores.
    return float(np.quantile(score, 1.0 - budget))


def selection_metrics(name, stage, budget, threshold, scores, cos, valid, bad):
    usable = valid
    # Invalid tiny true gradients excluded from directional metrics; same filter
    # applies to all methods, but is unavailable to real online selection.
    selected = scores >= threshold
    remaining = usable & ~selected
    positives = usable & bad
    selected_valid = selected & usable
    captured = (selected & positives).sum() / max(1, int(positives.sum()))
    row = {
        "stage": stage, "score": name, "target_exact_fraction": budget,
        "realized_exact_fraction_all": float(selected.mean()),
        "realized_exact_fraction_valid": float(selected_valid.sum()/max(1, usable.sum())),
        "bad_recall": float(captured),
        "bad_rate_among_kept": float((bad & remaining).sum()/max(1,int(remaining.sum()))),
        "non_descent_rate_kept": float(((cos <= 0) & remaining).sum()/max(1,int(remaining.sum()))),
        "mean_cosine_kept": float(cos[remaining].mean()) if remaining.any() else float("nan"),
        "n_valid_kept": int(remaining.sum()),
    }
    return row


def csv_write(path, rows):
    if rows:
        b.write_csv(path, rows)


def make_plots(out, selection_rows, auc_rows, no_plots):
    if no_plots:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Plots skipped (matplotlib not installed).")
        return
    plt.figure(figsize=(9, 6))
    names = list(dict.fromkeys(r["score"] for r in selection_rows))
    for name in names:
        group = [r for r in selection_rows if r["stage"] == "frozen" and r["score"] == name]
        group.sort(key=lambda r: r["realized_exact_fraction_all"])
        if group:
            plt.plot([r["realized_exact_fraction_all"] for r in group],
                     [r["bad_recall"] for r in group], marker="o", label=name)
    plt.plot([0, 1], [0, 1], "k--", lw=1, label="random expectation")
    plt.xlabel("Actual fraction sent to exact backprop")
    plt.ylabel("Recall of low-cosine ('bad') cases")
    plt.title("Frozen network: risk selection versus exact-backprop budget")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "phase_c_selection_curve.png", dpi=170)
    plt.close()

    plt.figure(figsize=(9, 5))
    for name in names:
        group = [r for r in auc_rows if r["score"] == name]
        if group:
            plt.plot(range(len(group)), [r["auroc_bad"] for r in group],
                     marker="o", label=name)
    stages = list(dict.fromkeys(r["stage"] for r in auc_rows))
    plt.xticks(range(len(stages)), stages)
    plt.ylabel("AUROC for low-cosine gradient risk")
    plt.title("Risk identification during classifier drift")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out / "phase_c_drift_auc.png", dpi=170)
    plt.close()


def main():
    args = parse_args()
    b.set_seed(args.seed)
    device = b.choose_device(args.device)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    _, loader = create_splits(args, device)
    print(f"device={device}", flush=True)
    if args.smoke_test:
        print("SMOKE TEST: random data; ALL metrics scientifically meaningless.", flush=True)

    print("\n[1/6] Training classifier:", flush=True)
    classifier = b.Classifier(args.hidden).to(device)
    b.train_classifier(classifier, loader("model", True), device,
                       args.class_epochs, args.model_lr)
    classifier.eval().requires_grad_(False)

    print("\n[2/6] Extracting exact activation gradients for independent splits:", flush=True)
    packed = {}
    for name in ("model", "risk", "cal", "test"):
        packed[name] = collect(classifier, loader(name), device, name)
    hfit, yfit, gfit_raw = packed["model"]
    scale = max(float(gfit_raw.square().mean().sqrt()), 1e-12)
    packed = {k: (h, y, g / scale) for k, (h, y, g) in packed.items()}
    hfit, yfit, gfit = packed["model"]
    means = mean_gradients(yfit, gfit)
    floor = max(float(np.quantile(gfit.norm(dim=-1).numpy(),
                                   args.direction_floor_quantile)), 1e-8)
    print(f"gradient scale={scale:.7g}, norm exclusion floor={floor:.7g}", flush=True)

    print("\n[3/6] Training cosine+MSE and Gaussian predictors on model split only:", flush=True)
    preds = {}
    for method in ("cosine_mse", "gaussian"):
        preds[method] = b.train_predictor(method, hfit, yfit, gfit, means,
                                          device, args, floor)
    mean_np = means.numpy()

    def evaluate_snapshot(name, keep_exact_labels=True):
        h, y, exact = packed[name]
        g_pred, _ = b.predict_all(preds["cosine_mse"], "cosine_mse", h, y,
                                 means, device, args.batch_size)
        _, gauss_sigma = b.predict_all(preds["gaussian"], "gaussian", h, y,
                                      means, device, args.batch_size)
        feat, cheap = cheap_features(classifier, h, y, g_pred, mean_np, device, args.batch_size)
        cheap["gaussian_sigma"] = gauss_sigma
        cos, true_norm = per_example_cosine(exact.numpy(), g_pred)
        valid = true_norm > floor
        return {"features": feat, "cheap": cheap, "cos": cos,
                "true_norm": true_norm, "valid": valid,
                "gauss_sigma": gauss_sigma}

    print("\n[4/6] Building risk dataset and calibration split:", flush=True)
    risk = evaluate_snapshot("risk")
    cal = evaluate_snapshot("cal")
    test = evaluate_snapshot("test")
    risk_threshold = float(np.quantile(risk["cos"][risk["valid"]], args.bad_quantile))
    print(f"Low-cosine risk event: cosine <= {risk_threshold:.5f}; "
          f"computed ONLY on risk-training labels.", flush=True)
    risk_labels = (risk["cos"] <= risk_threshold).astype(np.float32)
    cal_labels = (cal["cos"] <= risk_threshold).astype(np.float32)
    normed, feature_mu, feature_std = normalize(
        risk["features"], [cal["features"], test["features"]])
    x_risk, x_cal, x_test = normed
    risk_model = train_risk_network(x_risk[risk["valid"]],
                                    risk_labels[risk["valid"]], device, args)
    logits_cal = score_risk(risk_model, x_cal, device, args.batch_size)
    risk_slope, risk_intercept = calibrate_logistic(
        logits_cal[cal["valid"]], cal_labels[cal["valid"]])
    print(f"Risk probability Platt calibration (cal split): "
          f"slope={risk_slope:.3f}, intercept={risk_intercept:.3f}")

    budget_grid = [0.1, 0.2, 0.3, 0.5]
    methods = ["learned_risk", "small_pred_norm", "large_pred_norm", "low_confidence",
               "high_entropy", "class_mean_disagreement", "gaussian_sigma"]
    def scores_from(d, norm_x):
        result = dict(d["cheap"])
        result["learned_risk"] = sigmoid(risk_slope * score_risk(risk_model, norm_x, device, args.batch_size) + risk_intercept)
        return result
    cal_scores = scores_from(cal, x_cal)
    # Thresholds are set before testing, with only *calibration scores*.
    gate_thresholds = {m: {q: threshold_at_budget(cal_scores[m], q) for q in budget_grid}
                       for m in methods}

    auc_rows, selection_rows = [], []
    def assess(stage, d, normalized_features):
        scores = scores_from(d, normalized_features)
        event = d["cos"] <= risk_threshold
        valid = d["valid"]
        directional_error = 1 - d["cos"]
        for m in methods:
            sc = scores[m]
            y_good = event[valid]
            auc = auc_rank(y_good, sc[valid])
            ap = average_precision(y_good, sc[valid])
            row = {
                "stage": stage, "score": m, "n_valid": int(valid.sum()),
                "bad_prevalence": float(y_good.mean()),
                "non_descent_prevalence": float(np.mean(d["cos"][valid] <= 0)),
                "auroc_bad": auc, "average_precision_bad": ap,
                "spearman_direction_error": b.spearman(sc[valid], directional_error[valid]),
                "spearman_true_norm": b.spearman(sc[valid], d["true_norm"][valid]),
                "within_norm_decile_spearman": stratified_spearman(
                    sc, directional_error, d["true_norm"], valid),
            }
            if m == "learned_risk":
                row["brier_bad"] = float(np.mean((sc[valid] - y_good.astype(float))**2))
                row["ece_bad"] = calibration_ece(sc[valid], y_good.astype(float))
            auc_rows.append(row)
            for budget in budget_grid:
                selection_rows.append(selection_metrics(
                    m, stage, budget, gate_thresholds[m][budget], sc,
                    d["cos"], valid, event))
        print(f"  {stage}: bad cosine fraction={event[valid].mean():.3f}, "
              f"learned AUROC={auc_rows[-len(methods)][ 'auroc_bad']:.3f}, "
              f"Gaussian σ AUROC={auc_rows[-1]['auroc_bad']:.3f}", flush=True)
    print("\n[5/6] Testing independent held-out set, with frozen calibration:", flush=True)
    assess("frozen", test, x_test)

    torch.save({
        "classifier": classifier.state_dict(), "risk_net": risk_model.state_dict(),
        "cosine_predictor": preds["cosine_mse"].state_dict(),
        "gaussian_predictor": preds["gaussian"].state_dict(),
        "feature_mu": torch.from_numpy(feature_mu),
        "feature_std": torch.from_numpy(feature_std),
        "gradient_scale": scale, "class_means": means,
        "risk_cosine_cutoff": risk_threshold, "direction_floor": floor,
        "risk_calibration_slope": risk_slope,
        "risk_calibration_intercept": risk_intercept,
        "gate_thresholds": gate_thresholds,
    }, out / "phase_c_frozen_checkpoint.pt")

    print("\n[6/6] Classifier drift test (predictors/risk estimator not retrained):", flush=True)
    for drift in range(1, args.drift_epochs+1):
        classifier.requires_grad_(True)
        b.train_classifier(classifier, loader("model", True), device, 1, args.drift_lr)
        classifier.eval().requires_grad_(False)
        h_t, y_t, g_t = collect(classifier, loader("test"), device, f"drift_{drift}")
        packed["test"] = (h_t, y_t, g_t / scale)
        drift_data = evaluate_snapshot("test")
        x_drift = (drift_data["features"] - feature_mu) / feature_std
        assess(f"drift_{drift}", drift_data, x_drift)

    csv_write(out / "phase_c_ranking.csv", auc_rows)
    csv_write(out / "phase_c_selection.csv", selection_rows)
    make_plots(out, selection_rows, auc_rows, args.skip_plots)
    torch.save({
        "classifier": classifier.state_dict(), "risk_net": risk_model.state_dict(),
        "cosine_predictor": preds["cosine_mse"].state_dict(),
        "gaussian_predictor": preds["gaussian"].state_dict(),
        "feature_mu": torch.from_numpy(feature_mu), "feature_std": torch.from_numpy(feature_std),
        "gradient_scale": scale, "class_means": means,
        "risk_cosine_cutoff": risk_threshold, "direction_floor": floor,
        "risk_calibration_slope": risk_slope,
        "risk_calibration_intercept": risk_intercept, "gate_thresholds": gate_thresholds,
        "warning": "Classifier contains POST-DRIFT weights. Predictor/risk weights were fit pre-drift.",
    }, out / "phase_c_checkpoint.pt")
    with (out / "phase_c_settings.json").open("w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "risk_cosine_cutoff": risk_threshold,
                   "risk_calibration_slope": risk_slope,
                   "risk_calibration_intercept": risk_intercept, "split_sizes": {"model":len(packed["model"][0]),
                   "risk":len(packed["risk"][0]), "cal":len(packed["cal"][0]),
                   "test":len(packed["test"][0])},
                   "budgets": budget_grid,
                   "notes": "No held-out labels used for fitting or thresholds. "
                            "Risk refers to low activation-gradient cosine, not guaranteed loss descent."}, f, indent=2)
    print("\nRESULTS (frozen test + drift; risk AUROC higher is better)")
    print(f"{'stage':<12}{'score':<26}{'AUROC':>8}{'AP':>8}{'norm-strat rho':>17}")
    for row in auc_rows:
        print(f"{row['stage']:<12}{row['score']:<26}{row['auroc_bad']:>8.3f}"
              f"{row['average_precision_bad']:>8.3f}{row['within_norm_decile_spearman']:>17.3f}")
    print(f"\nSaved to {out.resolve()}")
    print("OFFLINE DIAGNOSTIC ONLY: no synthetic updates, speedup or online risk guarantee.")


if __name__ == "__main__":
    main()
