"""Phase B — controlled experiment for predicting intermediate gradients.

Purpose
-------
1. Train an ordinary classifier with exact backpropagation.
2. Freeze every classifier parameter.
3. Compute true, *per-example* activation gradients on disjoint train/holdout
   image splits, using the same frozen classifier for both.
4. Compare class-mean, MSE, cosine+MSE, isotropic Gaussian NLL, and a
   class-mean-plus-learned-residual predictor on held-out examples.
5. Save diagnostics, plots, model weights, and metrics.

This is NOT accelerated training and performs NO gradient substitution.
The held-out set is never used to fit predictors, class means, or scales.

Usage:
    py -3.14 gradient_phase_b.py --device cuda
    py -3.14 gradient_phase_b.py --smoke-test --device cpu
    py -3.14 gradient_phase_b.py --class-epochs 8 --pred-epochs 20

Requires: torch, torchvision, numpy, matplotlib (optional for plots).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset


class Classifier(nn.Module):
    """Same classifier structure and activation boundary as Phase A."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 10))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.head(h), h


class GradientPredictor(nn.Module):
    """Predicts gradient mean, optionally with a scalar log-variance per example."""

    def __init__(self, hidden: int, gaussian: bool = False):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden + 10, 192), nn.ReLU(),
            nn.Linear(192, 192), nn.ReLU(),
        )
        self.mean = nn.Linear(192, hidden)
        self.logvar = nn.Linear(192, 1) if gaussian else None

    def forward(self, h: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        z = torch.cat([h, F.one_hot(y, num_classes=10).to(h.dtype)], dim=-1)
        z = self.net(z)
        mean = self.mean(z)
        logvar = self.logvar(z).clamp(-10.0, 8.0) if self.logvar is not None else None
        return mean, logvar


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def choose_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested, but PyTorch cannot access CUDA. Try:\n"
            '  py -3.14 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"\n'
            "Check your NVIDIA driver and the CUDA-enabled PyTorch installation."
        )
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("WARNING: CUDA unavailable; running on CPU. Use --device cuda to require GPU.")
        return torch.device("cpu")
    return torch.device(requested)


def get_data(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    if args.smoke_test:
        rng = torch.Generator().manual_seed(args.seed)
        data = TensorDataset(torch.randn(768, 1, 28, 28, generator=rng),
                                torch.randint(0, 10, (768,), generator=rng))
        n_train = 640
    else:
        from torchvision import datasets, transforms
        data = datasets.FashionMNIST(
            args.data_dir, train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]),
        )
        n_train = 50_000

    # Deterministic disjoint split; caps apply AFTER splitting.
    perm = torch.randperm(len(data), generator=torch.Generator().manual_seed(args.seed)).tolist()
    train_ids = perm[:n_train]
    val_ids = perm[n_train:]
    if args.max_train:
        train_ids = train_ids[:args.max_train]
    if args.max_val:
        val_ids = val_ids[:args.max_val]
    train_data, val_data = Subset(data, train_ids), Subset(data, val_ids)
    kw = dict(batch_size=args.batch_size, num_workers=0,
              pin_memory=args.device_resolved.type == "cuda")
    train = DataLoader(train_data, shuffle=True,
                       generator=torch.Generator().manual_seed(args.seed + 1), **kw)
    validation = DataLoader(val_data, shuffle=False, **kw)
    return train, validation


def train_classifier(model: Classifier, loader: DataLoader, device: torch.device,
                     epochs: int, lr: float) -> None:
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for ep in range(1, epochs + 1):
        total_loss, total_ok, seen = 0.0, 0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = F.cross_entropy(logits, y)
            loss.backward()
            opt.step()
            b = y.numel()
            total_loss += float(loss.item()) * b
            total_ok += int((logits.argmax(-1) == y).sum().item())
            seen += b
        print(f"  classifier epoch {ep:2d}/{epochs}: CE={total_loss / seen:.4f} "
              f"train-accuracy={total_ok / seen:.3f}", flush=True)


def collect_frozen_gradients(model: Classifier, loader: DataLoader,
                             device: torch.device, name: str
                             ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """Return CPU tensors (h, label, per-example dCE/dh) and classifier accuracy.

    The head contains only example-independent Linear/ReLU layers; therefore
    derivative of SUM_i CE_i with respect to h[i] is exactly derivative of CE_i.
    No 1/batch_size scaling is introduced. Model parameters are frozen, and
    gradients are calculated only with respect to a detached activation leaf.
    """
    model.eval()
    hs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    gs: list[torch.Tensor] = []
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.no_grad():
            h0 = model.trunk(x)
        # Re-enable autograd for h even though classifier weights are frozen.
        with torch.enable_grad():
            h = h0.detach().requires_grad_(True)
            logits = model.head(h)
            loss = F.cross_entropy(logits, y, reduction="sum")
            grad = torch.autograd.grad(loss, h, create_graph=False)[0]
        hs.append(h0.detach().cpu())
        ys.append(y.detach().cpu())
        gs.append(grad.detach().cpu())
        correct += int((logits.argmax(1) == y).sum().item())
        total += y.numel()
    h_cpu, y_cpu, g_cpu = torch.cat(hs), torch.cat(ys), torch.cat(gs)
    print(f"  {name}: gradients={total}, classifier accuracy={correct / total:.3f}, "
          f"mean gradient L2={g_cpu.norm(dim=-1).mean().item():.6g}", flush=True)
    return h_cpu, y_cpu, g_cpu, correct / total


def gaussian_nll(target: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    # Isotropic Gaussian with one variance per example; Gaussian constant omitted.
    residual_mse = (target - mean).square().mean(dim=-1, keepdim=True)
    return (0.5 * (residual_mse * torch.exp(-logvar) + logvar)).mean()


def prediction_loss(method: str, pred: torch.Tensor, true: torch.Tensor,
                    logvar: torch.Tensor | None, cosine_weight: float,
                    cosine_floor: float) -> torch.Tensor:
    mse = F.mse_loss(pred, true)
    if method == "gaussian":
        assert logvar is not None
        return gaussian_nll(true, pred, logvar)
    if method == "cosine_mse":
        valid = true.norm(dim=-1) > cosine_floor
        if valid.any():
            cos = F.cosine_similarity(pred[valid], true[valid], dim=-1, eps=1e-8)
            return mse + cosine_weight * (1.0 - cos).mean()
    return mse


def train_predictor(method: str, h_train: torch.Tensor, y_train: torch.Tensor,
                    g_train: torch.Tensor, class_means: torch.Tensor,
                    device: torch.device, args: argparse.Namespace,
                    direction_floor: float) -> GradientPredictor:
    gaussian = method == "gaussian"
    predictor = GradientPredictor(h_train.shape[1], gaussian=gaussian).to(device)
    if method == "residual":
        # Initially match the class-mean baseline exactly, rather than random output.
        nn.init.zeros_(predictor.mean.weight)
        nn.init.zeros_(predictor.mean.bias)
    opt = torch.optim.AdamW(predictor.parameters(), lr=args.predictor_lr)
    data = TensorDataset(h_train, y_train, g_train)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=True, num_workers=0,
                        generator=torch.Generator().manual_seed(args.seed + 100))
    means = class_means.to(device)
    for ep in range(1, args.pred_epochs + 1):
        predictor.train()
        weighted_loss = 0.0
        for h, y, true in loader:
            h, y, true = h.to(device), y.to(device), true.to(device)
            opt.zero_grad(set_to_none=True)
            pred, logvar = predictor(h, y)
            if method == "residual":
                pred = pred + means[y]
            loss = prediction_loss(method, pred, true, logvar,
                                   args.cosine_weight, direction_floor)
            loss.backward()
            opt.step()
            weighted_loss += loss.item() * y.numel()
        if ep == 1 or ep == args.pred_epochs or ep % max(1, args.pred_epochs // 4) == 0:
            print(f"  {method:11s} epoch {ep:2d}/{args.pred_epochs}: "
                  f"loss={weighted_loss / len(data):.5f}", flush=True)
    predictor.eval()
    return predictor


@torch.inference_mode()
def predict_all(model: GradientPredictor, method: str,
                h: torch.Tensor, y: torch.Tensor, class_means: torch.Tensor,
                device: torch.device, batch_size: int
                ) -> tuple[np.ndarray, np.ndarray | None]:
    predictions: list[torch.Tensor] = []
    uncertainties: list[torch.Tensor] = []
    means = class_means.to(device)
    for h_batch, y_batch in DataLoader(TensorDataset(h, y), batch_size=batch_size):
        h_batch, y_batch = h_batch.to(device), y_batch.to(device)
        pred, logvar = model(h_batch, y_batch)
        if method == "residual":
            pred = pred + means[y_batch]
        predictions.append(pred.cpu())
        if logvar is not None:
            uncertainties.append(torch.exp(0.5 * logvar).cpu())
    return (torch.cat(predictions).numpy(),
            torch.cat(uncertainties).numpy()[:, 0] if uncertainties else None)


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Tie-aware average ranks, compatible with standard Spearman correlation."""
    order = np.argsort(values, kind="mergesort")
    sorted_values = values[order]
    ranks = np.empty(len(values), dtype=np.float64)
    pos = 0
    while pos < len(values):
        end = pos + 1
        while end < len(values) and sorted_values[end] == sorted_values[pos]:
            end += 1
        ranks[order[pos:end]] = (pos + end - 1) / 2.0
        pos = end
    return ranks


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3:
        return float("nan")
    ra, rb = average_ranks(a), average_ranks(b)
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def metrics_for(name: str, true: np.ndarray, pred: np.ndarray,
                direction_floor: float, sigma: np.ndarray | None = None
                ) -> tuple[dict[str, float | str], dict[str, np.ndarray]]:
    residual = pred - true
    true_norm = np.linalg.norm(true, axis=1)
    pred_norm = np.linalg.norm(pred, axis=1)
    error_norm = np.linalg.norm(residual, axis=1)
    valid = true_norm > direction_floor
    denom = np.maximum(true_norm, direction_floor)
    relative_error = error_norm / denom
    cos = np.sum(pred * true, axis=1) / np.maximum(true_norm * pred_norm, 1e-12)
    cos = np.clip(cos, -1.0, 1.0)
    cos_valid = cos[valid]
    rel_valid = relative_error[valid]
    row: dict[str, float | str] = {
        "method": name,
        "n_holdout": len(true),
        "direction_valid_fraction": float(valid.mean()),
        "direction_floor_normalized": float(direction_floor),
        "cosine_mean": float(cos_valid.mean()) if len(cos_valid) else float("nan"),
        "cosine_median": float(np.median(cos_valid)) if len(cos_valid) else float("nan"),
        "non_descent_fraction": float((cos_valid <= 0).mean()) if len(cos_valid) else float("nan"),
        "relative_l2_median": float(np.median(rel_valid)) if len(rel_valid) else float("nan"),
        "relative_l2_p90": float(np.percentile(rel_valid, 90)) if len(rel_valid) else float("nan"),
        "absolute_l2_mean_normalized": float(error_norm.mean()),
        "rmse_coord_normalized": float(np.sqrt(np.mean(residual ** 2))),
        "true_norm_mean_normalized": float(true_norm.mean()),
        "pred_norm_mean_normalized": float(pred_norm.mean()),
    }
    diagnostics = {
        "true_norm": true_norm,
        "error_norm": error_norm,
        "relative_error": relative_error,
        "cosine": cos,
    }
    if sigma is not None:
        # Correctly account for each coordinate, not just examples.
        z = residual / sigma[:, None]
        abs_z = np.abs(z)
        row.update({
            "uncertainty_error_spearman": spearman(sigma, error_norm),
            "uncertainty_relative_error_spearman": spearman(sigma[valid], relative_error[valid]),
            "uncertainty_directional_error_spearman": spearman(sigma[valid], 1 - cos[valid]),
            "uncertainty_true_norm_spearman": spearman(sigma, true_norm),
            "coverage_1sigma": float(np.mean(abs_z <= 1.0)),
            "coverage_1_96sigma": float(np.mean(abs_z <= 1.96)),
            "normalized_mse": float(np.mean(z ** 2)),
            "mean_sigma_normalized": float(sigma.mean()),
        })
        diagnostics["sigma"] = sigma
    return row, diagnostics


def make_plots(out_dir: Path, true_train: np.ndarray, true_val: np.ndarray,
               diagnostics: dict[str, dict[str, np.ndarray]],
               results: list[dict[str, float | str]], seed: int) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  Plotting skipped: install matplotlib with "
              "'py -3.14 -m pip install matplotlib'")
        return

    train_norm = np.linalg.norm(true_train, axis=1)
    val_norm = np.linalg.norm(true_val, axis=1)
    plt.figure(figsize=(8, 5))
    log_train = np.log10(np.maximum(train_norm, 1e-12))
    log_val = np.log10(np.maximum(val_norm, 1e-12))
    plt.hist(log_train, bins=45, density=True, alpha=0.55, label="train")
    plt.hist(log_val, bins=45, density=True, alpha=0.55, label="held-out")
    plt.title("Frozen-classifier gradient norm distribution")
    plt.xlabel("log10(||gradient||), normalized units")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "phase_b_gradient_norms.png", dpi=160)
    plt.close()

    rng = np.random.default_rng(seed)
    n = len(true_val)
    selected = rng.choice(n, size=min(n, 800), replace=False)
    plt.figure(figsize=(8, 6))
    for name, d in diagnostics.items():
        plt.scatter(np.maximum(d["true_norm"][selected], 1e-12),
                    np.maximum(d["error_norm"][selected], 1e-12),
                    s=9, alpha=0.25, label=name, rasterized=True)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Exact gradient norm (normalized)")
    plt.ylabel("Prediction error norm (normalized)")
    plt.title("Error versus true gradient magnitude (held-out)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "phase_b_error_vs_gradient_norm.png", dpi=160)
    plt.close()

    names = [r["method"] for r in results]
    cosines = [r["cosine_mean"] for r in results]
    plt.figure(figsize=(9, 5))
    plt.bar(names, cosines)
    plt.axhline(0.0, color="black", lw=0.8)
    plt.ylim(min(-0.2, min(cosines) - 0.05), max(1.0, max(cosines) + 0.05))
    plt.ylabel("Held-out mean cosine similarity (higher is better)")
    plt.title("Phase B: frozen-classifier gradient prediction")
    plt.xticks(rotation=18, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "phase_b_cosine_comparison.png", dpi=160)
    plt.close()


def write_csv(path: Path, rows: list[dict[str, float | str]]) -> None:
    all_keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase B: fixed-network gradient prediction study")
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--out-dir", default="./phase_b_output")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--class-epochs", type=int, default=6)
    p.add_argument("--pred-epochs", type=int, default=12)
    p.add_argument("--model-lr", type=float, default=0.001)
    p.add_argument("--predictor-lr", type=float, default=0.001)
    p.add_argument("--cosine-weight", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train", type=int, default=0, help="0 = full 50,000 training split")
    p.add_argument("--max-val", type=int, default=0, help="0 = full 10,000 holdout split")
    p.add_argument("--direction-floor-quantile", type=float, default=0.01,
                   help="Exclude train-norm quantile (default 1%%) of tiny true gradients from direction metrics")
    p.add_argument("--smoke-test", action="store_true", help="Random inputs, for verifying execution only")
    p.add_argument("--skip-plots", action="store_true")
    p.add_argument("--classifier-checkpoint", type=str, default="",
                   help="Optional existing classifier checkpoint; skips classifier training")
    args = p.parse_args()
    if (args.batch_size <= 0 or args.hidden <= 0 or args.class_epochs <= 0 or args.pred_epochs <= 0
            or args.model_lr <= 0 or args.predictor_lr <= 0 or args.cosine_weight < 0
            or args.max_train < 0 or args.max_val < 0
            or not 0 <= args.direction_floor_quantile < 1):
        p.error("Invalid arguments: positive sizes/epochs/rates, nonnegative weights/caps, 0<=quantile<1")
    return args


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = choose_device(args.device)
    args.device_resolved = device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_loader, val_loader = get_data(args)
    print(f"device={device} train={len(train_loader.dataset)} heldout={len(val_loader.dataset)}")
    if args.smoke_test:
        print("SMOKE TEST uses random labels and images: metrics are NOT scientifically meaningful.")
    classifier = Classifier(hidden=args.hidden).to(device)

    if args.classifier_checkpoint:
        payload = torch.load(args.classifier_checkpoint, map_location=device, weights_only=True)
        # Accept either a state dictionary or one saved by this script.
        state = payload["classifier"] if "classifier" in payload else payload
        classifier.load_state_dict(state, strict=True)
        print(f"Loaded frozen classifier from {args.classifier_checkpoint}")
    else:
        print("Training classifier (exact backpropagation):", flush=True)
        train_classifier(classifier, train_loader, device, args.class_epochs, args.model_lr)

    classifier.eval()
    classifier.requires_grad_(False)
    torch.save({"classifier": classifier.state_dict(), "hidden": args.hidden,
                "seed": args.seed, "smoke_test": args.smoke_test},
               out_dir / "classifier_frozen.pt")

    print("Extracting fixed-network activation-gradient labels:", flush=True)
    h_train, y_train, g_train_raw, train_acc = collect_frozen_gradients(
        classifier, DataLoader(train_loader.dataset, batch_size=args.batch_size, shuffle=False),
        device, "train")
    h_val, y_val, g_val_raw, val_acc = collect_frozen_gradients(
        classifier, val_loader, device, "held-out")

    # Training data ONLY. Scalar target normalization improves regression conditioning.
    # Per-coordinate RMS, not a per-example norm, gives average target squared coord ~1.
    gradient_scale = max(float(g_train_raw.square().mean().sqrt().item()), 1e-12)
    g_train = g_train_raw / gradient_scale
    g_val = g_val_raw / gradient_scale
    gradient_norm_train = g_train.norm(dim=-1).numpy()
    direction_floor = max(float(np.quantile(gradient_norm_train,
                                         args.direction_floor_quantile)), 1e-8)
    class_sums = torch.zeros(10, args.hidden, dtype=g_train.dtype)
    class_counts = torch.bincount(y_train, minlength=10)
    class_sums.index_add_(0, y_train, g_train)
    # Classes absent from a tiny smoke sample fall back to the global mean.
    class_means = class_sums / class_counts.clamp(min=1).unsqueeze(-1)
    class_means[class_counts == 0] = g_train.mean(dim=0)
    print(f"gradient scale (raw coordinate RMS)={gradient_scale:.7g}; "
          f"direction threshold (normalized L2)={direction_floor:.7g}", flush=True)

    true = g_val.numpy()
    base_pred = class_means[y_val].numpy()
    baseline, baseline_diag = metrics_for("class_mean", true, base_pred, direction_floor)
    rows = [baseline]
    diagnostics = {"class_mean": baseline_diag}
    print(f"  class_mean held-out cosine={baseline['cosine_mean']:.4f}, "
          f"median relative L2={baseline['relative_l2_median']:.4f}", flush=True)

    for method in ("mse", "cosine_mse", "gaussian", "residual"):
        print(f"Training {method} predictor:", flush=True)
        predictor = train_predictor(method, h_train, y_train, g_train,
                                    class_means, device, args, direction_floor)
        pred, sigma = predict_all(predictor, method, h_val, y_val,
                                  class_means, device, args.batch_size)
        row, diag = metrics_for(method, true, pred, direction_floor, sigma)
        rows.append(row)
        diagnostics[method] = diag
        torch.save({"predictor": predictor.cpu().state_dict(),
                    "method": method, "hidden": args.hidden,
                    "gradient_scale": gradient_scale,
                    "class_means": class_means}, out_dir / f"predictor_{method}.pt")
        print(f"  HELD-OUT {method}: cosine={row['cosine_mean']:.4f}, "
              f"relL2-median={row['relative_l2_median']:.4f}, "
              f"non-descent={row['non_descent_fraction']:.3f}", flush=True)
        if sigma is not None:
            print(f"    uncertainty-error Spearman={row['uncertainty_error_spearman']:.3f}, "
                  f"uncertainty-norm Spearman={row['uncertainty_true_norm_spearman']:.3f}, "
                  f"coverage(1σ,1.96σ)=({row['coverage_1sigma']:.3f},"
                  f"{row['coverage_1_96sigma']:.3f})", flush=True)

    write_csv(out_dir / "phase_b_metrics.csv", rows)
    with (out_dir / "phase_b_run.json").open("w", encoding="utf-8") as f:
        json.dump({"settings": {k: str(v) if isinstance(v, torch.device) else v
                                for k, v in vars(args).items()},
                   "classifier_accuracy_train": train_acc,
                   "classifier_accuracy_heldout": val_acc,
                   "gradient_scale_raw_coordinate_rms": gradient_scale,
                   "direction_floor_normalized": direction_floor,
                   "class_counts": class_counts.tolist()}, f, indent=2)
    if not args.skip_plots:
        make_plots(out_dir, g_train.numpy(), true, diagnostics, rows, args.seed)
    print("\nRESULT SUMMARY (held-out; bigger cosine is better)")
    print(f"{'method':12s} {'cosine':>10s} {'relL2-med':>12s} {'bad-direction':>14s}")
    for row in rows:
        print(f"{row['method']:12s} {row['cosine_mean']:10.4f} "
              f"{row['relative_l2_median']:12.4f} {row['non_descent_fraction']:14.3f}")
    print(f"\nSaved to {out_dir.resolve()}")
    print("All predictors fitted to the SAME frozen classifier; no held-out fitting.")
    print("No synthetic updates were made; this is a prediction diagnostic experiment.")


if __name__ == "__main__":
    main()
