"""Phase A: predict intermediate activation gradients and test uncertainty.

Research prototype, NOT an accelerated/backprop-free trainer. Gradient labels
come from exact backprop; the base classifier always trains with exact gradients.

Train predictor on training examples for the first N epochs, freeze it, then
measure predictive accuracy and calibration on held-out examples at later model
states. No held-out data updates either model.
"""

from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset


class Classifier(nn.Module):
    """An MLP exposing the activation whose downstream gradient is predicted."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 10))

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return class logits and the first hidden activation."""
        h = self.trunk(images)
        return self.head(h), h


class GradientPredictor(nn.Module):
    """Predict gradient mean and an isotropic log variance per input example."""

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden + 10, 192), nn.ReLU(),
            nn.Linear(192, 192), nn.ReLU(),
        )
        self.mean = nn.Linear(192, hidden)
        self.logvar = nn.Linear(192, 1)

    def forward(self, h: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the predicted scaled gradient and log variance."""
        inputs = torch.cat((h.detach(), F.one_hot(targets, 10).to(h.dtype)), dim=-1)
        features = self.net(inputs)
        return self.mean(features), self.logvar(features).clamp(-12.0, 7.0)


def gaussian_nll(target: torch.Tensor, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """Average diagonal Gaussian NLL, omitting the constant term."""
    return 0.5 * (((target - mean).square().mean(dim=-1, keepdim=True)
                   * torch.exp(-logvar)) + logvar).mean()


def loaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader]:
    """Create fixed disjoint training and validation example sets."""
    if args.smoke_test:
        # Random tensors only verify execution; they cannot validate the hypothesis.
        generator = torch.Generator().manual_seed(args.seed)
        images = torch.randn(768, 1, 28, 28, generator=generator)
        labels = torch.randint(10, (768,), generator=generator)
        dataset = TensorDataset(images, labels)
        n_train = 640
    else:
        from torchvision import datasets, transforms

        dataset = datasets.FashionMNIST(
            root=args.data_dir,
            train=True,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]),
        )
        n_train = 50000

    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.seed))
    train = Subset(dataset, indices[:n_train].tolist())
    validation = Subset(dataset, indices[n_train:].tolist())
    train_loader = DataLoader(
        train, batch_size=args.batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(args.seed + 1), num_workers=0,
    )
    val_loader = DataLoader(validation, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader


def train_epoch(
    model: Classifier,
    predictor: GradientPredictor,
    train_loader: DataLoader,
    model_opt: torch.optim.Optimizer,
    pred_opt: torch.optim.Optimizer,
    class_sums: torch.Tensor,
    class_counts: torch.Tensor,
    scale: float,
    train_predictor: bool,
    device: torch.device,
) -> tuple[float, float | None]:
    """Train classifier exactly; optionally learn from its true activation gradients."""
    model.train()
    predictor.train()
    total_loss = 0.0
    total_pred_loss = 0.0
    examples = 0
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        model_opt.zero_grad(set_to_none=True)
        logits, h = model(x)
        if train_predictor:
            h.retain_grad()
        ce = F.cross_entropy(logits, y, reduction="mean")
        ce.backward()

        if train_predictor:
            # Mean-reduced batch loss scales each individual sample's dL/dh by 1/B.
            # Correct this to recover the per-example gradient. No cross-sample layers.
            exact = h.grad.detach() * y.numel() * scale
            features = h.detach()
            with torch.no_grad():
                class_sums.index_add_(0, y, exact)
                class_counts.index_add_(0, y, torch.ones_like(y, dtype=class_counts.dtype))

            pred_opt.zero_grad(set_to_none=True)
            mean, logvar = predictor(features, y)
            p_loss = gaussian_nll(exact, mean, logvar)
            p_loss.backward()
            pred_opt.step()
            total_pred_loss += float(p_loss.item()) * y.numel()

        model_opt.step()
        total_loss += float(ce.item()) * y.numel()
        examples += y.numel()
    return total_loss / examples, (total_pred_loss / examples if train_predictor else None)


def ranks(a: np.ndarray) -> np.ndarray:
    """Return ranks for continuous-valued data (ties are unlikely here)."""
    order = np.argsort(a, kind="mergesort")
    out = np.empty(len(a), dtype=np.float64)
    out[order] = np.arange(len(a), dtype=np.float64)
    return out


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Compute a Spearman rank correlation without an additional dependency."""
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(ranks(x), ranks(y))[0, 1])


def evaluate(
    model: Classifier,
    predictor: GradientPredictor,
    loader: DataLoader,
    class_sums: torch.Tensor,
    class_counts: torch.Tensor,
    scale: float,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate future-state gradient prediction on disjoint held-out images."""
    model.eval()
    predictor.eval()
    all_error, all_sigma = [], []
    cos_pred, rel_pred, cos_baseline, rel_baseline = [], [], [], []
    covered_one, covered_95, zsq, accuracy = [], [], [], []
    class_means = class_sums / class_counts.clamp_min(1).unsqueeze(-1)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        # Exact gradients are OBSERVED for evaluation only, never used to fit the predictor.
        with torch.enable_grad():
            logits, h = model(x)
            true = torch.autograd.grad(F.cross_entropy(logits, y, reduction="sum"), h)[0].detach() * scale
        with torch.no_grad():
            mean, logvar = predictor(h.detach(), y)
            sigma = torch.exp(0.5 * logvar)
            base = class_means[y]
            residual = mean - true
            rmse = residual.square().mean(-1).sqrt()
            cosine = F.cosine_similarity(mean, true, dim=-1, eps=1e-8)
            rel = residual.norm(dim=-1) / true.norm(dim=-1).clamp_min(1e-8)
            base_cos = F.cosine_similarity(base, true, dim=-1, eps=1e-8)
            base_rel = (base - true).norm(dim=-1) / true.norm(dim=-1).clamp_min(1e-8)
            all_error.append(rmse.cpu())
            all_sigma.append(sigma[:, 0].cpu())
            cos_pred.append(cosine.cpu())
            rel_pred.append(rel.cpu())
            cos_baseline.append(base_cos.cpu())
            rel_baseline.append(base_rel.cpu())
            covered_one.append((residual.abs() <= sigma).float().mean(-1).cpu())
            covered_95.append((residual.abs() <= (1.96 * sigma)).float().mean(-1).cpu())
            zsq.append(((residual / sigma).square().mean(-1)).cpu())
            accuracy.append((logits.argmax(-1) == y).float().cpu())

    def avg(values: list[torch.Tensor]) -> float:
        """Average per-example metrics over all minibatches."""
        return float(torch.cat(values).mean().item())

    e = torch.cat(all_error).numpy()
    u = torch.cat(all_sigma).numpy()
    q1, q3 = np.quantile(u, [0.25, 0.75])
    return {
        "classifier_val_accuracy": avg(accuracy),
        "predictor_cosine": avg(cos_pred),
        "predictor_relative_l2": avg(rel_pred),
        "class_mean_cosine": avg(cos_baseline),
        "class_mean_relative_l2": avg(rel_baseline),
        "uncertainty_error_spearman": spearman(u, e),
        "coverage_1sigma": avg(covered_one),
        "coverage_1_96sigma": avg(covered_95),
        "normalized_mse": avg(zsq),
        "lowest_uncertainty_quartile_rmse": float(e[u <= q1].mean()),
        "highest_uncertainty_quartile_rmse": float(e[u >= q3].mean()),
        "mean_predicted_sigma": float(u.mean()),
        "mean_actual_rmse": float(e.mean()),
    }


def main() -> None:
    """Run the phase-A experiment and export per-epoch held-out metrics."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--predictor-train-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--model-lr", type=float, default=0.001)
    parser.add_argument("--predictor-lr", type=float, default=0.001)
    parser.add_argument("--gradient-scale", type=float, default=100.0,
                        help="Fixed numerical scale used for the predictor targets and reported errors")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--out", default="phase_a_results.csv")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--smoke-test", action="store_true", help="Use random tensors to test code execution only")
    args = parser.parse_args()
    if args.epochs <= args.predictor_train_epochs or args.predictor_train_epochs < 1:
        parser.error("require 1 <= --predictor-train-epochs < --epochs")
    if args.batch_size < 1 or args.gradient_scale <= 0 or args.hidden < 1:
        parser.error("batch size, gradient scale and hidden dimension must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda" if (args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available())) else "cpu")
    train_loader, val_loader = loaders(args)
    model = Classifier(args.hidden).to(device)
    predictor = GradientPredictor(args.hidden).to(device)
    model_opt = torch.optim.AdamW(model.parameters(), lr=args.model_lr)
    pred_opt = torch.optim.AdamW(predictor.parameters(), lr=args.predictor_lr)
    class_sums = torch.zeros(10, args.hidden, device=device)
    class_counts = torch.zeros(10, device=device)
    print(f"device={device} train={len(train_loader.dataset)} heldout={len(val_loader.dataset)}")
    if args.smoke_test:
        print("SMOKE TEST: random data; metrics have no scientific meaning")

    rows: list[dict[str, float | int]] = []
    for epoch in range(1, args.epochs + 1):
        predictor_training = epoch <= args.predictor_train_epochs
        ce, nll = train_epoch(
            model, predictor, train_loader, model_opt, pred_opt,
            class_sums, class_counts, args.gradient_scale, predictor_training, device,
        )
        print(f"epoch {epoch}/{args.epochs}: classifier CE={ce:.4f} "
              + (f"predictor NLL={nll:.4f}" if nll is not None else "predictor FROZEN"))
        if not predictor_training:
            metrics = evaluate(
                model, predictor, val_loader, class_sums, class_counts, args.gradient_scale, device,
            )
            rows.append({"epoch": epoch, **metrics})
            print("  cosine={predictor_cosine:.3f} (class baseline {class_mean_cosine:.3f}) "
                  "relL2={predictor_relative_l2:.3f} ".format(**metrics))
            print("  uncertainty/error Spearman={uncertainty_error_spearman:.3f}; "
                  "coverage ±1σ={coverage_1sigma:.3f}, ±1.96σ={coverage_1_96sigma:.3f}; "
                  "normalized MSE={normalized_mse:.3f}".format(**metrics))
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved held-out metrics to {path.resolve()}")
    print("Note: predictor scores and errors use the fixed scaled-gradient units; "
          "cosine, relative L2, and calibration coverages are scale invariant.")


if __name__ == "__main__":
    main()
