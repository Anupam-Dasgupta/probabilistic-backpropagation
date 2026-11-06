"""R0/R1: robust, reproducible intermediate-gradient prediction study.

This experiment is deliberately OFFLINE: it does not substitute gradients or
claim accelerated training. For the full pre-registered study, see README_R1.md.

Usage (Windows CMD):
    py -3.14 gradient_robust_r1.py --smoke-test --device cpu
    py -3.14 gradient_robust_r1.py --device cuda --out r1_study
    py -3.14 gradient_robust_r1.py --device cuda --mode classifiers --out r1_study
    py -3.14 gradient_robust_r1.py --device cuda --mode predictors --out r1_study

Dependencies: torch torchvision numpy. matplotlib optional for plots.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, TensorDataset


METHODS = ("mse", "cosine_mse", "gaussian", "residual")


class Classifier(nn.Module):
    def __init__(self, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 10))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.head(h), h


class Predictor(nn.Module):
    def __init__(self, hidden: int, width: int = 192, gaussian: bool = False):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(hidden + 10, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
        )
        self.mean = nn.Linear(width, hidden)
        self.logvar = nn.Linear(width, 1) if gaussian else None

    def forward(self, h: torch.Tensor, y: torch.Tensor):
        z = torch.cat((h, F.one_hot(y, 10).to(h.dtype)), dim=-1)
        z = self.features(z)
        return self.mean(z), (self.logvar(z).clamp(-12, 8) if self.logvar is not None else None)


def set_seed(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)


def device_from_arg(value: str) -> torch.device:
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable in this Python's PyTorch installation.\n"
            'Check: py -3.14 -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"'
        )
    if value == "auto":
        value = "cuda" if torch.cuda.is_available() else "cpu"
        if value == "cpu":
            print("WARNING: CUDA unavailable; running on CPU.", flush=True)
    return torch.device(value)


def parse_ints(value: str) -> list[int]:
    result = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not result or len(set(result)) != len(result):
        raise argparse.ArgumentTypeError("Use a nonempty, unique comma-separated integer list")
    return result


def parse_methods(value: str) -> list[str]:
    methods = [x.strip() for x in value.split(",") if x.strip()]
    if not methods or len(set(methods)) != len(methods) or any(m not in METHODS for m in methods):
        raise argparse.ArgumentTypeError(f"Methods must be unique and from {METHODS}")
    return methods


def args_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--mode", choices=["all", "classifiers", "predictors"], default="all")
    p.add_argument("--out", default="r1_study")
    p.add_argument("--data-dir", default="./data")
    p.add_argument("--seeds", type=parse_ints, default=[0, 1, 2])
    p.add_argument("--split-seed", type=int, default=20261025)
    p.add_argument("--class-epochs", type=int, default=60)
    p.add_argument("--checkpoints", type=parse_ints, default=[5, 10, 20, 40, 60])
    p.add_argument("--predictor-epochs", type=int, default=50)
    p.add_argument("--predictor-snapshots", type=parse_ints, default=[10, 25, 50])
    p.add_argument("--methods", type=parse_methods, default=list(METHODS))
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--predictor-batch-size", type=int, default=512)
    p.add_argument("--class-lr", type=float, default=1e-3)
    p.add_argument("--predictor-lr", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--pred-width", type=int, default=192)
    p.add_argument("--cosine-weight", type=float, default=0.25)
    p.add_argument("--min-norm-ratio", type=float, default=1e-3,
                   help="Direction valid if true norm > ratio * training median true norm; fixed per checkpoint")
    p.add_argument("--deterministic", action="store_true")
    p.add_argument("--save-predictions", action="store_true", help="Save per-example diagnostics for selected models")
    p.add_argument("--smoke-test", action="store_true", help="Small structured synthetic test; NOT scientific results")
    return p


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def make_data(args: argparse.Namespace):
    if args.smoke_test:
        # Structured labels prevent all-random "learning" in smoke mode.
        rng = torch.Generator().manual_seed(args.split_seed)
        templates = torch.randn(10, 1, 28, 28, generator=rng)
        labels = torch.arange(320) % 10
        images = templates[labels] + 0.7 * torch.randn(320, 1, 28, 28, generator=rng)
        dataset = TensorDataset(images, labels)
        sizes = (200, 60, 60)
    else:
        from torchvision import datasets, transforms
        dataset = datasets.FashionMNIST(
            root=args.data_dir, train=True, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.2860,), (0.3530,)),
            ]),
        )
        # The official FashionMNIST test set is intentionally NEVER loaded here.
        sizes = (40_000, 10_000, 10_000)
    if sum(sizes) != len(dataset):
        raise RuntimeError("Dataset has unexpected size")
    perm = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(args.split_seed)).tolist()
    i, j = sizes[0], sizes[0] + sizes[1]
    splits = {"train": perm[:i], "val": perm[i:j], "dev": perm[j:]}
    out = {name: Subset(dataset, ids) for name, ids in splits.items()}
    return out, splits


def data_loader(data, batch: int, seed: int, shuffle: bool, device: torch.device):
    return DataLoader(data, batch_size=batch, shuffle=shuffle, num_workers=0,
                      pin_memory=device.type == "cuda",
                      generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def evaluate_classifier(model, dataset, args, device) -> dict:
    model.eval()
    loss_sum, correct, total = 0., 0, 0
    for x, y in data_loader(dataset, args.batch_size, 0, False, device):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits, _ = model(x)
        loss_sum += F.cross_entropy(logits, y, reduction="sum").item()
        correct += (logits.argmax(-1) == y).sum().item()
        total += len(y)
    return {"cross_entropy": float(loss_sum / total), "accuracy": float(correct / total)}


def audit_gradients(device: torch.device, hidden: int) -> dict:
    """R0: loop-vs-batch activation grads, and upstream Jacobian VJP vs backprop."""
    set_seed(777, deterministic=False)
    model = Classifier(hidden=hidden).to(device).eval()
    x = torch.randn(4, 1, 28, 28, device=device)
    y = torch.tensor([0, 1, 2, 3], device=device)
    h = model.trunk(x).detach().requires_grad_(True)
    loss = F.cross_entropy(model.head(h), y, reduction="sum")
    vectorized = torch.autograd.grad(loss, h)[0]
    individual = []
    for i in range(len(y)):
        hi = h[i:i+1].detach().requires_grad_(True)
        li = F.cross_entropy(model.head(hi), y[i:i+1], reduction="sum")
        individual.append(torch.autograd.grad(li, hi)[0].detach())
    loop = torch.cat(individual)
    error_examples = (vectorized - loop).abs().max().item()

    # Exact full-network parameter gradient of first Linear weight.
    first_weight = model.trunk[1].weight
    full_loss = F.cross_entropy(model(x)[0], y, reduction="sum")
    direct = torch.autograd.grad(full_loss, first_weight)[0]
    trunk_h = model.trunk(x)
    upstream = torch.autograd.grad(trunk_h, first_weight,
                                   grad_outputs=vectorized.detach())[0]
    error_vjp = (direct - upstream).abs().max().item()
    threshold = 3e-5 if device.type == "cuda" else 1e-5
    if max(error_examples, error_vjp) >= threshold:
        raise AssertionError(f"R0 gradient audit failed: {error_examples=}, {error_vjp=}")
    # Independent finite-difference check with a double-precision classifier.
    double_model = Classifier(hidden=hidden).double().eval()
    dh = torch.randn(2, hidden, dtype=torch.float64, requires_grad=True)
    dy = torch.tensor([0, 2])
    grad_d = torch.autograd.grad(F.cross_entropy(double_model.head(dh), dy, reduction="sum"), dh)[0]
    epsilon = 1e-5
    plus = dh.detach().clone()
    minus = dh.detach().clone()
    plus[0, 0] += epsilon
    minus[0, 0] -= epsilon
    fd = (F.cross_entropy(double_model.head(plus), dy, reduction="sum")
          - F.cross_entropy(double_model.head(minus), dy, reduction="sum"))/(2*epsilon)
    error_fd = abs(fd.item()-grad_d[0, 0].item())
    if error_fd > 1e-6:
        raise AssertionError(f"Finite-difference audit failed: {error_fd=}")
    return {"max_abs_batch_vs_loop": error_examples,
            "max_abs_direct_vs_vjp": error_vjp, "finite_difference_error": error_fd,
            "threshold": threshold,
            "gradient_reduction": "sum (per-example gradients; no / batch_size)"}


def train_classifiers(args, splits, out, device):
    for seed in args.seeds:
        set_seed(seed, args.deterministic)
        sdir = out / f"seed_{seed}"
        sdir.mkdir(parents=True, exist_ok=True)
        model = Classifier(args.hidden).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.class_lr)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[max(1, round(args.class_epochs*0.5)),
                                   max(2, round(args.class_epochs*0.75))], gamma=0.2)
        loader = data_loader(splits["train"], args.batch_size, seed+123, True, device)
        history = []
        print(f"\n[Classifier seed {seed}]", flush=True)
        for epoch in range(1, args.class_epochs+1):
            model.train()
            total, accum, correct = 0, 0., 0
            for x, y in loader:
                x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                logits, _ = model(x)
                loss = F.cross_entropy(logits, y)
                loss.backward()
                optimizer.step()
                accum += float(loss.detach())*len(y)
                correct += (logits.argmax(-1) == y).sum().item()
                total += len(y)
            scheduler.step()
            if epoch in args.checkpoints or epoch == 1 or epoch == args.class_epochs:
                v = evaluate_classifier(model, splits["val"], args, device)
                row = {"seed":seed, "epoch":epoch, "train_ce":accum/total,
                       "train_accuracy": correct/total, "val_ce":v["cross_entropy"],
                       "val_accuracy":v["accuracy"], "learning_rate":optimizer.param_groups[0]["lr"]}
                history.append(row)
                print(f"  epoch {epoch:3d}: train CE={accum/total:.4f}, "
                      f"val CE={v['cross_entropy']:.4f}, val acc={v['accuracy']:.4f}", flush=True)
            if epoch in args.checkpoints:
                torch.save({"model":{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                            "seed":seed, "epoch":epoch, "hidden":args.hidden}, sdir/f"classifier_e{epoch}.pt")
        write_csv(sdir/"classifier_history.csv", history)


def extract_gradient_data(model, dataset, device, batch_size):
    """Per-example dCE_i/dh_i, exact and unaveraged; disjoint fixed splits."""
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    hs, ys, gs, confs, correct = [], [], [], [], []
    for x, y in data_loader(dataset, batch_size, 0, False, device):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.no_grad():
            h0 = model.trunk(x)
        with torch.enable_grad():
            h = h0.detach().requires_grad_(True)
            logits = model.head(h)
            grad = torch.autograd.grad(F.cross_entropy(logits, y, reduction="sum"), h)[0]
        probs = logits.detach().softmax(-1)
        hs.append(h0.cpu())
        ys.append(y.cpu())
        gs.append(grad.detach().cpu())
        confs.append(probs.max(dim=-1).values.cpu())
        correct.append((logits.detach().argmax(-1) == y).cpu())
    return {"h":torch.cat(hs), "y":torch.cat(ys), "g":torch.cat(gs),
            "confidence":torch.cat(confs), "correct":torch.cat(correct)}


def fit_class_means(g: torch.Tensor, y: torch.Tensor):
    means = torch.zeros(10, g.shape[1], dtype=g.dtype)
    for c in range(10):
        mask = (y==c)
        if mask.any():
            means[c] = g[mask].mean(dim=0)
    return means


def loss_for(method, mu, target, logvar, cosine_weight, direction_floor):
    mse = F.mse_loss(mu, target)
    if method == "gaussian":
        assert logvar is not None
        coord_mse = (target-mu).square().mean(-1, keepdim=True)
        return (0.5*(torch.exp(-logvar)*coord_mse + logvar)).mean()
    if method == "cosine_mse":
        valid = target.norm(dim=-1)>direction_floor
        if valid.any():
            cosine = F.cosine_similarity(mu[valid], target[valid], dim=-1, eps=1e-8)
            return mse + cosine_weight*(1-cosine).mean()
    return mse


@torch.no_grad()
def predict_all(method, predictor, split, class_means, scale, device, batch):
    predictor.eval()
    out, sigma = [], []
    loader = DataLoader(TensorDataset(split["h"], split["y"]), batch_size=batch, shuffle=False)
    for h, y in loader:
        h, y = h.to(device), y.to(device)
        mu, logvar = predictor(h, y)
        if method == "residual":
            mu = mu + class_means.to(device)[y]
        out.append(mu.cpu()*scale)
        if logvar is not None:
            sigma.append(torch.exp(0.5*logvar).cpu()*scale)
    return torch.cat(out), (torch.cat(sigma) if sigma else None)


def metric_dict(true, pred, floor, correct=None, sigma=None):
    """Include metric denominators; do not ignore zero predictions on valid targets."""
    true, pred = true.float(), pred.float()
    norm = true.norm(dim=-1)
    pnorm = pred.norm(dim=-1)
    valid = norm > floor
    e = (pred-true).norm(dim=-1)
    dot = (pred*true).sum(-1)
    cosine = dot/(norm.clamp_min(1e-12)*pnorm.clamp_min(1e-12))
    cosine = cosine.clamp(-1,1)
    if valid.any():
        c = cosine[valid]
        cmean, cmed = c.mean().item(), c.median().item()
        nond = (c <= 0).float().mean().item()
    else:
        cmean, cmed, nond = float("nan"),float("nan"),float("nan")
    result = {
        "n_total":int(len(norm)),"n_direction_valid":int(valid.sum()),
        "direction_valid_fraction":float(valid.float().mean()),
        "cosine_mean":cmean, "cosine_median":cmed,
        "nonpositive_cosine_fraction":nond,
        "relative_l2_median":float((e/norm.clamp_min(floor)).median()),
        "relative_l2_p90":float(torch.quantile(e/norm.clamp_min(floor),0.9)),
        "normalized_rmse":float(torch.sqrt((pred-true).square().mean()/true.square().mean().clamp_min(1e-15))),
        "mean_absolute_l2":float(e.mean()),
        "true_norm_mean":float(norm.mean()),
        "pred_norm_mean":float(pnorm.mean()),
        "near_zero_true_fraction":float((~valid).float().mean()),
    }
    if correct is not None:
        for name, mask in (("correct",correct.bool()),("incorrect",~correct.bool())):
            v = valid & mask
            result[f"cosine_{name}"] = float(cosine[v].mean()) if v.any() else float("nan")
            result[f"n_{name}"] = int(v.sum())
    if sigma is not None:
        # Scalar isotropic sigma per example: coordinate coverage, descriptive only.
        z = (pred-true).abs()
        result["coverage_1sigma"] = float((z<=sigma).float().mean())
        result["coverage_1_96sigma"] = float((z<=1.96*sigma).float().mean())
        result["uncertainty_true_norm_spearman"] = spearman(sigma.squeeze(-1), norm)
        result["uncertainty_relative_error_spearman"] = spearman(sigma.squeeze(-1),e/norm.clamp_min(floor))
    return result


def spearman(a, b):
    # Stable average ranks for ties; n <=10K. numpy only, no scipy dependency.
    def rank(x):
        x = np.asarray(x, dtype=np.float64)
        order = np.argsort(x, kind="stable")
        xsort = x[order]
        breaks = np.r_[0, np.flatnonzero(np.diff(xsort))+1, len(x)]
        result = np.empty(len(x),dtype=np.float64)
        for lo, hi in zip(breaks[:-1],breaks[1:]):
            result[order[lo:hi]] = (lo+hi-1)*0.5
        return result
    aa = rank(a.detach().cpu().numpy() if torch.is_tensor(a) else a)
    bb = rank(b.detach().cpu().numpy() if torch.is_tensor(b) else b)
    if aa.std()==0 or bb.std()==0:
        return float("nan")
    return float(np.corrcoef(aa,bb)[0,1])


def run_predictors(args, splits, out, device):
    for seed in args.seeds:
        sdir = out / f"seed_{seed}"
        for epoch in args.checkpoints:
            source = sdir/f"classifier_e{epoch}.pt"
            if not source.exists():
                raise FileNotFoundError(f"Classifier checkpoint missing: {source}. Run --mode classifiers first.")
            obj = torch.load(source, map_location="cpu", weights_only=True)
            model = Classifier(args.hidden).to(device)
            model.load_state_dict(obj["model"])
            model.eval()
            print(f"\n[Predictors seed={seed}, classifier epoch={epoch}] Extracting exact gradients ...",flush=True)
            data = {name:extract_gradient_data(model, split,device,args.batch_size)
                    for name,split in splits.items()}
            del model
            raw_g = data["train"]["g"]
            scale = raw_g.square().mean().sqrt().clamp_min(1e-10).item()
            scaled_true = raw_g / scale
            direction_floor = max(float(raw_g.norm(dim=-1).median())*args.min_norm_ratio,1e-10)
            scaled_floor = direction_floor/scale
            # These quantities MUST come from training only.
            means_scaled = fit_class_means(scaled_true, data["train"]["y"])
            checkpoint_dir = sdir/f"checkpoint_{epoch}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            with (checkpoint_dir/"gradient_stats.json").open("w",encoding="utf-8") as f:
                json.dump({"gradient_scale_train_only":scale,"gradient_floor_train_only":direction_floor,
                           "train_gradient_norm_mean":float(raw_g.norm(dim=-1).mean()),
                           "train_gradient_norm_median":float(raw_g.norm(dim=-1).median()),
                           "n_train":len(raw_g),"n_val":len(data["val"]["g"]),
                           "n_dev":len(data["dev"]["g"])},f,indent=2)

            base_rows = []
            for base in ("zero","global_mean","class_mean"):
                for part in ("val","dev"):
                    truth=data[part]["g"]
                    if base=="zero":
                        predictions = torch.zeros_like(truth)
                    elif base=="global_mean":
                        predictions = raw_g.mean(0).expand_as(truth)
                    else:
                        predictions = means_scaled[data[part]["y"]]*scale
                    row={"seed":seed,"classifier_epoch":epoch,"method":base,
                         "pred_epochs":0,"split":part,"selected_by_val":True}
                    row.update(metric_dict(truth,predictions,direction_floor,data[part]["correct"]))
                    base_rows.append(row)
            write_csv(checkpoint_dir/"baselines.csv",base_rows)
            print("  class-mean dev cosine=%.4f; gradient scale=%.5g; valid fraction=%.3f"%(
                  next(r["cosine_mean"] for r in base_rows if r["method"]=="class_mean" and r["split"]=="dev"),
                  scale,next(r["direction_valid_fraction"] for r in base_rows if r["method"]=="class_mean" and r["split"]=="dev")),flush=True)

            for method in args.methods:
                method_i=METHODS.index(method)
                metrics_path=checkpoint_dir/f"{method}_metrics.csv"
                # Skip completed methods only under identical experiment config (checked in main).
                if metrics_path.exists():
                    print(f"  {method}: existing results, skipping",flush=True)
                    continue
                set_seed(seed*10_000+epoch*50+method_i+6000,args.deterministic)
                net=Predictor(args.hidden,args.pred_width,gaussian=(method=="gaussian")).to(device)
                if method=="residual":
                    nn.init.zeros_(net.mean.weight)
                    nn.init.zeros_(net.mean.bias)
                opt=torch.optim.AdamW(net.parameters(),lr=args.predictor_lr)
                dataset=TensorDataset(data["train"]["h"],data["train"]["y"],scaled_true)
                loader=data_loader(dataset,args.predictor_batch_size,seed*100+epoch,True,device)
                rows=[];best_score=-float("inf");best_epoch=None;best_state=None
                print(f"  training {method} ...",flush=True)
                for pe in range(1,args.predictor_epochs+1):
                    net.train()
                    loss_sum=0.
                    for h,y,g in loader:
                        h,y,g=h.to(device,non_blocking=True),y.to(device,non_blocking=True),g.to(device,non_blocking=True)
                        opt.zero_grad(set_to_none=True)
                        mu,logvar=net(h,y)
                        if method=="residual":
                            mu=mu+means_scaled.to(device)[y]
                        loss=loss_for(method,mu,g,logvar,args.cosine_weight,scaled_floor)
                        if not torch.isfinite(loss):
                            raise RuntimeError(f"Nonfinite loss for seed={seed}, epoch={epoch}, method={method}")
                        loss.backward()
                        opt.step()
                        loss_sum+=loss.detach().item()*len(y)
                    if pe in args.predictor_snapshots:
                        pred,sigma=predict_all(method,net,data["val"],means_scaled,scale,device,args.predictor_batch_size)
                        metrics=metric_dict(data["val"]["g"],pred,direction_floor,data["val"]["correct"],sigma)
                        row={"seed":seed,"classifier_epoch":epoch,"method":method,
                             "pred_epochs":pe,"train_objective":loss_sum/len(dataset),"split":"val", "selected_by_val":False}
                        row.update(metrics)
                        rows.append(row)
                        # Predeclared selection criterion per method. Never use dev for selection.
                        if method=="cosine_mse":
                            score=metrics["cosine_mean"]
                        else:
                            score=-metrics["normalized_rmse"]
                        if math.isfinite(score) and score>best_score:
                            best_score=score
                            best_epoch=pe
                            best_state={k:v.detach().cpu().clone() for k,v in net.state_dict().items()}
                        print(f"    {pe:3d}/{args.predictor_epochs}: train loss={row['train_objective']:.4f} "
                              f"val cos={metrics['cosine_mean']:.3f} val nRMSE={metrics['normalized_rmse']:.3f}",flush=True)
                if best_state is None:
                    raise RuntimeError("No finite predictor validation score")
                net.load_state_dict(best_state)
                pred,sigma=predict_all(method,net,data["dev"],means_scaled,scale,device,args.predictor_batch_size)
                metrics=metric_dict(data["dev"]["g"],pred,direction_floor,data["dev"]["correct"],sigma)
                final={"seed":seed,"classifier_epoch":epoch,"method":method,
                       "pred_epochs":best_epoch,"train_objective":float("nan"),"split":"dev","selected_by_val":True}
                final.update(metrics)
                rows.append(final)
                for r in rows:
                    r["selected_by_val"] = r["split"]=="dev" or (r["split"]=="val" and r["pred_epochs"]==best_epoch)
                torch.save({"predictor":best_state,"method":method,"classifier_epoch":epoch,
                            "seed":seed,"selected_predictor_epoch":best_epoch,
                            "scale_train_only":scale,"class_means_train_only":means_scaled},
                           checkpoint_dir/f"{method}_selected.pt")
                if args.save_predictions:
                    np.savez_compressed(checkpoint_dir/f"{method}_dev_diagnostics.npz",
                                        true_norm=data["dev"]["g"].norm(dim=-1).numpy(),
                                        pred_norm=pred.norm(dim=-1).numpy(),
                                        cosine=F.cosine_similarity(data["dev"]["g"],pred,dim=-1).numpy(),
                                        abs_l2=(data["dev"]["g"]-pred).norm(dim=-1).numpy(),
                                        correct=data["dev"]["correct"].numpy(),
                                        sigma=sigma.numpy() if sigma is not None else np.array([]))
                write_csv(metrics_path,rows)
                print(f"    SELECTED val epoch {best_epoch}, dev cosine={metrics['cosine_mean']:.4f} "
                      f"dev nRMSE={metrics['normalized_rmse']:.4f} "
                      f"nonpositive={metrics['nonpositive_cosine_fraction']:.3f}",flush=True)
                del net
            del data
        aggregate_metrics(args,out)


def aggregate_metrics(args,out):
    rows=[]
    for seed in args.seeds:
        for ep in args.checkpoints:
            checkpoint_dir=out/f"seed_{seed}"/f"checkpoint_{ep}"
            rows.extend(load_csv(checkpoint_dir/"baselines.csv"))
            for m in args.methods:
                rows.extend(load_csv(checkpoint_dir/f"{m}_metrics.csv"))
    # Dev must be evaluated only for validation-selected model; all snapshots on val.
    write_csv(out/"all_metrics.csv",rows)
    dev=[r for r in rows if r.get("split")=="dev"]
    summary=[]
    metrics=("cosine_mean","normalized_rmse","nonpositive_cosine_fraction",
             "relative_l2_median","near_zero_true_fraction")
    for ep in args.checkpoints:
        for m in ("zero","global_mean","class_mean",*args.methods):
            members=[r for r in dev if int(r["classifier_epoch"])==ep and r["method"]==m]
            if not members:
                continue
            record={"classifier_epoch":ep,"method":m,"n_seeds":len(members)}
            for metric in metrics:
                values=np.array([float(r[metric]) for r in members],dtype=np.float64)
                values=values[np.isfinite(values)]
                record[f"{metric}_mean"]=float(values.mean()) if len(values) else float("nan")
                record[f"{metric}_std"]=float(values.std(ddof=1)) if len(values)>1 else float("nan")
            summary.append(record)
    write_csv(out/"summary_dev.csv",summary)
    # Pair each learned method to the class-mean baseline ON THE SAME run.
    paired=[]
    for ep in args.checkpoints:
        for method in args.methods:
            deltas=[]
            for seed in args.seeds:
                a=next((r for r in dev if int(r["classifier_epoch"])==ep and
                        int(r["seed"])==seed and r["method"]==method),None)
                b=next((r for r in dev if int(r["classifier_epoch"])==ep and
                        int(r["seed"])==seed and r["method"]=="class_mean"),None)
                if a is not None and b is not None:
                    deltas.append((float(a["cosine_mean"])-float(b["cosine_mean"]),
                                   float(a["normalized_rmse"])-float(b["normalized_rmse"])))
            if deltas:
                arr=np.asarray(deltas)
                paired.append({"classifier_epoch":ep,"method":method,"n_paired_seeds":len(deltas),
                               "cosine_delta_vs_class_mean_mean":float(arr[:,0].mean()),
                               "cosine_delta_vs_class_mean_sd":float(arr[:,0].std(ddof=1)) if len(arr)>1 else float("nan"),
                               "nrmse_delta_vs_class_mean_mean":float(arr[:,1].mean()),
                               "nrmse_delta_vs_class_mean_sd":float(arr[:,1].std(ddof=1)) if len(arr)>1 else float("nan")})
    write_csv(out/"paired_vs_class_mean.csv",paired)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for metric,ylabel,filename in (("cosine_mean","Mean cosine similarity","cosine_by_checkpoint.png"),
                                        ("normalized_rmse","Normalized RMSE","rmse_by_checkpoint.png")):
            fig,ax=plt.subplots(figsize=(8,4.5))
            for method in ("class_mean",*args.methods):
                rs=[r for r in summary if r["method"]==method]
                if rs:
                    xx=[int(r["classifier_epoch"]) for r in rs]
                    yy=[r[f"{metric}_mean"] for r in rs]
                    yerr=[r[f"{metric}_std"] if math.isfinite(r[f"{metric}_std"]) else 0 for r in rs]
                    ax.errorbar(xx,yy,yerr=yerr,marker="o",capsize=3,label=method)
            ax.set_xlabel("Classifier training epoch")
            ax.set_ylabel(ylabel)
            ax.set_title("Development evaluation; mean ± SD across classifier seeds")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out/filename,dpi=160)
            plt.close(fig)
    except ImportError:
        pass
    return summary


def main():
    args=args_parser().parse_args()
    if args.smoke_test:
        if args.out=="r1_study":
            args.out="r1_smoke_output"
        args.seeds=[0]
        args.class_epochs=2
        args.checkpoints=[1,2]
        args.predictor_epochs=2
        args.predictor_snapshots=[1,2]
        args.methods=list(METHODS)
        args.batch_size=64
        args.predictor_batch_size=64
    if (any(x<=0 for x in args.checkpoints+args.predictor_snapshots)
        or max(args.checkpoints)>args.class_epochs
        or max(args.predictor_snapshots)>args.predictor_epochs
        or args.batch_size<=0 or args.predictor_batch_size<=0
        or args.min_norm_ratio<=0):
        raise ValueError("Check positive epochs, snapshot ranges, batch sizes, and norm ratio")
    args.checkpoints=sorted(args.checkpoints)
    args.predictor_snapshots=sorted(args.predictor_snapshots)
    if args.deterministic:
        # Must be set before CUDA BLAS kernels initialize in this process.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    device=device_from_arg(args.device)
    out=Path(args.out)
    out.mkdir(parents=True,exist_ok=True)
    fingerprint={k:v for k,v in vars(args).items() if k not in ("mode","out","device","save_predictions")}
    manifest=out/"config.json"
    if manifest.exists():
        prior=json.loads(manifest.read_text(encoding="utf-8"))
        if prior["experiment"]!=fingerprint:
            raise ValueError("Existing output directory has DIFFERENT settings. Use a new --out folder.")
    else:
        manifest.write_text(json.dumps({"experiment":fingerprint,"torch":torch.__version__,
                                        "numpy":np.__version__,"cuda_build":torch.version.cuda,
                                        "python":__import__("sys").version},indent=2),encoding="utf-8")
    set_seed(args.split_seed,args.deterministic)
    print(f"Device: {device}; seeds={args.seeds}; checkpoints={args.checkpoints}",flush=True)
    audit=audit_gradients(device,args.hidden)
    (out/"gradient_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8")
    print(f"R0 audit PASS: batch error={audit['max_abs_batch_vs_loop']:.2e}, "
          f"chain-rule error={audit['max_abs_direct_vs_vjp']:.2e}",flush=True)
    splits,indices=make_data(args)
    np.savez_compressed(out/"split_indices.npz",**{k:np.asarray(v,dtype=np.int32) for k,v in indices.items()})
    print("Data splits: "+str({k:len(v) for k,v in indices.items()})+"; official test set untouched",flush=True)
    if args.mode in ("all","classifiers"):
        # Reuse if all checkpoints exist. No unsafe partial-resume from an old optimizer state.
        complete=all((out/f"seed_{s}"/f"classifier_e{ep}.pt").exists()
                     for s in args.seeds for ep in args.checkpoints)
        if complete:
            print("All classifier checkpoints exist: reusing.",flush=True)
        else:
            train_classifiers(args,splits,out,device)
    if args.mode in ("all","predictors"):
        run_predictors(args,splits,out,device)
    print(f"\nFinished. See {out.resolve()}\n"
          "Reminder: development evidence only; no gradient substitution or GPU speedup test.",flush=True)


if __name__ == "__main__":
    main()
