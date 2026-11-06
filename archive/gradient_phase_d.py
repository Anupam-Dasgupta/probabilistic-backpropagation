"""Phase D: actual exact vs scheduled vs feedback synthetic-gradient training.

Standalone Fashion-MNIST pilot. The synthetic step deliberately omits the
DOWNSTREAM HEAD'S BACKWARD PASS and does not update head weights that step.
It DOES update upstream trunk parameters through h.backward(predicted_dLdh/B).

This is a software analogue of alternating BP/GP, *not* a reproduction of
ADA-GP's hardware results or proof that the method is faster.

Usage (Windows CMD):
  py -3.13 gradient_phase_d.py --smoke-test --device cpu --out d_smoke
  py -3.13 gradient_phase_d.py --device cuda --out d_study --seeds 0 1 2

Requires: torch, torchvision (for real Fashion-MNIST), numpy, matplotlib (plots).
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Subset, SubsetRandomSampler, TensorDataset


class SplitMLP(nn.Module):
    """Explicit gradient boundary: trunk produces hidden activation h."""

    def __init__(self, hidden: int = 128):
        super().__init__()
        self.trunk = nn.Sequential(nn.Flatten(), nn.Linear(784, hidden), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 10))

    def forward(self, x):
        h = self.trunk(x)
        return self.head(h)


class SyntheticGradient(nn.Module):
    def __init__(self, hidden: int = 128, width: int = 96):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(hidden + 10, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, hidden),
        )

    def forward(self, h, y):
        z = torch.cat([h.detach(), F.one_hot(y, 10).to(dtype=h.dtype)], dim=1)
        return self.network(z)


def set_seed(seed: int, deterministic: bool = False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = False


def device_from(request: str):
    if request == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this Python environment; check torch.cuda.is_available().')
    if request == 'auto':
        request = 'cuda' if torch.cuda.is_available() else 'cpu'
        if request == 'cpu':
            print('WARNING: CUDA unavailable; using CPU', flush=True)
    return torch.device(request)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def datasets(args):
    if args.smoke_test:
        gen = torch.Generator().manual_seed(991)
        images = torch.randn(384, 1, 28, 28, generator=gen)
        labels = torch.randint(10, (384,), generator=gen)
        data = TensorDataset(images, labels)
        return Subset(data, range(320)), Subset(data, range(320, 384))
    from torchvision import datasets as tv, transforms
    data = tv.FashionMNIST(
        root=args.data_dir, train=True, download=True,
        transform=transforms.Compose([
            transforms.ToTensor(), transforms.Normalize((0.2860,), (0.3530,)),
        ]),
    )
    # Fixed, disjoint dev split; official test set remains untouched.
    ix = torch.randperm(len(data), generator=torch.Generator().manual_seed(23117)).tolist()
    return Subset(data, ix[:50_000]), Subset(data, ix[50_000:])


def train_loader(dataset, args, seed, epoch, device):
    generator = torch.Generator().manual_seed(3_000_001 + 10_007 * seed + epoch)
    sampler = SubsetRandomSampler(range(len(dataset)), generator=generator)
    return DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=0,
                      pin_memory=(device.type == 'cuda'))


def valid_loader(dataset, args, device):
    return DataLoader(dataset, batch_size=args.eval_batch_size, shuffle=False,
                      num_workers=0, pin_memory=(device.type == 'cuda'))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = correct = 0
    ce_total = 0.0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        ce_total += F.cross_entropy(logits, y, reduction='sum').item()
        correct += int((logits.argmax(1) == y).sum().item())
        total += y.numel()
    model.train()
    return {'val_ce': ce_total / total, 'val_accuracy': correct / total}


def fresh_optimizers(model, predictor, args):
    trunk_opt = torch.optim.AdamW(model.trunk.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    head_opt = torch.optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    pred_opt = torch.optim.Adam(predictor.parameters(), lr=args.pred_lr)
    return trunk_opt, head_opt, pred_opt


def predictor_loss(pred_normed, target_normed, cosine_weight):
    mse = F.mse_loss(pred_normed, target_normed)
    good = (target_normed.norm(dim=1) > 1e-5)
    if bool(good.any()):
        cosine = F.cosine_similarity(pred_normed[good], target_normed[good], dim=1)
        return mse + cosine_weight * (1 - cosine).mean()
    return mse


def exact_step(model, predictor, optimizers, x, y, scale, args, update_predictor=True):
    """Exact head+trunk BP, then update predictor from true per-example dCE/dh.

    cross_entropy reduction='mean' yields dL/dh = per-example dCE/dh / B;
    explicitly multiply by B to train on per-example targets. The same scale
    is reversed when a synthetic gradient is injected into the trunk.
    """
    trunk_opt, head_opt, pred_opt = optimizers
    trunk_opt.zero_grad(set_to_none=True)
    head_opt.zero_grad(set_to_none=True)
    if not update_predictor:
        # Unmodified exact-backpropagation baseline: absolutely no predictor
        # forward pass or predictor optimizer step on these batches.
        logits = model.head(model.trunk(x))
        loss = F.cross_entropy(logits, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        trunk_opt.step()
        head_opt.step()
        return float(loss.item()), None, None
    h = model.trunk(x)
    h.retain_grad()
    logits = model.head(h)
    loss = F.cross_entropy(logits, y)
    loss.backward()
    true_g = h.grad.detach() * len(y)
    with torch.no_grad():
        pred_g_before = predictor(h.detach(), y) * scale
        # Aggregate normalized L2 is less vulnerable to tiny per-example norms.
        denom = true_g.square().mean().sqrt().clamp_min(scale * 0.05)
        feedback_error = ((pred_g_before - true_g).square().mean().sqrt() / denom).item()
        normalized_true = true_g / scale
    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    trunk_opt.step()
    head_opt.step()
    pred_opt.zero_grad(set_to_none=True)
    estimate = predictor(h.detach(), y)
    pred_loss = predictor_loss(estimate, normalized_true, args.cosine_weight)
    pred_loss.backward()
    nn.utils.clip_grad_norm_(predictor.parameters(), args.clip_grad)
    pred_opt.step()
    return float(loss.item()), float(pred_loss.item()), feedback_error


def synthetic_step(model, predictor, optimizers, x, y, scale, args):
    """NO exact-gradient computation and NO head backward/update in this step."""
    trunk_opt, head_opt, _ = optimizers
    trunk_opt.zero_grad(set_to_none=True)
    head_opt.zero_grad(set_to_none=True)
    h = model.trunk(x)
    with torch.no_grad():
        # Keep the full head FORWARD PASS in both methods for comparable costs,
        # but do not build its backward graph (or update its parameters).
        ce = F.cross_entropy(model.head(h.detach()), y)
        estimated_per_example = predictor(h.detach(), y) * scale
    # This is the only backward computation in a synthetic step.
    h.backward(estimated_per_example.detach() / len(y))
    if args.smoke_test or args.audit_every_step:
        assert all(p.grad is None for p in model.head.parameters()), 'Head BP ran on synthetic step!'
    nn.utils.clip_grad_norm_(model.trunk.parameters(), args.clip_grad)
    trunk_opt.step()
    return float(ce.item())


def audit_gradient_injection(device):
    """Verify a substitute h.backward(g) matches exact trunk BP when g is exact."""
    set_seed(81492)
    m = SplitMLP(hidden=16).to(device)
    xx = torch.randn(7, 1, 28, 28, device=device)
    yy = torch.randint(0, 10, (7,), device=device)
    m.zero_grad(set_to_none=True)
    h = m.trunk(xx)
    logits = m.head(h)
    loss = F.cross_entropy(logits, yy)
    true_g = torch.autograd.grad(loss, h, retain_graph=True)[0]
    loss.backward()
    true_params = [p.grad.detach().clone() for p in m.trunk.parameters()]
    m.zero_grad(set_to_none=True)
    new_h = m.trunk(xx)
    new_h.backward(true_g.detach())
    inject_params = [p.grad.detach().clone() for p in m.trunk.parameters()]
    max_err = max((a-b).abs().max().item() for a,b in zip(true_params, inject_params))
    assert max_err < (1e-5 if device.type=='cuda' else 1e-6), max_err
    assert all(p.grad is None for p in m.head.parameters()), 'Synthetic path touched head!'
    # Check per-example scaling of label targets.
    leaf = h.detach().requires_grad_(True)
    sum_loss = F.cross_entropy(m.head(leaf), yy, reduction='sum')
    raw = torch.autograd.grad(sum_loss, leaf)[0]
    per_ex_error = (raw - true_g.detach() * len(yy)).abs().max().item()
    assert per_ex_error < (1e-5 if device.type=='cuda' else 1e-6), per_ex_error
    print(f'gradient audit PASS: injection maxdiff={max_err:.2g}, per-example scaling maxdiff={per_ex_error:.2g}',flush=True)
    return {'trunk_vjp_maxabs':max_err,'per_example_scale_maxabs':per_ex_error}


def bootstrap(args, train, valid, seed, device, out):
    """Shared pretrained state. Saved once, reused for every regime."""
    path = out / f'seed_{seed}_warmup.pt'
    if path.exists() and not args.no_cache:
        try:
            s = torch.load(path, map_location='cpu', weights_only=True)
            if s['config'] == warmup_signature(args):
                print(f'  Reusing warmup checkpoint {path}', flush=True)
                return s
        except Exception as exc:
            print(f'  Cached warmup unreadable; training again: {exc}',flush=True)
    set_seed(12_345 + seed, args.deterministic)
    model = SplitMLP(args.hidden).to(device)
    predictor = SyntheticGradient(args.hidden, args.predictor_width).to(device)
    ops = fresh_optimizers(model, predictor, args)
    scale = None
    history=[]
    plain_initial_model = copy.deepcopy(model)
    plain_ops = fresh_optimizers(plain_initial_model, copy.deepcopy(predictor), args)
    warmup_train_seconds=0.0
    warmup_val_seconds=0.0
    vloader=valid_loader(valid,args,device)
    for ep in range(args.warmup_epochs):
        model.train(); predictor.train()
        sync(device); t0=time.perf_counter()
        losses=[]
        for x,y in train_loader(train,args,seed,ep,device):
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            # Calibrate a fixed scale from the first warmup exact BP batch.
            if scale is None:
                with torch.no_grad(): h0=model.trunk(x)
                lh=h0.detach().requires_grad_(True)
                ce=F.cross_entropy(model.head(lh), y,reduction='sum')
                g=torch.autograd.grad(ce,lh)[0]
                scale=max(g.square().mean().sqrt().item(),1e-6)
            ce,_,_=exact_step(model,predictor,ops,x,y,scale,args)
            losses.append(ce)
        sync(device); warmup_train_seconds+=time.perf_counter()-t0
        sync(device); vt0=time.perf_counter()
        v=evaluate(model,vloader,device)
        sync(device); warmup_val_seconds+=time.perf_counter()-vt0
        history.append({'seed':seed,'method':'warmup','epoch':ep+1,'train_ce':float(np.mean(losses)), **v})
        print(f'  seed={seed} warmup {ep+1}/{args.warmup_epochs}: val_acc={v["val_accuracy"]:.4f}',flush=True)
    # Measure a separate ordinary-BP warmup for the true exact baseline.
    # It has identical classifier initialization, minibatch order and updates,
    # but never fits a synthetic-gradient model. Without this accounting,
    # the exact baseline would be unfairly charged predictor-training costs.
    baseline_warmup_train_seconds = 0.0
    baseline_warmup_val_seconds = 0.0
    for ep in range(args.warmup_epochs):
        plain_initial_model.train()
        sync(device); bt0=time.perf_counter()
        for x,y in train_loader(train,args,seed,ep,device):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            exact_step(plain_initial_model, None, plain_ops, x, y, 1.0, args,
                       update_predictor=False)
        sync(device); baseline_warmup_train_seconds+=time.perf_counter()-bt0
        sync(device); bv0=time.perf_counter()
        evaluate(plain_initial_model,vloader,device)
        sync(device); baseline_warmup_val_seconds+=time.perf_counter()-bv0
    # Check equivalence of the classifier states after both warmups.
    max_weight_diff=max((a-b).abs().max().item() for a,b in zip(
        model.parameters(),plain_initial_model.parameters()))
    assert max_weight_diff < 1e-5, (
        f'Plain vs predictor-trained warmup classifier changed: {max_weight_diff}')
    print(f'  Warmup classifier equivalence maxdiff={max_weight_diff:.2g}; '
          f'BP-only warmup train time={baseline_warmup_train_seconds:.1f}s',flush=True)
    state={
        'baseline_warmup_train_seconds':baseline_warmup_train_seconds,
        'baseline_warmup_val_seconds':baseline_warmup_val_seconds,
        'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
        'predictor':{k:v.detach().cpu() for k,v in predictor.state_dict().items()},
        'opt_trunk':copy.deepcopy(ops[0].state_dict()),
        'opt_head':copy.deepcopy(ops[1].state_dict()),
        'opt_predictor':copy.deepcopy(ops[2].state_dict()),
        'scale':float(scale),'warmup_train_seconds':warmup_train_seconds,
        'warmup_val_seconds':warmup_val_seconds,'warmup_history':history,
        'config':warmup_signature(args),
    }
    torch.save(state,path)
    return state


def warmup_signature(args):
    return {k:getattr(args,k) for k in (
        'smoke_test','batch_size','hidden','predictor_width','warmup_epochs',
        'lr','pred_lr','weight_decay','clip_grad','cosine_weight','deterministic')}


def restore(args,state,device):
    model=SplitMLP(args.hidden).to(device)
    predictor=SyntheticGradient(args.hidden,args.predictor_width).to(device)
    model.load_state_dict(state['model']); predictor.load_state_dict(state['predictor'])
    ops=fresh_optimizers(model,predictor,args)
    for i,k in enumerate(('opt_trunk','opt_head','opt_predictor')):
        ops[i].load_state_dict(state[k])
    return model,predictor,ops


def schedule_ratio(epoch, total):
    q=int(4*epoch/max(total,1))
    return (4,3,2,1)[min(q,3)]


def update_quota(err_ema, args):
    if err_ema < args.error_low:
        return args.max_synthetic
    if err_ema < args.error_high:
        return min(2,args.max_synthetic)
    return 0


def method_masks(method, args, total_steps, matched_exact=None):
    if method != 'matched_periodic':
        return None
    if matched_exact is None:
        raise ValueError('matched_periodic requires feedback exact-step count')
    if not (0<=matched_exact<=total_steps):
        raise ValueError('Invalid matched BP count')
    result=np.zeros(total_steps,dtype=bool)
    if matched_exact:
        indices=np.floor((np.arange(matched_exact)+0.5)*total_steps/matched_exact).astype(int)
        result[indices]=True
    assert int(result.sum())==matched_exact
    return result


def dump_csv(path,records):
    if not records: return
    path.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for row in records for k in row.keys()))
    with path.open('w',newline='',encoding='utf-8') as fp:
        w=csv.DictWriter(fp,fieldnames=fields)
        w.writeheader();w.writerows(records)


def run_method(args,state,train,valid,seed,device,method,out,matched_exact=None):
    model,predictor,opts=restore(args,state,device)
    n_per_epoch=math.ceil(len(train)/args.batch_size)
    total_steps=n_per_epoch*args.epochs
    mask=method_masks(method,args,total_steps,matched_exact)
    vloader=valid_loader(valid,args,device)
    scale=state['scale']
    epoch_rows=[]; train_seconds=0.0; val_seconds=0.0
    bp_count=gp_count=0
    ema=None; syn_left=0
    global_step=0
    wtrain = (state['baseline_warmup_train_seconds'] if method=='exact' else state['warmup_train_seconds'])
    wval = (state['baseline_warmup_val_seconds'] if method=='exact' else state['warmup_val_seconds'])
    warmup_reached = any(r['val_accuracy'] >= args.target_accuracy for r in state['warmup_history'])
    first_target_epoch=args.warmup_epochs if warmup_reached else None
    first_target_seconds=wtrain+wval if warmup_reached else None
    for ep in range(args.epochs):
        model.train();predictor.train()
        sync(device); t0=time.perf_counter()
        ce_vals=[]; pred_vals=[]; observed_errors=[]
        exact_ep=synthetic_ep=0
        for batch_idx,(x,y) in enumerate(train_loader(train,args,seed,ep+args.warmup_epochs,device)):
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            if method=='exact': exact=True
            elif method=='scheduled':
                q=schedule_ratio(ep,args.epochs)
                exact=(batch_idx%(q+1)==0)
            elif method=='matched_periodic':
                exact=bool(mask[global_step])
            elif method=='feedback':
                exact=(syn_left==0)
            else: raise ValueError(method)
            if exact:
                loss,pred_loss,err=exact_step(model,predictor,opts,x,y,scale,args,
                                              update_predictor=(method!='exact'))
                if pred_loss is not None:pred_vals.append(pred_loss)
                if err is not None:observed_errors.append(err)
                bp_count+=1;exact_ep+=1
                if method=='feedback':
                    ema=err if ema is None else (1-args.ema_alpha)*ema+args.ema_alpha*err
                    syn_left=update_quota(ema,args)
            else:
                loss=synthetic_step(model,predictor,opts,x,y,scale,args)
                gp_count+=1;synthetic_ep+=1
                if method=='feedback':syn_left-=1
            ce_vals.append(loss)
            global_step+=1
        sync(device);dt=time.perf_counter()-t0
        train_seconds+=dt
        sync(device);vt0=time.perf_counter()
        v=evaluate(model,vloader,device)
        sync(device);val_seconds+=time.perf_counter()-vt0
        overall_time=(wtrain+wval+train_seconds+val_seconds)
        epoch_row={'seed':seed,'method':method,'epoch':args.warmup_epochs+ep+1,
                   'postwarm_epoch':ep+1,'train_ce':float(np.mean(ce_vals)),
                   'pred_loss_exact_mean':float(np.mean(pred_vals)) if pred_vals else '',
                   'exact_gradient_error_mean':float(np.mean(observed_errors)) if observed_errors else '',
                   'risk_ema':ema if ema is not None else '',
                   'exact_batches_epoch':exact_ep,'synthetic_batches_epoch':synthetic_ep,
                   'exact_batches_total':bp_count,'synthetic_batches_total':gp_count,
                   'exact_fraction_total':bp_count/(bp_count+gp_count),
                   'train_seconds_cum':train_seconds,'eval_seconds_cum':val_seconds,
                   'wall_seconds_cum_with_warmup':overall_time,**v}
        epoch_rows.append(epoch_row)
        if first_target_epoch is None and v['val_accuracy'] >= args.target_accuracy:
            first_target_epoch=epoch_row['epoch']
            first_target_seconds=overall_time
        print(f'  seed={seed} {method:<16} epoch {ep+1}/{args.epochs}: '
              f'acc={v["val_accuracy"]:.4f} exact={exact_ep}/{n_per_epoch} '
              f'error={np.mean(observed_errors) if observed_errors else float("nan"):.3f} '
              f'train_s={train_seconds:.1f}',flush=True)
        dump_csv(out/'epochs.csv', getattr(args,'_global_epochs',[])+epoch_rows)
    final={
        'seed':seed,'method':method,'val_accuracy':epoch_rows[-1]['val_accuracy'],
        'val_ce':epoch_rows[-1]['val_ce'], 'best_val_accuracy':max(r['val_accuracy'] for r in epoch_rows),
        'exact_batches_postwarm':bp_count,'synthetic_batches_postwarm':gp_count,
        'exact_fraction_postwarm':bp_count/total_steps,'total_batches_postwarm':total_steps,
        'warmup_exact_batches':args.warmup_epochs*n_per_epoch,
        'total_exact_batches':bp_count+args.warmup_epochs*n_per_epoch,
        'exact_fraction_including_warmup':(bp_count+args.warmup_epochs*n_per_epoch)/((args.warmup_epochs+args.epochs)*n_per_epoch),
        'postwarm_train_seconds':train_seconds,
        'warmup_train_seconds':wtrain,
        'warmup_val_seconds':wval,
        'total_train_seconds':train_seconds+wtrain,
        'total_wall_seconds_with_validation':overall_time,
        'target_accuracy':args.target_accuracy,'first_target_epoch':first_target_epoch or '',
        'seconds_to_target_with_validation':first_target_seconds if first_target_seconds is not None else '',
    }
    torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},
                'predictor':{k:v.detach().cpu() for k,v in predictor.state_dict().items()},
                'config':warmup_signature(args),'final':final},
               out/f'seed_{seed}_{method}_final.pt')
    return epoch_rows,final


def aggregate_summary(final_rows, out):
    fields = ('val_accuracy','best_val_accuracy','exact_fraction_postwarm',
              'total_train_seconds','total_wall_seconds_with_validation',
              'seconds_to_target_with_validation')
    grouped=[]
    for method in sorted(set(r['method'] for r in final_rows)):
        rows=[r for r in final_rows if r['method']==method]
        result={'method':method,'n_seeds':len(rows)}
        for field in fields:
            values=[float(r[field]) for r in rows if r[field]!='']
            result[field+'_n']=len(values)
            result[field+'_mean']=float(np.mean(values)) if values else ''
            result[field+'_sd']=float(np.std(values,ddof=1)) if len(values)>=2 else ''
        grouped.append(result)
    dump_csv(out/'summary_aggregate.csv',grouped)
    pair=[]
    for seed in sorted(set(r['seed'] for r in final_rows)):
        matches={r['method']:r for r in final_rows if r['seed']==seed}
        if 'feedback' not in matches:continue
        for baseline in ('exact','scheduled','matched_periodic'):
            if baseline not in matches:continue
            f,b=matches['feedback'],matches[baseline]
            pair.append({'seed':seed,'versus':baseline,
                'feedback_minus_baseline_accuracy':f['val_accuracy']-b['val_accuracy'],
                'feedback_minus_baseline_train_seconds':f['total_train_seconds']-b['total_train_seconds'],
                'feedback_minus_baseline_exact_fraction':f['exact_fraction_postwarm']-b['exact_fraction_postwarm'],
                'feedback_minus_baseline_wall_seconds':f['total_wall_seconds_with_validation']-b['total_wall_seconds_with_validation']})
    dump_csv(out/'paired_vs_baselines.csv',pair)


def plots(rows,out):
    try: import matplotlib
    except ImportError:return
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    methods=sorted(set(x['method'] for x in rows))
    seeds=sorted(set(x['seed'] for x in rows))
    for metric,file,ylabel in [('val_accuracy','accuracy.png','Validation accuracy'),
                               ('wall_seconds_cum_with_warmup','time_by_epoch.png','Elapsed training + evaluation time (s)')]:
        fig,ax=plt.subplots(figsize=(8,4.5))
        for method in methods:
            groups=[]
            for seed in seeds:
                data=sorted((r for r in rows if r['method']==method and r['seed']==seed),key=lambda d:d['epoch'])
                if data:groups.append([float(x[metric]) for x in data])
            if not groups:continue
            if not all(len(x)==len(groups[0]) for x in groups):continue
            y=np.asarray(groups)
            x=np.asarray([int(r['epoch']) for r in sorted((r for r in rows if r['method']==method and r['seed']==seeds[0]),key=lambda d:d['epoch'])])
            if len(x)!=y.shape[1]: x=np.arange(y.shape[1])
            ax.plot(x,y.mean(0),label=method)
            if y.shape[0]>=2:ax.fill_between(x,y.mean(0)-y.std(0,ddof=1),y.mean(0)+y.std(0,ddof=1),alpha=.12)
        ax.set_xlabel('Total classifier epochs');ax.set_ylabel(ylabel);ax.legend()
        ax.grid(True,alpha=.2);fig.tight_layout();fig.savefig(out/file,dpi=150);plt.close(fig)


def parse():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--out',default='phase_d_study')
    p.add_argument('--data-dir',default='data')
    p.add_argument('--device',choices=['auto','cpu','cuda'],default='auto')
    p.add_argument('--seeds',nargs='+',type=int,default=[0,1,2])
    p.add_argument('--modes',nargs='+',choices=['exact','scheduled','feedback'],default=['exact','scheduled','feedback'])
    p.add_argument('--no-matched',action='store_true',help='Do not add BP-budget-matched periodic baseline (requires feedback)')
    p.add_argument('--smoke-test',action='store_true')
    p.add_argument('--deterministic',action='store_true')
    p.add_argument('--no-cache',action='store_true',help='Recompute warmup instead of reusing saved warmup state')
    p.add_argument('--audit-every-step',action='store_true')
    p.add_argument('--warmup-epochs',type=int,default=4)
    p.add_argument('--epochs',type=int,default=16,help='Post-warmup training epochs')
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--eval-batch-size',type=int,default=512)
    p.add_argument('--hidden',type=int,default=128)
    p.add_argument('--predictor-width',type=int,default=96)
    p.add_argument('--lr',type=float,default=1e-3)
    p.add_argument('--pred-lr',type=float,default=1e-3)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--clip-grad',type=float,default=5.0)
    p.add_argument('--cosine-weight',type=float,default=0.1)
    p.add_argument('--ema-alpha',type=float,default=0.2)
    p.add_argument('--error-low',type=float,default=0.7)
    p.add_argument('--error-high',type=float,default=1.1)
    p.add_argument('--max-synthetic',type=int,default=4)
    p.add_argument('--target-accuracy',type=float,default=0.88)
    args=p.parse_args()
    if args.smoke_test:
        args.warmup_epochs=1;args.epochs=2;args.batch_size=64;args.seeds=[0];args.target_accuracy=0.10
    if args.warmup_epochs<1 or args.epochs<1 or args.batch_size<1 or args.max_synthetic<0:
        p.error('warmup-epochs, epochs, batch-size must be >=1; max-synthetic must be >=0')
    if not 0<args.ema_alpha<=1 or not 0<=args.error_low<=args.error_high:
        p.error('Invalid feedback EMA/threshold configuration')
    return args


def main():
    args=parse()
    if args.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    device=device_from(args.device)
    print(f'Phase D device={device} torch={torch.__version__} modes={args.modes}',flush=True)
    audit=audit_gradient_injection(device)
    train,valid=datasets(args)
    print(f'dataset train={len(train)} val={len(valid)}; validation/test data never fit predictors',flush=True)
    config={k:v for k,v in vars(args).items() if not k.startswith('_')}
    config['torch_version']=torch.__version__
    config['device_used']=str(device)
    config['cuda_gpu']=torch.cuda.get_device_name(0) if device.type=='cuda' else None
    config['audit']=audit
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    all_rows=[];all_final=[]
    for seed in args.seeds:
        print(f'\n[seed {seed}] shared warmup',flush=True)
        state=bootstrap(args,train,valid,seed,device,out)
        # Matched-periodic baseline needs the realized feedback BP count, so
        # the feedback run is executed before it. Execution order does not
        # affect initialization, optimizer state, or training permutations.
        modes=list(args.modes)
        if 'feedback' in modes and not args.no_matched:
            modes.append('matched_periodic')
        feedback_exact=None
        for method in modes:
            print(f'\n[seed {seed}] mode={method}',flush=True)
            rows,final=run_method(args,state,train,valid,seed,device,method,out,
                                  matched_exact=feedback_exact if method=='matched_periodic' else None)
            all_rows.extend(rows);all_final.append(final)
            args._global_epochs=all_rows
            dump_csv(out/'epochs.csv',all_rows)
            dump_csv(out/'summary.csv',all_final)
            aggregate_summary(all_final,out)
            if method=='feedback': feedback_exact=final['exact_batches_postwarm']
        plots(all_rows,out)
    print(f'\nSaved to {out.resolve()}',flush=True)
    print('READ summary.csv. Speedups must include predictor-training, warmup and validation costs.',flush=True)
    print('This is an end-to-end software pilot; no claim of ADA-GP reproduction.',flush=True)


if __name__=='__main__':
    main()
