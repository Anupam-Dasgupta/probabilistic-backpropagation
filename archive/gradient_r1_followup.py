"""R1+ follow-up: long gradient-predictor training, magnitude correction, risk & drift.

Requires gradient_robust_r1.py next to this file and the ORIGINAL R1 study folder
containing seed_<n>/classifier_e<epoch>.pt and config.json.

This is an OFFLINE diagnostic, not a synthetic-gradient training or speedup claim.
Run --smoke-test to generate a standalone, tiny synthetic test and check the pipeline.

Windows CMD:
    py -3.13 gradient_r1_followup.py --smoke-test --device cpu
    py -3.13 gradient_r1_followup.py --r1-root r1_study --device cuda --out r1_followup
    py -3.13 gradient_r1_followup.py --r1-root r1_study --device cuda --seeds 0 --checkpoints 10
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import TensorDataset
from torch.func import functional_call

import gradient_robust_r1 as r1


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--r1-root', type=Path, default=Path('r1_study'), help='EXTRACTED original R1 output folder')
    p.add_argument('--out', type=Path, default=Path('r1_followup'))
    p.add_argument('--data-dir', default='./data')
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--seeds', type=r1.parse_ints, default=[0, 1, 2])
    p.add_argument('--checkpoints', type=r1.parse_ints, default=[10, 20, 40, 60])
    p.add_argument('--pred-epochs', type=int, default=120)
    p.add_argument('--pred-snapshots', type=r1.parse_ints, default=[20, 40, 60, 80, 100, 120])
    p.add_argument('--mag-epochs', type=int, default=40)
    p.add_argument('--risk-epochs', type=int, default=25)
    p.add_argument('--predictor-batch-size', type=int, default=512)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--pred-lr', type=float, default=1e-3)
    p.add_argument('--cosine-weight', type=float, default=0.25)
    p.add_argument('--risk-cosine-threshold', type=float, default=0.8)
    p.add_argument('--risk-relative-threshold', type=float, default=2.0)
    p.add_argument('--exact-budgets', type=r1.parse_ints, default=[10, 20, 30])
    p.add_argument('--probe-lr', type=float, default=0.05)
    p.add_argument('--probe-batches', type=int, default=2)
    p.add_argument('--deterministic', action='store_true')
    p.add_argument('--smoke-test', action='store_true')
    return p.parse_args()


def save_csv(file, rows):
    r1.write_csv(file, rows)


def select_split(data, idx):
    return {k: v[idx] for k, v in data.items()}


def to_state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def device_for(args):
    if args.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    return r1.device_from_arg(args.device)


def configure(args):
    if args.smoke_test:
        args.r1_root = args.out / '_smoke_original_r1'
        args.seeds = [0]
        args.checkpoints = [1, 2]
        args.pred_epochs = 3
        args.pred_snapshots = [1, 2, 3]
        args.mag_epochs = 3
        args.risk_epochs = 3
        args.batch_size = 64
        args.predictor_batch_size = 64
        args.probe_batches = 1
    args.checkpoints = sorted(args.checkpoints)
    args.pred_snapshots = sorted(args.pred_snapshots)
    if (not args.checkpoints or min(args.checkpoints) <= 0 or
        not args.pred_snapshots or min(args.pred_snapshots) <= 0 or
        max(args.pred_snapshots) > args.pred_epochs or
        min([args.pred_epochs, args.mag_epochs, args.risk_epochs,
             args.batch_size, args.predictor_batch_size]) <= 0 or
        any(b < 0 or b > 100 for b in args.exact_budgets) or
        args.probe_lr <= 0 or args.probe_batches < 0):
        raise ValueError('Invalid epoch, budget, batch-size, or learning-rate settings')


def make_smoke_original(args, device):
    root = args.r1_root
    root.mkdir(parents=True, exist_ok=True)
    config = {'experiment': {'hidden': 128, 'pred_width': 192, 'split_seed': 20261025,
                             'smoke_test': True, 'seeds': [0], 'checkpoints': [1, 2],
                             'min_norm_ratio': 1e-3}}
    (root / 'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    ra = argparse.Namespace(hidden=128, split_seed=20261025, smoke_test=True,
                            deterministic=args.deterministic, seeds=[0], class_epochs=2,
                            checkpoints=[1, 2], class_lr=1e-3, batch_size=64)
    splits, _ = r1.make_data(ra)
    if not all((root / 'seed_0' / f'classifier_e{e}.pt').exists() for e in [1, 2]):
        r1.train_classifiers(ra, splits, root, device)
    return splits


def get_original(args, device):
    if args.smoke_test:
        splits = make_smoke_original(args, device)
    else:
        if not (args.r1_root / 'config.json').is_file():
            raise FileNotFoundError(f'Original config missing: {args.r1_root / "config.json"}. '
                                    'Pass --r1-root to the EXTRACTED R1 folder.')
        cfg = json.loads((args.r1_root / 'config.json').read_text(encoding='utf-8'))['experiment']
        ns = argparse.Namespace(split_seed=cfg['split_seed'], data_dir=args.data_dir,
                                smoke_test=False)
        splits, _ = r1.make_data(ns)
        index_file = args.r1_root / 'split_indices.npz'
        if not index_file.exists():
            raise FileNotFoundError('Missing original split_indices.npz: refusing to reconstruct splits blindly.')
        with np.load(index_file) as indices:
            # Verify that source train/val/dev maps onto exactly the ORIGINAL R1 samples.
            _, regenerated = r1.make_data(ns)
            for key in ('train', 'val', 'dev'):
                if not np.array_equal(indices[key], np.asarray(regenerated[key])):
                    raise RuntimeError(f'Original R1 split mismatch in {key}; aborting.')
        if cfg.get('smoke_test'):
            raise RuntimeError('This source was a smoke test; use --smoke-test instead.')
    cfg = json.loads((args.r1_root / 'config.json').read_text(encoding='utf-8'))['experiment']
    if any(s not in cfg['seeds'] for s in args.seeds):
        raise ValueError('Requested seed absent from source R1')
    if any(e not in cfg['checkpoints'] for e in args.checkpoints):
        raise ValueError('Requested checkpoint absent from source R1')
    for s in args.seeds:
        for e in args.checkpoints:
            path = args.r1_root / f'seed_{s}' / f'classifier_e{e}.pt'
            if not path.is_file():
                raise FileNotFoundError(path)
    return cfg, splits


def get_model(args, cfg, seed, epoch, device):
    file = args.r1_root / f'seed_{seed}' / f'classifier_e{epoch}.pt'
    stored = torch.load(file, map_location='cpu', weights_only=True)
    model = r1.Classifier(hidden=int(cfg['hidden'])).to(device)
    model.load_state_dict(stored['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def extract(model, split, args, device):
    data = r1.extract_gradient_data(model, split, device, args.batch_size)
    # The first-layer parameter gradient norm includes the input L2 norm.
    norms = []
    for x, _ in r1.data_loader(split, args.batch_size, 0, False, device):
        norms.append(x.flatten(1).norm(dim=-1).cpu())
    data['input_norm'] = torch.cat(norms)
    return data


def train_gradient_predictor(args, method, source, train, val, scale, floor,
                             class_means, hidden, width, device, out, seed, epoch):
    file = out / f'{method}_long.pt'
    if file.exists():
        saved = torch.load(file, map_location='cpu', weights_only=True)
        net = r1.Predictor(hidden, width, gaussian=(method == 'gaussian')).to(device)
        net.load_state_dict(saved['state'])
        return net, saved['best_epoch']
    r1.set_seed(15000 + seed * 1000 + epoch * 10 + (0 if method == 'cosine_mse' else 1),args.deterministic)
    net = r1.Predictor(hidden, width, gaussian=(method == 'gaussian')).to(device)
    optim = torch.optim.AdamW(net.parameters(), lr=args.pred_lr)
    dataset = TensorDataset(train['h'],train['y'],train['g']/scale)
    floor_scaled = floor / scale
    best = -float('inf')
    best_epoch = None
    best_state = None
    curve = []
    training_ckpt = out / f'{method}_training_progress.pt'
    starting_epoch = 1
    if training_ckpt.exists():
        previous = torch.load(training_ckpt, map_location='cpu', weights_only=True)
        net.load_state_dict(previous['state'])
        optim.load_state_dict(previous['optimizer'])
        # Moving optimizer state to device supports CPU<>GPU resumption.
        for group in optim.state.values():
            for key, val in group.items():
                if torch.is_tensor(val):
                    group[key] = val.to(device)
        starting_epoch = int(previous['last_epoch']) + 1
        curve = previous['curve']
        best = float(previous['best_score'])
        best_epoch = previous['best_epoch']
        best_state = previous['best_state']
        print(f'  RESUME {method} at epoch {starting_epoch}', flush=True)
    for pe in range(starting_epoch,args.pred_epochs+1):
        # Re-seed shuffling per epoch so continuation reproduces the same data ordering.
        loader = r1.data_loader(dataset, args.predictor_batch_size,
                                300+seed*10000+epoch*1000+pe, True, device)
        net.train()
        total_loss = 0.0
        for h,y,g in loader:
            h,y,g = h.to(device),y.to(device),g.to(device)
            optim.zero_grad(set_to_none=True)
            mu,logvar = net(h,y)
            loss = r1.loss_for(method,mu,g,logvar,args.cosine_weight,floor_scaled)
            if not torch.isfinite(loss):
                raise RuntimeError(f'Non-finite predictor loss at {method} seed={seed} checkpoint={epoch}')
            loss.backward()
            optim.step()
            total_loss += loss.detach().item()*len(y)
        if pe not in args.pred_snapshots:
            continue
        p,sigma = r1.predict_all(method,net,val,class_means,scale,device,args.predictor_batch_size)
        metrics = r1.metric_dict(val['g'],p,floor,val['correct'],sigma)
        score = metrics['cosine_mean'] if method=='cosine_mse' else -metrics['normalized_rmse']
        curve.append({'seed':seed,'classifier_epoch':epoch,'method':method,
                      'predictor_epoch':pe,'train_loss':total_loss/len(dataset),**metrics})
        if math.isfinite(score) and score > best:
            best, best_epoch, best_state = score,pe,to_state(net)
        # Atomic progress checkpoint; interrupted runs resume from the last snapshot.
        progress={'state':to_state(net),'optimizer':optim.state_dict(),
                  'last_epoch':pe,'curve':curve,'best_score':best,
                  'best_epoch':best_epoch,'best_state':best_state}
        pending = training_ckpt.with_suffix('.tmp')
        torch.save(progress, pending)
        pending.replace(training_ckpt)
        print(f'  {method}: {pe:3d}/{args.pred_epochs} train={total_loss/len(dataset):.4f} '
              f'val_cos={metrics["cosine_mean"]:.4f} val_nRMSE={metrics["normalized_rmse"]:.4f}',flush=True)
    if best_state is None:
        raise RuntimeError(f'No valid best checkpoint for {method}')
    save_csv(out / f'{method}_learning_curve.csv',curve)
    torch.save({'state':best_state,'best_epoch':best_epoch,'method':method},file)
    training_ckpt.unlink(missing_ok=True)
    net.load_state_dict(best_state)
    return net,best_epoch


@torch.no_grad()
def pred_on(net, data, scale, device, batch):
    net.eval()
    mu_list, sigma_list = [], []
    loader = r1.data_loader(TensorDataset(data['h'],data['y']),batch,0,False,device)
    for h,y in loader:
        h,y=h.to(device),y.to(device)
        mean,logvar = net(h,y)
        mu_list.append(mean.cpu()*scale)
        if logvar is not None:
            sigma_list.append(torch.exp(0.5*logvar).cpu()*scale)
    return torch.cat(mu_list), (torch.cat(sigma_list) if sigma_list else None)


def alpha_fit(pred, true, mask=None):
    if mask is not None:
        pred,true=pred[mask],true[mask]
    denom = (pred*pred).sum().item()
    alpha = (pred*true).sum().item()/max(denom,1e-15)
    return max(0.0,min(alpha,10.0))


class MagnitudeNet(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(hidden+10+2,96),nn.ReLU(),
                               nn.Linear(96,96),nn.ReLU(),nn.Linear(96,1))

    def forward(self,h,y,base_norm,confidence,scale):
        features=torch.cat([h,F.one_hot(y,10).to(h.dtype),
                            torch.log1p(base_norm.clamp_min(0)/scale).unsqueeze(-1),
                            confidence.unsqueeze(-1)],dim=-1)
        # Predict log(1 + norm / scale), constrained nonnegative for stability.
        return F.softplus(self.net(features).squeeze(-1))


@torch.no_grad()
def mag_predict(net,data,raw,scale,maxnorm,device,batch):
    net.eval()
    norms=[]
    for i in range(0,len(raw),batch):
        sl=slice(i,i+batch)
        h=data['h'][sl].to(device)
        y=data['y'][sl].to(device)
        conf=data['confidence'][sl].to(device)
        pn=raw[sl].norm(dim=-1).to(device)
        pred_log=net(h,y,pn,conf,scale)
        pred_norm=torch.expm1(pred_log.clamp(0,15))*scale
        norms.append(pred_norm.clamp(0,maxnorm).cpu())
    return torch.cat(norms)


def renormalize(raw, norms):
    # Near-zero predictors have no trustworthy direction; retain zero.
    return F.normalize(raw,dim=-1,eps=1e-12)*norms[:,None]


def fit_magnitude_net(args,train,val,train_pred,val_pred,scale,out,hidden,device,seed,epoch):
    file=out/'magnitude.pt'
    norm_cap=float(torch.quantile(train['g'].norm(dim=-1),0.999).clamp_min(1e-9))
    if file.exists():
        saved=torch.load(file,map_location='cpu',weights_only=True)
        net=MagnitudeNet(hidden).to(device)
        net.load_state_dict(saved['state'])
        return net,float(saved['norm_cap']),int(saved['best_epoch'])
    r1.set_seed(22000+seed*1000+epoch,args.deterministic)
    net=MagnitudeNet(hidden).to(device)
    optimizer=torch.optim.AdamW(net.parameters(),lr=1e-3)
    dataset=TensorDataset(train['h'],train['y'],train['confidence'],
                          train_pred.norm(dim=-1),
                          torch.log1p(train['g'].norm(dim=-1)/scale))
    loader=r1.data_loader(dataset,args.predictor_batch_size,seed+epoch+901,True,device)
    best=float('inf')
    best_epoch=None
    state=None
    curves=[]
    for ep in range(1,args.mag_epochs+1):
        net.train()
        acc=0.0
        for h,y,c,pn,target in loader:
            h,y,c,pn,target=h.to(device),y.to(device),c.to(device),pn.to(device),target.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits=net(h,y,pn,c,scale)
            loss=F.smooth_l1_loss(logits,target)
            loss.backward()
            optimizer.step()
            acc+=float(loss.detach())*len(y)
        if ep == 1 or ep % 5 == 0 or ep == args.mag_epochs:
            m=mag_predict(net,val,val_pred,scale,norm_cap,device,args.predictor_batch_size)
            hybrid=renormalize(val_pred,m)
            nrmse=r1.metric_dict(val['g'],hybrid,1e-10)['normalized_rmse']
            curves.append({'magnitude_epoch':ep,'train_log_norm_loss':acc/len(dataset),
                           'val_hybrid_nrmse':nrmse})
            if nrmse < best:
                best,best_epoch,state=nrmse,ep,to_state(net)
    if state is None:
        raise RuntimeError('Magnitude training failed')
    net.load_state_dict(state)
    save_csv(out/'magnitude_learning_curve.csv',curves)
    torch.save({'state':state,'best_epoch':best_epoch,'norm_cap':norm_cap},file)
    return net,norm_cap,best_epoch


@torch.no_grad()
def head_features(model,data,device,batch):
    # Every feature can be calculated from a normal forward pass, no true-gradient inputs.
    conf=[]; ent=[]
    for h in data['h'].split(batch):
        logits=model.head(h.to(device))
        probs=logits.softmax(-1)
        conf.append(probs.max(-1).values.cpu())
        ent.append((-(probs*(probs.clamp_min(1e-12)).log()).sum(-1)).cpu())
    return torch.cat(conf),torch.cat(ent)


def risk_features(data, pred, means, entropy):
    baseline=means[data['y']]
    dis=1-F.cosine_similarity(pred,baseline,dim=-1,eps=1e-12)
    pnorm=pred.norm(dim=-1)
    feats=torch.cat([data['h'],F.one_hot(data['y'],10).float(),
                     torch.log1p(pnorm).unsqueeze(1),
                     dis.unsqueeze(1),
                     data['confidence'].unsqueeze(1),
                     entropy.unsqueeze(1)],-1)
    return feats, {'disagreement':dis,
                   'small_pred_norm':-pnorm,
                   'low_confidence':-data['confidence']}


def unsafe_labels(data, pred, floor, cosine_threshold, relative_threshold):
    norm=data['g'].norm(dim=-1)
    valid=norm>floor
    cosine=F.cosine_similarity(pred,data['g'],dim=-1,eps=1e-12)
    rel=(pred-data['g']).norm(dim=-1)/norm.clamp_min(floor)
    poor_cos=valid & (cosine < cosine_threshold)
    big_error=rel > relative_threshold
    # Combines direction and magnitude concerns; pre-specified thresholds.
    unsafe=poor_cos | big_error
    return unsafe.float(), {'direction_failure':poor_cos.float(),
                            'magnitude_failure':big_error.float(),
                            'direction_valid':valid,'cosine':cosine,'relative_error':rel}


class RiskNet(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(hidden+14,128),nn.ReLU(),
                               nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x):
        return self.net(x).squeeze(-1)


def risk_probs(net,features,device,batch):
    with torch.no_grad():
        net.eval()
        return torch.cat([net(x.to(device)).sigmoid().cpu()
                          for x in features.split(batch)])


def fit_risk(args,source_risk,source_val,pred_risk,pred_val,
             means,source_model,scale,floor,hidden,device,out,seed,epoch):
    file=out/'risk_model.pt'
    _,er=head_features(source_model,source_risk,device,args.predictor_batch_size)
    _,ev=head_features(source_model,source_val,device,args.predictor_batch_size)
    fr,_=risk_features(source_risk,pred_risk,means,er)
    fv,_=risk_features(source_val,pred_val,means,ev)
    target,_=unsafe_labels(source_risk,pred_risk,floor,args.risk_cosine_threshold,args.risk_relative_threshold)
    if file.exists():
        saved=torch.load(file,map_location='cpu',weights_only=True)
        net=RiskNet(hidden).to(device);net.load_state_dict(saved['state'])
        return net, float(target.mean())
    r1.set_seed(33000+seed*1000+epoch,args.deterministic)
    net=RiskNet(hidden).to(device)
    opt=torch.optim.AdamW(net.parameters(),lr=1e-3)
    dataset=TensorDataset(fr,target)
    loader=r1.data_loader(dataset,args.predictor_batch_size,seed+epoch+311,True,device)
    best=float('inf');best_state=None;best_ep=0
    # Calibration evaluation split is disjoint from risk fitting; selection uses Brier.
    val_target,_=unsafe_labels(source_val,pred_val,floor,args.risk_cosine_threshold,args.risk_relative_threshold)
    pos=float(target.mean())
    for ep in range(1,args.risk_epochs+1):
        net.train()
        for x,y in loader:
            x,y=x.to(device),y.to(device)
            opt.zero_grad(set_to_none=True)
            loss=F.binary_cross_entropy_with_logits(net(x),y)
            loss.backward();opt.step()
        if ep==1 or ep%5==0 or ep==args.risk_epochs:
            pred=risk_probs(net,fv,device,args.predictor_batch_size)
            brier=float(((pred-val_target)**2).mean())
            if brier < best:
                best=brier;best_ep=ep;best_state=to_state(net)
    net.load_state_dict(best_state)
    torch.save({'state':best_state,'best_epoch':best_ep,'risk_positive_fraction':pos},file)
    return net,pos


def auc_score(labels, scores):
    yy=np.asarray(labels,dtype=np.int32)
    ss=np.asarray(scores,dtype=np.float64)
    n1=int(yy.sum());n0=len(yy)-n1
    if n1==0 or n0==0: return float('nan')
    order=np.argsort(ss,kind='mergesort')
    sorted_scores=ss[order]
    ranks=np.empty(len(ss),dtype=np.float64)
    start=0
    while start<len(ss):
        end=start+1
        while end<len(ss) and sorted_scores[end]==sorted_scores[start]:end+=1
        ranks[order[start:end]]=(start+end+1)/2.0
        start=end
    return float((ranks[yy==1].sum()-n1*(n1+1)/2)/(n1*n0))


def norm_and_direction_metrics(data,pred,floor):
    out=r1.metric_dict(data['g'],pred,floor,data['correct'])
    true=data['g']; pnorm=pred.norm(dim=-1);tnorm=true.norm(dim=-1)
    delta=(true-pred)
    mask=(data['h']>0).float() # ReLU gate for first linear layer
    original=true*mask; approximate=pred*mask
    denom=(original.square().sum(-1)*data['input_norm'].square()).sum().clamp_min(1e-20)
    num=((original-approximate).square().sum(-1)*data['input_norm'].square()).sum()
    out['first_linear_weight_gradient_nrmse']=float(torch.sqrt(num/denom))
    valid=original.norm(dim=-1)>floor
    out['first_linear_weight_direction_cosine']=float(
        F.cosine_similarity(original[valid],approximate[valid],dim=-1).mean()) if valid.any() else float('nan')
    out['norm_ratio_median_valid']=float((pnorm[tnorm>floor]/tnorm[tnorm>floor]).median()) if (tnorm>floor).any() else float('nan')
    return out


def risk_report(args, data, pred, gaussian_sigma, means, risk_net, model, floor,device,seed,source_epoch,target_epoch):
    _,entropy=head_features(model,data,device,args.predictor_batch_size)
    feats,baselines=risk_features(data,pred,means,entropy)
    learned=risk_probs(risk_net,feats,device,args.predictor_batch_size)
    labels,detail=unsafe_labels(data,pred,floor,args.risk_cosine_threshold,args.risk_relative_threshold)
    if gaussian_sigma is not None:
        baselines['gaussian_sigma']=gaussian_sigma.squeeze(-1)
    baselines['learned_risk']=learned
    gen=torch.Generator().manual_seed(40000+seed*1000+source_epoch+target_epoch)
    baselines['random']=torch.rand(len(labels),generator=gen)
    y=labels.numpy();n=len(y);positives=int(labels.sum())
    ranking=[];selections=[]
    for method,score in baselines.items():
        score=np.asarray(score,dtype=np.float64)
        ranking.append({'seed':seed,'source_epoch':source_epoch,'target_epoch':target_epoch,
                        'method':method,'n_examples':n,'n_unsafe':positives,
                        'unsafe_fraction':positives/n,
                        'risk_auc':auc_score(y,score),
                        'spearman_vs_directional_error':r1.spearman(score,1-detail['cosine'])})
        # Stable exact top-k for every strategy. Never use labels for selecting batches.
        order=np.argsort(-score,kind='mergesort')
        for budget in args.exact_budgets:
            k=round(n*budget/100)
            selected=np.zeros(n,dtype=bool);selected[order[:k]]=True
            kept=~selected
            v=detail['direction_valid'].numpy() & kept
            selections.append({'seed':seed,'source_epoch':source_epoch,'target_epoch':target_epoch,
               'method':method,'exact_budget_percent':budget,'realized_budget':k/n,
               'unsafe_recall':float(y[selected].sum()/positives) if positives else float('nan'),
               'unsafe_rate_among_kept':float(y[kept].mean()) if kept.any() else float('nan'),
               'kept_cosine_valid_mean':float(detail['cosine'].numpy()[v].mean()) if v.any() else float('nan'),
               'n_kept':int(kept.sum())})
    return ranking,selections


def probe_sgd(args, classifier, selected_source, methods, source_scale,device,seed,src_ep,tgt_ep):
    """Diagnostic: virtual ONE-STEP SGD update to first trunk layer, not actual training."""
    results=[]
    if args.probe_batches==0:return results
    classifier.eval()
    for p in classifier.head.parameters():p.requires_grad_(False)
    params=dict(classifier.trunk.named_parameters())
    for p in params.values():p.requires_grad_(True)
    for i,(x,y) in enumerate(r1.data_loader(selected_source,args.batch_size,0,False,device)):
        if i>=args.probe_batches:break
        x,y=x.to(device),y.to(device)
        h=classifier.trunk(x)
        logits=classifier.head(h)
        base=float(F.cross_entropy(logits,y).detach())
        g=torch.autograd.grad(F.cross_entropy(logits,y,reduction='sum'),h,retain_graph=True)[0]
        predictions={'exact':g.detach(),'zero':torch.zeros_like(g)}
        for name,fn in methods.items():predictions[name]=fn(h.detach(),y).detach()
        for name,v in predictions.items():
            trunk_grad=torch.autograd.grad(h,tuple(params.values()),grad_outputs=v/len(y),retain_graph=True)
            updated={key:p.detach()-args.probe_lr*dg.detach() for (key,p),dg in zip(params.items(),trunk_grad)}
            with torch.no_grad():
                shifted_h=functional_call(classifier.trunk,updated,(x,))
                updated_loss=float(F.cross_entropy(classifier.head(shifted_h),y))
            results.append({'seed':seed,'source_epoch':src_ep,'target_epoch':tgt_ep,
                            'method':name,'probe_batch':i,'n_examples':len(y),
                            'sgd_probe_lr':args.probe_lr,'loss_before':base,
                            'loss_after':updated_loss,'loss_improvement':base-updated_loss})
        del h,g
    for p in params.values():p.requires_grad_(False)
    return results



def summarize_outputs(out: Path) -> None:
    """Summarize independent seeds (not the 10k examples) and make optional plots."""
    metrics = r1.load_csv(out / 'metrics.csv')
    rankings = r1.load_csv(out / 'risk_ranking.csv')
    if not metrics:
        return
    summary = []
    keys = sorted({(int(r['source_epoch']), int(r['target_epoch']), r['model']) for r in metrics})
    cols = ['cosine_mean', 'normalized_rmse', 'first_linear_weight_gradient_nrmse',
            'nonpositive_cosine_fraction', 'near_zero_true_fraction']
    for src, tgt, method in keys:
        rows = [r for r in metrics if int(r['source_epoch']) == src and
                int(r['target_epoch']) == tgt and r['model'] == method]
        rec = {'source_epoch': src, 'target_epoch': tgt,
               'method': method, 'n_seeds': len(rows)}
        for col in cols:
            arr = np.asarray([float(r[col]) for r in rows], dtype=float)
            arr = arr[np.isfinite(arr)]
            rec[col + '_mean'] = float(arr.mean()) if len(arr) else float('nan')
            rec[col + '_sd'] = float(arr.std(ddof=1)) if len(arr)>1 else float('nan')
        summary.append(rec)
    save_csv(out / 'summary_by_checkpoint.csv', summary)
    risk_summary=[]
    for src, tgt, method in sorted({(int(r['source_epoch']),int(r['target_epoch']),r['method']) for r in rankings}):
        values=[float(r['risk_auc']) for r in rankings if int(r['source_epoch']) == src and
                int(r['target_epoch']) == tgt and r['method'] == method]
        a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
        risk_summary.append({'source_epoch':src,'target_epoch':tgt,'method':method,
                             'n_valid_seeds':len(a),
                             'risk_auc_mean':float(a.mean()) if len(a) else float('nan'),
                             'risk_auc_sd':float(a.std(ddof=1)) if len(a)>1 else float('nan')})
    save_csv(out / 'risk_summary.csv',risk_summary)
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed: generated CSV summaries, skipping plots.', flush=True)
        return
    for col, label, filename in [('cosine_mean','Mean gradient cosine','cosine_by_checkpoint.png'),
                                  ('normalized_rmse','Normalized RMSE','normalized_rmse_by_checkpoint.png'),
                                  ('first_linear_weight_gradient_nrmse',
                                   'First-layer weight-gradient normalized RMSE',
                                   'weight_gradient_rmse_by_checkpoint.png')]:
        fig, ax = plt.subplots(figsize=(8.5,4.7))
        # Only same-checkpoint predictors here; staleness is reported separately.
        for m in ['class_mean','cosine_raw','cosine_global_rescale','cosine_learned_magnitude','gaussian']:
            rr=sorted((r for r in summary if r['source_epoch']==r['target_epoch'] and r['method']==m),
                      key=lambda r:r['source_epoch'])
            if rr:
                ax.plot([r['source_epoch'] for r in rr], [r[col+'_mean'] for r in rr],
                        '-o', label=m)
        ax.set_xlabel('Frozen classifier checkpoint epoch')
        ax.set_ylabel(label)
        ax.set_title('Source = target; mean across seeds (development split)')
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=8)
        fig.tight_layout();fig.savefig(out/filename,dpi=140);plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.5,4.7))
    for src in sorted({r['source_epoch'] for r in risk_summary}):
        rr=sorted((r for r in risk_summary if r['source_epoch']==src and
                   r['method']=='learned_risk' and math.isfinite(r['risk_auc_mean'])),
                  key=lambda r:r['target_epoch'])
        if rr:
            ax.plot([r['target_epoch']-src for r in rr],
                    [r['risk_auc_mean'] for r in rr], '-o', label=f'Frozen at epoch {src}')
    ax.set_xlabel('Classifier epochs after risk model was fitted')
    ax.set_ylabel('Gradient-error risk AUROC')
    ax.set_ylim(0,1)
    if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=8)
    ax.set_title('Frozen directional/magnitude risk detector under classifier drift')
    fig.tight_layout();fig.savefig(out/'risk_auc_under_drift.png',dpi=140);plt.close(fig)


def main():
    args=parse_args();configure(args)
    device=device_for(args)
    args.out.mkdir(parents=True,exist_ok=True)
    cfg,splits=get_original(args,device)
    fingerprint={'original_root':str(args.r1_root.resolve()),
                 'seeds':args.seeds,'checkpoints':args.checkpoints,
                 'predictor_epochs':args.pred_epochs,'predictor_snapshots':args.pred_snapshots,
                 'mag_epochs':args.mag_epochs,'risk_epochs':args.risk_epochs,
                 'predictor_batch_size':args.predictor_batch_size,'batch_size':args.batch_size,
                 'pred_lr':args.pred_lr,'cosine_weight':args.cosine_weight,
                 'cosine_threshold':args.risk_cosine_threshold,
                 'relative_threshold':args.risk_relative_threshold,
                 'budgets':args.exact_budgets,'probe_lr':args.probe_lr,
                 'probe_batches':args.probe_batches,'source_config':cfg,
                 'torch_version':torch.__version__}
    manifest=args.out/'followup_config.json'
    if manifest.exists() and json.loads(manifest.read_text())!=fingerprint:
        raise RuntimeError(f'Existing output has different configuration: {manifest}; use a new --out directory.')
    manifest.write_text(json.dumps(fingerprint,indent=2),encoding='utf-8')
    print('R1 follow-up:',device,'checkpoints',args.checkpoints,'seeds',args.seeds,flush=True)
    print('No classifier retraining. Original held-out dev was previously inspected; results remain exploratory.',flush=True)
    allrows=[];rankrows=[];selectionrows=[];proberows=[]
    for seed in args.seeds:
        # Deterministic disjoint predictor-training and risk-training subsets of ORIGINAL train split.
        train_indices=torch.randperm(len(splits['train']),generator=torch.Generator().manual_seed(20261026))
        n_risk = max(32,len(train_indices)//5)
        pidx=train_indices[:-n_risk];ridx=train_indices[-n_risk:]
        for src_ep in args.checkpoints:
            print(f'\n[seed={seed} source epoch={src_ep}] extracting frozen model gradients',flush=True)
            dir=args.out/f'seed_{seed}'/f'source_{src_ep}'
            dir.mkdir(parents=True,exist_ok=True)
            model=get_model(args,cfg,seed,src_ep,device)
            data_full=extract(model,splits['train'],args,device)
            train=select_split(data_full,pidx)
            risk=select_split(data_full,ridx)
            val=extract(model,splits['val'],args,device)
            # Keep dev separate from all fitting / model-selection procedures.
            dev=extract(model,splits['dev'],args,device)
            del data_full
            scale=float(train['g'].square().mean().sqrt().clamp_min(1e-10))
            floor=max(float(train['g'].norm(dim=-1).median())*float(cfg.get('min_norm_ratio',1e-3)),1e-10)
            means=r1.fit_class_means(train['g'],train['y'])
            trained={}; selected_epochs={}
            for method in ('cosine_mse','gaussian'):
                trained[method],selected_epochs[method]=train_gradient_predictor(
                    args,method,None,train,val,scale,floor,means,int(cfg['hidden']),
                    int(cfg.get('pred_width',192)),device,dir,seed,src_ep)
            cos, _=pred_on(trained['cosine_mse'],train,scale,device,args.predictor_batch_size)
            cp_val,_=pred_on(trained['cosine_mse'],val,scale,device,args.predictor_batch_size)
            alpha=alpha_fit(cos,train['g'])
            mag,cap,mag_ep=fit_magnitude_net(args,train,val,cos,cp_val,scale,dir,int(cfg['hidden']),device,seed,src_ep)
            cos_risk,_=pred_on(trained['cosine_mse'],risk,scale,device,args.predictor_batch_size)
            risk_net,pos=fit_risk(args,risk,val,cos_risk,cp_val,means,model,
                                  scale,floor,int(cfg['hidden']),device,dir,seed,src_ep)
            # All correction methods use only source-training data. These parameters are frozen during drift.
            torch.save({'scale':scale,'floor':floor,'class_means':means,
                        'global_alpha':alpha,'magnitude_cap':cap,
                        'predictor_epochs_selected':selected_epochs,
                        'magnitude_epoch_selected':mag_ep},dir/'calibration_constants.pt')
            print(f'  selected predictor epochs {selected_epochs}; alpha={alpha:.4f}; '
                  f'magnitude epoch={mag_ep}; risk train positives={pos:.3f}',flush=True)
            del train,risk,cos,cp_val,cos_risk,val
            for tgt_ep in [e for e in args.checkpoints if e>=src_ep]:
                target_model=model if tgt_ep==src_ep else get_model(args,cfg,seed,tgt_ep,device)
                td=dev if tgt_ep==src_ep else extract(target_model,splits['dev'],args,device)
                p,_=pred_on(trained['cosine_mse'],td,scale,device,args.predictor_batch_size)
                ga,gs=pred_on(trained['gaussian'],td,scale,device,args.predictor_batch_size)
                mag_norm=mag_predict(mag,td,p,scale,cap,device,args.predictor_batch_size)
                predictions={'cosine_raw':p,'cosine_global_rescale':p*alpha,
                             'cosine_learned_magnitude':renormalize(p,mag_norm),
                             'gaussian':ga,'class_mean':means[td['y']]}
                for method,pred in predictions.items():
                    metrics=norm_and_direction_metrics(td,pred,floor)
                    allrows.append({'seed':seed,'source_epoch':src_ep,'target_epoch':tgt_ep,
                                    'model':method,'n_samples':len(pred),
                                    'predictor_epoch':selected_epochs['gaussian'] if method=='gaussian' else selected_epochs['cosine_mse'],
                                    **metrics})
                ranking,selection=risk_report(args,td,p,gs,means,risk_net,target_model,floor,
                                               device,seed,src_ep,tgt_ep)
                rankrows.extend(ranking);selectionrows.extend(selection)
                if args.probe_batches:
                    # Evaluate virtual SGD updates on the first few batches, on the target model.
                    def mk_forward(net, factor=1.0):
                        def fn(h,y):
                            with torch.no_grad(): return net(h,y)[0] * scale * factor
                        return fn
                    def mag_forward(h,y):
                        with torch.no_grad():
                            pred=trained['cosine_mse'](h,y)[0]*scale
                            conf=target_model.head(h).softmax(-1).max(-1).values
                            mlog=mag(h,y,pred.norm(dim=-1),conf,scale)
                            norms=(torch.expm1(mlog.clamp(0,15))*scale).clamp(0,cap)
                            return F.normalize(pred,dim=-1,eps=1e-12)*norms[:,None]
                    methods={'cosine_raw':mk_forward(trained['cosine_mse']),
                             'cosine_global_rescale':mk_forward(trained['cosine_mse'],alpha),
                             'cosine_learned_magnitude':mag_forward,
                             'gaussian':mk_forward(trained['gaussian'])}
                    proberows.extend(probe_sgd(args,target_model,splits['dev'],methods,scale,
                                               device,seed,src_ep,tgt_ep))
                print(f'  -> target={tgt_ep:3d}: raw cosine={next(r["cosine_mean"] for r in allrows[::-1] if r["source_epoch"]==src_ep and r["target_epoch"]==tgt_ep and r["seed"]==seed and r["model"]=="cosine_raw"):.4f}; '
                      f'risk AUROC={next(r["risk_auc"] for r in ranking if r["method"]=="learned_risk"):.4f}',flush=True)
                save_csv(args.out/'metrics.csv',allrows)
                save_csv(args.out/'risk_ranking.csv',rankrows)
                save_csv(args.out/'risk_selection.csv',selectionrows)
                save_csv(args.out/'virtual_sgd_probe.csv',proberows)
                if tgt_ep!=src_ep:
                    del target_model,td
            del model,dev,trained,mag,risk_net
    summarize_outputs(args.out)
    print('\nCOMPLETE. See summary_by_checkpoint.csv, metrics.csv, risk_ranking.csv, risk_selection.csv and virtual_sgd_probe.csv.',flush=True)
    print('All results exploratory; OFFLINE only. No synthetic updates or GPU speedup shown.',flush=True)

if __name__=='__main__':
    main()
