"""Phase D v2: calibrated error feedback with head-update controls and GPU profiling.

Companion to gradient_phase_d.py (both files must be in the same directory).
Not an exact implementation of ADA-GP; this is a controlled Fashion-MNIST pilot.

Windows CMD:
  py -3.13 gradient_phase_d_v2.py --smoke-test --device cpu --out d2_smoke
  py -3.13 gradient_phase_d_v2.py --device cuda --deterministic --out phase_d_v2
  py -3.13 gradient_phase_d_v2.py --device cuda --warmup-root phase_d_study --out phase_d_v2
"""
from __future__ import annotations

import argparse
import collections
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

import gradient_phase_d as base


METHODS = (
    'exact', 'scheduled', 'feedback', 'matched_periodic',
    'exact_head_feedback_mask', 'exact_head_periodic_mask',
)


def save_csv(path, records):
    if not records:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in records for k in r))
    with path.open('w', encoding='utf-8', newline='') as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader(); wr.writerows(records)


def load_warmup(args, seed, train, valid, device, out):
    # Reuse v1 warmup checkpoints only if the entire model/training signature matches.
    cache_dir = Path(args.warmup_root) if args.warmup_root else out
    cache = cache_dir / f'seed_{seed}_warmup.pt'
    if cache.exists() and not args.no_cache:
        state = torch.load(cache, map_location='cpu', weights_only=True)
        if state.get('config') == base.warmup_signature(args):
            print(f'  loaded compatible warmup: {cache}', flush=True)
            return state
        if args.warmup_root:
            raise ValueError(f'Warmup checkpoint incompatible: {cache}; check CLI training parameters')
        print('  warmup signature differs; training fresh', flush=True)
    if args.warmup_root and not cache.exists():
        raise FileNotFoundError(f'Warmup checkpoint not found: {cache}')
    return base.bootstrap(args, train, valid, seed, device, out)


def gradient_rms_error(truth, predicted, scale):
    denom = truth.square().mean().sqrt().clamp_min(scale * 0.05)
    return float(((truth - predicted).square().mean().sqrt() / denom).item())


def calibrate_errors(args, state, train, seed, device, out):
    """Calibrate only from training data, without model or optimizer updates.

    Include the entire calibration pass in feedback timing. This avoids
    choosing thresholds from the evaluation set or after seeing feedback results.
    """
    model, predictor, _ = base.restore(args, state, device)
    model.eval(); predictor.eval()
    scores = []
    base.sync(device); start = time.perf_counter()
    it = iter(base.train_loader(train, args, seed + 84, 0, device))
    for i in range(args.calibration_batches):
        try:
            x, y = next(it)
        except StopIteration:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        # The head backward signal is exact but model parameters are unchanged.
        with torch.no_grad():
            h = model.trunk(x)
        leaf = h.detach().requires_grad_(True)
        ce = F.cross_entropy(model.head(leaf), y, reduction='sum')
        truth = torch.autograd.grad(ce, leaf)[0].detach()
        with torch.no_grad():
            estimate = predictor(h, y) * state['scale']
            scores.append(gradient_rms_error(truth, estimate, state['scale']))
    base.sync(device); dt = time.perf_counter() - start
    if len(scores) < 3:
        raise ValueError('At least 3 calibration batches required (increase dataset size).')
    out_json = {'seed': seed, 'n_batches': len(scores), 'seconds': dt,
                'median_error': float(np.median(scores)),
                'p33': float(np.quantile(scores, args.low_quantile)),
                'p67': float(np.quantile(scores, args.high_quantile)),
                'errors': scores}
    (out/f'seed_{seed}_calibration.json').write_text(json.dumps(out_json, indent=2))
    print(f'  calibration: n={len(scores)} median={np.median(scores):.4f} '
          f'quantiles=({out_json["p33"]:.4f},{out_json["p67"]:.4f}) '
          f'cost={dt:.2f}s', flush=True)
    return out_json


class ErrorController:
    """Rolling pre-update quantile controller with optional drift safety guard.

    The current exact gradient error is compared to quantiles computed from PRIOR
    exact-gradient observations, then added to the buffer. No future information.
    It chooses 0, 2 or 4 synthetic batches (subject to max_synthetic).
    """
    def __init__(self, initial_errors, args):
        if len(initial_errors) < 3:
            raise ValueError('Insufficient calibration scores')
        self.buffer = collections.deque(
            list(initial_errors)[-args.error_window:], maxlen=args.error_window)
        self.base_median = float(np.median(initial_errors))
        self.args = args
        self.counts = {0: 0, 2: 0, 4: 0}
        self.guard_count = 0

    def choose(self, err):
        old = np.asarray(self.buffer, dtype=float)
        lo = float(np.quantile(old, self.args.low_quantile))
        hi = float(np.quantile(old, self.args.high_quantile))
        percentile = float(np.mean(old <= err))
        drift_median = float(np.median(old))
        guard = bool(self.args.drift_guard_ratio > 0 and
                     drift_median > self.base_median * self.args.drift_guard_ratio)
        if guard or err > self.base_median * self.args.emergency_ratio:
            quota = 0
            self.guard_count += 1
        elif err <= lo:
            quota = min(4, self.args.max_synthetic)
        elif err <= hi:
            quota = min(2, self.args.max_synthetic)
        else:
            quota = 0
        self.counts[quota] = self.counts.get(quota, 0) + 1
        self.buffer.append(float(err))
        return quota, {'error': err, 'q_low': lo, 'q_high': hi,
                       'percentile': percentile, 'rolling_median': drift_median,
                       'calibration_median': self.base_median,
                       'emergency_guard': int(guard), 'quota': quota}


def controller_selftest(args):
    # Verify that error values on either side of learned thresholds cause
    # multiple actions. This tests code behavior, not empirical adaptivity.
    x = [0.8,0.82,0.84,0.86,0.88,0.90,0.92,0.94,0.96,0.98]
    c = ErrorController(x, args)
    observed = [c.choose(e)[0] for e in [0.8,0.9,0.98]]
    if args.max_synthetic >= 4 and len(set(observed)) < 3:
        raise AssertionError(f'Controller self-check failed: {observed}')
    return observed


def run_exact_with_head_mask(model, opts, x, y, args, head_update):
    """Full exact BP on every batch; optionally withhold ONLY the head update."""
    trunk_opt, head_opt, _ = opts
    trunk_opt.zero_grad(set_to_none=True)
    head_opt.zero_grad(set_to_none=True)
    loss = F.cross_entropy(model.head(model.trunk(x)), y)
    loss.backward()
    nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    trunk_opt.step()
    if head_update:
        head_opt.step()
    else:
        head_opt.zero_grad(set_to_none=True)
    return float(loss.item())


def spaced_mask(total_steps, bp_count):
    arr = np.zeros(total_steps, dtype=np.bool_)
    if bp_count:
        points = np.floor((np.arange(bp_count)+0.5)*total_steps/bp_count).astype(int)
        arr[points] = True
    assert int(arr.sum()) == bp_count
    return arr


def run_regime(args, state, train, valid, seed, device, mode, out,
               feedback_mask=None, periodic_mask=None, calibration=None):
    m, p, opts = base.restore(args, state, device)
    n_batches = math.ceil(len(train)/args.batch_size)
    total = n_batches*args.epochs
    if mode in ('matched_periodic','exact_head_periodic_mask') and periodic_mask is None:
        raise ValueError('Periodic mask is required')
    if mode == 'exact_head_feedback_mask' and feedback_mask is None:
        raise ValueError('Feedback mask is required')
    if mode == 'feedback' and calibration is None:
        raise ValueError('Calibration is required for feedback')
    controller = ErrorController(calibration['errors'], args) if mode == 'feedback' else None
    used_mask = np.zeros(total, dtype=np.bool_)
    event_rows = []
    epoch_rows = []
    e_count = s_count = head_steps = 0
    quota_left = 0
    total_train_seconds = total_val_seconds = 0.0
    warmup_train = (state['baseline_warmup_train_seconds'] if mode in
                    ('exact','exact_head_feedback_mask','exact_head_periodic_mask')
                    else state['warmup_train_seconds'])
    warmup_val = (state['baseline_warmup_val_seconds'] if mode in
                  ('exact','exact_head_feedback_mask','exact_head_periodic_mask')
                  else state['warmup_val_seconds'])
    cal_seconds = calibration['seconds'] if mode == 'feedback' else 0.0
    vloader = base.valid_loader(valid, args, device)
    target_first = None
    if any(r['val_accuracy'] >= args.target_accuracy for r in state['warmup_history']):
        target_first = warmup_train+warmup_val
    step = 0
    for ep in range(args.epochs):
        m.train(); p.train()
        base.sync(device); t0 = time.perf_counter()
        epoch_exact = epoch_synthetic = epoch_head = 0
        errs = []; losses = []
        for x, y in base.train_loader(train,args,seed,ep+args.warmup_epochs,device):
            x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
            if mode == 'feedback':
                take_exact = quota_left == 0
            elif mode in ('matched_periodic','exact_head_periodic_mask'):
                take_exact = bool(periodic_mask[step])
            elif mode == 'exact_head_feedback_mask':
                take_exact = bool(feedback_mask[step])
            elif mode == 'scheduled':
                take_exact = (step % (base.schedule_ratio(ep,args.epochs)+1) == 0)
            else:
                take_exact = True

            # For head controls, take_exact describes the replayed HEAD update;
            # BP remains exact on *every* minibatch.
            used_mask[step] = take_exact
            if mode.startswith('exact_head_'):
                loss=run_exact_with_head_mask(m,opts,x,y,args,take_exact)
                e_count+=1; epoch_exact+=1
                if take_exact: head_steps+=1; epoch_head+=1
            elif take_exact:
                loss, _, err=base.exact_step(
                    m,p,opts,x,y,state['scale'],args,update_predictor=(mode!='exact'))
                e_count+=1;epoch_exact+=1;head_steps+=1;epoch_head+=1
                if mode == 'feedback':
                    quota_left, detail=controller.choose(err)
                    event_rows.append({'seed':seed,'epoch':args.warmup_epochs+ep+1,
                                       'step':step, **detail})
                    errs.append(err)
            else:
                loss=base.synthetic_step(m,p,opts,x,y,state['scale'],args)
                s_count+=1;epoch_synthetic+=1
                if mode=='feedback':
                    quota_left-=1
            losses.append(loss)
            step+=1
        base.sync(device); train_dt=time.perf_counter()-t0
        total_train_seconds+=train_dt
        base.sync(device); vt=time.perf_counter()
        v=base.evaluate(m,vloader,device)
        base.sync(device); val_dt=time.perf_counter()-vt
        total_val_seconds+=val_dt
        wall=(warmup_train+warmup_val+cal_seconds+
              total_train_seconds+total_val_seconds)
        if target_first is None and v['val_accuracy'] >= args.target_accuracy:
            target_first=wall
        row = {'seed':seed,'method':mode,'epoch':args.warmup_epochs+ep+1,
               'val_accuracy':v['val_accuracy'],'val_ce':v['val_ce'],
               'train_ce':float(np.mean(losses)),
               'exact_backward_epoch':epoch_exact,'synthetic_epoch':epoch_synthetic,
               'head_updates_epoch':epoch_head,
               'head_update_fraction_cum':head_steps/step,
               'exact_backward_fraction_cum':e_count/step,
               'controller_error_mean':float(np.mean(errs)) if errs else '',
               'controller_quota_0':controller.counts.get(0,0) if controller else '',
               'controller_quota_2':controller.counts.get(2,0) if controller else '',
               'controller_quota_4':controller.counts.get(4,0) if controller else '',
               'calibration_seconds':cal_seconds,
               'train_seconds_cum':total_train_seconds,
               'eval_seconds_cum':total_val_seconds,
               'wall_seconds_with_warmup':wall}
        epoch_rows.append(row)
        print(f'  seed {seed} {mode:<26} epoch {ep+1}/{args.epochs}: '
              f'val={v["val_accuracy"]:.4f} exact={epoch_exact} syn={epoch_synthetic} '
              f'head={epoch_head} ctrl={controller.counts if controller else "-"}',flush=True)
    if mode == 'exact_head_feedback_mask':
        assert np.array_equal(used_mask, feedback_mask), 'Head replay mask mismatch'
        assert head_steps == int(feedback_mask.sum()), 'Wrong replayed head-update count'
    if mode in ('matched_periodic', 'exact_head_periodic_mask'):
        assert np.array_equal(used_mask, periodic_mask), 'Periodic mask mismatch'
        assert head_steps == int(periodic_mask.sum()), 'Wrong periodic head-update count'
    if mode == 'feedback':
        assert len(event_rows) == e_count
        assert sum(controller.counts.values()) == e_count
    counts = controller.counts if controller else {}
    observed_states=sum(n>0 for n in counts.values())
    if mode=='feedback' and observed_states < 2:
        print('  WARNING: feedback remained in one state. This is not an adaptive comparison.', flush=True)
    rec = {'seed':seed,'method':mode,
           'final_accuracy':epoch_rows[-1]['val_accuracy'],
           'best_accuracy':max(r['val_accuracy'] for r in epoch_rows),
           'final_ce':epoch_rows[-1]['val_ce'],
           'exact_backward_batches':e_count,
           'synthetic_batches':s_count,
           'head_updates':head_steps,
           'postwarm_batches':total,
           'exact_backward_fraction':e_count/total,
           'head_update_fraction':head_steps/total,
           'feedback_states_used':observed_states if controller else '',
           'feedback_guard_events':controller.guard_count if controller else '',
           'calibration_seconds':cal_seconds,
           'warmup_train_seconds':warmup_train,
           'warmup_eval_seconds':warmup_val,
           'postwarm_train_seconds':total_train_seconds,
           'postwarm_eval_seconds':total_val_seconds,
           'total_train_seconds':warmup_train+cal_seconds+total_train_seconds,
           'total_wall_seconds':warmup_train+warmup_val+cal_seconds+total_train_seconds+total_val_seconds,
           'target_accuracy':args.target_accuracy,
           'seconds_to_target':target_first if target_first is not None else ''}
    torch.save({'model':{k:v.detach().cpu() for k,v in m.state_dict().items()},
                'predictor':{k:v.detach().cpu() for k,v in p.state_dict().items()},
                'record':rec},out/f'seed_{seed}_{mode}_final.pt')
    if mode=='feedback':
        save_csv(out/f'seed_{seed}_controller_events.csv', event_rows)
    return epoch_rows,rec,used_mask


def profile_segments(args,state,train,seed,device,out):
    """Separate diagnostic microbenchmark (no interference with training run).

    Splits head/trunk backward via autograd.grad, unlike ordinary full loss.backward;
    therefore these component timings are diagnostic, NOT the primary speed metric.
    GPU sync is intentionally used around every segment, outside full training.
    """
    profile_rows=[]
    loader=iter(base.train_loader(train,args,seed+18,0,device))
    preloaded=[]
    for j in range(args.profile_batches + 1):
        try:preloaded.append(next(loader))
        except StopIteration:break
    if not preloaded: return []
    for variant in ('exact_plain','exact_predictor','synthetic'):
        model,pred,ops=base.restore(args,state,device)
        model.train();pred.train()
        for batch_id,(xb,yb) in enumerate(preloaded):
            segments={}
            def timed(name, action):
                base.sync(device);start=time.perf_counter()
                v=action()
                base.sync(device);seconds=time.perf_counter()-start
                segments[name]=segments.get(name,0.)+seconds
                return v
            def move():
                return xb.to(device,non_blocking=True),yb.to(device,non_blocking=True)
            x,y=timed('data_to_device',move)
            tr,hd,pr=ops
            tr.zero_grad(set_to_none=True); hd.zero_grad(set_to_none=True)
            if variant == 'synthetic':
                h=timed('trunk_forward',lambda:model.trunk(x))
                timed('head_forward',lambda: F.cross_entropy(model.head(h.detach()),y).detach())
                with torch.no_grad():
                    grad=timed('predictor_inference',lambda:pred(h.detach(),y)*state['scale']/len(y))
                timed('trunk_backward',lambda:h.backward(grad.detach()))
                def update():
                    nn.utils.clip_grad_norm_(model.trunk.parameters(),args.clip_grad)
                    tr.step()
                timed('optimizer',update)
            else:
                h=timed('trunk_forward',lambda:model.trunk(x))
                loss=timed('head_forward',lambda:F.cross_entropy(model.head(h),y))
                head_params=list(model.head.parameters())
                grads=timed('head_backward',lambda:torch.autograd.grad(loss,(h,*head_params)))
                gh,*head_grads=grads
                for param,g in zip(head_params,head_grads):param.grad=g.detach()
                timed('trunk_backward',lambda:h.backward(gh.detach()))
                if variant=='exact_predictor':
                    with torch.no_grad():
                        timed('predictor_inference',lambda:pred(h.detach(),y)*state['scale'])
                def update():
                    nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)
                    tr.step();hd.step()
                timed('optimizer',update)
                if variant=='exact_predictor':
                    true=(gh.detach()*len(y)/state['scale'])
                    def pred_train():
                        pr.zero_grad(set_to_none=True)
                        estimate=pred(h.detach(),y)
                        pl=base.predictor_loss(estimate,true,args.cosine_weight)
                        pl.backward()
                        nn.utils.clip_grad_norm_(pred.parameters(),args.clip_grad)
                        pr.step()
                    timed('predictor_training',pred_train)
            # Exclude a first unmeasured warm-up iteration (kernel/cache startup).
            if batch_id > 0:
                profile_rows += [{'seed':seed,'variant':variant,'component':c,
                                  'seconds':value} for c,value in segments.items()]
    save_csv(out/f'seed_{seed}_profile_components.csv', profile_rows)
    return profile_rows


def profile_aggregate(rows,out):
    groups={}
    for r in rows:
        groups.setdefault((r['variant'],r['component']),[]).append(float(r['seconds']))
    records=[{'variant':v,'component':c,'n':len(vals),
              'mean_ms':1000*float(np.mean(vals)),
              'median_ms':1000*float(np.median(vals))}
             for (v,c),vals in sorted(groups.items())]
    for v in sorted(set(r['variant'] for r in rows)):
        component_means=[r['mean_ms'] for r in records if r['variant']==v]
        records.append({'variant':v,'component':'TOTAL_DIAGNOSTIC',
                        'n':'','mean_ms':sum(component_means),
                        'median_ms':''})
    save_csv(out/'profile_summary.csv',records)


def aggregate(all_final, out):
    summary=[]; paired=[]
    for method in METHODS:
        subset=[r for r in all_final if r['method']==method]
        if not subset:continue
        row={'method':method,'n_seeds':len(subset)}
        for field in ('final_accuracy','total_wall_seconds','total_train_seconds',
                      'exact_backward_fraction','head_update_fraction',
                      'feedback_states_used'):
            v=[float(r[field]) for r in subset if r[field]!='']
            row[field+'_mean']=float(np.mean(v)) if v else ''
            row[field+'_sd']=float(np.std(v,ddof=1)) if len(v)>1 else ''
        summary.append(row)
    for seed in sorted(set(r['seed'] for r in all_final)):
        items={r['method']:r for r in all_final if r['seed']==seed}
        if 'feedback' not in items:continue
        f=items['feedback']
        for baseline in METHODS:
            if baseline=='feedback' or baseline not in items:continue
            b=items[baseline]
            paired.append({'seed':seed,'against':baseline,
                          'accuracy_diff':f['final_accuracy']-b['final_accuracy'],
                          'train_seconds_diff':f['total_train_seconds']-b['total_train_seconds'],
                          'wall_seconds_diff':f['total_wall_seconds']-b['total_wall_seconds'],
                          'exact_fraction_diff':f['exact_backward_fraction']-b['exact_backward_fraction'],
                          'head_update_fraction_diff':f['head_update_fraction']-b['head_update_fraction']})
    save_csv(out/'summary_aggregate.csv',summary)
    save_csv(out/'paired_vs_feedback.csv',paired)


def make_plots(rows,out):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return
    for field,title,name in [
        ('val_accuracy','Validation accuracy','accuracy.png'),
        ('wall_seconds_with_warmup','Elapsed training and validation (seconds)','time.png'),
        ('head_update_fraction_cum','Cumulative head update fraction','head_updates.png')]:
        fig, ax=plt.subplots(figsize=(9,5))
        for method in METHODS:
            curves=[]
            for s in sorted(set(r['seed'] for r in rows)):
                sub=sorted((r for r in rows if r['seed']==s and r['method']==method),key=lambda r:r['epoch'])
                if sub:curves.append(np.array([float(r[field]) for r in sub]))
            if curves and len(set(map(len,curves)))==1:
                vals=np.stack(curves);epochs=range(1,vals.shape[1]+1)
                ax.plot(epochs,vals.mean(0),label=method)
                if len(curves)>1:
                    ax.fill_between(epochs,vals.mean(0)-vals.std(0,ddof=1),
                                    vals.mean(0)+vals.std(0,ddof=1),alpha=.12)
        ax.set_xlabel('Epochs after warm-up');ax.set_ylabel(title)
        ax.legend(fontsize=8);ax.grid(alpha=.2);fig.tight_layout()
        fig.savefig(out/name,dpi=140);plt.close(fig)


def parser():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--out',default='phase_d_v2')
    p.add_argument('--warmup-root',default=None,help='Reuse seed_*_warmup.pt from previous Phase D folder')
    p.add_argument('--device',choices=('auto','cpu','cuda'),default='auto')
    p.add_argument('--data-dir',default='data')
    p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    p.add_argument('--smoke-test',action='store_true')
    p.add_argument('--deterministic',action='store_true')
    p.add_argument('--no-cache',action='store_true')
    p.add_argument('--audit-every-step',action='store_true')
    p.add_argument('--skip-profile',action='store_true')
    p.add_argument('--profile-batches',type=int,default=8)
    p.add_argument('--warmup-epochs',type=int,default=4)
    p.add_argument('--epochs',type=int,default=16)
    p.add_argument('--batch-size',type=int,default=128)
    p.add_argument('--eval-batch-size',type=int,default=512)
    p.add_argument('--hidden',type=int,default=128)
    p.add_argument('--predictor-width',type=int,default=96)
    p.add_argument('--lr',type=float,default=0.001)
    p.add_argument('--pred-lr',type=float,default=0.001)
    p.add_argument('--weight-decay',type=float,default=1e-4)
    p.add_argument('--clip-grad',type=float,default=5.0)
    p.add_argument('--cosine-weight',type=float,default=0.1)
    p.add_argument('--target-accuracy',type=float,default=0.88)
    p.add_argument('--max-synthetic',type=int,default=4)
    p.add_argument('--calibration-batches',type=int,default=64)
    p.add_argument('--error-window',type=int,default=96)
    p.add_argument('--low-quantile',type=float,default=1/3)
    p.add_argument('--high-quantile',type=float,default=2/3)
    p.add_argument('--drift-guard-ratio',type=float,default=2.0)
    p.add_argument('--emergency-ratio',type=float,default=3.0)
    p.add_argument('--ema-alpha',type=float,default=0.2,help='compatibility with v1 warmup signature (not used for v2 controller)')
    p.add_argument('--error-low',type=float,default=0.7,help='compatibility only, v2 uses empirical quantiles')
    p.add_argument('--error-high',type=float,default=1.1,help='compatibility only, v2 uses empirical quantiles')
    args=p.parse_args()
    if args.smoke_test:
        args.warmup_epochs=1; args.epochs=2; args.batch_size=64
        args.seeds=[0];args.calibration_batches=5;args.profile_batches=2
        args.target_accuracy=0.10
    if not (0<args.low_quantile<args.high_quantile<1):p.error('invalid quantiles')
    if min(args.epochs,args.warmup_epochs,args.batch_size,args.calibration_batches,args.error_window)<1:
        p.error('epochs, warmup, batch size, calibration and window must be positive')
    if args.max_synthetic not in (2,4):p.error('max-synthetic must be 2 or 4')
    if args.drift_guard_ratio<=1 or args.emergency_ratio<=1:p.error('guard ratios must exceed 1')
    if args.warmup_root and args.smoke_test:p.error('smoke test cannot reuse real warmup checkpoints')
    return args


def main():
    args=parser()
    if args.deterministic:
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    device=base.device_from(args.device)
    observed=controller_selftest(args)
    print('controller unit check: quotas',observed,flush=True)
    print('device:',device,'torch:',torch.__version__,flush=True)
    audit=base.audit_gradient_injection(device)
    train,valid=base.datasets(args)
    config={k:v for k,v in vars(args).items() if not k.startswith('_')}
    config.update({'torch_version':torch.__version__,'device_used':str(device),
                   'gpu':torch.cuda.get_device_name(0) if device.type=='cuda' else None,
                   'audit':audit,'controller_test_actions':observed})
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    all_epochs=[];all_final=[];all_profiles=[]
    for seed in args.seeds:
        print('\n=== SEED',seed,'===',flush=True)
        state=load_warmup(args,seed,train,valid,device,out)
        cal=calibrate_errors(args,state,train,seed,device,out)
        if not args.skip_profile:
            profile=profile_segments(args,state,train,seed,device,out)
            all_profiles+=profile
            save_csv(out/'profile_components.csv',all_profiles)
            profile_aggregate(all_profiles,out)
        masks={}
        for mode in METHODS:
            periodic=masks.get('matched_periodic')
            feedback=masks.get('feedback')
            if mode == 'matched_periodic':
                periodic=spaced_mask(len(feedback),int(feedback.sum()))
                masks['matched_periodic']=periodic
            rows,rec,decisions=run_regime(
                args,state,train,valid,seed,device,mode,out,
                feedback_mask=feedback,periodic_mask=periodic,calibration=cal)
            if mode=='feedback':masks['feedback']=decisions
            all_epochs+=rows;all_final.append(rec)
            save_csv(out/'epochs.csv',all_epochs)
            save_csv(out/'summary.csv',all_final)
            aggregate(all_final,out)
        np.savez_compressed(out/f'seed_{seed}_masks.npz',
                            feedback=masks['feedback'],
                            matched_periodic=masks['matched_periodic'])
        make_plots(all_epochs,out)
    print('\nDone:',out.resolve(),flush=True)
    print('Inspect summary.csv, paired_vs_feedback.csv, controller_events and component profile.',flush=True)
    print('Time-to-target includes warmup and feedback calibration. This is not a hardware ADA-GP replication.',flush=True)


if __name__ == '__main__':
    main()
