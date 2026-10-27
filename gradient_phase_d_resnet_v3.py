"""Phase D-ResNet: CIFAR-10 selective synthetic-gradient training.

Standalone; requires torch, torchvision and numpy. No previous Phase D files.
The experiment compares exact BP, scheduled GP/BP, rolling-relative feedback,
fixed-calibration absolute-error feedback, matched periodic, and a head-update control.

The split is AFTER ResNet layer2. On synthetic steps the head forward/backward is
SKIPPED and only the trunk is updated with a learned activation gradient.
This is a pilot algorithm, NOT an exact reproduction of ADA-GP.

All methods share classifier/predictor warm-up weights, postwarm learning-rate
schedule, exact minibatch sequence and data augmentation RNG. Head BatchNorm
running statistics are frozen after warm-up for all methods because the synthetic
method does not execute head forward passes. This is an explicit experimental
choice and can change accuracy relative to ordinary ResNet training.
"""
from __future__ import annotations

import argparse
import collections
import copy
import csv
import json
import math
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset, Subset, TensorDataset


METHODS = ('exact', 'scheduled', 'relative_feedback', 'absolute_feedback',
           'matched_periodic', 'exact_head_mask')


def seed_everything(seed, deterministic=False):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try: torch.use_deterministic_algorithms(True, warn_only=True)
        except TypeError: torch.use_deterministic_algorithms(True)


def sync(device):
    if device.type == 'cuda': torch.cuda.synchronize()


def save_csv(path, rows):
    if not rows: return
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as f:
        w=csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


class BasicBlock(nn.Module):
    def __init__(self, n_in, n_out, stride=1):
        super().__init__()
        self.conv1=nn.Conv2d(n_in, n_out, 3, stride, 1, bias=False)
        self.bn1=nn.BatchNorm2d(n_out)
        self.conv2=nn.Conv2d(n_out, n_out, 3, 1, 1, bias=False)
        self.bn2=nn.BatchNorm2d(n_out)
        self.skip=(nn.Identity() if n_in==n_out and stride==1 else
                   nn.Sequential(nn.Conv2d(n_in,n_out,1,stride,bias=False),nn.BatchNorm2d(n_out)))

    def forward(self, x):
        return F.relu(self.bn2(self.conv2(F.relu(self.bn1(self.conv1(x))))) + self.skip(x))


def make_layer(ch_in, ch_out, blocks, stride):
    pieces=[BasicBlock(ch_in,ch_out,stride)]
    pieces += [BasicBlock(ch_out,ch_out) for _ in range(blocks-1)]
    return nn.Sequential(*pieces)


class SplitResNet(nn.Module):
    def __init__(self, width=64, blocks=2):
        super().__init__()
        self.width=width
        self.trunk=nn.Sequential(
            nn.Conv2d(3,width,3,1,1,bias=False),nn.BatchNorm2d(width),nn.ReLU(),
            make_layer(width,width,blocks,1),
            make_layer(width,width*2,blocks,2),
        )
        self.head=nn.Sequential(
            make_layer(width*2,width*4,blocks,2),
            make_layer(width*4,width*8,blocks,2),
            nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(width*8,10),
        )

    def forward(self,x): return self.head(self.trunk(x))

    def postwarm_mode(self):
        self.trunk.train()
        # Freeze head BN statistics, but leave all affine parameters trainable.
        self.head.eval()


class GradientPredictor(nn.Module):
    """Cheap spatial synthetic-gradient predictor with class conditioning.

    Feature tensor is [B,2*width,H,W]. Prediction uses a 1x1 bottleneck and
    depthwise 3x3 to avoid another large ResNet-style head.
    """
    def __init__(self, channels, width=32):
        super().__init__()
        self.label=nn.Embedding(10, channels)
        self.net=nn.Sequential(
            nn.Conv2d(channels, width, 1),nn.ReLU(),
            nn.Conv2d(width,width,3,padding=1,groups=width),nn.ReLU(),
            nn.Conv2d(width,channels,1),
        )

    def forward(self,h,y):
        return self.net(h.detach() + self.label(y)[:,:,None,None])


def predictor_loss(estimate, target, cosine_weight):
    # Coordinates are normalized by one fixed scale. Larger cosine_weight
    # favors directional prediction, never substitutes a training gradient here.
    mse=(estimate-target).square().mean()
    a=estimate.flatten(1); b=target.flatten(1)
    valid=b.norm(dim=1) > 1e-5
    if valid.any():
        cosine=(1-F.cosine_similarity(a[valid],b[valid],dim=1,eps=1e-8)).mean()
    else: cosine=mse*0
    return mse + cosine_weight*cosine


def train_data(args):
    if args.smoke_test:
        g=torch.Generator().manual_seed(991)
        x=torch.randn(64,3,32,32,generator=g)
        y=torch.randint(0,10,(64,),generator=g)
        return TensorDataset(x[:48],y[:48]),TensorDataset(x[48:],y[48:])
    from torchvision import datasets, transforms
    norm=transforms.Normalize((0.4914,0.4822,0.4465),(0.2470,0.2435,0.2616))
    train_trans=transforms.Compose([transforms.RandomCrop(32,padding=4),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),norm])
    eval_trans=transforms.Compose([transforms.ToTensor(),norm])
    ds_aug=datasets.CIFAR10(root=args.data_dir,train=True,download=True,transform=train_trans)
    ds_val=datasets.CIFAR10(root=args.data_dir,train=True,download=False,transform=eval_trans)
    g=torch.Generator().manual_seed(71503)
    inds=torch.randperm(len(ds_aug),generator=g).tolist()
    n_val=5000
    return Subset(ds_aug,inds[n_val:]),Subset(ds_val,inds[:n_val])


def loader(ds,args,seed,epoch,shuffle=True):
    # Generator plus deterministic worker seeds; same augmentation/order for all methods.
    # With workers=0, torchvision's crop/flip use the main-process RNG.
    aug_seed=245600+100_000*seed+epoch
    if shuffle:
        random.seed(aug_seed); np.random.seed(aug_seed % (2**32-1))
        torch.manual_seed(aug_seed)
    g=torch.Generator().manual_seed(aug_seed)
    return DataLoader(ds,batch_size=args.batch_size,shuffle=shuffle,generator=g,
                      num_workers=args.workers,pin_memory=args.pin_memory,
                      persistent_workers=(args.workers>0))


def create(args,device):
    model=SplitResNet(args.width,args.blocks).to(device)
    pred=GradientPredictor(args.width*2,args.predictor_width).to(device)
    return model,pred


def opts_for(m,p,args):
    return (torch.optim.SGD(m.trunk.parameters(),lr=args.lr,momentum=.9,weight_decay=args.weight_decay),
            torch.optim.SGD(m.head.parameters(),lr=args.lr,momentum=.9,weight_decay=args.weight_decay),
            torch.optim.AdamW(p.parameters(),lr=args.pred_lr,weight_decay=1e-4))


def exact_step(model,pred,opts,x,y,scale,args,fit_predictor=True,head_update=True):
    t_opt,h_opt,p_opt=opts
    t_opt.zero_grad(set_to_none=True);h_opt.zero_grad(set_to_none=True)
    h=model.trunk(x)
    if fit_predictor: h.retain_grad()
    loss=F.cross_entropy(model.head(h),y)
    loss.backward()
    if fit_predictor:
        truth=h.grad.detach()*len(y) / scale  # per-example target, not batch-mean
        with torch.no_grad():
            estimated=pred(h.detach(),y)
            # Absolute normalized RMSE of per-example activation gradients.
            err=float((estimated-truth).square().mean().sqrt().item())
        p_opt.zero_grad(set_to_none=True)
        pl=predictor_loss(pred(h.detach(),y),truth,args.cosine_weight)
        pl.backward()
        nn.utils.clip_grad_norm_(pred.parameters(),args.clip_grad)
        p_opt.step()
    else: err=float('nan')
    nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)
    t_opt.step()
    if head_update: h_opt.step()
    else: h_opt.zero_grad(set_to_none=True)
    return float(loss.item()),err


def synthetic_step(model,pred,opts,x,y,scale,args):
    t_opt,h_opt,_=opts
    t_opt.zero_grad(set_to_none=True);h_opt.zero_grad(set_to_none=True)
    h=model.trunk(x)
    with torch.no_grad(): g=pred(h.detach(),y)*scale/len(y)
    # The downstream head is never called: no head forward graph or backward graph.
    h.backward(g.detach())
    nn.utils.clip_grad_norm_(model.trunk.parameters(),args.clip_grad)
    t_opt.step()
    if args.smoke_test:
        assert all(v.grad is None for v in model.head.parameters())
    return float('nan')


@torch.no_grad()
def evaluate(model,val_loader,device):
    model.eval();good=n=0; summed=0.0
    for xx,yy in val_loader:
        xx=xx.to(device,non_blocking=True);yy=yy.to(device,non_blocking=True)
        out=model(xx);summed+=F.cross_entropy(out,yy,reduction='sum').item()
        good+=(out.argmax(1)==yy).sum().item();n+=len(yy)
    return {'accuracy':good/max(1,n),'ce':summed/max(1,n)}


def audit_injection(device):
    seed_everything(444)
    m=SplitResNet(width=4,blocks=1).to(device)
    m.eval()   # avoid BatchNorm buffers changing between the two audit runs
    x=torch.randn(2,3,32,32,device=device);y=torch.tensor([1,5],device=device)
    m.zero_grad(set_to_none=True)
    h=m.trunk(x); loss=F.cross_entropy(m.head(h),y)
    grad=torch.autograd.grad(loss,h,retain_graph=True)[0].detach()
    loss.backward()
    actual=[p.grad.clone() for p in m.trunk.parameters()]
    m.zero_grad(set_to_none=True)
    m.trunk(x).backward(grad)
    inject=[p.grad.clone() for p in m.trunk.parameters()]
    diff=max((a-b).abs().max().item() for a,b in zip(actual,inject))
    assert diff < (3e-4 if device.type=='cuda' else 2e-5), f'injection error {diff}'
    assert all(p.grad is None for p in m.head.parameters())
    print(f'PASS: exact gradient injection, max abs parameter-gradient error={diff:.3g}',flush=True)


def signature(args):
    return {k:getattr(args,k) for k in ('smoke_test','width','blocks','predictor_width',
             'warmup_epochs','batch_size','lr','pred_lr','weight_decay','cosine_weight',
             'clip_grad','deterministic','epochs','lr_min')}


def warmup(args,train,val,device,seed,out):
    path=out/f'seed_{seed}_warmup.pt'
    if path.exists() and not args.overwrite:
        try:
            ck=torch.load(path,map_location='cpu',weights_only=True)
            if ck['signature']==signature(args):
                print(f'Reusing warm-up {path}',flush=True);return ck
        except Exception as e: print(f'Cache invalid: {e}',flush=True)
    seed_everything(18371+seed,args.deterministic)
    m,p=create(args,device);ops=opts_for(m,p,args)
    # Baseline separate trajectory from same initialization; same batches and updates.
    plain=copy.deepcopy(m); dummy=copy.deepcopy(p)
    plain_ops=opts_for(plain,dummy,args)
    scale=None; t_syn=v_syn=t_plain=v_plain=0.0
    epoch_logs=[]
    for ep in range(args.warmup_epochs):
        set_epoch_lr(ops,args,ep)
        m.train();p.train();sync(device);t0=time.perf_counter()
        for x,y in loader(train,args,seed,ep):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            if scale is None:
                # one fixed gradient-unit scale for the entire experiment
                # Freeze BN updates during scale estimation, otherwise the
                # predictor-trained and ordinary-BP baselines differ in BN state.
                m.eval()
                with torch.no_grad(): h=m.trunk(x)
                leaf=h.detach().requires_grad_(True)
                target=torch.autograd.grad(F.cross_entropy(m.head(leaf),y,reduction='sum'),leaf)[0]
                scale=max(float(target.square().mean().sqrt()),1e-6)
                m.train()
            exact_step(m,p,ops,x,y,scale,args,fit_predictor=True)
        sync(device);t_syn+=time.perf_counter()-t0
        sync(device);v0=time.perf_counter(); valinfo=evaluate(m,loader(val,args,seed,0,False),device)
        sync(device);v_syn+=time.perf_counter()-v0
        epoch_logs.append({'epoch':ep+1,**valinfo})
        print(f'  seed {seed} warmup {ep+1}/{args.warmup_epochs} val_accuracy={valinfo["accuracy"]:.3f}',flush=True)
    for ep in range(args.warmup_epochs):
        set_epoch_lr(plain_ops,args,ep)
        plain.train();sync(device);t0=time.perf_counter()
        for x,y in loader(train,args,seed,ep):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            exact_step(plain,None,plain_ops,x,y,1.,args,fit_predictor=False)
        sync(device);t_plain+=time.perf_counter()-t0
        sync(device);v0=time.perf_counter();evaluate(plain,loader(val,args,seed,0,False),device)
        sync(device);v_plain+=time.perf_counter()-v0
    diff=max((a-b).abs().max().item() for a,b in zip(m.parameters(),plain.parameters()))
    state_diff=max((a-b).abs().max().item() for a,b in zip(m.state_dict().values(),plain.state_dict().values()))
    assert diff < 1e-5 and state_diff < 1e-5, (
        f'Warm-up classifiers diverged: parameter={diff}, BN buffers included={state_diff}')
    ck={'signature':signature(args),'model':{k:v.detach().cpu() for k,v in m.state_dict().items()},
        'predictor':{k:v.detach().cpu() for k,v in p.state_dict().items()},
        'optim_trunk':copy.deepcopy(ops[0].state_dict()),
        'optim_head':copy.deepcopy(ops[1].state_dict()),
        'optim_predictor':copy.deepcopy(ops[2].state_dict()),
        'scale':float(scale),'warmup_synthetic_seconds':t_syn+v_syn,
        'warmup_plain_seconds':t_plain+v_plain,'warmup_metrics':epoch_logs}
    torch.save(ck,path)
    return ck


def restore(args,ck,device):
    m,p=create(args,device)
    m.load_state_dict(ck['model']);p.load_state_dict(ck['predictor'])
    ops=opts_for(m,p,args)
    for o,key in zip(ops,('optim_trunk','optim_head','optim_predictor')):o.load_state_dict(ck[key])
    return m,p,ops


def rel_error(estimated,truth):
    # Dimensionless error, unlike absolute feedback. Norm-clamped.
    norm=truth.square().mean().sqrt().clamp_min(0.05)
    return float(((estimated-truth).square().mean().sqrt()/norm).item())


def calibration(args,ck,train,device,seed,out):
    path=out/f'seed_{seed}_calibration.json'
    if path.exists() and not args.overwrite:
        r=json.loads(path.read_text(encoding='utf-8'))
        if (r.get('signature')==signature(args) and
            r.get('calibration_batches_requested')==args.calibration_batches and
            r.get('quantiles')==[args.low_q,args.high_q]):
            return r
    m,p,_=restore(args,ck,device);m.postwarm_mode();p.eval()
    vals_abs=[];vals_rel=[]
    sync(device);t0=time.perf_counter()
    for i,(x,y) in enumerate(loader(train,args,seed,911)):
        if i>=args.calibration_batches:break
        x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
        with torch.no_grad():h=m.trunk(x)
        leaf=h.detach().requires_grad_(True)
        truth=torch.autograd.grad(F.cross_entropy(m.head(leaf),y,reduction='sum'),leaf)[0]/ck['scale']
        with torch.no_grad():
            estimated=p(h,y)
            vals_abs.append(float((estimated-truth).square().mean().sqrt()))
            vals_rel.append(rel_error(estimated,truth))
    sync(device);seconds=time.perf_counter()-t0
    if len(vals_abs)<3:raise RuntimeError('Need at least three calibration batches')
    outp={'signature':signature(args),'quantiles':[args.low_q,args.high_q],
          'calibration_batches_requested':args.calibration_batches,
          'absolute_errors':vals_abs,'relative_errors':vals_rel,
          'seconds':seconds,'q_abs_low':float(np.quantile(vals_abs,args.low_q)),
          'q_abs_high':float(np.quantile(vals_abs,args.high_q)),
          'median_abs':float(np.median(vals_abs)),'median_rel':float(np.median(vals_rel))}
    path.write_text(json.dumps(outp,indent=2),encoding='utf-8')
    return outp


class Feedback:
    def __init__(self,kind,args,cal):
        self.kind=kind;self.args=args
        self.history=collections.deque(cal['relative_errors'][-args.error_window:],maxlen=args.error_window)
        self.lo=cal['q_abs_low'];self.hi=cal['q_abs_high'];self.base=cal['median_abs']
        self.ema=None;self.choices=collections.Counter()

    def choose(self,error_abs,error_rel):
        if self.kind=='absolute_feedback':
            # Calibrated ONCE at initial warm-up state. No moving-percentile drift masking.
            self.ema=(error_abs if self.ema is None else
                      self.args.ema_beta*self.ema+(1-self.args.ema_beta)*error_abs)
            score=self.ema
            if score > self.base*self.args.emergency_ratio: quota=0
            elif score<=self.lo:quota=4
            elif score<=self.hi:quota=2
            else:quota=0
            lo,hi=self.lo,self.hi
        else:
            hist=np.array(self.history)
            lo=float(np.quantile(hist,self.args.low_q))
            hi=float(np.quantile(hist,self.args.high_q))
            # Compared to prior errors, not future ones
            score=error_rel
            if score<=lo:quota=4
            elif score<=hi:quota=2
            else:quota=0
            self.history.append(error_rel)
        quota=min(quota,self.args.max_synthetic)
        self.choices[quota]+=1
        return quota,{'error_abs':error_abs,'error_relative':error_rel,
                      'controller_score':score,'threshold_low':lo,
                      'threshold_high':hi,'synthetic_quota':quota}


def audit_controller():
    class ToyArgs:
        low_q=.33;high_q=.67;error_window=12
        ema_beta=0;emergency_ratio=2;max_synthetic=4
    cal={'relative_errors':[.5,.6,.7,.8,.9,1.,1.1],
         'q_abs_low':.6,'q_abs_high':1.,'median_abs':.8}
    assert Feedback('absolute_feedback',ToyArgs,cal).choose(.2,.2)[0]==4
    assert Feedback('absolute_feedback',ToyArgs,cal).choose(.8,.8)[0]==2
    assert Feedback('absolute_feedback',ToyArgs,cal).choose(1.5,1.5)[0]==0
    mask=periodic_mask(33,9)
    assert int(mask.sum())==9
    print('PASS: fixed absolute-error controller thresholds and matched mask',flush=True)


def periodic_mask(n, exact_count):
    if exact_count<1:raise ValueError('At least one exact batch required')
    arr=np.zeros(n,dtype=bool)
    pts=np.floor((np.arange(exact_count)+.5)*n/exact_count).astype(int)
    arr[pts]=True
    assert int(arr.sum())==exact_count
    return arr


def scheduled_exact(step,epoch,total_epochs):
    # ADA-GP-inspired progression, not exact source implementation.
    quartile=min(3,int(4*epoch/total_epochs))
    ratio=(4,3,2,1)[quartile]
    return (step % (ratio+1))==0


def study_signature(args):
    return dict(signature(args),calibration_batches=args.calibration_batches,
                low_q=args.low_q,high_q=args.high_q,error_window=args.error_window,
                ema_beta=args.ema_beta,emergency_ratio=args.emergency_ratio,
                max_synthetic=args.max_synthetic,target_accuracy=args.target_accuracy)


def set_epoch_lr(opts,args,global_epoch):
    total=args.warmup_epochs+args.epochs
    phase=min(global_epoch,max(0,total-1))/max(1,total-1)
    value=args.lr_min+(args.lr-args.lr_min)*.5*(1+math.cos(math.pi*phase))
    for opt in opts[:2]:
        for group in opt.param_groups:group['lr']=value
    return value


def regime(args,ck,train,val,device,seed,method,out,cal=None,mask=None):
    result_path=out/f'seed_{seed}_{method}_result.pt'
    sig=study_signature(args)
    if result_path.exists() and not args.overwrite:
        saved=torch.load(result_path,map_location='cpu',weights_only=True)
        if saved.get('signature')!=sig:
            raise ValueError(f'Cached run incompatible: {result_path}. '
                             'Use fresh --out or --overwrite.')
        print(f'Reusing run {method} seed {seed}',flush=True)
        return saved['summary'],saved['epochs'],np.array(saved['exact_mask'],dtype=bool),saved['events']
    m,p,opts=restore(args,ck,device)
    vloader=loader(val,args,seed,0,False)
    controller=Feedback(method,args,cal) if 'feedback' in method else None
    if controller is not None and cal is None:raise ValueError('Controller calibration missing')
    if method in ('matched_periodic','exact_head_mask') and mask is None:raise ValueError('Required replay mask missing')
    nsteps=math.ceil(len(train)/args.batch_size)*args.epochs
    taken=np.zeros(nsteps,dtype=bool)
    epochs=[];events=[]
    e=s=head_updates=0;quota=0;step=0
    train_seconds=eval_seconds=0.
    warmup_seconds=ck['warmup_plain_seconds'] if method in ('exact','exact_head_mask') else ck['warmup_synthetic_seconds']
    cal_seconds=cal['seconds'] if controller is not None else 0.
    first_target=None
    if ck['warmup_metrics'][-1]['accuracy'] >= args.target_accuracy:first_target=warmup_seconds
    for epoch in range(args.epochs):
        set_epoch_lr(opts,args,epoch+args.warmup_epochs)
        m.postwarm_mode();p.train()
        sync(device);ts=time.perf_counter()
        epoch_exact=epoch_syn=0
        for x,y in loader(train,args,seed,epoch+args.warmup_epochs):
            x=x.to(device,non_blocking=True);y=y.to(device,non_blocking=True)
            if method=='exact':exact=True
            elif method=='scheduled':exact=scheduled_exact(step,epoch,args.epochs)
            elif method in ('matched_periodic','exact_head_mask'):exact=bool(mask[step])
            elif controller is not None:exact=(quota==0)
            else:raise ValueError(method)
            taken[step]=exact
            if method=='exact_head_mask':
                exact_step(m,None,opts,x,y,ck['scale'],args,fit_predictor=False,head_update=exact)
                e+=1;epoch_exact+=1;head_updates+=int(exact)
            elif exact:
                # Plain BP doesn't train predictor; all other methods do.
                if method=='exact':
                    exact_step(m,None,opts,x,y,ck['scale'],args,fit_predictor=False)
                else:
                    # To calculate BOTH controller statistics from same exact batch,
                    # use one forward/trunk/head backward and predictor pre-update.
                    tr,hd,pr=opts
                    tr.zero_grad(set_to_none=True);hd.zero_grad(set_to_none=True)
                    h=m.trunk(x);h.retain_grad()
                    loss=F.cross_entropy(m.head(h),y);loss.backward()
                    truth=h.grad.detach()*len(y)/ck['scale']
                    with torch.no_grad():
                        approx=p(h.detach(),y)
                        error_abs=float((approx-truth).square().mean().sqrt())
                        error_rel=rel_error(approx,truth)
                    pr.zero_grad(set_to_none=True)
                    pl=predictor_loss(p(h.detach(),y),truth,args.cosine_weight)
                    pl.backward()
                    nn.utils.clip_grad_norm_(p.parameters(),args.clip_grad);pr.step()
                    nn.utils.clip_grad_norm_(m.parameters(),args.clip_grad);tr.step();hd.step()
                    if controller is not None:
                        quota,details=controller.choose(error_abs,error_rel)
                        events.append({'seed':seed,'epoch':epoch+1,'step':step,**details})
                e+=1;epoch_exact+=1;head_updates+=1
            else:
                synthetic_step(m,p,opts,x,y,ck['scale'],args)
                s+=1;epoch_syn+=1
                if controller is not None:quota-=1
            step+=1
        sync(device);train_seconds+=time.perf_counter()-ts
        sync(device);vs=time.perf_counter();v=evaluate(m,vloader,device)
        sync(device);eval_seconds+=time.perf_counter()-vs
        total=warmup_seconds+cal_seconds+train_seconds+eval_seconds
        if first_target is None and v['accuracy'] >= args.target_accuracy:first_target=total
        epochs.append({'seed':seed,'method':method,'epoch':args.warmup_epochs+epoch+1,
                       'accuracy':v['accuracy'],'ce':v['ce'],
                       'exact_epoch':epoch_exact,'synthetic_epoch':epoch_syn,
                       'exact_frac':e/max(1,step),'wall_seconds':total,
                       'controller_0':controller.choices[0] if controller else '',
                       'controller_2':controller.choices[2] if controller else '',
                       'controller_4':controller.choices[4] if controller else ''})
        print(f'  seed {seed} {method:>17} {epoch+1:>2}/{args.epochs} '
              f'acc={v["accuracy"]:.3f} exact={e/max(1,step):.2%} '
              f'choices={dict(controller.choices) if controller else "-"}',flush=True)
    if method in ('matched_periodic','exact_head_mask'):
        assert np.array_equal(mask,taken)
    if controller and len([v for v in controller.choices.values() if v])<2:
        print('WARNING: controller used fewer than two switching actions.',flush=True)
    summary={'seed':seed,'method':method,'final_accuracy':epochs[-1]['accuracy'],
             'best_accuracy':max(v['accuracy'] for v in epochs),
             'final_ce':epochs[-1]['ce'],'exact_batches':e,
             'synthetic_batches':s,'head_updates':head_updates,
             'exact_fraction':e/nsteps,'warmup_seconds':warmup_seconds,
             'calibration_seconds':cal_seconds,'postwarm_train_seconds':train_seconds,
             'postwarm_eval_seconds':eval_seconds,
             'wall_seconds':warmup_seconds+cal_seconds+train_seconds+eval_seconds,
             'seconds_to_target':first_target if first_target is not None else '',
             'controller_states':len([v for v in controller.choices.values() if v]) if controller else ''}
    torch.save({'signature':sig,'summary':summary,'epochs':epochs,'exact_mask':taken.tolist(),
                'events':events,'model':{k:v.detach().cpu() for k,v in m.state_dict().items()}},result_path)
    if events:save_csv(out/f'seed_{seed}_{method}_controller.csv',events)
    return summary,epochs,taken,events


def profile(args,ck,train,device,seed,out):
    """Cost preflight: no data-loading; synchronized end-to-end steps after warm-up.

    This is only a feasibility gate. Final claims must use complete wall time.
    """
    bat=[]
    for i,(x,y) in enumerate(loader(train,args,seed,777)):
        if i>=args.profile_batches+2:break
        bat.append((x.to(device),y.to(device)))
    if len(bat)<3:raise RuntimeError('Insufficient profile batches')
    trials=[]
    for method in ('exact_plain','exact_plus_predictor','synthetic'):
        m,p,ops=restore(args,ck,device);m.postwarm_mode();p.train()
        if device.type=='cuda':torch.cuda.reset_peak_memory_stats(device)
        for i,(x,y) in enumerate(bat):
            sync(device);ts=time.perf_counter()
            if method=='exact_plain':exact_step(m,None,ops,x,y,ck['scale'],args,fit_predictor=False)
            elif method=='exact_plus_predictor':exact_step(m,p,ops,x,y,ck['scale'],args,fit_predictor=True)
            else:synthetic_step(m,p,ops,x,y,ck['scale'],args)
            sync(device);dt=time.perf_counter()-ts
            if i>=2:trials.append({'variant':method,'milliseconds':dt*1000,'seed':seed,
                                   'cuda_peak_allocated_mb':(torch.cuda.max_memory_allocated(device)/1048576
                                                             if device.type=='cuda' else '')})
    save_csv(out/f'seed_{seed}_profile.csv',trials)
    med={name:statistics.median(r['milliseconds'] for r in trials if r['variant']==name)
         for name in ('exact_plain','exact_plus_predictor','synthetic')}
    values=[]
    denom=med['exact_plus_predictor']-med['synthetic']
    # Maximum exact-BP fraction that could yield a benefit under this naive
    # timing model. Excludes validation, data loading, warm-up, and calibration.
    break_even_fraction=(med['exact_plain']-med['synthetic'])/denom if denom>0 else float('nan')
    for frac in (.2,.35,.5):
        estimated=frac*med['exact_plus_predictor']+(1-frac)*med['synthetic']
        values.append({'seed':seed,'exact_fraction':frac,'estimated_mixed_ms_per_batch':estimated,
                       'exact_plain_ms_per_batch':med['exact_plain'],
                       'predictor_training_overhead_ms':med['exact_plus_predictor']-med['exact_plain'],
                       'synthetic_batch_savings_ms':med['exact_plain']-med['synthetic'],
                       'modeled_break_even_exact_fraction':break_even_fraction,
                       'estimated_speedup_ratio':med['exact_plain']/estimated,
                       'predicted_faster':estimated<med['exact_plain']})
    save_csv(out/f'seed_{seed}_feasibility.csv',values)
    print(f'Preflight ms/batch {med}; modeled break-even exact fraction={break_even_fraction:.1%}',flush=True)
    for r in values:print(f'  exact={r["exact_fraction"]:.0%}: modeled speedup={r["estimated_speedup_ratio"]:.3f}x',flush=True)
    if not any(r['predicted_faster'] for r in values):
        print('WARNING: no modeled speed benefit for 20/35/50% exact-BP usage. '
              'Consider stopping before full study.',flush=True)
    return values


def aggregate(all_summaries,out):
    groups={}
    for r in all_summaries:groups.setdefault(r['method'],[]).append(r)
    rows=[]
    for method,items in groups.items():
        rec={'method':method,'seeds':len(items)}
        for key in ('final_accuracy','wall_seconds','exact_fraction','seconds_to_target'):
            nums=[float(x[key]) for x in items if x[key]!='']
            rec[key+'_mean']=float(np.mean(nums)) if nums else ''
            rec[key+'_sd']=float(np.std(nums,ddof=1)) if len(nums)>1 else ''
        rows.append(rec)
    save_csv(out/'summary_aggregate.csv',rows)
    paired=[]
    for seed in sorted(set(r['seed'] for r in all_summaries)):
        lookup={r['method']:r for r in all_summaries if r['seed']==seed}
        if 'absolute_feedback' not in lookup:continue
        ref=lookup['absolute_feedback']
        for comp in ('matched_periodic','relative_feedback','scheduled','exact'):
            if comp in lookup:
                b=lookup[comp]
                paired.append({'seed':seed,'baseline':comp,
                               'accuracy_diff_abs_minus_base':ref['final_accuracy']-b['final_accuracy'],
                               'wall_seconds_diff_abs_minus_base':ref['wall_seconds']-b['wall_seconds'],
                               'exact_fraction_diff':ref['exact_fraction']-b['exact_fraction']})
    save_csv(out/'paired_absolute_vs_baselines.csv',paired)


def plot_results(epoch_rows,out):
    """Optional figures; CSV remains the authoritative result."""
    if not epoch_rows:return
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib not installed; skipping optional plots.',flush=True)
        return
    for abscissa, filename, label in (
        ('epoch','accuracy_vs_epoch.png','Classifier epoch'),
        ('wall_seconds','accuracy_vs_walltime.png','Elapsed wall time (s; includes warm-up)'),
    ):
        fig,ax=plt.subplots(figsize=(8,5))
        for method in METHODS:
            subset=[r for r in epoch_rows if r['method']==method]
            if not subset:continue
            if abscissa=='epoch':
                grouped={}
                for r in subset:grouped.setdefault(int(r['epoch']),[]).append(float(r['accuracy']))
                xs=sorted(grouped);ys=[float(np.mean(grouped[x])) for x in xs]
            else:
                # For wall-time plot show each seed separately: elapsed time differs
                # across methods and seeds, so averaging by epoch distorts the x axis.
                for seed in sorted(set(r['seed'] for r in subset)):
                    rs=sorted((r for r in subset if r['seed']==seed),key=lambda q:q['epoch'])
                    ax.plot([float(r[abscissa]) for r in rs],
                            [float(r['accuracy']) for r in rs],alpha=.45,
                            label=method if seed==rs[0]['seed'] and seed==min(r['seed'] for r in subset) else None)
                continue
            ax.plot(xs,ys,marker='o',label=method)
        ax.set(xlabel=label,ylabel='Validation accuracy',title='CIFAR-10 synthetic-gradient training')
        ax.grid(alpha=.2);ax.legend(fontsize=8);fig.tight_layout()
        fig.savefig(out/filename,dpi=160);plt.close(fig)


def parse_args():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('--smoke-test',action='store_true')
    ap.add_argument('--device',choices=('cpu','cuda','auto'),default='auto')
    ap.add_argument('--mode',choices=('preflight','full'),default='preflight')
    ap.add_argument('--out',default='phase_d_resnet')
    ap.add_argument('--data-dir',default='data')
    ap.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    ap.add_argument('--warmup-epochs',type=int,default=4)
    ap.add_argument('--epochs',type=int,default=12)
    ap.add_argument('--batch-size',type=int,default=32)
    ap.add_argument('--width',type=int,default=64)
    ap.add_argument('--blocks',type=int,default=2)
    ap.add_argument('--predictor-width',type=int,default=32)
    ap.add_argument('--lr',type=float,default=0.04)
    ap.add_argument('--lr-min',type=float,default=0.001)
    ap.add_argument('--pred-lr',type=float,default=0.0008)
    ap.add_argument('--weight-decay',type=float,default=0.0005)
    ap.add_argument('--clip-grad',type=float,default=5.0)
    ap.add_argument('--cosine-weight',type=float,default=0.2)
    ap.add_argument('--calibration-batches',type=int,default=48)
    ap.add_argument('--profile-batches',type=int,default=12)
    ap.add_argument('--error-window',type=int,default=96)
    ap.add_argument('--low-q',type=float,default=.33)
    ap.add_argument('--high-q',type=float,default=.67)
    ap.add_argument('--ema-beta',type=float,default=.25)
    ap.add_argument('--emergency-ratio',type=float,default=2.0)
    ap.add_argument('--max-synthetic',type=int,default=4)
    ap.add_argument('--target-accuracy',type=float,default=.70)
    ap.add_argument('--workers',type=int,default=0)
    ap.add_argument('--pin-memory',action='store_true')
    ap.add_argument('--deterministic',action='store_true')
    ap.add_argument('--overwrite',action='store_true')
    return ap.parse_args()


def main():
    args=parse_args()
    if args.smoke_test:
        args.width=4;args.blocks=1;args.batch_size=16
        args.warmup_epochs=1;args.epochs=1;args.calibration_batches=3
        args.profile_batches=2;args.seeds=args.seeds[:1]
        args.workers=0;args.target_accuracy=.1
    if args.device=='cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable in this Python environment. Check torch.cuda.is_available()')
    device=torch.device('cuda' if args.device=='auto' and torch.cuda.is_available() else
                        'cpu' if args.device=='auto' else args.device)
    if args.deterministic:seed_everything(9231,True)
    if device.type=='cuda':torch.backends.cudnn.benchmark=not args.deterministic
    else:torch.set_num_threads(min(4,torch.get_num_threads()))
    out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    print('Device:',device,'mode:',args.mode,'seeds:',args.seeds,flush=True)
    if args.smoke_test:
        audit_injection(device)
        audit_controller()
    train,val=train_data(args)
    (out/'config.json').write_text(json.dumps(vars(args),indent=2),encoding='utf-8')
    summaries=[];epoch_rows=[];feasibility=[]
    for seed in args.seeds:
        ck=warmup(args,train,val,device,seed,out)
        feasibility+=profile(args,ck,train,device,seed,out)
        if args.mode=='preflight':continue
        cal=calibration(args,ck,train,device,seed,out)
        for method in ('exact','scheduled','relative_feedback','absolute_feedback'):
            summ,rows,mask,ev=regime(args,ck,train,val,device,seed,method,out,cal=cal)
            summaries.append(summ);epoch_rows+=rows
            if method=='absolute_feedback':absolute_mask=mask
        mask=periodic_mask(len(absolute_mask),int(absolute_mask.sum()))
        for method in ('matched_periodic','exact_head_mask'):
            appropriate=mask if method=='matched_periodic' else absolute_mask
            summ,rows,_,_=regime(args,ck,train,val,device,seed,method,out,mask=appropriate)
            summaries.append(summ);epoch_rows+=rows
        save_csv(out/'summary.csv',summaries)
        save_csv(out/'epochs.csv',epoch_rows)
        aggregate(summaries,out)
    save_csv(out/'feasibility.csv',feasibility)
    if args.mode=='full':plot_results(epoch_rows,out)
    if args.mode=='preflight':print('Preflight finished. Inspect feasibility.csv before --mode full.',flush=True)
    else:print('Study finished. Inspect summary_aggregate.csv and paired_absolute_vs_baselines.csv.',flush=True)


if __name__=='__main__':main()
