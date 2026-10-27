# Phase D v4 implementation and experimental plan

Status: planning complete; implementation and experiments require separate authorization.

This document specifies future work. Creating this document does not authorize changes to the training code, execution of experiments, or creation of `AGENTS.md`. Preserve the original source, specification, data, and results. Resolve the approval items in Section 12 before launching the corresponding scientific runs.

## 1. Objective, evidence, and scope

Answer these questions in order:

1. **Predictor usefulness:** do learned synthetic gradients improve learning over zero-gradient and skipped-update controls under the same exact-update mask?
2. **Controller usefulness:** does error-driven switching improve learning over schedules with matched budgets and controlled update spacing?
3. **Training efficiency:** does a deployable method improve accuracy at comparable actual elapsed time, or reach a predefined accuracy sooner, compared with exact BP?

The accuracy-budget optimizer, improved learned controllers, alternative predictor architectures, and broad architecture/dataset sweeps are future research. Do not implement them in this phase.

### 1.1 Authoritative local evidence

- Specification: `RESEARCH_SPEC.md`, including its final corrections.
- Actual reference source: `gradient_phase_d_resnet_v3.py`.
- Reference outputs: `cifar_d_resnet_v3/`.
- Historical context: `archive/`, especially the Phase D v1/v2 and R1/R1+ sources and results.
- The specification and README contain older names without `_v3`; use the actual paths above. Do not invent another experiment from the naming difference.

The preceding read-only audit established:

| Finding | Evidence and limit |
|---|---|
| Relative feedback final mean accuracy is 85.0467%, versus 89.8933% for exact BP | Recomputed from all three seeds; deficit 4.8467 percentage points (pp). |
| Recorded fixed-epoch speedups are approximately 1.15–1.22x | Historical laptop measurements; not a controlled T4 benchmark. |
| Exact BP first exceeds feedback's final accuracy at approximately 624/1,035/662 seconds; feedback finishes at 810/1,140/744 seconds | Epoch-resolution observations for seeds 0/1/2. Existing evidence does not establish better time to comparable accuracy. |
| Relative feedback uses 33.894% exact batches after warm-up, approximately 50.42% including warm-up | Report both denominators. These fractions are not FLOP or elapsed-time fractions. |
| Existing periodic and head-mask controls follow absolute feedback | Saved masks, events, and count totals were verified for all 18 finished method checkpoints. |
| All methods freeze downstream-head BatchNorm buffers after warm-up | Confirmed in source and saved buffers. Exact BP here is not the usual full train-mode ResNet reference. |
| Basic injection works | Existing checks and small CPU checks at batch sizes 1, 8, and 32 passed; maximum scaled-injection discrepancy was below 3e-8. This is not complete optimizer/resume validation. |
| Relative feedback's logged denominator floor never activates | The floor is not an established explanation for its poor predictor quality. Absolute-feedback seed 2 does encounter it. |
| Warm-up predictor quality is weak in both downstream BN modes | Three diagnostic minibatches per seed only. Switching the head to evaluation mode did not worsen relative error in those probes; no causal conclusion about collapse follows. |
| Finished checkpoints omit predictor and optimizer/controller state | They allow skipping finished methods, not faithful interrupted-training recovery or final-predictor diagnosis. Warm-up checkpoints contain more state. |

Other reconciliations: the reported old virtual-SGD improvement of about 26% describes particular settings, not a universal ratio; archived fresh and drifted predictor results vary widely. Recompute profile medians from all samples rather than copying approximate values in prose. No historical run should be described as confirmatory evidence for a newly selected protocol.

### 1.2 Classification of issues

- **Confirmed implementation/reporting defects or gaps:** inadequate result-cache identity; missing resumable method state; time-to-target logic records the end of warm-up rather than the first qualifying warm-up epoch; incomplete metadata and diagnostics. No fundamental injection bug was found.
- **Confounders:** unmatched budgets, early exact-update allocation, sparse head training, clipping differences, temporal spacing, fixed method order/hardware drift, and nonstandard exact-BP normalization.
- **Hypotheses:** useful synthetic gradients, useful error information about future updates, normalization-induced instability, harmful predictor magnitude, and vanishing-gradient false confidence.
- **Optional improvements:** new reliability safeguards, alternative scaling/clipping, alternative normalization, predictor optimization.
- **Future extensions:** longer horizons, new architectures/datasets, broader schedule sweeps, official test reporting, and accuracy-budget optimization.

Do not relabel a confounder as a software bug, or a hypothesis as a demonstrated cause.

## 2. Minimal implementation layout and boundaries

Only the following new files are initially needed when implementation is authorized:

| File | Responsibility |
|---|---|
| `gradient_phase_d_resnet_v4.py` | CLI, models, data construction, optimizer setup, training steps, controllers, masks, evaluation, diagnostic probes, experiment orchestration. Use focused functions and a few small state/config records, not a plugin framework. |
| `phase_d_v4_io.py` | Canonical identities, atomic checkpoint persistence, RNG serialization, stable metric schemas, run manifests, timing ledger. Must not import the training module. |
| `analyze_phase_d_v4.py` | Read-only result validation, paired statistics, time-to-target tables, Pareto comparisons, optional plots. No training or automatic tuning. |
| `tests/test_phase_d_v4.py` | Standard-library `unittest` correctness, replay, resume, accounting, and analysis tests using tiny models/data. |
| `configs/phase_d_v4_reference.json` | Explicit frozen reference defaults and schema version. Scientific study manifests are generated separately after approval. |
| `COLAB_PHASE_D_V4.md` | Setup, dependency verification, test/smoke/preflight/run/resume commands, output meanings, persistence and known limits. Write alongside working implementation, not in this planning task. |

Use Python, PyTorch, torchvision, NumPy, and the standard library. Matplotlib is optional for plots. No mandatory pandas, SciPy, experiment-tracking service, distributed training, or configuration framework. If a statistical computation needs a distribution quantile, use a small documented critical-value table for the supported seed counts and fail on unsupported counts rather than silently approximating it.

Keep v3 source/results read-only. All generated files go under a new root such as `runs_phase_d_v4/<study_id>/<run_id>/`. Refuse an output path inside `archive/`, `data/`, or `cifar_d_resnet_v3/`, or one that would overwrite a source/specification file. Do not implement a broad destructive `--overwrite`; use fresh run IDs and explicit compatible resume.

Archive preservation means hashing and referencing existing artifacts, not copying the complete dataset or result archive into every run. No Git repository is assumed; source SHA-256 hashes are mandatory, Git revision is optional when available.

## 3. Frozen reference contract

### 3.1 Data, architecture, and hyperparameters

| Setting | Required initial value/behavior |
|---|---|
| Dataset | CIFAR-10 official training set only; deterministic 45,000/5,000 train/validation partition using `torch.randperm` with generator seed 71503. Save index arrays and hashes. |
| Final test set | Not loaded by routine training, diagnostics, calibration, or development analysis. Final test evaluation is a separately authorized later phase. |
| Augmentation | Random crop to 32 with padding 4, random horizontal flip, tensor conversion, normalization mean `(0.4914,0.4822,0.4465)`, std `(0.2470,0.2435,0.2616)`. Validation has no random augmentation. |
| Architecture | CIFAR ResNet-18, width 64, two BasicBlocks per stage, 3x3 stem; split after layer 2. Boundary shape `[B,128,16,16]`. |
| Predictor | Existing label embedding plus 1x1 bottleneck, depthwise 3x3, 1x1 output; bottleneck width 32. Preserve activations, biases, label addition, and detach semantics. |
| Batch size | 32, `drop_last=False`; final training batch has 8 examples. 1,407 batches/epoch. |
| Horizon | Four exact warm-up epochs plus twelve post-warm-up epochs, total 16. |
| Model optimizers | Separate trunk/head SGD: momentum 0.9, weight decay 0.0005, dampening 0, Nesterov false. |
| Predictor optimizer | AdamW: LR 0.0008, weight decay 0.0001, betas `(0.9,0.999)`, eps `1e-8`; record resolved optimizer defaults/backend choices. |
| Model LR | Epoch-indexed cosine from 0.04 to 0.001 across all 16 epochs; no reset at warm-up boundary or on resume. |
| Predictor objective | Coordinate MSE plus 0.2 times per-example cosine loss; valid target flattened norm >1e-5; cosine eps 1e-8. |
| Clipping | Global L2 gradient-norm limit 5.0, with historical parameter groups retained as specified below. |
| Precision | FP32; no AMP, gradient accumulation, `torch.compile`, or automatic batch-size changes. Record matmul/cuDNN TF32 flags; use TF32 off for controlled v4 runs, and distinguish this backend setting from historical unknowns. |
| Data loading | Workers 0 and pin_memory false initially. Performance variants require a separately fingerprinted condition applied to all comparators. |
| Controller | History window 96; quantiles 0.33/0.67; maximum quota 4; absolute EMA old-value weight 0.25; emergency multiplier 2.0. |
| Calibration/profile | 48 training-stream calibration batches; 12 timed profile batches after 2 unreported settling batches. |

For global epoch `e=0,...,15`, use:

`lr(e) = 0.001 + 0.5*(0.04-0.001)*(1 + cos(pi*e/15))`.

For configurable toy/longer horizons, use the exact v3 generalization: `H=warmup_epochs+postwarm_epochs`, `phase=min(e,max(0,H-1))/max(1,H-1)`, and `lr=lr_min+0.5*(lr_start-lr_min)*(1+cos(pi*phase))`. Changing `H` changes the whole LR trajectory and therefore defines a new experimental condition, not merely an appended continuation.

Both model optimizers use this LR regardless of how often they step. Sparse-head comparisons intentionally preserve the original epoch clock; compensating head LR or decay for missed updates is a new condition, not a correction.

### 3.2 Identity and reproducibility

Reproduce v3 initialization with seed `18371 + seed`. Reproduce its training epoch loader/augmentation seed `245600 + 100000*seed + global_epoch`. Use the original dedicated DataLoader generator and reseeding behavior for workers 0. Do not silently replace this with a new per-example augmentation algorithm.

All methods within a paired comparison must see identical sample order and augmentation for the same epoch/batch. These models contain no dropout; controller/mask randomization must use a separate generator and must never consume the global augmentation RNG. Diagnostic probes must run on copies with restored RNG or in separate processes. Instrument the loader in tests to hash actual input tensors and indices across methods and resumed runs.

Use strict deterministic algorithms for the controlled v4 execution mode, record cuDNN flags, and fail with a clear unsupported-operation error. v3 used `warn_only=True`; expose that only as an explicitly labeled compatibility option. Do not promise bitwise agreement across PyTorch versions, operating systems, or GPU architectures. Historical checkpoints and new T4 runs must not be mixed into a timing pair.

### 3.3 Gradient definitions and update order

Let `h=f_theta(x)` and `L=mean_i CE(g_phi(h)_i,y_i)`, with actual batch size `B`.

1. Compute `g_h = dL/dh` before clipping model parameter gradients.
2. Predictor target is `z = B*g_h/s`, where `s` is the fixed positive v3 scale.
3. Predicted boundary gradient for injection is `g_hat_h = s*P(h.detach(),y)/B`.
4. Inject with `h.backward(g_hat_h.detach())`; gradients must not enter predictor parameters from this operation.

`B*g_h` is the gradient of the summed batch loss. It is **not** generally an independent per-example derivative when the downstream network contains train-mode BatchNorm or other cross-example operations. The scaling identity and VJP injection remain valid in that case; do not test them using an invalid independence assumption.

Preserve scale construction: on the first warm-up training batch, evaluate the initialized full model with frozen BN buffers, compute the boundary gradient of summed CE, and set `s=max(RMS(gradient),1e-6)`. Restore train mode before the actual warm-up update. Persist `s` without recomputation on resume. This scale is estimated from an untrained eval-mode model, so log normalized target magnitude and objective components; scale adequacy is a hypothesis to diagnose, not a change in the reference.

Exact update order:

1. Clear trunk/head gradients with `set_to_none=True`.
2. Run trunk and head; retain the boundary gradient only when needed; backward the classifier loss once.
3. For predictor-enabled methods, compute pre-fit predictor error, then fit the predictor to detached `z`; clip predictor gradients and step AdamW once.
4. Clip joint trunk/head parameter gradients; step trunk SGD once and head SGD only when prescribed.
5. The controller uses the pre-fit error to choose the quota for **future** synthetic batches. The measured batch is already exact.

Initially retain v3's two predictor evaluations on exact predictor-training steps. A later equivalent single-evaluation optimization requires a separate tested backend option, identical detached target/error values and parameter updates, and benchmarks for every affected method. Do not optimize the compatibility path before equivalence is established.

On learned synthetic steps, run only the trunk and predictor, inject the scaled prediction, clip trunk gradients, and step trunk SGD. Never run the head, fit the predictor, or step head SGD.

Predictor parameter `.grad` buffers can still contain values from the preceding exact step in v3. The invariant is **no new predictor backward contribution and no predictor parameter/optimizer-state change** on a synthetic step, not that every existing `.grad` buffer is necessarily `None`. Test by snapshotting existing buffers or clearing them deliberately in the isolated test fixture; do not change historical update semantics to satisfy an incorrect assertion.

### 3.4 Optimizer and clipping semantics

For each parameter with an allocated gradient, SGD conceptually forms `d=g+lambda*theta`, updates momentum with `v=0.9*v+d` (PyTorch's first-buffer initialization is `v=d`), and applies `theta -= lr*v`. Clipping is applied to data gradients **before** SGD adds weight decay. Parameters with `grad=None` are skipped by PyTorch optimizers; a tensor of zeros is different.

Define three synthetic-slot behaviors:

| Behavior | Trunk forward / BN | Trunk backward | Trunk optimizer | Head optimizer |
|---|---|---|---|---|
| `learned` | Yes, train mode | Inject prediction | Step | Do not step |
| `zero` | Yes, train mode | Inject `zeros_like(h)` to allocate true zero gradients through the same graph | Step: momentum/decay remain active | Do not step |
| `skip` | Yes, train mode, under no_grad when possible | None | Do not call step; parameters and momentum buffers unchanged | Do not step |

All three consume the same batch and preserve trunk BN forward-buffer updates. A method that skips the entire batch, including the forward pass and BN update, is a different future control. Never implement `skip` by supplying zero gradients or by setting LR to zero.

Retain historical clipping for primary attribution experiments: joint trunk+head clipping on exact steps, trunk-only clipping on learned/zero slots. In an exact-trunk sparse-head control, include head gradients in joint clipping even when the head optimizer will not step, then clear them. This preserves v3 semantics but means attribution is conditional on that clipping protocol. Log preclip norms, applied coefficients, and clipping frequencies. A trunk/head-separate clipping ablation is conditional; never silently substitute it in the primary methods.

### 3.5 Normalization, warm-up, and calibration

- `frozen_head_postwarm`: all BN layers train during warm-up; afterward trunk BN trains and head BN uses frozen running statistics, with head affine parameters still trainable on exact slots. This is the primary v3-compatible regime.
- `standard_bn`: all classifier BN layers train on every exact training batch, including after warm-up; validation uses eval mode. Initially this is required only for conventional exact BP.
- Synthetic paths cannot update head BN without executing the head. Do not claim normalization parity with conventional exact BP; report it as a separate practical reference.
- Calibration uses a disposable copy of warm-up state in post-warm-up mode, on training minibatches only, with no weight or optimizer updates. Trunk BN buffers on that copy may change, as in v3; discard the copy. Verify the reusable warm-up state is unchanged.
- Save the separate plain and predictor-trained warm-up states and timings. On fixed inputs they must produce the same classifier parameters, BN buffers, and model optimizer buffers within declared tolerances. Predictor training must not alter classifier gradients.
- A no-predictor practical method should instantiate no predictor and should not pay its initialization/calibration/training cost. A causal shadow-predictor control intentionally retains those costs; never confuse these records.

### 3.6 Historical controller definitions

With predicted normalized target `p` and true normalized target `z`:

- `error_abs = sqrt(mean((p-z)^2))` across the entire batch and all coordinates.
- `error_relative = error_abs / max(sqrt(mean(z^2)),0.05)`.
- Relative feedback compares the current pre-fit error with linear-interpolated NumPy quantiles of **previous** exact-batch errors, initialized from calibration history. Quota is 4 at/below the lower threshold, 2 at/below the upper threshold, otherwise 0; append current error afterward.
- Absolute feedback sets fixed calibration quantiles and median. EMA is `0.25*previous + 0.75*current` (first value=current). If EMA exceeds twice calibration median, quota=0; otherwise thresholds choose 4/2/0. Do not reverse the EMA weights.
- Apply the maximum-quota cap after choosing. Quota persists across epoch boundaries; do not reset it at an epoch boundary. Synthetic batches decrement it; exact batches choose a fresh value. Unused final quota is logged, not counted as executed batches.
- Historical scheduled switching uses ratios `(4,3,2,1)` in successive post-warm-up epoch quartiles and `global_postwarm_step % (ratio+1) == 0`. The step counter does not reset each epoch.

Historical method identifiers remain `exact`, `scheduled`, `relative_feedback`, `absolute_feedback`, `matched_periodic`, and `exact_head_mask`, with `algorithm_version="v3"` and explicit normalization/backend metadata. The last two still reference absolute feedback. New controls have new identifiers. Required v4 controls must never inherit misleading historical names.

## 4. Experimental controls and schedule construction

Represent a schedule as a one-dimensional boolean array of length `N=sum_e n_e`; true means an exact/head-update slot. Persist mask bytes, length, epoch offsets, SHA-256, source run/checkpoint identity, source seed, and construction algorithm/version. Distinguish the schedule mask from actual trunk-exact and head-update masks: the exact-trunk sparse-head control has an all-true trunk-exact mask but a sparse head-update mask.

### 4.1 Predictor attribution methods

Use relative feedback's recorded mask for all replay controls, with the same warm-up classifier/predictor/optimizer states, data stream, LR, normalization, and clipping.

| New identifier | Exact slots | Synthetic slots | Interpretation |
|---|---|---|---|
| `relative_replay_learned` | Exact classifier update and predictor fitting | Learned injection | Replay equivalence check; should reproduce its source trajectory. |
| `relative_replay_zero_shadow` | Same exact-step operations as learned replay | Infer predictor and discard output; inject zero; step trunk | Causal predictor-removal control retaining predictor supervision/inference overhead. |
| `relative_replay_skip_shadow` | Same exact-step operations as learned replay | Run trunk and predictor, discard prediction; no trunk optimizer step | Separates injection/momentum effects from skipped updates while retaining BN forward behavior. |
| `relative_replay_exact_trunk` | Exact trunk and head updates; no predictor fitting | Exact trunk backward/update; do not step head optimizer | Head-update control; expensive diagnostic, not a speed candidate. |

Shadow predictors receive their own trajectory's exact-gradient targets. They do not borrow predictions, weights, or labels from the source run beyond the shared initial state and replay mask. Their fitting cannot affect classifier parameters because inputs/targets are detached. Do not force shadow predictor weights to equal the source after trajectories diverge.

Learned replay equivalence is a correctness check, not an additional independent sample. For plots of replay-vs-zero accuracy, the source curve may stand in for replay only after equivalence has passed; use the replay's actual measured time when comparing replay execution costs.

Practical `*_zero_fast` / `*_skip_fast` conditions remove the predictor entirely and use plain warm-up. Their classifier/optimizer trajectory should equal the corresponding shadow condition at fixed mask, subject to identical RNG/data state. Calibration is unnecessary for a supplied mask. These are conditional Stage 4 benchmarks; a source-derived mask still makes them retrospective, not deployable. Pair them with an approved independently specified fixed schedule before claiming a practical sparse-BP competitor.

### 4.2 Periodic and front-loaded controls

New periodic mask helper `make_periodic_mask(n,k)` accepts `0<=k<=n`. For `k=0`, all false. Otherwise set positions `floor(j*n/k)` for `j=0,...,k-1` true. This gives exactly `k` unique positions, starts with an exact step, and avoids adding an initial synthetic-only prefix. It intentionally differs from v3's midpoint helper; preserve the midpoint helper for historical `matched_periodic` reproduction.

- `relative_periodic_total`: use `k=sum(source_mask)` over the complete post-warm-up horizon. Retrospective total-budget diagnostic; not per-epoch matched.
- `relative_periodic_epoch`: apply the new helper separately using each source epoch's `k_e`. Retrospective per-epoch-budget diagnostic; matches early exact allocation but not burst lengths or within-epoch phase.
- `fixed_frontloaded`: deployable, specified before evaluation; fractions 0.40/0.35/0.30/0.30 for the four three-epoch phases. Set `k_e=floor(n_e*f_e+0.5)` and use the new periodic helper per epoch. This is approximately budget-comparable, not exactly source-budget-matched. Report the actual difference.
- `fixed_uniform_third`: deployable secondary baseline with `k_e=floor(n_e/3+0.5)` per epoch. This provides a simple practical comparator if feedback looks no better than its matched controls.

Do not tune front-loading fractions on the fresh evaluation seeds. A later schedule sweep is development work with a new protocol version.

### 4.3 Randomization with exact invariants

Primary randomized control: `relative_gap_permutation_epoch`.

For each epoch, find exact indices `a_0<...<a_(k-1)`. If `k>=2`:

1. Preserve the leading synthetic prefix `L=a_0` and trailing suffix `T=n-1-a_(k-1)`.
2. Extract internal gaps `d_j=a_(j+1)-a_j-1` for `j=0,...,k-2`; zeros represent consecutive exact steps.
3. Permute this gap vector with a dedicated seeded NumPy generator; reconstruct `L` false entries, one true, then each permuted gap followed by one true, then `T` false entries.
4. If `k=0` or `k=1`, return the source epoch unchanged and report that no randomization is possible.

This preserves each epoch's length, exact count, first/last exact positions, internal gap multiset, and leading/trailing synthetic lengths. Therefore cross-epoch synthetic runs are also preserved. It removes the association between observed error and the placement of internal run lengths. It does **not** preserve the ordering of run lengths or all higher-order temporal correlations; a difference is evidence about temporal alignment, not proof of a perfectly isolated causal error statistic.

Use permutation RNG seed `900000 + 1000*classifier_seed + 100*replicate + postwarm_epoch`. Pilot replicate=0. Locked controller comparison uses replicates 0,1,2 and averages them within classifier seed before statistical comparisons. These replicas are not independent classifier seeds. Do not repeatedly redraw until a desired result appears. Log identical permutations and the fraction of moved gaps; an unchanged permutation is valid, and an entirely unrandomizable mask makes this contrast uninformative.

Secondary optional spacing control: uniformly select `k_e` distinct exact positions per epoch using a dedicated generator, fixing the first slot exact when `k_e>0` and drawing the remainder from later slots. This preserves counts but not gap lengths; give it a different identifier and interpretation. Not required in the initial matrix.

### 4.4 Data leakage and source dependence

All mask-derived controls are **retrospective diagnostics**. Label them that way in manifests, CSVs, plots, and summaries. They may depend on a feedback trajectory on the same training stream, but never on validation/test labels or which validation epoch looked best.

A feedback-derived mask may favor the source trajectory. Consequently replay results describe predictor utility under that mask, not universal utility under arbitrary schedules. If the result is positive, repeat learned-vs-zero/skip under a fixed schedule before generalizing. Keep that follow-up conditional to control compute cost.

## 5. Diagnostics and evaluation contract

### 5.1 Low-overhead telemetry in every run

Record per epoch: classifier epoch and post-warm-up epoch, LR, samples/batches seen, exact trunk backwards, synthetic backwards, zero backwards, trunk/head optimizer steps, predictor fits/inferences, exact fraction both post-warm-up and full-run, validation CE/accuracy, timing components, checkpoint status, and failure flags.

Controller events: global/epoch batch indices, pre-fit absolute/relative errors, thresholds, EMA/history state summary, chosen quota, and executed mask entry. Count actual executed quotas separately from selected quotas. Log target RMS and floor activation on controller exact steps; this uses the existing forward/backward data, not extra supervision.

Avoid extra host synchronization per tensor or batch. Accumulate counters and detached scalar tensors and transfer in bounded chunks or at epoch end. Existing controller decisions require host-readable errors; account for that cost. Bound telemetry memory rather than retaining graphs or full activations. Synthetic steps have no classifier loss; store null, not a fabricated CE value.

Detailed gradient diagnostics are performed in a separate diagnostic execution mode. Any sampled diagnostic collection retained in a timed run must be declared, consistently enabled across relevant methods, and included in elapsed time.

### 5.2 Gradient metrics

Compute in raw boundary-gradient units and, where needed, normalized target units. Name both units explicitly.

For each example's flattened boundary tensors `g` and `g_hat`, log:

- cosine `dot(g_hat,g)/(norm(g_hat)*norm(g))` only when both norms exceed the declared floor;
- true norm, predicted norm, residual norm;
- relative residual `norm(g_hat-g)/max(norm(g),epsilon)`;
- norm ratio `norm(g_hat)/max(norm(g),epsilon)`;
- projection `dot(g_hat,g)/max(norm(g)^2,epsilon^2)`.

Use normalized target-unit diagnostic floor `1e-8`; raw-unit floor is `s*1e-8/B` for gradients of the mean loss. Report invalid/near-zero counts; never silently discard them. This is separate from the unchanged controller RMS floor 0.05. Also report batch-global RMS/relative residual and batch-global norm ratio, because averages of per-example ratios are different quantities. Aggregate per-example means by sample count, not equal weighting of unequal batches; report medians and 10th/90th percentiles with sample counts.

Log predictor MSE, cosine component, their weighted contributions, and predictor/model gradient clipping coefficients. This checks whether fixed initialization-based scaling makes MSE dominate the cosine objective. It is a diagnosis, not authority to change the loss.

For parameter-gradient agreement, use a disposable model copy, the same input and saved pre-step state, and exact vs injected trunk VJPs. Compute global dot products/norms by summing over parameter tensors without concatenating a full parameter vector. Report preclip and postclip values. The exact-trunk/head computation and BN buffer mutation must be controlled on identical copies.

For update utility, clone the full optimizer state, execute one exact/learned/zero/skip update at the current LR/momentum/decay, and evaluate loss before/after in eval mode on (a) the same unaugmented training examples and (b) a separate fixed training probe batch. Hold the head update fixed across these alternatives: first use no head update in any alternative to isolate trunk utility. Report a separate full-exact reference if desired. Do not substitute plain SGD for the actual momentum optimizer or infer generalization from same-batch loss improvement.

### 5.3 Diagnostic sampling and normalization investigation

Default probe checkpoints: after warm-up and after post-warm-up epochs 1, 6, and 12. Use four fixed training probe batches of 32 examples with deterministic evaluation transforms, selected with a separate generator seed 88001 from training indices. Disclose that these are training observations, not an independent reliability test set. Do not fit parameters, thresholds, schedules, or stopping policies using their diagnostic outcomes during a locked run.

At warm-up, compare head train-mode and frozen-stat behavior on copied states using identical boundary features. Measure true-target change, predictor error, clipping, and BN buffers before/after. Repeat short transition probes on augmented training batches separately, since deterministic evaluation transforms differ from the actual training distribution.

At all probe checkpoints, use independent copied state for each probe/alternative. No diagnostics may modify the production run's model, optimizers, controller, BN buffers, RNG, or loader state. Tests must prove isolation.

If normalization remains a plausible dominant cause, propose a separately approved condition in which head BN statistics are consistently frozen from a defined earlier point, or an explicitly trained calibration adaptation period. Apply the same condition to its exact and switching comparators and account for added work. Do not automatically introduce GroupNorm or change architecture; those answer a different question.

### 5.4 Accuracy endpoints and statistical rules

Default targets: 85%, 87%, and 89% validation accuracy, fixed before fresh-seed evaluation. Historical 70% attainment can be reproduced but is not a primary endpoint. Evaluate after every warm-up and post-warm-up epoch; save the timestamp of each observation.

- First attainment is the first observed validation accuracy at/above the target, at the end of that validation pass. Do not interpolate an unobserved crossing or claim a target was reached sooner between epochs.
- Mark unattained targets `reached=false`, with null attainment time and recorded observation horizon. Do not replace them with the final duration, drop them when averaging, or claim speedup to an unattained target.
- Report final epoch accuracy as the primary fixed-horizon accuracy endpoint. Best validation accuracy is secondary and selection-biased; preserve its actual checkpoint/time.
- Plot both exact references and candidate accuracy trajectories from the first warm-up evaluation. Early-stopped exact comparisons use observed checkpoints of the same full-horizon LR schedule, not a silently reoptimized short schedule.
- Predefine a common time budget per seed as the fresh `exact` frozen-head reference's full-run duration, using an independent reference pass when necessary. At that budget, report each method's latest completed evaluation at/before the budget; report its timestamp/staleness. If no evaluation exists, report unavailable. This is a secondary retrospective comparison, not an online stopping rule.

Development: historical seeds 0/1/2 are already inspected; use seed 0 for engineering diagnostics and report all such work as exploratory. Recommended locked evaluation uses fresh seeds 3/4/5/6/7 after the protocol is approved and frozen. These share the same validation split, so seed uncertainty does not measure dataset-split uncertainty or remove validation reuse.

Paired reporting: list every seed's difference, mean paired difference, sample SD, and a Student-t 95% CI with `n-1` degrees of freedom. For five seeds the two-sided 95% critical value is 2.776445; for three it is 4.302653. Treat these small-sample intervals as assumption-dependent, not guaranteed coverage. For timing use paired log time ratios, then exponentiate interval endpoints. Require comparable hardware/timing conditions. Mask replicas are averaged within seed; hardware repetitions are summarized within seed and are not additional independent samples.

Practical accuracy margin proposal: 0.5 pp. Practical runtime margin proposal: 5%. These are proposed decision thresholds, not established domain facts; require approval before locked evaluation. A superiority conclusion should have a positive paired effect with a CI excluding zero and a point estimate exceeding the practical margin. A strong practical-improvement claim requires the CI itself to exceed the practical margin. Predefined approximate equivalence requires the whole 95% interval inside +/-0.5 pp; otherwise report inconclusive. Failure to find a significant difference is not equivalence.

Primary predictor contrasts are learned-vs-zero and learned-vs-skip; primary controller contrasts are feedback-vs-per-epoch-periodic and feedback-vs-gap-permutation. Report individual intervals descriptively. To make a family-level claim, use Bonferroni-adjusted intervals within each two-contrast family (97.5% two-sided intervals, t critical 3.495406 for five seeds); both required contrasts must support the claim. Do not select the favorable comparator and omit the other. Other methods/endpoints/targets remain explicitly exploratory unless a new locked protocol specifies otherwise. Do not pool epochs or minibatches as independent replicates.

## 6. Runtime, profiling, persistence, and outputs

### 6.1 Timing from the first scientific run

Timing infrastructure is a Stage 1 requirement. Stage 4 analyzes efficiency; it does not retrofit timing into earlier experiments.

Use synchronized CPU wall time at phase boundaries and CUDA events for separate microbenchmarks. Do not synchronize every batch in the normal training loop merely for logging. Record a non-overlapping ledger for setup, scale estimation, plain/predictor warm-up training, warm-up validation, calibration, post-warm-up training, post-warm-up validation, diagnostics, checkpoint/logging I/O, and finalization. Where scale estimation occurs inside a historical warm-up interval, split the interval without double counting, or mark it included.

Define:

- `T_learning = warmup_training + scale_if_needed + calibration_if_needed + postwarm_training`.
- `T_evaluated = T_learning + warmup_validation + postwarm_validation` (closest comparable v3 accounting).
- `T_workflow = setup + T_evaluated + required_checkpoint/log_IO + finalization`, including required diagnostics only if actually enabled in that workflow.
- `T_postwarm = postwarm_training + postwarm_validation`; also publish its training-only component.
- `S_full = T_workflow(exact)/T_workflow(candidate)` and `S_postwarm = T_postwarm(exact)/T_postwarm(candidate)` for comparable conditions.

Record dataset download/cache preparation separately from steady training; report it for a cold workflow and state when training timing assumes a ready dataset. Preflight profiles and optional scientific probes are research overhead, reported separately, not hidden or automatically charged to the deployable algorithm. Runtime accounting must never infer a speedup from exact-step counts.

Distinguish actual study resource use from per-method standalone costs. Shared warm-up execution is charged once in actual study resource totals, but its measured relevant plain/predictor cost is attributed to each method's standalone estimate. Label reconstructed totals as attributed, not continuously measured end-to-end executions. Final deployable efficiency claims require clean, independently timed complete runs including their own required warm-up/setup/persistence. Cache hits have no new runtime observation.

Retrospective replay methods have two cost records: execution conditional on an already available mask, and the dependency-inclusive workflow cost. The latter sums unique measured prerequisite phases, including feedback mask generation, without counting shared warm-up twice. Neither turns the retrospective control into an online policy. Do not place conditional replay timings on a deployable-method Pareto frontier.

Time-to-target includes work actually completed before the relevant validation endpoint. Future checkpoint writes do not count toward an earlier attainment time. Record checkpoint I/O separately at each boundary, so later timestamps include it. Algorithmic active time excludes disconnection downtime; resumed wall-span and reload/redo overhead are separate fields. Resumed runs can contribute accuracy evidence after equivalence checks but cannot serve as clean uninterrupted efficiency benchmarks.

### 6.2 Hardware and ordering

Record Python, PyTorch, torchvision, NumPy, CUDA/cuDNN versions, GPU name/UUID if available, driver, precision/TF32/determinism flags, CPU, loader options, source hashes, and backend optimizer settings. Record available temperature, power, clocks, and memory telemetry at phase boundaries without creating a required dependency or altering GPU settings.

Do not run another training job concurrently. No automatic CPU fallback for a requested CUDA run. Use the same batch size/precision on T4 for every comparison; do not compare laptop times with T4 times.

Mask dependencies constrain ordering: first generate donor masks; then run the measured comparison block, including a fresh feedback rerun and exact references, in a deterministic randomized order saved before observing results. Verify the measured feedback mask agrees with the donor in the same deterministic environment. If not, stop the corresponding mask-matched comparison and investigate; do not substitute a new mask after seeing outcomes.

Use a private ordering RNG seed 73001 plus study repetition index. Across repeated timing blocks use a cyclic rotation of the saved randomized order to reduce position effects. Pre/post-block identical short exact profiles diagnose hardware drift; a >10% median latency difference flags the block. Keep all records, mark the reason, and repeat the entire affected comparison block if authorized; never cherry-pick fast runs or discard a slow method alone. T4 does not guarantee stable clocks or an otherwise idle host.

### 6.3 Checkpoint and identity contract

Support **epoch-boundary resume first**, including warm-up and post-warm-up stages. Mid-epoch exact resume is out of scope. After an interrupted epoch, resume from the preceding committed boundary and repeat that epoch; log the interruption/redo rather than pretending it was continuous.

Checkpoint payload must include:

- schema/code/scientific/execution fingerprints, seed, method, normalization, stage, completed epoch and next epoch;
- trunk/head/predictor weights and all buffers, relevant optimizer states, fixed scale, LR clock;
- controller history in order, thresholds/base/EMA, pending quota, action counts;
- full mask or immutable mask reference/hash, source identity, consumed cursor, epoch offsets;
- Python, NumPy, CPU Torch and all CUDA RNG states, explicit loader/augmentation/generator states or reconstructible seeds;
- accumulated counters, epochs/events, target-attainment state, timing ledger and checkpoint sequence number;
- data split hashes, warm-up identity, environment metadata, and run completion state.

Store NumPy/Python RNG state as primitive containers/tensors supported by `torch.load(weights_only=True)`; do not require arbitrary pickle execution. Save model tensors without live computation graphs. Validate the requested device/backend compatibility before loading optimizers to that device.

Canonical scientific fingerprint: sorted JSON with explicit defaults covering architecture/split, data identities/transforms, optimizer/LR/clipping/scaling, controllers, horizon/batch size, precision/normalization, seeds, diagnostics affecting computation, masks/source hashes, algorithm version, and source hashes. Separate execution fingerprint includes hardware/software, workers/pinning/determinism, timing/checkpoint policy. Paths/timestamps are not scientific identifiers, but resolved input content hashes are. Reject mismatches rather than silently ignoring them. Environment changes require a new run with a provenance link, not an exact-resume claim.

Write a temporary checkpoint in the destination directory, flush/fsync where supported, then `os.replace` the final path. Keep the preceding valid checkpoint until the new one and its metadata commit successfully. On Colab, atomicity is guaranteed only for local filesystem writes; copy immutable completed checkpoints to Drive, verify SHA-256, and publish a completion marker last. Ignore partial/mismatched remote copies on restore.

Use a checkpoint-sequence commit journal for elapsed I/O time and output reconciliation. A checkpoint cannot contain the final duration of its own write; append that duration after commit. If a crash prevents the timing journal entry, flag the unmeasured interval rather than inventing it. State equivalence excludes timing fields. On resume, rebuild CSVs idempotently from committed records or truncate uncommitted rows by sequence number; never duplicate epochs/events.

### 6.4 Required outputs and schema

Each run directory contains `manifest.json`, `epochs.csv`, `controller_events.csv` where relevant, `masks.npz` plus `mask_manifest.json`, `summary.json`, `timing.json`, `checkpoints/`, and `errors.jsonl`. Diagnostic runs additionally contain `gradient_diagnostics.csv`, `update_probes.csv`, and `normalization_probes.csv`. Profiles contain `profile_samples.csv` and `profile_summary.json`. The study root contains locked protocol, run-order manifest, `paired_comparisons.csv`, `target_attainment.csv`, aggregate tables, and optional accuracy/time and Pareto plots.

Checkpoint retention default: retain initialization/stage-boundary state, the latest and preceding resumable checkpoints, the warm-up-final state, post-warm-up epochs 1/6/12 for probes, and the first-attainment checkpoint for each predefined target (deduplicate identical epochs). Save a separate final predictor even when no target is attained. Retention may remove only explicitly superseded files generated within that run, after the replacement has committed; it never removes v3 files or source masks. Estimate storage before the run. Persist checkpoints locally every epoch; copy each committed epoch checkpoint to Drive when persistence is enabled, then apply the same verified retention policy. Remote-copy time remains visible in workflow time.

Units: accuracy stored as fractions `[0,1]`; comparison columns ending `_pp` multiply differences by 100; wall time seconds; latency milliseconds; memory MiB; gradient units explicitly named. Use blank/null plus a status field for unavailable values, never numerical zero or infinity as a stand-in. Every row carries schema version, run/seed/method identity, and appropriate epoch/step/probe key. Publish the exact column dictionaries in code and future Colab documentation. Log finite failures with the last valid checkpoint, batch identity, and update type; do not silently recover by changing LR, gradients, quota, precision, or batch size.

## 7. Ordered implementation milestones

Each milestone must end with a small reviewable change and its tests. Passing engineering tests authorizes progression only within the separately approved implementation scope; it does not authorize the scientific experiment matrix.

### M0 — Freeze the specification and reference identity (Stage 1)

**Objective/motivation:** prevent a corrected implementation from changing the historical algorithm without disclosure.

**Tasks and interfaces:** create the new config/schema; encode Sections 3–6 as explicit values and enums; hash reference source/results and split identities; define `RunConfig`, `RunState`, `StepResult`, and `MaskMetadata` as small records. `StepResult` exposes update counts and detached metrics, not retained graphs. Add `validate_config(config)` and canonical `fingerprint(config, artifacts)`. Validate positive horizons, batch sizes, windows, ordered quantiles, bounded EMA, and supported method/normalization combinations. Preserve old method identifiers and define all new identifiers listed here.

**Files:** new reference JSON; initial v4 and I/O modules; initial tests. No edits to v3/spec/archive.

**Dependencies:** none beyond reading the source/spec and this plan.

**Tests/invariants:** config round-trip; stable hashes independent of dictionary order; each scientifically relevant field changes identity; rejected output roots; invalid combinations fail before training; historical source hashes unchanged.

**Acceptance:** every implemented condition has an unambiguous method/version/normalization identity; no inherited default is omitted from the saved resolved config; the issue register in Section 10 has an owner milestone.

**Outputs:** schema/config and a reference manifest; no scientific training results.

**Failure modes/detection:** stale filename assumptions, conflating head masks with trunk masks, omissions in fingerprint, or accidental writes to historical roots; detect with explicit path and mutation tests.

### M1 — Exact/synthetic updates and normalization correctness (Stage 1)

**Objective/motivation:** establish that differences between conditions reflect the intended updates.

**Tasks and interfaces:** implement `build_models`, `build_optimizers`, `set_epoch_lr`, `set_training_mode`, `exact_step`, and `synthetic_step(kind, shadow_predictor)` following Section 3. Add conventional `standard_bn` exact reference. Port historical controller logic without changing math. Keep timing/counter hooks in the step runner but no per-batch synchronized profiling by default. Document tensor dimensions, detach boundaries, loss reductions, clipping groups, and optimizer stepping.

**Files:** v4 module and tests; amend config only for already specified fields.

**Dependencies:** M0.

**Tests/invariants:** direct BP vs injected true boundary gradient at batch sizes 1, 8, 32, with a tiny split model in both downstream BN modes, using cloned pre-forward states; scale round-trip; full last-batch path; no new predictor gradients/state changes on synthetic steps; head forward hooks prove no head execution on synthetic slots; no stale head gradients; trainable BN affine parameters despite frozen buffers. Validate zero vs skip with nonzero momentum and weight decay, plus zero-momentum/zero-decay controls. Verify predictor fitting cannot change classifier gradients. Check joint/trunk clipping behavior and LR endpoints. On cloned tiny states, compare v4 historical exact/synthetic step outputs, optimizer buffers, predictor updates, and historical controller decisions directly with imported v3 functions, including a short alternating sequence. Import with bytecode writing disabled; never invoke v3 `main()` or output-writing helpers.

**Acceptance:** FP32 gradient comparisons use `torch.testing.assert_close` with CPU rtol=1e-5/atol=1e-6 and CUDA rtol=1e-4/atol=3e-5, reporting max absolute/relative discrepancy. Do not loosen tolerances automatically. Integer counters and masks match exactly. Optimizer parameters/buffers match a manually computed tiny reference; skip leaves them bitwise unchanged. Run existing v3 small audit helpers in memory without invoking v3 `main()` or writing old outputs.

**Outputs:** passing correctness report including device/version; no accuracy/speed claims.

**Failure modes/detection:** factor-of-B errors, independent-example assumptions under BN, predictor graph leakage, unintended head steps, momentum drift on skip, clipping-group changes; each has a dedicated assertion.

### M2 — Deterministic runner, persistence, timing, and Colab skeleton (Stage 1)

**Objective/motivation:** make subsequent evidence reproducible and time-accounted from its first run.

**Tasks and interfaces:** implement `run_epoch`, `evaluate`, `save_checkpoint`, `load_checkpoint`, `resume_run`, and a phase timer/ledger; separate warm-up and post-warm-up stages; persist per-epoch warm-up metrics/times; add mask and data-order hashes. Build local output commit/reconciliation and optional verified Drive persistence. Provide a CLI with explicit `audit`, `smoke`, `profile`, `run`, `resume`, and `diagnose` subcommands. A mode never silently launches another expensive mode. No automatic full study after preflight.

**Files:** v4, I/O, tests, and future Colab guide.

**Dependencies:** M1.

**Tests/invariants:** uninterrupted vs epoch-boundary resumed tiny runs must match model/predictor/optimizer/controller state, masks, LR, augmentations, metrics, and RNG; exclude elapsed time from equality. Include warm-up interruption and a quota crossing an epoch boundary. Simulate partial checkpoint/corrupt checksum, config/mask mismatch, failure between checkpoint and CSV commits, duplicated output rows, and missing timing-journal entry. Assert validation/probes do not alter training state. Test timing sums on synthetic timer inputs rather than wall-clock thresholds.

**Acceptance:** complete CPU tests and CUDA smoke when a CUDA device is available; unsupported strict determinism fails clearly. An interrupted epoch is explicitly restarted, not falsely resumed mid-batch. Every validation point, including warm-up, has a time and checkpoint identity. No code needs Linux-specific or Windows-specific paths. Code imports do not download data or start a run.

**Outputs:** clean toy run/resume artifacts, resolved environment manifest, setup/test/smoke instructions. Pin a tested compatible torch/torchvision pair after actual environment verification; do not blindly reinstall Colab's GPU stack.

**Failure modes/detection:** RNG changed by logging/mask construction, head mode left in eval/train incorrectly after evaluation, stale result cache, incomplete Drive copy, shared warm-up double counting, or missing time components. Fail on scientific-state mismatch; label timing incompleteness.

### M3 — Diagnostic probes and normalization baseline (Stage 1 completion)

**Objective/motivation:** distinguish gradient/normalization hypotheses before algorithm changes.

**Tasks and interfaces:** implement `probe_boundary_gradients`, `probe_parameter_updates`, `probe_normalization_transition`, and `profile_steps` with copied state and fixed sample IDs. Compute all Section 5 metrics for any predictor-bearing condition, including scheduled methods. Add a zero-predictor reference whose unfloored batch relative residual is 1 when the true gradient is nonzero. Recompute microbenchmark medians from all timed samples; include actual controller-path overhead separately from the exact-plus-predictor kernel profile.

**Files:** v4, initial analysis module, tests, Colab guide.

**Dependencies:** M2; requires full predictor/optimizer state in diagnostic checkpoints.

**Tests/invariants:** exact predictor yields cosine/norm ratio/projection 1 and residual 0; zero prediction gives residual ratio 1 away from floor and undefined cosine; norm-matched orthogonal prediction gives zero projection; near-zero targets remain explicitly flagged. Probe execution leaves every production state/RNG hash unchanged. Manual tensor examples verify sample vs batch aggregation. Time-to-target analysis detects attainment within warm-up and unattained targets correctly.

**Acceptance:** a tiny diagnostic pass produces finite interpretable metrics or explicit validity flags with no production mutation. The conventional exact reference trains downstream BN normally and is labeled separately from v3 exact. Probe overhead never disappears into a claimed deployable speedup.

**Outputs:** diagnostic CSVs and microbenchmark sample/summary schemas; a small engineering report. Real-data transition probes require experiment authorization, even though they are cheaper than training runs.

**Failure modes/detection:** probes consume augmentation RNG, reuse mutated BN states, use plain SGD rather than cloned momentum state, or log scaled/unscaled gradients under one name; hash isolation and analytic fixtures detect them.

### M4 — Predictor-usefulness replay controls (Stage 2)

**Objective/motivation:** determine whether the learned injected signal adds value beyond the exact-update schedule and optimizer dynamics.

**Tasks and interfaces:** implement the four replay conditions in Section 4.1; `load_mask` verifies full identity before training; `execute_slot` distinguishes trunk-exact, head-step, predictor-fit, and synthetic-kind decisions. Add source/replay equality checking and source-dependency cost records. Implement the no-predictor fast variants but do not automatically include them in the causal timing table.

**Files:** v4, I/O schema extensions, analysis contrasts, tests, Colab guide.

**Dependencies:** M3; a donor mask from the same approved condition is required for scientific replay.

**Tests/invariants:** learned replay follows exactly the donor mask and reproduces its trajectory on a tiny deterministic run; zero/skip have identical sample/augmentation hashes; head step count equals mask true count; exact-trunk control has `N` true trunk-exact entries; all zero/skip semantics from M1 hold after nonzero momentum warm-up. Shadow-vs-fast variants must agree in classifier trajectory at a fixed mask. Counts sum to total slots without conflating zero backwards with exact or learned synthetic backwards.

**Acceptance:** all replay invariants pass; malformed masks fail before model updates. A pilot may support an engineering conclusion or hypothesis only, not statistical equivalence or acceleration.

**Outputs:** paired learned/zero/skip/head-control tables, complete masks/state checkpoints, gradient/update-quality diagnostics, and provenance/cost labels.

**Failure modes/detection:** accidental online re-selection instead of replay, mask off-by-one after resume, shadow predictor affecting classifier, or treating head-mask entries as trunk-exact entries; assert at each slot in tests and reconcile epoch/final totals.

### M5 — Controller-usefulness schedules (Stage 3)

**Objective/motivation:** separate amount, early allocation, and temporal placement of exact computation from information in the feedback signal.

**Tasks and interfaces:** implement `make_periodic_mask`, `make_epoch_matched_mask`, `make_frontloaded_mask`, and `permute_epoch_gaps`. Keep v3 midpoint scheduling separately for historical controls. Assign independent RNG namespaces and source hashes. Log exact counts by full run/epoch, synthetic run-length histograms, first/last exact positions, and moved-gap fraction. Keep controller definitions frozen; do not tune thresholds in response to pilot results without a new protocol.

**Files:** v4, analysis, tests, config/guide method registry.

**Dependencies:** M4 and donor masks; deployable fixed schedules do not depend on a donor.

**Tests/invariants:** property-style loops over small `n,k`, including zero/full/one exact slot; periodic positions unique and exact counts correct; matched totals and epoch counts exact; gap permutation preserves every invariant in Section 4.3, including runs crossing epoch boundaries; source masks unchanged; RNG independence; degenerate masks labeled correctly. Verify quotas continue through epoch boundaries and historical scheduled phase boundaries retain the global step rule.

**Acceptance:** every claimed matching property has a test and is checked on run artifacts. Random replicas aggregate within seed. A report cannot label retrospective schedules deployable or describe per-epoch matching as full temporal matching.

**Outputs:** budget/spacing diagnostic tables; paired controller comparisons; saved schedule construction metadata and random seeds.

**Failure modes/detection:** reset quotas, incorrect final truncated gaps, randomization that changes budgets, front-loading mislabeled exactly matched, or future-count access hidden in an online baseline; tests and explicit provenance detect them.

### M6 — Efficiency analysis and controlled benchmark execution (Stage 4)

**Objective/motivation:** establish whether any deployable candidate improves a meaningful accuracy–time tradeoff.

**Tasks and interfaces:** implement result validators, paired accuracy/log-time statistics, target attainment, full and post-warm-up timing summaries, and plots using every seed. Distinguish conventional vs frozen-head exact references, attributed vs continuously measured totals, clean vs resumed runs, shadow vs fast costs, and retrospective vs deployable methods. Execute only the approved staged matrix in Section 8; generate donor masks before randomized measurement blocks. Include early-stopped exact checkpoints and report checkpoints that cannot meet the target as censored.

**Files:** analysis module, orchestration portions of v4/I/O, tests, Colab guide. No new framework.

**Dependencies:** M5; timing data already exists from M2 onward. Fresh clean full runs are needed for strong efficiency claims.

**Tests/invariants:** synthetic result fixtures cover early target attainment, unattained targets, interrupted runs, duplicate seeds, mismatched hardware, averaged replicas, missing epochs, unknown schema, and pp/fraction confusion. Validate CI/timing calculations against hand-calculated fixtures. A Pareto filter must not include retrospective conditional-mask timings among deployable points. Reject invalid/incomplete pairings instead of silently shortening comparisons.

**Acceptance:** all timed comparisons use the same approved hardware/precision protocol, disclose warm-up/calibration/I/O, and show attainment counts. Hardware repetitions and seed replicas cannot inflate sample size. Complete ledger reconciles within timer precision; unexplained residuals are failures, not a miscellaneous hidden saving.

**Outputs:** accuracy-vs-epoch/time figures, target-attainment and paired tables, deployment-eligible Pareto table, hardware/order diagnostics, actual study resource cost, and a concise limitations report.

**Failure modes/detection:** speed claims at unequal accuracy, exclusion of failed targets, laptop/T4 mixing, cached time reuse, fixed donor-first thermal bias, or cherry-picked timing runs. Automated eligibility checks and the predeclared order manifest prevent these errors.

### M7 — Scientific decision gate (Stage 5)

**Objective/motivation:** choose the next research direction based on evidence rather than defend the original hypothesis.

**Tasks:** analyze the locked contrasts and uncertainty using Section 9; distinguish numerical correctness, predictor utility, controller utility, and practical efficiency. Produce a short results interpretation only after approved experiments. Identify whether further seeds, a fixed-mask replication, or longer horizons are necessary. Do not implement a better controller or accuracy-budget optimizer automatically.

**Files:** analysis module only if needed for the already specified gate; future run-local `decision_report.md`. Do not rewrite v3 or this protocol to fit outcomes.

**Dependencies:** M6 and a sufficient approved evidence set; otherwise the conclusion is explicitly exploratory/inconclusive.

**Tests/invariants:** decision fixtures cover positive, negative, contradictory, equivalence, and wide-CI outcomes. Failed or underpowered results cannot be promoted to success through wording or best-epoch selection.

**Acceptance:** every conclusion names its contrast, endpoint, uncertainty, practical margin, normalization, horizon, and deployment eligibility. Unresolved evidence produces a concrete conditional next experiment, not a claimed proof.

**Outputs:** decision report with recommended direction and incremental compute estimate; no new algorithm implementation.

**Failure modes/detection:** interpreting nonsignificance as equivalence, treating local loss probes as generalization evidence, or advancing to constrained optimization without a useful Pareto frontier. Use the explicit decision table below.

## 8. Staged run matrix and relative compute cost

All runs below are planned, not authorized by this document alone. No full matrix should be launched as part of implementation tests.

### 8.1 Engineering and pilot sequence

1. **CPU unit/integration tests:** tiny synthetic images, width 4, one block, two short epochs and unequal last batch. Seconds/minutes scale, not scientific evidence.
2. **T4 smoke and resume:** similarly tiny data/model; all required methods, quotas, masks, and schemas; no accuracy claims.
3. **T4 cost preflight:** real-width model and copied warm-up state; measure actual step components and memory. If no synthetic-step savings exist, inspect overhead before expanding a speed study, but do not treat this as an accuracy conclusion.
4. **One-seed full-horizon exploratory pilot (seed 0):** core methods below; early-epoch snapshots/probes are inspected for numerical failures, not used to choose a flattering stopping epoch. Preserve runs that collapse as valid scientific outcomes unless a verified implementation defect invalidates them.
5. **Protocol lock and fresh-seed evaluation:** only after review of correctness, pilot stability, diagnostic costs, and budget. Fresh seeds are not used for iterative tuning.

### 8.2 Core and conditional matrix

| Group | Conditions | Initial status |
|---|---|---|
| Accuracy references | v3 `exact` frozen-head; conventional `exact_standard_bn` | Mandatory. |
| Candidate/history | v3 `relative_feedback`; v3 `scheduled` | Mandatory. Historical absolute feedback and its original controls are optional failure-mode reproduction. |
| Predictor attribution | relative replay zero-shadow, skip-shadow, exact-trunk | Mandatory. Learned replay is an equivalence check, not a separate independent scientific condition. |
| Controller attribution | relative per-epoch periodic; relative gap permutation | Mandatory. Pilot one permutation; locked controller question three per seed. |
| Budget allocation | relative total-periodic; fixed-frontloaded | Required implementation and pilot comparisons; promote to fresh-seed evaluation if feedback's advantage over the core controller controls remains plausible. |
| Practical sparse BP | fixed-uniform-third or the predeclared front-loaded schedule with zero-fast and skip-fast | Conditional: required if claiming synthetic-gradient efficiency or if zero/skip match learned replay closely. Choose the predeclared front-loaded schedule as default; do not choose using fresh-seed results. |
| Generality of predictor effect | learned/zero/skip under that same fixed schedule | Conditional follow-up after positive source-mask predictor evidence. |
| Normalization/clipping changes | matched exact/candidate conditions with a single explicit change | Conditional, separate protocol version; not a Cartesian product in the first study. |

The mandatory core has nine trajectories per classifier seed when there is one gap-permutation replica: two exact references, relative feedback, scheduled, three predictor controls, epoch-periodic, and gap permutation. Three permutation replicas raise it to eleven. Replay equivalence, donor generation, profiles, optional total/front-loaded runs, and clean timing repetitions are additional resource costs, not hidden in that count.

If an earlier stage decisively shows no useful learned contribution or an implementation failure, stop expanding later scientific comparisons and invoke the decision gate. A broad null claim still needs adequate precision; a one-seed pilot is insufficient.

### 8.3 Cost estimates and caps

Define `C` as measured T4 post-warm-up exact training+validation cost and `W_plain`, `W_pred` as measured four-epoch warm-up costs. Do not extrapolate laptop seconds to T4 or promise a fixed duration before profiling.

- Exact/head-trunk controls cost roughly one `C` each; the head-mask control still computes the full head backward.
- Synthetic/shadow-zero runs cost `N*(f*t_exact_pred+(1-f)*t_slot)` plus validation/host work. Estimate using the actual slot type, not the learned-synthetic latency for every control.
- Skip controls can be cheaper, but still execute trunk forward/BN and, in shadow mode, predictor inference.
- Each extra random-mask replica adds a full post-warm-up run. Three replicas across five seeds add ten runs beyond one replica/seed.
- Five-seed locked core is 55 post-warm-up trajectories with three random replicas, plus donor/replay work and timing repetitions. This is a ceiling proposal, not an instruction to launch 55 runs immediately.
- Shared warm-up reduces actual study resource use but not the reported standalone cost of an algorithm. Profile and copied-state probes are usually much cheaper than full training, while exact parameter-update probes require several model/optimizer copies and must be memory-bounded/sequential.
- Before any batch of full runs, print planned run count, estimated GPU-hours from measured pilot durations, expected storage, and remaining dependencies. Default to one run at a time. No unbounded sweep or automatic extension of epochs/seeds.

## 9. Scientific decision rules

Interpret fixed-mask comparisons within their normalization/horizon and report both accuracy and gradient/update diagnostics.

| Evidence | Interpretation | Next action |
|---|---|---|
| Learned replay beats zero and skip with meaningful paired effects | Predictor contributes under the feedback-derived mask | Confirm under the fixed schedule; then assess controller and time benefits. |
| Learned replay and zero/skip meet the predefined equivalence criterion | Little practical contribution detected at this horizon/mask | Prioritize prediction quality or sparse exact-BP schedules; do not call the effect synthetic-gradient acceleration. |
| Intervals are wide or the two predictor comparisons disagree | Attribution unresolved; momentum or spacing may matter | Examine update diagnostics and uncertainty; propose targeted additional evidence, not a positive/negative headline. |
| Feedback beats epoch-periodic but not gap permutation | Budget plus burst structure may explain the gain | Study schedule design; error information is not established as the useful ingredient. |
| Feedback beats both controller controls after paired/multiplicity checks | Evidence for useful temporal alignment beyond those matched properties | Confirm against deployable front-loaded/fixed schedules and assess future-error predictiveness separately. |
| Learned gradients help but feedback does not beat fixed schedules | Predictor utility without demonstrated controller utility | Prefer a simpler schedule or study predictor quality; defer complex adaptive control. |
| Candidate is faster at fixed epochs but exact reaches targets sooner | No demonstrated training-efficiency advantage for those targets | Do not proceed to accuracy-budget optimization on that evidence. |
| Candidate has a meaningful deployable accuracy–time advantage over both exact references and credible fixed/sparse alternatives | Promising practical benefit under tested conditions | Propose longer-horizon/held-out validation before an accuracy-budget controller. |
| Correctness fails, normalization/clipping dominates, or all useful comparisons are inconclusive | Existing mechanism interpretation is unreliable | Repair verified defects or run one controlled diagnostic; do not expand architecture/schedule sweeps. |

A formal accuracy-degradation guarantee is outside scope. Even a positive empirical frontier does not establish such a guarantee. Retrospective mask controls cannot be selected as the deployable final algorithm.

## 10. Issue-to-correction traceability

`S#` refers to the numbered issue in Section 7 of `RESEARCH_SPEC.md`; `A#` refers to an additional audit/planning concern. “Deferred” means explicitly excluded from initial implementation, not forgotten.

| ID / issue | Classification | Correction/control | Required test or evidence | Milestone |
|---|---|---|---|---|
| S1: relative feedback lacks matched comparator | Confounder | Total/epoch matched periodic controls | Exact total/epoch counts and source identity | M5 |
| S2: early exact allocation differs | Confounder | Epoch matching; predeclared front-loaded policy | Per-epoch allocation and LR comparison | M5–M6 |
| S3: BN transition | Hypothesis/confounder | Paired copied-state probes; separate conventional exact reference | BN mode/buffer tests and target/error transition metrics | M1, M3 |
| S4: sparse head training | Confounder | Relative-mask exact-trunk control | All-exact trunk mask, sparse verified head steps | M4 |
| S5: 70% target and warm-up attainment | Reporting defect/weak endpoint | Warm-up epoch timestamps; 85/87/89% targets; censoring | First-attainment fixtures including warm-up | M2, M3, M6 |
| S6: relative quantiles hide global deterioration | Algorithm limitation | Preserve historical controller; log absolute/relative reliability | Global-drift synthetic fixtures; no claim of safeguard | M1, M3; redesign deferred |
| S7: absolute controller degenerates | Algorithm limitation | Preserve optional baseline; log quotas, floor and outliers | EMA/threshold tests; fixed-schedule comparisons | M1, M3, M5 |
| S8: denominator floor | Interpretation risk | Preserve floor; measure activation; separate diagnostics | Known floor-active/inactive fixtures | M3 |
| S9: calibration uses training data | Protocol limitation | Declare online training-stream calibration; disjoint validation/test | Split hashes, no eval data in predictor/controller fitting | M0–M3 |
| S10: clipping groups differ | Confounder | Preserve primary semantics; measure coefficients; conditional alternative | Exact/synthetic/head-mask clipping tests | M1, M3 |
| S11: cache/mask reuse | Implementation gap | Config/source/mask fingerprints; reject mismatches | Cache identity and corrupted-mask tests | M0, M2, M4 |
| S12: preflight is not end-to-end | Accounting limitation | Separate profile, evaluated, workflow and study costs | Ledger reconciliation and eligibility checks | M2, M6 |
| S13: short horizon | Evidence limitation | Preserve 16-epoch reference; mark scope; gate longer runs | Metadata/horizon validation; no convergence claim | M6–M7; longer runs deferred |
| S14: one dataset/architecture | Evidence limitation | Restrict claims; expand only after mechanism evidence | Decision report explicitly names coverage | M7; expansion deferred |
| S15: incomplete gradient diagnostics | Measurement gap | Common fixed probes and predictor/error/update metrics | Analytic metrics and mutation-isolation tests | M3 |
| A1: learned vs zero vs skip unresolved | Central hypothesis | Shared-mask controls with precise optimizer semantics | Momentum/decay and paired utility evidence | M1, M4 |
| A2: exact wins at comparable observed accuracy | Central efficiency concern | Early-stopped exact and target/cost curves | Correct endpoint accounting and attainment counts | M6 |
| A3: burst spacing can mimic signal value | Confounder | Epoch gap permutation, independent replica RNG | Counts/gap/endpoints invariants and paired replicas | M5 |
| A4: score describes past exact batch, not future synthetic examples | Causal interpretation limit | Pre-fit timestamps; randomization control; no per-example selection claim | Event-order tests; optional future-error probes are diagnostic only | M1, M3, M5 |
| A5: missing predictor/resume state | Implementation gap | Full checkpoints including predictor/optimizer/controller/RNG | Uninterrupted/resumed state equivalence | M2 |
| A6: fixed method order and GPU variation | Timing confounder | Donor phase then randomized measured blocks; drift flags | Order manifest, repeat full flagged block, matched hardware | M6 |
| A7: frozen-head exact is not conventional BP | Comparator limitation | `exact_standard_bn` reference | Mode/buffer and label checks | M1, M6 |
| A8: scaled batch gradient mislabeled independent per-example target | Mathematical wording issue | Explicit summed-batch derivative identity | BN-aware VJP and last-batch tests | M1 |
| A9: initialization scale may distort objective balance | Hypothesis | Log target RMS, objective terms and clipping | Unit definitions and loss-component fixtures | M3; scale redesign deferred |
| A10: historical runtime/budget claims omit distinctions | Reporting limitation | Both full/post-warm budgets; recomputed medians; cost provenance | Summary reconciliation from raw rows | M2, M6 |
| A11: shadow controls vs practical sparse computation | Attribution/cost confounder | Separate shadow/fast identities and mask dependency costs | Same classifier trajectory, different disclosed computation | M4, M6 |
| A12: repeated validation use and small seed count | Statistical limitation | Development/locked split; paired seed intervals; no test access | Duplicate/pseudoreplication/multiplicity checks | M0, M6–M7 |
| A13: calibration BN copy mutation | State risk | Disposable copy; discard changed buffers | Warm-up state/RNG hashes unchanged | M2–M3 |
| A14: source mask favors source trajectory | Scope limitation | Fixed-schedule learned/zero/skip confirmation | Separately labeled follow-up | M4–M7 |
| A15: timing and probes perturb training | Measurement risk | Separate probe mode and bounded logging; stage-boundary sync | Production-state isolation and diagnostic-cost ledger | M2–M3 |
| A16: preservation/paths/environment uncertainty | Reproducibility gap | Actual `_v3` paths, source hashes, resolved dependency metadata | Read-only source checks and fresh-output validation | M0, M2 |
| A17: improving controller before mechanism proof | Scope risk | Locked controller semantics and scientific gate | Method registry/version and decision tests | M0, M7 |
| A18: accuracy budget interpreted as guarantee | Future-research risk | Defer optimizer; distinguish empirical satisfaction from guarantee | Explicit report limitation | M7 |

## 11. Completion and review checklist

Implementation is ready for an approved full T4 study only when:

- all M0–M5 correctness tests pass; a CUDA smoke and interrupted/resumed run pass in the intended environment;
- historical algorithm definitions remain identifiable and old files/hashes remain unchanged;
- all masks, data order, augmentations, optimizer steps, normalization modes, and last-batch scaling are verified;
- source/replay learned trajectories agree in the same execution environment;
- every run saves sufficient state to diagnose and resume, with atomic local and checksum-verified remote persistence;
- timing is collected from warm-up onward, reconciles, and distinguishes measured, attributed, replay-conditional, and deployment-eligible costs;
- all controller/methods have comparable diagnostic coverage without unreported probe overhead;
- no unattained target, failed run, cached result, shadow cost, or random-mask replica is misrepresented;
- the staged matrix, primary contrasts, thresholds, seeds, ordering, and budget are approved and saved before fresh-seed outcomes are examined;
- Colab instructions work from a fresh runtime and an explicit resume path, with no hidden dependency on laptop-only files.

Review the implementation in this order: mathematics/state transitions, data/mask identity, optimizer/normalization semantics, persistence/reproducibility, timing/accounting, statistical interpretation, then performance/readability. Do not use smoke-test accuracy or timing to support research conclusions.

## 12. Decisions proposed for user approval before scientific execution

These are the only unresolved scientific defaults in this plan. Implementation can expose them as explicit configuration values and tests; it must not choose replacements independently.

| Decision | Recommended default | Why approval matters |
|---|---|---|
| Practical effect margins | 0.5 pp accuracy and 5% runtime; report all estimates/intervals regardless | These express research priorities, not facts supplied by the data. |
| Locked seed/mask budget | Fresh seeds 3–7; three gap replicas/seed only for the locked controller question | Five seeds and repeated masks cost materially more than the pilot; precision may still be limited. |
| Full-run expansion | One-seed core pilot first; conditional controls promoted according to Section 8 | Avoid committing to the maximum 55-trajectory core plus dependencies/repetitions prematurely. |
| Confirmatory evaluation scope | Reused validation split for mechanism work; official test untouched until a separately locked final evaluation | Fresh initialization seeds do not undo prior validation-set use. |

No approval is requested here for an unspecified implementation or an open-ended compute budget. The next user authorization should identify whether to implement the framework only, run engineering tests, or execute a concrete scientific batch. Any unanticipated scientific ambiguity must be documented with a recommended default and its impact before dependent experiments proceed.

## 13. Plan consistency review

This plan has been reviewed against the source and prior audit for the following points:

1. Timing begins in Stage 1; Stage 4 uses already valid measurements and adds clean benchmarks.
2. Randomized masks have explicit count, gap-length, boundary, and RNG invariants rather than a vague spacing claim.
3. Causal shadow controls and practical no-predictor controls have different identifiers, costs, and eligibility.
4. Statistical units are classifier seeds, not epochs, batches, masks, or hardware repetitions; uncertainty and inconclusive outcomes remain visible.
5. The study is staged rather than a full Cartesian product; conditional diagnostics cannot silently expand the run matrix.
6. v3 controller/scaling/clipping definitions remain intact. Any backend difference is recorded, and algorithm changes require a new condition.
7. BN transition, predictor usefulness, and reliability-aware switching remain hypotheses; no unverified diagnosis mandates an architecture change.
8. Retrospective donor generation is compatible with controlled benchmark ordering and separately accounted dependency cost.
9. Resume state, logging reconciliation, and timing uncertainty after interruption are explicitly separated.
10. The final gate requires a practical accuracy–time advantage before advancing toward an accuracy-budget optimizer.

End of plan. Do not begin implementation or experiments without explicit authorization.
