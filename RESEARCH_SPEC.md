# Research Project Handoff: Adaptive Synthetic Gradients for Faster Neural Network Training

## 1. Project Overview

We are investigating whether neural networks can be trained faster by **replacing selected exact backpropagation steps with learned synthetic gradients**, while maintaining an acceptable level of predictive accuracy.

Our long-term objective is an **accuracy-constrained, compute-efficient training system**: the user specifies how much accuracy degradation they are willing to tolerate relative to conventional backpropagation, and the system tries to maximize training speed without exceeding that degradation budget.

The research direction builds on:

- **Decoupled Neural Interfaces Using Synthetic Gradients** — Jaderberg et al., 2017.
- Theoretical analyses of synthetic-gradient learning, including work by Czarnecki and collaborators.
- **ADA-GP (MICRO 2023)**, which combines gradient prediction with conventional backpropagation through scheduled switching.

Our initial target is an incremental but scientifically defensible improvement over scheduled gradient prediction—not necessarily a completely new learning algorithm.

We have completed multiple preliminary experiments, culminating in a three-seed CIFAR-10/ResNet-18 study that demonstrates measured runtime savings. However, important methodological problems and competing explanations remain.

**The immediate task is to audit the supplied code, implement a corrected experimental framework, and prepare it for Google Colab T4 GPUs. We must establish whether learned synthetic gradients and error-driven switching actually cause the improvement before developing the accuracy-budget optimization system.**

Do not treat the preliminary results as proof of our hypothesis.

---

## 2. The Core Research Idea

Consider a neural network partitioned into an upstream trunk and a downstream head:

\[
h=f_\theta(x), \qquad \hat y=g_\phi(h).
\]

Conventional backpropagation computes the downstream activation gradient

\[
g_h=\frac{\partial\mathcal L}{\partial h}
\]

and propagates it through the trunk:

\[
\nabla_\theta \mathcal L
=
\left(\frac{\partial h}{\partial\theta}\right)^T g_h.
\]

Instead, train a relatively inexpensive predictor

\[
\hat g_h=P_\psi(h,y)
\]

to approximate the gradient at the trunk–head boundary.

On a synthetic-gradient step, the head's backward computation is skipped. The predicted activation gradient is injected into the trunk:

\[
\widehat{\nabla_\theta\mathcal L}
=
\left(\frac{\partial h}{\partial\theta}\right)^T\hat g_h.
\]

This saves downstream backward computation, provided predictor inference is sufficiently cheap.

There is an unavoidable tradeoff:

- More exact BP steps increase computational cost but provide accurate gradients and fresh predictor-training targets.
- More synthetic steps reduce computation but introduce gradient errors and may destabilize training.

The central question is **when to use each type of update**.

### Our proposed improvement

Instead of following a fixed schedule, estimate synthetic-gradient reliability from recent exact-gradient observations and adapt the number of synthetic steps performed before the next exact correction.

The current implementation chooses among:

- 0 synthetic steps after an exact step.
- 2 synthetic steps after an exact step.
- 4 synthetic steps after an exact step.

Eventually, we want to optimize a policy \(\pi\) against an accuracy-degradation budget:

\[
\min_\pi T_\pi
\]

subject to

\[
A_{\mathrm{exact}}-A_\pi\leq\delta.
\]

Here:

- \(T_\pi\) is the total training time of policy \(\pi\).
- \(A_{\mathrm{exact}}\) is the accuracy of a conventional exact-BP baseline.
- \(A_\pi\) is the accuracy achieved by the proposed policy.
- \(\delta\) is the maximum allowed accuracy loss, measured in **percentage points**, not relative percentages.

Potential research targets are \(\delta=0.5,1,2,3,5\) percentage points.

The objective is to improve the **accuracy–compute Pareto frontier**, not merely to achieve the lowest number of exact backpropagation batches.

An important constraint: an online controller cannot know its final accuracy gap in advance. The eventual budget controller will need a defensible strategy for estimating or controlling accuracy risk, and we must distinguish empirical budget satisfaction from a formal guarantee.

---

## 3. Background: What We Already Investigated

The project originated from an idea initially described as probabilistic backpropagation: learning or approximating gradients and deciding when exact computation is necessary.

We progressively refined this into a concrete synthetic-gradient training system.

### Phase A — Initial gradient prediction

Architecture: small Fashion-MNIST MLP.

We trained a gradient predictor and examined whether Gaussian predictive uncertainty could identify unreliable gradients.

Findings:

- Initial gradient prediction was weak.
- A class-conditional mean-gradient baseline was surprisingly competitive.
- Gaussian uncertainty correlated strongly with absolute prediction error.
- However, uncertainty largely tracked gradient magnitude rather than dangerous update directions.

Conclusion: gradient uncertainty alone was insufficient.

### Phase B — Better predictors

We compared:

- MSE-trained predictors.
- Cosine-plus-MSE predictors.
- Gaussian predictors.
- Class-mean and residual predictors.

Cosine-plus-MSE training achieved strong directional predictions in the frozen-classifier experiment, reaching approximately 0.894 mean cosine similarity.

However, direction alone did not establish that the approximate parameter update would be useful.

### Phase C — Predicting gradient risk

We trained a risk estimator to identify potentially poor predicted gradients.

On held-out observations:

- Learned risk AUROC reached approximately 0.966 in one frozen-model setting.
- Performance deteriorated somewhat with model drift.
- Simple heuristics such as predicted gradient norm remained competitive.
- Better risk rankings did not necessarily imply proportionally better training updates.

Conclusion: ranking gradient errors was possible, but sophisticated risk-estimation models might cost more than they were worth.

### Robustness experiments R0/R1/R1+

We tested predictor quality at later classifier checkpoints and under drift.

The cosine-plus-MSE predictor maintained relatively strong gradient directions but showed substantial deterioration in normalized RMSE and gradient magnitude accuracy as classifier training progressed.

In a virtual SGD-update experiment, an approximate gradient produced only about 26% of the improvement achieved by an exact gradient on the same minibatch, despite reasonably high activation-gradient cosine similarity.

This was an important warning:

**High activation-gradient cosine similarity does not guarantee useful parameter updates.**

We therefore shifted toward inexpensive scalar feedback controllers rather than increasingly complicated uncertainty models.

### Phase D v1 — Actual synthetic-gradient training

Fashion-MNIST MLP.

We introduced actual synthetic-gradient substitution, comparing conventional BP, scheduled switching, feedback switching, and periodic switching.

Result: synthetic-gradient methods generally reduced accuracy and were **slower** than conventional backpropagation.

### Phase D v2 — Better feedback logic

We introduced rolling-quantile feedback, stronger controls, and profiling.

The relative-feedback controller consistently outperformed matched periodic switching on validation accuracy in the small MLP setting.

However, end-to-end runtime still favored exact BP.

Profiling showed why: the downstream MLP was too cheap, so predictor inference and training overhead eliminated the savings.

### Phase D v3 — CIFAR-10/ResNet-18

We moved to a deeper network to make gradient approximation potentially worthwhile.

**This is the current experiment and the code Codex must inspect and correct.**

The primary file is:

`gradient_phase_d_resnet.py`

The experimental outputs are in:

`cifar_d_resnet/`

We completed three seeds: 0, 1, and 2.

---

## 4. Existing CIFAR-10/ResNet Experiment

### Dataset and model

- CIFAR-10.
- Approximately 45,000 training examples and 5,000 validation examples.
- CIFAR-style ResNet-18, partitioned into a trunk and downstream head.
- Batch size: 32.
- Four exact-backpropagation warm-up epochs.
- Twelve subsequent training epochs.
- Sixteen epochs total.
- Approximately 1,407 minibatches per epoch.
- The existing script handles augmentation, shared initialization, optimizer setup, and reproducibility settings.

The exact architecture split, predictor dimensions, learning-rate implementation, and augmentation pipeline must be checked directly in the supplied source. Do not infer missing details from this document.

### Existing methods

**1. `exact`**

Conventional exact backpropagation on every batch.

This is the primary accuracy and runtime reference.

**2. `scheduled`**

An ADA-GP-inspired fixed switching schedule:

| Post-warm-up epochs | Synthetic : exact ratio | Exact fraction |
|---|---|---|
| 1–3 | 4:1 | 20% |
| 4–6 | 3:1 | 25% |
| 7–9 | 2:1 | 33.3% |
| 10–12 | 1:1 | 50% |

It is important to label this as **ADA-GP-inspired**, not an exact reproduction of published ADA-GP.

**3. `relative_feedback`**

Uses recent relative gradient prediction error to choose the synthetic quota.

The controller maintains a rolling distribution of errors and selects among 0, 2, and 4 synthetic steps.

Larger relative errors lead to more frequent exact-gradient corrections.

**4. `absolute_feedback`**

Uses absolute prediction error and fixed calibration thresholds to select the quota.

**5. `matched_periodic`**

A periodic baseline matched to the absolute-feedback method's exact-gradient count.

**6. `exact_head_mask`**

Uses exact trunk gradients but restricts head updates according to the absolute-feedback method's mask.

It helps investigate whether skipping head updates contributes to accuracy loss.

### Training process

The high-level workflow is:

1. Set random seeds and create fixed train/validation partitions.
2. Initialize the classifier and synthetic-gradient predictor.
3. Run exact-BP warm-up.
4. Train the predictor using exact activation gradients during warm-up or calibration according to the implementation.
5. Store reusable warm-up and calibration states.
6. Initialize method-specific copies from shared states.
7. Run twelve post-warm-up epochs using the selected switching strategy.
8. On exact batches, compute true activation gradients and train the predictor.
9. On synthetic batches, predict the activation gradient and backpropagate it only through the trunk.
10. Evaluate validation accuracy each epoch.
11. Save method results, controller logs, profiling, and summary CSVs.

The implementation uses a gradient-injection operation equivalent to:

```python
h.backward(predicted_activation_gradient)
```

with appropriate gradient scaling.

Earlier code review found no obvious fundamental error in this injection mechanism, but it must be revalidated through automated gradient-equivalence tests.

### Existing command-line usage

On the original Windows environment:

```bash
py -3.13 gradient_phase_d_resnet.py --device cuda --mode preflight --deterministic --seeds 0 --out cifar_d_resnet
```

```bash
py -3.13 gradient_phase_d_resnet.py --device cuda --mode full --deterministic --seeds 0 1 2 --out cifar_d_resnet
```

The revised implementation must additionally support Linux/Colab execution and reliable checkpoint-resume behavior.

---

## 5. Completed Three-Seed Results

These are the actual recorded results, not predictions.

### Final validation accuracy

| Method | Seed 0 | Seed 1 | Seed 2 | Mean |
|---|---:|---:|---:|---:|
| Exact BP | 90.00% | 89.98% | 89.70% | 89.89% |
| Relative feedback | 85.28% | 85.20% | 84.66% | 85.05% |
| Scheduled | 72.70% | 61.94% | 75.18% | 69.94% |
| Absolute feedback | 81.02% | 80.98% | 49.70% | 70.57% |
| Matched periodic | 80.52% | 80.32% | 49.40% | 70.08% |
| Exact head-mask control | 87.70% | 87.86% | 87.92% | 87.83% |

Relative feedback used approximately **33.9% exact-backpropagation batches**.

Scheduled switching used approximately **32.1%**.

Absolute feedback and its periodic baseline used approximately **20.1%**.

### Actual end-to-end runtime

| Seed | Exact BP | Relative feedback | Measured speedup |
|---|---:|---:|---:|
| 0 | 949.7 s | 809.6 s | 1.17× |
| 1 | 1391.2 s | 1139.6 s | 1.22× |
| 2 | 857.1 s | 744.4 s | 1.15× |

Relative feedback's mean validation-accuracy deficit was approximately **4.85 percentage points**.

Thus, we have observed an accuracy–runtime tradeoff: relative feedback is faster but less accurate.

The speedups are provisional because GPU performance varied substantially on the laptop, potentially due to thermals or power limits.

### Important observations

1. Relative feedback is substantially more accurate than the scheduled method in all three seeds.
2. Relative feedback produces strikingly consistent final accuracy across seeds.
3. The absolute-feedback and periodic methods both collapse in seed 2.
4. A scheduled method using almost the same overall exact-BP fraction as relative feedback performs much worse.
5. The exact head-update control reaches approximately 87.8%, suggesting that reduced head-update frequency alone may explain a meaningful portion of the accuracy deficit.

These are valid observations, but they do not establish that the proposed feedback signal is responsible for the gains.

---

## 6. Gradient Predictor and Controller Diagnostics

### 6.1 High relative gradient prediction error

The logged relative gradient errors are concerning.

Approximate average relative error during the first post-warm-up epoch:

| Seed | Mean relative error |
|---|---:|
| 0 | 0.985 |
| 1 | 0.981 |
| 2 | 0.988 |

By the final epoch:

| Seed | Mean relative error |
|---|---:|
| 0 | 0.994 |
| 1 | 0.992 |
| 2 | 0.995 |

An error close to 1 means the residual norm is roughly as large as the target-gradient norm, subject to the implementation's denominator floor.

This does **not** by itself prove that the predicted gradients are useless. We need directional cosine similarity, gradient norm ratios, and effective parameter-update measurements.

But it introduces a critical competing hypothesis:

**Maybe the method succeeds because its exact BP steps perform most of the useful learning, while synthetic-gradient steps contribute little or nothing.**

The next experiment must test this.

### 6.2 Relative feedback settles into a nearly fixed overall budget

Cumulative controller choices:

| Seed | Quota 0 | Quota 2 | Quota 4 |
|---|---:|---:|---:|
| 0 | 1,990 | 1,911 | 1,833 |
| 1 | 2,015 | 1,867 | 1,854 |
| 2 | 1,948 | 1,907 | 1,843 |

The policy continually updates rolling quantiles of relative prediction error.

Those quantiles move with the underlying distribution, resulting in approximately one-third of decisions selecting each quota.

This may be producing an advantageous exact-BP frequency without the error signal providing genuinely useful batch-selection information.

We must compare against policies that reproduce the same exact-gradient counts and temporal distribution without using prediction errors.

### 6.3 Absolute-feedback controller failure

The calibrated thresholds were approximately:

- Seed 0: 14.06 and 15.62.
- Seed 1: 14.15 and 16.11.
- Seed 2: 14.34 and 16.61.

In seed 2, mean absolute gradient error fell dramatically despite the classifier performing poorly.

After the early post-warm-up period, the absolute controller virtually always selected four synthetic updates.

For example, seed 2's total quota decisions were approximately:

- Quota 0: 5.
- Quota 2: 18.
- Quota 4: 3,365.

The fixed thresholds stopped being useful once the gradient-error scale shifted.

This is an important failure of the controller design—not necessarily a software bug.

### 6.4 Per-batch GPU profiling

Approximate median batch times on the RTX 3060 laptop:

| Operation | Seed 0 | Seed 1 | Seed 2 |
|---|---:|---:|---:|
| Exact BP | 25.87 ms | 24.00 ms | 24.19 ms |
| Exact BP + predictor training | 30.81 ms | 31.33 ms | 32.23 ms |
| Synthetic-gradient update | 12.89 ms | 13.45 ms | 13.03 ms |

Synthetic batches are genuinely cheaper per operation.

However, predictor training and inference have significant overhead, and the observed end-to-end speedup is much smaller than the reduction in exact-BP fraction.

We need to optimize the complete training process, not merely count skipped backward passes.

---

## 7. Problems Identified in the Existing Experimental Setup

These are the issues that must be addressed before making publication-level claims.

### Issue 1 — Relative feedback lacks a properly matched periodic baseline

**Severity: Critical**

The `matched_periodic` method is matched to `absolute_feedback`, not `relative_feedback`.

The comparison is therefore approximately:

- Relative feedback: 34% exact BP.
- Matched periodic: 20% exact BP.

A superiority claim based on that comparison is invalid.

**Fix:**

Add:

- Periodic baseline with the same overall exact-BP count as relative feedback.
- Periodic baseline with the same number of exact steps in every epoch.
- Optionally, a replayed-mask baseline with exact positions derived from relative feedback.

Distinguish retrospective matched controls from practical online policies. A periodic schedule constructed after observing a feedback run's future exact-step counts is a diagnostic control, not a deployable online competitor.

### Issue 2 — Different early-training exact-gradient allocation

**Severity: Critical**

Scheduled switching uses approximately 20% exact BP immediately after warm-up.

Relative feedback initially uses approximately 40%.

Both use similar total exact-gradient budgets over twelve epochs, but relative feedback spends more of that computation early.

This matters because early updates may have greater influence, especially at higher learning rates.

**Fix:**

Include:

- Early-front-loaded fixed schedule.
- Total-budget-matched fixed schedule.
- Per-epoch-budget-matched periodic schedule.
- A schedule matching relative feedback's per-epoch exact fractions but not its within-epoch switching decisions.

This allows us to separate the effects of **how much**, **when**, and **why** exact backpropagation is performed.

### Issue 3 — BatchNorm mode transition at warm-up boundary

**Severity: High**

The existing code uses:

```python
model.train()
```

during warm-up, then switches to behavior equivalent to:

```python
model.trunk.train()
model.head.eval()
```

during post-warm-up training.

Consequently, downstream BatchNorm layers change from using minibatch statistics to using stored running statistics.

The gradient predictor may have been trained on gradient targets generated under a different normalization regime from the one encountered during synthetic training.

Also, train-mode BatchNorm introduces cross-sample dependence. Multiplying the gradient of a batch-mean loss by batch size is not generally equivalent to independent per-example gradients when cross-sample operations are active.

**Fix:**

- Audit the exact location of all BatchNorm layers.
- Check whether gradient targets are computed under a consistent normalization regime.
- Explicitly measure predictor error across the warm-up-to-post-warm-up transition.
- Implement a normalization-consistent experimental condition.
- Consider a BN-free downstream head using suitable alternative normalization if justified.
- Preserve the original model as a separate reference condition.

Do not silently change the architecture and attribute all accuracy differences to the switching controller.

### Issue 4 — Head-update frequency is confounded with gradient prediction

**Severity: High**

Synthetic batches bypass downstream-head training.

Thus, more synthetic batches also mean fewer updates to head parameters.

The current head-update control follows the absolute-feedback mask rather than the relative-feedback mask.

**Fix:**

Create a control that:

- Uses exact trunk gradients on every batch.
- Performs head optimizer updates only at the exact-step positions used by relative feedback.
- Replays the same mask, data ordering, and optimizer schedule.

This isolates loss attributable to reduced head training rather than approximate gradients.

Also consider a separate ablation that updates the head on synthetic batches, but report the additional computation explicitly.

### Issue 5 — Time-to-accuracy target is meaningless

**Severity: High**

The current default target is 70% accuracy.

Some runs already satisfy this target during warm-up, before synthetic-gradient switching begins.

As a result, `seconds_to_target` does not meaningfully compare training policies.

**Fix:**

- Report warm-up accuracy separately.
- Predefine more challenging accuracy targets, such as 85%, 87%, and 89%, when appropriate for the selected setup.
- Record actual first attainment of each target.
- Mark unattained targets as censored/not reached.
- Report both post-warm-up time and full end-to-end time.
- Never claim a speedup to an accuracy level that the approximate method did not reach.

### Issue 6 — Relative feedback may not detect global deterioration

Rolling quantile thresholds track the error distribution.

If all gradient errors become worse, the threshold distribution can move upward too.

The controller can continue issuing approximately the same mixture of decisions.

**Fix:**

Keep relative quantile feedback as one baseline, but evaluate absolute reliability safeguards, calibrated normalized-error thresholds, and history-aware alternatives.

Avoid adding a complex neural risk estimator without evidence that it provides enough benefit to justify the overhead.

### Issue 7 — Absolute feedback almost never adapts

Its thresholds are not appropriately tracking changes in gradient scale.

The implementation effectively degenerates to a fixed 4:1 synthetic-to-exact schedule.

**Fix:**

Introduce a better normalized or dynamically calibrated error mechanism, but preserve the original method as a separate baseline to demonstrate the improvement.

A key design constraint is that low absolute error caused by vanishing gradients must not automatically be interpreted as safe synthetic training.

### Issue 8 — Arbitrary relative-error denominator floor

The original error score uses a minimum denominator around 0.05.

This can distort the interpretation when true gradient norms become small.

**Fix:**

Investigate scale-aware normalization and explicitly log:

- True gradient norms.
- Predicted gradient norms.
- Absolute residual norms.
- Relative residual norms.
- The fraction of samples or batches affected by the denominator floor.

Specify the exact aggregation convention.

### Issue 9 — Calibration uses training examples

The calibration data is not a statistically independent held-out reliability dataset.

This is not necessarily invalid for an online controller trained using training data, but it limits generalization and calibration claims.

**Fix:**

Separate:

- Data used to train the classifier.
- Data used for predictor supervision.
- Data used for controller calibration, where required.
- Validation data used for evaluation and model selection.
- Final held-out test data.

Avoid validation/test leakage into the online switching policy.

If a separate calibration split is costly, use an explicitly documented online training-stream calibration protocol and avoid claiming held-out calibration validity.

### Issue 10 — Different gradient clipping behavior

Exact steps clip joint trunk/head gradients, while synthetic steps clip trunk gradients.

This can create different effective update magnitudes independently of gradient quality.

**Fix:**

Define a consistent, scientifically justified gradient-clipping protocol.

Include a diagnostic that logs clipping frequency and pre/post-clipping norms by update type.

### Issue 11 — Checkpoint reuse and replay-mask verification

The original caching system may reuse a run without sufficiently verifying the requested replay mask.

**Fix:**

Every cache entry should include a configuration and mask fingerprint.

Checkpoint validation must verify:

- Architecture.
- Dataset partition.
- Random seed.
- Method.
- Hyperparameters.
- Training stage.
- Replay-mask identity.
- Predictor/controller configuration.
- Code version or experiment schema.

Never silently reuse an incompatible result.

### Issue 12 — Preflight estimates are not end-to-end runtime

The current feasibility estimate uses selected operation costs.

It does not fully capture:

- Data loading.
- Validation.
- Calibration.
- Predictor training throughout the run.
- Warm-up.
- Host overhead.
- Thermal or power variability.
- Additional training needed to reach the same accuracy.

**Fix:**

Report microbenchmarks and full-run timing separately.

Use actual end-to-end measurements for speedup claims.

### Issue 13 — Insufficient training horizon

Four warm-up plus twelve experimental epochs may not characterize final convergence.

A method that collapses initially might recover with longer training.

**Fix:**

First reproduce the sixteen-epoch setup, then conduct longer-horizon experiments under fixed and matched computational budgets.

Report both fixed-epoch and accuracy-to-target results.

### Issue 14 — Limited dataset/architecture coverage

Current acceleration results come from one model family and one dataset.

**Fix:**

After establishing a credible causal effect, evaluate at least one additional architecture or training setup.

Do not expand to expensive architectures before the core ResNet experiment is resolved.

### Issue 15 — Incomplete gradient-quality diagnostics

The scheduled baseline does not have a directly comparable gradient-error trace.

Current logged relative errors also do not establish cosine similarity, useful magnitude, or actual parameter-update quality.

**Fix:**

Log the same gradient-quality diagnostics for all switching policies on appropriately selected exact-gradient probe batches.

Do not perform expensive exact-gradient probes on nominally synthetic batches without accounting for their computational cost. Prefer separate diagnostic runs or a clearly identified probe overhead.

---

## 8. The New Essential Ablation: Do Synthetic Gradients Actually Help?

This is the most important additional experiment prompted by the three-seed results.

The predictor's relative L2 error is close to 1.

It is therefore possible that the primary benefit of relative feedback comes from its schedule of exact-gradient steps, rather than useful synthetic-gradient contributions.

We must distinguish these possibilities.

### A. Learned-gradient replay

Run the existing relative-feedback method and record its exact/synthetic update mask.

This is our candidate method.

### B. Zero-gradient replay

Use the identical exact/synthetic mask.

On synthetic-designated batches, inject a zero activation gradient instead of the predicted gradient.

Keep all other relevant behavior controlled.

**Important optimizer caveat:** A zero gradient supplied to AdamW or an optimizer with momentum is not necessarily equivalent to skipping the parameter update. Momentum, weight decay, and optimizer state may still change.

Therefore, distinguish:

1. A zero-gradient injection with normal optimizer stepping.
2. A true trunk-update skip with no optimizer step.

Document the behavior of both.

### C. Matched fixed-schedule synthetic gradients

Use learned synthetic gradients with a fixed schedule matched to the exact-gradient budget of relative feedback.

This isolates the controller's switching decisions from the predictor itself.

### D. Matched fixed-schedule zero gradients

Use zero or skipped synthetic updates with that same fixed schedule.

This provides a direct estimate of the benefit attributable to the predictor rather than just intermittent exact BP.

### E. Optional randomized-gradient control

Inject randomized or norm-matched synthetic gradients, if useful, to distinguish meaningful learning from generic update noise or regularization.

Do not make this mandatory for the first implementation.

### Interpreting possible outcomes

If learned gradients significantly outperform zero gradients under identical masks, we have evidence that the predictor supplies useful updates.

If learned and zero-gradient methods perform similarly, then our current predictor may be mostly unnecessary.

If relative feedback does not beat an identical-budget fixed schedule using the same predictor, its claimed intelligent switching advantage is unsubstantiated.

If relative feedback beats a per-epoch-matched periodic policy, we have evidence that within-epoch selection matters beyond early front-loading.

The project should be willing to reject its original hypothesis if these controls demonstrate a simpler explanation.

---

## 9. Required Gradient-Quality Diagnostics

For sampled exact-gradient batches, calculate:

### Direction

\[
\cos(g,\hat g)
=
\frac{\langle g,\hat g\rangle}
{\|g\|\|\hat g\|+\epsilon}.
\]

### Relative L2 error

\[
e_{\mathrm{rel}}
=
\frac{\|g-\hat g\|}
{\max(\|g\|,\epsilon)}.
\]

### Predicted-to-true norm ratio

\[
r_{\mathrm{norm}}
=
\frac{\|\hat g\|}
{\max(\|g\|,\epsilon)}.
\]

### Projection onto the true gradient

\[
r_{\mathrm{proj}}
=
\frac{\langle\hat g,g\rangle}
{\|g\|^2+\epsilon}.
\]

### Effective parameter-gradient agreement

Where computationally practical, examine the resulting trunk parameter-gradient vectors and their agreement with the exact parameter gradients.

### Short-horizon update utility

On a diagnostic minibatch, compare the loss changes produced by hypothetical exact, predicted, zero, and potentially randomized updates.

Do not confuse improvements on the same training batch with generalization improvements.

Report gradient norm distributions and near-zero-gradient frequencies.

The purpose is to determine whether the learned predictor approximates gradients meaningfully or simply produces small, relatively harmless updates.

---

## 10. Corrected Experimental Matrix

The next controlled experiment should include the following methods.

| Method | Purpose |
|---|---|
| Exact BP | Reference accuracy and runtime |
| Original scheduled | Preserve historical baseline |
| Original relative feedback | Preserve historical candidate |
| Original absolute feedback | Failure-mode comparison |
| Periodic, total-budget matched | Isolate overall exact-BP frequency |
| Periodic, per-epoch-budget matched | Isolate within-epoch selection |
| Early-front-loaded fixed schedule | Test importance of early exact BP |
| Relative-feedback mask + zero gradients | Test predictor usefulness |
| Relative-feedback mask + skipped updates | Distinguish optimizer-state effects |
| Relative-feedback mask + exact trunk gradients | Isolate head-update confounding |
| Optional improved relative/absolute controller | Evaluate corrected reliability logic |

Not every method needs to run in the first test.

Implement and validate the core controls first; prioritize the smallest experiment that can decisively isolate the mechanisms.

### Fair-comparison requirements

Hold fixed, wherever relevant:

- Initial model weights.
- Warm-up classifier state.
- Predictor initialization/training procedure.
- Data split and minibatch ordering.
- Augmentations.
- Optimizer families and hyperparameters.
- Learning-rate schedules.
- Batch size.
- Epoch budget.
- Normalization behavior.
- Gradient clipping.
- Evaluation protocol.

**An especially important principle:** The matched-periodic method must not get access to the feedback method's future decisions during its online training. Retrospective per-epoch matching is allowed as a clearly labeled diagnostic.

A separate online fixed-budget schedule should be available as a realistic deployable baseline.

---

## 11. Google Colab T4 Migration

The next corrected experiment will run primarily on **Google Colab with an NVIDIA T4 GPU**, rather than the original RTX 3060 laptop GPU.

Reasons:

- Avoid laptop overheating and performance throttling.
- Obtain 16 GB GPU memory.
- Run longer experiments without depending on laptop thermals.
- Evaluate the idea on a second GPU architecture.

The T4 is not necessarily faster than the RTX 3060.

Do not compare exact BP measured on the RTX 3060 against relative feedback measured on the T4. All methods within a speedup comparison must execute on the same hardware under comparable conditions.

### Environment requirements

Provide:

- A reproducible Python environment.
- Compatible PyTorch and torchvision versions.
- CUDA availability checks.
- A Colab setup notebook or documented setup commands.
- Dataset download/cache management.
- Google Drive persistence for checkpoints and results.
- Simple commands for preflight, smoke tests, and full runs.

Preserve batch size 32 initially so that architectural and scheduling changes can be studied without simultaneously changing batch-size dynamics.

Larger batch sizes can be investigated separately.

Disable mixed precision for the first corrected reference experiment unless it was already used consistently. If mixed precision is later introduced, benchmark all methods under the same controlled precision regime.

### Checkpoint/resume

Colab sessions can disconnect unexpectedly.

The new implementation must support checkpointing at least once per epoch, preferably with optional more frequent checkpoints.

A checkpoint must contain sufficient state for an accurate resume:

- Trunk and head weights, including BatchNorm state.
- Predictor weights.
- Model optimizer states.
- Predictor optimizer state.
- Learning-rate schedulers.
- Controller rolling-history/EMA state and thresholds.
- Synthetic quota and controller position.
- Epoch and global step.
- Data-ordering state.
- Python, NumPy, CPU PyTorch and CUDA RNG states.
- Any replay-mask information.
- Accumulated metrics and elapsed timing.
- Full experiment configuration.

For exact epoch-boundary resumption, save the random/data-loading state needed to reproduce the next epoch.

For mid-epoch resumption, ensure the sampler and batch cursor are recoverable. If this cannot be guaranteed, support documented epoch-boundary resumption first rather than claiming exact mid-epoch reproducibility.

Use atomic checkpoint writes and clearly distinguish resumed runs from fresh runs in the metadata.

### Outputs

Every run should produce:

- Configuration and environment metadata.
- Per-epoch metrics.
- Final summary.
- Controller decisions.
- Exact-gradient budget counts.
- Gradient-quality diagnostics.
- Profiling information.
- Resume/checkpoint metadata.
- Method masks and mask hashes.
- Error logs.
- Aggregate multi-seed tables.

All CSV outputs should have stable schemas and documented units.

---

## 12. Timing and Evaluation Standards

The original laptop results contain substantial timing variation.

For example, the all-exact, sparsely updated head control sometimes ran markedly faster than full exact BP despite performing comparable backward computation.

That is a warning that hardware conditions may confound the original timings.

The corrected experiment must measure:

1. Warm-up time.
2. Controller/predictor calibration time.
3. Post-warm-up training time.
4. Validation time.
5. Total training time.
6. Exact batch mean/median latency.
7. Synthetic batch mean/median latency.
8. Predictor-training overhead.
9. Inference overhead.
10. GPU peak memory.

GPU timing should use appropriate synchronization or CUDA events.

CPU wall-clock time should measure actual elapsed training.

Report both:

\[
S_{\mathrm{full}}
=
\frac{T_{\mathrm{exact,full}}}
{T_{\mathrm{method,full}}}
\]

and

\[
S_{\mathrm{postwarm}}
=
\frac{T_{\mathrm{exact,postwarm}}}
{T_{\mathrm{method,postwarm}}}.
\]

The second metric isolates the switching phase, while the first reflects the actual full training workflow.

Shared warm-up costs must be accounted for consistently.

A method that is fast but cannot reach the target accuracy must not be reported as faster *to that accuracy*.

For accuracy-to-target experiments, record unattained targets explicitly.

---

## 13. Experimental Roadmap

### Stage 0 — Code audit and reproducibility

Before changing algorithms:

1. Read the entire existing script.
2. Map all model, optimizer, predictor and controller state transitions.
3. Identify exact/synthetic data flow.
4. Audit normalization and gradient scaling.
5. Verify optimizer behavior.
6. Preserve a runnable reference version.
7. Implement correctness tests.
8. Add Colab execution and checkpointing.

**Deliverable:** an independently runnable and validated baseline.

### Stage 1 — Correct causal controls

Run the controlled experiments that answer:

- Are predicted gradients better than zero gradients?
- Is the benefit caused by more exact-gradient steps early?
- Does choosing batches based on measured error outperform an equally budgeted fixed schedule?
- Does BatchNorm inconsistency explain the collapse?
- How much does sparse head training affect accuracy?

This stage has the highest immediate scientific value.

**Deliverable:** a well-controlled mechanistic comparison.

### Stage 2 — Improve the switching controller

Only after establishing that synthetic gradients and/or feedback decisions are useful:

Evaluate better control mechanisms.

Candidates include:

- Improved normalized-error feedback.
- Combined directional/magnitude reliability scores.
- Error-drift detection.
- Safeguards against vanishing-gradient false confidence.
- Explicit exact-gradient fraction targets.
- Adaptive thresholds with stable long-term reference statistics.
- Recovery modes after reliability deterioration.

Keep controller overhead minimal.

Avoid unnecessary neural risk-estimation modules unless they demonstrate clear net benefit.

### Stage 3 — Accuracy-degradation-budget optimization

This is our primary intended research contribution.

The user specifies an allowable loss \(\delta\).

The controller aims to minimize runtime while satisfying that budget.

Potential formulations include adaptive exact-gradient frequency, constrained optimization, or a lightweight online controller.

Experiments should sweep accuracy budgets and compute costs, rather than reporting one arbitrarily chosen policy.

For each budget, record:

- Achieved validation accuracy.
- Gap relative to exact BP.
- Exact-gradient fraction.
- Total runtime.
- Speedup.
- Frequency and severity of budget violations across seeds.
- Stability over training.

Compare to a range of fixed schedules spanning the same compute budgets.

Produce an accuracy–runtime Pareto frontier.

Critically, distinguish a method that empirically respects the budget from one offering a theoretical constraint guarantee.

### Stage 4 — Generalization and publication-level study

Once the corrected controller outperforms credible baselines:

- Run more random seeds.
- Extend the training horizon.
- Evaluate an additional architecture or dataset.
- Include official held-out test performance after model selection.
- Measure accuracy-to-target runtime.
- Report memory and compute overhead.
- Analyze failure conditions.
- Conduct ablation studies.
- Compare appropriately with published synthetic-gradient methods, including a faithful ADA-GP implementation if a claim against ADA-GP is intended.

A publication claim requires improvement against strong baselines, not just the historically weak scheduled configuration.

---

## 14. Tests That Must Be Implemented

### Gradient injection correctness

Inject the true boundary gradient through the synthetic path and confirm that the resulting trunk gradients agree with conventional BP within suitable numerical tolerance.

Test multiple batch sizes, and explicitly handle last-batch scaling.

### Gradient-target scaling

Verify the relationship between mean-reduced loss gradients and the predictor target, particularly around BatchNorm and other cross-example layers.

### Optimizer-state correctness

Check that exact, synthetic, zero-gradient, and skipped-update paths have the intended behavior.

Verify momentum/AdamW state and weight decay.

### Switching-controller invariants

Confirm:

- Every minibatch receives exactly one prescribed update type.
- Exact and synthetic counts add to total batches.
- Quotas are implemented correctly.
- No out-of-range controller actions occur.
- Replay masks are applied to the intended batch indices.

### Match-budget invariants

For matched controls, verify exact-gradient counts by:

- Full run.
- Epoch.
- Optionally training phase.

Fail loudly if a baseline's declared matching condition is violated.

### Normalization-state correctness

Check BatchNorm train/eval modes and running-statistic behavior during:

- Warm-up.
- Calibration.
- Exact updates.
- Synthetic updates.
- Evaluation.

### Reproducibility

Check common initializations, data order, augmentations, and method-specific randomness.

### Checkpoint/resume equivalence

A short run interrupted and resumed from a checkpoint should match an uninterrupted run to the precision supported by the execution environment.

### No-leakage checks

Verify validation and test examples do not enter the training loss, controller supervision, or predictor fitting unless a clearly specified experimental protocol requires and discloses it.

---

## 15. Existing Output Files

The current experiment produces or has produced files including:

- `summary.csv`
- `summary_aggregate.csv`
- `epochs.csv`
- `feasibility.csv`
- `paired_absolute_vs_baselines.csv`
- `seed_0_profile.csv`
- `seed_1_profile.csv`
- `seed_2_profile.csv`
- `seed_0_relative_feedback_controller.csv`
- `seed_1_relative_feedback_controller.csv`
- `seed_2_relative_feedback_controller.csv`
- `seed_0_absolute_feedback_controller.csv`
- `seed_1_absolute_feedback_controller.csv`
- `seed_2_absolute_feedback_controller.csv`
- Method-specific `.pt` results/checkpoints.

Some uploaded files have duplicate-safe names such as `summary(1).csv` and `epochs(1).csv`; treat these as uploaded copies, not separate experiments without checking their contents.

The complete archive is called:

`cifar_d_resnet.zip`

The original source script is:

`gradient_phase_d_resnet.py`

Codex should request or locate those files in its project workspace before making implementation-specific changes.

---

## 16. Implementation Requirements for Codex

### Preserve the original work

Do not overwrite the original Phase D v3 script or experimental results.

Create a versioned corrected implementation, such as:

`gradient_phase_d_resnet_v4.py`

or a small, appropriately modularized project.

Maintain the ability to reproduce the original method definitions before modifying their behavior.

### Separate baseline reproduction from new methods

Existing method names should retain their historical meaning.

Introduce new method names for corrected or improved policies.

For instance, avoid silently changing what `absolute_feedback` means and then reporting the new result as though it came from the original implementation.

### Keep the code research-oriented

Prioritize:

1. Mathematical correctness.
2. Fair comparisons.
3. Explicit configuration.
4. Reproducibility.
5. Logging.
6. Hardware efficiency.
7. Clean separation between controllers and update logic.

Avoid unnecessary abstractions and large frameworks.

### Avoid premature algorithm development

Do not immediately implement reinforcement learning, a sophisticated learned controller, or an elaborate accuracy-budget optimizer.

First demonstrate that the underlying predictor and error-feedback logic have a genuine causal advantage over simpler alternatives.

### Make all experimental conditions explicit

Every run must record:

- Network architecture and split point.
- Predictor architecture.
- Controller.
- Dataset partition.
- Seed.
- Optimizer configuration.
- Learning-rate schedule.
- Batch size.
- Normalization regime.
- Update-mask identity.
- Gradient-clipping rules.
- Epoch count.
- Precision mode.
- GPU model.
- Code/configuration version.

---

## 17. Immediate Deliverables Requested

For the first implementation, produce:

**A. Corrected training code**

A runnable PyTorch implementation suitable for Colab T4.

**B. Correct experimental controls**

At minimum:

- Exact BP.
- Original relative feedback.
- Original scheduled switching.
- Relative-feedback-matched periodic controls.
- Early-front-loaded schedule.
- Relative-feedback-matched exact head-update control.
- Relative-feedback mask with zero-gradient updates.
- Relative-feedback mask with skipped updates.

Preserve original absolute feedback as an optional failure-mode baseline.

**C. Normalization correction and diagnostic**

Explicitly resolve or isolate the BatchNorm regime-change concern, without quietly confounding architectural changes with controller changes.

**D. Gradient-quality diagnostics**

Log cosine similarity, norm ratios, relative and absolute errors, and relevant update-quality metrics.

**E. Checkpoint/resume**

Reliable epoch-level checkpoints suitable for Colab disconnections, including training and controller state.

**F. Colab instructions**

Provide a simple setup and execution workflow for preflight, smoke test, and full experiments.

**G. Automated tests**

Especially true-gradient injection equivalence, replay-mask correctness, exact-budget matching, checkpoint/resume, and optimizer behavior.

**H. Research documentation**

Briefly explain:

- Which bugs were actually found.
- Which problems are methodological confounders.
- Which controls address each confounder.
- Which hypotheses each ablation tests.
- Which comparisons are valid and invalid.

---

## 18. Acceptance Criteria

The corrected implementation is ready for full T4 experiments only when:

1. All correctness tests pass.
2. Exact-gradient injection is numerically validated.
3. Baseline data splits and initial states are matched.
4. Normalization behavior is documented and controlled.
5. Matched controls satisfy their exact-gradient-budget invariants.
6. Zero-gradient and skipped-update controls are not accidentally conflated.
7. Replayed head-update masks are verified.
8. A short interrupted/resumed run reproduces an uninterrupted run.
9. Preflight and smoke-test runs execute successfully on CUDA.
10. Output files contain sufficient information for independent analysis.
11. The original v3 experiments remain untouched.

Before running all expensive experiments, run a small smoke test with limited batches and verify the outputs.

Do not make unsubstantiated accuracy or speedup claims based on smoke tests.

---

## 19. What Constitutes a Meaningful Research Result?

There are several possibilities.

### Strong result

Relative feedback consistently beats budget-matched fixed policies, and learned synthetic gradients demonstrably outperform zero-gradient controls.

The method produces a superior accuracy–runtime tradeoff across seeds.

This would strongly motivate developing the accuracy-budget optimizer.

### Moderate result

Synthetic gradients help, but a simple front-loaded fixed schedule performs about as well as feedback.

The research could pivot toward better compute-budget allocation and prediction quality rather than error-adaptive switching.

### Alternative result

Relative feedback is better than the original schedule, but zero-gradient or skipped-update controls perform similarly to learned gradients.

Then the result may be primarily about sparse exact backpropagation or optimization schedules, not synthetic-gradient prediction.

We should not misattribute the improvement.

### Negative result

After strong controls and consistent normalization, learned synthetic gradients do not produce a favorable accuracy–runtime frontier.

Then the method needs a fundamental redesign, possibly changing the split point, predictor architecture, or exact/synthetic update mechanism.

The project must prioritize accurate conclusions over confirming the original hypothesis.

---

## 20. Final Direction

The broad research goal remains:

**Find the fastest practical way to train a neural network while respecting a user-specified accuracy-degradation budget, using synthetic-gradient prediction and adaptive allocation of exact backpropagation.**

The existing ResNet experiments are promising because they established a reproducible accuracy gap of approximately 4.85 percentage points at approximately one-third exact-BP usage, accompanied by observed speedups of roughly 1.15–1.22×.

But our strongest unresolved question is:

**Is the performance advantage caused by useful synthetic gradients and intelligent reliability-aware switching, or simply by a favorable schedule of intermittent exact updates?**

The next corrected experiment must answer that first.

### Instruction to Codex

Start by inspecting the supplied `gradient_phase_d_resnet.py` and output CSVs. Reconcile this handoff against the real code rather than assuming every previous diagnosis is correct.

Then implement the corrected Phase D v4 framework, keep the original implementation and results intact, add reproducibility and correctness tests, and prepare a Colab T4 workflow.

Do not proceed straight to the final accuracy-budget controller until the mechanism-isolation experiments have produced interpretable evidence.

**Important corrections to RESEARCH_SPEC.md**

Before implementation, incorporate these clarifications:

1. The v3 method demonstrates lower runtime at a fixed epoch count, but exact BP reaches the feedback method's final accuracy earlier in all three seeds. Therefore, superior training efficiency has NOT been established. Add early-stopped exact BP and accuracy-versus-wall-time comparisons as primary controls.

2. The exact-BP baseline uses frozen downstream-head BatchNorm statistics after warm-up. This is shared across methods and must be documented.

3. The existing relative-error score is batch-wide normalized RMSE with a denominator floor of 0.05, not per-example relative L2. Preserve it for v3 reproducibility and add separately defined diagnostic metrics.

4. Trunk and head optimization use SGD with momentum 0.9; only the predictor uses AdamW. Zero-gradient and skipped-update ablations must account for optimizer momentum and weight decay.

5. Normalization-induced predictor distribution shift is a hypothesis, not a confirmed cause of training collapse. Test it before changing the architecture.

6. Replayed controller masks are retrospective diagnostic controls, not independently deployable online policies.

7. Preserve the original v3 implementation and experimental outputs unchanged. Independently verify all handoff claims against the source before implementing corrections.

The immediate objective is to establish whether synthetic gradients and error-based switching offer advantages over strong fixed schedules, zero-gradient controls, and early-stopped exact backpropagation at comparable computational cost.