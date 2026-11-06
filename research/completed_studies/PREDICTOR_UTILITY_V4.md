# Predictor utility follow-up

This is the separately authorized implementation following the literature review.
`IMPLEMENTATION_PLAN.md`, all historical source, the existing notebook, and prior
results remain unchanged. The new runner is `predictor_utility_v4.py`.
Implementation does not authorize launching this real-data study.

## Purpose and corrections

The project seeks better accuracy versus total computation or elapsed time through
adaptive allocation of exact backpropagation. Predictor regression accuracy alone
is not the success criterion.

| Review concern | Implemented correction |
|---|---|
| Reused tiny predictor holdout | Exclude every one of the old 512 diagnostic/fitting/held-out training positions; select 512 fresh evaluation positions. |
| Repeated fitting on very little data | Compare nested 256- and 4,096-example fitting sets at 256 updates, reporting 32 and 2 effective passes. |
| Width and context changed together | Compare local32, local64, global32 and global64 under the same examples, objective and update budget. |
| Fitted predictors evaluated only by gradient MSE | Measure their actual copied-state updates with full optimizer state against exact, zero and skip alternatives. |
| Current score assumed to describe future usefulness | Record the pre-fit error on an anchor, execute that exact anchor, then compare the next 1–4 steps on independent copies. |
| Recovery repeats fitting from scratch | Save predictor, AdamW moments, batch order and private RNG every 64 updates. Commit every completed probe separately. |
| Incorrect scope or efficiency conclusions | Reports explicitly identify exploratory development, unequal FLOP budgets, diagnostic overhead and unavailable total-compute savings. |

The old `scheduled` condition is an **ADA-GP-inspired schedule**, not a faithful
ADA-GP reproduction. Its historical identifier and results are preserved. Matching
ratios alone does not match the paper's weight-gradient prediction and update rules.

## Exact proposed diagnostic

- Frozen feedback-source checkpoints: epochs 5 and 16 from seed 0.
- Four candidate architectures × two fitting-set sizes × two checkpoints: 16 fits.
- Saved predictor at both checkpoints: two additional unfitted conditions.
- Reference AdamW, objective, scaling, clipping and classifier normalization remain.
- One fixed augmented view per fitting example; evaluation uses deterministic views.
- Eight disjoint episodes per condition, each with an augmented anchor, four
  augmented future batches and a separate deterministic evaluation batch.
- The anchor executes the original exact step, including head and predictor updates.
  Its score is measured before the predictor fit. Every branch begins afterward
  from the same model, BatchNorm buffers and complete optimizer state.
- Future alternatives are exact/learned/zero/skip trunk updates, with no head
  updates in any branch. The exact branch retains historical joint clipping.
- Positive `zero_minus_learned_ce` or `skip_minus_learned_ce` means learned updates
  reduced evaluation cross-entropy more than that comparator.
- Error/benefit correlation is descriptive within a checkpoint. It supplies no
  confidence interval, online controller claim or accuracy guarantee.

All selected images come from the official training partition's existing 45,000
training positions. They are unseen by the fresh predictor fit, but not necessarily
by the saved classifier or saved predictor. Evaluation images are never passed to
fitting, used for stopping, or used to select a best checkpoint inside the runner.
Once inspected for design decisions, these observations are development data.

The study performs 4,096 predictor-fitting updates and 144 copied episodes with
2,448 copied classifier steps, plus target construction and loss evaluation.
These are bounded diagnostics, not full classifier trajectories. Equal update
budgets do not imply equal FLOPs. At the real boundary shape, the largest CPU
feature/target cache is approximately 1.125 GiB before other model/data overhead.
Those targets are saved once per checkpoint with checksums (about 2.25 GiB total),
so runtime recovery reloads them rather than reconstructing their gradients.
GPU time has not been measured on T4; no runtime or compute savings are promised.

## Colab execution and recovery

Keep the original environment, Drive mounting, dataset and split setup cells.
Use the pinned historical source files that match the saved full checkpoints;
add this runner alongside the original investigation script. No old cells need
replacement. First inspect the proposed budget, without accessing CIFAR:

```python
run_checked(['predictor_utility_v4.py', '--describe'])
```

After the diagnostic is approved, execute one command with the actual paths:

```python
run_checked([
    'predictor_utility_v4.py', '--authorize-utility-study', '--device', 'cuda',
    '--checkpoint-dir', CHECKPOINT_DIR,
    '--data-dir', DATA,
    '--split-file', SPLIT,
    '--out', '/content/drive/MyDrive/phase_d_v4_predictor_utility_seed0_v1',
])
```

`CHECKPOINT_DIR` must contain the original full `checkpoint_000005.pt` and
`checkpoint_000016.pt` files and their `.pt.commit.json` markers. Summary ZIPs
without those tensors are insufficient. Data must already be present: this runner
does not download it or change the saved split.

After a runtime reset, perform the existing environment/mount/data setup and run
the **same command with the same output path**. Completed reports and episodes are
verified and skipped. An interrupted fit resumes its last committed 64-update
boundary, repeating at most the unfinished block. A partial episode is repeated.
The verified frozen-target cache is reused, as is the original dataset split.
Corrupt committed files cause an explicit failure; they are never silently ignored.
Keep the whole output folder, including snapshots and markers, on Drive. Results
can span allocations of the same GPU model only as labeled diagnostics, not clean
runtime measurements. Software, driver version, memory, precision and determinism
settings are bound to the study identity; changed settings require a new study.

## Decisions that still require evidence

This does not implement an arbitrary new controller threshold, change BatchNorm,
compensate head learning rates, reproduce ADA-GP, or launch longer training.
The review did not specify validated replacements for these scientific choices.
A structured loss surrogate remains a candidate requiring its own objective and
cost definition; it is not silently substituted for direct gradient prediction.

Before confirmatory training, select an accuracy-loss margin, count all executed
computation, and specify fresh evaluation seeds and a longer-horizon LR schedule.
Compare against ordinary exact training, earlier exact checkpoints, and competitive
deployable fixed schedules. Donor-derived masks remain retrospective controls.
No conclusion here establishes that the predictor or scheduling policy helps
online training. Favorable diagnostic evidence motivates that next comparison.
