# Training

`kaggle_train_dermascan.ipynb` retrains the EfficientNet-B3 classifier on
HAM10000 using a Kaggle GPU, and exports a CPU-ready artifact bundle that drops
into this repository.

## Running it

1. Upload the notebook to Kaggle (**Create → Notebook → File → Import**).
2. **Settings → Accelerator → GPU** (P100 or T4).
3. **+ Add Data →** search `skin-cancer-mnist-ham10000` and add it.
4. Run all. Roughly 2–4 hours at 30 epochs; lower `CFG["epochs"]` for a smoke run.

Outputs land in `/kaggle/working` and are downloadable from the Output panel.

## What it produces

| File | Purpose |
| :--- | :--- |
| `dermascan_b3.pt` | Self-describing bundle: weights plus architecture, class order, input size, normalization, readout, thresholds, temperature and a weight fingerprint |
| `class_thresholds.json` | Per-class decision thresholds, in the format the backend reads |
| `calibration.json` | Temperature, melanoma alert threshold and readout |
| `evaluation_results.json` | Measured test-set metrics and confusion matrix |
| `confusion_matrix_measured.png` | Measured, from the test split |
| `training_curves.png`, `training_history.csv` | Per-epoch history — **keep these**; the previous run kept no log and its curves could not be reproduced |
| `split_*.csv` | The exact lesion-disjoint splits used |

## Why the notebook is shaped the way it is

Each choice below traces to a defect found in the deployed system.

**The checkpoint describes itself.** A previous deployment served a checkpoint
whose architecture differed from the one the published metrics described — a
20-point accuracy gap that went unnoticed because nothing recorded what had been
evaluated. The bundle now carries `arch`, `head`, `classes`, `img_size`,
`norm_mean/std`, `readout` and a weight fingerprint.

**Validation transforms are identical to serving transforms.** Inference ran at
224 px with a centre crop against a model trained at 300 px. `img_size` travels
inside the checkpoint so the two cannot drift apart silently.

**The readout is recorded.** Thresholds were once fitted on softmax and applied
to sigmoid outputs. The notebook fits thresholds under `CFG["readout"]` and
exports it; the backend reads that field rather than assuming.

**Splits are lesion-disjoint.** HAM10000 holds several images of the same lesion.
Splitting by image leaks a lesion across partitions and inflates every metric.
The notebook groups by `lesion_id`, stratifies by diagnosis, and asserts zero
overlap before training starts.

**The test split is touched once.** Temperature, thresholds and the melanoma
alert cutoff are all fitted on a separate calibration split. Nothing is tuned
against the test numbers.

**Melanoma is weighted explicitly, and class balancing is applied once.**
Measured recall was 0.624, with misses reported as benign. Frequency is handled
by the sampler alone (`sampler_power`, square-root by default); loss weights are
uniform apart from the `mel_loss_weight` multiplier on melanoma. Doing both — a
balanced sampler *and* inverse-frequency loss weights — is a double correction
that collapsed run 1. Melanoma recall is logged every epoch, and
`min_mel_recall` prevents the threshold step from trading it away afterwards.

**Brightness augmentation, then a brightness check.** The deployed model changed
its answer when an image was darkened, which biases against darker skin and poor
lighting. Training jitters brightness and contrast; the notebook then measures
accuracy at several brightness scales before export and prints the spread.

**Only tensors and plain primitives are saved.** A numpy scalar in a checkpoint
previously broke `torch.load(weights_only=True)` on a CPU-only machine. Every
exported value passes through `float()`/`int()` first, and the final cell loads
the bundle back with `weights_only=True` on CPU and runs a forward pass.

## Deploying the result

```bash
cp dermascan_b3.pt         models/latest.pt
cp class_thresholds.json   models/
cp calibration.json        models/
cp evaluation_results.json docs/
```

Set `MODEL_ARCH` in `.env` to match the bundle's `head` (`plain` or `multihead`)
and `IMG_SIZE` to its `img_size`. The readout is picked up from
`calibration.json` automatically.

Then verify:

```bash
python -m pytest
python scripts/evaluate_model.py --data-dir path/to/test_set
```

The evaluator's output should match `evaluation_results.json`. If it does not,
the training and serving paths have diverged — investigate before publishing
either number.

## What this cannot fix

Weighting and thresholds move the operating point along a curve; they do not add
information. If melanoma recall is still short after retraining, the remaining
levers are more melanoma data, higher input resolution, or a dedicated
melanoma-versus-nevus head — not more threshold tuning.

## Run history

### Run 1 — 2 September 2026, Tesla T4, 30 epochs. Rejected.

Completed end to end and produced a model **worse than the deployed one on every
axis**. Nothing was exported; the run is kept because it diagnosed four real
defects in this notebook.

| | deployed | run 1 |
| :--- | ---: | ---: |
| Accuracy | 0.8505 | 0.7552 |
| Macro-F1 | 0.7450 | 0.6416 |
| Melanoma recall | 0.6236 | 0.5706 |
| ECE | 0.0511 | 0.2367 |

Notably, validation melanoma recall reached **0.76–0.78 at arg-max** — better
than the deployed model — and the threshold step then gave it away.

**What went wrong, and what changed:**

1. **Double class balancing.** A fully class-balanced sampler was combined with
   inverse-frequency loss weights, an effective ~1/freq² correction. Validation
   accuracy was 0.08 at epoch 1, when predicting only `nv` scores 0.68.
   *Fix:* `sampler_power` (0.5, square-root balancing) and uniform class weights.

2. **The melanoma multiplier did nothing.** Applied after normalisation, melanoma
   landed at 0.66 — below `akiec` (0.89) and `df` (2.92), so the class it was
   meant to prioritise ranked fifth of seven.
   *Fix:* weights are uniform; only melanoma is boosted.

3. **Temperature and ECE measured on different decision paths.** Temperature was
   fitted on `argmax` while ECE was reported on thresholded predictions, so calib
   ECE 0.037 and test ECE 0.237 were never comparable.
   *Fix:* both fitted on the served decision rule, alternating until they settle.

4. **Threshold fitting traded melanoma away.** Maximising macro-F1 alone pushed
   the melanoma threshold to 0.75 and cut recall from 0.76 to 0.57.
   *Fix:* `min_mel_recall` is a hard constraint, with an explicit warning when it
   cannot be met rather than a silent fallback.

Run 2 also compares against the deployed model and refuses to recommend export on
a regression.

### Run 2 — 3 September 2026, Tesla T4, 30 epochs. Rejected; results uninterpretable.

Training completed cleanly and looked healthy — best epoch 24 at **val accuracy
0.8109, macro-F1 0.6772, melanoma recall 0.6706**, no early stop, ~85s/epoch.
Everything measured *after* training is unusable, and the run diagnosed three
more defects.

| | val, epoch 24 | test, same kernel |
| :--- | ---: | ---: |
| Accuracy | 0.8109 | 0.0532 |
| Macro-F1 | 0.6772 | 0.0518 |
| Melanoma recall | 0.6706 | 0.0059 |

Test arg-max accuracy 0.0532 is *below* the 0.143 that uniform guessing gives,
and 0.0532 × 1503 ≈ 79 = the exact support of `bcc`: the evaluated model was
emitting one near-constant class. Thresholds cannot cause that — arg-max ignores
them — and val and test use the same `eval_tf` and the same loaders, so the fault
is the weights, not the data.

**What went wrong, and what changed:**

5. **The restored checkpoint was never verified.** `model.load_state_dict(
   torch.load(BEST_PATH))` is the only thing between the 0.81 val number and the
   0.05 test number. `/kaggle/working` is repopulated from the previous version
   when a notebook is re-opened, so `_best_weights.pt` from run 1 — the run that
   collapsed to val accuracy 0.08 — was sitting there to be loaded.
   *Fix:* checkpoints are named and stamped with `RUN_ID`, the restore asserts
   the id matches, `model` is rebuilt from scratch, and val macro-F1 is
   re-measured and compared against the logged value. Drift over 0.01 raises.
   A second guard in the test cell raises if arg-max accuracy is below chance.

6. **The melanoma floor made the threshold search degenerate.** Encoded as
   `if recall < floor: return -1`, the objective is *flat* at −1 across every
   infeasible threshold vector, so coordinate ascent had no direction to move in
   and returned its own start point. The recovery path then maximised
   `mel_recall + 0.001·macro_f1`, whose global optimum is "predict melanoma for
   every image": calib recall 1.0000, macro-F1 0.0300, **97.5% of test cases
   flagged for review**. A model that flags everything has said nothing.
   *Fix:* a linear penalty (`macro_f1 − 5·deficit`) that is feasibility-seeking
   where the floor is violated and plain macro-F1 where it is met, from three
   starts. On a synthetic replica this reaches macro-F1 0.7716 at melanoma
   recall 0.7182, against 0.6987/0.5727 for arg-max.

7. **The exported temperature and thresholds were never valid together.** They
   were fitted alternately and the *last* iteration's leftovers were shipped:
   thresholds fitted under the previous temperature, paired with a temperature
   fitted under the previous thresholds. On the synthetic replica the
   temperature oscillated 1.0 → 6.0 → 0.5 and the exported pair scored 0.699 /
   0.573 while a consistent pair from the same data reached 0.781 / 0.700.
   *Fix:* every `(T, thresholds)` pair is scored as a pair, the loop stops when
   the temperature starts cycling, and the best *jointly evaluated* pair wins.

Two smaller ones: the AMP/scheduler warning was real — `scaler.step()` skips the
update when gradients overflow while `scheduler.step()` advanced regardless, so
the schedule ran ahead of the updates; and the `head`/`backbone` learning-rate
split matched the name fragment `"head"`, which swept efficientnet's `conv_head`
(a backbone block) into the head group at 10× its intended rate.

### The incumbent comparison was not a fair fight

`docs/evaluation_results.json` reports accuracy 0.8505 / macro-F1 0.7450 over
n=1525 with **no lesion-disjoint splitting**. HAM10000 holds ~10015 images of
~7470 lesions, so an image-level split routinely puts two photographs of the same
lesion on opposite sides of the train/test line, and the published number is
optimistic by an unknown margin. Every number this notebook produces is
lesion-disjoint. Gating an honest model on an optimistic one is not a decision
procedure, so the deploy decision is now an **absolute release gate**
(`CFG["release_gate"]`: macro-F1 ≥ 0.70, melanoma recall ≥ 0.70, melanoma
surfaced ≥ 0.90, review rate ≤ 0.45, ECE ≤ 0.10). The published numbers are still
printed, labelled as not comparable. To settle it properly, upload
`models/latest.pt` as a Kaggle dataset: the fair-comparison cell re-measures the
incumbent on this split at arg-max.

### Recipe changes for run 3

Run 2 plateaued at val macro-F1 0.677 by epoch 16 while train loss kept falling
to 0.27 — overfitting, with a backbone at `lr` 3e-5 that was barely moving.

| | run 2 | run 3 |
| :--- | ---: | ---: |
| `lr_backbone` / `lr_head` | 3e-5 / 3e-4 | 1e-4 / 5e-4 |
| Mixup | — | α 0.2, p 0.5 |
| Weight EMA | — | decay 0.9995 (evaluated and exported) |
| `drop_rate` | 0.4 | 0.3 |
| `sampler_power` | 0.5 | 0.6 |
| Epochs | 30 | 35 |

Mixup pays for the higher learning rates and improves calibration, which matters
directly because the thresholds are fitted on probabilities. The EMA is what gets
checkpointed and exported: run 2's melanoma recall swung between 0.59 and 0.82 on
consecutive epochs, which makes checkpoint selection a lottery. Selection is
macro-F1 with a linear penalty below the melanoma floor, rather than macro-F1
alone — a hard floor would throw away good epochs over 170 validation melanomas'
worth of noise.

### Run 3 — 3 September 2026, Tesla T4, 31 of 35 epochs (early stop). **Passed the gate; artefacts lost.**

> The session was interactive and was closed without saving a version, so Kaggle
> deleted `/kaggle/working`. The weights are unrecoverable — `TRAINED_FILE/Run3.ipynb`
> keeps the printed metrics but not the 41 MB they describe. **The numbers below
> are real measurements of a checkpoint that no longer exists**, and are kept as
> the target a rerun should land near, not as the description of anything
> deployed. The notebook now refuses to end quietly on this: a final cell prints
> the artefact list with checksums and, in an interactive session, says loudly
> that nothing is saved yet.

Best epoch 21, restored and verified (`drift 0.0000` — the guard from run 2 does
what it was built to do). Measured on the lesion-disjoint test split, n=1503:

| | required | measured | |
| :--- | ---: | ---: | :--- |
| Macro-F1 | ≥ 0.70 | **0.7224** | pass |
| Melanoma recall | ≥ 0.70 | **0.8176** | pass |
| Melanoma surfaced | ≥ 0.90 | **0.9588** | pass |
| Review rate | ≤ 0.45 | **0.3781** | pass |
| ECE | ≤ 0.10 | **0.0404** | pass |

Accuracy is 0.8290 at arg-max and 0.7951 under the served threshold rule; the
3.4-point difference buys melanoma recall 0.6588 → 0.8176, which is the trade
this system exists to make. Brightness spread 0.0233 across ×1.0–×0.35
(acceptable). Exported: `T=1.35`, melanoma alert `p(mel) ≥ 0.05`, fingerprint
`9b08a89ec541b6c8`.

Against run 2, on validation: macro-F1 0.6772 → 0.7120, accuracy 0.8109 →
0.8318. Early stopping fired at epoch 31 with the best at 21, so 35 epochs is
about right and the recipe is no longer the binding constraint.

Per-class F1 on test: `nv` 0.892, `vasc` 0.878, `bcc` 0.758, `df` 0.723, `bkl`
0.619, `akiec` 0.610, `mel` 0.577. Melanoma precision is 0.4455 by design — the
threshold rule deliberately over-calls melanoma — and `bkl` recall (0.5443) is
now the weakest link, mostly lost to `nv` and `mel`.

### The incumbent comparison cannot be repaired

Run 3 ran the fair-comparison cell with `models/latest.pt` uploaded, and the
result invalidates the comparison rather than settling it:

| arg-max, this split | incumbent | run 3 |
| :--- | ---: | ---: |
| Accuracy | 0.9534 | 0.8290 |
| Macro-F1 | 0.9288 | 0.7226 |
| Melanoma recall | 0.8000 | 0.6588 |

The incumbent's own checkpoint records held-out accuracy **0.8637**, and the repo
publishes 0.8505. A model does not score *better* on a stricter split than on its
own held-out data. The explanation is that this split is drawn fresh from all
10015 images while the incumbent was trained on an unrecorded split of the same
corpus — so roughly 85% of what is called "test" here was in that model's
training set, and the table above measures memorisation. Solving
`0.9534 = seen + (1-seen)·0.8637` gives ~66% seen at perfect recall, or ~85% at
the ~0.98 a fitted model typically scores on its own training data.

This cannot be fixed without the incumbent's original split manifest, which was
never recorded. **The candidate must be judged on the release gate** — absolute
criteria on data it has never seen — and the incumbent's numbers treated as
unverifiable. The cell now detects this case and says so instead of printing a
table that looks authoritative.

### Run 3_1 — 3 September 2026, Tesla T4, 29 of 35 epochs (early stop). **Failed the gate.**

The rerun of run 3's recipe. The model reproduced; the decision layer fitted on
top of it did not.

| arg-max, test split | run 3 | run 3_1 |
| :--- | ---: | ---: |
| Accuracy | 0.8290 | 0.8244 |
| Macro-F1 | 0.7226 | 0.7121 |
| Melanoma recall | 0.6588 | 0.6471 |

That is a reproduction, inside the variance quoted above. But after thresholding:
macro-F1 **0.6904** (gate ≥ 0.70) and melanoma surfaced **0.8529** (gate ≥ 0.90),
so the run was rejected. Two causes, both in the fitting that happens *after*
training:

8. **Pair selection was blind to a criterion the gate enforces.** Two pairs were
   fitted: `T=1.00` (macro-F1 0.7219, ECE 0.0468, score 0.6985) and `T=1.30`
   (0.7133, 0.0317, score 0.6974). It took the first by 0.001. But temperature
   also decides whether the alert channel can reach its sensitivity target —
   flatter probabilities lift more melanomas over the alert threshold. At
   `T=1.00` no alert threshold reached 0.90 sensitivity, the search fell back to
   0.50, and melanoma surfaced came out at 0.8529 — *exactly* equal to
   thresholded recall, i.e. the channel contributed nothing. Run 3 chose `T=1.35`
   and surfaced 0.9588.
   *Fix:* pairs are now ranked in tiers — melanoma floor met, then alert target
   reachable, then review rate within the gate — and only then by macro-F1 minus
   half the ECE. A pair that cannot reach the target sensitivity loses to one
   that can, whatever its ECE.

9. **Seven thresholds fitted on 110 melanomas overfit.** Calib macro-F1 0.7219 →
   test 0.6904, a three-point drop. Run 3 got 0.7405 → 0.7224 from the same
   procedure; the difference between the two runs is threshold luck, not model
   quality.
   *Fix:* `CFG["fit_splits"]` — the decision layer is fitted on `calib + val`
   (2537 images, 280 melanomas) rather than `calib` alone. `val` already chose
   the checkpoint, so it is not virgin data and the fitted thresholds carry mild
   optimism; that is the deliberate trade for roughly half the variance. The test
   split is still touched exactly once and is still the number that counts.

### Refitting without retraining

`RESUME_FROM`: upload a `_best_*.pt` from an earlier run as a Kaggle dataset and
the notebook finds it, skips training, and re-fits and re-measures the decision
layer against those weights — about five minutes instead of an hour. The seed
must match the run that produced the checkpoint, or the splits will not line up;
the restore check catches that, which is what it is for.

## Status

Run 3 cleared the gate and its artefacts were lost; run 3_1 reproduced the model
and was rejected by the decision layer fitted on top of it. The recipe is
unchanged again — the two fixes above are both in the post-training fit, which is
where both failures were.

The fastest way to a deployable model is a **resume run**: upload
`NEW_MODEL/_best_20260903-043324.pt` (run 3_1's epoch-19 EMA weights, val
macro-F1 0.7012, verified loadable) as a Kaggle dataset with the same seed, and
the notebook refits and re-measures in about five minutes. If that clears the
gate, the model is deployable without another hour of training. If it does not,
the decision layer is not the whole story and a retrain is the next step.

Expect from any rerun: the splits are deterministic (`RandomState(42)`, same
images in the same four splits), but `cudnn.benchmark`, the weighted sampler and
mixup all draw from unseeded CUDA-side randomness. Arg-max macro-F1 within a
point or two of 0.72 is a reproduction.

Both notebook paths — training from scratch, and resuming from an uploaded
checkpoint — were executed end to end on CPU against a synthetic HAM10000
stand-in. The threshold search was separately validated on synthetic logits
starting *below* the melanoma floor, and the tiered pair selection was checked on
the exact failure run 3_1 hit: a lower-ECE pair whose alert channel could not
reach 0.90 sensitivity is now correctly rejected in favour of one that can.

What is still unmeasured: performance on skin tones outside HAM10000's
distribution (the brightness sweep is a crude proxy, not a substitute for a
Fitzpatrick-labelled set such as DDI), and any comparison against the deployed
model that is not contaminated.
