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

## Status

Logic validated locally without a GPU: splits were generated against the real
HAM10000 metadata (zero lesion leakage, all seven classes in all four splits),
and the loss, metrics, threshold fitting, calibration and export functions were
executed against synthetic tensors on CPU — including a `weights_only=True`
round-trip and a check that the melanoma-recall floor actually binds.

**The corrected recipe has not itself been run on a GPU.** Run 1's numbers are
real; run 2's are not yet. Treat the next Kaggle run as the test of these fixes,
and compare it against the incumbent table the notebook now prints.
