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

**Melanoma is weighted explicitly.** Measured recall was 0.624, with misses
reported as benign. The loss applies inverse-frequency weights plus an extra
`mel_loss_weight` multiplier, and melanoma recall is logged every epoch so a run
that trades it away for overall accuracy is visible while it happens.

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

## Status

The notebook's logic was validated locally: split generation was run against the
real HAM10000 metadata (zero lesion leakage, all seven classes present in all
four splits), and the loss, metrics, calibration and export functions were
executed against synthetic tensors on CPU, including a `weights_only=True`
round-trip. **The full training run has not been executed** — no GPU was
available — so treat the first Kaggle run as the real test.
