# Served model

Everything the running server loads lives in this directory. If you want to know
what is deployed, this file is the answer.

## What is in use

| | |
| :--- | :--- |
| **Checkpoint** | `models/latest.pt` — 41.3 MB |
| **Architecture** | timm `efficientnet_b3`, plain head, 10.79 M parameters, 7 classes |
| **Weights** | EMA, epoch 19 of training run `20260903-043324` |
| **Fingerprint** | `27629fd06174b38f` (sha256 of the sorted weight tensors) |
| **Input** | 300×300, `Resize` with no centre crop, ImageNet normalization |
| **Readout** | softmax at temperature 1.15 |
| **Decision rule** | `argmax(probability − class threshold)` |
| **Melanoma alert** | flags any scan with `p(mel) ≥ 0.10`, whatever class wins |
| **Deployed** | 3 September 2026 |

The checkpoint is self-describing: `arch`, `head`, `classes`, `img_size`,
`norm_mean/std`, `readout`, `temperature`, `thresholds` and the fingerprint are
stored *inside* it, so serving verifies rather than assumes. A previous
deployment served a checkpoint whose architecture differed from the one its
published metrics described, and nothing recorded the difference.

## Measured performance

Lesion-disjoint held-out split, 1503 images, touched exactly once:

| Metric | Value | Release gate |
| :--- | ---: | :--- |
| Accuracy | 0.8090 | — |
| Macro-F1 | 0.7123 | ≥ 0.70 ✓ |
| Melanoma recall | 0.8000 | ≥ 0.70 ✓ |
| Melanoma surfaced (prediction or alert) | 0.9294 | ≥ 0.90 ✓ |
| Review rate | 0.2356 | ≤ 0.45 ✓ |
| ECE | 0.0337 | ≤ 0.10 ✓ |

Per-class figures, the confusion matrix and the serving-side cross-check are in
[docs/MODEL_DETAILS.md](../docs/MODEL_DETAILS.md). Raw numbers:
`docs/evaluation_results.json`.

**These are not comparable to this project's pre-September-2026 figures**
(accuracy 0.8505, macro-F1 0.7450, melanoma recall 0.624). Those were measured on
an image-level split of a dataset with ~10015 images of ~7470 lesions, which
leaks other photographs of the same lesion into the test set.

## Files

| File | Purpose | Tracked |
| :--- | :--- | :--- |
| `latest.pt` | The served checkpoint | no (gitignored — 41 MB) |
| `class_thresholds.json` | Per-class decision thresholds, in the shape `backend/ml_engine.py` reads | yes |
| `calibration.json` | Temperature, melanoma alert threshold, readout | yes |
| `splits/split_*.csv` | The exact lesion-disjoint partition this model was trained under | yes |
| `splits/train_config.json` | The training configuration that produced it | yes |
| `checkpoints/` | Raw EMA weights for resuming a run without retraining | no |
| `backup/` | Previous deployments, for rollback | no |

`splits/` is what makes the evaluation reproducible and lets `samples/` be
regenerated from genuinely held-out images. Do not delete it: without the split
manifest there is no way to tell training images from test images later, which is
exactly the position the previous model left this project in.

## Replacing the model

Never copy files in by hand. Use the installer, which refuses to deploy anything
it cannot verify:

```bash
python scripts/deploy_checkpoint.py --from-dir ./kaggle_run --dry-run
python scripts/deploy_checkpoint.py --from-dir ./kaggle_run
```

It re-fingerprints the weights, loads them strictly into the architecture
`backend/model.py` will build, runs a CPU forward pass, checks `IMG_SIZE` and
`MODEL_ARCH`, confirms the sidecars agree with the bundle, and re-checks the
release gate. Every install backs up what it replaces.

## Rolling back

```bash
python scripts/deploy_checkpoint.py --rollback            # list backup sets
python scripts/deploy_checkpoint.py --rollback 20260903-112456
```

| Stamp | What it is |
| :--- | :--- |
| `20260903-112456` | The state immediately before the current model was installed |
| `20260317-000000` | The original March 2026 checkpoint, with its own thresholds and calibration |

Both currently hold the same weights — the March checkpoint was still live when
the current model was installed — and share one copy on disk.

Restart the server after any install or rollback.

## Retraining

See [training/README.md](../training/README.md). To refit only the thresholds and
calibration without retraining, upload `checkpoints/_best_20260903-043324.pt` to
Kaggle as a dataset: the notebook finds it, skips training, and re-measures in
about five minutes.
