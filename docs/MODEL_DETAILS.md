# Skin Cancer Classification Model Details

This document provides technical specifications and performance metrics for the skin cancer diagnostic model utilized in this project.

## Clinical Context: ABCDE Rules

Before specialized neural analysis, clinicians often use the **ABCDE criteria** for the visual assessment of pigmented lesions, particularly for identifying potential melanoma.

| Rule | Aspect | Description |
| :--- | :--- | :--- |
| **A** | Asymmetry | One half of the mole does not match the other. |
| **B** | Border | Edges are irregular, ragged, notched, or blurred. |
| **C** | Color | Pigmentation is not uniform across the lesion. |
| **D** | Diameter | Usually greater than 6mm (approx. size of a pencil eraser). |
| **E** | Evolving | The mole is changing in size, shape, or color. |

## Model Architecture

- **Base Model:** EfficientNet-B3 (via `timm` library)
- **Input Resolution:** 300x300 pixels
- **State:** Evaluation Mode (`eval()`) with pre-trained weights loaded from `models/latest.pt`.

## Data Processing & Preprocessing

The model uses a standardized ImageNet-based preprocessing pipeline:
1. **Resize:** Input images are resized directly to 300x300 (no centre crop),
   matching `img_size: 300` and `get_val_transforms` in the training repository.
2. **Normalization:**
   - Mean: `[0.485, 0.456, 0.406]`
   - Std Dev: `[0.229, 0.224, 0.225]`

## Classification Categories

The model is trained on the HAM10000 dataset to recognize the following 7 morphologies.

![Real HAM10000 examples, one per class](./class_samples_real.png)
*One real image per class from the HAM10000 test split, each labelled with its
ISIC id and ground-truth diagnosis so any panel can be traced to its dataset row.
Regenerate with `python scripts/build_class_samples.py`.*

| Label | Full Name | Description |
| :--- | :--- | :--- |
| **akiec** | Actinic Keratosis | Also includes Bowen's disease / intraepithelial carcinoma. |
| **bcc** | Basal Cell Carcinoma | A common form of skin cancer. |
| **bkl** | Benign Keratosis | Includes seborrheic keratoses and lichen-planus like keratoses. |
| **df** | Dermatofibroma | A benign fibrous skin lesion. |
| **mel** | Melanoma | The most serious type of skin cancer. |
| **nv** | Melanocytic Nevi | Common moles (benign). |
| **vasc** | Vascular Lesions | Includes angiomas, hemorrhage, and pyogenic granulomas. |

## Performance Metrics

Metrics are based on optimized classification thresholds to maximize the Macro F1-Score.

### Decision-layer fit (calib + val, n=2537)

The thresholds, temperature and melanoma alert cutoff in `models/` were fitted
together on the merged calibration and validation splits — 2537 images, 280
melanomas — and reported on a test split that was touched exactly once.

Fitting on `calib` alone (1009 images, 110 melanomas) was not enough to pin seven
thresholds: one run read calibration macro-F1 0.7219 and delivered 0.6904 on
test. `val` had already chosen the checkpoint, so these thresholds carry mild
optimism; that is the deliberate trade for roughly half the variance. See
[training/README.md](../training/README.md).

### Measured Test-Set Performance

Measured by running the served checkpoint over a **lesion-disjoint** held-out
split of HAM10000 (1503 images) — not reconstructed from summary statistics.

- **Served model:** `models/latest.pt` (plain timm EfficientNet-B3, `MODEL_ARCH=plain`),
  run `20260903-043324` epoch 19, EMA weights, fingerprint `27629fd06174b38f`
- **Preprocessing:** `Resize(300, 300)`, no centre crop — matches `img_size: 300`
  in the bundle
- **Readout:** softmax at temperature 1.15, decision rule
  `argmax(probability - class threshold)`
- **Accuracy:** 0.8090  **Macro-F1:** 0.7123  **ECE:** 0.0337

| Class | Support | Precision | Recall | F1 |
| :--- | ---: | ---: | ---: | ---: |
| **akiec** | 52 | 0.5312 | 0.6538 | 0.5862 |
| **bcc** | 79 | 0.8571 | 0.6835 | 0.7606 |
| **bkl** | 158 | 0.7073 | 0.5506 | 0.6192 |
| **df** | 22 | 0.6364 | 0.6364 | 0.6364 |
| **mel** | 170 | 0.4806 | 0.8000 | 0.6004 |
| **nv** | 1000 | 0.9397 | 0.8730 | 0.9051 |
| **vasc** | 22 | 0.9474 | 0.8182 | 0.8780 |

![Measured confusion matrix](./confusion_matrix_measured.png)
*Measured confusion matrix, n=1503. Counts in parentheses, row-normalised shading.*

Raw numbers: `docs/evaluation_results.json`. Reproduce with:

```bash
python scripts/evaluate_model.py --data-dir path/to/test_set
```

#### Serving reproduces training, to within fp16

The same checkpoint scored through the serving path (`backend/ml_engine.py`, CPU,
float32) rather than the training notebook (CUDA, autocast float16):

| | notebook | serving |
| :--- | ---: | ---: |
| Accuracy | 0.8090 | 0.8097 |
| Macro-F1 | 0.7123 | 0.7130 |
| Melanoma recall | 0.8000 | 0.8000 |

Five of seven classes are identical to four decimal places; `bkl` and `nv` each
move by one borderline image, which is what mixed-precision logits do to a
decision rule comparing probabilities against fixed thresholds. Full output:
`docs/evaluation_serving_check.json`.

#### Melanoma performance

Melanoma recall is **0.800** under the served rule, against 0.6471 at plain
arg-max: the thresholds are what buy that, and the cost is melanoma precision of
0.4806. The alert channel then surfaces **92.9%** of melanomas — flagging any scan
whose `p(mel)` clears 0.10 regardless of which class wins — at a **23.6%** review
rate.

The system is deliberately tuned to over-call melanoma. A false alarm costs a
review; a miss can cost a life. That is a clinical policy choice and should be
signed off as one, not inherited from a threshold search.

#### Historical: the previous served model

The figures below describe the checkpoint served before September 2026, measured
on an **image-level** split (n=1525). HAM10000 holds ~10015 images of ~7470
lesions, so that split places other photographs of the same lesion on both sides
of the train/test line, and every figure is optimistic by an unknown margin. They
are kept for provenance and are **not comparable** to the numbers above.

| Config (historical, image-level split) | Accuracy | Macro-F1 | Melanoma recall | Melanoma F1 |
| :--- | ---: | ---: | ---: | ---: |
| 300px softmax + margin | 0.8616 | 0.7668 | 0.5393 | 0.6316 |
| 300px sigmoid + margin (was shipped) | 0.8505 | 0.7450 | 0.6236 | 0.6361 |
| 224px + crop, sigmoid + margin | 0.8066 | 0.7116 | 0.6798 | 0.5641 |

When that model was re-scored on the current lesion-disjoint split it returned
accuracy 0.9534 and macro-F1 0.9288 — *higher* than its own held-out numbers,
which is only possible because it had trained on most of those images. There is
no way to compare the two models fairly without its original split manifest,
which was never recorded.

#### Historical: melanoma threshold sweep

Measured on the **previous** model and the image-level split, kept because the
shape of the tradeoff still holds. Sweeping the melanoma threshold alone (all
others at the macro-F1 fit), selected on calibration and reported on test:

| `mel` threshold | Accuracy | Melanoma recall | Melanoma F1 |
| ---: | ---: | ---: | ---: |
| 0.20 | 0.8256 | 0.7416 | 0.5986 |
| 0.25 | 0.8348 | 0.7135 | 0.6135 |
| 0.30 | 0.8393 | 0.6798 | 0.6189 |
| 0.35 | 0.8446 | 0.6404 | 0.6230 |
| 0.40 | 0.8531 | 0.6180 | 0.6377 |
| 0.45 | 0.8584 | 0.5843 | 0.6480 |
| 0.50 | 0.8557 | 0.5000 | 0.6075 |

Lowering the melanoma threshold buys recall at roughly **1 point of accuracy per
3-4 points of melanoma recall**. Choosing a different operating point is a change
to the `mel` entry in `models/class_thresholds.json` — a clinical decision about
the relative cost of a missed melanoma versus a false alarm, not a tuning detail.

Re-fit with:

```bash
python scripts/optimize_thresholds.py     --calib-dir path/to/calib_set --test-dir path/to/test_set     --readout sigmoid --objective mel_recall
```

It fits on the calibration split, reports on the test split, writes only with
`--write`, and refuses to write thresholds that cost more than 2 points of test
macro-F1.

#### On training curves

Per-epoch curves for the **previous** checkpoint are not available: it stored only
a single epoch's metrics and kept no history log. Figures once shown here were
generated from hardcoded exponentials rather than measured, and were removed.

The current notebook logs per-epoch loss, accuracy, macro-F1 and melanoma recall
to `training_history.csv` and plots them, so this gap does not recur. The deployed
checkpoint came from a resumed run that refitted only the decision layer, so its
curves belong to the run that produced the weights rather than the one that
produced the thresholds.

#### A note on checkpoints

`skin_cancer/checkpoints/best.pt` uses a different architecture (two-layer head,
`MODEL_ARCH=multihead`) and is an **earlier, weaker** run: measured accuracy
0.6570 / macro-F1 0.4464 on the same test set. It is not the served model. The
architecture is selected explicitly by `MODEL_ARCH` and loaded strictly, so
swapping checkpoints fails loudly rather than silently changing the network.

## Confidence calibration and the melanoma alert channel

Two distinct problems: the model **misses melanomas**, and it does so
**confidently**. They have different fixes, and neither requires retraining. The
analysis below was carried out on the previous checkpoint, where arg-max melanoma
recall was 0.624 and mean stated confidence on wrong answers was 0.942. The
mechanism is unchanged, and the current model uses the same two devices — now
fitted as a single pair.

### Why the alert channel works

On the melanomas the model misses, it still assigns a substantial melanoma
probability — median p(mel) = 0.539 across the 67 missed cases. The recall ceiling is
imposed by the **argmax**, not by what the model knows: `nv` simply edges ahead.
Flagging on p(mel) directly, independently of which class wins, recovers most of them.

### Fitted values — current model

`temperature = 1.15`, `mel_alert_threshold = 0.10` in `models/calibration.json`,
fitted on calib+val and reported on the lesion-disjoint test split (n=1503):

| Measure | Value |
| :--- | ---: |
| Expected calibration error | 0.0337 |
| Melanoma recall (served rule) | 0.8000 |
| **Melanoma surfaced (prediction or alert)** | **0.9294** |
| Cases flagged for review | 0.2356 |

Temperature and thresholds are fitted **as a pair** and scored together, because
they interact: temperature decides whether the alert channel can reach its
sensitivity target at all, since flatter probabilities lift more melanomas over
the alert cutoff. Selecting temperature on calibration error alone once produced a
model whose alert channel could not reach 90% sensitivity at any threshold,
leaving it inert — melanoma surfaced came out exactly equal to thresholded recall.

### Fitted values — previous model, historical

Temperature was chosen on the calibration split (n=997) by minimising expected
calibration error; the alert threshold was the lowest review rate reaching 90%
melanoma sensitivity there. Both reported on the image-level test split (n=1525).

| Measure | Before | After |
| :--- | ---: | ---: |
| Accuracy | 0.8505 | **0.8544** |
| Expected calibration error | 0.1162 | **0.0511** |
| Mean confidence when wrong | 0.9418 | **0.8141** |
| Melanoma recall (argmax) | 0.6236 | 0.6236 |
| **Melanoma surfaced (argmax or alert)** | 0.8764 | **0.8989** |
| Cases flagged for review | 0.2420 | 0.3051 |

`temperature = 2.0`, `mel_alert_threshold = 0.45` — the values that model shipped with.

Temperature scaling is not a trade here: accuracy rises slightly and melanoma recall is
unchanged, because dividing the logits leaves most of the ordering intact. What changes
is that stated confidence stops being uniformly ~0.97.

The alert channel is a genuine trade: **90% of melanomas surfaced against a
31% review rate**. It does not alter the primary prediction — it is an
additional output, shown in the UI as "melanoma not excluded".

Re-fit with:

```bash
python scripts/fit_calibration.py --calib-dir <calib> --test-dir <test> --write
```

`--target-sensitivity` moves the operating point; higher sensitivity means more cases
flagged. With no calibration file present, confidences are raw and the alert is off.

### What this does not do

Neither device improves what the model actually knows. Thresholds and the alert
channel move the operating point along a fixed curve and raise the review burden;
they do not add discrimination. On the current model, arg-max melanoma recall is
0.6471 and the served rule lifts it to 0.8000 — bought entirely with melanoma
precision, which sits at 0.4806.

Real improvement needs the model: more melanoma data, higher input resolution, an
ensemble, or a dedicated melanoma-versus-nevus head. The retraining that produced
the current checkpoint already applies melanoma-weighted focal loss, so that lever
is spent.

## Out-of-Distribution (OOD) Gatekeeper

Three stages guard inference.

### Stage 1: Illumination-invariant image statistics

Every metric is a **ratio**, so scaling an image's brightness leaves it unchanged.

| Metric | Rule | Rejects |
| :--- | :--- | :--- |
| `rel_contrast` (luminance std / mean) | `< 0.04` | Flat colour fields |
| `hf_ratio` (high-frequency residual / total) | `> 0.45` | Pixel noise |
| `blue_green` (over chromatic pixels only) | `> 0.60` | Sky, foliage, surgical drape |

Hue checks apply only when at least 25% of pixels carry a numerically meaningful
hue. Grayscale images fall below that and **skip** the hue checks rather than
being rejected.

#### Why this replaced the previous rules

The previous gate used absolute channel standard deviation (`avg_std`), which
scales with brightness. Identical lesions rejected or accepted purely on how dark
the image was — a direct bias against darker skin tones and underexposed
captures. Measured on a HAM10000 image, `ISIC_0024307`
darkened to 35% brightness was rejected as `too_uniform` while the identical
lighter image passed. `rel_contrast` is constant across the same range
(0.119 → 0.120).

Two further rules were removed: `grayscale_not_allowed`, which excluded
legitimate grayscale dermoscopy, and `avg_std > 65`, which rejected
high-contrast dermoscopic images — a dark lesion on pale skin, i.e. the
presentation of most concern.

On a 21-case check of skin images (including darkened, grayscale and
contrast-boosted variants) plus non-skin controls, the old gate produced 6
incorrect verdicts and the new gate 1.

### Stage 2: Feature-space Mahalanobis distance

Colour statistics cannot reject a photograph of another real object: a
desaturated animal photo has statistics inside the dermoscopic range. Semantic
rejection requires the classifier's own feature space.

This stage is **not active until fitted**. Run:

```bash
python scripts/calibrate_ood.py --data-dir path/to/dermoscopic_images
```

Until then the system runs on Stage 1 alone and will accept some non-clinical
photographs. Calibrate on data spanning the full range of skin tones you intend
to serve — a gate fitted only to light skin reintroduces the bias removed above.

#### How much data this stage needs

**It cannot be fitted from the 21 images in `samples/`, and attempting it
produces a gate that rejects every scan.** This was measured, not assumed.

EfficientNet-B3 emits 1536-dimensional features, so the covariance has
1,180,416 free parameters. Estimated from 21 samples it is rank-deficient; the
pseudo-inverse then explodes along the ~1515 directions the data never
constrained. Under leave-one-out — fit on 20, judge the held-out lesion —
**21 of 21 held-out lesions were rejected**, at Mahalanobis distances around
10⁶ against a cutoff of 18.

The same run reported "OOD correctly rejected: 3/3 (100.0%)". That number is
worthless on its own: a gate that rejects *everything* rejects out-of-
distribution input perfectly. The script also reported "in-distribution wrongly
rejected: 4.8%", because it measured false rejects on the very images it had
just fitted to, where the cutoff is by construction the 99th percentile of those
distances.

`scripts/calibrate_ood.py` now holds a fraction of `--data-dir` back from
fitting, reports the false-reject rate on that held-out portion, and **refuses
to write** a gate that rejects more than `--max-false-reject` of it (default
5%), or that cannot be validated at all. Override with `--force` only with a
reason.

Fitting this stage therefore requires the HAM10000 image set — the manifests in
`models/splits/` reference `/kaggle/input/...` paths and the images are not in
this repository. With the 5,975-image training split available, run:

```bash
python scripts/calibrate_ood.py --data-dir path/to/ham10000_images \
    --ood-dir path/to/non_skin_images
```

and read the held-out false-reject rate before trusting the result.

### Stage 3: Confidence margin

`argmax(probability - class threshold)`; a negative best margin is rejected as
`low_confidence`.

> **Do not add max-softmax or energy-based OOD to this checkpoint.** Both were
> measured and are anti-correlated: a blank white field scores max-softmax
> **0.994** and a *more* in-distribution energy (-3.87) than any real lesion
> image (-2.23 to -3.20). The feature-space stage rejects that same white field
> by a wide margin.

---
*Last Updated: 2026-03-27*
