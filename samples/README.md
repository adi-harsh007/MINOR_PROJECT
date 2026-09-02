# Test Samples

Labelled dermoscopic images for exercising the classifier.

**These are from the held-out `test` split.** The model was not trained on them, so
predictions here reflect real generalisation. They were selected by manifest order,
**not** by whether the model classifies them correctly — a set curated on model
success would look better than the model actually is.

## Naming

`<class>_<n>_<ISIC id>.jpg` — the prefix is the ground-truth diagnosis and the
ISIC id identifies the exact dataset row.

| Code | Diagnosis | Risk | Files |
| :--- | :--- | :--- | ---: |
| `akiec` | Actinic keratosis / intraepithelial carcinoma | Pre-malignant | 3 |
| `bcc` | Basal cell carcinoma | Malignant | 3 |
| `bkl` | Benign keratosis | Benign | 3 |
| `df` | Dermatofibroma | Benign | 3 |
| `mel` | Melanoma | Malignant | 3 |
| `nv` | Melanocytic nevus | Benign | 3 |
| `vasc` | Vascular lesion | Benign | 3 |

## Files

| File | Ground truth | ISIC id |
| :--- | :--- | :--- |
| `akiec_1_ISIC_0024450.jpg` | Actinic keratosis / intraepithelial carcinoma (`akiec`) | ISIC_0024450 |
| `akiec_2_ISIC_0024843.jpg` | Actinic keratosis / intraepithelial carcinoma (`akiec`) | ISIC_0024843 |
| `akiec_3_ISIC_0024923.jpg` | Actinic keratosis / intraepithelial carcinoma (`akiec`) | ISIC_0024923 |
| `bcc_1_ISIC_0024332.jpg` | Basal cell carcinoma (`bcc`) | ISIC_0024332 |
| `bcc_2_ISIC_0024454.jpg` | Basal cell carcinoma (`bcc`) | ISIC_0024454 |
| `bcc_3_ISIC_0024573.jpg` | Basal cell carcinoma (`bcc`) | ISIC_0024573 |
| `bkl_1_ISIC_0024336.jpg` | Benign keratosis (`bkl`) | ISIC_0024336 |
| `bkl_2_ISIC_0024408.jpg` | Benign keratosis (`bkl`) | ISIC_0024408 |
| `bkl_3_ISIC_0024409.jpg` | Benign keratosis (`bkl`) | ISIC_0024409 |
| `df_1_ISIC_0024386.jpg` | Dermatofibroma (`df`) | ISIC_0024386 |
| `df_2_ISIC_0024973.jpg` | Dermatofibroma (`df`) | ISIC_0024973 |
| `df_3_ISIC_0025622.jpg` | Dermatofibroma (`df`) | ISIC_0025622 |
| `mel_1_ISIC_0024310.jpg` | Melanoma (`mel`) | ISIC_0024310 |
| `mel_2_ISIC_0024313.jpg` | Melanoma (`mel`) | ISIC_0024313 |
| `mel_3_ISIC_0024496.jpg` | Melanoma (`mel`) | ISIC_0024496 |
| `nv_1_ISIC_0024307.jpg` | Melanocytic nevus (`nv`) | ISIC_0024307 |
| `nv_2_ISIC_0024308.jpg` | Melanocytic nevus (`nv`) | ISIC_0024308 |
| `nv_3_ISIC_0024325.jpg` | Melanocytic nevus (`nv`) | ISIC_0024325 |
| `vasc_1_ISIC_0024669.jpg` | Vascular lesion (`vasc`) | ISIC_0024669 |
| `vasc_2_ISIC_0024904.jpg` | Vascular lesion (`vasc`) | ISIC_0024904 |
| `vasc_3_ISIC_0025250.jpg` | Vascular lesion (`vasc`) | ISIC_0025250 |

## Other files

| File | Purpose |
| :--- | :--- |
| `cat.jpg` | Non-skin control. Should be **rejected** by the OOD gate, not classified. |
| `nv.jpg`, `ISIC_0024307.jpg` | Earlier reference images kept because the sample gallery links to them. |

## What to expect

Snapshot taken 2 September 2026 with the shipped configuration (`models/latest.pt`,
300x300, sigmoid readout, per-class thresholds). **17 of 21 correct** — in line with the
0.8505 accuracy measured over the full test split.

| Ground truth | Correct | Notes |
| :--- | :--- | :--- |
| `akiec` | 3/3 | |
| `bcc` | 2/3 | one read as `bkl` |
| `bkl` | 3/3 | |
| `df` | 2/3 | one read as `nv` |
| `mel` | **1/3** | two read as `nv` at 0.969 and 0.984 confidence |
| `nv` | 3/3 | |
| `vasc` | 3/3 | |

The melanoma column is the one that matters. Two of three melanomas were missed, and
both were called benign nevus with very high confidence — the model is **confidently
wrong**, not uncertain, so a low confidence score cannot be used as a safety signal.
This matches the measured melanoma recall of 0.624 across the full test set.

`cat.jpg` should be rejected outright rather than classified. Note that until the
feature-space OOD stage is fitted (`scripts/calibrate_ood.py`), the colour gate alone
may accept some non-clinical photographs.

## Regenerating

```bash
python scripts/build_test_samples.py --per-class 3
```

Re-running replaces the class-prefixed files and leaves everything else alone.
