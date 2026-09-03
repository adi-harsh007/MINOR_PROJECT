# Test Samples

Labelled dermoscopic images for exercising the classifier.

**These are from the held-out `test` split of the lesion-disjoint partition the
deployed model was trained under** (`models/splits/split_test.csv`). The model was
not trained on them, so predictions here reflect real generalisation. Regenerate
them whenever the served model changes to a different split — an earlier sample
set became 15/21 *training* images the moment the split was redrawn, while still
claiming to be held out. They were selected by manifest order,
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
| `akiec_1_ISIC_0029659.jpg` | Actinic keratosis / intraepithelial carcinoma (`akiec`) | ISIC_0029659 |
| `akiec_2_ISIC_0025178.jpg` | Actinic keratosis / intraepithelial carcinoma (`akiec`) | ISIC_0025178 |
| `akiec_3_ISIC_0028730.jpg` | Actinic keratosis / intraepithelial carcinoma (`akiec`) | ISIC_0028730 |
| `bcc_1_ISIC_0029230.jpg` | Basal cell carcinoma (`bcc`) | ISIC_0029230 |
| `bcc_2_ISIC_0031513.jpg` | Basal cell carcinoma (`bcc`) | ISIC_0031513 |
| `bcc_3_ISIC_0028238.jpg` | Basal cell carcinoma (`bcc`) | ISIC_0028238 |
| `bkl_1_ISIC_0025915.jpg` | Benign keratosis (`bkl`) | ISIC_0025915 |
| `bkl_2_ISIC_0031029.jpg` | Benign keratosis (`bkl`) | ISIC_0031029 |
| `bkl_3_ISIC_0029836.jpg` | Benign keratosis (`bkl`) | ISIC_0029836 |
| `df_1_ISIC_0029760.jpg` | Dermatofibroma (`df`) | ISIC_0029760 |
| `df_2_ISIC_0030555.jpg` | Dermatofibroma (`df`) | ISIC_0030555 |
| `df_3_ISIC_0030244.jpg` | Dermatofibroma (`df`) | ISIC_0030244 |
| `mel_1_ISIC_0026120.jpg` | Melanoma (`mel`) | ISIC_0026120 |
| `mel_2_ISIC_0028412.jpg` | Melanoma (`mel`) | ISIC_0028412 |
| `mel_3_ISIC_0030443.jpg` | Melanoma (`mel`) | ISIC_0030443 |
| `nv_1_ISIC_0032285.jpg` | Melanocytic nevus (`nv`) | ISIC_0032285 |
| `nv_2_ISIC_0029979.jpg` | Melanocytic nevus (`nv`) | ISIC_0029979 |
| `nv_3_ISIC_0029961.jpg` | Melanocytic nevus (`nv`) | ISIC_0029961 |
| `vasc_1_ISIC_0029486.jpg` | Vascular lesion (`vasc`) | ISIC_0029486 |
| `vasc_2_ISIC_0031901.jpg` | Vascular lesion (`vasc`) | ISIC_0031901 |
| `vasc_3_ISIC_0029404.jpg` | Vascular lesion (`vasc`) | ISIC_0029404 |

## Other files

| File | Purpose |
| :--- | :--- |
| `cat.jpg` | Non-skin control. Should be **rejected** by the OOD gate, not classified. |
| `nv.jpg`, `ISIC_0024307.jpg` | Earlier reference images kept because the sample gallery links to them. |

## Regenerating

```bash
python scripts/build_test_samples.py --per-class 3
```

Re-running replaces the class-prefixed files and leaves everything else alone.
