# DermaScan AI 4.0 — Clinical Skin Cancer Diagnostic Studio

DermaScan AI is a **research prototype** for skin lesion classification, built on an
**EfficientNet-B3** network trained on HAM10000. It predicts across 7 lesion classes and
provides Grad-CAM attribution, out-of-distribution rejection, and a local history of past
scans.

It is **not a diagnostic device and not a substitute for examination by a qualified
clinician.** Measured melanoma recall is 0.800 — one melanoma in five is missed by the
prediction itself, and the alert channel surfaces 92.9% of them for review. See
[Measured performance](#measured-performance).

**Deployed model:** EfficientNet-B3, run `20260903-043324`, fingerprint
`27629fd06174b38f` — accuracy 0.8090, macro-F1 0.7123, melanoma recall 0.800 on a
lesion-disjoint held-out split. [models/README.md](models/README.md) is the single
source of truth for what is running, how to replace it, and how to roll back.

---

## 📁 Project Architecture & Directory Structure

```text
MODEL_Skin-Cancer/
├── backend/
│   ├── main.py                 # App entrypoint, CORS policy, static mounts
│   ├── config.py               # Paths, upload limits, security configuration
│   ├── model.py                # Network architectures (plain / multihead)
│   ├── ml_engine.py            # Predictor: preprocessing, readout, Grad-CAM
│   ├── ood.py                  # Out-of-distribution gate
│   ├── database.py             # SQLAlchemy engine and sessions
│   ├── models.py               # ORM schema (diagnostic_sessions)
│   └── routers/
│       └── diagnostics.py      # /api/analyze, history, delete endpoints
├── frontend/
│   ├── index.html              # UI layout, viewport, diagnostic views
│   └── js/
│       └── app.js              # View router, heatmap renderer, API client
├── models/
│   ├── README.md               # WHAT IS DEPLOYED — start here
│   ├── latest.pt               # Served EfficientNet-B3 bundle (41 MB, gitignored)
│   ├── class_thresholds.json   # Per-class decision thresholds
│   ├── calibration.json        # Temperature, melanoma alert threshold, readout
│   ├── splits/                 # The lesion-disjoint partition this model was trained under
│   ├── checkpoints/            # Raw weights for resuming a run (gitignored)
│   └── backup/                 # Previous deployments, for rollback (gitignored)
├── data/                       # gitignored
│   ├── pathology.db            # SQLite diagnostic history
│   └── uploads/                # Stored lesion images
├── docs/
│   ├── MODEL_DETAILS.md        # Measured performance, thresholds, OOD logic
│   ├── PROJECT_ARCHITECTURE.md # Stack, request flow, API surface
│   ├── FEATURE_STATUS.md       # What the UI actually implements
│   ├── evaluation_results.json # Raw measured test-set metrics
│   ├── evaluation_serving_check.json  # The same model scored through the serving path
│   ├── confusion_matrix_measured.png
│   ├── class_samples_real.png  # Real HAM10000 example per class
│   └── DermaScan_Serving_Audit.pdf
├── samples/                    # 21 labelled test images (3/class) + a non-skin control
├── scripts/
│   ├── deploy_checkpoint.py    # Verify and install a training bundle; rollback
│   ├── evaluate_model.py       # Measure performance on a labelled hold-out set
│   ├── build_test_samples.py   # Refresh samples/ from a split manifest
│   ├── optimize_thresholds.py  # Fit thresholds to the served configuration
│   ├── calibrate_ood.py        # Fit the OOD gate to real data
│   ├── build_class_samples.py  # Build the class figure from real images
│   └── build_audit_pdf.py      # Build the audit PDF
├── training/
│   ├── kaggle_train_dermascan.ipynb  # GPU retraining -> CPU-ready bundle
│   ├── README.md               # The recipe, and the run history behind it
│   └── runs/                   # Executed notebooks kept as run records
├── tests/                      # pytest suite (runs against a temp database)
├── LICENSE                     # MIT (code) + third-party data terms
├── pytest.ini
├── requirements.txt
└── start.py                    # Single-command startup
```

---

## 🚀 How to Start Frontend & Backend

### **Method 1: Unified Single Command (Recommended)**
This launches the FastAPI backend and serves the frontend SPA interface at `http://localhost:8088`.

```bash
python start.py
```
- **Web Console (Frontend + App)**: [http://localhost:8088](http://localhost:8088)
- **Interactive OpenAPI Docs**: [http://localhost:8088/docs](http://localhost:8088/docs)

---

### **Method 2: Running Uvicorn directly**

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8088 --reload
```

The frontend is served by this same process at `/`. It calls the API at the relative
path `/api`, so it cannot be hosted on a separate port without changing `API_BASE` in
`frontend/js/app.js` or putting a proxy in front of it.

---

## ⚙️ How Frontend & Backend Work Together

1. **User Upload & Configuration**:
   - The user selects or drags a dermatoscopic lesion scan into the Scan Console (`#console-viewport-img`).
   - Anatomic site metadata (e.g., *Anterior Torso*, *Head/Neck*) is tagged.

2. **API Inference Request**:
   - Clicking **"Begin AI Analysis"** sends a multipart `POST` to `/api/analyze` with the
     image binary and a `site` field. The site is validated against the six values the
     UI offers and stored on the record; it does not affect inference.

3. **Backend ML Pipeline Execution**:
   - **OOD Gatekeeper**: `backend/ood.py` applies illumination-invariant image statistics (relative contrast, high-frequency ratio, chromatic hue), then a feature-space check once fitted.
   - **EfficientNet-B3 Pass**: Image is resized to 300x300 (matching training) and normalized. Forward pass computes logits for 7 pathology classes (MEL, NV, BCC, AKIEC, BKL, DF, VASC).
   - **Grad-CAM Activation Engine**: Backward pass on `conv_head` layer computes gradient-weighted feature maps, rendering a Jet colormap (Blue -> Red) heatmap.
   - **Database Log**: SQLAlchemy writes the prediction, confidence, threshold, all
     seven scores, the anatomic site and a high-risk flag to `data/pathology.db`.

4. **Frontend Response & Visualization**:
   - The API returns `session_id`, `prediction`, `confidence`, `threshold`, the seven
     class `scores`, an `is_high_risk` boolean, `anatomic_site`, and `heatmap_base64`
     (which is `null` when
     attribution could not be computed). Risk wording and any ABCDE guidance shown in the
     UI are produced client-side from the predicted class — they are not API fields.
   - `app.js` updates view to **Diagnostic Results**, draws the Grad-CAM heatmap onto `<canvas id="console-heatmap-canvas">`, and synchronizes zoom/pan transforms.

---

## 📖 Component Index

Verified against the source. Anything not listed here does not exist.

### Backend (`backend/`)

#### `main.py`
- `serve_index()` — serves `frontend/index.html`.
- `favicon()` — returns 204 No Content.
- `lifespan()` — creates database tables on startup.
- `health_check()` — reports the real serving configuration (architecture, checkpoint
  name and presence, input size, whether the model has been loaded). It does not force
  a model load.

#### `ml_engine.py` — `SkinCancerPredictor`
- `__init__()` — builds the architecture named by `MODEL_ARCH`, loads the checkpoint
  strictly (failure is fatal), loads per-class thresholds, registers Grad-CAM hooks on
  `conv_head`, and sets up preprocessing (300x300 resize, ImageNet normalization).
- `_check_ood(image, tensor)` — runs the stages in `ood.py`; returns a rejection with a
  reason, or `None` to proceed.
- `generate_gradcam_base64(image, target_class_idx)` — gradient-weighted activation map
  over `conv_head` with a Jet colormap, as a base64 PNG. Returns `None` when attribution
  cannot be computed; it never substitutes a placeholder.
- `predict(image)` — OOD gate, forward pass, sigmoid readout, per-class threshold margin
  rule, Grad-CAM. Temperature scaling is **not** applied: the upstream calibration run
  measured it as harmful (NLL 1.045 -> 1.173) and the reported metrics use temperature 1.0.

#### `ood.py`
- `compute_metrics(image)` — illumination-invariant statistics (`rel_contrast`,
  `hf_ratio`, `blue_green`, `chromatic_fraction`).
- `color_gate(image, thresholds)` — stage 1; returns `is_ood`, `reason`, `detail`, `metrics`.
- `FeatureSpaceOOD` — stage 2, Mahalanobis distance in the classifier's feature space.
  Reports itself unavailable until fitted by `scripts/calibrate_ood.py`.

#### `model.py`
- `build_model(arch, num_classes)` — `plain` (timm EfficientNet-B3, single linear head)
  or `multihead` (two-layer head). The architectures are not interchangeable.
- `get_conv_head(model)` / `get_pooled_features(model, x)` — architecture-agnostic accessors.

#### `routers/diagnostics.py`
- `POST /api/analyze` — validates the upload, runs inference, records the session.
- `GET /api/history` — completed sessions, newest first (`limit` 1–200).
- `DELETE /api/history/all` and `DELETE /api/history/{id}` — both require `X-Admin-Token`
  and are disabled (403) while `ADMIN_TOKEN` is unset.
- `require_admin(x_admin_token)` — the guard for both delete routes.

#### `database.py` / `models.py`
- `init_db()` — creates the schema.
- `get_db()` — session dependency.
- `DiagnosticSession` — the only table (`diagnostic_sessions`), with fields
  `id`, `image_path`, `status`, `prediction`, `confidence`, `threshold_used`,
  `all_scores`, `is_high_risk`, `anatomic_site`, `created_at`, `completed_at`.

### Frontend (`frontend/js/app.js`)
- `navigate(viewId, payload)` — swaps between `view-console`, `view-results`,
  `view-compare`, `view-analytics` and `view-knowledge`.
- `toast(message, type)` — floating notifications; creates its own container.
- `runAnalysis()` — submits the scan and routes to the results view. If the selected
  image cannot be fetched it raises rather than substituting anything.
- `renderGradCamHeatmap(base64Data)` — draws the attribution map onto
  `#console-heatmap-canvas`. With no data the canvas stays empty.
- `toggleGradCamHeatmap()`, `updateViewportTransform()`, `zoomIn()`, `zoomOut()`,
  `resetViewportTransform()` — viewport controls.
- `updateCalibrationFilters()` / `resetCalibration()` — brightness, contrast and
  saturation CSS filters. Display only; they do not affect what is sent for inference.
- `handleFileSelected(file)`, `renderResultsView(result)`, `renderHistoryTable(logs)`.
- `deleteHistoryRecord(id)` — deletes one session, sending the `X-Admin-Token` header.
  The token is prompted for once and stored in `localStorage`; `getStoredAdminToken()`,
  `setStoredAdminToken()` and `requestAdminToken()` manage it.
- `downloadClinicalReport()` — client-side export via html2pdf, falling back to
  `window.print()`. There is no server-side PDF endpoint.

---

## 🔬 Test Samples

`samples/` holds **21 labelled dermoscopic images**, three per diagnostic class, for
exercising the classifier without hunting for data.

| Code | Diagnosis | Risk | Files | Example |
| :--- | :--- | :--- | ---: | :--- |
| `akiec` | Actinic keratosis / intraepithelial carcinoma | Pre-malignant | 3 | `akiec_1_ISIC_0029659.jpg` |
| `bcc` | Basal cell carcinoma | Malignant | 3 | `bcc_1_ISIC_0029230.jpg` |
| `bkl` | Benign keratosis | Benign | 3 | `bkl_1_ISIC_0025915.jpg` |
| `df` | Dermatofibroma | Benign | 3 | `df_1_ISIC_0029760.jpg` |
| `mel` | Melanoma | Malignant | 3 | `mel_1_ISIC_0026120.jpg` |
| `nv` | Melanocytic nevus | Benign | 3 | `nv_1_ISIC_0032285.jpg` |
| `vasc` | Vascular lesion | Benign | 3 | `vasc_1_ISIC_0029486.jpg` |

Naming is `<class>_<n>_<ISIC id>.jpg`, so the ground truth is visible in the filename and
every image traces back to its dataset row. All filenames were verified against the split
manifest.

Two properties make these usable as an honest test:

- **They come from the held-out test split of the partition the deployed model was
  trained under** (`models/splits/split_test.csv`). The model never trained on them, so
  results reflect real generalisation rather than memorisation. Regenerate them whenever
  the served model changes: `python scripts/build_test_samples.py --manifest
  models/splits/split_test.csv --image-root path/to/ham10000`.
- **They were selected by manifest order, not by what the model gets right.** A sample set
  curated on model success would look better than the model is.

`cat.jpg` is a non-skin control: it should be **rejected** by the OOD gate, not classified.
(`nv.jpg` and `ISIC_0024307.jpg` are earlier reference images retained because the sample
gallery links to them.)

### Current behaviour on these samples

Measured with the deployed checkpoint — **13 of 21 correct**, melanoma **3/3**:

| Ground truth | Correct | Notes |
| :--- | :--- | :--- |
| `mel`, `nv`, `df` | 3/3, 3/3, 2/3 | every melanoma caught by the prediction itself |
| `akiec` | 2/3 | one read as `df` |
| `bkl` | 1/3 | one as `nv`, one as `akiec` |
| `bcc` | 1/3 | two read as `nv` |
| `vasc` | 1/3 | two read as `mel` — surfaced for review, so they fail safe |

**Do not read 13/21 as accuracy.** This set is balanced three-per-class, while
the real distribution is 66% `nv`; measured accuracy on the full 1503-image test
split is 0.8090. Twenty-one images also carry enormous variance — the `bcc` and
`vasc` rows here are worse than their measured test recalls (0.68 and 0.82). It
is a smoke test that the serving path works end to end, not a metric.

The melanoma row is the one that matters, and it is why the alert channel exists:
all three melanomas are caught by the prediction itself, and the alert flags a
further four cases (two `vasc`, one `df`, one `bkl`) for review. Over-calling
melanoma on a vascular lesion is the direction this system is deliberately tuned
to fail in.

Regenerate or resize the set at any time:

```bash
python scripts/build_test_samples.py --per-class 5
```

It replaces the class-prefixed files and leaves the controls alone. Full manifest and the
per-file breakdown: [samples/README.md](samples/README.md). These images are from HAM10000
and carry that dataset's licence (6 MB in-repo).

---

## 🧬 Model & Pathology Specifications

The model classifies cutaneous lesions across **7 clinical categories**:

| Pathology Code | Disease Name | Risk Stratification |
|---|---|---|
| **MEL** | Melanoma | **HIGH** (Malignant) |
| **NV** | Melanocytic Nevi (Common Mole) | **LOW** (Benign) |
| **BCC** | Basal Cell Carcinoma | **HIGH** (Malignant) |
| **AKIEC** | Actinic Keratoses / Intraepithelial Carcinoma | **MODERATE** (Pre-cancerous) |
| **BKL** | Benign Keratosis (Seborrheic Keratosis) | **LOW** (Benign) |
| **DF** | Dermatofibroma | **LOW** (Benign) |
| **VASC** | Vascular Lesions | **LOW** (Benign) |

The API exposes a single `is_high_risk` boolean, which is true for `mel`, `bcc` and
`akiec`. It groups the pre-malignant class with the malignant ones, so it is a triage
hint rather than clinical staging. The three-level wording above is presentation only
and is applied in the frontend.

### Measured performance

Measured on a **lesion-disjoint** held-out split of HAM10000 (1503 images) with
the served configuration — `models/latest.pt` at 300x300, softmax readout,
temperature 1.15, per-class thresholds:

- **Accuracy:** 0.8090   **Macro-F1:** 0.7123   **ECE:** 0.0337
- **Melanoma recall: 0.800** — 20% of melanomas are missed by the prediction;
  the alert channel surfaces **92.9%** of them at a 23.6% review rate.

| Class | Support | Precision | Recall | F1 |
| :--- | ---: | ---: | ---: | ---: |
| `akiec` | 52 | 0.5312 | 0.6538 | 0.5862 |
| `bcc` | 79 | 0.8571 | 0.6835 | 0.7606 |
| `bkl` | 158 | 0.7073 | 0.5506 | 0.6192 |
| `df` | 22 | 0.6364 | 0.6364 | 0.6364 |
| `mel` | 170 | 0.4806 | 0.8000 | 0.6004 |
| `nv` | 1000 | 0.9397 | 0.8730 | 0.9051 |
| `vasc` | 22 | 0.9474 | 0.8182 | 0.8780 |

Melanoma precision is 0.48 by design: the decision rule deliberately over-calls
melanoma, because a false alarm costs a review and a miss can cost a life. That
choice is what makes 23.6% of scans reviewable, and it should be signed off as a
clinical policy rather than inherited from a threshold search.

**These numbers are not comparable to this project's earlier published figures**
(accuracy 0.8505, macro-F1 0.7450, melanoma recall 0.624). Those were measured on
an image-level split. HAM10000 holds ~10015 images of ~7470 lesions, so splitting
by image puts other photographs of the same lesion on both sides of the
train/test line and inflates every metric by an unknown margin. The figures above
are lower on accuracy, far higher on melanoma recall, and honest. See
[training/README.md](training/README.md) for why the two cannot be compared
directly, and what happened when the old model was re-measured on this split.

This system is a research prototype and is **not a substitute for examination by
a qualified clinician.**

To reproduce on a labelled hold-out set:

```bash
python scripts/evaluate_model.py --data-dir path/to/test_set
```

## Documentation

- **[docs/MODEL_DETAILS.md](docs/MODEL_DETAILS.md)** — measured per-class metrics, the
  confusion matrix, threshold operating point, and the OOD gate.
- **[docs/PROJECT_ARCHITECTURE.md](docs/PROJECT_ARCHITECTURE.md)** — stack, request flow,
  API surface, configuration.
- **[docs/FEATURE_STATUS.md](docs/FEATURE_STATUS.md)** — what the UI implements, what it
  does not, and known issues.
- **[docs/DermaScan_Serving_Audit.pdf](docs/DermaScan_Serving_Audit.pdf)** — printable
  audit report.
- **[training/README.md](training/README.md)** — retraining on a Kaggle GPU, and why the
  notebook is built the way it is.

Every figure in these documents is measured. Reproduce them with
`scripts/evaluate_model.py` and `scripts/build_class_samples.py`.

---

## ⚖️ Licence

Source code is MIT licensed — see [LICENSE](LICENSE).

The dermoscopic images in `samples/` and `docs/class_samples_real.png` are **not**
covered by MIT. They come from the **HAM10000** dataset, which is distributed under a
Creative Commons **NonCommercial** licence: redistribution is permitted with attribution
for non-commercial use only. The trained weights are derived from that dataset and
inherit the same restriction.

> Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of
> multi-source dermatoscopic images of common pigmented skin lesions.* Sci. Data 5,
> 180161 (2018). <https://doi.org/10.1038/sdata.2018.161>

`samples/cat.jpg` has unknown provenance and should be replaced with an image of known
licence before public redistribution.

**This is a research prototype, not a medical device.** It has not been clinically
validated. Measured melanoma recall is 0.800 — a confident output is not evidence of a
benign lesion. See the disclaimer in [LICENSE](LICENSE).
