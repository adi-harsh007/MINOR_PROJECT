# DermaScan AI 4.0 — Clinical Skin Cancer Diagnostic Studio

DermaScan AI is a **research prototype** for skin lesion classification, built on an
**EfficientNet-B3** network trained on HAM10000. It predicts across 7 lesion classes and
provides Grad-CAM attribution, out-of-distribution rejection, and a local history of past
scans.

It is **not a diagnostic device and not a substitute for examination by a qualified
clinician.** Measured melanoma recall is 0.624 — roughly 38% of melanomas in the held-out
test set are missed. See [Measured performance](#measured-performance).

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
├── models/                     # gitignored
│   ├── latest.pt               # EfficientNet-B3 weights (123 MB)
│   └── class_thresholds.json   # Per-class decision thresholds
├── data/                       # gitignored
│   ├── pathology.db            # SQLite diagnostic history
│   └── uploads/                # Stored lesion images
├── docs/
│   ├── MODEL_DETAILS.md        # Measured performance, thresholds, OOD logic
│   ├── PROJECT_ARCHITECTURE.md # Stack, request flow, API surface
│   ├── FEATURE_STATUS.md       # What the UI actually implements
│   ├── evaluation_results.json # Raw measured test-set metrics
│   ├── confusion_matrix_measured.png
│   ├── class_samples_real.png  # Real HAM10000 example per class
│   └── DermaScan_Serving_Audit.pdf
├── samples/                    # Reference dermoscopic images + a non-skin control
├── scripts/
│   ├── evaluate_model.py       # Measure performance on a labelled hold-out set
│   ├── optimize_thresholds.py  # Fit thresholds to the served configuration
│   ├── calibrate_ood.py        # Fit the OOD gate to real data
│   ├── build_class_samples.py  # Build the class figure from real images
│   └── build_audit_pdf.py      # Build the audit PDF
├── tests/                      # pytest suite (runs against a temp database)
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
     image binary. A `site` field is also sent, but the backend currently ignores it — the
     anatomic site is not stored and does not affect inference.

3. **Backend ML Pipeline Execution**:
   - **OOD Gatekeeper**: `backend/ood.py` applies illumination-invariant image statistics (relative contrast, high-frequency ratio, chromatic hue), then a feature-space check once fitted.
   - **EfficientNet-B3 Pass**: Image is resized to 300x300 (matching training) and normalized. Forward pass computes logits for 7 pathology classes (MEL, NV, BCC, AKIEC, BKL, DF, VASC).
   - **Grad-CAM Activation Engine**: Backward pass on `conv_head` layer computes gradient-weighted feature maps, rendering a Jet colormap (Blue -> Red) heatmap.
   - **Database Log**: SQLAlchemy saves diagnosis details to `data/pathology.db`.

4. **Frontend Response & Visualization**:
   - The API returns `session_id`, `prediction`, `confidence`, `threshold`, the seven
     class `scores`, an `is_high_risk` boolean, and `heatmap_base64` (which is `null` when
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
- `health_check()` — returns a static `{status, engine, version}` payload. The engine
  string is hardcoded and does not reflect the loaded checkpoint.

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
  `all_scores`, `is_high_risk`, `created_at`, `completed_at`.

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
- `downloadClinicalReport()` — client-side export via html2pdf, falling back to
  `window.print()`. There is no server-side PDF endpoint.

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

Measured on the full held-out HAM10000 test split (1525 images) with the served
configuration — `models/latest.pt` at 300x300, sigmoid readout, per-class
thresholds:

- **Accuracy:** 0.8505   **Macro-F1:** 0.7450
- **Melanoma recall: 0.624** — roughly 38% of melanomas are missed, most
  of them misread as benign nevi.

Melanoma recall is the limiting factor and the dominant clinical risk. This
system is a research prototype and is **not a substitute for examination by a
qualified clinician.**

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

Every figure in these documents is measured. Reproduce them with
`scripts/evaluate_model.py` and `scripts/build_class_samples.py`.
