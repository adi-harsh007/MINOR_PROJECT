# DermaScan AI 4.0 — Clinical Skin Cancer Diagnostic Studio

DermaScan AI is a clinical-grade skin lesion classification system powered by an **EfficientNet-B3** deep neural network trained on the ISIC/HAM10000 dataset. It provides real-time pathology prediction across 7 skin cancer classes, spatial **Grad-CAM AI Heatmap** activation visualization, Out-of-Distribution (OOD) non-skin rejection, and persistent clinical diagnostic history.

---

## 📁 Project Architecture & Directory Structure

```text
MODEL_Skin-Cancer/
├── backend/                  # FastAPI backend & PyTorch inference
│   ├── main.py               # App entrypoint, CORS policy & static mounts
│   ├── config.py             # Paths, upload limits & security configuration
│   ├── model.py              # Network architectures (plain / multihead)
│   ├── ml_engine.py          # Predictor: preprocessing, readout & Grad-CAM
│   ├── ood.py                # Out-of-distribution gate (colour + feature space)
│   ├── database.py           # SQLAlchemy engine & session management
│   ├── models.py             # ORM schema for diagnostic records
│   └── routers/
│       └── diagnostics.py    # /api/analyze and history endpoints
├── frontend/                 # Single-page web application
│   ├── index.html            # UI layout, viewport & diagnostic views
│   └── js/app.js             # View router, heatmap renderer & API client
├── models/                   # Weights & decision thresholds (gitignored)
│   ├── latest.pt             # EfficientNet-B3 weights (123 MB)
│   └── class_thresholds.json # Per-class decision thresholds
├── data/
│   ├── pathology.db          # SQLite diagnostic history
│   └── uploads/              # Stored lesion images
├── docs/
│   ├── MODEL_DETAILS.md      # Measured performance, thresholds & OOD logic
│   ├── PROJECT_ARCHITECTURE.md
│   ├── evaluation_results.json      # Raw measured test-set metrics
│   └── confusion_matrix_measured.png
├── samples/                  # Reference dermoscopic images + a non-skin control
├── scripts/
│   ├── evaluate_model.py     # Measure performance on a labelled hold-out set
│   ├── optimize_thresholds.py# Fit per-class thresholds to the served config
│   └── calibrate_ood.py      # Fit the OOD gate to real data
├── tests/                    # pytest suite (runs against a temp database)
├── pytest.ini
├── requirements.txt
└── start.py                  # Single-command startup
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

### **Method 2: Starting Backend & Frontend Separately**

If you prefer running the API server and web interface in separate terminal windows:

#### **Step 1: Start Backend Server**
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8088 --reload
```
- **Backend API URL**: `http://localhost:8088`
- **Swagger Docs**: `http://localhost:8088/docs`

#### **Step 2: Start Frontend Web Server**
```bash
python -m http.server 8080 --directory frontend
```
- **Frontend SPA URL**: `http://localhost:8080`

---

## ⚙️ How Frontend & Backend Work Together

1. **User Upload & Configuration**:
   - The user selects or drags a dermatoscopic lesion scan into the Scan Console (`#console-viewport-img`).
   - Anatomic site metadata (e.g., *Anterior Torso*, *Head/Neck*) is tagged.

2. **API Inference Request**:
   - Clicking **"Begin AI Analysis"** sends a multipart HTTP `POST` request to `http://localhost:8088/api/analyze` with the image binary and anatomic tag.

3. **Backend ML Pipeline Execution**:
   - **OOD Gatekeeper**: `backend/ood.py` applies illumination-invariant image statistics (relative contrast, high-frequency ratio, chromatic hue), then a feature-space check once fitted.
   - **EfficientNet-B3 Pass**: Image is resized to 300x300 (matching training) and normalized. Forward pass computes logits for 7 pathology classes (MEL, NV, BCC, AKIEC, BKL, DF, VASC).
   - **Grad-CAM Activation Engine**: Backward pass on `conv_head` layer computes gradient-weighted feature maps, rendering a Jet colormap (Blue -> Red) heatmap.
   - **Database Log**: SQLAlchemy saves diagnosis details to `data/pathology.db`.

4. **Frontend Response & Visualization**:
   - The API returns probabilities, risk level (*HIGH*, *MODERATE*, *LOW*), ABCDE clinical findings, and Base64 Grad-CAM heatmap data.
   - `app.js` updates view to **Diagnostic Results**, draws the Grad-CAM heatmap onto `<canvas id="console-heatmap-canvas">`, and synchronizes zoom/pan transforms.

---

## 📖 Comprehensive Function & Component Index

### 🐍 Backend Python Functions (`backend/`)

#### **`backend/main.py`**
- `serve_index()`: Serves `frontend/index.html` as the main application SPA entrypoint.
- `favicon()`: Returns 204 No Content.
- `lifespan()`: Initializes database tables on startup.
- `health_check()`: Returns system health status, model architecture details, and version string.

#### **`backend/ml_engine.py` (`SkinCancerPredictor` Class)**
- `__init__()`: Builds the architecture named by `MODEL_ARCH`, strictly loads the checkpoint (failure is fatal), loads per-class thresholds, registers Grad-CAM hooks on `conv_head`, and sets up preprocessing (300x300 resize, ImageNet normalization).
- `_check_ood(image, tensor)`: Runs the OOD stages in `backend/ood.py` - illumination-invariant image statistics, then feature-space Mahalanobis distance when fitted. Returns a rejection with a reason, or `None` to proceed.
- `generate_gradcam_base64(image, target_class_idx)`: Computes a class activation map from `conv_head` activations and gradients, applies a Jet colormap, and returns a Base64 PNG. Returns `None` if attribution cannot be computed - never a synthetic stand-in.
- `predict(image)`: Full inference path - OOD gate, forward pass, sigmoid readout, per-class threshold margin rule, and Grad-CAM generation. Temperature scaling is **not** applied: the calibration run measured it as harmful (NLL 1.045 -> 1.173) and the reported metrics use temperature 1.0.

#### **`backend/routers/diagnostics.py`**
- `/api/analyze` (`analyze_lesion(file, site, db)`): End-to-end API endpoint that receives lesion scans, calls `predictor.predict()`, records diagnosis entries in SQLite database, and returns clinical JSON payloads + heatmap overlays.
- `/api/history` (`get_history(limit, db)`): Retrieves past diagnostic scan records sorted by timestamp.
- `/api/export-pdf/{scan_id}` (`export_pdf(scan_id, db)`): Generates printable clinical PDF reports containing diagnostic summaries, pathology codes, and risk assessments.

#### **`backend/database.py` & `backend/models.py`**
- `init_db()`: Creates SQLite database schema defined in `DiagnosticRecord`.
- `get_db()`: SQLAlchemy database session dependency injector.
- `DiagnosticRecord`: Database ORM model defining schema fields (`id`, `filename`, `anatomic_site`, `primary_diagnosis`, `pathology_code`, `risk_level`, `confidence`, `probabilities`, `timestamp`).

---

### 🌐 Frontend JavaScript Functions (`frontend/js/app.js`)

#### **SPA Routing & State Engine**
- `navigate(viewId, payload)`: Swaps active view (`view-console`, `view-results`, `view-history`, `view-knowledge`) and executes view initializer functions.
- `toast(message, type)`: Displays non-intrusive floating toast notifications with severity styling.

#### **Diagnostic Analysis & Viewport Engine**
- `runAnalysis()`: Captures lesion scan binary and anatomic site, triggers `/api/analyze` request, handles loading animations, and updates results view.
- `renderGradCamHeatmap(base64Data)`: Renders AI-generated Base64 activation map onto `<canvas id="console-heatmap-canvas">` or draws fallback thermal gradient.
- `toggleGradCamHeatmap()`: Toggles state and visibility of Grad-CAM overlay canvas with button style highlights.
- `updateViewportTransform()`: Applies CSS `scale()` and `translate()` properties synchronously to both `#console-viewport-img` and `#console-heatmap-canvas`.
- `zoomIn()` / `zoomOut()` / `resetViewportTransform()`: Handles zoom and pan viewport controls.
- `updateCalibrationFilters()` / `resetCalibration()`: Adjusts brightness, contrast, and saturation CSS filters on the lesion scan in real time.
- `handleFileSelected(file)`: Validates image drop/upload, creates object preview URLs, and updates console UI state.

#### **Results & History Management**
- `renderResultsView(result)`: Renders primary pathology prognosis, confidence gauges, risk tags, probability distribution bars, and ABCDE diagnostic breakdown.
- `fetchHistory()` / `renderHistoryTable(historyData)`: Fetches diagnostic scan logs from `/api/history` and populates the clinical history data table.
- `downloadClinicalReport()`: Formats and initiates print/PDF export for clinical documentation.

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

For per-class metrics and model details, see **[docs/MODEL_DETAILS.md](docs/MODEL_DETAILS.md)**.
