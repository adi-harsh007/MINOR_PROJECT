# DermaScan AI 4.0 — Clinical Skin Cancer Diagnostic Studio

DermaScan AI is a clinical-grade skin lesion classification system powered by an **EfficientNet-B3** deep neural network trained on the ISIC/HAM10000 dataset. It provides real-time pathology prediction across 7 skin cancer classes, spatial **Grad-CAM AI Heatmap** activation visualization, Out-of-Distribution (OOD) non-skin rejection, and persistent clinical diagnostic history.

---

## 📁 Project Architecture & Directory Structure

```text
MODEL_Skin-Cancer/
├── backend/                  # FastAPI REST Backend & PyTorch ML Engine
│   ├── main.py               # FastAPI application entrypoint & static file mounts
│   ├── ml_engine.py          # PyTorch EfficientNet-B3 predictor & Grad-CAM hook engine
│   ├── database.py           # SQLAlchemy SQLite connection & session manager
│   ├── models.py             # Database ORM schema for clinical records
│   ├── config.py             # Directory paths & environment configuration
│   └── routers/
│       └── diagnostics.py    # Diagnostic analysis, history & report export endpoints
├── frontend/                 # Single-Page Web Application (SPA)
│   ├── index.html            # Core UI layout, viewport canvas & diagnostic views
│   └── js/
│       └── app.js            # SPA view router, canvas heatmap renderer & API handler
├── models/                   # Neural Network Models & Thresholds
│   ├── latest.pt             # EfficientNet-B3 PyTorch model weights (129 MB)
│   └── class_thresholds.json # Optimized per-class decision thresholds
├── data/                     # Data Persistence & Image Uploads
│   ├── pathology.db          # SQLite clinical history database
│   └── uploads/              # Diagnostic lesion image storage
├── docs/                     # Technical Documentation & Performance Graphs
│   ├── MODEL_DETAILS.md      # Architecture metrics, confusion matrices & OOD logic
│   └── PROJECT_ARCHITECTURE.md# System architecture breakdown
├── samples/                  # Representative test images (ISIC/HAM10000)
├── scripts/                  # Model evaluation scripts
├── tests/                    # Out-of-Distribution test scripts
├── requirements.txt          # Python dependencies
└── start.py                  # Unified single-command startup script
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
   - **OOD Gatekeeper**: `SkinCancerPredictor.is_out_of_distribution()` checks color histograms (HSV hue variance, saturation levels) to filter non-skin images.
   - **EfficientNet-B3 Pass**: Image is resized to 224x224 and normalized. Forward pass computes logits for 7 pathology classes (MEL, NV, BCC, AKIEC, BKL, DF, VASC).
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
- `favicon()`: Handles favicon requests with index/fallback.
- `on_startup()`: Initializes SQLite database tables on server start.
- `health_check()`: Returns system health status, model architecture details, and version string.

#### **`backend/ml_engine.py` (`SkinCancerPredictor` Class)**
- `__init__()`: Loads PyTorch model weights (`models/latest.pt`), decision thresholds (`models/class_thresholds.json`), registers PyTorch hooks on `conv_head`, and initializes ImageNet normalization pipelines.
- `is_out_of_distribution(image_bytes)`: Dual-layer OOD analyzer evaluating HSV color space histograms (saturation, value, hue variance) to reject non-lesion/non-skin images.
- `generate_gradcam_base64(image_bytes, target_class_idx)`: Calculates class activation maps using feature maps and gradients from `conv_head`, applies a Jet colormap, and returns a Base64-encoded PNG image.
- `predict(image_bytes)`: Complete inference workflow including OOD check, forward pass, temperature-scaled confidence normalization, per-class thresholding, risk stratification, and Grad-CAM map generation.

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

For complete performance metrics, confusion matrices, and ROC curves, see **[docs/MODEL_DETAILS.md](file:///d:/ML/MODEL_Skin-Cancer/docs/MODEL_DETAILS.md)**.
