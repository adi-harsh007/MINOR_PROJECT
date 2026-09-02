# DermaScan AI — Project Architecture

How the pieces fit together: tech stack, request flow, and the API surface.
Every statement here is checked against the code; where behaviour is not
implemented, this document says so rather than describing an intention.

## Technology Stack

### Backend
- **[FastAPI](https://fastapi.tiangolo.com/)** — HTTP API and static file serving.
- **[Uvicorn](https://www.uvicorn.org/)** — ASGI server. Launched by `start.py`.
- **[SQLite](https://www.sqlite.org/)** — single-file database at `data/pathology.db`.
- **[SQLAlchemy](https://www.sqlalchemy.org/)** — ORM; one table, `diagnostic_sessions`.

### Machine learning
- **[PyTorch](https://pytorch.org/)** (>= 2.6) — inference only. No training code lives
  in this repository; the model is trained separately and the checkpoint is copied in.
- **[timm](https://github.com/huggingface/pytorch-image-models)** — supplies the
  EfficientNet-B3 backbone.
- **[Torchvision](https://pytorch.org/vision/stable/index.html)** — resize and
  normalize. No centre crop: the model is fed a plain 300x300 resize, matching how
  it was trained and evaluated.
- **[Pillow](https://python-pillow.org/)** and **[NumPy](https://numpy.org/)** —
  image decoding and the out-of-distribution statistics in `backend/ood.py`.

### Frontend
- **Single-page application**, vanilla ES6 JavaScript. No build step, no framework.
- **[Tailwind CSS](https://tailwindcss.com/)** via CDN, and
  **[Material Symbols](https://fonts.google.com/icons)** for icons.
- **[html2pdf.js](https://github.com/eKoopmans/html2pdf.js)** via CDN for the
  client-side report export, falling back to `window.print()` when it fails to load.
- Dark theme, served from `frontend/index.html` and `frontend/js/app.js`.

The frontend is served by the same FastAPI process and calls the API at the
relative path `/api`. It therefore cannot be hosted separately without changing
`API_BASE` in `app.js` or adding a proxy.

---

## Directory Layout

- `backend/` — API routes, ORM models, and inference.
  - `model.py` — the two network architectures; selected by `MODEL_ARCH`.
  - `ml_engine.py` — preprocessing, readout, decision rule, Grad-CAM.
  - `ood.py` — out-of-distribution gate.
- `frontend/` — `index.html` and `js/app.js`.
- `models/` — `latest.pt` and `class_thresholds.json` (gitignored; not in the repo).
- `data/` — `pathology.db` and uploaded images.
- `scripts/` — evaluation, threshold fitting, OOD calibration, figure builders.
- `tests/` — pytest suite; runs against a temporary database, never `data/pathology.db`.

---

## Request Flow

1. **Upload.** The browser POSTs `multipart/form-data` to `/api/analyze`, including a
   `site` field. The site is validated against the six values the UI offers and stored
   on the record; unrecognised values are discarded. It does not affect inference.
2. **Validation.** The extension is checked against an allowlist, the body against
   a 10 MB cap, and the image is validated by decoding it. The client-supplied
   `content_type` is not trusted.
3. **OOD gate** (`backend/ood.py`). Illumination-invariant image statistics reject
   flat colour fields, pixel noise, and non-skin hues. A feature-space Mahalanobis
   stage exists but is **inactive until fitted** by `scripts/calibrate_ood.py`.
   Rejections return HTTP 422 with a machine-readable reason.
4. **Inference.** Resize to 300x300, ImageNet normalization, forward pass through
   EfficientNet-B3, sigmoid readout.
5. **Decision.** `argmax(probability - class threshold)` using the per-class
   thresholds in `models/class_thresholds.json`. A negative best margin is rejected
   as `low_confidence`.
6. **Attribution.** Grad-CAM over `conv_head`, returned as a base64 PNG. If it
   cannot be computed the API returns `null` — never a placeholder image.
7. **Persistence.** A `diagnostic_sessions` row is written with the prediction,
   confidence, threshold used, all seven scores, the anatomic site, and a high-risk
   flag. Columns added after a database was created are applied on startup by
   `_add_missing_columns()`, since `create_all()` never alters an existing table.
8. **Response.** JSON is rendered by the SPA, which draws the heatmap on a canvas
   overlaying the lesion image.

Rejected scans are not persisted, and their uploaded file is deleted.

---

## API

| Method | Path | Notes |
| :--- | :--- | :--- |
| `POST` | `/api/analyze` | Inference and OOD gating. Returns `session_id`, `prediction`, `confidence`, `threshold`, `scores`, `is_high_risk`, `anatomic_site`, `heatmap_base64`. |
| `GET` | `/api/history` | Completed sessions, newest first. `limit` is bounded to 1–200. |
| `DELETE` | `/api/history/all` | Deletes every session and its image. **Requires `X-Admin-Token`.** |
| `DELETE` | `/api/history/{id}` | Deletes one session. **Requires `X-Admin-Token`.** |
| `GET` | `/api/health` | Serving configuration: architecture, checkpoint name and presence, input size, and whether the model has been loaded yet. Does not trigger a load. |

Both `DELETE` endpoints return 403 when `ADMIN_TOKEN` is unset, which is the
default — they are disabled unless deliberately enabled.

There is no server-side PDF export endpoint. Report export is client-side only.

`is_high_risk` is true for `mel`, `bcc`, and `akiec`. Note this groups `akiec`
(pre-malignant) with the malignant classes; the flag is a triage hint, not a
clinical staging.

---

## Configuration

Set via environment or `.env`:

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `MODEL_PATH` | `models/latest.pt` | Served checkpoint. |
| `MODEL_ARCH` | `plain` | `plain` or `multihead`. Must match the checkpoint; loading is strict. |
| `IMG_SIZE` | `300` | Inference resolution. Must match training. |
| `THRESHOLD_PATH` | `models/class_thresholds.json` | Per-class decision thresholds. |
| `DATABASE_URL` | SQLite in `data/` | Database connection. |
| `CORS_ORIGINS` | localhost only | Allowed browser origins. |
| `ADMIN_TOKEN` | unset | Enables the delete endpoints. |
| `MAX_UPLOAD_BYTES` | `10485760` | Upload size cap. |

---

## Operation

`start.py` launches Uvicorn on port 8088 as a subprocess. It does **not**
initialise the database or validate the model itself — the database schema is
created by the FastAPI lifespan handler on startup, and the model is loaded
lazily on the first `/api/analyze` request. A missing or mismatched checkpoint
therefore surfaces as a 500 on that first request, not at launch.

Nothing in this system is encrypted or anonymised. Uploaded images are written to
`data/uploads/` in the clear and retained indefinitely; the database is a plain
SQLite file. There is no authentication on inference or history reads.

*Related:* [Model details and measured performance](./MODEL_DETAILS.md) · [README](../README.md)
