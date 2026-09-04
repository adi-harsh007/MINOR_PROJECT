# Feature Status

What the interface actually does, checked against `frontend/index.html` and
`frontend/js/app.js`. This replaces the previous feature roadmap, which described
planned work in the present tense and listed capabilities that were never built.

## Implemented

| Feature | Where | Notes |
| :--- | :--- | :--- |
| Image upload and drag-drop | `handleFileSelected()` | Extension and size are enforced server-side. |
| Reference sample gallery | `CLINICAL_SAMPLES` | Three real images served from `/samples`: two HAM10000 lesions with labels sourced from the split manifests, and one non-skin control. |
| Anatomic site tags | `view-console` | Submitted with the scan and stored on the record. Only the six sites the UI offers are accepted; anything else is discarded. It does not affect inference — the model sees only the image. |
| Zoom / pan viewport | `updateViewportTransform()` | Image and heatmap canvas transform together. |
| Brightness / contrast / saturation | `updateCalibrationFilters()` | CSS filters for viewing only. They do not change the image sent for inference. |
| Grad-CAM overlay | `renderGradCamHeatmap()` | Real attribution only. When the API returns `null` the canvas stays empty and the UI says so. |
| 7-class probability bars | `renderResultsView()` | Rendered from the API `scores` object. |
| Radial confidence gauge | `#gauge-confidence-circle` | Displays the predicted class confidence. |
| OOD rejection feedback | `apiCall()` | Shows the specific reason returned by the API (`uniform_field`, `pixel_noise`, `non_skin_colour`, `low_confidence`). |
| Diagnostic history table | `renderHistoryTable()` | Reads `GET /api/history`. |
| Analytics view | `view-analytics` | Scan counts and class distribution computed client-side from history. |
| Side-by-side comparison | `view-compare` | Two scans in adjacent viewports with synchronised pan and zoom. Visual inspection only — nothing is measured or differenced, and the view says so. |
| Knowledge reference | `view-knowledge` | Static reference cards for the 7 classes. |
| History search / filter | `history-search-input` | Filters the table by record id, class code or pathology name. |
| CSV export | history view | Client-side download of the history table, including the anatomic site. |
| OOD gate readout | `renderOodPanel()` | Shows the statistics the gate measured on this image (`rel_contrast`, `hf_ratio`, `blue_green`) and states plainly whether the thresholds are fitted and whether the feature-space stage ran. |
| Clinical summary | `buildClinicalSummary()` | Derived from the prediction, the high-risk flag and the melanoma alert. Never a fixed string. |
| PDF report export | `downloadClinicalReport()` | Client-side via html2pdf, falling back to `window.print()`. There is no server-side PDF endpoint. |
| Toast notifications | `toast()` | Creates its own container; `index.html` does not define one. |
| Delete a history record | `deleteHistoryRecord()` | Sends `X-Admin-Token`. The token is requested once and kept in this browser's `localStorage`; a rejected token is discarded so the next attempt re-asks. |

## Not implemented

These appeared in the previous roadmap. None of them exist in the code:

- **Magnifier reticle / zoom lens.** Only a comment in `index.html` mentions it.
- **Sharpness adjustment.** Only brightness, contrast and saturation exist.
- **Dermoscopic colour filters** (monochrome, inverted, edge isolation).
- **Live camera / dermoscope capture.** No `getUserMedia` call anywhere.
- **Interactive body-map selector.** Anatomic site is a set of text buttons.
- **Progression flags in comparison view.** The two scans are shown side by side; no
  differential is computed. A previous build shipped a "differential engine" that
  appeared to compute one: growth percentage, confidence, pigmentation density and
  border irregularity were all derived from `img.src.length % 25` — the length of the
  base64 string, which is a file size — and it printed progression verdicts including
  "Recommending urgent biopsy evaluation" on that basis. It ran automatically whenever
  an image was loaded into either slot. It has been removed in full, along with the
  fabricated defaults the markup shipped (invented scan dates, a hardcoded
  `MEL (Melanoma)` tag on scan B, and a "delta heatmap" overlay that stroked a fixed
  ellipse at a fixed position regardless of the image). The view now states that it
  performs visual comparison only.
(Previously this list also claimed history search was missing. It is not: the
`history-search-input` handler filters by id, class code and pathology name. The
entry was wrong and has been moved to the implemented table above.)

## Known issues

- **Deleting requires `ADMIN_TOKEN` to be set server-side.** With it unset the API
  returns 403 and the UI reports deletion as disabled, which is the default state.
- **Anatomic site is recorded but unused by the model.** It is stored for the record
  and shown in exports; inference takes the image alone.
