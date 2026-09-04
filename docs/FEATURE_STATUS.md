# Feature Status

What the interface actually does, checked against `frontend/index.html` and
`frontend/js/app.js`. This replaces the previous feature roadmap, which described
planned work in the present tense and listed capabilities that were never built.

## Implemented

| Feature | Where | Notes |
| :--- | :--- | :--- |
| Image upload and drag-drop | `handleFileSelected()` | Extension and size are enforced server-side. |
| Reference sample gallery | `CLINICAL_SAMPLES` | Eight real images served from `/samples`: one HAM10000 lesion per class, labels sourced from the split manifests, plus one non-skin control. |
| Anatomic site tags | `view-console` | Submitted with the scan and stored on the record. Only the six sites the UI offers are accepted; anything else is discarded. It does not affect inference — the model sees only the image. |
| Zoom / pan viewport | `updateViewportTransform()` | Image and heatmap canvas transform together. |
| Brightness / contrast / saturation | `updateCalibrationFilters()` | CSS filters for viewing only. They do not change the image sent for inference, and the panel says so. |
| Grad-CAM overlay | `renderGradCamHeatmap()` | Real attribution only. When the API returns `null` the canvas stays empty and the UI says so. `syncHeatmapCanvasToImage()` sizes the canvas onto the image's rendered box so the overlay lines up with the lesion. |
| 7-class probability bars | `renderResultsView()` | Rendered from the API `scores` object. |
| Radial confidence gauge | `#gauge-confidence-circle` | Displays the predicted class confidence. |
| Analysed image on the result panel | `renderResultsImage()` | The stored upload, refetched from the record via `GET /api/history/{id}/image`, so it is the image the model was given rather than whatever the console currently shows. Carries its own Grad-CAM toggle, sized onto the image by `syncResultsHeatmapCanvas()`. |
| OOD rejection feedback | `apiCall()` | Shows the specific reason returned by the API (`uniform_field`, `pixel_noise`, `non_skin_colour`, `low_confidence`). |
| Diagnostic history table | `renderHistoryTable()` | Reads `GET /api/history`. |
| Analytics view | `view-analytics` | Scan counts and class distribution computed client-side from history. |
| Side-by-side comparison | `renderCompare()` | Two **recorded** scans chosen from history. Shows both stored images, per-class softmax for each with the arithmetic difference B − A, and every persisted field (prediction, confidence, high-risk flag, melanoma alert, p(mel), decision threshold, site, timestamps). Reads `GET /api/history/{id}` and `GET /api/history/{id}/image`. Differences are between two model outputs; nothing about the skin is measured, and the view says so. |
| Knowledge reference | `view-knowledge` | Static reference cards for the 7 classes. |
| Model card | `loadModelCard()` | Publishes the recorded evaluation from `GET /api/model`: accuracy, macro F1, ECE, the melanoma operating point (recall, surfaced, review rate), a per-class table with the live thresholds, and the confusion matrix. Leads with whether those figures were measured under the thresholds actually being served. Every number is read from a file; an unavailable artifact is reported as unavailable. |
| History search / filter | `history-search-input` | Filters the table by record id, class code or pathology name. |
| CSV export | history view | Client-side download of the history table, including the anatomic site. |
| OOD gate readout | `renderOodPanel()` | Shows the statistics the gate measured on this image (`rel_contrast`, `hf_ratio`, `blue_green`) and states plainly whether the thresholds are fitted and whether the feature-space stage ran. |
| Clinical summary | `buildClinicalSummary()` | Derived from the prediction, the high-risk flag and the melanoma alert. Never a fixed string. The melanoma-recall figure it quotes comes from `operating_point` in the API response, which the predictor reads out of `models/class_thresholds.json`; if the server sends none, the sentence is omitted rather than filled with a remembered number. |
| PDF / print report | `buildPrintReport()`, `downloadClinicalReport()` | Renders a purpose-built light document (`#print-report`) rather than screenshotting the app: submitted image, Grad-CAM, classification, all seven probabilities, input-screening readout, derived summary and disclaimer. Client-side via html2pdf, falling back to `window.print()`, which prints the same node. There is no server-side PDF endpoint. |
| Toast notifications | `toast()` | Creates its own container; `index.html` does not define one. |
| Analysis loading state | `setAnalysisOverlay()` | Covers the viewport for the duration of the request. |
| Server-side timings | `formatMs()` | `/api/analyze` returns `inference_ms` (forward pass) and `queue_ms` (wait for a concurrency slot); the result panel shows both. They are measured and reported separately on purpose — summed, "inference" would track load rather than the model. Not persisted: `DiagnosticSession` has no column for them, so history rows and the CSV do not carry them. |
| Responsive layout | `index.html` breakpoints | Single column below `xl`; below `lg` the sidebar is an off-canvas drawer (`openNavDrawer()` / `closeNavDrawer()`), opened from a hamburger and closed by the backdrop, Escape, or picking a view. |
| Delete a history record | `deleteHistoryRecord()` | Sends `X-Admin-Token`. The token is requested once and kept in this browser's `localStorage`; a rejected token is discarded so the next attempt re-asks. |

## Not implemented

These appeared in the previous roadmap. None of them exist in the code:

- **Magnifier reticle / zoom lens.** Never built. The stray `index.html` comment that
  referred to one is gone.
- **Sharpness adjustment.** Only brightness, contrast and saturation exist.
- **Dermoscopic colour filters** (monochrome, inverted, edge isolation).
- **Live camera / dermoscope capture.** No `getUserMedia` call anywhere.
- **Interactive body-map selector.** Anatomic site is a set of text buttons.
- **Loose images in the comparison view.** Both slots used to accept arbitrary
  uploads. An upload has not been through the model, so there was nothing to put
  beside it — the view could show pictures it had no output for, or output it had no
  picture for, but never both. Comparison is now anchored to recorded scans; to
  compare a new image, run it through the scan console first.
- **Progression assessment in the comparison view.** The view differences the two
  *model outputs* — that is arithmetic on stored softmax vectors. It computes nothing
  about the lesion itself: no growth, diameter, pigmentation or border metric exists
  anywhere in the codebase, and nothing links two records to the same lesion or
  patient, so no progression claim is available to make. A previous build shipped a
  "differential engine" that
  appeared to compute one: growth percentage, confidence, pigmentation density and
  border irregularity were all derived from `img.src.length % 25` — the length of the
  base64 string, which is a file size — and it printed progression verdicts including
  "Recommending urgent biopsy evaluation" on that basis. It ran automatically whenever
  an image was loaded into either slot. It has been removed in full, along with the
  fabricated defaults the markup shipped (invented scan dates, a hardcoded
  `MEL (Melanoma)` tag on scan B, and a "delta heatmap" overlay that stroked a fixed
  ellipse at a fixed position regardless of the image).
(Previously this list also claimed history search was missing. It is not: the
`history-search-input` handler filters by id, class code and pathology name. The
entry was wrong and has been moved to the implemented table above.)

## Known issues

- **`models/class_thresholds.json` mislabels its own metrics.** The file records
  `fitted_on: "calib+val splits (lesion-disjoint)"`, but the `per_class_metrics` stored
  beside the thresholds are byte-identical to the 1503-sample test evaluation in
  `docs/evaluation_serving_check.json` — so that label describes where the *thresholds*
  were fitted, not where those metrics were measured. The API passes the value through
  as `operating_point.thresholds_fitted_on` and the UI does not restate the split,
  because the served artifact does not establish it. The model card shows the label
  verbatim under Serving configuration, so the ambiguity is visible rather than
  hidden behind a figure.

- **Deleting requires `ADMIN_TOKEN` to be set server-side.** With it unset the API
  returns 403 and the UI reports deletion as disabled, which is the default state.
- **Anatomic site is recorded but unused by the model.** It is stored for the record
  and shown in exports; inference takes the image alone.
- **The interface is dark only.** `<html class="dark">` is fixed and there is no
  light palette. `prefers-color-scheme` is not consulted.
