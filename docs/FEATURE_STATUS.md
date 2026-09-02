# Feature Status

What the interface actually does, checked against `frontend/index.html` and
`frontend/js/app.js`. This replaces the previous feature roadmap, which described
planned work in the present tense and listed capabilities that were never built.

## Implemented

| Feature | Where | Notes |
| :--- | :--- | :--- |
| Image upload and drag-drop | `handleFileSelected()` | Extension and size are enforced server-side. |
| Reference sample gallery | `CLINICAL_SAMPLES` | Three real images served from `/samples`: two HAM10000 lesions with labels sourced from the split manifests, and one non-skin control. |
| Anatomic site tags | `view-console` | Recorded in frontend state only. The backend ignores the `site` field — it is not stored and does not affect inference. |
| Zoom / pan viewport | `updateViewportTransform()` | Image and heatmap canvas transform together. |
| Brightness / contrast / saturation | `updateCalibrationFilters()` | CSS filters for viewing only. They do not change the image sent for inference. |
| Grad-CAM overlay | `renderGradCamHeatmap()` | Real attribution only. When the API returns `null` the canvas stays empty and the UI says so. |
| 7-class probability bars | `renderResultsView()` | Rendered from the API `scores` object. |
| Radial confidence gauge | `#gauge-confidence-circle` | Displays the predicted class confidence. |
| OOD rejection feedback | `apiCall()` | Shows the specific reason returned by the API (`uniform_field`, `pixel_noise`, `non_skin_colour`, `low_confidence`). |
| Diagnostic history table | `renderHistoryTable()` | Reads `GET /api/history`. |
| Analytics view | `view-analytics` | Scan counts and class distribution computed client-side from history. |
| Side-by-side comparison | `view-compare` | Two scans in adjacent viewports. |
| Knowledge reference | `view-knowledge` | Static reference cards for the 7 classes. |
| CSV export | history view | Client-side download of the history table. |
| PDF report export | `downloadClinicalReport()` | Client-side via html2pdf, falling back to `window.print()`. There is no server-side PDF endpoint. |
| Toast notifications | `toast()` | Creates its own container; `index.html` does not define one. |

## Not implemented

These appeared in the previous roadmap. None of them exist in the code:

- **Magnifier reticle / zoom lens.** Only a comment in `index.html` mentions it.
- **Sharpness adjustment.** Only brightness, contrast and saturation exist.
- **Dermoscopic colour filters** (monochrome, inverted, edge isolation).
- **Live camera / dermoscope capture.** No `getUserMedia` call anywhere.
- **Interactive body-map selector.** Anatomic site is a set of text buttons.
- **Progression flags in comparison view.** The two scans are shown side by side; no
  differential is computed.
- **History search and filtering.** The table is unfiltered.

## Known issues

- **History deletion returns 401 from the UI.** `app.js` calls
  `DELETE /api/history/{id}`, which now requires an `X-Admin-Token` header that the
  frontend does not send. The delete control will fail until either the frontend
  sends the token or the endpoint is reworked for the local single-user case.
- **Anatomic site is collected but discarded.** The UI implies it is part of the
  record; it is not persisted.
- **`/api/health` is static.** It reports a hardcoded engine name and version rather
  than the loaded checkpoint.
