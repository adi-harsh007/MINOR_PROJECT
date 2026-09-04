# Changelog

## Unreleased — integrity fixes and backend hardening

A defect-driven pass over the whole application. Every figure quoted below was
measured on this machine; nothing here is estimated.

Test suite: **41 → 107 tests**. A lint gate was added. The served model, its
weights and its decision thresholds are unchanged — no retraining happened and
no published metric moved.

---

### Interface redesigned: calm clinical

The interface presented a cancer-triage tool as a neon sci-fi console. Glow
shadows on every panel, a spinning crosshair reticle over the lesion image, and
labels such as `SYSTEM STATUS: OPTIMAL`, `OOD Safety Shield Enabled` and
`REAL-TIME HARDWARE SHADER ACCELERATED` — the last of which described three CSS
filters. The chrome oversold the system in the same way the removed clinical
output did, so it was rebuilt on a restrained palette.

**Design tokens.** The Material-3 token set carried 60+ colours, most unused and
several redundant, which is how three different greens came to mean "fine" on
three different screens. It is now neutral slate surfaces, one accent (`primary`,
#62a8e8) used only for interactive elements, and exactly three semantic colours —
`risk-low`, `risk-mid`, `risk-high` — which are the only colours that carry
meaning. Every remaining token is used. All text pairs measured at or above WCAG
AA on their own ground.

**Typography.** Inter throughout, with IBM Plex Mono confined to identifiers and
measured numbers (tabular figures). Body copy was previously IBM Plex Mono at
10–12px uppercase across the OOD readout, the probability rows and the result
banner — the least legible font at the smallest size carrying the most important
numbers. The radius scale, in which `rounded-full` resolved to 0.75rem and every
pill and status dot therefore rendered as a rounded square, was corrected.

**Responsive layout.** There were no breakpoints. At 375px the fixed 288px
sidebar took 77% of the screen and content was squeezed into an 87px strip; at
~800px the console grid rendered one word per line. The sidebar is now an
off-canvas drawer below `lg` with a hamburger, backdrop and Escape-to-close, and
the console, result, compare and history grids all stack. The console was also a
fixed-height shell that clipped the display-adjustment sliders off the bottom at
1440×900 with nothing able to scroll to them.

**States.** Hover and selected were styled almost identically in the sidebar, so
the current view was not identifiable at a glance. Selected now carries a rail
and a tinted ground; hover only lightens. `aria-current`, `aria-pressed` and
`aria-expanded` are set, every interactive element has a visible focus ring, the
icon-only viewport buttons have accessible labels, and the drop zone answers
Enter and Space. Scrollbars are styled rather than hidden outright, which had
left scrollable panels with no indication that they scrolled.

State that was applied by rewriting whole Tailwind class strings from JavaScript
is now a single class toggle against component styles (`.nav-item.is-active`,
`.seg.is-active`, `.btn.is-on`, `.chip--high`), so the markup and the script can
no longer drift apart. Several of those strings referenced `bg-surface-elevated`,
`bg-surface-base` and `border-surface-elevated`, which were never defined in the
Tailwind config and silently rendered as nothing.

### A model card, from figures that were already measured

The repository's premise is honest measurement, and none of it reached the
interface. Accuracy, per-class recall, calibration error and the confusion matrix
sat in `docs/` where only someone reading the source would find them, while the
UI asserted a single recall figure hardcoded in JavaScript. `GET /api/model` now
publishes them, and a **Model card** view renders them: headline metrics, the
melanoma operating point, a per-class table carrying the live decision
thresholds, the confusion matrix, the serving configuration read from the running
process, and the limits.

**The page leads with whether the figures are even about this model.** Publishing
measured numbers beside a configuration they were not measured on would make the
page worse than not having one, so the endpoint compares the thresholds recorded
in `docs/evaluation_serving_check.json` against the ones the server is actually
using and answers in three states: they match, they differ (naming which classes
and both values), or it cannot tell. The interface renders each differently, and
an absent artifact reports as `evaluation_available: false` rather than as
blanks. Nothing is computed in the client beyond percentage formatting.

Two details worth recording. The confusion matrix in `evaluation_results.json` is
a bare 7×7 array with **no class labels of its own**, so the axes are labelled
with the order `/api/model` declares — an assumption that would mislabel every
cell while looking entirely plausible if an evaluation run ever wrote a different
order. It is now pinned by a test to two independent properties of the same file:
each row must sum to that class's recorded support, and each diagonal ratio must
reproduce its recorded recall. And the matrix is shaded per row rather than
globally, because NV is 1000 of 1503 images and a global scale renders every
other row blank.

The container image previously copied no part of `docs/`, so the endpoint would
have reported no evaluation inside it. The Dockerfile now copies the two JSON
artifacts and nothing else — the rest is prose and PDFs.

### The quoted melanoma recall is read, not remembered

Every summary the interface writes — on screen and in the exported report —
states this model's melanoma recall. It was the literal `0.800` typed into
`buildClinicalSummary()`, alongside the claim that it was measured "on a
lesion-disjoint hold-out". A retrain would have moved the real figure and left
that sentence asserting the old one indefinitely, with nothing in the pipeline
able to notice: no test referenced it, and it is presented as a measured
property of the served model.

The predictor now keeps the `per_class_metrics` recorded beside the thresholds in
`models/class_thresholds.json` — a file it already refuses to start without, and
which ships with the weights — and `/api/analyze` returns them as
`operating_point`. Sending it with each result rather than once at startup keeps
the figure tied to the checkpoint that answered that request. The client derives
the "roughly one melanoma in N" gloss from the value rather than restating "one
in five", suppresses it where it would be misleadingly precise (a recall of 0.999
is not "one in 1000"), still reports a recall of zero rather than hiding it, and
drops the sentence entirely when the server reports nothing. Two tests pin the
served value to the file.

The provenance claim was dropped rather than moved. `class_thresholds.json`
records `fitted_on: "calib+val splits (lesion-disjoint)"`, but the metrics stored
next to those thresholds are byte-identical to the 1503-sample test evaluation in
`docs/evaluation_serving_check.json` — so the label describes where the
thresholds were fitted, not where the metrics were measured. The value is
reported; the split is not asserted, because the served artifact does not
establish it. Recorded under Known issues.

### The result panel shows the lesion, and the report is a document

**The analysed image was missing from the result panel.** A diagnosis was
presented with no sight of what produced it, and `downloadClinicalReport()` fed
that same panel to html2pdf — so the exported "clinical report" carried
probabilities, a confidence gauge, and no picture of the lesion anywhere in it.
The panel now leads with the image, refetched from the record rather than reused
from the console, so it is provably the image the model was given and is
unaffected by the display sliders or by loading a different sample afterwards. It
carries its own Grad-CAM toggle, sized onto the image the same way the console's
is, and reports honestly when the API returned no attribution. The results grid
was rebalanced around it: image and classification on the left, the full
probability distribution and the screening readout on the right, summary across
the foot.

**The PDF export was a screenshot of a dark dashboard.** html2pdf was handed the
live results view, so the report was neon-on-near-black, cropped mid-card
wherever the page break fell, and inherited the missing image. It now builds
`#print-report`, a separate light document laid out for paper: header with
session id and generation time, the submitted image beside its Grad-CAM,
classification with a risk flag, the full seven-class probability table, the
input-screening statistics and calibration state, the derived summary, and a
disclaimer stating plainly that the system measures nothing about the lesion and
that an unflagged result is not a clearance. Sections are marked
`page-break-inside: avoid`, so one may move to the next page but is never sliced
across it. The `window.print()` fallback prints the same node, so both routes
produce the same document.

Two things this exposed. **html2pdf silently returns a blank page for an
absolutely or fixed positioned source element** — measured as a 3 KB PDF against
728 KB for identical content in normal flow, with no error thrown either way. The
report therefore stays statically positioned and is hidden by
`#print-report-clip`, a zero-height `overflow: hidden` parent, which leaves the
document's own height untouched. And **the capture has to wait for the images**:
`whenReportImagesSettle()` resolves only once every figure has loaded or failed,
because capturing earlier produced a report with empty frames where the lesion
should have been.

### The comparison view became a study view

It previously took two arbitrary image uploads and placed them side by side with
nothing else — no prediction, no probabilities, no record. That is the weakest
possible version of the feature, and it was structurally stuck there: an upload
has not been through the model, so there was never anything to display beside it.
The view could show pictures it had no output for, or output it had no picture
for, but never both at once.

It now compares two **recorded scans**, picked from history. Both stored images
are shown, and with them everything the database holds about each: per-class
softmax with the arithmetic difference B − A on a diverging scale, predicted
class and confidence, high-risk flag, melanoma alert and p(mel), the decision
threshold, anatomic site, timestamps, and whether the image file is still
retained. Zoom can be linked across the pair or driven independently.

Two endpoints made this possible, both gated by the existing history read guard:

* `GET /api/history/{id}` — one record, with the fields the list omitted
  (`threshold_used`, `melanoma_probability`) plus `has_image`.
* `GET /api/history/{id}/image` — the stored upload. There was previously **no
  way at all** to retrieve a historical scan's image; `syncActiveScanToB()`
  referenced an `image_url` the API has never returned, so that branch was dead.
  The path comes from the database and is resolved through
  `resolve_stored_path()`, which refuses anything outside the upload directory.
  A record whose file retention or the orphan sweep has removed returns 404, and
  the client renders the record's numbers with a placeholder where the picture
  was.

The list and the detail response are produced by one serialiser
(`_session_summary`) so the pickers and the comparison cannot disagree about a
record. Seven tests cover the pair, including the missing-file path and the read
guard.

**What it deliberately still refuses to say.** Nothing in the schema links a scan
to a lesion or a patient, so the application cannot know whether two records show
the same lesion, and it does not imply that it does. Every difference shown is a
difference between two model outputs — no growth, diameter, pigmentation or
border metric is computed anywhere in this codebase, the elapsed time is labelled
as the gap between two database rows rather than lesion age, and the view says in
plain text that a probability moving between two scans can reflect lighting,
framing, focus or scale. This is the line the removed "differential engine"
crossed when it derived lesion growth from the length of a base64 string; adding
real data to the view is not a licence to re-cross it.

### Defects found during the redesign

**Three result fields were fixed placeholders that nothing ever updated.** The
markup shipped `SESSION: SCN-2026-LIVE`, an anatomic site of `Anterior Torso` and
a model latency of `142ms`, and no code path wrote to any of them — so every
scan, whatever its site, reported the same three values. All three now come from
the API response. The threshold is per-class rather than global, so the predicted
class is named beside it (a bare `0.000` reads as a missing value rather than a
permissive threshold).

**`/api/analyze` now returns its own timings.** `inference_ms` and `queue_ms`
were already measured in `backend/routers/diagnostics.py` and fed to the metrics
endpoint, but were not in the response, which is why the panel had nothing real
to show. They are returned — and displayed — as two separate fields, for the same
reason they are measured separately: summed, "inference" would track concurrency
rather than the model. Covered by
`test_analyze_reports_its_own_timings`. `DiagnosticSession` has no column for
either, so history rows and the CSV export still do not carry them.

**The default console image was a remote asset.** `console-viewport-img` pointed
at a Google-hosted composite figure — a dermoscopy teaching plate with black
bars and annotation text baked in, not a lesion photograph. Pressing *Run
analysis* on a fresh page load therefore submitted it and the colour gate
rejected it as blue/green-dominated. It now points at a local HAM10000 sample.

**The Grad-CAM overlay did not cover the image it explained.** The canvas carried
intrinsic 224×224 dimensions and only `max-h-full max-w-full`, so it rendered as
a 224px square floating in the middle of a 600×450 photograph — the attribution
hot spot pointed at the wrong part of the lesion. The canvas is now measured onto
the image's rendered content box, and re-measured on load, on resize and when the
overlay is toggled.

**The analysis loading overlay was dead markup.** `#analysis-overlay` existed but
nothing unhid it, so a scan produced no feedback over the image. It is now shown
for the duration of the request.

**The source chip went stale on upload.** It kept naming the last reference sample
after an upload replaced it, so the toolbar described an image that was no longer
on screen.

---

### Fabricated clinical output removed

Three separate places presented invented numbers as analysis. This was the most
serious category of defect in the repository, because the project's entire
premise is honest measurement.

**The comparison view fabricated progression findings.**
`runDifferentialAnalysis()` derived "lesion area growth", "confidence index",
pigmentation density and border irregularity from `img.src.length % 25` — the
length of a base64 string, which is a file size, not a measurement of anything.
It then printed verdicts up to *"Recommending urgent biopsy evaluation"* on that
basis, and `renderCanvasDiffOverlay()` stroked a fixed ellipse at a hard-coded
position regardless of the image. It ran automatically on every image load, not
just on the button. Removed in full, along with the fabricated defaults the
markup shipped: invented scan dates, a hardcoded `MEL (Melanoma)` tag on scan B,
and a decorative "delta heatmap". The view now states plainly that it performs
visual comparison only.

**The results panel asserted a clinical recommendation on every scan.**
`res-clinical-summary` was hardcoded in the markup and never written by script:
*"High probability of malignant lesion detected. Immediate histopathological
biopsy and dermatologic evaluation are recommended."* — shown on benign results
too, directly contradicting the "NO MALIGNANT PATTERN IDENTIFIED" banner one
card above it. Reproduced on a 96.8 %-confidence nevus. The summary is now
derived from the prediction, the high-risk flag and the melanoma alert, and
states the measured operating point (recall 0.800) wherever it appears.

**The OOD panel showed constants.** "Skin hue ratio 84.2 %", "blue-green ratio
0.92", "QUALIFIED SCAN" were static markup, identical for every scan; the script
meant to update them keyed off an `ood_metrics` field the API never returned,
and its fallbacks were invented too. The API now returns the statistics the gate
actually computed — they were already being calculated and discarded — and the
panel reports those plus whether the gate's thresholds are fitted at all.

### Frontend correctness

- **Only one view rendered at a time.** `view-console` was missing the
  `view-container` class that `navigate()` uses to hide views, so the Scan
  Console never hid and every other view stacked beneath it.
- **Anatomic site reached the server.** Sample cards carried HAM10000's own
  vocabulary ("Back", "Trunk", "Face"), which the API's six-value allowlist
  discarded — the site was recorded as null and every tag button deselected.
  Now mapped to the accepted vocabulary.
- **History timestamps.** The API emitted naive UTC that browsers parsed as
  local time, and the table mixed a UTC date with a local time in one row.
- **Report export.** The always-visible export button captured the results panel
  even when hidden and empty, producing a blank PDF presented as a clinical
  report.

### Serving configuration is no longer silently wrong

- **A missing `calibration.json` is now fatal.** It previously fell through to
  the config defaults — sigmoid instead of softmax, no temperature scaling, and
  the melanoma alert channel switched off — a different decision rule from the
  one every published metric describes, announced only by a `print()`.
  `ALLOW_UNCALIBRATED=1` is the deliberate escape hatch.
- **`/api/health` reports what is actually in force**: readout, temperature,
  melanoma alert threshold, OOD calibration state, concurrency settings and
  which hardening is enabled, plus a `degraded` list. It still does not load the
  model.
- **Internal errors no longer leak.** `HTTPException(500, detail=str(e))`
  forwarded absolute build paths, usernames and library internals to an
  unauthenticated endpoint. Callers now receive only a reference, which is the
  request id stamped on every log line the request produced.

### Concurrency

- **Inference no longer blocks the event loop.** `/api/analyze` was `async def`
  but called blocking CPU-bound inference directly. Measured with 12 concurrent
  scans while polling `/api/health` every 50 ms:

  | | before | after |
  |---|---|---|
  | health probes completed | 7 | 257 |
  | health max latency | 12 531 ms | 44 ms |

  The before-p95 looked fine only because seven samples got through; one probe
  sat blocked for the entire duration of every scan.

- **The model loads once.** `get_predictor()` had no lock, so concurrent
  cold-start requests could each build a predictor and read the 41 MB checkpoint.
- **Threads no longer oversubscribe.** Torch defaulted to 10 threads per
  inference against a concurrency limit of 2 — 20 threads on 12 cores.
  `TORCH_NUM_THREADS = cpu_count // MAX_CONCURRENT_INFERENCE`. Measured at
  6-way concurrency: 60 threads 14.66 s → 12 threads 12.20 s (+16.8 %). At the
  default concurrency of 2 the improvement was within noise.
- In-flight inference is bounded by a semaphore so overload queues instead of
  thrashing.

### Storage and the database

- **SQLite survives concurrent access**: WAL journal, `busy_timeout=5000`,
  `synchronous=NORMAL`.
- **Image paths are stored relative to the upload directory.** Absolute host
  paths meant moving, renaming or containerising the install orphaned every
  historical image. Migrated 76 existing rows in place; the migration also
  re-homes images from a previous install location.
- **Uploads are written atomically.** The file was previously written before
  inference and deleted afterwards on failure, so a crash in between left it on
  disk permanently. Now written to a `.part` name and published only once the
  owning row commits. A conservative startup sweep reclaims genuine orphans.
- **Optional retention** via `UPLOAD_RETENTION_DAYS`, off by default.

  > During development the orphan sweep deleted eight real images and
  > `data/uploads/.gitkeep`. Both were caught before touching live data — the
  > first by running the migration against a copy of the database, the second in
  > `git status`. The sweep now matches on stored filenames rather than resolved
  > paths (an unresolvable path previously contributed no reference, so its
  > image looked orphaned) and only recognises its own naming. Both failures are
  > covered by regression tests.

### Observability

- Every request carries an id, echoed as `X-Request-ID` and stamped on every log
  line it produces — including the lazy model load it triggers.
- All 11 `print()` calls replaced with levelled, timestamped logging.
- `/api/metrics` reports counters and latency. Inference time and queue wait are
  measured **separately**: conflated, p95 "inference" climbed with queue depth
  and pointed at the wrong bottleneck.

### Hardening and deployment

- Rate limiting per client address, with inference on a tighter budget than
  reads. Verified: 60 analyses allowed, the 61st returns 429 with `Retry-After`,
  and `/api/health` is unaffected.
- Oversized uploads rejected on `Content-Length` before the body is parsed.
- Optional history read guard (`REQUIRE_HISTORY_TOKEN`), off by default because
  the bundled UI reads history unauthenticated.
- `Cache-Control: no-cache` on the SPA's own assets — unversioned `app.js` was
  being heuristically cached, so a deploy could pair stale frontend with a new
  API.
- **CI** (`.github/workflows/ci.yml`) on Python 3.11 and 3.13: lint, tests, and
  a check that the app boots without a checkpoint. Tests needing
  `models/latest.pt` (41 MB, gitignored) skip themselves, so a clean clone runs
  62 of 80 and a red run means a real regression.
- **Dockerfile**, CPU-only, non-root, no host-specific paths. The checkpoint is
  mounted, not baked in — it carries HAM10000's non-commercial licence.
- **Lint gate** (`ruff.toml`) scoped to `F` + `E9`. Ruff's defaults report 158
  findings on this codebase, almost all import ordering; a gate that fails on
  its first run gets switched off. It immediately found a stale import left by
  the logging work.

### Out-of-distribution gate — still not fitted, and now provably so

The feature-space stage has never been calibrated. It **cannot** be fitted from
the 21 images in `samples/`, and attempting it produces a gate that rejects
every scan.

EfficientNet-B3 emits 1536-dimensional features, so the covariance has 1 180 416
free parameters. Fitted from 21 samples it is rank-deficient. Under
leave-one-out, **21 of 21 held-out real lesions were rejected**, at Mahalanobis
distances around 10⁶ against a cutoff of 18 — while the script reported "OOD
correctly rejected 3/3" and "in-distribution wrongly rejected 4.8 %", because it
measured false rejects on the images it had just fitted to. A gate that rejects
everything rejects out-of-distribution input perfectly.

`scripts/calibrate_ood.py` now holds data back from fitting, reports the
false-reject rate on that held-out portion, and **refuses to install** a gate
that rejects more than `--max-false-reject` of it, or that cannot be validated
at all. Fitting this stage requires the HAM10000 image set, which is not in this
repository.

---

### Known and unchanged

- Melanoma recall is **0.800** — one melanoma in five is missed by the
  prediction; the alert channel surfaces 92.9 % at a 23.6 % review rate.
- The OOD gate runs on provisional thresholds with the feature stage inactive.
  `samples/cat.jpg` is still accepted rather than rejected.
- The interface has no ARIA attributes, loads Tailwind and html2pdf from CDNs,
  and keeps the admin token in `localStorage`.
- This is a research prototype and not a medical device.
