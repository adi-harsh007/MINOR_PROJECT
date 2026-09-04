# Changelog

## Unreleased — integrity fixes and backend hardening

A defect-driven pass over the whole application. Every figure quoted below was
measured on this machine; nothing here is estimated.

Test suite: **41 → 80 tests**. A lint gate was added. The served model, its
weights and its decision thresholds are unchanged — no retraining happened and
no published metric moved.

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
