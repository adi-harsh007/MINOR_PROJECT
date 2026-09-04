import json
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response, JSONResponse
import uvicorn

from .config import (CORS_ORIGINS, FRONTEND_DIR, SAMPLES_DIR, MODEL_PATH,
                     MODEL_ARCH, IMG_SIZE, THRESHOLD_PATH,
                     MAX_CONCURRENT_INFERENCE, TORCH_NUM_THREADS,
                     MAX_UPLOAD_BYTES, RATE_LIMIT_PER_MINUTE,
                     ANALYZE_RATE_LIMIT_PER_MINUTE, REQUIRE_HISTORY_TOKEN,
                     UPLOAD_RETENTION_DAYS, EVALUATION_PATH, SERVING_CHECK_PATH)
from .database import init_db
from .routers import diagnostics
from . import metrics, ratelimit
from .logging_setup import (get_logger, adopt_request_id, set_request_id,
                            reset_request_id)

log = get_logger("app")
access_log = get_logger("access")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    log.info("database ready; serving %s at %dpx, decision layer %s",
             MODEL_ARCH, IMG_SIZE,
             "calibrated" if os.path.exists(
                 os.path.join(os.path.dirname(MODEL_PATH), "calibration.json"))
             else "UNCALIBRATED")
    yield
    log.info("shutting down")


app = FastAPI(
    title="DermaScan AI — Diagnostic Engine",
    description="EfficientNet-B3 skin cancer classification API",
    version="3.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "DELETE"],
    allow_headers=["*"],
)

@app.middleware("http")
async def guard_request(request, call_next):
    """Reject oversized and over-frequent requests before they cost anything.

    The size check reads Content-Length and answers before the multipart parser
    runs. The endpoint's own check still exists and still matters - a chunked
    upload declares no length - but by then Starlette has already spooled the
    whole body, so a caller could make the server buffer a gigabyte to be told
    the limit is ten megabytes.
    """
    if request.url.path.startswith("/api/"):
        declared = request.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > MAX_UPLOAD_BYTES:
            metrics.incr("rejected_oversized")
            return JSONResponse(
                status_code=413,
                content={"detail": "Image exceeds the {} MB upload limit.".format(
                    MAX_UPLOAD_BYTES // (1024 * 1024))},
            )

        client = request.client.host if request.client else "unknown"
        allowed, retry_after, limit = ratelimit.check(client, request.url.path)
        if not allowed:
            metrics.incr("rate_limited")
            access_log.warning("rate limit hit: client=%s path=%s limit=%d/min",
                               client, request.url.path, limit)
            return JSONResponse(
                status_code=429,
                headers={"Retry-After": str(retry_after)},
                content={"detail": "Too many requests. Try again in {}s.".format(
                    retry_after)},
            )

    return await call_next(request)


@app.middleware("http")
async def request_context(request, call_next):
    """Tag every request with an id and record how it went.

    The id is echoed as `X-Request-ID` and stamped on every log line the request
    produces, so a user-reported failure reference leads straight to its
    traceback. API requests are logged with their status and duration; static
    asset fetches are not, since they would drown everything else.
    """
    request_id = adopt_request_id(request.headers.get("x-request-id"))
    token = set_request_id(request_id)
    is_api = request.url.path.startswith("/api/")
    started = time.perf_counter()
    try:
        try:
            response = await call_next(request)
        except Exception:
            elapsed = (time.perf_counter() - started) * 1000
            metrics.incr("requests_unhandled_error")
            access_log.exception("%s %s -> unhandled exception after %.0fms",
                                 request.method, request.url.path, elapsed)
            raise

        elapsed = (time.perf_counter() - started) * 1000
        response.headers["X-Request-ID"] = request_id
        if is_api:
            metrics.incr("requests_total")
            if response.status_code >= 500:
                metrics.incr("requests_5xx")
            elif response.status_code >= 400:
                metrics.incr("requests_4xx")
            access_log.info("%s %s -> %d in %.0fms",
                            request.method, request.url.path,
                            response.status_code, elapsed)
        return response
    finally:
        # Reset last: the access log above must still carry the id.
        reset_request_id(token)


@app.middleware("http")
async def no_stale_frontend(request, call_next):
    """Force revalidation of the SPA's own assets.

    The frontend is served unversioned (`/` and `/js/app.js`), and neither
    FileResponse nor StaticFiles sets Cache-Control. Browsers then apply
    heuristic freshness and can keep serving an old app.js for hours after a
    deploy, silently pairing stale frontend code with a new API. Both already
    send an ETag, so `no-cache` costs one conditional request and returns 304
    when nothing changed.
    """
    response = await call_next(request)
    path = request.url.path
    if path == "/" or path.startswith("/js/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


app.include_router(diagnostics.router)

# Serve frontend and the reference lesion images (paths are absolute so the
# server can be started from any working directory).
app.mount("/js", StaticFiles(directory=os.path.join(FRONTEND_DIR, "js")), name="js")
app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


@app.get("/")
def serve_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return Response(status_code=204)


@app.get("/api/health")
def health_check():
    """Reports the actual serving configuration.

    Deliberately does not touch the predictor: the model is loaded lazily on the
    first analysis, and a health check should not pull 123 MB into memory. It
    reports whether that load has happened rather than triggering it.

    The decision layer and OOD gate are read from their config files rather than
    from a live predictor, for the same reason. Reporting them matters: the
    difference between a calibrated and an uncalibrated deployment is a different
    decision rule and a disabled melanoma alert channel, and that was previously
    invisible to anything but stdout.
    """
    from .routers import diagnostics
    from .ml_engine import decision_config
    from .ood import OOD_CONFIG_PATH, OOD_STATS_PATH

    decision = decision_config()
    ood = {
        "thresholds_fitted": os.path.exists(OOD_CONFIG_PATH),
        "feature_stage_fitted": os.path.exists(OOD_STATS_PATH),
    }

    degraded = []
    if not decision["calibration_loaded"]:
        degraded.append("decision_layer_uncalibrated")
    if not ood["thresholds_fitted"]:
        degraded.append("ood_thresholds_provisional")
    if not ood["feature_stage_fitted"]:
        degraded.append("ood_feature_stage_inactive")
    if not os.path.exists(MODEL_PATH):
        degraded.append("checkpoint_missing")

    return {
        "status": "ok",
        "version": app.version,
        "degraded": degraded,
        "model": {
            "architecture": MODEL_ARCH,
            "checkpoint": os.path.basename(MODEL_PATH),
            "checkpoint_present": os.path.exists(MODEL_PATH),
            "thresholds_present": os.path.exists(THRESHOLD_PATH),
            "input_size": IMG_SIZE,
            "loaded": diagnostics._predictor is not None,
        },
        "decision_layer": decision,
        "ood_gate": ood,
        "concurrency": {
            "max_concurrent_inference": MAX_CONCURRENT_INFERENCE,
            "torch_num_threads": TORCH_NUM_THREADS,
        },
        # What is actually switched on. Several of these default to off because
        # enabling them changes behaviour the bundled UI depends on; a public
        # deployment should be able to see at a glance which are not.
        "hardening": {
            "rate_limit_per_minute": RATE_LIMIT_PER_MINUTE,
            "analyze_rate_limit_per_minute": ANALYZE_RATE_LIMIT_PER_MINUTE,
            "max_upload_mb": MAX_UPLOAD_BYTES // (1024 * 1024),
            "history_requires_token": REQUIRE_HISTORY_TOKEN,
            "delete_endpoints_enabled": bool(os.getenv("ADMIN_TOKEN")),
            "upload_retention_days": UPLOAD_RETENTION_DAYS or None,
        },
    }


def _read_json(path):
    """Optional artifact. Absent or unreadable is a reportable state, not a crash."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _serving_check_thresholds(check):
    """The thresholds the recorded evaluation was measured under."""
    if not isinstance(check, dict):
        return None
    thresholds = check.get("thresholds")
    return thresholds if isinstance(thresholds, dict) else None


@app.get("/api/model")
def model_card():
    """What this deployment is serving, and what it measured.

    The repository's whole premise is honest measurement, and until now none of
    it reached the interface: accuracy, per-class recall and the confusion matrix
    sat in docs/ where only someone reading the source would find them, while the
    UI stated a single recall figure hardcoded in JavaScript.

    Everything here is read from files. Nothing is computed, rounded up or filled
    in, and a missing artifact is reported as missing.

    Deliberately does not touch the predictor, for the same reason /api/health
    does not: the checkpoint is loaded lazily on the first analysis and a page
    view should not pull it into memory. The thresholds come from the same file
    the predictor reads.
    """
    from .ml_engine import decision_config, read_threshold_file

    classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

    thresholds = None
    threshold_metrics = None
    thresholds_fitted_on = None
    try:
        thresholds, threshold_metrics, thresholds_fitted_on = read_threshold_file(
            THRESHOLD_PATH, classes)
    except Exception as e:
        log.warning("model card: thresholds unreadable at %s: %s", THRESHOLD_PATH, e)

    evaluation = _read_json(EVALUATION_PATH)
    check = _read_json(SERVING_CHECK_PATH)

    # Does the recorded evaluation describe what this server is actually doing?
    #
    # Publishing figures beside a model they were not measured on is the failure
    # this guards against: the numbers would look authoritative and be about a
    # different decision rule. Comparable only when both the live thresholds and
    # the thresholds the evaluation ran under are known - otherwise the answer is
    # "unknown", which the UI must be able to say.
    recorded_thresholds = _serving_check_thresholds(check)
    threshold_mismatches = []
    describes_this_configuration = None
    if thresholds and recorded_thresholds:
        for cls in classes:
            live = thresholds.get(cls)
            recorded = recorded_thresholds.get(cls)
            if live is None or recorded is None or abs(float(live) - float(recorded)) > 1e-9:
                threshold_mismatches.append({
                    "class": cls, "serving": live, "evaluated_with": recorded,
                })
        describes_this_configuration = not threshold_mismatches

    return {
        "architecture": MODEL_ARCH,
        "backbone": "EfficientNet-B3",
        "input_size": IMG_SIZE,
        "classes": classes,
        "dataset": "HAM10000",
        "checkpoint": {
            "name": os.path.basename(MODEL_PATH),
            "present": os.path.exists(MODEL_PATH),
            "loaded": diagnostics._predictor is not None,
        },
        "decision_layer": decision_config(),
        "thresholds": thresholds,
        # Precision, recall and F1 as recorded beside each threshold. Note that
        # `thresholds_fitted_on` describes where the thresholds were fitted, not
        # necessarily the split these metrics were measured on - see the note in
        # docs/FEATURE_STATUS.md. The label is passed through verbatim rather
        # than reinterpreted here.
        "threshold_metrics": threshold_metrics,
        "thresholds_fitted_on": thresholds_fitted_on,
        "evaluation": evaluation,
        "evaluation_available": evaluation is not None,
        "evaluation_describes_this_configuration": describes_this_configuration,
        "evaluation_threshold_mismatches": threshold_mismatches,
    }


@app.get("/api/metrics")
def metrics_endpoint():
    """Counters and inference latency for this process.

    In-memory and per-process: it resets on restart and does not aggregate
    across workers. Enough to answer "how slow is inference right now, and what
    is the gate rejecting?" without standing up a metrics stack.
    """
    return metrics.snapshot()


if __name__ == "__main__":
    uvicorn.run("backend.main:app", host="0.0.0.0", port=8088, reload=True)
