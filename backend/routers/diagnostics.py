from fastapi import APIRouter, HTTPException, UploadFile, File, Form, Query, Depends, Header
from sqlalchemy.orm import Session
from datetime import datetime, timezone
import os
import time
import secrets
import threading
from PIL import Image
from io import BytesIO

from ..database import get_db
from ..models import DiagnosticSession
from ..config import (UPLOAD_DIR, ALLOWED_EXTENSIONS, MAX_UPLOAD_BYTES, ADMIN_TOKEN,
                      ANATOMIC_SITES, MAX_CONCURRENT_INFERENCE,
                      REQUIRE_HISTORY_TOKEN)
from ..ml_engine import SkinCancerPredictor
from ..logging_setup import get_logger, get_request_id
from ..storage import (new_upload_name, to_stored_path, delete_stored_file,
                       PARTIAL_SUFFIX)
from .. import metrics

log = get_logger("diagnostics")


def _discard(path):
    """Remove a file this request owns. Absence is not a failure."""
    try:
        os.remove(path)
    except FileNotFoundError:
        pass
    except OSError as e:
        log.warning("could not remove %s: %s", path, e)

router = APIRouter(prefix="/api", tags=["Diagnostics"])

_predictor = None
_predictor_lock = threading.Lock()

# Inference is CPU-bound and single-threaded per call. Without a bound, every
# request that arrives at once gets its own threadpool thread and they all
# contend for the same cores, so latency degrades faster than throughput
# improves. Queueing past this point is preferable to thrashing.
_inference_semaphore = threading.BoundedSemaphore(MAX_CONCURRENT_INFERENCE)


def get_predictor():
    """Build the predictor once, even under concurrent cold-start requests.

    Without the lock, two requests arriving before the first load completed
    would each construct a SkinCancerPredictor: the 41 MB checkpoint read twice,
    two sets of Grad-CAM hooks registered, and whichever finished last silently
    winning. The fast path stays lock-free — reading a module global is atomic
    under the GIL, so only the cold start pays for synchronisation.
    """
    global _predictor
    if _predictor is not None:
        return _predictor
    with _predictor_lock:
        # Re-check inside the lock: another thread may have built it while we waited.
        if _predictor is None:
            _predictor = SkinCancerPredictor()
    return _predictor

def require_admin(x_admin_token: str = Header(default="")):
    """Guards destructive endpoints.

    ADMIN_TOKEN is unset by default, which disables these endpoints entirely
    rather than leaving them open to unauthenticated callers.
    """
    if not ADMIN_TOKEN:
        raise HTTPException(
            403,
            detail="Destructive endpoints are disabled. Set ADMIN_TOKEN to enable them.",
        )
    if not secrets.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(401, detail="Invalid or missing X-Admin-Token header.")



# ─── Analyze ────────────────────────────────────────────────

@router.post("/analyze")
def analyze_lesion(
    file: UploadFile = File(...),
    site: str = Form(default=""),
    db: Session = Depends(get_db),
):
    """Upload an image and run synchronous EfficientNet-B3 inference.

    Deliberately a plain `def`, not `async def`. The body is CPU-bound and
    blocking - a forward pass plus a Grad-CAM backward pass, hundreds of
    milliseconds to seconds on CPU - and FastAPI runs a sync endpoint in a
    worker thread. Declared `async`, that same work ran directly on the event
    loop and stalled every other request in the process, `/api/health`
    included, for the duration of each scan.

    `site` is the anatomic site chosen in the UI. It is recorded alongside the
    result but does not influence inference: the model takes only the image.
    """
    # Only values the UI actually offers are stored; anything else is discarded
    # rather than written through to the record.
    anatomic_site = site.strip() if site and site.strip() in ANATOMIC_SITES else None
    # The client-supplied extension is never trusted as a path component.
    ext = (file.filename or "").rsplit(".", 1)[-1].lower() if "." in (file.filename or "") else ""
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            400,
            detail=f"Invalid file type. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}.",
        )

    # Sync endpoint: read the underlying spooled file directly rather than
    # awaiting UploadFile.read().
    contents = file.file.read()
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413,
            detail=f"Image exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB upload limit.",
        )

    # Validate by decoding, not by the client-declared content type.
    try:
        Image.open(BytesIO(contents)).verify()
    except Exception:
        raise HTTPException(400, detail="File is not a readable image.")

    # Write to a partial name first and commit it only once the record that owns
    # it exists. Previously the final file was written before inference ran, so a
    # crash in between left it on disk forever with nothing referring to it.
    filename = new_upload_name(ext)
    filepath = os.path.join(UPLOAD_DIR, filename)
    partial_path = filepath + PARTIAL_SUFFIX

    with open(partial_path, "wb") as f:
        f.write(contents)

    committed = False   # partial renamed to its final name
    recorded = False    # a row now owns that file
    try:
        predictor = get_predictor()
        image = Image.open(BytesIO(contents))
        # Time the forward pass and the wait for a slot separately. Measured
        # together, p95 "inference" would climb with queue depth and no longer
        # describe how long the model actually takes.
        queued_at = time.perf_counter()
        with _inference_semaphore:
            started = time.perf_counter()
            result = predictor.predict(image)
            inference_ms = (time.perf_counter() - started) * 1000
        queue_ms = (started - queued_at) * 1000
        metrics.observe_inference_ms(inference_ms)
        metrics.observe_queue_ms(queue_ms)

        if result.get("is_ood"):
            # Rejected scans are not persisted, so the partial write is simply
            # discarded by the cleanup in `finally`.
            reason = result.get("reason") or "unspecified"
            metrics.incr("ood_rejections_total")
            metrics.incr("ood_rejection_{}".format(reason))
            log.info("scan rejected by the OOD gate: reason=%s inference=%.0fms",
                     reason, inference_ms)
            detail = result.get("detail") or (
                "Image does not appear to be a clinical skin lesion."
            )
            raise HTTPException(
                422,
                detail={
                    "message": "Invalid Scan Detected: " + detail,
                    "reason": result.get("reason"),
                },
            )

        # The record owns the file from here on, so publish it under its final
        # name before the row referring to it is written.
        os.replace(partial_path, filepath)
        committed = True

        # Create session only for valid scans
        session = DiagnosticSession(
            image_path=to_stored_path(filepath),
            anatomic_site=anatomic_site,
            status="completed",
            prediction=result["prediction"],
            confidence=result["confidence"],
            threshold_used=result["threshold"],
            all_scores=result["scores"],
            is_high_risk=result["prediction"] in ["mel", "bcc", "akiec"],
            melanoma_alert=result.get("melanoma_alert", False),
            melanoma_probability=result.get("melanoma_probability"),
            completed_at=datetime.now(timezone.utc)
        )
        db.add(session)
        db.commit()
        db.refresh(session)
        recorded = True

        metrics.incr("analyses_completed")
        metrics.incr("prediction_{}".format(result["prediction"]))
        if session.melanoma_alert:
            metrics.incr("melanoma_alerts")
        log.info("scan %d: prediction=%s confidence=%.3f high_risk=%s "
                 "melanoma_alert=%s site=%s inference=%.0fms queue=%.0fms",
                 session.id, result["prediction"], result["confidence"],
                 session.is_high_risk, session.melanoma_alert,
                 anatomic_site or "-", inference_ms, queue_ms)

        return {
            "session_id": session.id,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "threshold": result["threshold"],
            "scores": result["scores"],
            "is_high_risk": session.is_high_risk,
            "anatomic_site": session.anatomic_site,
            "melanoma_alert": session.melanoma_alert,
            "melanoma_probability": session.melanoma_probability,
            "heatmap_base64": result.get("heatmap_base64"),
            # Real measurements from the OOD gate, plus its calibration state.
            # The UI reports these; it must never assert a gate result it was
            # not given.
            "ood_metrics": result.get("ood_metrics"),
            "ood_calibrated": result.get("ood_calibrated", False),
            "ood_feature_stage_active": result.get("ood_feature_stage_active", False),
        }
    except HTTPException:
        raise
    except Exception:
        # The exception text is for the operator, not the caller: it carries
        # absolute filesystem paths, checkpoint names and library internals, and
        # this endpoint is unauthenticated. Log the traceback against a reference
        # the user can quote, and return only that reference.
        # The reference is this request's id, so quoting it leads straight to
        # every log line the request produced, not just this traceback.
        reference = get_request_id()
        metrics.incr("analyses_failed")
        log.exception("analyze failed: site=%s file=%s", anatomic_site, filename)
        raise HTTPException(
            500,
            detail=("Analysis failed because of an internal error. "
                    "Quote reference {} when reporting this.".format(reference)),
        )
    finally:
        # Whatever this request wrote and did not hand over to a record is its
        # own to clean up.
        if not committed:
            # Never published - the partial write is all there is.
            _discard(partial_path)
        elif not recorded:
            # Published, then the row failed to commit. Remove it now rather
            # than leaving an image nothing points at.
            _discard(filepath)


# ─── History ────────────────────────────────────────────────

def _utc_isoformat(value):
    """Serialise a timestamp as unambiguous UTC.

    SQLite has no timezone type, so a value written as timezone-aware UTC comes
    back naive. Emitting that bare ("2026-09-04T06:11:00") makes every browser
    read it as *local* time, shifting displayed timestamps by the client's
    offset. Marking it UTC explicitly is what makes the client render it right.
    """
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def require_history_access(x_admin_token: str = Header(default="")):
    """Optional read guard for history.

    Off by default: the bundled UI reads history unauthenticated and would break.
    Any deployment reachable by someone other than the operator should set
    REQUIRE_HISTORY_TOKEN=1, because these rows describe real uploaded images.
    """
    if not REQUIRE_HISTORY_TOKEN:
        return
    if not ADMIN_TOKEN:
        raise HTTPException(
            403,
            detail="History access requires a token but ADMIN_TOKEN is not set.")
    if not secrets.compare_digest(x_admin_token, ADMIN_TOKEN):
        raise HTTPException(401, detail="Invalid or missing X-Admin-Token header.")


@router.get("/history", dependencies=[Depends(require_history_access)])
def get_history(limit: int = Query(50, ge=1, le=200), db: Session = Depends(get_db)):
    """Return completed diagnostic sessions ordered by newest first."""
    sessions = (
        db.query(DiagnosticSession)
        .filter(DiagnosticSession.status == "completed")
        .order_by(DiagnosticSession.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": s.id,
            "prediction": s.prediction,
            "confidence": s.confidence,
            "is_high_risk": s.is_high_risk,
            "anatomic_site": s.anatomic_site,
            "melanoma_alert": s.melanoma_alert,
            "scores": s.all_scores,
            "created_at": _utc_isoformat(s.created_at),
        }
        for s in sessions
    ]


@router.delete("/history/all", dependencies=[Depends(require_admin)])
def delete_all_history(db: Session = Depends(get_db)):
    """Delete ALL diagnostic sessions and their uploaded images."""
    sessions = db.query(DiagnosticSession).all()
    count = len(sessions)
    for s in sessions:
        delete_stored_file(s.image_path)
        db.delete(s)
    db.commit()
    log.info("deleted all %d diagnostic session(s) and their images", count)
    return {"message": f"Deleted {count} sessions"}

@router.delete("/history/{session_id}", dependencies=[Depends(require_admin)])
def delete_history_entry(session_id: int, db: Session = Depends(get_db)):
    """Delete a single diagnostic session by ID."""
    session = db.query(DiagnosticSession).filter(DiagnosticSession.id == session_id).first()
    if not session:
        raise HTTPException(404, detail="Session not found")

    # Delete the uploaded image file. Resolution handles both the current
    # storage-relative form and legacy absolute paths.
    delete_stored_file(session.image_path)

    db.delete(session)
    db.commit()
    return {"message": "Deleted", "id": session_id}
