from fastapi import APIRouter, HTTPException, UploadFile, File, Query, Depends
from sqlalchemy.orm import Session
from datetime import datetime
import os
import uuid
from PIL import Image
from io import BytesIO

from ..database import get_db
from ..models import DiagnosticSession
from ..config import UPLOAD_DIR
from ..ml_engine import SkinCancerPredictor

router = APIRouter(prefix="/api", tags=["Diagnostics"])

_predictor = None

def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = SkinCancerPredictor()
    return _predictor


# ─── Analyze ────────────────────────────────────────────────

@router.post("/analyze")
async def analyze_lesion(
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
):
    """Upload an image and run synchronous EfficientNet-B3 inference."""
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(400, detail="Invalid file type. Upload an image (jpg/png).")

    ext = file.filename.split(".")[-1] if "." in file.filename else "jpg"
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(UPLOAD_DIR, filename)

    contents = await file.read()
    with open(filepath, "wb") as f:
        f.write(contents)

    try:
        predictor = get_predictor()
        image = Image.open(BytesIO(contents))
        result = predictor.predict(image)

        if result.get("is_ood"):
            # Clean up the file and reject
            if os.path.exists(filepath):
                os.remove(filepath)
            raise HTTPException(422, detail="Invalid Scan Detected: Image does not appear to be a clinical skin lesion.")

        # Create session only for valid scans
        session = DiagnosticSession(
            image_path=filepath, 
            status="completed",
            prediction=result["prediction"],
            confidence=result["confidence"],
            threshold_used=result["threshold"],
            all_scores=result["scores"],
            is_high_risk=result["prediction"] in ["mel", "bcc", "akiec"],
            completed_at=datetime.utcnow()
        )
        db.add(session)
        db.commit()
        db.refresh(session)

        return {
            "session_id": session.id,
            "prediction": result["prediction"],
            "confidence": result["confidence"],
            "threshold": result["threshold"],
            "scores": result["scores"],
            "is_high_risk": session.is_high_risk,
        }
    except HTTPException:
        raise
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise HTTPException(500, detail=str(e))


# ─── History ────────────────────────────────────────────────

@router.get("/history")
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
            "scores": s.all_scores,
            "created_at": s.created_at.isoformat() if s.created_at else None,
        }
        for s in sessions
    ]


@router.delete("/history/all")
def delete_all_history(db: Session = Depends(get_db)):
    """Delete ALL diagnostic sessions and their uploaded images."""
    sessions = db.query(DiagnosticSession).all()
    count = len(sessions)
    for s in sessions:
        if s.image_path and os.path.exists(s.image_path):
            os.remove(s.image_path)
        db.delete(s)
    db.commit()
    return {"message": f"Deleted {count} sessions"}

@router.delete("/history/{session_id}")
def delete_history_entry(session_id: int, db: Session = Depends(get_db)):
    """Delete a single diagnostic session by ID."""
    session = db.query(DiagnosticSession).filter(DiagnosticSession.id == session_id).first()
    if not session:
        raise HTTPException(404, detail="Session not found")

    # Delete the uploaded image file
    if session.image_path and os.path.exists(session.image_path):
        os.remove(session.image_path)

    db.delete(session)
    db.commit()
    return {"message": "Deleted", "id": session_id}
