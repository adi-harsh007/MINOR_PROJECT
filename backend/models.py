from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean
from datetime import datetime, timezone
from .database import Base


class DiagnosticSession(Base):
    __tablename__ = "diagnostic_sessions"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String(500), nullable=False)
    status = Column(String(50), default="pending")  # pending | processing | completed | failed

    # Results
    prediction = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    threshold_used = Column(Float, nullable=True)
    all_scores = Column(JSON, nullable=True)
    is_high_risk = Column(Boolean, default=False)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
