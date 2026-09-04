from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, Boolean
from datetime import datetime, timezone
from .database import Base


class DiagnosticSession(Base):
    __tablename__ = "diagnostic_sessions"

    id = Column(Integer, primary_key=True, index=True)
    image_path = Column(String(500), nullable=False)
    anatomic_site = Column(String(64), nullable=True)
    # Operator-assigned grouping: which lesion this scan is of. Nothing infers
    # it and nothing validates it against reality - it records an assertion the
    # person making it is responsible for, which is exactly why the comparison
    # view attributes it to them rather than presenting it as a finding.
    case_label = Column(String(64), nullable=True, index=True)
    status = Column(String(50), default="pending")  # pending | processing | completed | failed

    # Results
    prediction = Column(String(50), nullable=True)
    confidence = Column(Float, nullable=True)
    threshold_used = Column(Float, nullable=True)
    all_scores = Column(JSON, nullable=True)
    is_high_risk = Column(Boolean, default=False)
    melanoma_alert = Column(Boolean, default=False)
    melanoma_probability = Column(Float, nullable=True)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    completed_at = Column(DateTime, nullable=True)
