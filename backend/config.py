# Backend Configuration
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'data', 'pathology.db')}")

# Served checkpoint. models/latest.pt is the plain-timm network whose measured
# test-set performance is what the docs report. See backend/model.py.
MODEL_PATH = os.path.join(BASE_DIR, "models", "latest.pt")

# Architecture of MODEL_PATH: "plain" or "multihead". Selected explicitly so a
# mismatched checkpoint fails loudly rather than loading a different network.
MODEL_ARCH = os.getenv("MODEL_ARCH", "plain")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "class_thresholds.json")

# Allow environment overrides but fallback to verified structure
MODEL_PATH = os.getenv("MODEL_PATH", MODEL_PATH)
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", THRESHOLD_PATH)

# Inference resolution. Must match configs/default.yaml img_size in the training
# repository; the model was trained and evaluated at 300px with no centre crop.
IMG_SIZE = int(os.getenv("IMG_SIZE", 300))

UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Static asset directories ────────────────────────────────
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")

# ── Upload constraints ──────────────────────────────────────
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", 10 * 1024 * 1024))  # 10 MB

# Anatomic sites offered by the UI (data-site attributes in index.html).
# Anything else submitted is stored as None rather than trusted.
ANATOMIC_SITES = {
    "Head & Neck",
    "Anterior Torso",
    "Posterior Torso",
    "Upper Extremities",
    "Lower Extremities",
    "Palms & Soles",
}

# ── Security ────────────────────────────────────────────────
# Comma-separated list of allowed browser origins. Defaults to local dev only.
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv(
        "CORS_ORIGINS",
        "http://localhost:8088,http://127.0.0.1:8088",
    ).split(",")
    if o.strip()
]

# Required to call destructive endpoints. When unset, those endpoints are disabled.
ADMIN_TOKEN = os.getenv("ADMIN_TOKEN") or None
