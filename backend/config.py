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

# Confidence calibration and the melanoma alert channel, fitted by
# scripts/fit_calibration.py. Defaults below apply when the file is absent.
CALIBRATION_PATH = os.getenv(
    "CALIBRATION_PATH", os.path.join(BASE_DIR, "models", "calibration.json"))

# Temperature > 1 flattens over-confident probabilities. 1.0 disables it.
DEFAULT_TEMPERATURE = float(os.getenv("TEMPERATURE", 1.0))

# Flag "melanoma not excluded" when p(mel) reaches this, whatever class wins.
# None disables the alert channel.
DEFAULT_MEL_ALERT_THRESHOLD = None

# Probability readout. Must match what the thresholds were fitted under: the
# current checkpoint was fitted with sigmoid, a softmax-trained model exports
# "softmax" in its calibration file and this follows it.
DEFAULT_READOUT = os.getenv("READOUT", "sigmoid")

# The defaults above (sigmoid, T=1.0, no melanoma alert) describe NO validated
# configuration. Every published figure was measured with the values in
# models/calibration.json. Serving without that file therefore serves a model
# the documentation does not describe, with the melanoma alert channel silently
# off. Refuse by default; set ALLOW_UNCALIBRATED=1 to serve anyway.
ALLOW_UNCALIBRATED = os.getenv("ALLOW_UNCALIBRATED", "").strip().lower() in {"1", "true", "yes"}

# Inference resolution. Must match configs/default.yaml img_size in the training
# repository; the model was trained and evaluated at 300px with no centre crop.
IMG_SIZE = int(os.getenv("IMG_SIZE", 300))

# Case labels are free text, so unlike the anatomic site there is no vocabulary
# to validate against - only a length the column can hold.
CASE_LABEL_MAX_LENGTH = 64

# Recorded evaluation of the served checkpoint, surfaced by /api/model.
#
# EVALUATION_PATH holds the headline figures; SERVING_CHECK_PATH records the
# thresholds those figures were measured under, which is what lets the endpoint
# say whether the evaluation describes the configuration actually being served
# rather than simply printing numbers next to an unrelated model. Both are
# optional: absent, /api/model reports the serving configuration and says the
# evaluation is unavailable, instead of inventing one.
EVALUATION_PATH = os.getenv(
    "EVALUATION_PATH", os.path.join(BASE_DIR, "docs", "evaluation_results.json"))
SERVING_CHECK_PATH = os.getenv(
    "SERVING_CHECK_PATH",
    os.path.join(BASE_DIR, "docs", "evaluation_serving_check.json"))

UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── Static asset directories ────────────────────────────────
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
SAMPLES_DIR = os.path.join(BASE_DIR, "samples")

# ── Inference concurrency ───────────────────────────────────
# How many analyses may run at once. Inference is CPU-bound, so beyond roughly
# the core count extra concurrency costs latency without adding throughput.
# Requests past this limit queue rather than competing for the same cores.
MAX_CONCURRENT_INFERENCE = max(1, int(os.getenv("MAX_CONCURRENT_INFERENCE", 2)))

# Intra-op threads per inference. Torch defaults to roughly the core count, which
# is right for one inference at a time and wrong for several: with the default,
# MAX_CONCURRENT_INFERENCE simultaneous scans each spawn a full-width thread pool
# and oversubscribe the machine several times over, so every scan gets slower.
# Dividing the cores between the permitted concurrent scans keeps total demand at
# roughly one thread per core. Set TORCH_NUM_THREADS to override.
_CPU_COUNT = os.cpu_count() or 2
TORCH_NUM_THREADS = max(
    1, int(os.getenv("TORCH_NUM_THREADS", _CPU_COUNT // MAX_CONCURRENT_INFERENCE)))

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

# ── Rate limiting ───────────────────────────────────────────
# Requests per minute per client address. 0 disables a bucket. Generous by
# default: the aim is to stop one caller monopolising a single-process,
# CPU-bound service, not to police normal use. Inference gets a tighter budget
# than the read endpoints because it is what actually costs.
RATE_LIMIT_PER_MINUTE = max(0, int(os.getenv("RATE_LIMIT_PER_MINUTE", 240)))
ANALYZE_RATE_LIMIT_PER_MINUTE = max(
    0, int(os.getenv("ANALYZE_RATE_LIMIT_PER_MINUTE", 60)))

# ── Data retention ──────────────────────────────────────────
# Delete diagnostic sessions and their images older than this many days, at
# startup. 0 keeps everything for ever, which is the current documented
# behaviour and stays the default: silently deleting a clinician's records
# would be worse than keeping them.
UPLOAD_RETENTION_DAYS = max(0, int(os.getenv("UPLOAD_RETENTION_DAYS", 0)))

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

# Require ADMIN_TOKEN to *read* history as well as to delete it. Off by default
# because the bundled UI reads history without credentials and would break; turn
# it on for any deployment reachable by anyone but the operator. History rows
# carry predictions and anatomic sites for real uploaded images.
REQUIRE_HISTORY_TOKEN = os.getenv(
    "REQUIRE_HISTORY_TOKEN", "").strip().lower() in {"1", "true", "yes"}
