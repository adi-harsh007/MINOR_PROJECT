# Backend Configuration
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{os.path.join(BASE_DIR, 'data', 'pathology.db')}")

MODEL_PATH = os.path.join(BASE_DIR, "models", "latest.pt")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "class_thresholds.json")

# Allow environment overrides but fallback to verified structure
MODEL_PATH = os.getenv("MODEL_PATH", MODEL_PATH)
THRESHOLD_PATH = os.getenv("THRESHOLD_PATH", THRESHOLD_PATH)

UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
