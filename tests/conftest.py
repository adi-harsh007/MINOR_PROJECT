"""Test fixtures.

Every test runs against a throwaway database and upload directory. This is not
optional hygiene: the API exposes bulk-delete endpoints, and pointing a test run
at the real database once already destroyed stored diagnostic sessions and their
images permanently.
"""

import os
import sys
import tempfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Redirect storage before any application module is imported, since paths and the
# engine are resolved at import time.
_TMP = tempfile.mkdtemp(prefix="dermascan-tests-")
os.environ["DATABASE_URL"] = "sqlite:///" + os.path.join(_TMP, "test.db").replace("\\", "/")
os.environ.setdefault("ADMIN_TOKEN", "test-admin-token")

import backend.config as config  # noqa: E402

config.UPLOAD_DIR = os.path.join(_TMP, "uploads")
os.makedirs(config.UPLOAD_DIR, exist_ok=True)

import backend.routers.diagnostics as diagnostics  # noqa: E402
from backend.database import Base, engine  # noqa: E402

diagnostics.UPLOAD_DIR = config.UPLOAD_DIR


@pytest.fixture(scope="session", autouse=True)
def _schema():
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture(autouse=True)
def _clean_tables():
    """Each test starts with an empty history."""
    from backend.database import SessionLocal
    from backend.models import DiagnosticSession

    db = SessionLocal()
    db.query(DiagnosticSession).delete()
    db.commit()
    db.close()
    yield


@pytest.fixture(scope="session")
def db_path():
    return _TMP


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient
    from backend.main import app

    with TestClient(app) as c:
        yield c


@pytest.fixture(scope="session")
def admin_headers():
    return {"X-Admin-Token": os.environ["ADMIN_TOKEN"]}


@pytest.fixture(scope="session")
def project_root():
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
