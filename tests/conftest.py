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

import backend.config as config_module  # noqa: E402
config = config_module

config.UPLOAD_DIR = os.path.join(_TMP, "uploads")
os.makedirs(config.UPLOAD_DIR, exist_ok=True)

import backend.routers.diagnostics as diagnostics  # noqa: E402
from backend.database import Base, engine  # noqa: E402

diagnostics.UPLOAD_DIR = config.UPLOAD_DIR


def pytest_collection_modifyitems(config, items):
    """Skip checkpoint-dependent tests when the checkpoint is not present.

    models/latest.pt is 41 MB and gitignored, so a clean clone - CI included -
    does not have it. Everything that does not need the network still runs
    there: upload validation, storage, the OOD statistics, admin guards, and
    the observability contract. Tests that genuinely need to run inference are
    skipped with a reason rather than failing, so a red CI run means a real
    regression.
    """
    if os.path.exists(config_module.MODEL_PATH):
        return
    skip = pytest.mark.skip(
        reason="needs models/latest.pt (41 MB, gitignored); "
               "run scripts/deploy_checkpoint.py to install it")
    for item in items:
        if "slow" in item.keywords or "requires_model" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session", autouse=True)
def _schema():
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture(autouse=True)
def _clean_process_state():
    """Reset in-process counters and rate-limit windows between tests.

    Both are module-level and would otherwise carry across the suite: the rate
    limiter would eventually reject a later test's request because of requests
    an earlier one made, which is a flake that only appears as the suite grows.
    """
    from backend import metrics, ratelimit

    metrics.reset()
    ratelimit.reset()
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
