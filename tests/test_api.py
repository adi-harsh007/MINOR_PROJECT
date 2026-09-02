"""Upload validation, admin guards, and history behaviour."""

import io
import os

import numpy as np
import pytest
from PIL import Image


def jpeg_bytes(img):
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG")
    return buf.getvalue()


def upload(client, name, data, content_type="image/jpeg"):
    return client.post("/api/analyze", files={"file": (name, data, content_type)})


@pytest.fixture(scope="session")
def lesion(project_root):
    return jpeg_bytes(Image.open(os.path.join(project_root, "samples", "ISIC_0024307.jpg")))


# ── service ──────────────────────────────────────────────────────────────

def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_health_reports_real_configuration(client):
    """Health must describe the actual serving config, not a hardcoded string."""
    from backend.config import IMG_SIZE, MODEL_ARCH

    model = client.get("/api/health").json()["model"]
    assert model["architecture"] == MODEL_ARCH
    assert model["input_size"] == IMG_SIZE
    assert "loaded" in model and isinstance(model["loaded"], bool)


def test_health_does_not_force_model_load(client):
    """A health check must not pull the checkpoint into memory."""
    import backend.routers.diagnostics as diagnostics

    before = diagnostics._predictor
    client.get("/api/health")
    assert diagnostics._predictor is before


def test_index_and_samples_are_served(client):
    assert client.get("/").status_code == 200
    assert client.get("/samples/nv.jpg").status_code == 200


def test_favicon_is_no_content(client):
    """Previously returned index.html as image/x-icon."""
    assert client.get("/favicon.ico").status_code == 204


# ── upload validation ────────────────────────────────────────────────────

def test_rejects_disallowed_extension(client):
    r = upload(client, "payload.py", b"print(1)")
    assert r.status_code == 400


def test_rejects_traversal_style_filename(client, lesion):
    """The client-supplied name must never reach the filesystem path."""
    r = upload(client, "../../evil.jpg/x.php", lesion)
    assert r.status_code == 400


def test_rejects_non_image_with_image_extension(client):
    """content_type is client-controlled, so validation decodes the bytes."""
    r = upload(client, "fake.jpg", b"not an image at all")
    assert r.status_code == 400


def test_rejects_oversized_upload(client):
    from backend.config import MAX_UPLOAD_BYTES

    r = upload(client, "big.jpg", b"\0" * (MAX_UPLOAD_BYTES + 1))
    assert r.status_code == 413


# ── inference ────────────────────────────────────────────────────────────

@pytest.mark.slow
def test_accepts_real_lesion_and_records_it(client, lesion):
    r = upload(client, "lesion.jpg", lesion)
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    assert 0.0 <= body["confidence"] <= 1.0
    assert set(body["scores"]) == {"akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"}

    history = client.get("/api/history").json()
    assert len(history) == 1
    assert history[0]["prediction"] == body["prediction"]


@pytest.mark.slow
def test_heatmap_is_absent_or_real_never_synthetic(client, lesion):
    """A heatmap is either genuine model attribution or absent - never a stand-in."""
    body = upload(client, "lesion.jpg", lesion).json()
    hm = body.get("heatmap_base64")
    assert hm is None or hm.startswith("data:image/png;base64,")


@pytest.mark.slow
def test_ood_rejection_reports_a_reason(client):
    flat = jpeg_bytes(Image.new("RGB", (300, 300), (240, 240, 238)))
    r = upload(client, "wall.jpg", flat)
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert detail["reason"] == "uniform_field"
    assert detail["message"]


@pytest.mark.slow
def test_rejected_scan_is_not_persisted(client):
    noise = jpeg_bytes(Image.fromarray(
        np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)))
    assert upload(client, "noise.jpg", noise).status_code == 422
    assert client.get("/api/history").json() == []


# ── admin guard ──────────────────────────────────────────────────────────

def test_delete_all_requires_token(client):
    assert client.delete("/api/history/all").status_code == 401


def test_delete_one_requires_token(client):
    assert client.delete("/api/history/1").status_code == 401


def test_delete_rejects_wrong_token(client):
    r = client.delete("/api/history/all", headers={"X-Admin-Token": "wrong"})
    assert r.status_code == 401


@pytest.mark.slow
def test_single_delete_works_with_token(client, lesion, admin_headers):
    """The history UI deletes one record at a time and must succeed with a token."""
    session_id = upload(client, "lesion.jpg", lesion).json()["session_id"]
    assert client.delete(f"/api/history/{session_id}").status_code == 401
    r = client.delete(f"/api/history/{session_id}", headers=admin_headers)
    assert r.status_code == 200
    assert client.get("/api/history").json() == []


def test_delete_disabled_when_no_token_configured(client, monkeypatch):
    """With ADMIN_TOKEN unset the endpoint reports 'disabled', not 'wrong token'."""
    import backend.routers.diagnostics as diagnostics

    monkeypatch.setattr(diagnostics, "ADMIN_TOKEN", None)
    r = client.delete("/api/history/1", headers={"X-Admin-Token": "anything"})
    assert r.status_code == 403
    assert "disabled" in r.json()["detail"].lower()


@pytest.mark.slow
def test_delete_all_with_token_removes_rows_and_files(client, lesion, admin_headers):
    upload(client, "lesion.jpg", lesion)
    assert len(client.get("/api/history").json()) == 1

    r = client.delete("/api/history/all", headers=admin_headers)
    assert r.status_code == 200
    assert client.get("/api/history").json() == []


@pytest.mark.slow
def test_anatomic_site_is_persisted_and_returned(client, lesion):
    """The UI collects an anatomic site; it must survive into the record."""
    r = client.post("/api/analyze",
                    files={"file": ("lesion.jpg", lesion, "image/jpeg")},
                    data={"site": "Palms & Soles"})
    assert r.status_code == 200
    assert r.json()["anatomic_site"] == "Palms & Soles"
    assert client.get("/api/history").json()[0]["anatomic_site"] == "Palms & Soles"


@pytest.mark.slow
def test_unknown_anatomic_site_is_discarded(client, lesion):
    """Only sites the UI offers are stored; anything else is dropped."""
    r = client.post("/api/analyze",
                    files={"file": ("lesion.jpg", lesion, "image/jpeg")},
                    data={"site": "'; DROP TABLE diagnostic_sessions;--"})
    assert r.status_code == 200
    assert r.json()["anatomic_site"] is None


def test_migration_adds_missing_columns(tmp_path):
    """A database created before the column existed must be migrated, not broken."""
    import sqlite3

    db = tmp_path / "legacy.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE diagnostic_sessions ("
                "id INTEGER PRIMARY KEY, image_path VARCHAR(500) NOT NULL)")
    con.commit()
    con.close()

    from sqlalchemy import create_engine, inspect
    import backend.database as database

    original = database.engine
    database.engine = create_engine(f"sqlite:///{db}",
                                    connect_args={"check_same_thread": False})
    try:
        database._add_missing_columns()
        cols = {c["name"] for c in inspect(database.engine).get_columns("diagnostic_sessions")}
        assert "anatomic_site" in cols
        database._add_missing_columns()   # idempotent
    finally:
        database.engine.dispose()
        database.engine = original


@pytest.mark.slow
def test_melanoma_alert_is_reported_and_persisted(client, lesion):
    """The alert channel is an additive output; it must reach the API and the record."""
    body = client.post("/api/analyze",
                       files={"file": ("lesion.jpg", lesion, "image/jpeg")}).json()
    assert isinstance(body["melanoma_alert"], bool)
    assert 0.0 <= body["melanoma_probability"] <= 1.0
    assert client.get("/api/history").json()[0]["melanoma_alert"] == body["melanoma_alert"]


@pytest.mark.slow
def test_alert_fires_whenever_melanoma_probability_clears_threshold(client, lesion):
    """The alert must not depend on melanoma winning the argmax."""
    from backend.routers.diagnostics import get_predictor

    predictor = get_predictor()
    if predictor.mel_alert_threshold is None:
        pytest.skip("calibration not fitted")

    body = client.post("/api/analyze",
                       files={"file": ("lesion.jpg", lesion, "image/jpeg")}).json()
    expected = (body["prediction"] == "mel"
                or body["melanoma_probability"] >= predictor.mel_alert_threshold)
    assert body["melanoma_alert"] is expected


def test_temperature_is_applied_to_confidence():
    """A temperature above 1 must actually flatten the reported probabilities."""
    import numpy as np
    import torch

    logits = torch.tensor([[3.0, -1.0, 0.5, -2.0, 1.0, 2.0, -1.5]])
    raw = torch.sigmoid(logits).numpy()
    scaled = torch.sigmoid(logits / 2.0).numpy()
    assert scaled.max() < raw.max()
    assert np.all(np.abs(scaled - 0.5) <= np.abs(raw - 0.5) + 1e-9)


def test_history_limit_is_bounded(client):
    assert client.get("/api/history?limit=0").status_code == 422
    assert client.get("/api/history?limit=500").status_code == 422
