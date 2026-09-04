"""Upload validation, admin guards, and history behaviour."""

import io
import re
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


# A real dermoscopic lesion from samples/. Named once so that regenerating the
# sample set (scripts/build_test_samples.py) is a one-line change here rather
# than a hunt through the suite - the previous fixture pointed at an image that
# was removed when the samples were redrawn from the current split.
LESION_SAMPLE = "nv_1_ISIC_0032285.jpg"


@pytest.fixture(scope="session")
def lesion(project_root):
    return jpeg_bytes(Image.open(os.path.join(project_root, "samples", LESION_SAMPLE)))


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
    assert client.get("/samples/" + LESION_SAMPLE).status_code == 200


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
def test_operating_point_comes_from_the_served_threshold_file(client, lesion,
                                                              project_root):
    """The recall the UI states must be read, not remembered.

    Every summary the interface writes quotes this model's melanoma recall. It
    used to be the literal 0.800 typed into `buildClinicalSummary()`, which would
    have gone on asserting itself through any retrain with nothing in the
    pipeline able to notice. It is now read out of the same
    models/class_thresholds.json the predictor refuses to start without, so a new
    checkpoint moves the number the client prints.
    """
    import json

    with open(os.path.join(project_root, "models", "class_thresholds.json")) as f:
        shipped = json.load(f)

    body = client.post("/api/analyze",
                       files={"file": ("lesion.jpg", lesion, "image/jpeg")}).json()
    point = body["operating_point"]

    assert point["melanoma_recall"] == shipped["per_class_metrics"]["mel"]["recall"]
    assert point["melanoma_precision"] == shipped["per_class_metrics"]["mel"]["precision"]
    assert point["thresholds_fitted_on"] == shipped.get("fitted_on")


def test_threshold_file_carries_the_metrics_the_ui_quotes(project_root):
    """The recall the interface prints has to be in the file it claims to read.

    Deliberately tests the reader rather than a constructed predictor, so it runs
    in CI: models/latest.pt is gitignored and absent there, but
    models/class_thresholds.json is tracked. A rename or a silently empty dict
    here is exactly how the hardcoded 0.800 would creep back into the summary.
    """
    from backend.ml_engine import read_threshold_file

    classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    thresholds, metrics, fitted_on = read_threshold_file(
        os.path.join(project_root, "models", "class_thresholds.json"), classes)

    assert set(thresholds) == set(classes)
    assert set(metrics) == set(classes)
    assert "recall" in metrics["mel"], "the summary has nothing to quote without this"
    assert 0.0 <= metrics["mel"]["recall"] <= 1.0
    assert isinstance(fitted_on, (str, type(None)))


@pytest.mark.requires_model
def test_predictor_exposes_the_metrics_beside_its_thresholds():
    """And the predictor carries them through to the response builder."""
    from backend.routers.diagnostics import get_predictor

    predictor = get_predictor()
    assert set(predictor.class_metrics) == set(predictor.classes)
    assert "recall" in predictor.class_metrics["mel"]
    assert 0.0 <= predictor.class_metrics["mel"]["recall"] <= 1.0


@pytest.mark.slow
def test_analyze_reports_its_own_timings(client, lesion):
    """The UI shows these two numbers, so they have to be real and separate.

    The results panel previously printed a hardcoded "142ms" that no code path
    ever wrote to, because the response carried no timing at all. Inference and
    queue wait are measured apart on purpose - summed, "inference" would climb
    with concurrency and stop describing the model - so both must survive to the
    client as distinct fields.
    """
    body = client.post("/api/analyze",
                       files={"file": ("lesion.jpg", lesion, "image/jpeg")}).json()

    assert isinstance(body["inference_ms"], (int, float))
    assert isinstance(body["queue_ms"], (int, float))
    # A forward pass cannot take zero time; an unloaded queue legitimately can.
    assert body["inference_ms"] > 0
    assert body["queue_ms"] >= 0


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


# ── record retrieval, for the side-by-side comparison view ───────────────
# The comparison view is built entirely on these two endpoints: without them it
# could only ever show loose uploads it had no model output for, which is what
# it did before.

@pytest.mark.slow
def test_history_detail_carries_everything_the_comparison_shows(client, lesion):
    """A single record must expose every field the comparison view prints."""
    created = client.post("/api/analyze",
                          files={"file": ("lesion.jpg", lesion, "image/jpeg")},
                          data={"site": "Anterior Torso"}).json()

    body = client.get(f"/api/history/{created['session_id']}").json()

    for field in ("id", "prediction", "confidence", "scores", "is_high_risk",
                  "anatomic_site", "melanoma_alert", "melanoma_probability",
                  "threshold_used", "created_at", "has_image"):
        assert field in body, f"{field} missing from the history detail payload"

    assert body["id"] == created["session_id"]
    assert body["prediction"] == created["prediction"]
    assert set(body["scores"]) == set(created["scores"])
    assert body["has_image"] is True


@pytest.mark.slow
def test_history_list_and_detail_agree(client, lesion):
    """Both are rendered by one serialiser; they must not drift apart.

    The list is what the comparison pickers read and the detail is what the
    comparison itself reads, so a field present in one and absent from the other
    shows up as a record that changes when you select it.
    """
    created = client.post("/api/analyze",
                          files={"file": ("lesion.jpg", lesion, "image/jpeg")}).json()
    row = next(r for r in client.get("/api/history").json()
               if r["id"] == created["session_id"])
    assert client.get(f"/api/history/{created['session_id']}").json() == row


def test_history_detail_404s_for_an_unknown_id(client):
    assert client.get("/api/history/99999999").status_code == 404


@pytest.mark.slow
def test_history_image_serves_the_stored_upload(client, lesion):
    created = client.post("/api/analyze",
                          files={"file": ("lesion.jpg", lesion, "image/jpeg")}).json()

    r = client.get(f"/api/history/{created['session_id']}/image")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("image/")
    # Round-trips as a decodable image, not just as bytes.
    Image.open(io.BytesIO(r.content)).verify()


def test_history_image_404s_for_an_unknown_id(client):
    assert client.get("/api/history/99999999/image").status_code == 404


@pytest.mark.slow
def test_history_image_reports_a_missing_file_as_404(client, lesion, monkeypatch):
    """Retention and the orphan sweep outlive the rows that referred to files.

    A record whose image is gone has to say so, not raise. The client uses this
    to show the record's numbers with a placeholder where the picture was.
    """
    import backend.routers.diagnostics as diagnostics

    created = client.post("/api/analyze",
                          files={"file": ("lesion.jpg", lesion, "image/jpeg")}).json()
    monkeypatch.setattr(diagnostics, "resolve_stored_path", lambda stored: None)

    r = client.get(f"/api/history/{created['session_id']}/image")
    assert r.status_code == 404
    assert "no longer stored" in r.json()["detail"]


@pytest.mark.slow
def test_record_endpoints_honour_the_history_read_guard(client, lesion,
                                                        monkeypatch, admin_headers):
    """Both new endpoints expose the same rows /history does, so both are gated."""
    import backend.routers.diagnostics as diagnostics

    created = client.post("/api/analyze",
                          files={"file": ("lesion.jpg", lesion, "image/jpeg")}).json()
    session_id = created["session_id"]

    monkeypatch.setattr(diagnostics, "REQUIRE_HISTORY_TOKEN", True)
    assert client.get(f"/api/history/{session_id}").status_code == 401
    assert client.get(f"/api/history/{session_id}/image").status_code == 401
    assert client.get(f"/api/history/{session_id}",
                      headers=admin_headers).status_code == 200
    assert client.get(f"/api/history/{session_id}/image",
                      headers=admin_headers).status_code == 200


# ── serving-configuration guards ─────────────────────────────────────────
# Each of these locks down a defect where the server would quietly serve
# something other than the configuration its published metrics describe.

def test_refuses_to_serve_without_decision_calibration(monkeypatch):
    """A missing calibration file must be fatal, not a silent fallback.

    Falling through to the config defaults means sigmoid instead of softmax, no
    temperature scaling, and the melanoma alert channel switched off - a
    different decision rule from the measured one, with nothing downstream able
    to tell.
    """
    import backend.ml_engine as ml_engine

    monkeypatch.setattr(ml_engine, "CALIBRATION_PATH", "models/__absent__.json")
    monkeypatch.setattr(ml_engine, "ALLOW_UNCALIBRATED", False)

    with pytest.raises(RuntimeError, match="calibration is unavailable"):
        ml_engine.SkinCancerPredictor()


@pytest.mark.requires_model
def test_uncalibrated_serving_requires_an_explicit_opt_in(monkeypatch):
    """ALLOW_UNCALIBRATED is the only way through, and the state stays visible."""
    import backend.ml_engine as ml_engine

    monkeypatch.setattr(ml_engine, "CALIBRATION_PATH", "models/__absent__.json")
    monkeypatch.setattr(ml_engine, "ALLOW_UNCALIBRATED", True)

    predictor = ml_engine.SkinCancerPredictor()
    assert predictor.calibration_loaded is False


def test_health_reports_the_decision_rule_in_force(client):
    """The decision layer must be inspectable without running a scan."""
    body = client.get("/api/health").json()

    decision = body["decision_layer"]
    assert decision["calibration_loaded"] is True
    assert decision["readout"] in {"softmax", "sigmoid"}
    assert decision["temperature"] > 0
    assert "decision_layer_uncalibrated" not in body["degraded"]


def test_health_reports_ood_gate_state(client):
    """An unfitted OOD gate must be visible, not implied to be working."""
    body = client.get("/api/health").json()

    ood = body["ood_gate"]
    assert set(ood) == {"thresholds_fitted", "feature_stage_fitted"}
    if not ood["feature_stage_fitted"]:
        assert "ood_feature_stage_inactive" in body["degraded"]


def test_inference_threads_do_not_oversubscribe_the_cpu(client):
    """Concurrent scans must not each claim a full-width torch thread pool."""
    import os
    import torch

    body = client.get("/api/health").json()["concurrency"]
    total = body["max_concurrent_inference"] * torch.get_num_threads()
    assert torch.get_num_threads() == body["torch_num_threads"]
    assert total <= (os.cpu_count() or 2)


def test_internal_errors_do_not_leak_exception_text(client, monkeypatch, caplog):
    """A 500 must carry a reference, not the exception message.

    /api/analyze is unauthenticated, and torch failures quote absolute build
    paths, usernames and checkpoint names. The operator needs the detail; the
    caller needs only something to quote back.
    """
    import logging

    import backend.routers.diagnostics as diagnostics

    secret = "cuDNN handle creation failed at " + os.path.join("private", "latest.pt")

    class Exploder:
        mel_alert_threshold = 0.1

        def predict(self, image):
            raise RuntimeError(secret)

    monkeypatch.setattr(diagnostics, "_predictor", Exploder())

    img = Image.new("RGB", (300, 300), (180, 120, 100))
    with caplog.at_level(logging.ERROR, logger="dermascan.diagnostics"):
        response = upload(client, "scan.jpg", jpeg_bytes(img))

    assert response.status_code == 500
    detail = response.json()["detail"]

    # Nothing internal reaches the caller.
    assert secret not in detail
    assert "cuDNN" not in detail
    assert "Traceback" not in detail
    assert "RuntimeError" not in detail

    # A reference does, and it is this request's id - the same one stamped on
    # every log line the request produced, including the one carrying the cause.
    import re
    match = re.search(r"reference ([0-9a-f]{12})", detail)
    assert match, detail
    reference = match.group(1)

    assert secret in caplog.text
    failure_records = [r for r in caplog.records if "analyze failed" in r.getMessage()]
    assert failure_records, caplog.text
    assert failure_records[0].request_id == reference

    # And it is handed back on the response, so a caller can correlate without
    # reading the body.
    assert response.headers["X-Request-ID"] == reference


def test_failed_analysis_does_not_leave_the_upload_behind(client, monkeypatch):
    """A crash mid-analysis must not orphan the file it just wrote."""
    import backend.routers.diagnostics as diagnostics

    class Exploder:
        mel_alert_threshold = 0.1

        def predict(self, image):
            raise RuntimeError("boom")

    monkeypatch.setattr(diagnostics, "_predictor", Exploder())

    before = set(os.listdir(diagnostics.UPLOAD_DIR))
    upload(client, "scan.jpg", jpeg_bytes(Image.new("RGB", (300, 300), (170, 110, 90))))
    after = set(os.listdir(diagnostics.UPLOAD_DIR))

    assert after == before


# ── observability ────────────────────────────────────────────────────────

def test_every_response_carries_a_request_id(client):
    response = client.get("/api/health")
    assert re.fullmatch(r"[0-9a-f]{12}", response.headers["X-Request-ID"])


def test_a_caller_supplied_request_id_is_echoed(client):
    """Lets a caller correlate its logs with ours."""
    response = client.get("/api/health", headers={"X-Request-ID": "caller-abc_123"})
    assert response.headers["X-Request-ID"] == "caller-abc_123"


def test_an_unsafe_request_id_is_replaced_not_echoed(client):
    """The id lands in every log line, so it is untrusted input."""
    injected = "abc\nFAKE LOG LINE"
    response = client.get("/api/health", headers={"X-Request-ID": injected})
    returned = response.headers["X-Request-ID"]
    assert returned != injected
    assert re.fullmatch(r"[0-9a-f]{12}", returned)


def test_log_records_carry_the_request_id(client, caplog):
    """Every record, from any module, must be attributable to a request."""
    import logging

    with caplog.at_level(logging.INFO, logger="dermascan"):
        response = client.get("/api/health")

    request_id = response.headers["X-Request-ID"]
    access = [r for r in caplog.records if r.name == "dermascan.access"]
    assert access, caplog.text
    assert access[0].request_id == request_id


@pytest.mark.requires_model
def test_metrics_report_inference_latency_and_outcomes(client, lesion):
    from backend import metrics as metrics_module

    metrics_module.reset()
    client.post("/api/analyze", files={"file": ("lesion.jpg", lesion, "image/jpeg")})

    body = client.get("/api/metrics").json()
    assert body["counters"]["analyses_completed"] == 1
    assert body["inference_ms"]["count"] == 1
    assert body["inference_ms"]["p50"] > 0
    # The predicted class is counted, so the mix is visible without log scraping.
    assert any(k.startswith("prediction_") for k in body["counters"])


@pytest.mark.requires_model
def test_metrics_count_ood_rejections_by_reason(client):
    from backend import metrics as metrics_module

    metrics_module.reset()
    flat = Image.new("RGB", (300, 300), (128, 128, 128))
    response = upload(client, "flat.jpg", jpeg_bytes(flat))
    assert response.status_code == 422

    counters = client.get("/api/metrics").json()["counters"]
    assert counters["ood_rejections_total"] == 1
    assert counters["ood_rejection_uniform_field"] == 1


# ── hardening ────────────────────────────────────────────────────────────

def test_oversized_upload_is_refused_before_the_body_is_parsed(client):
    """Content-Length is checked in middleware, before Starlette spools the body.

    The endpoint's own check still exists for chunked uploads that declare no
    length, but by then the whole body has already been buffered.
    """
    from backend.config import MAX_UPLOAD_BYTES

    response = client.post(
        "/api/analyze",
        headers={"Content-Length": str(MAX_UPLOAD_BYTES + 1),
                 "Content-Type": "application/octet-stream"},
        content=b"x" * 16,
    )
    assert response.status_code == 413


def test_rate_limit_rejects_a_flood_with_retry_after(client, monkeypatch):
    from backend import ratelimit

    monkeypatch.setattr(ratelimit, "RATE_LIMIT_PER_MINUTE", 3)
    ratelimit.reset()

    codes = [client.get("/api/health").status_code for _ in range(5)]
    assert codes[:3] == [200, 200, 200]
    assert codes[3:] == [429, 429]

    blocked = client.get("/api/health")
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) > 0


def test_inference_has_its_own_tighter_budget(client, monkeypatch):
    """Polling history freely must not buy the right to queue hundreds of scans."""
    from backend import ratelimit

    monkeypatch.setattr(ratelimit, "RATE_LIMIT_PER_MINUTE", 100)
    monkeypatch.setattr(ratelimit, "ANALYZE_RATE_LIMIT_PER_MINUTE", 1)
    ratelimit.reset()

    assert upload(client, "a.jpg", b"not an image").status_code == 400   # counted
    assert upload(client, "b.jpg", b"not an image").status_code == 429   # over budget
    assert client.get("/api/health").status_code == 200                  # other bucket


def test_history_read_guard_is_off_by_default(client):
    """The bundled UI reads history unauthenticated; the guard must be opt-in."""
    from backend.config import REQUIRE_HISTORY_TOKEN

    assert REQUIRE_HISTORY_TOKEN is False
    assert client.get("/api/history").status_code == 200


def test_history_can_be_put_behind_the_admin_token(client, monkeypatch,
                                                   admin_headers):
    import backend.routers.diagnostics as diagnostics

    monkeypatch.setattr(diagnostics, "REQUIRE_HISTORY_TOKEN", True)
    assert client.get("/api/history").status_code == 401
    assert client.get("/api/history", headers=admin_headers).status_code == 200


# ── model card ───────────────────────────────────────────────────────────
# /api/model publishes the recorded evaluation. These figures carry more weight
# than anything else the interface shows, so the endpoint has to be right about
# whose numbers they are.

def test_model_card_reports_the_serving_configuration(client):
    """Readable without a checkpoint: it reads files, it does not load the model."""
    body = client.get("/api/model").json()

    assert body["backbone"] == "EfficientNet-B3"
    assert body["classes"] == ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    assert body["thresholds"] is not None, "thresholds ship with the repo"
    assert set(body["thresholds"]) == set(body["classes"])
    assert "readout" in body["decision_layer"]
    # Publishing a page about the model must not drag 123 MB into memory.
    assert body["checkpoint"]["loaded"] is False or body["checkpoint"]["present"]


def test_model_card_says_whether_the_evaluation_matches_what_is_served(client):
    """The whole point: numbers measured under other thresholds are not ours."""
    body = client.get("/api/model").json()

    if not body["evaluation_available"]:
        pytest.skip("no evaluation artifact in this checkout")

    assert body["evaluation_describes_this_configuration"] is True, (
        "the shipped evaluation should have been measured under the shipped "
        f"thresholds; mismatches: {body['evaluation_threshold_mismatches']}")
    assert body["evaluation_threshold_mismatches"] == []


def test_model_card_flags_an_evaluation_measured_under_other_thresholds(client,
                                                                       monkeypatch,
                                                                       tmp_path):
    """A drifted evaluation must be called out, not quietly published."""
    import json
    import backend.main as main

    check = json.loads(open(main.SERVING_CHECK_PATH, encoding="utf-8").read())
    check["thresholds"]["mel"] = 0.99          # not what the server is using
    drifted = tmp_path / "drifted.json"
    drifted.write_text(json.dumps(check), encoding="utf-8")
    monkeypatch.setattr(main, "SERVING_CHECK_PATH", str(drifted))

    body = client.get("/api/model").json()
    assert body["evaluation_describes_this_configuration"] is False
    mismatch = [m for m in body["evaluation_threshold_mismatches"] if m["class"] == "mel"]
    assert mismatch and mismatch[0]["evaluated_with"] == 0.99


def test_model_card_degrades_when_the_evaluation_is_absent(client, monkeypatch,
                                                           tmp_path):
    """Absent artifacts are reported as absent. Nothing is filled in."""
    import backend.main as main

    monkeypatch.setattr(main, "EVALUATION_PATH", str(tmp_path / "nope.json"))
    monkeypatch.setattr(main, "SERVING_CHECK_PATH", str(tmp_path / "nope2.json"))

    body = client.get("/api/model").json()
    assert body["evaluation_available"] is False
    assert body["evaluation"] is None
    assert body["evaluation_describes_this_configuration"] is None
    # The serving configuration is still reported - that part needs no artifact.
    assert body["thresholds"] is not None


def test_confusion_matrix_axes_are_in_the_order_the_api_declares(client):
    """The matrix carries no class labels of its own.

    docs/evaluation_results.json stores `confusion_matrix` as a bare 7x7 array
    with no `classes` key, and the model card labels its axes with the order
    /api/model declares. If an evaluation run ever writes a different order, the
    page would mislabel every cell while looking entirely plausible - so the
    order is pinned to two independent properties of the same file: each row must
    sum to that class's recorded support, and each diagonal ratio must reproduce
    its recorded recall.
    """
    body = client.get("/api/model").json()
    if not body["evaluation_available"]:
        pytest.skip("no evaluation artifact in this checkout")

    classes = body["classes"]
    matrix = body["evaluation"]["confusion_matrix"]
    per_class = body["evaluation"]["per_class"]

    assert len(matrix) == len(classes)
    for i, cls in enumerate(classes):
        row_total = sum(matrix[i])
        assert row_total == per_class[cls]["support"], (
            f"row {i} sums to {row_total}, but {cls} has support "
            f"{per_class[cls]['support']} - the axes are not in this order")
        assert matrix[i][i] / row_total == pytest.approx(per_class[cls]["recall"])


def test_health_reports_which_hardening_is_active(client):
    hardening = client.get("/api/health").json()["hardening"]
    assert hardening["max_upload_mb"] >= 1
    assert hardening["history_requires_token"] is False
    assert "rate_limit_per_minute" in hardening
    assert "upload_retention_days" in hardening
