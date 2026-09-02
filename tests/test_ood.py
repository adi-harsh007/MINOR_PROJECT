"""Out-of-distribution gate.

The regression these lock down: the previous gate keyed on absolute channel
standard deviation, which scales with image brightness, so identical lesions were
accepted or rejected purely on how dark the image was.
"""

import os

import numpy as np
import pytest
from PIL import Image, ImageEnhance

from backend.ood import color_gate, compute_metrics


@pytest.fixture(scope="module")
def lesion(project_root):
    return Image.open(os.path.join(project_root, "samples", "ISIC_0024307.jpg")).convert("RGB")


def accepted(img):
    return not color_gate(img)["is_ood"]


def reason(img):
    return color_gate(img)["reason"]


# ── the bias regression ──────────────────────────────────────────────────

@pytest.mark.parametrize("factor", [1.0, 0.75, 0.5, 0.35, 0.25, 0.15])
def test_accepts_lesion_at_any_brightness(lesion, factor):
    """Darkening must not change the verdict. The old gate rejected 0.35 and below."""
    assert accepted(ImageEnhance.Brightness(lesion).enhance(factor))


def test_metrics_are_illumination_invariant(lesion):
    full = compute_metrics(lesion)
    dark = compute_metrics(ImageEnhance.Brightness(lesion).enhance(0.35))
    assert dark["rel_contrast"] == pytest.approx(full["rel_contrast"], abs=0.02)
    assert dark["hf_ratio"] == pytest.approx(full["hf_ratio"], abs=0.02)


def test_accepts_grayscale_dermoscopy(lesion):
    """Grayscale input was previously rejected outright."""
    assert accepted(lesion.convert("L").convert("RGB"))


def test_accepts_high_contrast_lesion(lesion):
    """A dark lesion on pale skin tripped the old avg_std upper bound."""
    assert accepted(ImageEnhance.Contrast(lesion).enhance(1.8))


# ── genuine rejections ───────────────────────────────────────────────────

def test_rejects_uniform_field():
    assert reason(Image.new("RGB", (300, 300), (240, 240, 238))) == "uniform_field"


def test_rejects_saturated_flat_colour():
    assert reason(Image.new("RGB", (300, 300), (220, 40, 40))) == "uniform_field"


def test_rejects_pixel_noise():
    noise = Image.fromarray(np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8))
    assert reason(noise) == "pixel_noise"


def test_rejects_grayscale_noise():
    g = np.random.randint(100, 200, (300, 300), dtype=np.uint8)
    assert reason(Image.fromarray(np.stack([g] * 3, axis=-1))) == "pixel_noise"


def test_rejection_carries_human_readable_detail():
    result = color_gate(Image.new("RGB", (300, 300), (240, 240, 238)))
    assert result["detail"] and result["detail"][0].isupper()


# ── documented limitation ────────────────────────────────────────────────

def test_feature_stage_is_inactive_until_fitted():
    """Semantic OOD needs the fitted Mahalanobis stage; it reports itself unavailable."""
    from backend.ood import FeatureSpaceOOD

    detector = FeatureSpaceOOD(stats_path="does-not-exist.npz")
    assert detector.available is False
    assert detector.check(np.zeros(8)) is None
