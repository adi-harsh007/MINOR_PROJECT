"""Out-of-distribution gating for the DermaScan classifier.

Two independent stages:

1. `color_gate` — cheap image-statistics sanity checks. Every metric here is
   **illumination-invariant**: it is computed as a ratio, so scaling an image's
   brightness does not change it. This matters clinically. The previous gate
   used absolute channel standard deviation, which scales with brightness, so
   darker images — darker skin tones, underexposed captures, poor lighting —
   drifted toward the "too uniform" rejection while identical lighter images
   passed. It also rejected all grayscale input, which excluded legitimate
   grayscale dermoscopy.

   This stage reliably rejects flat colour fields and pixel noise. It cannot
   reject a photograph of some other real object: a desaturated photo of an
   animal has image statistics well inside the dermoscopic range. That is a
   limitation of colour statistics, not a tuning problem.

2. `FeatureSpaceOOD` — Mahalanobis distance in the classifier's own penultimate
   feature space, which *can* separate semantic OOD. It requires fitting on
   in-distribution data first (see `scripts/calibrate_ood.py`). When no fitted
   statistics are present it reports itself unavailable and the system falls
   back to stage 1 alone.

Note on model confidence: max-softmax and energy scores were measured on this
checkpoint and are NOT usable as an OOD signal. A blank white field scores
max-softmax 0.994 and a *lower* (more in-distribution) energy than any real
lesion image. Do not reintroduce confidence-based OOD without re-measuring.
"""

import json
import os

import numpy as np
from PIL import Image, ImageFilter

from .config import BASE_DIR
from .logging_setup import get_logger

log = get_logger("ood")

OOD_CONFIG_PATH = os.path.join(BASE_DIR, "models", "ood_config.json")
OOD_STATS_PATH = os.path.join(BASE_DIR, "models", "ood_stats.npz")

# Provisional defaults.
#
# These are set to be *permissive*: each is far from the values measured on real
# dermoscopic images, so the gate rejects only clear-cut non-clinical input. They
# were derived from a very small sample and are not properly calibrated. Run
# `scripts/calibrate_ood.py --data-dir <in-distribution images>` to fit them to a
# real dataset, which writes models/ood_config.json and overrides these.
DEFAULT_THRESHOLDS = {
    # Relative luminance contrast (std / mean). Flat colour fields sit at ~0.00;
    # real dermoscopy measured 0.12-0.44.
    "min_rel_contrast": 0.04,
    # High-frequency residual ratio. Real dermoscopy measured 0.04-0.16;
    # pixel noise sits above 0.90.
    "max_hf_ratio": 0.45,
    # Fraction of chromatically meaningful pixels whose hue is blue/green.
    # Sky and foliage sit at 1.00; skin at 0.00.
    "max_blue_green": 0.60,
    # Hue checks are only applied when at least this fraction of pixels carry a
    # numerically meaningful hue (enough saturation and value). Grayscale images
    # fall below this and skip hue checks entirely rather than being rejected.
    "min_chromatic_fraction": 0.25,
    # Mahalanobis distance percentile cutoff, used only when fitted stats exist.
    "max_mahalanobis": None,
}

_ANALYSIS_SIZE = (128, 128)


def load_thresholds():
    """Returns the active thresholds, preferring a fitted models/ood_config.json."""
    thresholds = dict(DEFAULT_THRESHOLDS)
    calibrated = False
    try:
        with open(OOD_CONFIG_PATH) as f:
            data = json.load(f)
        thresholds.update(data.get("thresholds", {}))
        calibrated = True
    except FileNotFoundError:
        pass
    except Exception as e:
        log.warning("could not read %s: %s", OOD_CONFIG_PATH, e)
    return thresholds, calibrated


def compute_metrics(pil_image):
    """Illumination-invariant image statistics used by the colour gate."""
    rgb_img = pil_image.convert("RGB").resize(_ANALYSIS_SIZE)
    rgb = np.asarray(rgb_img).astype(np.float32)

    lum = rgb.mean(axis=2)
    mean_lum = float(lum.mean())
    # Ratio, so uniform scaling of brightness leaves it unchanged.
    rel_contrast = float(lum.std() / (mean_lum + 1e-6))

    gray = np.asarray(rgb_img.convert("L"))
    blurred = np.asarray(
        Image.fromarray(gray).filter(ImageFilter.GaussianBlur(1.5))
    ).astype(np.float32)
    residual = gray.astype(np.float32) - blurred
    hf_ratio = float(residual.std() / (gray.astype(np.float32).std() + 1e-6))

    hsv = np.asarray(rgb_img.convert("HSV")).astype(np.float32)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # Hue is numerically meaningless at low saturation or value; only count
    # pixels where it carries information.
    chromatic = (s > 25) & (v > 25)
    chromatic_fraction = float(chromatic.mean())
    if chromatic.sum() > 0:
        hue = h[chromatic]
        blue_green = float(((hue > 60) & (hue < 170)).sum() / chromatic.sum())
    else:
        blue_green = 0.0

    return {
        "rel_contrast": rel_contrast,
        "hf_ratio": hf_ratio,
        "blue_green": blue_green,
        "chromatic_fraction": chromatic_fraction,
        "mean_luminance": mean_lum,
    }


def color_gate(pil_image, thresholds=None):
    """Stage 1. Returns {'is_ood', 'reason', 'detail', 'metrics'}."""
    if thresholds is None:
        thresholds, _ = load_thresholds()

    m = compute_metrics(pil_image)

    if m["rel_contrast"] < thresholds["min_rel_contrast"]:
        return _reject(
            "uniform_field", m,
            "Image is a near-uniform colour field with no visible structure.",
        )

    if m["hf_ratio"] > thresholds["max_hf_ratio"]:
        return _reject(
            "pixel_noise", m,
            "Image is dominated by high-frequency noise rather than tissue detail.",
        )

    # Hue is only informative when enough pixels are chromatic. Grayscale
    # dermoscopy legitimately falls below this and skips the check.
    if m["chromatic_fraction"] >= thresholds["min_chromatic_fraction"]:
        if m["blue_green"] > thresholds["max_blue_green"]:
            return _reject(
                "non_skin_colour", m,
                "Image is dominated by blue/green hues not found in skin tissue.",
            )

    return {"is_ood": False, "reason": None, "detail": None, "metrics": m}


def _reject(reason, metrics, detail):
    return {"is_ood": True, "reason": reason, "detail": detail, "metrics": metrics}


class FeatureSpaceOOD:
    """Stage 2. Mahalanobis distance in the classifier's penultimate features.

    Unavailable until fitted. `scripts/calibrate_ood.py` writes the statistics.
    """

    def __init__(self, stats_path=OOD_STATS_PATH):
        self.available = False
        self.mean = None
        self.inv_cov = None
        self.threshold = None
        try:
            data = np.load(stats_path)
            self.mean = data["mean"]
            self.inv_cov = data["inv_cov"]
            self.threshold = float(data["threshold"])
            self.available = True
            log.info("feature-space OOD detector loaded (threshold %.2f)",
                     self.threshold)
        except FileNotFoundError:
            log.warning("feature-space OOD detector not fitted; colour gate only. "
                        "Run scripts/calibrate_ood.py to enable it")
        except Exception as e:
            log.warning("could not load OOD stats: %s", e)

    def distance(self, features):
        """Squared Mahalanobis distance of a 1-D feature vector."""
        delta = np.asarray(features, dtype=np.float64) - self.mean
        return float(delta @ self.inv_cov @ delta)

    def check(self, features):
        """Returns {'is_ood', 'reason', 'detail', 'distance'} or None if unfitted."""
        if not self.available:
            return None
        d = self.distance(features)
        if d > self.threshold:
            return {
                "is_ood": True,
                "reason": "unfamiliar_image",
                "detail": (
                    "Image does not resemble the dermoscopic images this model "
                    "was trained on."
                ),
                "distance": d,
            }
        return {"is_ood": False, "reason": None, "detail": None, "distance": d}
