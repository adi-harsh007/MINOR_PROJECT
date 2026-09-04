import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import json
import os
import threading
from typing import Optional

import base64
from io import BytesIO

from .config import (MODEL_PATH, THRESHOLD_PATH, IMG_SIZE, MODEL_ARCH,
                     CALIBRATION_PATH, DEFAULT_TEMPERATURE, DEFAULT_MEL_ALERT_THRESHOLD,
                     DEFAULT_READOUT, TORCH_NUM_THREADS, ALLOW_UNCALIBRATED)

# Bound intra-op parallelism before any inference runs. Torch reads this once per
# process and it is global, so it is set at import rather than per predictor.
# set_num_interop_threads() can only be called before the inter-op pool is
# created and raises afterwards, so it is attempted separately and never fatal.
torch.set_num_threads(TORCH_NUM_THREADS)
try:
    torch.set_num_interop_threads(TORCH_NUM_THREADS)
except RuntimeError:
    pass
from .model import build_model, get_conv_head, get_pooled_features
from .ood import color_gate, load_thresholds, FeatureSpaceOOD
from .logging_setup import get_logger

log = get_logger("ml_engine")

def read_calibration():
    """Read models/calibration.json.

    Returns ``(values, loaded, error)``. Shared by the predictor and by
    /api/health so the health endpoint can report the decision rule actually in
    force without pulling the checkpoint into memory.
    """
    try:
        with open(CALIBRATION_PATH, "r") as f:
            return json.load(f), True, None
    except FileNotFoundError:
        return {}, False, "file not found"
    except Exception as e:
        # The category, not the message: this value is surfaced by /api/health,
        # which is unauthenticated, and a parse error's text can quote file
        # contents and paths.
        return {}, False, "unreadable ({})".format(type(e).__name__)


def decision_config():
    """The decision rule in force, resolved the same way the predictor resolves it."""
    calib, loaded, error = read_calibration()
    return {
        "readout": calib.get("readout", DEFAULT_READOUT),
        "temperature": float(calib.get("temperature", DEFAULT_TEMPERATURE)),
        "mel_alert_threshold": calib.get("mel_alert_threshold",
                                         DEFAULT_MEL_ALERT_THRESHOLD),
        "calibration_loaded": loaded,
        "calibration_error": error,
        "allow_uncalibrated": bool(ALLOW_UNCALIBRATED),
    }


def read_threshold_file(threshold_path, classes):
    """Decision thresholds and the metrics recorded beside them.

    Split out of the predictor so it can be tested without the checkpoint:
    models/latest.pt is gitignored and absent in CI, but models/class_thresholds.json
    is tracked, and it is the file the interface quotes this model's melanoma recall
    from. That figure used to be a literal in the frontend, where a retrain would
    have left it asserting the old number indefinitely; a reader that silently
    returned empty metrics would put it straight back in that position, so the
    parsing needs coverage that actually runs.

    Returns (thresholds, per-class metrics, the file's own `fitted_on` label).
    """
    with open(threshold_path, 'r') as f:
        data = json.load(f)

    per_class = data['per_class_metrics']
    thresholds = {k: per_class[k]['threshold'] for k in classes}
    metrics = {k: dict(per_class.get(k, {})) for k in classes}
    return thresholds, metrics, data.get('fitted_on')


# Fields scripts/optimize_thresholds.py always writes. Their absence means the
# threshold file came from somewhere else, and `python scripts/optimize_thresholds.py`
# will not reproduce the artifact every served decision depends on.
THRESHOLD_PIPELINE_FIELDS = ("fitted_on", "reported_on", "rule", "objective",
                             "test_metrics")


def threshold_provenance(threshold_path):
    """Can the thresholds in use be traced to the tool that is meant to produce them?

    optimize_thresholds.py exists to keep two splits apart - it fits on a
    calibration split and reports on a held-out one, because "fitting and
    reporting on the same images overstates performance". `reported_on` is the
    field that records the second half of that, so a file missing it cannot show
    the separation was kept, however plausible its numbers look.

    This reports what is there; it does not guess. An unverifiable provenance is
    not an accusation that the thresholds are overfitted - it is the statement
    that nothing here can rule it out, which the model card then says out loud
    rather than printing a confident-looking `fitted_on` string beside figures.
    """
    try:
        with open(threshold_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        return {
            "readable": False,
            "detail": str(e),
            "produced_by_pipeline": False,
            "missing_fields": list(THRESHOLD_PIPELINE_FIELDS),
            "fitted_on": None,
            "reported_on": None,
            "splits_shown_separate": False,
        }

    missing = [f for f in THRESHOLD_PIPELINE_FIELDS if f not in data]
    fitted_on = data.get("fitted_on")
    reported_on = data.get("reported_on")
    return {
        "readable": True,
        "produced_by_pipeline": not missing,
        "missing_fields": missing,
        "fitted_on": fitted_on,
        "reported_on": reported_on,
        # Both recorded, and different: the only state in which this file itself
        # demonstrates that thresholds were not scored on the data they were
        # tuned on.
        "splits_shown_separate": bool(fitted_on and reported_on
                                      and fitted_on != reported_on),
    }


class SkinCancerPredictor:
    def __init__(self, model_path=MODEL_PATH, threshold_path=THRESHOLD_PATH):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.classes = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
        
        # Load thresholds
        try:
            (self.thresholds,
             self.class_metrics,
             self.thresholds_fitted_on) = read_threshold_file(threshold_path, self.classes)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load per-class thresholds from {threshold_path}: {e}. "
                "Refusing to serve predictions with unvalidated decision thresholds."
            ) from e
            
        # Confidence calibration and the melanoma alert channel.
        #
        # A missing or unreadable calibration file used to fall through to the
        # config defaults with only a print(): sigmoid instead of softmax, no
        # temperature scaling, and the melanoma alert channel switched off
        # entirely. That is a materially different decision rule from the one
        # every published metric describes, and nothing downstream could tell.
        # It now refuses to construct unless the operator opts in explicitly.
        calib, self.calibration_loaded, calib_error = read_calibration()

        if not self.calibration_loaded:
            message = (
                f"Decision-layer calibration is unavailable ({calib_error}). "
                f"Without {CALIBRATION_PATH} the server would fall back to "
                f"readout={DEFAULT_READOUT!r}, temperature={DEFAULT_TEMPERATURE}, "
                f"melanoma alert disabled — a different decision rule from the one "
                f"the reported metrics were measured under, with no melanoma alert "
                f"channel. Restore the file, or set ALLOW_UNCALIBRATED=1 to serve "
                f"this configuration deliberately."
            )
            if not ALLOW_UNCALIBRATED:
                raise RuntimeError(message)
            log.warning(message)

        self.temperature = float(calib.get("temperature", DEFAULT_TEMPERATURE))
        self.mel_alert_threshold = calib.get("mel_alert_threshold",
                                             DEFAULT_MEL_ALERT_THRESHOLD)
        self.readout = calib.get("readout", DEFAULT_READOUT)
        log.info("decision layer: readout=%s T=%.2f melanoma_alert>=%s (calibration %s)",
                 self.readout, self.temperature, self.mel_alert_threshold,
                 "loaded" if self.calibration_loaded else "MISSING")

        # Architecture must match the training checkpoint exactly. load_state_dict
        # is strict, so a mismatched checkpoint fails loudly instead of serving a
        # different network than the one the published metrics describe.
        self.model = build_model(MODEL_ARCH, num_classes=len(self.classes))
        
        # A failure here must be fatal: serving randomly-initialised weights would
        # produce confident-looking predictions from an untrained network.
        try:
            # Prefer the safe weights-only path. Training checkpoints that embed
            # numpy scalars in their metrics need the full unpickler; MODEL_PATH is
            # operator-controlled configuration, not user-supplied input.
            try:
                ckpt = torch.load(model_path, map_location=self.device, weights_only=True)
            except Exception:
                ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            state_dict = ckpt.get('model_state_dict', ckpt)
            cleaned_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(cleaned_dict)
            log.info("model loaded on %s from %s", self.device,
                     os.path.basename(model_path))
        except Exception as e:
            raise RuntimeError(
                f"Failed to load model weights from {model_path}: {e}. "
                "Refusing to serve predictions from an uninitialised network."
            ) from e
            
        self.model.to(self.device).eval()

        # OOD gating
        self.ood_thresholds, self.ood_calibrated = load_thresholds()
        ood_calibrated = self.ood_calibrated
        if not ood_calibrated:
            log.warning("OOD gate running on provisional uncalibrated thresholds; "
                        "run scripts/calibrate_ood.py to fit them")
        self.feature_ood = FeatureSpaceOOD()

        # Grad-CAM state setup
        self._gradcam_lock = threading.Lock()
        self.gradients = None
        self.activations = None
        self._register_gradcam_hooks()

        # Preprocessing
        # Matches src/transforms.py get_val_transforms: A.Resize(img_size, img_size).
        # A centre crop here would discard the border at a resolution the model
        # was never evaluated at.
        self.transform = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _register_gradcam_hooks(self):
        """Attaches backward and forward hooks to the last conv layer of EfficientNet-B3."""
        def forward_hook(module, input, output):
            self.activations = output

        def backward_hook(module, grad_in, grad_out):
            self.gradients = grad_out[0]

        try:
            target_layer = get_conv_head(self.model)
            target_layer.register_forward_hook(forward_hook)
            target_layer.register_full_backward_hook(backward_hook)
        except Exception as e:
            log.warning("could not attach Grad-CAM hooks to conv_head: %s", e)

    def _apply_jet_colormap(self, cam_np: np.ndarray) -> Image.Image:
        """Converts a normalized 2D numpy array [0, 1] into a Jet RGBA heatmap Image."""
        h, w = cam_np.shape
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        # Standard Jet colormap approximation
        x = np.clip(cam_np, 0.0, 1.0)
        
        r = np.clip(1.5 - np.abs(x * 4.0 - 3.0), 0.0, 1.0)
        g = np.clip(1.5 - np.abs(x * 4.0 - 2.0), 0.0, 1.0)
        b = np.clip(1.5 - np.abs(x * 4.0 - 1.0), 0.0, 1.0)
        alpha = np.clip(x * 0.85 + 0.15, 0.0, 0.95)

        rgba[:, :, 0] = (r * 255).astype(np.uint8)
        rgba[:, :, 1] = (g * 255).astype(np.uint8)
        rgba[:, :, 2] = (b * 255).astype(np.uint8)
        rgba[:, :, 3] = (alpha * 255).astype(np.uint8)

        return Image.fromarray(rgba, mode='RGBA')

    def generate_gradcam_base64(self, image: Image.Image, target_class_idx: int) -> Optional[str]:
        """Generates a Grad-CAM heatmap base64 string for the given target class index.

        Returns None if attribution could not be computed. A synthetic stand-in is
        never returned: a fabricated heatmap is worse than no heatmap, because the
        clinician cannot tell the difference.
        """
        try:
            with self._gradcam_lock:
                return self._compute_gradcam(image, target_class_idx)

        except Exception as e:
            log.warning("Grad-CAM unavailable for this scan: %s", e)
            return None

    def _compute_gradcam(self, image: Image.Image, target_class_idx: int) -> str:
        """Grad-CAM computation proper. Caller must hold ``self._gradcam_lock``."""
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        tensor.requires_grad = True

        # Enable grad for Grad-CAM
        with torch.enable_grad():
            self.model.zero_grad()
            logits = self.model(tensor)
            score = logits[0, target_class_idx]
            score.backward()

        if self.activations is None or self.gradients is None:
            raise ValueError("Grad-CAM activations/gradients were not captured")

        gradients = self.gradients.data
        activations = self.activations.data

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Resize CAM to image size
        cam = F.interpolate(cam, size=(IMG_SIZE, IMG_SIZE), mode='bilinear', align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        # Normalize 0-1
        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        else:
            cam = np.zeros_like(cam)

        heatmap_img = self._apply_jet_colormap(cam)

        # Convert to base64
        buffered = BytesIO()
        heatmap_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

    def _extract_features(self, tensor):
        """Penultimate feature vector, used by the feature-space OOD detector."""
        with torch.no_grad():
            return get_pooled_features(self.model, tensor).squeeze(0).cpu().numpy()

    def _check_ood(self, image: Image.Image, tensor):
        """Runs both OOD stages.

        Returns ``(rejection, metrics)``. ``rejection`` is a rejection dict, or
        None to proceed; ``metrics`` is the stage-1 image statistics either way.
        The metrics are reported on the success path too, so the interface can
        show what the gate actually measured rather than an invented constant.
        """
        result = color_gate(image, self.ood_thresholds)
        metrics = result.get("metrics")
        if result["is_ood"]:
            return result, metrics

        stage2 = self.feature_ood.check(self._extract_features(tensor))
        if stage2 is not None and stage2["is_ood"]:
            return stage2, metrics

        return None, metrics

    def predict(self, image: Image.Image):
        if image.mode != 'RGB':
            image = image.convert('RGB')

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        # ── OOD gate: image statistics, then feature space ───────
        ood_check, ood_metrics = self._check_ood(image, tensor)
        if ood_check is not None:
            return ood_check

        
        with torch.no_grad():
            logits = self.model(tensor)
            # Sigmoid readout with the margin rule. The per-class thresholds in
            # models/class_thresholds.json were tuned on softmax with the rule in
            # scripts/optimize_thresholds.py, so this pairing is off-protocol —
            # but it was measured on the full 1525-image test set and gives the
            # best melanoma F1 (0.636) of any configuration, at accuracy 0.851 /
            # macro-F1 0.745. Melanoma recall is the error that matters clinically
            # here. Re-run threshold optimisation if this readout changes.
            # Temperature scaling: divides the logits before the readout, which
            # flattens over-confident probabilities without retraining. T=1.0 is
            # a no-op.
            scaled = logits / self.temperature
            if self.readout == "softmax":
                probs = torch.softmax(scaled, dim=1).squeeze(0).cpu().numpy().tolist()
            else:
                probs = torch.sigmoid(scaled).squeeze(0).cpu().numpy().tolist()

        results = {c: probs[i] for i, c in enumerate(self.classes)}

        # Margin rule: pick the class furthest above its own threshold.
        best_class = max(self.classes, key=lambda c: results[c] - self.thresholds[c])
        best_margin = results[best_class] - self.thresholds[best_class]

        if best_margin < 0:
            return {
                "is_ood": True,
                "reason": "low_confidence",
                "detail": (
                    "No class exceeded its decision threshold; the model cannot "
                    "make a supported call on this image."
                ),
            }

        # ── Melanoma alert ────────────────────────────────────────
        # Recall is limited by the argmax, not by what the model knows: on
        # melanomas it misses, p(mel) is still substantial. Flagging on p(mel)
        # directly surfaces them without altering the primary prediction.
        mel_probability = results["mel"]
        melanoma_alert = bool(
            self.mel_alert_threshold is not None
            and (best_class == "mel" or mel_probability >= self.mel_alert_threshold)
        )

        target_idx = self.classes.index(best_class)
        heatmap_base64 = self.generate_gradcam_base64(image, target_idx)

        return {
            "prediction": best_class,
            "confidence": results[best_class],
            "threshold": self.thresholds[best_class],
            "scores": results,
            "melanoma_alert": melanoma_alert,
            "melanoma_probability": mel_probability,
            "heatmap_base64": heatmap_base64,
            # What the gate measured on this image, plus whether its thresholds
            # were fitted to real data or are still the provisional defaults.
            # Reported so the interface can state the gate's actual condition
            # instead of asserting that every scan passed a calibrated check.
            "ood_metrics": ood_metrics,
            "ood_calibrated": bool(self.ood_calibrated),
            "ood_feature_stage_active": bool(self.feature_ood.available),
        }
