"""Fit confidence calibration and the melanoma alert threshold.

Two independent problems, two fixes:

1. **Temperature scaling.** The raw model is badly calibrated: it states ~0.97
   confidence on answers that are wrong almost as often as on answers that are
   right. Dividing logits by a temperature T flattens that without retraining.

2. **Melanoma alert threshold.** Melanoma recall is limited by the argmax, not by
   the model's knowledge: on melanomas it misses, it still assigns a substantial
   melanoma probability - `nv` simply edges ahead. Flagging on p(mel) directly,
   independently of which class wins, surfaces far more of them.

    python scripts/fit_calibration.py --calib-dir path/to/calib --test-dir path/to/test

Both directories take one subdirectory per class. T is chosen on the calibration
split by minimising expected calibration error; the alert threshold is chosen for
the target melanoma sensitivity. Results are reported on the test split, and
nothing is written without --write.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import CALIBRATION_PATH  # noqa: E402
from backend.ml_engine import SkinCancerPredictor  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def collect(data_dir, classes):
    out = []
    for idx, cls in enumerate(classes):
        d = os.path.join(data_dir, cls)
        if not os.path.isdir(d):
            continue
        for n in sorted(os.listdir(d)):
            if n.lower().endswith(IMAGE_EXTS):
                out.append((os.path.join(d, n), idx))
    if not out:
        raise SystemExit("No images found under " + data_dir)
    return out


@torch.no_grad()
def logits_for(predictor, samples, batch_size=16):
    outs, labels = [], []
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i + batch_size]
        tensors, labs = [], []
        for path, label in chunk:
            try:
                tensors.append(predictor.transform(Image.open(path).convert("RGB")))
                labs.append(label)
            except Exception as e:
                print("  Skipping {}: {}".format(path, e))
        if not tensors:
            continue
        outs.append(predictor.model(torch.stack(tensors).to(predictor.device)).cpu().numpy())
        labels.extend(labs)
        print("  {}/{}".format(min(i + batch_size, len(samples)), len(samples)), end="\r")
    print()
    return np.concatenate(outs), np.array(labels)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def ece(confidence, correct, bins=15):
    """Expected calibration error: gap between stated confidence and actual accuracy."""
    total = 0.0
    for i in range(bins):
        m = (confidence > i / bins) & (confidence <= (i + 1) / bins)
        if m.sum():
            total += m.mean() * abs(correct[m].mean() - confidence[m].mean())
    return float(total)


def predict(probs, thresholds):
    return np.argmax(probs - thresholds[None, :], axis=1)


def fit_temperature(logits, y, thresholds, grid=np.arange(1.0, 8.01, 0.1)):
    best_t, best_e = 1.0, float("inf")
    for t in grid:
        probs = sigmoid(logits / t)
        yp = predict(probs, thresholds)
        conf = probs[np.arange(len(yp)), yp]
        e = ece(conf, yp == y)
        if e < best_e:
            best_t, best_e = float(t), e
    return best_t, best_e


def fit_alert(logits, y, thresholds, mel_idx, temperature, target_sensitivity):
    """Lowest review rate that still reaches the target melanoma sensitivity."""
    probs = sigmoid(logits / temperature)
    yp = predict(probs, thresholds)
    mel = y == mel_idx
    chosen, chosen_rate = None, 1.0
    for t in np.round(np.arange(0.20, 0.91, 0.05), 2):
        surfaced = (probs[:, mel_idx] >= t) | (yp == mel_idx)
        if surfaced[mel].mean() >= target_sensitivity:
            rate = surfaced[~mel].mean()
            if rate <= chosen_rate:
                chosen, chosen_rate = float(t), float(rate)
    return chosen


def report(logits, y, thresholds, mel_idx, temperature, alert_t, label):
    probs = sigmoid(logits / temperature)
    yp = predict(probs, thresholds)
    conf = probs[np.arange(len(yp)), yp]
    ok = yp == y
    mel = y == mel_idx
    surfaced = (probs[:, mel_idx] >= alert_t) | (yp == mel_idx)
    print("\n-- {} (n={}) {}".format(label, len(y), "-" * max(0, 30 - len(label))))
    print("  accuracy                 {:.4f}".format(ok.mean()))
    print("  ECE                      {:.4f}".format(ece(conf, ok)))
    print("  mean confidence  right   {:.4f}".format(conf[ok].mean()))
    print("  mean confidence  wrong   {:.4f}".format(conf[~ok].mean()))
    print("  melanoma recall (argmax) {:.4f}".format((yp[mel] == mel_idx).mean()))
    print("  melanoma surfaced        {:.4f}   <- with the alert channel".format(surfaced[mel].mean()))
    print("  cases flagged for review {:.4f}".format(surfaced[~mel].mean()))
    return {
        "accuracy": float(ok.mean()),
        "ece": ece(conf, ok),
        "melanoma_recall_argmax": float((yp[mel] == mel_idx).mean()),
        "melanoma_surfaced": float(surfaced[mel].mean()),
        "review_rate": float(surfaced[~mel].mean()),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib-dir", required=True, help="Split to FIT on.")
    ap.add_argument("--test-dir", required=True, help="Split to REPORT on.")
    ap.add_argument("--target-sensitivity", type=float, default=0.90,
                    help="Fraction of melanomas the alert channel should surface "
                         "(default 0.90). Higher means more cases flagged for review.")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    predictor = SkinCancerPredictor()
    classes = predictor.classes
    mel_idx = classes.index("mel")
    thresholds = np.array([predictor.thresholds[c] for c in classes])

    print("Scoring calibration split...")
    cal_logits, cal_y = logits_for(predictor, collect(args.calib_dir, classes), args.batch_size)
    print("Scoring test split...")
    test_logits, test_y = logits_for(predictor, collect(args.test_dir, classes), args.batch_size)

    temperature, cal_ece = fit_temperature(cal_logits, cal_y, thresholds)
    print("\nTemperature fitted on calibration: T = {:.1f} (calib ECE {:.4f})".format(
        temperature, cal_ece))

    alert_t = fit_alert(cal_logits, cal_y, thresholds, mel_idx, temperature,
                        args.target_sensitivity)
    if alert_t is None:
        raise SystemExit("No alert threshold reaches the target sensitivity.")
    print("Alert threshold fitted on calibration: p(mel) >= {:.2f}".format(alert_t))

    report(cal_logits, cal_y, thresholds, mel_idx, 1.0, alert_t, "CALIB before (T=1.0)")
    report(cal_logits, cal_y, thresholds, mel_idx, temperature, alert_t, "CALIB after")
    before = report(test_logits, test_y, thresholds, mel_idx, 1.0, alert_t, "TEST before (T=1.0)")
    after = report(test_logits, test_y, thresholds, mel_idx, temperature, alert_t, "TEST after")

    if not args.write:
        print("\nNot written. Re-run with --write to save.")
        return

    payload = {
        "temperature": temperature,
        "mel_alert_threshold": alert_t,
        "target_sensitivity": args.target_sensitivity,
        "fitted_on": os.path.abspath(args.calib_dir),
        "reported_on": os.path.abspath(args.test_dir),
        "test_metrics_before": before,
        "test_metrics_after": after,
    }
    with open(CALIBRATION_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print("\nWrote " + CALIBRATION_PATH)


if __name__ == "__main__":
    main()
