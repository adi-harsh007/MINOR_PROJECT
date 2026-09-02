"""Fit per-class decision thresholds to the configuration actually served.

The thresholds shipped in `models/class_thresholds.json` were fitted upstream for
a *softmax* readout using a different decision rule. This script fits them for
the rule this backend actually uses -- ``argmax(probability - threshold)`` on the
configured readout -- so the thresholds and the serving code agree.

Thresholds are fitted on a calibration split and must be reported on a separate
test split. Fitting and reporting on the same images overstates performance.

Usage
-----
    python scripts/optimize_thresholds.py \\
        --calib-dir path/to/calib_set \\
        --test-dir  path/to/test_set \\
        --objective macro_f1

Both directories take one subdirectory per class (akiec, bcc, ... vasc).

Objectives
----------
    macro_f1    maximise macro-F1 (default)
    mel_recall  maximise melanoma recall subject to macro-F1 staying within
                --max-f1-drop of the macro_f1 optimum. Use when missing a
                melanoma is judged costlier than a false alarm.

Writes the fitted thresholds only with --write, and never overwrites without
reporting test-set numbers first.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.config import THRESHOLD_PATH  # noqa: E402
from backend.ml_engine import SkinCancerPredictor  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png")
GRID = np.round(np.arange(0.05, 0.96, 0.05), 2)


def collect(data_dir, classes):
    samples = []
    for idx, cls in enumerate(classes):
        d = os.path.join(data_dir, cls)
        if not os.path.isdir(d):
            continue
        for n in sorted(os.listdir(d)):
            if n.lower().endswith(IMAGE_EXTS):
                samples.append((os.path.join(d, n), idx))
    if not samples:
        raise SystemExit("No images found under " + data_dir)
    return samples


@torch.no_grad()
def probabilities(predictor, samples, readout, batch_size=16):
    """Probabilities under the served preprocessing and readout."""
    outs, labels = [], []
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        tensors, labs = [], []
        for path, label in chunk:
            try:
                tensors.append(predictor.transform(Image.open(path).convert("RGB")))
                labs.append(label)
            except Exception as e:
                print("  Skipping {}: {}".format(path, e))
        if not tensors:
            continue
        logits = predictor.model(torch.stack(tensors).to(predictor.device))
        probs = (torch.sigmoid(logits) if readout == "sigmoid"
                 else torch.softmax(logits, dim=1))
        outs.append(probs.cpu().numpy())
        labels.extend(labs)
        print("  {}/{}".format(min(start + batch_size, len(samples)), len(samples)),
              end="\r")
    print()
    return np.concatenate(outs), np.array(labels)


def predict(probs, thresholds):
    """The served decision rule."""
    return np.argmax(probs - thresholds[None, :], axis=1)


def scores(y_true, y_pred, n_classes=7):
    f1s, recalls = [], []
    for i in range(n_classes):
        tp = int(((y_pred == i) & (y_true == i)).sum())
        fp = int(((y_pred == i) & (y_true != i)).sum())
        fn = int(((y_pred != i) & (y_true == i)).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)
        recalls.append(r)
    return {
        "accuracy": float((y_pred == y_true).mean()),
        "macro_f1": float(np.mean(f1s)),
        "per_class_f1": f1s,
        "per_class_recall": recalls,
    }


def coordinate_ascent(probs, y, classes, objective, mel_idx, f1_floor=None,
                      rounds=6):
    """Greedy per-class threshold search against the served rule.

    The margin rule is invariant to adding a constant to every threshold, so the
    search is over relative values only; the grid keeps them in a sane range.
    """
    thresholds = np.full(len(classes), 0.5)

    def value(t):
        s = scores(y, predict(probs, t), len(classes))
        if objective == "macro_f1":
            return s["macro_f1"]
        # mel_recall: only consider points that keep macro-F1 above the floor.
        if f1_floor is not None and s["macro_f1"] < f1_floor:
            return -1.0
        return s["per_class_recall"][mel_idx]

    best = value(thresholds)
    for _ in range(rounds):
        improved = False
        for ci in range(len(classes)):
            current = thresholds[ci]
            for candidate in GRID:
                if candidate == current:
                    continue
                trial = thresholds.copy()
                trial[ci] = candidate
                v = value(trial)
                if v > best + 1e-9:
                    best, thresholds, improved = v, trial, True
        if not improved:
            break
    return thresholds, best


def report(tag, s, classes, mel_idx):
    print("  {:<28} acc {:.4f}  macroF1 {:.4f}  melRecall {:.4f}  melF1 {:.4f}".format(
        tag, s["accuracy"], s["macro_f1"], s["per_class_recall"][mel_idx],
        s["per_class_f1"][mel_idx]))


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib-dir", required=True, help="Split to FIT thresholds on.")
    ap.add_argument("--test-dir", required=True, help="Split to REPORT on.")
    ap.add_argument("--readout", choices=["sigmoid", "softmax"], default="sigmoid",
                    help="Must match the readout in backend/ml_engine.py.")
    ap.add_argument("--objective", choices=["macro_f1", "mel_recall"],
                    default="macro_f1")
    ap.add_argument("--max-f1-drop", type=float, default=0.01,
                    help="For --objective mel_recall: how much calibration macro-F1 "
                         "may fall below its optimum (default 0.01).")
    ap.add_argument("--write", action="store_true",
                    help="Write the fitted thresholds to the threshold file.")
    args = ap.parse_args()

    predictor = SkinCancerPredictor()
    classes = predictor.classes
    mel_idx = classes.index("mel")

    print("Scoring calibration split...")
    cal_p, cal_y = probabilities(predictor, collect(args.calib_dir, classes),
                                 args.readout)
    print("Scoring test split...")
    test_p, test_y = probabilities(predictor, collect(args.test_dir, classes),
                                   args.readout)

    current = np.array([predictor.thresholds[c] for c in classes])

    f1_floor = None
    if args.objective == "mel_recall":
        _, best_f1 = coordinate_ascent(cal_p, cal_y, classes, "macro_f1", mel_idx)
        f1_floor = best_f1 - args.max_f1_drop
        print("\nmacro-F1 optimum on calibration: {:.4f}; floor {:.4f}".format(
            best_f1, f1_floor))

    fitted, _ = coordinate_ascent(cal_p, cal_y, classes, args.objective, mel_idx,
                                  f1_floor)

    print("\nCalibration split (fitted on these - optimistic):")
    report("current thresholds", scores(cal_y, predict(cal_p, current)), classes, mel_idx)
    report("fitted thresholds", scores(cal_y, predict(cal_p, fitted)), classes, mel_idx)

    print("\nTest split (held out - the honest numbers):")
    cur_test = scores(test_y, predict(test_p, current))
    fit_test = scores(test_y, predict(test_p, fitted))
    report("current thresholds", cur_test, classes, mel_idx)
    report("fitted thresholds", fit_test, classes, mel_idx)

    print("\n  {:<8}{:>10}{:>10}".format("class", "current", "fitted"))
    for i, c in enumerate(classes):
        print("  {:<8}{:>10.2f}{:>10.2f}".format(c, current[i], fitted[i]))

    if not args.write:
        print("\nNot written. Re-run with --write to save these thresholds.")
        return

    if fit_test["macro_f1"] < cur_test["macro_f1"] - 0.02:
        print("\nRefusing to write: fitted thresholds lose more than 2 points of "
              "test macro-F1 against the current ones.")
        return

    payload = {
        "fitted_on": os.path.abspath(args.calib_dir),
        "reported_on": os.path.abspath(args.test_dir),
        "readout": args.readout,
        "rule": "argmax(probability - threshold)",
        "objective": args.objective,
        "class_thresholds": {c: float(t) for c, t in zip(classes, fitted)},
        "test_metrics": {
            "accuracy": fit_test["accuracy"],
            "macro_f1": fit_test["macro_f1"],
        },
        "per_class_metrics": {
            c: {
                "threshold": float(fitted[i]),
                "f1": fit_test["per_class_f1"][i],
                "recall": fit_test["per_class_recall"][i],
            }
            for i, c in enumerate(classes)
        },
    }
    with open(THRESHOLD_PATH, "w") as f:
        json.dump(payload, f, indent=2)
    print("\nWrote " + THRESHOLD_PATH)


if __name__ == "__main__":
    main()
