"""Genuine hold-out evaluation for the DermaScan EfficientNet-B3 classifier.

This script runs the real model over a real labelled dataset and reports the
metrics it actually measures. It does not synthesise figures: if the data is
not supplied, it fails rather than inventing results.

Usage
-----
    python scripts/evaluate_model.py --data-dir path/to/test_set

`--data-dir` must contain one subdirectory per class, named for the class:

    test_set/
      akiec/ *.jpg
      bcc/   *.jpg
      ...    (bkl, df, mel, nv, vasc)

Outputs `docs/evaluation_results.json` plus a measured confusion matrix, and
prints metrics under both sigmoid and softmax readouts so the activation used
at training time can be identified.

Note on training curves: per-epoch loss/F1/accuracy cannot be reconstructed
after the fact. They require the training run's logged history. This script
deliberately does not produce them.
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.ml_engine import SkinCancerPredictor  # noqa: E402

RESULTS_DIR = "docs"
IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def collect_samples(data_dir, classes):
    """Returns [(path, class_index)] for every image under data_dir/<class>/."""
    samples = []
    missing = []
    for idx, cls in enumerate(classes):
        cls_dir = os.path.join(data_dir, cls)
        if not os.path.isdir(cls_dir):
            missing.append(cls)
            continue
        for name in sorted(os.listdir(cls_dir)):
            if name.lower().endswith(IMAGE_EXTS):
                samples.append((os.path.join(cls_dir, name), idx))

    if missing:
        print("  Warning: no directory for class(es): " + ", ".join(missing))
    if not samples:
        raise SystemExit(
            "No images found under {}. Expected one subdirectory per class: {}".format(
                data_dir, ", ".join(classes)
            )
        )
    return samples


@torch.no_grad()
def collect_logits(predictor, samples, batch_size=32):
    """Runs the served model over every sample. Returns (logits, labels)."""
    logits_all, labels_all = [], []
    total = len(samples)

    for start in range(0, total, batch_size):
        batch = samples[start:start + batch_size]
        tensors, labels = [], []
        for path, label in batch:
            try:
                img = Image.open(path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                tensors.append(predictor.transform(img))
                labels.append(label)
            except Exception as e:
                print("  Skipping unreadable image {}: {}".format(path, e))

        if not tensors:
            continue

        x = torch.stack(tensors).to(predictor.device)
        logits_all.append(predictor.model(x).cpu().numpy())
        labels_all.extend(labels)
        print("  Evaluated {}/{}".format(min(start + batch_size, total), total), end="\r")

    print()
    return np.concatenate(logits_all, axis=0), np.array(labels_all)


def softmax(x):
    shifted = x - x.max(axis=1, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=1, keepdims=True)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def predict_with_thresholds(probs, thresholds):
    """Reproduces the serving rule: argmax over (probability - class threshold)."""
    return np.argmax(probs - thresholds[None, :], axis=1)


def score(y_true, y_pred, classes):
    """Per-class precision/recall/F1 and the measured confusion matrix."""
    n = len(classes)
    cm = np.zeros((n, n), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    per_class, f1s = {}, []
    for i, cls in enumerate(classes):
        tp = int(cm[i, i])
        fp = int(cm[:, i].sum() - tp)
        fn = int(cm[i, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[cls] = {
            "support": int(cm[i, :].sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
        f1s.append(f1)

    accuracy = float(np.trace(cm) / cm.sum()) if cm.sum() else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": float(np.mean(f1s)),
        "per_class_metrics": per_class,
        "confusion_matrix": cm.tolist(),
    }


def print_report(title, result, classes):
    print("\n-- {} {}".format(title, "-" * max(0, 46 - len(title))))
    print("  Accuracy : {:.4f}".format(result["accuracy"]))
    print("  Macro-F1 : {:.4f}".format(result["macro_f1"]))
    print("  {:<8}{:>9}{:>9}{:>9}{:>9}".format("class", "support", "prec", "recall", "f1"))
    for cls in classes:
        m = result["per_class_metrics"][cls]
        print("  {:<8}{:>9}{:>9.4f}{:>9.4f}{:>9.4f}".format(
            cls, m["support"], m["precision"], m["recall"], m["f1"]))


def plot_confusion_matrix(cm, classes, path, title):
    """Plots the measured confusion matrix. Row-normalised; empty rows stay 0."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not installed - skipping confusion matrix plot.")
        return

    cm = np.asarray(cm, dtype=np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums > 0)

    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(classes)), classes)
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label="Fraction of true class")

    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, "{:.2f}\n({})".format(norm[i, j], int(cm[i, j])),
                    ha="center", va="center", fontsize=8,
                    color="white" if norm[i, j] > 0.5 else "black")

    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print("  Wrote " + path)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", required=True,
        help="Labelled hold-out set: one subdirectory per class.",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--activation", choices=["both", "sigmoid", "softmax"], default="both",
        help="Readout to score. 'both' (default) reports each, which identifies "
             "the activation the stored thresholds were tuned under: the one "
             "whose macro-F1 matches models/class_thresholds.json.",
    )
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise SystemExit("--data-dir not found: " + args.data_dir)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    predictor = SkinCancerPredictor()
    classes = predictor.classes
    thresholds = np.array([predictor.thresholds[c] for c in classes])

    samples = collect_samples(args.data_dir, classes)
    print("Evaluating {} images from {}".format(len(samples), args.data_dir))

    logits, y_true = collect_logits(predictor, samples, args.batch_size)

    readouts = ["sigmoid", "softmax"] if args.activation == "both" else [args.activation]
    results = {}
    for name in readouts:
        probs = sigmoid(logits) if name == "sigmoid" else softmax(logits)
        results[name] = {
            "thresholded": score(y_true, predict_with_thresholds(probs, thresholds), classes),
            "plain_argmax": score(y_true, np.argmax(probs, axis=1), classes),
        }
        print_report(name + " + per-class thresholds", results[name]["thresholded"], classes)
        print_report(name + " + plain argmax", results[name]["plain_argmax"], classes)

    # Compare against the macro-F1 recorded alongside the thresholds.
    recorded = None
    try:
        with open(os.path.join("models", "class_thresholds.json")) as f:
            recorded = json.load(f).get("optimized_f1_macro")
    except Exception:
        pass

    if recorded is not None and len(readouts) > 1:
        print("\n-- Activation check " + "-" * 40)
        print("  Recorded optimized macro-F1 : {:.4f}".format(recorded))
        for name in readouts:
            measured = results[name]["thresholded"]["macro_f1"]
            print("  {:<28}: {:.4f}   delta {:.4f}".format(
                name + " (thresholded)", measured, abs(measured - recorded)))
        print("  The closer match indicates the activation used at training time.")

    best = readouts[0]
    plot_confusion_matrix(
        results[best]["thresholded"]["confusion_matrix"], classes,
        os.path.join(RESULTS_DIR, "confusion_matrix_measured.png"),
        "Measured confusion matrix ({} + thresholds, n={})".format(best, len(y_true)),
    )

    out = {
        "data_dir": os.path.abspath(args.data_dir),
        "num_samples": int(len(y_true)),
        "classes": classes,
        "thresholds": {c: float(t) for c, t in zip(classes, thresholds)},
        "results": results,
    }
    out_path = os.path.join(RESULTS_DIR, "evaluation_results.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print("\nWrote " + out_path)


if __name__ == "__main__":
    main()
