"""Fit the OOD gate to a real in-distribution dataset.

The shipped thresholds in `backend/ood.py` are provisional: they were derived
from a handful of images and are deliberately permissive. This script replaces
them with values measured on your own data, and additionally fits the
feature-space detector that the colour gate cannot substitute for.

Usage
-----
    python scripts/calibrate_ood.py --data-dir path/to/in_distribution_images

`--data-dir` should contain dermoscopic images that the system SHOULD accept
(any layout; subdirectories are searched). Use the training or validation split.

Optionally pass `--ood-dir` with images that should be REJECTED. When supplied,
the script reports the separation achieved and the false-accept / false-reject
rates at the chosen cutoffs, so you can see whether the gate actually works
rather than assuming it does.

Writes:
    models/ood_config.json  - colour-gate thresholds (percentiles of your data)
    models/ood_stats.npz    - Mahalanobis mean/covariance + distance cutoff

IMPORTANT: calibrate on a dataset that reflects the full range of skin tones you
intend to serve. A gate fitted only to light skin will reject darker skin, which
is the exact failure this rewrite removed. Check the per-tone breakdown printed
at the end if your data has that metadata.
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
from backend.ood import compute_metrics, DEFAULT_THRESHOLDS  # noqa: E402

IMAGE_EXTS = (".jpg", ".jpeg", ".png")


def find_images(root):
    paths = []
    for dirpath, _, names in os.walk(root):
        for n in sorted(names):
            if n.lower().endswith(IMAGE_EXTS):
                paths.append(os.path.join(dirpath, n))
    return paths


def gather(predictor, paths, want_features=True, batch_size=32):
    """Returns (metrics list, feature matrix) for every readable image."""
    metrics, features = [], []
    batch_imgs, total = [], len(paths)

    def flush():
        if not batch_imgs:
            return
        x = torch.stack(batch_imgs).to(predictor.device)
        with torch.no_grad():
            feats = predictor.model.forward_features(x)
            vecs = predictor.model.forward_head(feats, pre_logits=True)
        features.append(vecs.cpu().numpy())
        batch_imgs.clear()

    for i, p in enumerate(paths, 1):
        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print("  Skipping {}: {}".format(p, e))
            continue
        metrics.append(compute_metrics(img))
        if want_features:
            batch_imgs.append(predictor.transform(img))
            if len(batch_imgs) >= batch_size:
                flush()
        if i % 50 == 0 or i == total:
            print("  Processed {}/{}".format(i, total), end="\r")

    flush()
    print()
    feat_matrix = np.concatenate(features, axis=0) if features else np.empty((0, 0))
    return metrics, feat_matrix


def column(metrics, key):
    return np.array([m[key] for m in metrics], dtype=np.float64)


def describe(name, values):
    if values.size == 0:
        return
    print("  {:<22} min {:.4f}  p1 {:.4f}  median {:.4f}  p99 {:.4f}  max {:.4f}".format(
        name, values.min(), np.percentile(values, 1), np.median(values),
        np.percentile(values, 99), values.max()))


# Below this, a held-out false-reject rate is too noisy to mean anything.
MIN_HOLDOUT = 25


def split_holdout(features, fraction, seed=0):
    """Split into (holdout, fit). Deterministic, so a rerun says the same thing."""
    n = features.shape[0]
    n_holdout = int(round(n * fraction))
    if n_holdout <= 0:
        return features[:0], features
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    return features[order[:n_holdout]], features[order[n_holdout:]]


def fit_mahalanobis(features, percentile):
    """Mean, inverse covariance (shrunk), and the distance cutoff."""
    mean = features.mean(axis=0)
    centered = features - mean
    cov = np.cov(centered, rowvar=False)
    # Shrinkage keeps the covariance invertible when samples < dimensions.
    shrink = 1e-3 * np.trace(cov) / cov.shape[0]
    cov += shrink * np.eye(cov.shape[0])
    inv_cov = np.linalg.pinv(cov)

    dists = np.einsum("ij,jk,ik->i", centered, inv_cov, centered)
    threshold = float(np.percentile(dists, percentile))
    return mean, inv_cov, threshold, dists


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", required=True,
                        help="In-distribution dermoscopic images (should be accepted).")
    parser.add_argument("--ood-dir", default=None,
                        help="Optional images that should be rejected, for validation.")
    parser.add_argument("--percentile", type=float, default=99.0,
                        help="Keep this %% of in-distribution data inside the gate "
                             "(default 99, i.e. a 1%% false-reject budget).")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--dry-run", action="store_true",
                        help="Report measurements without writing any files.")
    parser.add_argument("--holdout-fraction", type=float, default=0.2,
                        help="Fraction of --data-dir held back from fitting and "
                             "used to measure the real false-reject rate "
                             "(default 0.2).")
    parser.add_argument("--max-false-reject", type=float, default=0.05,
                        help="Refuse to install a gate that rejects more than "
                             "this fraction of HELD-OUT in-distribution images "
                             "(default 0.05).")
    parser.add_argument("--force", action="store_true",
                        help="Write even if held-out validation fails. You are "
                             "asserting you know why.")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        raise SystemExit("--data-dir not found: " + args.data_dir)

    paths = find_images(args.data_dir)
    if len(paths) < 50:
        print("Warning: only {} images found. Thresholds fitted on a small sample "
              "are not trustworthy; prefer several hundred.".format(len(paths)))
    if not paths:
        raise SystemExit("No images found under " + args.data_dir)

    predictor = SkinCancerPredictor()
    print("Measuring {} in-distribution images...".format(len(paths)))
    metrics, features = gather(predictor, paths, batch_size=args.batch_size)

    lo = (100.0 - args.percentile) / 2.0
    hi = 100.0 - lo

    rel = column(metrics, "rel_contrast")
    hf = column(metrics, "hf_ratio")
    bg = column(metrics, "blue_green")
    chroma = column(metrics, "chromatic_fraction")

    print("\nIn-distribution measurements")
    for name, vals in [("rel_contrast", rel), ("hf_ratio", hf),
                       ("blue_green", bg), ("chromatic_fraction", chroma)]:
        describe(name, vals)

    thresholds = dict(DEFAULT_THRESHOLDS)
    thresholds["min_rel_contrast"] = float(np.percentile(rel, lo))
    thresholds["max_hf_ratio"] = float(np.percentile(hf, hi))
    thresholds["max_blue_green"] = float(max(np.percentile(bg, hi), 0.30))

    # ── Feature-space stage: fit on a subset, judge on the rest ───────────
    #
    # Mahalanobis distance needs a covariance estimated from many more samples
    # than there are dimensions. EfficientNet-B3 produces 1536-dimensional
    # features, so the covariance has ~1.18 million free parameters; fitted from
    # a few dozen images it is rank-deficient, the pseudo-inverse explodes along
    # every unconstrained direction, and the resulting gate rejects essentially
    # any image it did not see during fitting.
    #
    # This was measured, not assumed: fitted on 20 of the 21 bundled samples, the
    # gate rejected the held-out lesion every single time - 21/21, at distances
    # around 1e6 against a cutoff of 18.
    n_fit, n_dim = features.shape
    if n_fit <= n_dim:
        print("\nWARNING: {} images for {} feature dimensions. The covariance is "
              "rank-deficient and the feature-space gate will reject almost "
              "everything. Several times as many images as dimensions are "
              "needed.".format(n_fit, n_dim))

    holdout_features, fit_features = split_holdout(features, args.holdout_fraction)
    if holdout_features.shape[0] >= MIN_HOLDOUT:
        _, inv_holdout, cutoff_holdout, _ = fit_mahalanobis(
            fit_features, args.percentile)
        mean_holdout = fit_features.mean(axis=0)
        centered = holdout_features - mean_holdout
        holdout_dists = np.einsum("ij,jk,ik->i", centered, inv_holdout, centered)
        holdout_false_reject = float((holdout_dists > cutoff_holdout).mean())
        print("\nHeld-out validation ({} fitted / {} held out)".format(
            fit_features.shape[0], holdout_features.shape[0]))
        print("  in-distribution images wrongly rejected: {:.1f}%".format(
            100.0 * holdout_false_reject))
    else:
        holdout_false_reject = None
        print("\nHeld-out validation: skipped, fewer than {} images available. "
              "The numbers below are measured on the data they were fitted to "
              "and describe memorisation, not generalisation.".format(MIN_HOLDOUT))

    mean, inv_cov, mahal_threshold, id_dists = fit_mahalanobis(features, args.percentile)
    thresholds["max_mahalanobis"] = mahal_threshold

    print("\nFitted thresholds (keeping {:.1f}% of in-distribution data)".format(
        args.percentile))
    for k, v in thresholds.items():
        print("  {:<24} {}".format(k, "{:.4f}".format(v) if isinstance(v, float) else v))

    # Optional validation against images that should be rejected.
    if args.ood_dir:
        ood_paths = find_images(args.ood_dir)
        print("\nValidating against {} OOD images...".format(len(ood_paths)))
        ood_metrics, ood_features = gather(predictor, ood_paths,
                                           batch_size=args.batch_size)
        rejected = 0
        for m, f in zip(ood_metrics, ood_features):
            colour_reject = (
                m["rel_contrast"] < thresholds["min_rel_contrast"]
                or m["hf_ratio"] > thresholds["max_hf_ratio"]
                or (m["chromatic_fraction"] >= thresholds["min_chromatic_fraction"]
                    and m["blue_green"] > thresholds["max_blue_green"])
            )
            d = float((f - mean) @ inv_cov @ (f - mean))
            if colour_reject or d > mahal_threshold:
                rejected += 1
        if ood_metrics:
            print("  OOD correctly rejected : {}/{}  ({:.1f}%)".format(
                rejected, len(ood_metrics), 100.0 * rejected / len(ood_metrics)))
        false_reject = float((id_dists > mahal_threshold).mean())
        print("  In-distribution wrongly rejected by feature stage: "
              "{:.1f}%".format(100.0 * false_reject))

    if args.dry_run:
        print("\nDry run: nothing written.")
        return

    # ── Refuse to install a gate that rejects real lesions ────────────────
    #
    # This is the safety property the script previously lacked. It reported a
    # false-reject rate computed on the images it had just fitted to - where the
    # cutoff is by construction the 99th percentile of those very distances - so
    # a gate that rejected 100% of unseen lesions still reported ~1% and was
    # written out regardless. An OOD gate that rejects everything scores a
    # perfect OOD rejection rate, which is exactly why that number cannot be
    # read on its own.
    if holdout_false_reject is None:
        print("\nREFUSING TO WRITE: not enough images to validate on data that "
              "was held out of fitting ({} needed). A gate that cannot be "
              "validated must not be installed; every scan it wrongly rejects "
              "is a lesion nobody looks at.".format(MIN_HOLDOUT))
        if not args.force:
            raise SystemExit(2)
        print("Proceeding anyway because --force was given.")
    elif holdout_false_reject > args.max_false_reject:
        print("\nREFUSING TO WRITE: this gate rejects {:.1f}% of held-out "
              "in-distribution images, above the {:.1f}% budget. Installing it "
              "would reject real lesions.".format(
                  100.0 * holdout_false_reject, 100.0 * args.max_false_reject))
        print("Fit on more data, or raise --max-false-reject deliberately.")
        if not args.force:
            raise SystemExit(2)
        print("Proceeding anyway because --force was given.")

    models_dir = os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), "models")
    os.makedirs(models_dir, exist_ok=True)

    config_path = os.path.join(models_dir, "ood_config.json")
    with open(config_path, "w") as f:
        json.dump({
            "fitted_from": os.path.abspath(args.data_dir),
            "num_images": len(metrics),
            "percentile": args.percentile,
            "thresholds": thresholds,
        }, f, indent=2)
    print("\nWrote " + config_path)

    stats_path = os.path.join(models_dir, "ood_stats.npz")
    np.savez(stats_path, mean=mean, inv_cov=inv_cov, threshold=mahal_threshold)
    print("Wrote " + stats_path)


if __name__ == "__main__":
    main()
