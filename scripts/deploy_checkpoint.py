"""Install a training bundle from training/kaggle_train_dermascan.ipynb into
models/ and docs/, after verifying it can actually be served.

The previous deployment served a checkpoint that differed from the evaluated one
and nobody noticed for months, so nothing here is taken on trust: the weights are
re-fingerprinted, loaded strictly into the architecture the backend will build,
run through a forward pass, and checked against the serving configuration before
a single file is copied. The bundle's own test metrics are checked against the
release gate too - a bundle the notebook refused to recommend is not installed
without --force.

Usage:
    python scripts/deploy_checkpoint.py --from-dir path/to/kaggle/output
    python scripts/deploy_checkpoint.py --from-dir ... --dry-run
    python scripts/deploy_checkpoint.py --rollback
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS = os.path.join(BASE, "models")
DOCS = os.path.join(BASE, "docs")
BACKUP = os.path.join(MODELS, "backup")

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Same criteria the notebook gates on, restated here so a bundle cannot be
# installed just because someone re-ran the export cell past a failing gate.
GATE = {"macro_f1": (0.70, "min"), "melanoma_recall": (0.70, "min"),
        "melanoma_surfaced": (0.90, "min"), "review_rate": (0.45, "max"),
        "ece": (0.10, "max")}

# What lands where. (bundle filename, destination directory, required)
ARTIFACTS = [
    ("dermascan_b3.pt", MODELS, True),          # installed as latest.pt
    ("class_thresholds.json", MODELS, True),
    ("calibration.json", MODELS, True),
    ("evaluation_results.json", DOCS, True),
    ("confusion_matrix_measured.png", DOCS, False),
    ("training_curves.png", DOCS, False),
    ("training_history.csv", DOCS, False),
]


def fail(msg):
    print("REFUSING TO DEPLOY: " + msg)
    sys.exit(1)


def verify_bundle(path):
    """Load the bundle the way the server will, and check every claim it makes."""
    import torch

    print(f"bundle: {path}  ({os.path.getsize(path)/1024/1024:.0f} MB)")
    try:
        ck = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:
        fail(f"torch.load(weights_only=True) failed: {e}\n"
             "  The bundle contains something that is not a tensor or a plain "
             "primitive - most likely a numpy scalar in its metrics.")

    required = ["arch", "head", "classes", "img_size", "norm_mean", "norm_std",
                "readout", "temperature", "thresholds", "mel_alert_threshold",
                "model_state_dict", "weight_fingerprint"]
    missing = [k for k in required if k not in ck]
    if missing:
        fail(f"bundle is missing {missing}. This is not a bundle from the "
             f"current notebook - deploy it by hand if you know what it is.")

    if ck["classes"] != CLASSES:
        fail(f"class order is {ck['classes']}, expected {CLASSES}. Every "
             f"threshold and every stored metric is indexed by this order.")

    state = ck["model_state_dict"]
    fingerprint = hashlib.sha256(
        b"".join(state[k].numpy().tobytes() for k in sorted(state))).hexdigest()[:16]
    if fingerprint != ck["weight_fingerprint"]:
        fail(f"weight fingerprint mismatch: bundle says {ck['weight_fingerprint']}, "
             f"the weights hash to {fingerprint}. The file has been altered since "
             f"export.")
    print(f"  fingerprint {fingerprint} verified")
    print(f"  run {ck.get('run_id', '(unrecorded)')}, epoch "
          f"{ck.get('best_epoch', '?')}, weights {ck.get('weights', 'raw')}")

    # Build the architecture the BACKEND will build, not the one we assume.
    sys.path.insert(0, BASE)
    from backend.model import build_model
    from backend import config

    if ck["arch"] != "efficientnet_b3":
        fail(f"bundle arch is {ck['arch']}; backend/model.py builds "
             f"efficientnet_b3 for both MODEL_ARCH settings.")
    if ck["head"] != config.MODEL_ARCH:
        fail(f"bundle head is {ck['head']!r} but MODEL_ARCH is "
             f"{config.MODEL_ARCH!r}. Set MODEL_ARCH={ck['head']} in .env first, "
             f"or the server builds a different network than was evaluated.")

    model = build_model(config.MODEL_ARCH, num_classes=len(CLASSES))
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in state.items()})
    model.eval()
    probe = torch.randn(1, 3, ck["img_size"], ck["img_size"])
    with torch.no_grad():
        out = model(probe)
    if tuple(out.shape) != (1, len(CLASSES)):
        fail(f"forward pass produced {tuple(out.shape)}, expected (1, {len(CLASSES)})")
    print(f"  strict load into backend build_model({config.MODEL_ARCH!r}): OK")
    print(f"  CPU forward pass at {ck['img_size']}px: OK")

    if ck["img_size"] != config.IMG_SIZE:
        fail(f"bundle was evaluated at {ck['img_size']}px but IMG_SIZE is "
             f"{config.IMG_SIZE}. Set IMG_SIZE={ck['img_size']} in .env first. "
             f"Serving at a different resolution than evaluation is the exact "
             f"bug that cost the last deployment 20 points of accuracy.")
    print(f"  img_size {ck['img_size']} matches IMG_SIZE")
    return ck


def verify_sidecars(src_dir, ck):
    """The server reads thresholds and calibration from JSON, not the bundle."""
    th_path = os.path.join(src_dir, "class_thresholds.json")
    with open(th_path) as f:
        th = json.load(f)
    try:
        served = {c: th["per_class_metrics"][c]["threshold"] for c in CLASSES}
    except KeyError as e:
        fail(f"class_thresholds.json has no per_class_metrics{e} - this is the "
             f"path backend/ml_engine.py reads, and it refuses to serve without it.")
    for c in CLASSES:
        if abs(served[c] - ck["thresholds"][c]) > 1e-6:
            fail(f"threshold for {c} is {served[c]} in the sidecar but "
                 f"{ck['thresholds'][c]} in the bundle.")
    print("  class_thresholds.json agrees with the bundle on all 7 thresholds")

    with open(os.path.join(src_dir, "calibration.json")) as f:
        cal = json.load(f)
    for key, want in [("temperature", ck["temperature"]),
                      ("mel_alert_threshold", ck["mel_alert_threshold"]),
                      ("readout", ck["readout"])]:
        got = cal.get(key)
        if isinstance(want, float) and isinstance(got, (int, float)):
            ok = abs(got - want) < 1e-6
        else:
            ok = got == want
        if not ok:
            fail(f"calibration.json {key}={got!r}, bundle says {want!r}")
    print(f"  calibration.json agrees: readout={cal['readout']}, "
          f"T={cal['temperature']}, mel alert p>={cal['mel_alert_threshold']}")

    if cal["readout"] != ck["readout"]:
        fail("readout mismatch between bundle and calibration.json")
    return cal


def check_gate(src_dir, force):
    with open(os.path.join(src_dir, "evaluation_results.json")) as f:
        ev = json.load(f)
    print(f"\nrelease gate, on {ev.get('test_set_size', '?')} held-out images:")
    failures = []
    for key, (bound, kind) in GATE.items():
        got = ev.get(key)
        if got is None:
            failures.append(f"{key} not reported")
            print(f"  {key:<20} {'(not reported)':>12}   FAIL")
            continue
        ok = got >= bound if kind == "min" else got <= bound
        rel = ">=" if kind == "min" else "<="
        if not ok:
            failures.append(key)
        print(f"  {key:<20} {rel}{bound:<7.3f} {got:>8.4f}   {'pass' if ok else 'FAIL'}")
    if failures:
        if not force:
            fail("the bundle does not meet the release gate: " + ", ".join(failures) +
                 "\n  Retrain. Pass --force only if you are deliberately deploying "
                 "a model that failed it, and write down why.")
        print("  --force: installing a bundle that FAILED the gate")
    else:
        print("  gate: PASS")
    return ev


def backup_current(stamp):
    os.makedirs(BACKUP, exist_ok=True)
    saved = []
    for name, dest, _ in ARTIFACTS:
        live = os.path.join(MODELS, "latest.pt") if name.endswith(".pt") \
            else os.path.join(dest, name)
        if os.path.exists(live):
            target = os.path.join(BACKUP, f"{stamp}_{os.path.basename(live)}")
            shutil.copy2(live, target)
            saved.append(os.path.relpath(target, BASE))
    return saved


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", help="directory holding the Kaggle output files")
    ap.add_argument("--dry-run", action="store_true",
                    help="verify everything, copy nothing")
    ap.add_argument("--force", action="store_true",
                    help="install even if the release gate fails")
    ap.add_argument("--rollback", metavar="STAMP", nargs="?", const="list",
                    help="restore a backup set, or list them with no argument")
    args = ap.parse_args()

    if args.rollback:
        stamps = sorted({f.split("_")[0] for f in os.listdir(BACKUP)
                         if re.match(r"^\d{8}-\d{6}_", f)}) \
            if os.path.isdir(BACKUP) else []
        if args.rollback == "list":
            print("backup sets:", ", ".join(stamps) or "(none)")
            return
        if args.rollback not in stamps:
            fail(f"no backup set {args.rollback}. Available: {stamps}")
        for f in os.listdir(BACKUP):
            if not f.startswith(args.rollback + "_"):
                continue
            name = f.split("_", 1)[1]
            dest = DOCS if name in ("evaluation_results.json",
                                    "confusion_matrix_measured.png",
                                    "training_curves.png",
                                    "training_history.csv") else MODELS
            shutil.copy2(os.path.join(BACKUP, f), os.path.join(dest, name))
            print("restored", name)
        print("\nRestart the server for the change to take effect.")
        return

    if not args.from_dir:
        ap.error("--from-dir is required (or use --rollback)")
    src = os.path.abspath(args.from_dir)
    if not os.path.isdir(src):
        fail(f"{src} is not a directory")

    absent = [n for n, _, req in ARTIFACTS if req
              and not os.path.exists(os.path.join(src, n))]
    if absent:
        fail(f"{src} is missing {absent}.\n"
             "  Download the whole Output panel from the Kaggle run, not just "
             "the .pt - the thresholds and calibration are what make the "
             "checkpoint servable.")

    ck = verify_bundle(os.path.join(src, "dermascan_b3.pt"))
    verify_sidecars(src, ck)
    ev = check_gate(src, args.force)

    print()
    if args.dry_run:
        print("--dry-run: verification passed, nothing copied.")
        return

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    saved = backup_current(stamp)
    print(f"backed up {len(saved)} file(s) as {stamp}_*  in models/backup/")

    for name, dest, _ in ARTIFACTS:
        s = os.path.join(src, name)
        if not os.path.exists(s):
            continue
        target = os.path.join(MODELS, "latest.pt") if name.endswith(".pt") \
            else os.path.join(dest, name)
        shutil.copy2(s, target)
        print("  installed", os.path.relpath(target, BASE))

    print(f"\ndeployed run {ck.get('run_id', '?')} "
          f"(fingerprint {ck['weight_fingerprint']}, "
          f"macro-F1 {ev.get('macro_f1', float('nan')):.4f}, "
          f"melanoma recall {ev.get('melanoma_recall', float('nan')):.4f})")
    print("\nNext:")
    print("  1. Restart the server. backend/ml_engine.py takes readout={} from"
          .format(ck["readout"]))
    print("     calibration.json, so no .env change is needed - but if READOUT is")
    print("     set there it is only a fallback, and a stale value will mislead")
    print("     anyone reading the config.")
    print("  2. python -m pytest")
    print("  3. python scripts/evaluate_model.py --data-dir <test images>")
    print("     Its numbers must match docs/evaluation_results.json. If they do")
    print("     not, serving and training have diverged - stop and find out why.")
    print(f"\nRollback:  python scripts/deploy_checkpoint.py --rollback {stamp}")


if __name__ == "__main__":
    main()
