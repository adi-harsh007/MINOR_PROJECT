"""Populate samples/ with labelled test images, one set per diagnostic class.

Images are taken from the **held-out test split** — the model never saw them
during training, so predictions on them are meaningful. Selection is by manifest
order, not by whether the model gets them right: picking images the model happens
to classify correctly would make the sample set flatter than reality.

    python scripts/build_test_samples.py --per-class 3

Files are named `<class>_<n>_<ISIC id>.jpg` so the ground truth is visible in the
filename and every image is traceable to its dataset row. A README manifest is
written alongside them.
"""

import argparse
import os
import shutil

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
FULL_NAME = {
    "akiec": "Actinic keratosis / intraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}
RISK = {
    "akiec": "Pre-malignant",
    "bcc": "Malignant",
    "bkl": "Benign",
    "df": "Benign",
    "mel": "Malignant",
    "nv": "Benign",
    "vasc": "Benign",
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="D:/ML/skin_cancer",
                    help="Training repository containing data/processed/test.csv")
    ap.add_argument("--per-class", type=int, default=3)
    ap.add_argument("--split", default="test", choices=["test", "calib", "val"],
                    help="Held-out split to draw from. Never use 'train'.")
    args = ap.parse_args()

    import pandas as pd
    from PIL import Image

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out_dir = os.path.join(root, "samples")
    os.makedirs(out_dir, exist_ok=True)

    csv = os.path.join(args.data_root, "data", "processed", args.split + ".csv")
    if not os.path.exists(csv):
        raise SystemExit("Split manifest not found: " + csv)

    df = pd.read_csv(csv)
    df["path"] = df["image_path"].apply(
        lambda p: os.path.join(args.data_root, str(p).replace(os.sep, "/")))
    df = df[df["path"].apply(os.path.exists)]
    if df.empty:
        raise SystemExit("No images from the manifest are present under " + args.data_root)

    # Drop any file this script wrote previously so re-runs don't accumulate.
    for name in os.listdir(out_dir):
        if any(name.startswith(c + "_") for c in CLASSES):
            os.remove(os.path.join(out_dir, name))

    written, missing = [], []
    for cls in CLASSES:
        rows = df[df["dx"] == cls]
        if rows.empty:
            missing.append(cls)
            continue
        take = rows.head(args.per_class)
        if len(take) < args.per_class:
            print("  %s: only %d image(s) available in the %s split"
                  % (cls, len(take), args.split))
        for i, (_, row) in enumerate(take.iterrows(), 1):
            name = "%s_%d_%s.jpg" % (cls, i, row["image_id"])
            dest = os.path.join(out_dir, name)
            shutil.copy(row["path"], dest)
            with Image.open(dest) as im:
                size = im.size
            written.append((cls, name, row["image_id"], size))

    lines = [
        "# Test Samples",
        "",
        "Labelled dermoscopic images for exercising the classifier.",
        "",
        "**These are from the held-out `%s` split.** The model was not trained on them, so"
        % args.split,
        "predictions here reflect real generalisation. They were selected by manifest order,",
        "**not** by whether the model classifies them correctly — a set curated on model",
        "success would look better than the model actually is.",
        "",
        "## Naming",
        "",
        "`<class>_<n>_<ISIC id>.jpg` — the prefix is the ground-truth diagnosis and the",
        "ISIC id identifies the exact dataset row.",
        "",
        "| Code | Diagnosis | Risk | Files |",
        "| :--- | :--- | :--- | ---: |",
    ]
    for cls in CLASSES:
        n = sum(1 for w in written if w[0] == cls)
        lines.append("| `%s` | %s | %s | %d |" % (cls, FULL_NAME[cls], RISK[cls], n))

    lines += [
        "",
        "## Files",
        "",
        "| File | Ground truth | ISIC id |",
        "| :--- | :--- | :--- |",
    ]
    for cls, name, iid, _ in written:
        lines.append("| `%s` | %s (`%s`) | %s |" % (name, FULL_NAME[cls], cls, iid))

    lines += [
        "",
        "## Other files",
        "",
        "| File | Purpose |",
        "| :--- | :--- |",
        "| `cat.jpg` | Non-skin control. Should be **rejected** by the OOD gate, not classified. |",
        "| `nv.jpg`, `ISIC_0024307.jpg` | Earlier reference images kept because the sample gallery links to them. |",
        "",
        "## Regenerating",
        "",
        "```bash",
        "python scripts/build_test_samples.py --per-class %d" % args.per_class,
        "```",
        "",
        "Re-running replaces the class-prefixed files and leaves everything else alone.",
        "",
    ]
    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    print("Wrote %d images to samples/" % len(written))
    for cls in CLASSES:
        got = [w for w in written if w[0] == cls]
        print("  %-6s %d file(s)%s" % (cls, len(got),
              "" if got else "   <-- none available"))
    if missing:
        print("No images for: " + ", ".join(missing))


if __name__ == "__main__":
    main()
