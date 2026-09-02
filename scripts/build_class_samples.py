"""Build docs/class_samples_real.png from actual HAM10000 images.

Every panel is a real dermoscopic image from the dataset, labelled with its
ground-truth diagnosis taken from the split manifest. Nothing is illustrated or
generated.

    python scripts/build_class_samples.py --data-root path/to/skin_cancer

--data-root must contain data/processed/test.csv and the images it references
(the training repository layout). The figure names each image by its ISIC id so
any panel can be traced back to the dataset row it came from.
"""

import argparse
import os
import sys

CLASSES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
FULL_NAME = {
    "akiec": "Actinic keratosis /\nintraepithelial carcinoma",
    "bcc": "Basal cell carcinoma",
    "bkl": "Benign keratosis",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic nevus",
    "vasc": "Vascular lesion",
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default="D:/ML/skin_cancer",
                    help="Training repository root containing data/processed/test.csv")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "calib"])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from PIL import Image

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    out = args.out or os.path.join(root, "docs", "class_samples_real.png")

    csv = os.path.join(args.data_root, "data", "processed", args.split + ".csv")
    if not os.path.exists(csv):
        raise SystemExit("Split manifest not found: " + csv)

    df = pd.read_csv(csv)
    df["path"] = df["image_path"].apply(
        lambda p: os.path.join(args.data_root, str(p).replace(os.sep, "/")))
    df = df[df["path"].apply(os.path.exists)]
    if df.empty:
        raise SystemExit("No images from the manifest exist on disk under " + args.data_root)

    counts = df["dx"].value_counts().to_dict()

    fig, axes = plt.subplots(2, 4, figsize=(13, 7.4))
    fig.patch.set_facecolor("white")

    for ax, cls in zip(axes.flat, CLASSES):
        rows = df[df["dx"] == cls]
        if rows.empty:
            ax.axis("off")
            continue
        row = rows.iloc[0]
        ax.imshow(Image.open(row["path"]).convert("RGB"))
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_edgecolor("#b9b8b2")
        ax.set_title("%s  \u2014  %s" % (cls, FULL_NAME[cls]),
                     fontsize=9.5, fontweight="bold", color="#17181c", pad=6)
        ax.set_xlabel("%s\n%d images in %s split"
                      % (row["image_id"], counts.get(cls, 0), args.split),
                      fontsize=7.6, color="#5a5f68", family="monospace")

    axes.flat[7].axis("off")
    axes.flat[7].text(
        0.02, 0.92,
        "Real HAM10000 images.\n\n"
        "One example per diagnostic class,\n"
        "drawn from the %s split manifest.\n"
        "Labels are ground truth, not model\n"
        "predictions. Filenames identify the\n"
        "exact dataset row for each panel.\n\n"
        "n = %d images in this split."
        % (args.split, len(df)),
        transform=axes.flat[7].transAxes, va="top", fontsize=8.4,
        color="#41454d", family="monospace", linespacing=1.5)

    fig.suptitle("HAM10000 diagnostic classes \u2014 real dataset examples",
                 fontsize=13.5, fontweight="bold", color="#17181c", y=0.975)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=120, facecolor="white")
    print("Wrote %s (%.0f KB)" % (out, os.path.getsize(out) / 1024))
    for c in CLASSES:
        print("  %-6s %4d images" % (c, counts.get(c, 0)))


if __name__ == "__main__":
    main()
