"""
Build a leak-free copy of a Roboflow YOLOv8 export.

Roboflow sometimes keeps copies of the same photo (`<name>_aug`, `<name>_original`)
in different splits. A copy of a test photo inside the training set makes the
test score look better than it really is. This script groups images by their
source photo and keeps each group in ONE split:

  * a photo that has any copy in train   -> all its copies go to train
  * a photo in both valid and test        -> kept in test only

Usage:
    python scripts/make_clean_split.py <roboflow_export_dir> <output_dir>
The output has the usual train/valid/test layout plus a data.yaml.
"""
import re
import shutil
import sys
from pathlib import Path

SPLITS = ("train", "valid", "test")


def source_id(filename: str) -> str:
    """'Image_123_aug_jpg.rf.abc.jpg' -> 'Image_123'"""
    stem = filename.split(".rf.")[0]
    stem = re.sub(r"_(jpe?g|png)$", "", stem, flags=re.I)
    return re.sub(r"(_aug|_original)+$", "", stem)


def clean_split(src: Path, dst: Path) -> dict:
    files = {s: sorted((src / s / "images").iterdir()) for s in SPLITS}
    owner = {}                                    # source photo -> split it belongs to
    for split in ("train", "test", "valid"):      # priority: train > test > valid
        for img in files[split]:
            owner.setdefault(source_id(img.name), split)

    moved = {s: 0 for s in SPLITS}
    for split in SPLITS:
        for img in files[split]:
            target = owner[source_id(img.name)]
            if target != split:
                moved[split] += 1
            label = src / split / "labels" / (img.stem + ".txt")
            for f, kind in ((img, "images"), (label, "labels")):
                if f.exists():
                    out = dst / target / kind / f.name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(f, out)

    yaml_text = (src / "data.yaml").read_text()
    yaml_text = re.sub(r"(?m)^(train|val|test):.*$", "", yaml_text).strip()
    (dst / "data.yaml").write_text(
        f"path: {dst.resolve().as_posix()}\n"
        "train: train/images\nval: valid/images\ntest: test/images\n" + yaml_text + "\n"
    )
    counts = {s: len(list((dst / s / "images").glob("*"))) for s in SPLITS}
    return {"moved_out_of": moved, "final_counts": counts}


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if dst.exists() and any(dst.iterdir()):
        sys.exit(f"Output folder {dst} is not empty; choose a new folder.")
    print(clean_split(src, dst))
