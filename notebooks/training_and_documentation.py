# -*- coding: utf-8 -*-
"""PrintVision AI - training and documentation notebook (Google Colab export).

This file is the .py export of the Colab notebook used to download the
Roboflow dataset, train YOLOv8n, save the final weights to Google Drive
and record the experiment. It contains Colab-only lines (`!pip`, drive.mount)
and is meant to be read or run cell by cell in Colab, not as a script.

#LIBRARIES
"""

# Commented out IPython magic to ensure Python compatibility.
# %pip install -q ultralytics pyyaml streamlit pyngrok opencv-python-headless

from google.colab import drive
drive.mount("/content/drive")

!pip install roboflow

import os
from getpass import getpass
from roboflow import Roboflow

# Never hard-code the key. Set ROBOFLOW_API_KEY (Colab: Secrets panel) or type it when asked.
ROBOFLOW_API_KEY = os.environ.get("ROBOFLOW_API_KEY") or getpass("Roboflow API key: ")
rf = Roboflow(api_key=ROBOFLOW_API_KEY)
project = rf.workspace("abilesh-athimulam").project("my-first-project-krsyz")
version = project.version(5)
dataset = version.download("yolov8")

from pathlib import Path
from collections import Counter
import yaml
import os
import json

from ultralytics import YOLO

# Dataset
DATASET_DIR = Path("/content/My-First-Project-5")
DATA_YAML = DATASET_DIR / "data.yaml"

# Permanent training output location
PROJECT_DIR = Path("/content/drive/MyDrive/PrintVision_Training")

# Existing trained model from your previous run
EXISTING_MODEL = (
    PROJECT_DIR / "detection_baseline" / "weights" / "best.pt"
)

print("Dataset exists:", DATASET_DIR.exists())
print("Dataset YAML exists:", DATA_YAML.exists())
print("Existing model exists:", EXISTING_MODEL.exists())
print("Training output:", PROJECT_DIR)

if not DATA_YAML.exists():
    raise FileNotFoundError(f"Dataset YAML not found: {DATA_YAML}")

with open(DATA_YAML, "r") as f:
    data_config = yaml.safe_load(f)

print(yaml.safe_dump(data_config, sort_keys=False))

print("\nDataset classes:")
names = data_config["names"]

if isinstance(names, dict):
    for class_id, class_name in names.items():
        print(f"{class_id}: {class_name}")
else:
    for class_id, class_name in enumerate(names):
        print(f"{class_id}: {class_name}")

"""#image and label counts"""

image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

splits = ["train", "valid", "test"]

for split in splits:
    images_dir = DATASET_DIR / split / "images"
    labels_dir = DATASET_DIR / split / "labels"

    print(f"\n--- {split.upper()} ---")

    if not images_dir.exists() or not labels_dir.exists():
        print("Missing images or labels directory")
        continue

    images = [
        p for p in images_dir.iterdir()
        if p.suffix.lower() in image_extensions
    ]

    labels = list(labels_dir.glob("*.txt"))
    label_stems = {p.stem for p in labels}
    image_stems = {p.stem for p in images}

    missing_labels = image_stems - label_stems
    orphan_labels = label_stems - image_stems

    print("Images:", len(images))
    print("Label files:", len(labels))
    print("Images without label files:", len(missing_labels))
    print("Labels without images:", len(orphan_labels))

"""#Mix up"""

from collections import Counter

nc = int(data_config.get("nc", len(data_config["names"])))

for split in splits:
    labels_dir = DATASET_DIR / split / "labels"

    if not labels_dir.exists():
        continue

    row_types = Counter()
    class_counts = Counter()
    invalid_rows = []
    empty_files = 0

    for label_file in labels_dir.glob("*.txt"):
        lines = [
            line.strip()
            for line in label_file.read_text().splitlines()
            if line.strip()
        ]

        if not lines:
            empty_files += 1
            continue

        for line_number, line in enumerate(lines, start=1):
            parts = line.split()

            try:
                values = [float(v) for v in parts]
            except ValueError:
                invalid_rows.append(
                    (label_file.name, line_number, "Non-numeric values")
                )
                continue

            if len(values) < 5 or (len(values) - 1) % 2 != 0 and len(values) > 5:
                invalid_rows.append(
                    (label_file.name, line_number, "Unexpected number of values")
                )
                continue

            class_id = values[0]

            if not class_id.is_integer() or not 0 <= int(class_id) < nc:
                invalid_rows.append(
                    (label_file.name, line_number, "Invalid class ID")
                )
                continue

            coords = values[1:]

            if any(v < 0 or v > 1 for v in coords):
                invalid_rows.append(
                    (label_file.name, line_number, "Coordinate outside [0, 1]")
                )
                continue

            if len(values) == 5:
                row_types["detection boxes"] += 1
            elif len(values) >= 7:
                row_types["segmentation polygons"] += 1
            else:
                invalid_rows.append(
                    (label_file.name, line_number, "Too few polygon coordinates")
                )
                continue

            class_counts[int(class_id)] += 1

    print(f"\n--- {split.upper()} LABEL AUDIT ---")
    print("Row formats:", dict(row_types))
    print("Empty label files:", empty_files)
    print("Class counts:", dict(class_counts))
    print("Invalid rows:", len(invalid_rows))

    for item in invalid_rows[:10]:
        print("  ", item)

    if (
        row_types["detection boxes"] > 0
        and row_types["segmentation polygons"] > 0
    ):
        print("WARNING: Mixed detection and segmentation annotations detected.")

for split in splits:
    labels_dir = DATASET_DIR / split / "labels"

    if not labels_dir.exists():
        continue

    counts = Counter()

    for label_file in labels_dir.glob("*.txt"):
        for line in label_file.read_text().splitlines():
            parts = line.split()
            if not parts:
                continue

            try:
                class_id = int(float(parts[0]))
                counts[class_id] += 1
            except ValueError:
                pass

    print(f"\n{split.upper()} class distribution:")

    for class_id, count in sorted(counts.items()):
        print(f"  {class_id}: {data_config['names'][class_id] if isinstance(data_config['names'], list) else data_config['names'].get(class_id, data_config['names'].get(str(class_id), 'unknown'))} — {count}")

"""#YOLOv8n for 50 epochs"""

MODEL_NAME = "yolov8n.pt"

results = YOLO(MODEL_NAME).train(
    data=str(DATA_YAML),
    epochs=30,
    imgsz=768,
    batch=8,
    patience=15,
    project=str(PROJECT_DIR),
    name="detection_baseline_v2_768",
    exist_ok=False,
    seed=0
)

BEST_MODEL = (
    PROJECT_DIR / "detection_baseline_v2_768" / "weights" / "best_2.pt"
)

print("Training completed.")
print("Best checkpoint:", BEST_MODEL)
print("Checkpoint exists:", BEST_MODEL.exists())

from google.colab import drive
drive.mount('/content/gdrive')

from pathlib import Path

SEARCH_DIR = Path("/content/gdrive/MyDrive/PrintVision_Training")

print("Training folder exists:", SEARCH_DIR.exists())

if SEARCH_DIR.exists():
    for p in SEARCH_DIR.rglob("best.pt"):
        print(p)

from ultralytics import YOLO
from pathlib import Path

BEST_MODEL = Path(
    "/content/gdrive/MyDrive/PrintVision_Training/"
    "detection_baseline_v2-5/weights/best.pt"
)

if not BEST_MODEL.is_file():
    raise FileNotFoundError(f"Checkpoint not found: {BEST_MODEL}")

model = YOLO(str(BEST_MODEL))

print("Loaded checkpoint:", BEST_MODEL)
print("Model classes:", model.names)

import shutil
import json

FINAL_DIR = Path("/content/drive/MyDrive/PrintVision_Final")
FINAL_DIR.mkdir(parents=True, exist_ok=True)

final_model_path = FINAL_DIR / "best.pt"
shutil.copy2(BEST_MODEL, final_model_path)

metadata = {
    "model": str(final_model_path),
    "dataset_yaml": str(DATA_YAML),
    "input_size": 640,
    "target_defect_classes": {
        "0": "crack",
        "1": "failed print",
        "5": "stringing defect"
    },
    "all_model_classes": {
        str(k): v for k, v in model.names.items()
    }
}

with open(FINAL_DIR / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("Final model saved to:", final_model_path)
print("Metadata saved to:", FINAL_DIR / "model_metadata.json")

"""#Streamlit INterface"""

# Commented out IPython magic to ensure Python compatibility.
# %%writefile /content/printvision_app.py
# import streamlit as st
# from ultralytics import YOLO
# from PIL import Image, ImageDraw
# from pathlib import Path
# 
# st.set_page_config(
#     page_title="PrintVision AI",
#     page_icon="🔬",
#     layout="wide"
# )
# 
# MODEL_PATH = "/content/gdrive/MyDrive/PrintVision_Training/detection_baseline_v2-5/weights/best.pt"
# 
# # IDs from the six-class checkpoint
# DEFECT_CLASSES = {
#     0: "Crack",
#     1: "Print failed",
#     5: "Stringing defect"
# }
# 
# @st.cache_resource
# def load_model():
#     if not Path(MODEL_PATH).exists():
#         raise FileNotFoundError(f"Model not found: {MODEL_PATH}")
#     return YOLO(MODEL_PATH)
# 
# st.title("PrintVision AI")
# st.caption("AI-assisted 3D-print defect inspection")
# 
# with st.sidebar:
#     st.header("Inspection settings")
#     confidence = st.slider(
#         "Detection confidence threshold",
#         min_value=0.05,
#         max_value=0.90,
#         value=0.25,
#         step=0.05
#     )
#     image_size = st.selectbox("Inference image size", [640, 800, 1024], index=0)
# 
# uploaded_file = st.file_uploader(
#     "Upload a 3D-printed object image",
#     type=["jpg", "jpeg", "png", "webp"]
# )
# 
# try:
#     model = load_model()
#     st.sidebar.success("Model loaded")
#     st.sidebar.write("Model classes:", model.names)
# except Exception as e:
#     st.error(f"Unable to load the model: {e}")
#     st.stop()
# 
# if uploaded_file is not None:
#     original = Image.open(uploaded_file).convert("RGB")
# 
#     with st.spinner("Inspecting image..."):
#         result = model.predict(
#             source=original,
#             conf=confidence,
#             imgsz=image_size,
#             verbose=False
#         )[0]
# 
#     annotated = original.copy()
#     draw = ImageDraw.Draw(annotated)
#     defects = []
# 
#     for box in result.boxes:
#         class_id = int(box.cls.item())
# 
#         # Ignore non-defect classes
#         if class_id not in DEFECT_CLASSES:
#             continue
# 
#         score = float(box.conf.item())
#         x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
#         name = DEFECT_CLASSES[class_id]
# 
#         draw.rectangle(
#             (x1, y1, x2, y2),
#             outline="red",
#             width=max(3, original.width // 300)
#         )
# 
#         label = f"{name} {score:.0%}"
#         draw.text(
#             (x1, max(0, y1 - 22)),
#             label,
#             fill="red",
#             stroke_width=1,
#             stroke_fill="white"
#         )
# 
#         defects.append({
#             "name": name,
#             "confidence": score,
#             "box": (x1, y1, x2, y2)
#         })
# 
#     left, right = st.columns(2)
# 
#     with left:
#         st.subheader("Original image")
#         st.image(original, use_container_width=True)
# 
#     with right:
#         st.subheader("Inspection result")
#         st.image(annotated, use_container_width=True)
# 
#     st.divider()
# 
#     if defects:
#         st.error(f"Potential defects detected: {len(defects)}")
# 
#         for i, defect in enumerate(defects, start=1):
#             st.write(
#                 f"{i}. **{defect['name']}** — "
#                 f"confidence {defect['confidence']:.1%}"
#             )
#     else:
#         st.success("No defects found in the printing object.")
#         st.caption(
#             "No target defect was detected above the selected confidence "
#             "threshold. This does not guarantee that the object is defect-free."
#         )
# else:
#     st.info("Upload an image to begin inspection.")

from getpass import getpass
from pyngrok import ngrok

NGROK_TOKEN = getpass("Enter your ngrok authtoken: ")
ngrok.set_auth_token(NGROK_TOKEN)

"""#Launch Interface"""

from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path("/content/drive/MyDrive/PrintVision_Final/best.pt")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found at {MODEL_PATH}. "
        "Check where your best.pt file is saved."
    )

model = YOLO(str(MODEL_PATH))
print("Model loaded successfully.")
print("Classes:", model.names)

!pip -q install ipywidgets pillow

import io
import numpy as np
import ipywidgets as widgets

from PIL import Image, ImageDraw
from IPython.display import display, clear_output

# Interface title
title = widgets.HTML("""
<div style="
    background: linear-gradient(120deg, #101827, #183b4e);
    color: white;
    padding: 22px;
    border-radius: 12px;
    font-family: Arial;">
    <h1 style="margin:0;">PrintVision AI</h1>
    <p style="margin:8px 0 0 0;">
        AI-powered 3D-print defect inspection
    </p>
</div>
""")

uploader = widgets.FileUpload(
    accept="image/*",
    multiple=False,
    description="Upload image"
)

confidence = widgets.FloatSlider(
    value=0.25,
    min=0.05,
    max=0.95,
    step=0.05,
    description="Confidence:",
    continuous_update=False,
    readout_format=".2f",
    style={"description_width": "initial"},
    layout=widgets.Layout(width="450px")
)

run_button = widgets.Button(
    description="Inspect print",
    button_style="primary",
    icon="search",
    layout=widgets.Layout(width="180px", height="42px")
)

output = widgets.Output()

# Change these IDs only if your model uses a different class mapping.
DEFECT_CLASSES = {
    0: "Crack",
    1: "Failed print",
    5: "Stringing defect"
}

def get_uploaded_image(upload_value):
    """Support common ipywidgets FileUpload output formats."""
    if isinstance(upload_value, dict):
        item = next(iter(upload_value.values()))
        image_bytes = item["content"]
    elif isinstance(upload_value, (tuple, list)):
        item = upload_value[0]
        image_bytes = item["content"]
    else:
        raise ValueError("Please upload an image first.")

    if hasattr(image_bytes, "tobytes"):
        image_bytes = image_bytes.tobytes()

    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def inspect_image(_):
    with output:
        clear_output(wait=True)

        try:
            if not uploader.value:
                print("Upload an image before starting inspection.")
                return

            image = get_uploaded_image(uploader.value)

            # Run YOLO inference
            results = model.predict(
                source=np.array(image),
                conf=confidence.value,
                verbose=False
            )

            result = results[0]
            draw = ImageDraw.Draw(image)
            detections = []

            for box in result.boxes:
                class_id = int(box.cls[0].item())
                score = float(box.conf[0].item())

                # Ignore non-defect classes in the final defect summary.
                if class_id not in DEFECT_CLASSES:
                    continue

                x1, y1, x2, y2 = [
                    int(v) for v in box.xyxy[0].tolist()
                ]

                label = f"{DEFECT_CLASSES[class_id]} {score:.2f}"

                draw.rectangle(
                    [x1, y1, x2, y2],
                    outline="red",
                    width=4
                )

                # Draw a readable label above the box.
                text_box = draw.textbbox((x1, max(0, y1 - 22)), label)
                draw.rectangle(text_box, fill="red")
                draw.text(
                    (x1, max(0, y1 - 22)),
                    label,
                    fill="white"
                )

                detections.append({
                    "defect": DEFECT_CLASSES[class_id],
                    "confidence": score,
                    "box": (x1, y1, x2, y2)
                })

            print("INSPECTION RESULTS")
            print("-" * 40)

            if detections:
                print(f"Potential defects detected: {len(detections)}")
                for i, item in enumerate(detections, start=1):
                    print(
                        f"{i}. {item['defect']} | "
                        f"Confidence: {item['confidence']:.1%} | "
                        f"Box: {item['box']}"
                    )
            else:
                print("No target defects detected above the selected threshold.")
                print(
                    "This does not guarantee that the print is defect-free."
                )

            display(image)

        except Exception as e:
            print(f"Inspection failed: {type(e).__name__}: {e}")


run_button.on_click(inspect_image)

display(widgets.VBox([
    title,
    widgets.HTML("<h3>Upload a 3D-print image</h3>"),
    uploader,
    confidence,
    run_button,
    output
], layout=widgets.Layout(gap="12px")))

!pip -q install -U gradio ultralytics pillow

from google.colab import drive
drive.mount("/content/drive")

# Local install (run in a terminal, not here):
#   python -m pip install streamlit ultralytics opencv-python pillow pandas

"""#Final Documentation"""

import json
from datetime import datetime

experiment_record = {
    "project": "PrintVision AI",
    "task": "3D-print defect detection",
    "architecture": "YOLOv8n",
    "dataset": str(DATASET_DIR),
    "dataset_yaml": str(DATA_YAML),
    "checkpoint": str(BEST_MODEL),
    "epochs_requested": 50,
    "image_size": 640,
    "batch_size": 8,
    "target_defects": [
        "crack",
        "failed print",
        "stringing defect"
    ],
    "no_defect_logic": (
        "Report no target defects when no target-class detection "
        "passes the confidence threshold."
    ),
    "recorded_at": datetime.now().isoformat(timespec="seconds")
}

record_path = FINAL_DIR / "experiment_record.json"

with open(record_path, "w") as f:
    json.dump(experiment_record, f, indent=2)

print("Experiment record saved:", record_path)

import re
from pathlib import Path
import numpy as np
import pandas as pd

MODEL_PATH = "/content/drive/MyDrive/PrintVision_Final/best.pt"
PHOTO_DIR = Path("/content/drive/MyDrive/PrintVision_Experiment")
SCORE_CONF = 0.03          # low on purpose (illustrative run); confidence is kept as a feature
FAIL_CONF = 0.25           # a 'failed print' box above this marks the print failed
DEFECTS = ["crack", "stringing defect"]

NAME_RE = re.compile(r"T(?P<T>[\d.]+)_v(?P<v>\d)(?:_rep(?P<rep>\d+))?", re.I)


def extract_features(model, path):
    r = model.predict(str(path), conf=SCORE_CONF, imgsz=640, verbose=False)[0]
    h, w = r.orig_shape
    obj_area, failed = 0.0, False
    masks = {d: np.zeros((h, w), bool) for d in DEFECTS}      # union of boxes per defect
    stats = {d: dict(n=0, wconf_area=0.0, max_box=0.0, max_conf=0.0, box_w=0.0, box_h=0.0)
             for d in DEFECTS}
    for b in r.boxes:
        name, conf = model.names[int(b.cls)], float(b.conf)
        x1, y1, x2, y2 = [max(0.0, v) for v in b.xyxy[0].tolist()]
        bw, bh = x2 - x1, y2 - y1
        area = bw * bh
        if name == "object":
            obj_area = max(obj_area, area)
        elif name == "failed print" and conf >= FAIL_CONF:
            failed = True
        elif name in DEFECTS:
            masks[name][int(y1):int(y2), int(x1):int(x2)] = True
            s = stats[name]
            s["n"] += 1
            s["wconf_area"] += conf * area
            s["max_box"] = max(s["max_box"], area)
            s["max_conf"] = max(s["max_conf"], conf)
            s["box_w"] += bw
            s["box_h"] += bh
    ref = float(h * w)   # always normalise by image area (object boxes are inconsistent across photos)
    row = dict(failed=failed, ref_area_px=ref)
    for d in DEFECTS:
        k = d.split()[0]                                       # 'crack', 'stringing'
        s = stats[d]
        row[f"{k}_n"] = s["n"]
        row[f"{k}_area_frac"] = masks[d].sum() / ref           # box size vs part size (union)
        row[f"{k}_wconf_area_frac"] = s["wconf_area"] / ref    # confidence-weighted
        row[f"{k}_max_box_frac"] = s["max_box"] / ref          # largest single box
        row[f"{k}_max_conf"] = s["max_conf"]
        row[f"{k}_box_w_px"] = s["box_w"]
        row[f"{k}_box_h_px"] = s["box_h"]
    return row


def main():
    from ultralytics import YOLO
    model = YOLO(MODEL_PATH)
    rows = []
    files = sorted(PHOTO_DIR.glob("*.jp*g")) + sorted(PHOTO_DIR.glob("*.png"))
    for p in files:
        m = NAME_RE.search(p.name)
        if not m:
            print("Skipped (name not matched):", p.name)
            continue
        feat = extract_features(model, p)
        rows.append(dict(file=p.name, T=float(m["T"]), v=int(m["v"]),
                         rep=int(m["rep"] or 1), **feat))
    df = pd.DataFrame(rows)
    out = PHOTO_DIR / "image_features.csv"
    df.to_csv(out, index=False)
    print(df[["file", "T", "v", "stringing_n", "stringing_area_frac",
              "crack_n", "crack_area_frac", "failed"]].round(3).to_string(index=False))
    print("\nSaved:", out)
    return df


from pathlib import Path
import numpy as np
import pandas as pd

FEATURES_CSV = PHOTO_DIR / "image_features.csv"

# ---- scoring recipe -------------------------------------------------------
# score = sum over defects of  DEFECT_W * ( AREA_W * area_fraction + COUNT_W * count )
DEFECT_W = {"crack": 1.0, "stringing": 0.6}
AREA_W = 1.0              # weight on box size relative to the part (amount of defect)
COUNT_W = 0.05            # weight on number of boxes
AREA_MODE = "wconf_area_frac"   # confidence-weighted: works with weak detections
THRESH = None             # None = illustrative auto threshold (median setting score); or set a number
PLAN_RANGE = (215, 235)
SPEED_BONUS = 0.02        # tie-break toward 2X
# ---------------------------------------------------------------------------


def image_score(df):
    s = np.zeros(len(df))
    for d, wd in DEFECT_W.items():
        area = df[f"{d}_{AREA_MODE}"].clip(upper=1.0)
        s += wd * (AREA_W * area + COUNT_W * df[f"{d}_n"])
    return s


def per_setting(df):
    df = df.copy()
    df["score"] = image_score(df)
    pr = df.groupby(["T", "v", "rep"]).agg(score=("score", "mean"),
                                           failed=("failed", "max")).reset_index()
    penalty = max(pr.score.max(), THRESH or 0.0) + 1.0        # failure is a penalty, never zero
    pr.loc[pr.failed, "score"] = penalty
    st = pr.groupby(["T", "v"]).agg(score=("score", "mean"), sd=("score", "std"),
                                    n=("rep", "count"), failed=("failed", "max")).reset_index()
    st["objective"] = st.score - SPEED_BONUS * (st.v == 2)
    return st


def analyse(st, thresh=None):
    if thresh is None:
        thresh = THRESH if THRESH is not None else float(st.score.median())
    print(f"Acceptable-score threshold used: {thresh:.3f}")
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
    from scipy.stats import norm

    for v, g in st.groupby("v"):
        g = g.sort_values("T")
        print(f"\n=== Speed {v}X ===")
        print(g[["T", "score", "sd", "n", "failed"]].round(3).to_string(index=False))
        out = g[(g["T"] < PLAN_RANGE[0]) | (g["T"] > PLAN_RANGE[1])]
        if len(out):
            print("Note: outside planned window:", list(out["T"]))
        best = g.loc[g.objective.idxmin()]
        print(f"Best TESTED setting: {best['T']:.0f} C (score {best['score']:.3f})")

        if len(g) < 4:
            print(f"Only {len(g)} temperatures: too few to fit a curve or state a range. "
                  "Add temperatures (and the other speed) first.")
            continue
        lo, hi = g["T"].min(), g["T"].max()
        X = ((g["T"].values - lo) / (hi - lo)).reshape(-1, 1)
        gp = GaussianProcessRegressor(
            C(1.0) * Matern(length_scale=0.4, nu=2.5, length_scale_bounds=(0.15, 5))
            + WhiteKernel(0.05, (1e-4, 1.0)), normalize_y=True,
            n_restarts_optimizer=10).fit(X, g.objective.values)
        Tg = np.arange(lo, hi + 0.01, 0.5)             # no extrapolation
        mu, sd = gp.predict(((Tg - lo) / (hi - lo)).reshape(-1, 1), return_std=True)
        ok = (mu + 1.28 * sd) < thresh
        if ok.any():
            print(f"Predicted acceptable range: {Tg[ok].min():.1f} - {Tg[ok].max():.1f} C")
        else:
            print("No temperature meets the threshold with confidence.")
        ymin = g.objective.min()
        z = (ymin - mu) / np.maximum(sd, 1e-9)
        ei = (ymin - mu) * norm.cdf(z) + sd * norm.pdf(z)
        print(f"GP minimum at {Tg[np.argmin(mu)]:.1f} C; next suggested print: {Tg[np.argmax(ei)]:.1f} C")




def plot_results(st, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    for v, g in st.groupby("v"):
        g = g.sort_values("T")
        ax.plot(g["T"], g["score"], marker="o", label=f"{v}X")
    thr = THRESH if THRESH is not None else float(st.score.median())
    ax.axhline(thr, ls="--", color="gray", label="threshold")
    ax.set_xlabel("Nozzle temperature (C)"); ax.set_ylabel("Defect score (lower is better)")
    ax.legend(); ax.grid(alpha=0.3); fig.tight_layout(); fig.savefig(out_png, dpi=200)
    print("Saved plot:", out_png)


if __name__ == "__main__":
    df = main()                       # step 1: YOLO -> image_features.csv
    st = per_setting(df)              # step 2: features -> score per setting
    st.to_csv(FEATURES_CSV.parent / "setting_scores.csv", index=False)
    analyse(st)
    plot_results(st, FEATURES_CSV.parent / "score_vs_temperature.png")