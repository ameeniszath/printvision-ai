
import os
import re
import warnings
from pathlib import Path

import altair as alt
import numpy as np
import streamlit as st
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from ultralytics import YOLO


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="PrintVision AI | Inspection Console",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# 2. CUSTOM DASHBOARD STYLE
# ============================================================

st.markdown(
    """
    <style>
    .stApp {
        background-color: #080d17;
        color: #edf2f7;
    }

    section[data-testid="stSidebar"] {
        background-color: #0b1422;
        border-right: 1px solid #273449;
    }

    .main-title {
        font-size: 34px;
        font-weight: 800;
        letter-spacing: 1px;
        color: #f3f7ff;
        margin-bottom: 0;
    }

    .main-title span {
        color: #37d6bd;
    }

    .subtitle {
        color: #9aabc2;
        margin-top: 6px;
        margin-bottom: 24px;
    }

    .metric-card {
        background: #111d2d;
        border: 1px solid #293950;
        border-radius: 15px;
        padding: 20px;
        min-height: 112px;
    }

    .metric-label {
        color: #aab9cc;
        font-size: 14px;
        margin-bottom: 10px;
    }

    .metric-value {
        color: #f3f7ff;
        font-size: 27px;
        font-weight: 700;
    }

    .section-title {
        color: #f0f5ff;
        font-size: 21px;
        font-weight: 700;
        margin-top: 18px;
        margin-bottom: 12px;
    }

    .status-box {
        padding: 15px;
        border-radius: 10px;
        margin: 12px 0 20px 0;
        background: #123629;
        color: #58e69a;
    }

    .warning-box {
        padding: 15px;
        border-radius: 10px;
        margin: 12px 0 20px 0;
        background: #37351a;
        color: #f4e99a;
    }

    div.stButton > button {
        background-color: #1c8f83;
        color: white;
        border: none;
        border-radius: 9px;
        font-weight: 600;
        min-height: 42px;
    }

    div.stButton > button:hover {
        background-color: #26aa9a;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. MODEL CONFIGURATION
# ============================================================

# Put the trained best.pt in ./weights, or point PRINTVISION_WEIGHTS at it.
DEFAULT_MODEL_PATH = os.environ.get(
    "PRINTVISION_WEIGHTS",
    str(Path(__file__).parent / "weights" / "best.pt"),
)

# Only these classes are displayed as defects.
# General objects, print bed and printer head are excluded.
TARGET_DEFECTS = {
    "crack": "Crack",
    "failed print": "Failed print",
    "print failed": "Failed print",
    "stringing": "Stringing defect",
    "stringing defect": "Stringing defect",
}


# Class IDs from the training notebook's model_metadata.json.
# Used only if the class names in the weights file do not match above.
DOCUMENTED_DEFECT_IDS = {
    0: "Crack",
    1: "Failed print",
    5: "Stringing defect",
}


def normalize_name(name):
    """Normalize class names for matching."""
    return (
        str(name)
        .lower()
        .replace("_", " ")
        .replace("-", " ")
        .strip()
    )


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_model(model_path, file_mtime):
    """Load and cache the model for the selected weights file."""
    return YOLO(model_path)



# ============================================================
# 3b. SCORING + OPTIMISATION ENGINE (no Streamlit code here)
# ============================================================
# >>> ENGINE START

# Filenames must contain T<temp>_v<speed>, optionally _rep<n>.
# Example: T225_v1_a1.jpg  ->  225 C, speed 1X.
NAME_RE = re.compile(r"T(?P<T>[\d.]+)_v(?P<v>\d)(?:_rep(?P<rep>\d+))?", re.I)

# Map the display names used above to scoring roles.
ROLE_BY_DISPLAY = {
    "Crack": "crack",
    "Stringing defect": "stringing",
    "Failed print": "failed",
}
DEFECT_KEYS = ["crack", "stringing"]


def extract_features(result, class_roles, score_conf, fail_conf):
    """One YOLO result -> one row of image features.

    Defect size is the union of that defect's boxes divided by the
    FULL IMAGE area (object boxes are inconsistent between photos).
    Confidence is kept as a feature (confidence-weighted area).
    """
    h, w = result.orig_shape
    ref = float(h * w)
    failed = False
    masks = {d: np.zeros((h, w), bool) for d in DEFECT_KEYS}
    stats = {d: dict(n=0, wconf_area=0.0, max_conf=0.0) for d in DEFECT_KEYS}

    for b in result.boxes:
        role = class_roles.get(int(b.cls[0].item()))
        conf = float(b.conf[0].item())
        if role is None or conf < score_conf:
            continue
        x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
        x1, x2 = max(0.0, x1), min(float(w), x2)
        y1, y2 = max(0.0, y1), min(float(h), y2)
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

        if role == "failed":
            if conf >= fail_conf:
                failed = True
        elif role in DEFECT_KEYS:
            masks[role][int(y1):int(y2), int(x1):int(x2)] = True
            s = stats[role]
            s["n"] += 1
            s["wconf_area"] += conf * area
            s["max_conf"] = max(s["max_conf"], conf)

    row = dict(failed=failed, ref_area_px=ref)
    for d in DEFECT_KEYS:
        s = stats[d]
        row[f"{d}_n"] = s["n"]
        row[f"{d}_area_frac"] = float(masks[d].sum()) / ref
        row[f"{d}_wconf_area_frac"] = s["wconf_area"] / ref
        row[f"{d}_max_conf"] = s["max_conf"]
    return row


def image_score(df, p):
    """score = sum_d DEFECT_W[d] * (AREA_W * area_fraction + COUNT_W * count)
    Lower is better."""
    s = np.zeros(len(df))
    for d in DEFECT_KEYS:
        area = df[f"{d}_wconf_area_frac"].clip(upper=1.0)
        s += p["defect_w"][d] * (p["area_w"] * area + p["count_w"] * df[f"{d}_n"])
    return s


def per_setting(df, p):
    """Image features -> one score per (temperature, speed)."""
    df = df.copy()
    df["score"] = image_score(df, p)
    pr = (
        df.groupby(["T", "v", "rep"])
        .agg(score=("score", "mean"), failed=("failed", "max"))
        .reset_index()
    )
    pr["failed"] = pr["failed"].astype(bool)
    # failure is a penalty, never zero
    penalty = max(pr["score"].max(), p["thresh"] or 0.0) + 1.0
    pr.loc[pr["failed"], "score"] = penalty
    out = (
        pr.groupby(["T", "v"])
        .agg(score=("score", "mean"), sd=("score", "std"),
             n=("rep", "count"), failed=("failed", "max"))
        .reset_index()
    )
    out["objective"] = out["score"] - p["speed_bonus"] * (out["v"] == 2)
    return out


def acceptable_range(grid, ok, i_best):
    """Contiguous acceptable window on the GP grid.

    Returns the unbroken run of acceptable temperatures that contains the
    GP minimum (or, if the minimum itself is not acceptable, the longest
    acceptable run), plus the number of separate runs. Taking min/max of
    every acceptable point would wrongly include bad temperatures that sit
    between two good ones.
    """
    runs, start = [], None
    for i, flag in enumerate(ok):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(ok) - 1):
            end = i if flag else i - 1
            runs.append((start, end))
            start = None
    if not runs:
        return None, 0
    chosen = next((r for r in runs if r[0] <= i_best <= r[1]), None)
    if chosen is None:
        chosen = max(runs, key=lambda r: r[1] - r[0])
    return (float(grid[chosen[0]]), float(grid[chosen[1]])), len(runs)


def analyse_settings(st_df, p):
    """Per speed: best tested setting, plus a Gaussian-process fit
    (needs >= 4 temperatures) giving an acceptable range, the GP
    minimum and the next most informative temperature to print."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import Matern, WhiteKernel, ConstantKernel as C
    from scipy.stats import norm

    thresh = p["thresh"] if p["thresh"] is not None else float(st_df["score"].median())
    out = {"thresh": thresh, "speeds": {}}

    for v, g in st_df.groupby("v"):
        g = g.sort_values("T").reset_index(drop=True)
        lo_plan, hi_plan = p["plan_range"]
        info = {
            "table": g,
            "outside": [float(t) for t in g["T"] if t < lo_plan or t > hi_plan],
            "best": g.loc[g["objective"].idxmin()],
            "gp": None,
            "gp_msg": None,
        }

        if len(g) < 4:
            info["gp_msg"] = (
                f"Only {len(g)} temperatures: too few to fit a curve or state "
                "a range. Add temperatures first."
            )
            out["speeds"][int(v)] = info
            continue

        lo, hi = float(g["T"].min()), float(g["T"].max())
        X = ((g["T"].values - lo) / (hi - lo)).reshape(-1, 1)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                gp = GaussianProcessRegressor(
                    C(1.0) * Matern(length_scale=0.4, nu=2.5,
                                    length_scale_bounds=(0.15, 5))
                    + WhiteKernel(0.05, (1e-4, 1.0)),
                    normalize_y=True,
                    n_restarts_optimizer=10,
                    random_state=0,
                ).fit(X, g["objective"].values)
                Tg = np.arange(lo, hi + 0.01, 0.5)          # no extrapolation
                mu, sd = gp.predict(((Tg - lo) / (hi - lo)).reshape(-1, 1),
                                    return_std=True)
        except Exception as exc:                              # pragma: no cover
            info["gp_msg"] = f"GP fit failed: {exc}"
            out["speeds"][int(v)] = info
            continue

        # Acceptable = 90 % upper confidence bound below the threshold.
        ok = (mu + 1.28 * sd) < thresh
        i_best = int(np.argmin(mu))
        rng, n_runs = acceptable_range(Tg, ok, i_best)
        ymin = float(g["objective"].min())
        z = (ymin - mu) / np.maximum(sd, 1e-9)
        ei = (ymin - mu) * norm.cdf(z) + sd * norm.pdf(z)   # expected improvement
        info["gp"] = {
            "grid": Tg,
            "mu": mu,
            "sd": sd,
            "range": rng,
            "n_runs": n_runs,
            "min_T": float(Tg[i_best]),
            "next_T": float(Tg[np.argmax(ei)]),
        }
        out["speeds"][int(v)] = info
    return out

# <<< ENGINE END

# ============================================================
# 4. SIDEBAR: INSPECTION SETTINGS
# ============================================================

with st.sidebar:
    st.markdown(
        """
        <div style="color:#37d6bd;font-weight:800;
                    letter-spacing:2px;font-size:12px;">
            PV // CONTROL PANEL
        </div>
        <h3 style="margin-top:10px;">Inspection settings</h3>
        """,
        unsafe_allow_html=True,
    )

    model_path = st.text_input(
        "Model weights (.pt)",
        value=DEFAULT_MODEL_PATH,
        help="Enter the full path to your trained best.pt file.",
    )

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.01,
        help="Default 0.25 ≈ the F1-optimal threshold of the v2 model "
             "(best F1 0.64 at 0.22 on the held-out test set).",
    )

    show_confidence = st.checkbox(
        "Show confidence on boxes",
        value=True,
    )

    st.divider()

    with st.expander("Scoring & optimisation settings"):
        score_conf = st.slider(
            "Scoring confidence floor",
            min_value=0.01, max_value=0.50, value=0.03, step=0.01,
            help="Low on purpose: confidence is kept as a feature "
                 "(confidence-weighted area).",
        )
        fail_conf = st.slider(
            "Failed-print confidence",
            min_value=0.05, max_value=0.95, value=0.75, step=0.05,
            help="A 'failed print' box above this marks the print failed. "
                 "Kept high because 'failed print' is the weakest class "
                 "(test AP 0.12) and gave false alarms on complete prints.",
        )
        w_crack = st.number_input("Crack weight", 0.0, 10.0, 1.0, 0.1)
        w_string = st.number_input("Stringing weight", 0.0, 10.0, 0.6, 0.1)
        area_w = st.number_input("Area weight", 0.0, 10.0, 1.0, 0.1)
        count_w = st.number_input("Count weight", 0.0, 1.0, 0.05, 0.01)
        speed_bonus = st.number_input(
            "2X speed bonus", 0.0, 1.0, 0.02, 0.01,
            help="Tie-break toward the faster speed.",
        )
        thresh_mode = st.radio(
            "Acceptable-score threshold",
            ["Auto (median score)", "Manual"],
        )
        manual_thresh = st.number_input(
            "Manual threshold", 0.0, 10.0, 0.05, 0.01,
            disabled=(thresh_mode == "Auto (median score)"),
        )
        plan_lo = st.number_input("Planned window: min (C)", 150, 300, 215, 5)
        plan_hi = st.number_input("Planned window: max (C)", 150, 300, 235, 5)

    score_params = {
        "defect_w": {"crack": w_crack, "stringing": w_string},
        "area_w": area_w,
        "count_w": count_w,
        "speed_bonus": speed_bonus,
        "thresh": None if thresh_mode.startswith("Auto") else float(manual_thresh),
        "plan_range": (plan_lo, plan_hi),
    }

    st.divider()

    st.markdown("### Input source")
    st.caption(
        "Image upload (single inspection or batch scoring). "
        "IP-camera streaming can be added separately."
    )

    st.divider()

    st.caption(
        "LOCAL INFERENCE · IMAGE PROCESSING ON THIS COMPUTER"
    )


# ============================================================
# 5. DASHBOARD HEADER
# ============================================================

st.markdown(
    """
    <div class="main-title">PRINTVISION <span>AI</span></div>
    <div class="subtitle">
        3D-print defect inspection console · Upload an image,
        run inference, review detections.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 6. LOAD MODEL AND CHECK CLASSES
# ============================================================

model = None
model_error = None
available_defect_classes = {}
class_roles = {}

try:
    path = Path(model_path.strip().strip('"'))

    if not path.is_file():
        raise FileNotFoundError(
            f"Model weights were not found at:\n{path}"
        )

    model = load_model(
        str(path.resolve()),
        path.stat().st_mtime,
    )

    for class_id, class_name in model.names.items():
        normalized = normalize_name(class_name)

        if normalized in TARGET_DEFECTS:
            available_defect_classes[int(class_id)] = (
                TARGET_DEFECTS[normalized]
            )

    if not available_defect_classes:
        # Fall back to the IDs saved in model_metadata.json during
        # training: 0 = crack, 1 = failed print, 5 = stringing defect.
        for class_id, display_name in DOCUMENTED_DEFECT_IDS.items():
            if class_id in model.names:
                available_defect_classes[class_id] = display_name

    class_roles = {
        cid: ROLE_BY_DISPLAY[name]
        for cid, name in available_defect_classes.items()
    }

    if not available_defect_classes:
        actual_classes = ", ".join(
            str(name) for name in model.names.values()
        )

        raise ValueError(
            "None of the target defect classes matched the model. "
            f"Model classes: {actual_classes}"
        )

except Exception as exc:
    model_error = str(exc)


# ============================================================
# 7. TOP STATUS CARDS
# ============================================================

metric1, metric2, metric3 = st.columns(3)

with metric1:
    status = "CONNECTED" if model is not None else "NOT CONNECTED"

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">DETECTION ENGINE</div>
            <div class="metric-value">{status}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric2:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">INPUT PIPELINE</div>
            <div class="metric-value">IMAGE UPLOAD</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with metric3:
    st.markdown(
        """
        <div class="metric-card">
            <div class="metric-label">INFERENCE MODE</div>
            <div class="metric-value">LOCAL</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


if model_error:
    st.markdown(
        f'<div class="warning-box"><b>Model unavailable</b><br>'
        f'{model_error}</div>',
        unsafe_allow_html=True,
    )

else:
    st.success(
        "Model loaded. Displaying only target defect classes: "
        "crack, failed print, and stringing defect."
    )


# ============================================================
# 8. TABS: SINGLE INSPECTION | BATCH SCORING & OPTIMISATION
# ============================================================

tab_single, tab_batch = st.tabs(
    ["🔍 Single image inspection", "📈 Batch scoring & optimisation"]
)


# ------------------------------------------------------------
# TAB 1: SINGLE IMAGE INSPECTION (+ defect score)
# ------------------------------------------------------------

with tab_single:
    left, right = st.columns([0.95, 1.25], gap="large")

    with left:
        st.markdown(
            '<div class="section-title">01 / Image input</div>',
            unsafe_allow_html=True,
        )

        uploaded_file = st.file_uploader(
            "Choose a photo of a 3D-printed part",
            type=["png", "jpg", "jpeg", "bmp", "webp"],
            key="single_upload",
        )

        st.caption(
            "Tip: keep the defect in focus and avoid strong glare "
            "or motion blur."
        )

        run_inspection = st.button(
            "▶ Run inspection",
            use_container_width=True,
            disabled=(model is None or uploaded_file is None),
        )

    # --------------------------------------------------------
    # 9. INFERENCE, FILTERING, DRAWING AND SCORING
    # --------------------------------------------------------

    if uploaded_file is not None:
        original_image = Image.open(uploaded_file).convert("RGB")
    else:
        original_image = None

    annotated_image = None
    detections = []
    inference_error = None
    image_score_value = None
    image_failed = False

    if run_inspection and model is not None and original_image is not None:
        try:
            with st.spinner("Inspecting image..."):
                # One prediction at the lower of the two confidences.
                # Boxes are DRAWN at the display threshold; the SCORE
                # uses everything above the scoring floor.
                result = model.predict(
                    source=original_image,
                    conf=min(confidence, score_conf),
                    verbose=False,
                )[0]

                feat = extract_features(
                    result, class_roles, score_conf, fail_conf
                )
                image_failed = bool(feat["failed"])
                image_score_value = float(
                    image_score(pd.DataFrame([feat]), score_params)[0]
                )

                annotated_image = original_image.copy()
                draw = ImageDraw.Draw(annotated_image)

                try:
                    font = ImageFont.truetype("arial.ttf", 15)
                except OSError:
                    font = ImageFont.load_default()

                # Thin boxes: change this to 1 for even thinner lines.
                BOX_THICKNESS = 2

                for box in result.boxes:
                    class_id = int(box.cls[0].item())

                    # FILTER BEFORE DRAWING OR ADDING TO TABLE.
                    # This excludes object, print bed and printer head.
                    if class_id not in available_defect_classes:
                        continue

                    class_name = available_defect_classes[class_id]
                    score = float(box.conf[0].item())

                    # Display threshold (the score ignores this).
                    if score < confidence:
                        continue

                    x1, y1, x2, y2 = [
                        int(value)
                        for value in box.xyxy[0].tolist()
                    ]

                    detections.append(
                        {
                            "Defect / class": class_name,
                            "Confidence": round(score, 3),
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                        }
                    )

                    # Draw thinner bounding box.
                    for offset in range(BOX_THICKNESS):
                        draw.rectangle(
                            [
                                x1 + offset,
                                y1 + offset,
                                x2 - offset,
                                y2 - offset,
                            ],
                            outline="#18d6bb",
                        )

                    # Draw a compact label.
                    if show_confidence:
                        label = f"{class_name} {score:.2f}"
                    else:
                        label = class_name

                    text_bbox = draw.textbbox((0, 0), label, font=font)
                    text_width = text_bbox[2] - text_bbox[0]
                    text_height = text_bbox[3] - text_bbox[1]

                    label_x = max(0, x1)
                    label_y = max(0, y1 - text_height - 8)

                    draw.rectangle(
                        [
                            label_x,
                            label_y,
                            label_x + text_width + 10,
                            label_y + text_height + 6,
                        ],
                        fill="#087f75",
                    )

                    draw.text(
                        (label_x + 5, label_y + 3),
                        label,
                        fill="white",
                        font=font,
                    )

        except Exception as exc:
            inference_error = str(exc)

    # --------------------------------------------------------
    # 10. DETECTION VIEWPORT AND RESULTS
    # --------------------------------------------------------

    with right:
        st.markdown(
            '<div class="section-title">02 / Detection viewport</div>',
            unsafe_allow_html=True,
        )

        if inference_error:
            st.error(f"Inspection failed: {inference_error}")

        elif annotated_image is not None:
            st.image(
                annotated_image,
                caption="Filtered defect detections",
                use_container_width=True,
            )

        elif original_image is not None:
            st.image(
                original_image,
                caption="Input image — select Run inspection",
                use_container_width=True,
            )

        else:
            st.info("Inspection viewport is waiting for an image.")

    # Display results only after inference has been run.
    if annotated_image is not None:
        st.divider()

        result_left, result_right = st.columns([1, 2])

        with result_left:
            st.markdown(
                '<div class="section-title">03 / Inspection summary</div>',
                unsafe_allow_html=True,
            )

            m_a, m_b = st.columns(2)
            m_a.metric("Defect detections", len(detections))
            m_b.metric(
                "Defect score",
                f"{image_score_value:.3f}",
                help="Lower is better. Uses the scoring settings in the "
                     "sidebar and every detection above the scoring "
                     "confidence floor.",
            )

            if image_failed:
                st.error("A 'failed print' detection exceeded the "
                         "failed-print confidence.")

            if detections:
                st.success(
                    "Potential defect detection(s) returned above "
                    "the selected confidence threshold."
                )
            else:
                st.info(
                    "No target defects detected above the selected "
                    "confidence threshold. This does not prove the "
                    "part is defect-free."
                )

        with result_right:
            st.markdown(
                '<div class="section-title">04 / Detection details</div>',
                unsafe_allow_html=True,
            )

            if detections:
                table = pd.DataFrame(detections)
                st.dataframe(
                    table,
                    use_container_width=True,
                    hide_index=True,
                )

                csv_data = table.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download detection report (CSV)",
                    data=csv_data,
                    file_name="printvision_inspection.csv",
                    mime="text/csv",
                )
            else:
                st.caption("No target defect detections to display.")


# ------------------------------------------------------------
# TAB 2: BATCH SCORING AND OPTIMISATION
# ------------------------------------------------------------

with tab_batch:
    st.markdown(
        '<div class="section-title">05 / Labelled photo set</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Name each photo T<temp>_v<speed> (optional _rep<n>), e.g. "
        "T225_v1_a1.jpg = 225 C at 1X. Photos of the same setting are "
        "averaged. Images are scored locally; weights, threshold and "
        "planned window can be changed in the sidebar without "
        "re-running the model."
    )

    batch_files = st.file_uploader(
        "Upload the labelled photos",
        type=["png", "jpg", "jpeg", "bmp", "webp"],
        accept_multiple_files=True,
        key="batch_upload",
    )

    b_run, b_clear = st.columns(2)
    run_batch = b_run.button(
        "▶ Score and optimise",
        use_container_width=True,
        disabled=(model is None or not batch_files),
        key="run_batch",
    )
    clear_batch = b_clear.button(
        "Clear results",
        use_container_width=True,
        key="clear_batch",
    )

    if clear_batch:
        st.session_state.pop("batch_features", None)

    # ---- step 1: YOLO -> image features (cached in session) ----
    if run_batch and model is not None and batch_files:
        rows, skipped = [], []
        progress = st.progress(0.0, text="Detecting defects...")

        for i, f in enumerate(batch_files):
            m = NAME_RE.search(f.name)
            try:
                if not m:
                    raise ValueError("name not matched")
                temp = float(m["T"])
            except ValueError:
                skipped.append(f.name)
                progress.progress((i + 1) / len(batch_files))
                continue

            try:
                img = Image.open(f).convert("RGB")
                res = model.predict(
                    source=img, conf=score_conf, imgsz=640, verbose=False
                )[0]
                feat = extract_features(
                    res, class_roles, score_conf, fail_conf
                )
                rows.append(
                    dict(
                        file=f.name,
                        T=temp,
                        v=int(m["v"]),
                        rep=int(m["rep"] or 1),
                        **feat,
                    )
                )
            except Exception as exc:
                skipped.append(f"{f.name} ({exc})")

            progress.progress((i + 1) / len(batch_files))

        progress.empty()

        if rows:
            st.session_state["batch_features"] = pd.DataFrame(rows)
        if skipped:
            st.warning("Skipped (name not matched or unreadable): "
                       + ", ".join(skipped))

    # ---- step 2 + 3: score, per-setting, optimise (live) ----
    feats = st.session_state.get("batch_features")

    if feats is None:
        st.info("Upload labelled photos and select Score and optimise.")
    else:
        settings = per_setting(feats, score_params)
        analysis = analyse_settings(settings, score_params)
        thr = analysis["thresh"]

        st.divider()
        st.markdown(
            '<div class="section-title">06 / Optimisation result</div>',
            unsafe_allow_html=True,
        )

        overall = settings.loc[settings["objective"].idxmin()]
        c1, c2, c3 = st.columns(3)
        c1.metric(
            "Best tested setting",
            f"{overall['T']:.0f} °C · {int(overall['v'])}X",
        )
        c2.metric("Its defect score", f"{overall['score']:.3f}")
        c3.metric("Acceptable-score threshold", f"{thr:.3f}",
                  help="Median score if Auto; otherwise your manual value.")

        if score_params["thresh"] is None:
            st.caption(
                "Threshold is Auto (median score), so about half the "
                "settings always count as acceptable. Set a manual "
                "threshold in the sidebar once you know what a "
                "good print scores."
            )

        # ---- per-speed findings ----
        speed_cols = st.columns(max(1, len(analysis["speeds"])))
        for col, (v, info) in zip(speed_cols, sorted(analysis["speeds"].items())):
            with col:
                st.markdown(f"**Speed {v}X**")
                best = info["best"]
                st.write(
                    f"Best tested: **{best['T']:.0f} °C** "
                    f"(score {best['score']:.3f})"
                )
                if info["outside"]:
                    st.caption(
                        "Outside planned window: "
                        + ", ".join(f"{t:.0f}" for t in info["outside"])
                    )
                gp = info["gp"]
                if gp is None:
                    st.warning(info["gp_msg"])
                else:
                    if gp["range"]:
                        st.write(
                            f"Predicted acceptable range: "
                            f"**{gp['range'][0]:.1f} – {gp['range'][1]:.1f} °C**"
                        )
                        if gp["n_runs"] > 1:
                            st.caption(
                                f"{gp['n_runs']} separate acceptable windows "
                                "found; showing the one around the GP minimum."
                            )
                    else:
                        st.write("No temperature meets the threshold "
                                 "with confidence.")
                    st.write(f"GP minimum: **{gp['min_T']:.1f} °C**")
                    st.write(f"Next suggested print: **{gp['next_T']:.1f} °C**")

        # ---- chart: tested scores, GP fit, threshold ----
        tested = settings.assign(speed=settings["v"].astype(int).astype(str) + "X")
        fit_frames = []
        for v, info in analysis["speeds"].items():
            if info["gp"] is not None:
                fit_frames.append(
                    pd.DataFrame(
                        {
                            "T": info["gp"]["grid"],
                            "score": info["gp"]["mu"],
                            "speed": f"{v}X",
                        }
                    )
                )

        x_enc = alt.X("T:Q", title="Nozzle temperature (°C)",
                      scale=alt.Scale(zero=False))
        y_enc = alt.Y("score:Q", title="Defect score (lower is better)")
        c_enc = alt.Color("speed:N", title="Speed")

        layers = [
            alt.Chart(tested[["T", "score", "speed"]])
            .mark_line(point=True)
            .encode(x=x_enc, y=y_enc, color=c_enc)
        ]
        if fit_frames:
            layers.append(
                alt.Chart(pd.concat(fit_frames))
                .mark_line(strokeDash=[4, 3], opacity=0.6)
                .encode(x=x_enc, y=y_enc, color=c_enc)
            )
        layers.append(
            alt.Chart(pd.DataFrame({"y": [thr]}))
            .mark_rule(strokeDash=[6, 4], color="#9aabc2")
            .encode(y="y:Q")
        )

        chart = (
            alt.layer(*layers)
            .properties(height=340, background="#111d2d")
            .configure_axis(labelColor="#c5d0e0", titleColor="#c5d0e0",
                            gridColor="#273449")
            .configure_legend(labelColor="#c5d0e0", titleColor="#c5d0e0")
            .configure_view(strokeWidth=0)
        )
        st.markdown(
            '<div class="section-title">07 / Score vs temperature</div>',
            unsafe_allow_html=True,
        )
        st.altair_chart(chart, theme=None, use_container_width=True)
        st.caption(
            "Solid = tested settings, dashed = GP fit (needs 4+ "
            "temperatures at that speed), grey line = threshold."
        )

        # ---- tables + downloads ----
        t_left, t_right = st.columns(2)

        with t_left:
            st.markdown(
                '<div class="section-title">08 / Score per setting</div>',
                unsafe_allow_html=True,
            )
            shown = settings.rename(
                columns={"T": "Temp (C)", "v": "Speed (X)"}
            ).round(3)
            st.dataframe(shown, use_container_width=True, hide_index=True)
            st.download_button(
                "Download setting scores (CSV)",
                data=settings.to_csv(index=False).encode("utf-8"),
                file_name="setting_scores.csv",
                mime="text/csv",
            )

        with t_right:
            st.markdown(
                '<div class="section-title">09 / Per-image features</div>',
                unsafe_allow_html=True,
            )
            per_image = feats.copy()
            per_image["score"] = image_score(per_image, score_params)
            cols = ["file", "T", "v", "failed", "crack_n", "crack_area_frac",
                    "stringing_n", "stringing_area_frac", "score"]
            st.dataframe(per_image[cols].round(3),
                         use_container_width=True, hide_index=True)
            st.download_button(
                "Download image features (CSV)",
                data=per_image.to_csv(index=False).encode("utf-8"),
                file_name="image_features.csv",
                mime="text/csv",
            )

        st.caption(
            "Scores depend on the weights, threshold and confidence floor "
            "you set; state them when reporting. If the temperature labels "
            "were assigned for illustration, the result demonstrates the "
            "pipeline and is not a validated optimum."
        )


# ============================================================
# 11. FOOTER
# ============================================================

st.divider()

st.caption(
    "PRINTVISION AI · Prototype interface. Predictions depend on "
    "model quality, training labels, lighting, camera angle, and "
    "the confidence threshold. An empty result is not a guarantee "
    "that a printed part is defect-free."
)
