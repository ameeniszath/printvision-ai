"""
PrintVision AI - scoring + Gaussian-process optimisation (one file).
Needs the trained weights (best.pt) and photos named T215_v1_a1.jpg etc.

Locally:  python scripts/scoring_and_optimisation.py
  (defaults: weights/best.pt and assets/temperature_tower; override with
   PRINTVISION_WEIGHTS and PRINTVISION_PHOTOS environment variables)
In Colab: set the two variables to your Drive paths, or upload and run.
"""
# %% [setup]  (uncomment in Colab)
# !pip -q install ultralytics scikit-learn scipy pandas matplotlib
# from google.colab import drive; drive.mount("/content/drive")

# %% [step 1: scoring]
import os
import re
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = os.environ.get("PRINTVISION_WEIGHTS", str(ROOT / "weights" / "best.pt"))
PHOTO_DIR = Path(os.environ.get("PRINTVISION_PHOTOS", ROOT / "assets" / "temperature_tower"))
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




# %% [step 2: score + optimise]
import warnings

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
    pr["failed"] = pr["failed"].astype(bool)                  # object dtype breaks .loc masking
    penalty = max(pr.score.max(), THRESH or 0.0) + 1.0        # failure is a penalty, never zero
    pr.loc[pr.failed, "score"] = penalty
    st = pr.groupby(["T", "v"]).agg(score=("score", "mean"), sd=("score", "std"),
                                    n=("rep", "count"), failed=("failed", "max")).reset_index()
    st["objective"] = st.score - SPEED_BONUS * (st.v == 2)
    return st


def acceptable_range(grid, ok, i_best):
    """Unbroken acceptable window containing the GP minimum (else the longest one).
    min/max over all acceptable points would include bad temperatures between good ones."""
    runs, start = [], None
    for i, flag in enumerate(ok):
        if flag and start is None:
            start = i
        if start is not None and (not flag or i == len(ok) - 1):
            runs.append((start, i if flag else i - 1))
            start = None
    if not runs:
        return None, 0
    chosen = next((r for r in runs if r[0] <= i_best <= r[1]), max(runs, key=lambda r: r[1] - r[0]))
    return (float(grid[chosen[0]]), float(grid[chosen[1]])), len(runs)


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
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")            # GP optimiser convergence noise
            gp = GaussianProcessRegressor(
                C(1.0) * Matern(length_scale=0.4, nu=2.5, length_scale_bounds=(0.15, 5))
                + WhiteKernel(0.05, (1e-4, 1.0)), normalize_y=True,
                n_restarts_optimizer=10, random_state=0,   # reproducible results
            ).fit(X, g.objective.values)
            Tg = np.arange(lo, hi + 0.01, 0.5)         # no extrapolation
            mu, sd = gp.predict(((Tg - lo) / (hi - lo)).reshape(-1, 1), return_std=True)
        ok = (mu + 1.28 * sd) < thresh                 # 90 % upper bound below threshold
        i_best = int(np.argmin(mu))
        rng, n_runs = acceptable_range(Tg, ok, i_best)
        if rng:
            print(f"Predicted acceptable range: {rng[0]:.1f} - {rng[1]:.1f} C")
            if n_runs > 1:
                print(f"  ({n_runs} separate acceptable windows; showing the one around the GP minimum)")
        else:
            print("No temperature meets the threshold with confidence.")
        ymin = g.objective.min()
        z = (ymin - mu) / np.maximum(sd, 1e-9)
        ei = (ymin - mu) * norm.cdf(z) + sd * norm.pdf(z)  # expected improvement
        print(f"GP minimum at {Tg[i_best]:.1f} C; next suggested print: {Tg[np.argmax(ei)]:.1f} C")




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


# %% [run everything]
if __name__ == "__main__":
    df = main()                       # step 1: YOLO -> image_features.csv
    st = per_setting(df)              # step 2: features -> score per setting
    st.to_csv(FEATURES_CSV.parent / "setting_scores.csv", index=False)
    analyse(st)
    plot_results(st, FEATURES_CSV.parent / "score_vs_temperature.png")
