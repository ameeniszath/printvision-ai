"""
Evaluate one or more YOLO weight files on the same dataset split and print a
comparison table (mAP50, mAP50-95, precision, recall, best F1 and its
confidence, per-class AP50).

Usage:
    python scripts/evaluate.py <data.yaml> <split> <weights1.pt> [weights2.pt ...]
    e.g. python scripts/evaluate.py datasets/v5_clean/data.yaml test weights/best.pt
"""
import json
import sys

import numpy as np
from ultralytics import YOLO


def evaluate(weights, data, split, imgsz=640):
    m = YOLO(weights).val(data=data, split=split, imgsz=imgsz, conf=0.001, batch=8,
                          device="cpu", plots=False, verbose=False)
    f1 = m.box.f1_curve.mean(0)
    i = int(np.argmax(f1))
    return {
        "mAP50": round(float(m.box.map50), 3),
        "mAP50_95": round(float(m.box.map), 3),
        "precision": round(float(m.box.mp), 3),
        "recall": round(float(m.box.mr), 3),
        "best_F1": round(float(f1[i]), 3),
        "best_conf": round(float(m.box.px[i]), 3),
        "per_class_AP50": {m.names[int(c)]: round(float(a), 3)
                           for c, a in zip(m.box.ap_class_index, m.box.ap50)},
    }


if __name__ == "__main__":
    data, split, *weights = sys.argv[1:]
    results = {w: evaluate(w, data, split) for w in weights}
    print(json.dumps(results, indent=2))
