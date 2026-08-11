#!/usr/bin/env python3

from pathlib import Path
import ast
import pandas as pd
import numpy as np


RESCUE_CSV = Path(
    "results/baseline_complementarity/audit35/baseline_rescued_fn.csv"
)

BASE = Path(
    "results/baseline_complementarity/cache_original"
)

RARE_O = Path(
    "results/fn_analysis/cache"
)

RARE_H = Path(
    "results/fn_analysis/cache_hflip"
)


def iou_one_to_many(box, boxes):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)

    area1 = max(0, box[2] - box[0]) * max(0, box[3] - box[1])

    area2 = (
        np.maximum(0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0, boxes[:, 3] - boxes[:, 1])
    )

    union = area1 + area2 - inter

    return np.divide(
        inter,
        union,
        out=np.zeros_like(inter),
        where=union > 0,
    )


df = pd.read_csv(RESCUE_CSV)

print("=" * 88)
print("BASELINE RESCUE STRUCTURE AUDIT")
print("=" * 88)

for i, r in df.iterrows():

    stem = Path(r["image"]).stem
    cls = int(r["gt_class_id"])

    target_score = float(r["baseline_best_score"])
    target_box = np.asarray(
        ast.literal_eval(str(r["baseline_best_box"])),
        dtype=np.float32,
    )

    b = np.load(BASE / f"{stem}.npz")["candidates"]
    o = np.load(RARE_O / f"{stem}.npz")["candidates"]
    h = np.load(RARE_H / f"{stem}.npz")["candidates"]

    b_same = b[b[:, 0].astype(int) == cls]
    o_same = o[o[:, 0].astype(int) == cls]
    h_same = h[h[:, 0].astype(int) == cls]

    # Score ranks
    rank_same = int((b_same[:, 1] > target_score).sum()) + 1
    rank_all = int((b[:, 1] > target_score).sum()) + 1

    # Baseline rescue box vs RareOS proposal space
    oi = iou_one_to_many(target_box, o_same[:, 10:14])
    hi = iou_one_to_many(target_box, h_same[:, 10:14])

    omax = float(oi.max()) if len(oi) else 0.0
    hmax = float(hi.max()) if len(hi) else 0.0

    # Retrieve closest RareOS proposal metadata
    if len(oi):
        oj = int(np.argmax(oi))
        obox = o_same[oj, 10:14]
        oscore = float(o_same[oj, 1])
    else:
        obox = None
        oscore = 0.0

    if len(hi):
        hj = int(np.argmax(hi))
        hbox = h_same[hj, 10:14]
        hscore = float(h_same[hj, 1])
    else:
        hbox = None
        hscore = 0.0

    print()
    print("-" * 88)
    print(f"RESCUE #{i + 1}")
    print("image                  :", r["image"])
    print("class                  :", r["gt_class_name"])
    print("failure                :", r["failure_type"])
    print()
    print("GT                     :",
          [r["gt_x1"], r["gt_y1"], r["gt_x2"], r["gt_y2"]])
    print("Baseline rescue box    :", target_box.tolist())
    print("Baseline -> GT IoU     :", f"{r['baseline_best_iou']:.6f}")
    print("Baseline score         :", f"{target_score:.8g}")
    print()
    print("Baseline same-class N  :", len(b_same))
    print("same-class score rank  :", f"{rank_same}/{len(b_same)}")
    print("same-class percentile  :", f"{rank_same / len(b_same):.4f}")
    print("all-class score rank   :", f"{rank_all}/{len(b)}")
    print()
    print("Baseline box -> RareOS Original max same-class IoU :",
          f"{omax:.6f}")
    print("closest O score        :", f"{oscore:.8g}")
    print("closest O box          :",
          None if obox is None else obox.tolist())
    print()
    print("Baseline box -> RareOS HFlip max same-class IoU    :",
          f"{hmax:.6f}")
    print("closest H score        :", f"{hscore:.8g}")
    print("closest H box          :",
          None if hbox is None else hbox.tolist())
    print()
    print("survives class Top50   :", rank_same <= 50)
    print("survives class Top100  :", rank_same <= 100)
    print("survives class Top200  :", rank_same <= 200)
    print("survives class Top400  :", rank_same <= 400)
    print("survives class Top600  :", rank_same <= 600)
    print("survives class Top1000 :", rank_same <= 1000)

print()
print("=" * 88)
