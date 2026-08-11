#!/usr/bin/env python3

from pathlib import Path
import ast
import numpy as np
import pandas as pd
import yaml


BASE = Path("results/baseline_complementarity/cache_original")
RO = Path("results/fn_analysis/cache")
RH = Path("results/fn_analysis/cache_hflip")

RESCUE = Path(
    "results/baseline_complementarity/audit35/baseline_rescued_fn.csv"
)

YAML = Path("configs/steel_tiles_1280.yaml")

MIN_SCORE = 2e-5

UPPER_SCORES = [
    3e-4,
    1e-3,
    1e-2,
    float("inf"),
]

OVERLAP_GATES = [
    0.60,
    0.65,
    0.70,
    0.80,
    1.00,
]


def iou_many_to_many(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)

    ax1 = a[:, 0][:, None]
    ay1 = a[:, 1][:, None]
    ax2 = a[:, 2][:, None]
    ay2 = a[:, 3][:, None]

    bx1 = b[:, 0][None, :]
    by1 = b[:, 1][None, :]
    bx2 = b[:, 2][None, :]
    by2 = b[:, 3][None, :]

    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)

    iw = np.maximum(0, ix2 - ix1)
    ih = np.maximum(0, iy2 - iy1)

    inter = iw * ih

    aa = (
        np.maximum(0, ax2 - ax1)
        * np.maximum(0, ay2 - ay1)
    )

    ba = (
        np.maximum(0, bx2 - bx1)
        * np.maximum(0, by2 - by1)
    )

    union = aa + ba - inter

    return np.divide(
        inter,
        union,
        out=np.zeros_like(inter),
        where=union > 0,
    )


def load_names():
    with YAML.open("r", encoding="utf-8") as f:
        y = yaml.safe_load(f)

    names = y["names"]

    if isinstance(names, list):
        return {str(v): i for i, v in enumerate(names)}

    return {str(v): int(k) for k, v in names.items()}


name_to_id = load_names()
cls = name_to_id["huashang"]

rescues = pd.read_csv(RESCUE)

# rescue lookup:
# exact Baseline box for each known oracle rescue
rescue_boxes = []

for _, r in rescues.iterrows():
    rescue_boxes.append(
        (
            Path(r["image"]).stem,
            np.asarray(
                ast.literal_eval(r["baseline_best_box"]),
                dtype=np.float32,
            ),
            float(r["baseline_best_score"]),
        )
    )


results = []

files = sorted(BASE.glob("*.npz"))

print("Images:", len(files))
print("Class : huashang =", cls)
print()


for upper in UPPER_SCORES:

    counts = {
        gate: 0
        for gate in OVERLAP_GATES
    }

    rescue_kept = {
        gate: [False] * len(rescue_boxes)
        for gate in OVERLAP_GATES
    }

    for n, bp in enumerate(files, 1):

        stem = bp.stem

        b = np.load(bp)["candidates"]

        b = b[
            (b[:, 0].astype(int) == cls)
            & (b[:, 1] >= MIN_SCORE)
            & (b[:, 1] <= upper)
        ]

        if len(b) == 0:
            continue

        op = RO / bp.name
        hp = RH / bp.name

        o = np.load(op)["candidates"]
        h = np.load(hp)["candidates"]

        o = o[o[:, 0].astype(int) == cls]
        h = h[h[:, 0].astype(int) == cls]

        ref_boxes = []

        if len(o):
            ref_boxes.append(o[:, 10:14])

        if len(h):
            ref_boxes.append(h[:, 10:14])

        if ref_boxes:
            ref = np.concatenate(ref_boxes, axis=0)

            ious = iou_many_to_many(
                b[:, 10:14],
                ref,
            )

            max_iou = ious.max(axis=1)

        else:
            max_iou = np.zeros(
                len(b),
                dtype=np.float32,
            )

        for gate in OVERLAP_GATES:

            keep = max_iou < gate

            counts[gate] += int(keep.sum())

            # Check exact known rescue boxes.
            for ri, (rstem, rbox, rscore) in enumerate(rescue_boxes):

                if stem != rstem:
                    continue

                if not (MIN_SCORE <= rscore <= upper):
                    continue

                boxes = b[:, 10:14]

                exact = np.all(
                    np.isclose(
                        boxes,
                        rbox[None, :],
                        atol=1e-4,
                    ),
                    axis=1,
                )

                inds = np.where(exact)[0]

                if len(inds) == 0:
                    continue

                j = int(inds[0])

                if keep[j]:
                    rescue_kept[gate][ri] = True

        if n % 50 == 0:
            print(
                f"upper={upper} "
                f"{n}/{len(files)}"
            )

    for gate in OVERLAP_GATES:

        kept = sum(rescue_kept[gate])

        results.append(
            {
                "min_score": MIN_SCORE,
                "upper_score": upper,
                "overlap_gate": gate,
                "added_boxes": counts[gate],
                "oracle_rescues_kept": kept,
            }
        )


df = pd.DataFrame(results)

print()
print("=" * 76)
print("HUASHANG BASELINE COMPLEMENT GATE SWEEP")
print("=" * 76)

print(
    df.to_string(
        index=False,
    )
)

out = Path(
    "results/baseline_complementarity/"
    "huashang_gate_sweep.csv"
)

df.to_csv(
    out,
    index=False,
)

print()
print("Saved:", out)
