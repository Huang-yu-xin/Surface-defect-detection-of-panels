#!/usr/bin/env python3

from pathlib import Path
import argparse
import csv

import numpy as np
import yaml


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--rareos-original",
        type=Path,
        default=Path("results/fn_analysis/cache"),
    )
    p.add_argument(
        "--rareos-hflip",
        type=Path,
        default=Path("results/fn_analysis/cache_hflip"),
    )
    p.add_argument(
        "--baseline-cache",
        type=Path,
        default=Path("results/baseline_complementarity/cache_original"),
    )
    p.add_argument(
        "--dataset-yaml",
        type=Path,
        default=Path("configs/steel_tiles_1280.yaml"),
    )

    p.add_argument("--min-score", type=float, default=2e-5)
    p.add_argument("--max-score", type=float, default=3e-4)
    p.add_argument("--overlap-gate", type=float, default=0.65)

    p.add_argument(
        "--output-cache",
        type=Path,
        required=True,
    )

    return p.parse_args()


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

    inter = (
        np.maximum(0.0, ix2 - ix1)
        * np.maximum(0.0, iy2 - iy1)
    )

    area_a = (
        np.maximum(0.0, ax2 - ax1)
        * np.maximum(0.0, ay2 - ay1)
    )

    area_b = (
        np.maximum(0.0, bx2 - bx1)
        * np.maximum(0.0, by2 - by1)
    )

    union = area_a + area_b - inter

    return np.divide(
        inter,
        union,
        out=np.zeros_like(inter, dtype=np.float32),
        where=union > 0,
    )


def get_class_id(yaml_path, target):
    with yaml_path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)

    names = obj["names"]

    if isinstance(names, list):
        for i, name in enumerate(names):
            if str(name) == target:
                return i
    else:
        for k, name in names.items():
            if str(name) == target:
                return int(k)

    raise RuntimeError(f"Class not found: {target}")


def main():
    args = parse_args()
    args.output_cache.mkdir(parents=True, exist_ok=True)

    cls = get_class_id(args.dataset_yaml, "huashang")

    files = sorted(args.rareos_original.glob("*.npz"))

    total_original = 0
    total_selected = 0
    total_output = 0

    per_image = []

    print("===== BUILD BASELINE HUASHANG COMPLEMENT =====")
    print("images       :", len(files))
    print("class        : huashang", cls)
    print("score range  :", args.min_score, "~", args.max_score)
    print("overlap gate :", args.overlap_gate)
    print()

    for n, op in enumerate(files, 1):
        name = op.name

        hp = args.rareos_hflip / name
        bp = args.baseline_cache / name

        if not hp.exists():
            raise FileNotFoundError(hp)

        if not bp.exists():
            raise FileNotFoundError(bp)

        with np.load(op, allow_pickle=False) as d:
            odata = {k: np.asarray(d[k]) for k in d.files}

        with np.load(hp, allow_pickle=False) as d:
            h = np.asarray(d["candidates"], dtype=np.float32)

        with np.load(bp, allow_pickle=False) as d:
            b = np.asarray(d["candidates"], dtype=np.float32)

        o = np.asarray(odata["candidates"], dtype=np.float32)

        # Candidate Baseline huashang proposals.
        mask = (
            (b[:, 0].astype(np.int64) == cls)
            & (b[:, 1] >= args.min_score)
            & (b[:, 1] <= args.max_score)
        )

        cand = b[mask]

        # RareOS O/H huashang proposal space.
        o_same = o[o[:, 0].astype(np.int64) == cls]
        h_same = h[h[:, 0].astype(np.int64) == cls]

        refs = []

        if len(o_same):
            refs.append(o_same[:, 10:14])

        if len(h_same):
            refs.append(h_same[:, 10:14])

        if len(cand) == 0:
            selected = cand

        elif refs:
            ref_boxes = np.concatenate(refs, axis=0)

            ious = iou_many_to_many(
                cand[:, 10:14],
                ref_boxes,
            )

            max_iou = ious.max(axis=1)

            selected = cand[
                max_iou < args.overlap_gate
            ]

        else:
            selected = cand

        merged = np.concatenate(
            [o, selected],
            axis=0,
        ).astype(np.float32, copy=False)

        odata["candidates"] = merged

        np.savez(
            args.output_cache / name,
            **odata,
        )

        total_original += len(o)
        total_selected += len(selected)
        total_output += len(merged)

        per_image.append(
            {
                "image": op.stem,
                "original": len(o),
                "selected_baseline": len(selected),
                "output": len(merged),
            }
        )

        if n % 50 == 0 or n == len(files):
            print(
                f"{n:3d}/{len(files)} | "
                f"selected={total_selected:,}"
            )

    summary_csv = args.output_cache.parent / (
        args.output_cache.name + "_summary.csv"
    )

    with summary_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "original",
                "selected_baseline",
                "output",
            ],
        )
        writer.writeheader()
        writer.writerows(per_image)

    print()
    print("===== DONE =====")
    print("Original candidates :", f"{total_original:,}")
    print("Selected Baseline   :", f"{total_selected:,}")
    print("Output candidates   :", f"{total_output:,}")
    print("Output cache        :", args.output_cache)
    print("Summary CSV         :", summary_csv)


if __name__ == "__main__":
    main()
