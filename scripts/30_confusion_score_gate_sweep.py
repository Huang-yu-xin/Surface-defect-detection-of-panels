#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


CLASS_NAMES = [
    "jieba", "zonglie", "qilie", "jiaza", "yiwuyaru",
    "huashang", "mamianmakeng", "yanghuatiepi", "gunyin",
]


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--original-cache", type=Path, required=True)
    p.add_argument("--hflip-cache", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dup-score", type=float, default=1e-6)
    return p.parse_args()


def make_dup(final, rules, dup_score):
    """
    rules: [(src, dst, topk, min_source_score), ...]
    """
    cls = final[:, 0].astype(np.int32)
    pieces = []

    for src, dst, topk, score_floor in rules:
        rows = final[cls == src]

        if not len(rows):
            continue

        order = np.argsort(-rows[:, 1], kind="stable")
        rows = rows[order[:topk]]

        rows = rows[
            rows[:, 1] >= score_floor
        ]

        if not len(rows):
            continue

        dup = rows.copy()
        dup[:, 0] = dst
        dup[:, 1] = dup_score
        pieces.append(dup)

    if not pieces:
        return np.empty(
            (0, final.shape[1]),
            dtype=np.float32,
        )

    return np.concatenate(pieces, axis=0)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combo = load_module(
        Path("scripts/18_final_combo_from_cache.py"),
        "combo18_s30",
    )
    diag = load_module(
        Path("scripts/14_fn_diagnostic.py"),
        "diag14_s30",
    )

    device = torch.device(args.device)

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "18_final_combo_from_cache.py",
            "--mode", "val",
            "--original-cache", str(args.original_cache),
            "--hflip-cache", str(args.hflip_cache),
            "--labels", str(args.labels),
            "--output-dir", str(args.output_dir / "_dummy"),
            "--device", args.device,
        ]
        defaults = combo.parse_args()
    finally:
        sys.argv = old_argv

    stitch_args = SimpleNamespace(**vars(defaults))
    stitch_args.min_aspect = 5.0
    stitch_args.x_tol = 64.0
    stitch_args.max_y_gap = 64.0
    stitch_args.min_merged_height = 1300.0

    hclasses = {0, 2, 3, 4, 7}

    om = combo.load_manifest(args.original_cache)
    hm = combo.load_manifest(args.hflip_cache)
    hmap = combo.manifest_map(hm)

    # Fixed coarse thresholds only.
    variants = {
        # Atomic 7 -> 0
        "7to0_t20_s0":       [(7, 0, 20, 0.0)],
        "7to0_t20_s1e5":     [(7, 0, 20, 1e-5)],
        "7to0_t20_s3e5":     [(7, 0, 20, 3e-5)],
        "7to0_t20_s1e4":     [(7, 0, 20, 1e-4)],
        "7to0_t20_s3e4":     [(7, 0, 20, 3e-4)],
        "7to0_t20_s1e3":     [(7, 0, 20, 1e-3)],

        # Atomic 6 -> 0
        "6to0_t5_s0":        [(6, 0, 5, 0.0)],
        "6to0_t5_s1e4":      [(6, 0, 5, 1e-4)],
        "6to0_t5_s3e4":      [(6, 0, 5, 3e-4)],
        "6to0_t5_s1e3":      [(6, 0, 5, 1e-3)],
        "6to0_t5_s3e3":      [(6, 0, 5, 3e-3)],
        "6to0_t5_s1e2":      [(6, 0, 5, 1e-2)],

        # Mixed candidates
        "mix_original": [
            (7, 0, 20, 0.0),
            (6, 0, 5, 0.0),
        ],
        "mix_loose": [
            (7, 0, 20, 3e-5),
            (6, 0, 5, 3e-4),
        ],
        "mix_mid": [
            (7, 0, 20, 1e-4),
            (6, 0, 5, 1e-3),
        ],
        "mix_strict": [
            (7, 0, 20, 3e-4),
            (6, 0, 5, 3e-3),
        ],
        "mix_t15_mid": [
            (7, 0, 15, 1e-4),
            (6, 0, 5, 1e-3),
        ],
    }

    totals = {"baseline": Counter()}
    added = Counter()

    for v in variants:
        totals[v] = Counter()

    print("===== CONFUSION SCORE-GATE SWEEP =====")
    print("Images:", om["images_count"])
    print()

    for idx, oitem in enumerate(om["items"], start=1):
        name = oitem["image_name"]
        hitem = hmap[name]

        onpz = np.load(
            args.original_cache / oitem["cache_file"]
        )
        hnpz = np.load(
            args.hflip_cache / hitem["cache_file"]
        )

        orig = onpz["candidates"].astype(
            np.float32, copy=False
        )
        hflip = hnpz["candidates"].astype(
            np.float32, copy=False
        )

        height, width = map(int, onpz["image_shape"])

        hsel = hflip[
            np.isin(
                hflip[:, 0].astype(np.int32),
                list(hclasses),
            )
        ]

        union_pre = (
            np.concatenate([orig, hsel], axis=0)
            if len(hsel)
            else orig
        )

        post, _ = combo.nms(
            union_pre,
            defaults.global_iou,
            diag,
            device,
        )

        final, _ = combo.add_stitched(
            post,
            stitch_args,
            device,
        )

        gt = diag.read_yolo_gt(
            args.labels / f"{Path(name).stem}.txt",
            width,
            height,
        )

        tp, fp, fn, *_ = diag.match_predictions(
            final,
            gt,
            defaults.match_iou,
        )

        totals["baseline"]["tp"] += tp
        totals["baseline"]["fp"] += fp
        totals["baseline"]["fn"] += fn

        for vname, rules in variants.items():
            dup = make_dup(
                final,
                rules,
                args.dup_score,
            )

            aug = (
                np.concatenate([final, dup], axis=0)
                if len(dup)
                else final
            )

            vtp, vfp, vfn, *_ = diag.match_predictions(
                aug,
                gt,
                defaults.match_iou,
            )

            totals[vname]["tp"] += vtp
            totals[vname]["fp"] += vfp
            totals[vname]["fn"] += vfn
            added[vname] += len(dup)

        if idx % 50 == 0 or idx == int(om["images_count"]):
            print(f"{idx}/{om['images_count']}")

    b = totals["baseline"]

    print()
    print("===== BASELINE CHECK =====")
    print(
        f"TP={b['tp']} FP={b['fp']} FN={b['fn']}"
    )

    if (
        b["tp"] != 824
        or b["fp"] != 1631208
        or b["fn"] != 21
    ):
        raise RuntimeError("Baseline mismatch")

    print()
    print("===== SCORE-GATE RESULTS =====")

    rows = []

    for vname in variants:
        r = totals[vname]

        dtp = r["tp"] - b["tp"]
        dfn = r["fn"] - b["fn"]
        dfp = r["fp"] - b["fp"]

        recall = r["tp"] / (r["tp"] + r["fn"])

        print(
            f"{vname:20s} "
            f"TP={r['tp']} "
            f"FN={r['fn']} "
            f"ScoreLike={recall*100:.2f} "
            f"dTP={dtp:+d} "
            f"dFN={dfn:+d} "
            f"dFP={dfp:+d} "
            f"added={added[vname]:,}"
        )

        rows.append({
            "variant": vname,
            "tp": r["tp"],
            "fp": r["fp"],
            "fn": r["fn"],
            "recall": recall,
            "score_like": recall * 100,
            "delta_tp": dtp,
            "delta_fp": dfp,
            "delta_fn": dfn,
            "added_boxes": added[vname],
        })

    out = args.output_dir / "score_gate_summary.csv"

    with out.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        w.writeheader()
        w.writerows(rows)

    print()
    print("Saved:", out)


if __name__ == "__main__":
    main()
