#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
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
    p.add_argument("--pair-topk", type=int, default=3)
    p.add_argument("--pair-chunk", type=int, default=32)
    return p.parse_args()


def pass_range(value, cfg, key):
    lo = cfg.get("min_" + key, None)
    hi = cfg.get("max_" + key, None)

    if lo is not None and value < lo:
        return False
    if hi is not None and value > hi:
        return False
    return True


def extended_gate(p, cfg):
    if p["cls"] not in cfg["classes"]:
        return False

    for key in [
        "pair_iou",
        "dxn",
        "dyn",
        "wr",
        "hr",
        "ar",
        "score_ratio",
    ]:
        if not pass_range(p[key], cfg, key):
            return False

    # Optional individual parent-score limits.
    oscore = float(p["o"][1])
    hscore = float(p["h"][1])

    if not pass_range(oscore, cfg, "oscore"):
        return False

    if not pass_range(hscore, cfg, "hscore"):
        return False

    return True


def make_rows(s20, pairs, cfg, cols):
    rows = []

    for p in pairs:
        if not extended_gate(p, cfg):
            continue

        for method in cfg["methods"]:
            box = s20.fuse_box(p, method)

            row = np.zeros(cols, dtype=np.float32)

            row[0] = p["cls"]

            # Keep all existing Final Combo predictions ahead
            # of synthetic fusion proposals.
            row[1] = (
                min(
                    float(p["o"][1]),
                    float(p["h"][1]),
                )
                * 0.10
            )

            row[2:6] = box
            rows.append(row)

    if not rows:
        return np.empty((0, cols), dtype=np.float32)

    return np.stack(rows)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combo = load_module(
        Path("scripts/18_final_combo_from_cache.py"),
        "combo18_s23",
    )
    diag = load_module(
        Path("scripts/14_fn_diagnostic.py"),
        "diag14_s23",
    )
    s19 = load_module(
        Path("scripts/19_final_combo_fn_diagnostic.py"),
        "diag19_s23",
    )
    s20 = load_module(
        Path("scripts/20_crossview_fusion_sweep.py"),
        "fusion20_s23",
    )

    device = torch.device(args.device)

    om = combo.load_manifest(args.original_cache)
    hm = combo.load_manifest(args.hflip_cache)

    omap = combo.manifest_map(om)
    hmap = combo.manifest_map(hm)

    # Exact script-18 defaults.
    old_argv = sys.argv[:]

    try:
        sys.argv = [
            "18_final_combo_from_cache.py",
            "--mode", "val",
            "--original-cache", "_dummy_orig",
            "--hflip-cache", "_dummy_hflip",
            "--output-dir", "_dummy_out",
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

    #
    # ORIGINAL rules from script 22.
    #
    rules = {
        "J0": {
            "classes": {0},
            "min_pair_iou": 0.00,
            "max_dxn": 1.00,
            "max_dyn": 1.00,
            "max_wr": 4.0,
            "max_hr": 4.0,
            "max_ar": 6.0,
            "methods": ["envelope"],
        },

        "H0": {
            "classes": {5},
            "min_pair_iou": 0.25,
            "max_dxn": 0.60,
            "max_dyn": 0.60,
            "max_wr": 3.0,
            "max_hr": 3.0,
            "max_ar": 4.0,
            "methods": ["envelope"],
        },

        "A0": {
            "classes": {3},
            "min_pair_iou": 0.25,
            "max_dxn": 0.60,
            "max_dyn": 0.60,
            "max_wr": 3.0,
            "max_hr": 3.0,
            "max_ar": 4.0,
            "methods": ["avg50"],
        },

        "B0": {
            "classes": {3},
            "min_pair_iou": 0.00,
            "max_dxn": 2.10,
            "max_dyn": 0.60,
            "max_wr": 2.0,
            "max_hr": 3.0,
            "max_ar": 4.0,
            "methods": ["envelope"],
        },

        #
        # J: horizontal-separated but similar-size boxes.
        #
        "J1": {
            "classes": {0},
            "min_pair_iou": 0.00,
            "max_pair_iou": 0.12,

            "min_dxn": 0.75,
            "max_dxn": 1.05,

            "max_dyn": 0.25,

            "max_wr": 1.60,
            "max_hr": 1.60,
            "max_ar": 2.00,

            "max_score_ratio": 10.0,

            "methods": ["envelope"],
        },

        "J2": {
            "classes": {0},
            "min_pair_iou": 0.00,
            "max_pair_iou": 0.08,

            "min_dxn": 0.85,
            "max_dxn": 1.00,

            "max_dyn": 0.18,

            "max_wr": 1.40,
            "max_hr": 1.40,
            "max_ar": 1.70,

            "max_score_ratio": 6.0,

            "methods": ["envelope"],
        },

        #
        # H: y-aligned / horizontally shifted boxes.
        #
        "H1": {
            "classes": {5},

            "min_pair_iou": 0.25,
            "max_pair_iou": 0.50,

            "min_dxn": 0.30,
            "max_dxn": 0.55,

            "max_dyn": 0.06,

            "min_wr": 1.45,
            "max_wr": 2.20,

            "max_hr": 1.30,

            "min_ar": 1.30,
            "max_ar": 2.30,

            "methods": ["envelope"],
        },

        # Same geometry + strong cross-view confidence asymmetry.
        "H2": {
            "classes": {5},

            "min_pair_iou": 0.25,
            "max_pair_iou": 0.50,

            "min_dxn": 0.30,
            "max_dxn": 0.55,

            "max_dyn": 0.06,

            "min_wr": 1.45,
            "max_wr": 2.20,

            "max_hr": 1.30,

            "min_ar": 1.30,
            "max_ar": 2.30,

            "min_score_ratio": 100.0,

            "methods": ["envelope"],
        },

        #
        # A: x centers aligned but large height disagreement.
        #
        "A1": {
            "classes": {3},

            "min_pair_iou": 0.25,
            "max_pair_iou": 0.40,

            "max_dxn": 0.15,

            "min_dyn": 0.25,
            "max_dyn": 0.50,

            "max_wr": 1.50,

            "min_hr": 2.00,
            "max_hr": 3.30,

            "min_ar": 2.40,
            "max_ar": 4.00,

            "max_score_ratio": 5.0,

            "methods": ["avg50"],
        },

        "A2": {
            "classes": {3},

            "min_pair_iou": 0.28,
            "max_pair_iou": 0.36,

            "max_dxn": 0.10,

            "min_dyn": 0.30,
            "max_dyn": 0.45,

            "max_wr": 1.35,

            "min_hr": 2.40,
            "max_hr": 3.20,

            "min_ar": 2.80,
            "max_ar": 3.70,

            "max_score_ratio": 4.0,

            "methods": ["avg50"],
        },

        #
        # B: non-overlapping, horizontally separated,
        # similar-size jiaza boxes.
        #
        "B1": {
            "classes": {3},

            "min_pair_iou": 0.00,
            "max_pair_iou": 0.05,

            "min_dxn": 1.60,
            "max_dxn": 2.10,

            "min_dyn": 0.10,
            "max_dyn": 0.45,

            "max_wr": 1.70,
            "max_hr": 1.70,
            "max_ar": 2.30,

            "max_score_ratio": 6.0,

            "methods": ["envelope"],
        },

        "B2": {
            "classes": {3},

            "min_pair_iou": 0.00,
            "max_pair_iou": 0.02,

            "min_dxn": 1.75,
            "max_dxn": 2.05,

            "min_dyn": 0.15,
            "max_dyn": 0.40,

            "max_wr": 1.65,
            "max_hr": 1.60,
            "max_ar": 2.10,

            "max_score_ratio": 5.5,

            "methods": ["envelope"],
        },
    }

    #
    # Main combinations.
    #
    variants = {
        # Exact script-22 candidate:
        "original_targeted":
            ["J0", "H0", "A0", "B0"],

        # Replace one rule at a time.
        "prune_J":
            ["J1", "H0", "A0", "B0"],

        "prune_H":
            ["J0", "H1", "A0", "B0"],

        "prune_H_score":
            ["J0", "H2", "A0", "B0"],

        "prune_A":
            ["J0", "H0", "A1", "B0"],

        "prune_B":
            ["J0", "H0", "A0", "B1"],

        # First fully-pruned candidate.
        "pruned_v1":
            ["J1", "H1", "A1", "B1"],

        "pruned_v1_Hscore":
            ["J1", "H2", "A1", "B1"],

        # More aggressive.
        "pruned_v2":
            ["J2", "H1", "A2", "B2"],

        "pruned_v2_Hscore":
            ["J2", "H2", "A2", "B2"],

        # Atomic pruned rules, useful for checking whether
        # each rescue survives independently.
        "J1_only": ["J1"],
        "J2_only": ["J2"],
        "H1_only": ["H1"],
        "H2_only": ["H2"],
        "A1_only": ["A1"],
        "A2_only": ["A2"],
        "B1_only": ["B1"],
        "B2_only": ["B2"],
    }

    stats = {
        name: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "added": 0,
        }
        for name in variants
    }

    rule_added = {
        name: 0
        for name in rules
    }

    base = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }

    print("===== TARGETED FUSION PRUNE SWEEP =====")
    print("Images     :", len(omap))
    print("Pair top-k :", args.pair_topk)
    print("Pair chunk :", args.pair_chunk)
    print()

    for idx, (name, item) in enumerate(omap.items(), 1):
        hitem = hmap[name]

        onpz = np.load(
            args.original_cache / item["cache_file"]
        )
        hnpz = np.load(
            args.hflip_cache / hitem["cache_file"]
        )

        original = onpz["candidates"].astype(
            np.float32,
            copy=False,
        )

        hflip = hnpz["candidates"].astype(
            np.float32,
            copy=False,
        )

        # Exact Final Combo.
        hsel = hflip[
            np.isin(
                hflip[:, 0].astype(np.int32),
                list(hclasses),
            )
        ]

        union_pre = (
            np.concatenate([original, hsel], axis=0)
            if len(hsel)
            else original
        )

        post, _ = combo.nms(
            union_pre,
            0.90,
            diag,
            device,
        )

        final, _ = combo.add_stitched(
            post,
            stitch_args,
            device,
        )

        width, height = s19.image_size(
            item,
            om,
            name,
            onpz,
        )

        label_path = s19.find_label(
            args.labels,
            name,
        )

        gt = diag.read_yolo_gt(
            label_path,
            width,
            height,
        )

        btp, bfp, bfn, *_ = diag.match_predictions(
            final,
            gt,
            0.50,
        )

        base["tp"] += int(btp)
        base["fp"] += int(bfp)
        base["fn"] += int(bfn)

        # Only 0 / 3 / 5 are needed.
        pairs = []

        for cls in [0, 3, 5]:
            pairs.extend(
                s20.build_pair_features(
                    original,
                    hflip,
                    cls,
                    args.pair_topk,
                    args.pair_chunk,
                )
            )

        rule_boxes = {}

        for rname, cfg in rules.items():
            fused = make_rows(
                s20,
                pairs,
                cfg,
                final.shape[1],
            )

            rule_boxes[rname] = fused
            rule_added[rname] += len(fused)

        for vname, names in variants.items():
            pieces = [final]
            added = 0

            for rname in names:
                x = rule_boxes[rname]

                if len(x):
                    pieces.append(x)
                    added += len(x)

            augmented = (
                np.concatenate(pieces, axis=0)
                if len(pieces) > 1
                else final
            )

            tp, fp, fn, *_ = diag.match_predictions(
                augmented,
                gt,
                0.50,
            )

            st = stats[vname]

            st["tp"] += int(tp)
            st["fp"] += int(fp)
            st["fn"] += int(fn)
            st["added"] += added

        if idx % 50 == 0 or idx == len(omap):
            print(
                f"{idx}/{len(omap)} "
                f"baseTP={base['tp']} "
                f"baseFN={base['fn']}"
            )

    print()
    print("===== BASELINE CHECK =====")

    print(
        f"TP={base['tp']} "
        f"FP={base['fp']} "
        f"FN={base['fn']}"
    )

    if base["tp"] != 824 or base["fn"] != 21:
        raise RuntimeError(
            f"Baseline mismatch: {base}"
        )

    print()
    print("===== RULE BOX COUNTS =====")

    order = [
        "J0", "J1", "J2",
        "H0", "H1", "H2",
        "A0", "A1", "A2",
        "B0", "B1", "B2",
    ]

    for x in order:
        print(
            f"{x:3s} "
            f"added={rule_added[x]:,}"
        )

    rows = []

    for name, s in stats.items():
        denom = s["tp"] + s["fn"]

        recall = (
            s["tp"] / denom
            if denom else 0.0
        )

        rows.append({
            "config": name,
            "tp": s["tp"],
            "fp": s["fp"],
            "fn": s["fn"],
            "recall": recall,
            "score_like": recall * 100,
            "delta_tp": s["tp"] - base["tp"],
            "delta_fn": base["fn"] - s["fn"],
            "delta_fp": s["fp"] - base["fp"],
            "added_fusion_boxes": s["added"],
        })

    rows.sort(
        key=lambda r: (
            -r["tp"],
            r["added_fusion_boxes"],
            r["delta_fp"],
        )
    )

    print()
    print("===== PRUNE RESULTS =====")

    for r in rows:
        print(
            f"{r['config']:22s} "
            f"TP={r['tp']:3d} "
            f"FN={r['fn']:2d} "
            f"Recall={r['recall']:.6f} "
            f"ScoreLike={r['score_like']:.2f} "
            f"dTP={r['delta_tp']:+d} "
            f"dFP={r['delta_fp']:+d} "
            f"added={r['added_fusion_boxes']:,}"
        )

    out = args.output_dir / "prune_summary.csv"

    with out.open(
        "w",
        newline="",
        encoding="utf-8",
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
