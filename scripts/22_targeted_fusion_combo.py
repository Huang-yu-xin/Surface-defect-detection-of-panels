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


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combo = load_module(
        Path("scripts/18_final_combo_from_cache.py"),
        "combo18_s22",
    )
    diag = load_module(
        Path("scripts/14_fn_diagnostic.py"),
        "diag14_s22",
    )
    s19 = load_module(
        Path("scripts/19_final_combo_fn_diagnostic.py"),
        "diag19_s22",
    )
    s20 = load_module(
        Path("scripts/20_crossview_fusion_sweep.py"),
        "fusion20_s22",
    )

    device = torch.device(args.device)

    om = combo.load_manifest(args.original_cache)
    hm = combo.load_manifest(args.hflip_cache)

    omap = combo.manifest_map(om)
    hmap = combo.manifest_map(hm)

    #
    # Exact script-18 defaults.
    #
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

    # Existing selective HFlip classes.
    hclasses = {0, 2, 3, 4, 7}

    #
    # Four targeted rules.
    #
    rules = {
        #
        # jieba:
        # loose_env rescue
        #
        "J": {
            "name": "jieba_loose_env",
            "classes": {0},
            "min_pair_iou": 0.00,
            "max_dxn": 1.00,
            "max_dyn": 1.00,
            "max_wr": 4.0,
            "max_hr": 4.0,
            "max_ar": 6.0,
            "methods": ["envelope"],
        },

        #
        # huashang:
        # use tight_env instead of loose_env,
        # because both rescued the same GT but tight is safer.
        #
        "H": {
            "name": "huashang_tight_env",
            "classes": {5},
            "min_pair_iou": 0.25,
            "max_dxn": 0.60,
            "max_dyn": 0.60,
            "max_wr": 3.0,
            "max_hr": 3.0,
            "max_ar": 4.0,
            "methods": ["envelope"],
        },

        #
        # jiaza GT#3:
        # tight_avg
        #
        "A": {
            "name": "jiaza_tight_avg",
            "classes": {3},
            "min_pair_iou": 0.25,
            "max_dxn": 0.60,
            "max_dyn": 0.60,
            "max_wr": 3.0,
            "max_hr": 3.0,
            "max_ar": 4.0,
            "methods": ["avg50"],
        },

        #
        # jiaza GT#13:
        # wide envelope
        #
        "B": {
            "name": "jiaza_wide_env",
            "classes": {3},
            "min_pair_iou": 0.00,
            "max_dxn": 2.10,
            "max_dyn": 0.60,
            "max_wr": 2.0,
            "max_hr": 3.0,
            "max_ar": 4.0,
            "methods": ["envelope"],
        },
    }

    #
    # Ablations + combined variants.
    #
    variants = {
        "jieba_only": ["J"],
        "huashang_only": ["H"],
        "jiaza_tight_only": ["A"],
        "jiaza_wide_only": ["B"],

        "jiaza_both": ["A", "B"],
        "jieba_huashang": ["J", "H"],

        "targeted_no_wide": ["J", "H", "A"],

        # Main candidate.
        "targeted_all": ["J", "H", "A", "B"],
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

    base = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }

    rule_added = {
        k: 0
        for k in rules
    }

    print("===== TARGETED FUSION COMBO =====")
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

        #
        # Exact Final Combo.
        #
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

        #
        # Only classes actually needed:
        # 0 jieba
        # 3 jiaza
        # 5 huashang
        #
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

        #
        # Build each atomic rule once.
        #
        rule_boxes = {}

        for key, cfg in rules.items():
            fused = s20.make_fusion_rows(
                pairs,
                cfg,
                final.shape[1],
            )

            rule_boxes[key] = fused
            rule_added[key] += len(fused)

        #
        # Evaluate combinations.
        #
        for vname, keys in variants.items():
            pieces = [final]

            added = 0

            for key in keys:
                x = rule_boxes[key]

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

            s = stats[vname]

            s["tp"] += int(tp)
            s["fp"] += int(fp)
            s["fn"] += int(fn)
            s["added"] += added

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
    print("===== ATOMIC RULE BOX COUNTS =====")

    for key in ["J", "H", "A", "B"]:
        print(
            f"{key} "
            f"{rules[key]['name']:24s} "
            f"added={rule_added[key]:,}"
        )

    rows = []

    for name, s in stats.items():
        denom = s["tp"] + s["fn"]
        recall = (
            s["tp"] / denom
            if denom
            else 0.0
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
        )
    )

    print()
    print("===== TARGETED RESULTS =====")

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

    out_csv = args.output_dir / "targeted_summary.csv"

    with out_csv.open(
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
    print("Saved:", out_csv)


if __name__ == "__main__":
    main()
