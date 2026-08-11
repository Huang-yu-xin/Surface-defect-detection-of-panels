#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import sys
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


CLASS_NAMES = [
    "jieba",
    "zonglie",
    "qilie",
    "jiaza",
    "yiwuyaru",
    "huashang",
    "mamianmakeng",
    "yanghuatiepi",
    "gunyin",
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
    p.add_argument("--pair-topk", type=int, default=3)
    p.add_argument("--pair-chunk", type=int, default=32)

    return p.parse_args()


def get_unmatched_indices(
    diag,
    s19,
    pred,
    gt_obj,
    gt_array,
    match_iou=0.50,
):
    result = diag.match_predictions(
        pred,
        gt_obj,
        match_iou,
    )

    tp, fp, fn = map(int, result[:3])
    unmatched_ref = result[3]

    idx = s19.map_reference_unmatched(
        unmatched_ref,
        gt_array,
    )

    own = s19.our_unmatched_gt(
        pred,
        gt_array,
        match_iou,
    )

    if idx is None or len(idx) != fn:
        idx = own

    if len(idx) != fn:
        raise RuntimeError(
            f"Cannot map unmatched GT: "
            f"fn={fn}, mapped={len(idx)}, own={len(own)}"
        )

    return tp, fp, fn, set(idx)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combo = load_module(
        Path("scripts/18_final_combo_from_cache.py"),
        "combo18_s21",
    )

    diag = load_module(
        Path("scripts/14_fn_diagnostic.py"),
        "diag14_s21",
    )

    s19 = load_module(
        Path("scripts/19_final_combo_fn_diagnostic.py"),
        "diag19_s21",
    )

    s20 = load_module(
        Path("scripts/20_crossview_fusion_sweep.py"),
        "fusion20_s21",
    )

    device = torch.device(args.device)

    om = combo.load_manifest(args.original_cache)
    hm = combo.load_manifest(args.hflip_cache)

    omap = combo.manifest_map(om)
    hmap = combo.manifest_map(hm)

    #
    # Exact validated script-18 defaults.
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
        combo_defaults = combo.parse_args()
    finally:
        sys.argv = old_argv

    stitch_args = SimpleNamespace(
        **vars(combo_defaults)
    )

    stitch_args.min_aspect = 5.0
    stitch_args.x_tol = 64.0
    stitch_args.max_y_gap = 64.0
    stitch_args.min_merged_height = 1300.0

    hclasses = {0, 2, 3, 4, 7}

    main_classes = {0, 2, 3, 5}

    #
    # Only analyze the informative configs.
    #
    configs = [
        {
            "name": "tight_env",
            "classes": main_classes,
            "min_pair_iou": 0.25,
            "max_dxn": 0.60,
            "max_dyn": 0.60,
            "max_wr": 3.0,
            "max_hr": 3.0,
            "max_ar": 4.0,
            "methods": ["envelope"],
        },
        {
            "name": "tight_avg",
            "classes": main_classes,
            "min_pair_iou": 0.25,
            "max_dxn": 0.60,
            "max_dyn": 0.60,
            "max_wr": 3.0,
            "max_hr": 3.0,
            "max_ar": 4.0,
            "methods": ["avg50"],
        },
        {
            "name": "tight_both",
            "classes": main_classes,
            "min_pair_iou": 0.25,
            "max_dxn": 0.60,
            "max_dyn": 0.60,
            "max_wr": 3.0,
            "max_hr": 3.0,
            "max_ar": 4.0,
            "methods": ["envelope", "avg50"],
        },
        {
            "name": "center_both",
            "classes": main_classes,
            "min_pair_iou": 0.00,
            "max_dxn": 0.60,
            "max_dyn": 0.60,
            "max_wr": 3.0,
            "max_hr": 3.0,
            "max_ar": 4.0,
            "methods": ["envelope", "avg50"],
        },
        {
            "name": "loose_env",
            "classes": main_classes,
            "min_pair_iou": 0.00,
            "max_dxn": 1.00,
            "max_dyn": 1.00,
            "max_wr": 4.0,
            "max_hr": 4.0,
            "max_ar": 6.0,
            "methods": ["envelope"],
        },
        {
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
    ]

    stats = {
        c["name"]: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "added": 0,
            "rescued": [],
            "regressed": [],
        }
        for c in configs
    }

    base_totals = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }

    rescue_to_configs = defaultdict(list)
    details = []

    print("===== FUSION RESCUE ATTRIBUTION =====")
    print("Images     :", len(omap))
    print("Pair top-k :", args.pair_topk)
    print("Pair chunk :", args.pair_chunk)
    print()

    for img_idx, (name, item) in enumerate(
        omap.items(),
        1,
    ):
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

        hsel = hflip[
            np.isin(
                hflip[:, 0].astype(np.int32),
                list(hclasses),
            )
        ]

        union_pre = (
            np.concatenate(
                [original, hsel],
                axis=0,
            )
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

        gt_array = s19.read_yolo_gt(
            label_path,
            width,
            height,
        )

        gt_obj = diag.read_yolo_gt(
            label_path,
            width,
            height,
        )

        (
            btp,
            bfp,
            bfn,
            base_unmatched,
        ) = get_unmatched_indices(
            diag,
            s19,
            final,
            gt_obj,
            gt_array,
        )

        base_totals["tp"] += btp
        base_totals["fp"] += bfp
        base_totals["fn"] += bfn

        #
        # Build pair pool once.
        #
        pairs = []

        for cls in sorted(main_classes):
            pairs.extend(
                s20.build_pair_features(
                    original,
                    hflip,
                    cls,
                    args.pair_topk,
                    args.pair_chunk,
                )
            )

        for cfg in configs:
            fused = s20.make_fusion_rows(
                pairs,
                cfg,
                final.shape[1],
            )

            augmented = (
                np.concatenate(
                    [final, fused],
                    axis=0,
                )
                if len(fused)
                else final
            )

            (
                tp,
                fp,
                fn,
                aug_unmatched,
            ) = get_unmatched_indices(
                diag,
                s19,
                augmented,
                gt_obj,
                gt_array,
            )

            st = stats[cfg["name"]]

            st["tp"] += tp
            st["fp"] += fp
            st["fn"] += fn
            st["added"] += len(fused)

            rescued = sorted(
                base_unmatched - aug_unmatched
            )

            regressed = sorted(
                aug_unmatched - base_unmatched
            )

            for gi in rescued:
                gt = gt_array[gi]
                cls = int(gt[0])

                key = (
                    name,
                    cls,
                    int(gi),
                )

                rescue_to_configs[key].append(
                    cfg["name"]
                )

                best_o = s19.best_candidate(
                    original,
                    gt[1:5],
                    cls,
                )

                best_h = s19.best_candidate(
                    hflip,
                    gt[1:5],
                    cls,
                )

                rec = {
                    "config": cfg["name"],
                    "image_name": name,
                    "class_id": cls,
                    "class_name": CLASS_NAMES[cls],
                    "gt_index": gi,
                    "gt_width": float(gt[3] - gt[1]),
                    "gt_height": float(gt[4] - gt[2]),
                    "best_original_iou":
                        best_o["iou"]
                        if best_o else "",
                    "best_hflip_iou":
                        best_h["iou"]
                        if best_h else "",
                }

                st["rescued"].append(rec)
                details.append(rec)

            for gi in regressed:
                gt = gt_array[gi]
                cls = int(gt[0])

                st["regressed"].append({
                    "image_name": name,
                    "class_id": cls,
                    "class_name": CLASS_NAMES[cls],
                    "gt_index": gi,
                })

        if (
            img_idx % 50 == 0
            or img_idx == len(omap)
        ):
            print(
                f"{img_idx}/{len(omap)} "
                f"baseTP={base_totals['tp']} "
                f"baseFN={base_totals['fn']}"
            )

    print()
    print("===== BASELINE =====")
    print(
        f"TP={base_totals['tp']} "
        f"FP={base_totals['fp']} "
        f"FN={base_totals['fn']}"
    )

    if (
        base_totals["tp"] != 824
        or base_totals["fn"] != 21
    ):
        raise RuntimeError(
            f"Baseline mismatch: {base_totals}"
        )

    print()
    print("===== CONFIG ATTRIBUTION =====")

    for cfg in configs:
        name = cfg["name"]
        s = stats[name]

        print()
        print(
            f"[{name}] "
            f"TP={s['tp']} "
            f"FN={s['fn']} "
            f"dTP={s['tp'] - 824:+d} "
            f"dFP={s['fp'] - 1631208:+d} "
            f"added={s['added']:,}"
        )

        print(
            f"rescued={len(s['rescued'])}, "
            f"regressed={len(s['regressed'])}"
        )

        for r in s["rescued"]:
            print(
                "  RESCUE "
                f"{r['class_name']:16s} "
                f"{r['image_name']} "
                f"GT#{r['gt_index']} "
                f"O={float(r['best_original_iou']):.4f} "
                f"H={float(r['best_hflip_iou']):.4f}"
            )

        for r in s["regressed"]:
            print(
                "  REGRESS "
                f"{r['class_name']:16s} "
                f"{r['image_name']} "
                f"GT#{r['gt_index']}"
            )

    print()
    print("===== UNIQUE RESCUES ACROSS CONFIGS =====")

    keys = sorted(
        rescue_to_configs,
        key=lambda x: (
            x[1],
            x[0],
            x[2],
        ),
    )

    for key in keys:
        image_name, cls, gi = key

        print(
            f"{CLASS_NAMES[cls]:16s} "
            f"{image_name} "
            f"GT#{gi} "
            f"<- {', '.join(rescue_to_configs[key])}"
        )

    print()
    print(
        "Unique rescued GT count:",
        len(rescue_to_configs),
    )

    #
    # Save detailed CSV.
    #
    out_csv = (
        args.output_dir
        / "rescue_attribution.csv"
    )

    if details:
        with out_csv.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            w = csv.DictWriter(
                f,
                fieldnames=list(details[0].keys()),
            )
            w.writeheader()
            w.writerows(details)

    #
    # Save config summary.
    #
    summary_csv = (
        args.output_dir
        / "config_summary.csv"
    )

    with summary_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.writer(f)

        w.writerow([
            "config",
            "tp",
            "fp",
            "fn",
            "delta_tp",
            "delta_fp",
            "added",
            "rescued",
            "regressed",
        ])

        for cfg in configs:
            name = cfg["name"]
            s = stats[name]

            w.writerow([
                name,
                s["tp"],
                s["fp"],
                s["fn"],
                s["tp"] - 824,
                s["fp"] - 1631208,
                s["added"],
                len(s["rescued"]),
                len(s["regressed"]),
            ])

    print()
    print("Saved:", out_csv)
    print("Saved:", summary_csv)


if __name__ == "__main__":
    main()
