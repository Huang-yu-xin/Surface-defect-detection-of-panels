#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import sys
from collections import Counter, defaultdict
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
    if spec is None or spec.loader is None:
        raise RuntimeError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--original-cache",
        type=Path,
        default=Path("results/fn_analysis/cache"),
    )

    p.add_argument(
        "--hflip-cache",
        type=Path,
        default=Path("results/fn_analysis/cache_hflip"),
    )

    p.add_argument(
        "--labels",
        type=Path,
        default=Path("datasets/yolo_split/labels/val"),
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/class_confusion_dup_v1"),
    )

    p.add_argument(
        "--device",
        default="cpu",
    )

    p.add_argument(
        "--dup-score",
        type=float,
        default=1e-6,
    )

    return p.parse_args()


def iou_one_to_many(gt, boxes):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)

    gx1 = float(gt.xmin)
    gy1 = float(gt.ymin)
    gx2 = float(gt.xmax)
    gy2 = float(gt.ymax)

    x1 = np.maximum(gx1, boxes[:, 2])
    y1 = np.maximum(gy1, boxes[:, 3])
    x2 = np.minimum(gx2, boxes[:, 4])
    y2 = np.minimum(gy2, boxes[:, 5])

    iw = np.maximum(0.0, x2 - x1)
    ih = np.maximum(0.0, y2 - y1)

    inter = iw * ih

    ga = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)

    ba = (
        np.maximum(0.0, boxes[:, 4] - boxes[:, 2])
        *
        np.maximum(0.0, boxes[:, 5] - boxes[:, 3])
    )

    union = ga + ba - inter

    return np.divide(
        inter,
        union,
        out=np.zeros_like(inter, dtype=np.float32),
        where=union > 0,
    )


def duplicate_mappings(final, mappings, dup_score):
    """
    Duplicate selected source-class detections into target classes.

    Existing detections are preserved.
    Duplicates are appended with tiny scores so they cannot outrank
    the original Final Combo detections during score-ordered matching.
    """

    pieces = [final]

    added = 0
    by_mapping = Counter()

    cls = final[:, 0].astype(np.int32)

    for src, dst in mappings:
        src_rows = final[cls == src]

        if not len(src_rows):
            continue

        dup = src_rows.copy()
        dup[:, 0] = dst
        dup[:, 1] = dup_score

        pieces.append(dup)

        n = len(dup)
        added += n
        by_mapping[(src, dst)] += n

    if len(pieces) == 1:
        return final, 0, by_mapping

    return (
        np.concatenate(pieces, axis=0),
        added,
        by_mapping,
    )


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combo = load_module(
        Path("scripts/18_final_combo_from_cache.py"),
        "combo18_s26",
    )

    diag = load_module(
        Path("scripts/14_fn_diagnostic.py"),
        "diag14_s26",
    )

    device = torch.device(args.device)

    #
    # Load exact script18 defaults.
    #
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

    #
    # Exact validated stitch configuration.
    #
    stitch_args = SimpleNamespace(**vars(defaults))

    stitch_args.min_aspect = 5.0
    stitch_args.x_tol = 64.0
    stitch_args.max_y_gap = 64.0
    stitch_args.min_merged_height = 1300.0

    hclasses = {0, 2, 3, 4, 7}

    #
    # source predicted class -> duplicated target class
    #
    mappings = {
        "7to0": [(7, 0)],  # yanghuatiepi -> jieba
        "6to0": [(6, 0)],  # mamianmakeng -> jieba
        "0to3": [(0, 3)],  # jieba -> jiaza
        "5to4": [(5, 4)],  # huashang -> yiwuyaru
        "4to3": [(4, 3)],  # yiwuyaru -> jiaza

        "targeted5": [
            (7, 0),
            (6, 0),
            (0, 3),
            (5, 4),
            (4, 3),
        ],

        #
        # A deliberately broader diagnostic variant.
        # Not intended for submission.
        #
        "symmetric10": [
            (7, 0), (0, 7),
            (6, 0), (0, 6),
            (0, 3), (3, 0),
            (5, 4), (4, 5),
            (4, 3), (3, 4),
        ],
    }

    om = combo.load_manifest(args.original_cache)
    hm = combo.load_manifest(args.hflip_cache)
    hmap = combo.manifest_map(hm)

    if int(om["images_count"]) != int(hm["images_count"]):
        raise RuntimeError("Original/HFlip count mismatch")

    totals = {
        "baseline": Counter()
    }

    for name in mappings:
        totals[name] = Counter()

    added_counts = Counter()
    added_by_mapping = {
        name: Counter()
        for name in mappings
    }

    #
    # Whole-Val support audit:
    # For every GT, find the best WRONG-CLASS Final Combo box.
    #
    wrong_support = Counter()
    wrong_support_images = defaultdict(set)
    wrong_support_rows = []

    total_stitched = 0

    print("===== CLASS CONFUSION DUPLICATION SWEEP =====")
    print("Images   :", om["images_count"])
    print("Device   :", device)
    print("Dup score:", args.dup_score)
    print()

    for idx, oitem in enumerate(om["items"], start=1):
        name = oitem["image_name"]

        if name not in hmap:
            raise RuntimeError(
                f"HFlip cache missing {name}"
            )

        hitem = hmap[name]

        onpz = np.load(
            args.original_cache / oitem["cache_file"]
        )

        hnpz = np.load(
            args.hflip_cache / hitem["cache_file"]
        )

        orig = onpz["candidates"].astype(
            np.float32,
            copy=False,
        )

        hflip = hnpz["candidates"].astype(
            np.float32,
            copy=False,
        )

        height, width = map(
            int,
            onpz["image_shape"],
        )

        #
        # Exact Final Combo reconstruction.
        #
        hsel = hflip[
            np.isin(
                hflip[:, 0].astype(np.int32),
                list(hclasses),
            )
        ]

        union_pre = (
            np.concatenate(
                [orig, hsel],
                axis=0,
            )
            if len(hsel)
            else orig
        )

        post, _ = combo.nms(
            union_pre,
            defaults.global_iou,
            diag,
            device,
        )

        final, merged_count = combo.add_stitched(
            post,
            stitch_args,
            device,
        )

        total_stitched += merged_count

        label_path = (
            args.labels
            / f"{Path(name).stem}.txt"
        )

        gt = diag.read_yolo_gt(
            label_path,
            width,
            height,
        )

        #
        # Exact baseline.
        #
        tp, fp, fn, *_ = diag.match_predictions(
            final,
            gt,
            defaults.match_iou,
        )

        totals["baseline"]["tp"] += tp
        totals["baseline"]["fp"] += fp
        totals["baseline"]["fn"] += fn

        #
        # Whole-Val wrong-class overlap support.
        #
        final_cls = final[:, 0].astype(np.int32)

        for gi, g in enumerate(gt):
            wrong = final[
                final_cls != int(g.class_id)
            ]

            if not len(wrong):
                continue

            ious = iou_one_to_many(
                g,
                wrong,
            )

            bi = int(np.argmax(ious))
            best_iou = float(ious[bi])

            if best_iou < defaults.match_iou:
                continue

            src = int(wrong[bi, 0])
            dst = int(g.class_id)

            wrong_support[(src, dst)] += 1
            wrong_support_images[(src, dst)].add(name)

            wrong_support_rows.append({
                "image_name": name,
                "gt_index": gi,
                "target_class_id": dst,
                "target_class": CLASS_NAMES[dst],
                "wrong_source_class_id": src,
                "wrong_source_class": CLASS_NAMES[src],
                "wrong_iou": best_iou,
                "wrong_score": float(wrong[bi, 1]),
            })

        #
        # Duplication variants.
        #
        for vname, vmaps in mappings.items():
            aug, added, bymap = duplicate_mappings(
                final,
                vmaps,
                args.dup_score,
            )

            vtp, vfp, vfn, *_ = diag.match_predictions(
                aug,
                gt,
                defaults.match_iou,
            )

            totals[vname]["tp"] += vtp
            totals[vname]["fp"] += vfp
            totals[vname]["fn"] += vfn

            added_counts[vname] += added

            for k, v in bymap.items():
                added_by_mapping[vname][k] += v

        if (
            idx % 25 == 0
            or idx == int(om["images_count"])
        ):
            print(
                f"{idx}/{om['images_count']} "
                f"baseline_TP={totals['baseline']['tp']} "
                f"baseline_FN={totals['baseline']['fn']} "
                f"stitched={total_stitched}"
            )

    #
    # Exact baseline guard.
    #
    b = totals["baseline"]

    print()
    print("===== BASELINE CHECK =====")
    print(
        f"TP={b['tp']} FP={b['fp']} FN={b['fn']}"
    )

    if (
        b["tp"] != 824
        or b["fn"] != 21
        or b["fp"] != 1631208
    ):
        raise RuntimeError(
            "Final Combo baseline mismatch"
        )

    #
    # Whole-Val support.
    #
    print()
    print("===== WHOLE-VAL WRONG-CLASS SUPPORT =====")

    support_rows = []

    for (src, dst), n in sorted(
        wrong_support.items(),
        key=lambda x: (-x[1], x[0]),
    ):
        ni = len(
            wrong_support_images[(src, dst)]
        )

        mark = ""

        if (src, dst) in mappings["targeted5"]:
            mark = "  <TARGETED>"

        print(
            f"{CLASS_NAMES[src]:16s}"
            f" -> "
            f"{CLASS_NAMES[dst]:16s}"
            f" GT={n:3d}"
            f" images={ni:3d}"
            f"{mark}"
        )

        support_rows.append({
            "source_class_id": src,
            "source_class": CLASS_NAMES[src],
            "target_class_id": dst,
            "target_class": CLASS_NAMES[dst],
            "gt_support": n,
            "image_support": ni,
            "targeted5": int(
                (src, dst)
                in mappings["targeted5"]
            ),
        })

    #
    # Sweep results.
    #
    result_rows = []

    print()
    print("===== DUPLICATION RESULTS =====")

    ordered = [
        "7to0",
        "6to0",
        "0to3",
        "5to4",
        "4to3",
        "targeted5",
        "symmetric10",
    ]

    for name in ordered:
        r = totals[name]

        dtp = r["tp"] - b["tp"]
        dfp = r["fp"] - b["fp"]
        dfn = r["fn"] - b["fn"]

        denom = r["tp"] + r["fn"]
        recall = (
            r["tp"] / denom
            if denom
            else 0.0
        )

        print(
            f"{name:14s} "
            f"TP={r['tp']} "
            f"FN={r['fn']} "
            f"Recall={recall:.6f} "
            f"ScoreLike={recall * 100:.2f} "
            f"dTP={dtp:+d} "
            f"dFN={dfn:+d} "
            f"dFP={dfp:+d} "
            f"added={added_counts[name]:,}"
        )

        result_rows.append({
            "variant": name,
            "tp": r["tp"],
            "fp": r["fp"],
            "fn": r["fn"],
            "recall": recall,
            "score_like": recall * 100,
            "delta_tp": dtp,
            "delta_fp": dfp,
            "delta_fn": dfn,
            "added_boxes": added_counts[name],
        })

    #
    # Save.
    #
    with (
        args.output_dir
        / "duplication_summary.csv"
    ).open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(
                result_rows[0].keys()
            ),
        )
        w.writeheader()
        w.writerows(result_rows)

    if support_rows:
        with (
            args.output_dir
            / "wrong_class_support_summary.csv"
        ).open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            w = csv.DictWriter(
                f,
                fieldnames=list(
                    support_rows[0].keys()
                ),
            )
            w.writeheader()
            w.writerows(support_rows)

    if wrong_support_rows:
        with (
            args.output_dir
            / "wrong_class_support_detail.csv"
        ).open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            w = csv.DictWriter(
                f,
                fieldnames=list(
                    wrong_support_rows[0].keys()
                ),
            )
            w.writeheader()
            w.writerows(wrong_support_rows)

    print()
    print(
        "Saved:",
        args.output_dir
        / "duplication_summary.csv",
    )


if __name__ == "__main__":
    main()
