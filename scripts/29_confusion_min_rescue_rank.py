#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import sys
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
        required=True,
    )

    p.add_argument(
        "--hflip-cache",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--labels",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--remaining-fn",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
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

    ga = (
        max(0.0, gx2 - gx1)
        * max(0.0, gy2 - gy1)
    )

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


def make_topk_dup(final, src, dst, topk, dup_score):
    cls = final[:, 0].astype(np.int32)
    rows = final[cls == src]

    if not len(rows):
        return np.empty(
            (0, final.shape[1]),
            dtype=np.float32,
        )

    order = np.argsort(
        -rows[:, 1],
        kind="stable",
    )

    rows = rows[order[:topk]]

    dup = rows.copy()
    dup[:, 0] = dst
    dup[:, 1] = dup_score

    return dup


def main():
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    combo = load_module(
        Path("scripts/18_final_combo_from_cache.py"),
        "combo18_s29",
    )

    diag = load_module(
        Path("scripts/14_fn_diagnostic.py"),
        "diag14_s29",
    )

    device = torch.device(args.device)

    old_argv = sys.argv[:]

    try:
        sys.argv = [
            "18_final_combo_from_cache.py",
            "--mode", "val",
            "--original-cache",
            str(args.original_cache),
            "--hflip-cache",
            str(args.hflip_cache),
            "--labels",
            str(args.labels),
            "--output-dir",
            str(args.output_dir / "_dummy"),
            "--device",
            args.device,
        ]

        defaults = combo.parse_args()

    finally:
        sys.argv = old_argv

    stitch_args = SimpleNamespace(
        **vars(defaults)
    )

    stitch_args.min_aspect = 5.0
    stitch_args.x_tol = 64.0
    stitch_args.max_y_gap = 64.0
    stitch_args.min_merged_height = 1300.0

    hclasses = {0, 2, 3, 4, 7}

    om = combo.load_manifest(
        args.original_cache
    )

    hm = combo.load_manifest(
        args.hflip_cache
    )

    hmap = combo.manifest_map(hm)

    remaining = list(
        csv.DictReader(
            args.remaining_fn.open(
                encoding="utf-8-sig",
            )
        )
    )

    remaining_map = {
        (
            r["image_name"],
            int(r["gt_index"]),
        ): r
        for r in remaining
    }

    #
    # We only care about the two robust mappings.
    #
    mapping_cfg = {
        "7to0": (7, 0, range(1, 31)),
        "6to0": (6, 0, range(1, 21)),
    }

    #
    # Store exact Final Combo per image.
    #
    finals = {}
    gts = {}

    baseline_tp = 0
    baseline_fp = 0
    baseline_fn = 0

    print("===== BUILD EXACT FINAL COMBO =====")

    for idx, oitem in enumerate(
        om["items"],
        start=1,
    ):
        name = oitem["image_name"]

        hitem = hmap[name]

        onpz = np.load(
            args.original_cache
            / oitem["cache_file"]
        )

        hnpz = np.load(
            args.hflip_cache
            / hitem["cache_file"]
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

        final, _ = combo.add_stitched(
            post,
            stitch_args,
            device,
        )

        gt = diag.read_yolo_gt(
            args.labels
            / f"{Path(name).stem}.txt",
            width,
            height,
        )

        tp, fp, fn, *_ = (
            diag.match_predictions(
                final,
                gt,
                defaults.match_iou,
            )
        )

        baseline_tp += tp
        baseline_fp += fp
        baseline_fn += fn

        finals[name] = final
        gts[name] = gt

        if (
            idx % 50 == 0
            or idx == int(om["images_count"])
        ):
            print(
                f"{idx}/{om['images_count']}"
            )

    print()
    print("===== BASELINE CHECK =====")
    print(
        f"TP={baseline_tp} "
        f"FP={baseline_fp} "
        f"FN={baseline_fn}"
    )

    if (
        baseline_tp != 824
        or baseline_fp != 1631208
        or baseline_fn != 21
    ):
        raise RuntimeError(
            "Baseline mismatch"
        )

    #
    # First: inspect all IoU >= .5 source candidates
    # for the relevant remaining FN.
    #
    print()
    print("===== REMAINING FN SOURCE CANDIDATES =====")

    for key_name, (
        src,
        dst,
        _,
    ) in mapping_cfg.items():

        print()
        print(
            f"[{CLASS_NAMES[src]} "
            f"-> {CLASS_NAMES[dst]}]"
        )

        for (
            image_name,
            gt_index
        ), meta in remaining_map.items():

            if meta["class_name"] != CLASS_NAMES[dst]:
                continue

            final = finals[image_name]
            gt = gts[image_name][gt_index]

            src_rows = final[
                final[:, 0].astype(np.int32)
                == src
            ]

            if not len(src_rows):
                continue

            order = np.argsort(
                -src_rows[:, 1],
                kind="stable",
            )

            ranked = src_rows[order]

            ious = iou_one_to_many(
                gt,
                ranked,
            )

            good = np.where(
                ious >= 0.50
            )[0]

            if not len(good):
                continue

            print(
                f"{image_name} "
                f"GT#{gt_index} "
                f"failure={meta['failure_type']}"
            )

            print(
                "  qualifying ranks:",
                [
                    int(i + 1)
                    for i in good[:20]
                ],
            )

            print(
                "  first qualifying:",
                f"rank={int(good[0])+1}",
                f"IoU={float(ious[good[0]]):.4f}",
                f"score={float(ranked[good[0],1]):.8f}",
            )

            bi = int(
                np.argmax(ious)
            )

            print(
                "  best IoU:",
                f"rank={bi+1}",
                f"IoU={float(ious[bi]):.4f}",
                f"score={float(ranked[bi,1]):.8f}",
            )

    #
    # Exact K sweep.
    #
    print()
    print("===== EXACT MINIMUM-RANK SWEEP =====")

    sweep_rows = []

    for key_name, (
        src,
        dst,
        krange,
    ) in mapping_cfg.items():

        print()
        print(f"[{key_name}]")

        first_gain = None

        for k in krange:
            tp_sum = 0
            fp_sum = 0
            fn_sum = 0
            added = 0

            for name in finals:
                final = finals[name]
                gt = gts[name]

                dup = make_topk_dup(
                    final,
                    src,
                    dst,
                    k,
                    args.dup_score,
                )

                if len(dup):
                    aug = np.concatenate(
                        [final, dup],
                        axis=0,
                    )
                else:
                    aug = final

                tp, fp, fn, *_ = (
                    diag.match_predictions(
                        aug,
                        gt,
                        defaults.match_iou,
                    )
                )

                tp_sum += tp
                fp_sum += fp
                fn_sum += fn
                added += len(dup)

            dtp = tp_sum - baseline_tp

            if dtp > 0 and first_gain is None:
                first_gain = k

            print(
                f"Top{k:<2d} "
                f"TP={tp_sum} "
                f"FN={fn_sum} "
                f"dTP={dtp:+d} "
                f"added={added:,}"
            )

            sweep_rows.append({
                "mapping": key_name,
                "topk": k,
                "tp": tp_sum,
                "fp": fp_sum,
                "fn": fn_sum,
                "delta_tp": dtp,
                "added_boxes": added,
            })

        print(
            "FIRST GAIN TOP-K:",
            first_gain,
        )

    #
    # Exact robust combo attribution.
    #
    print()
    print("===== ROBUST_20_5 RESCUE ATTRIBUTION =====")

    rescued = []

    for name in finals:
        final = finals[name]
        gt = gts[name]

        btp, bfp, bfn, bunmatched, *_ = (
            diag.match_predictions(
                final,
                gt,
                defaults.match_iou,
            )
        )

        d1 = make_topk_dup(
            final,
            7,
            0,
            20,
            args.dup_score,
        )

        d2 = make_topk_dup(
            final,
            6,
            0,
            5,
            args.dup_score,
        )

        pieces = [final]

        if len(d1):
            pieces.append(d1)

        if len(d2):
            pieces.append(d2)

        aug = np.concatenate(
            pieces,
            axis=0,
        )

        atp, afp, afn, aunmatched, *_ = (
            diag.match_predictions(
                aug,
                gt,
                defaults.match_iou,
            )
        )

        if atp <= btp:
            continue

        before = set(bunmatched)
        after = set(aunmatched)

        newly = sorted(
            before - after
        )

        for gi in newly:
            meta = remaining_map.get(
                (name, gi),
                {},
            )

            row = {
                "image_name": name,
                "gt_index": gi,
                "class_name":
                    CLASS_NAMES[
                        int(gt[gi].class_id)
                    ],
                "failure_type":
                    meta.get(
                        "failure_type",
                        "UNKNOWN",
                    ),
            }

            rescued.append(row)

            print(
                f"{name} "
                f"GT#{gi} "
                f"{row['class_name']} "
                f"{row['failure_type']}"
            )

    with (
        args.output_dir
        / "min_rank_sweep.csv"
    ).open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(
                sweep_rows[0].keys()
            ),
        )
        w.writeheader()
        w.writerows(sweep_rows)

    if rescued:
        with (
            args.output_dir
            / "robust_20_5_rescued.csv"
        ).open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            w = csv.DictWriter(
                f,
                fieldnames=list(
                    rescued[0].keys()
                ),
            )
            w.writeheader()
            w.writerows(rescued)

    print()
    print(
        "Saved:",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
