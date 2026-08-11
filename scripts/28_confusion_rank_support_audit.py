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


MAPPINGS = [
    (7, 0),  # yanghuatiepi -> jieba
    (6, 0),  # mamianmakeng -> jieba
    (0, 3),  # jieba -> jiaza
    (5, 4),  # huashang -> yiwuyaru
    (4, 3),  # yiwuyaru -> jiaza
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
        default=Path(
            "results/final_combo_fn21/remaining_fn_21.csv"
        ),
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


def rank_of_index(rows, idx):
    """
    1-based score rank within one source class.
    """

    scores = rows[:, 1]

    order = np.argsort(
        -scores,
        kind="stable",
    )

    where = np.where(order == idx)[0]

    if len(where) == 0:
        return None

    return int(where[0]) + 1


def duplicate_topk(
    final,
    rules,
    dup_score,
):
    """
    rules:
        [(src, dst, topk), ...]
    """

    cls = final[:, 0].astype(np.int32)
    pieces = []

    for src, dst, topk in rules:
        rows = final[cls == src]

        if not len(rows):
            continue

        order = np.argsort(
            -rows[:, 1],
            kind="stable",
        )

        rows = rows[order[:topk]]

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

    return np.concatenate(
        pieces,
        axis=0,
    )


def percentile(vals, q):
    if not vals:
        return None

    return float(
        np.percentile(
            np.asarray(vals, dtype=np.float64),
            q,
        )
    )


def main():
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    combo = load_module(
        Path("scripts/18_final_combo_from_cache.py"),
        "combo18_s28",
    )

    diag = load_module(
        Path("scripts/14_fn_diagnostic.py"),
        "diag14_s28",
    )

    device = torch.device(args.device)

    #
    # Exact Final Combo defaults.
    #
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

    #
    # Remaining-FN identity.
    #
    remaining_rows = list(
        csv.DictReader(
            args.remaining_fn.open(
                encoding="utf-8-sig",
            )
        )
    )

    remaining_keys = set()

    for r in remaining_rows:
        remaining_keys.add(
            (
                r["image_name"],
                int(r["gt_index"]),
            )
        )

    #
    # Per mapping support information.
    #
    support_ranks = defaultdict(list)
    support_scores = defaultdict(list)
    support_ious = defaultdict(list)
    support_images = defaultdict(set)

    remaining_support = defaultdict(list)

    detail_rows = []

    #
    # Mixed candidates.
    #
    candidates = {
        "7to0_top20": [
            (7, 0, 20),
        ],

        "6to0_top5": [
            (6, 0, 5),
        ],

        "robust_20_5": [
            (7, 0, 20),
            (6, 0, 5),
        ],

        "robust_30_10": [
            (7, 0, 30),
            (6, 0, 10),
        ],

        "robust_50_20": [
            (7, 0, 50),
            (6, 0, 20),
        ],
    }

    totals = {
        "baseline": Counter()
    }

    added = Counter()

    for name in candidates:
        totals[name] = Counter()

    print(
        "===== CONFUSION RANK SUPPORT AUDIT ====="
    )
    print(
        "Images:",
        om["images_count"],
    )
    print()

    for idx, oitem in enumerate(
        om["items"],
        start=1,
    ):
        image_name = oitem["image_name"]

        hitem = hmap[image_name]

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

        label_path = (
            args.labels
            / f"{Path(image_name).stem}.txt"
        )

        gt = diag.read_yolo_gt(
            label_path,
            width,
            height,
        )

        #
        # Baseline exact.
        #
        tp, fp, fn, *_ = (
            diag.match_predictions(
                final,
                gt,
                defaults.match_iou,
            )
        )

        totals["baseline"]["tp"] += tp
        totals["baseline"]["fp"] += fp
        totals["baseline"]["fn"] += fn

        final_cls = (
            final[:, 0]
            .astype(np.int32)
        )

        #
        # Whole-Val rank support.
        #
        for gi, g in enumerate(gt):

            for src, dst in MAPPINGS:

                if int(g.class_id) != dst:
                    continue

                source_rows = final[
                    final_cls == src
                ]

                if not len(source_rows):
                    continue

                ious = iou_one_to_many(
                    g,
                    source_rows,
                )

                bi = int(
                    np.argmax(ious)
                )

                best_iou = float(
                    ious[bi]
                )

                if best_iou < 0.50:
                    continue

                rank = rank_of_index(
                    source_rows,
                    bi,
                )

                score = float(
                    source_rows[bi, 1]
                )

                key = (src, dst)

                support_ranks[key].append(
                    rank
                )

                support_scores[key].append(
                    score
                )

                support_ious[key].append(
                    best_iou
                )

                support_images[key].add(
                    image_name
                )

                is_remaining = (
                    image_name,
                    gi,
                ) in remaining_keys

                if is_remaining:
                    remaining_support[key].append({
                        "image_name":
                            image_name,
                        "gt_index":
                            gi,
                        "rank":
                            rank,
                        "score":
                            score,
                        "iou":
                            best_iou,
                    })

                detail_rows.append({
                    "image_name":
                        image_name,

                    "gt_index":
                        gi,

                    "source_class_id":
                        src,

                    "source_class":
                        CLASS_NAMES[src],

                    "target_class_id":
                        dst,

                    "target_class":
                        CLASS_NAMES[dst],

                    "source_rank":
                        rank,

                    "source_score":
                        score,

                    "source_iou":
                        best_iou,

                    "is_remaining_fn":
                        int(is_remaining),
                })

        #
        # Mixed duplication candidates.
        #
        for vname, rules in (
            candidates.items()
        ):
            dup = duplicate_topk(
                final,
                rules,
                args.dup_score,
            )

            if len(dup):
                aug = np.concatenate(
                    [final, dup],
                    axis=0,
                )
            else:
                aug = final

            vtp, vfp, vfn, *_ = (
                diag.match_predictions(
                    aug,
                    gt,
                    defaults.match_iou,
                )
            )

            totals[vname]["tp"] += vtp
            totals[vname]["fp"] += vfp
            totals[vname]["fn"] += vfn

            added[vname] += len(dup)

        if (
            idx % 25 == 0
            or idx == int(
                om["images_count"]
            )
        ):
            print(
                f"{idx}/{om['images_count']}"
            )

    b = totals["baseline"]

    print()
    print("===== BASELINE CHECK =====")
    print(
        f"TP={b['tp']} "
        f"FP={b['fp']} "
        f"FN={b['fn']}"
    )

    if (
        b["tp"] != 824
        or b["fp"] != 1631208
        or b["fn"] != 21
    ):
        raise RuntimeError(
            "Final Combo baseline mismatch"
        )

    #
    # Rank distributions.
    #
    print()
    print(
        "===== WHOLE-VAL SOURCE RANK DISTRIBUTION ====="
    )

    summary_rows = []

    for src, dst in MAPPINGS:
        key = (src, dst)

        ranks = support_ranks[key]

        if not ranks:
            print()
            print(
                f"{CLASS_NAMES[src]} "
                f"-> {CLASS_NAMES[dst]}: "
                f"NO SUPPORT"
            )
            continue

        arr = np.asarray(
            ranks,
            dtype=np.int32,
        )

        n = len(arr)
        ni = len(
            support_images[key]
        )

        counts = {
            k: int(
                np.sum(arr <= k)
            )
            for k in [
                1, 3, 5, 10,
                20, 50, 100,
            ]
        }

        print()
        print(
            f"[{CLASS_NAMES[src]} "
            f"-> {CLASS_NAMES[dst]}]"
        )

        print(
            f"support GT     : {n}"
        )

        print(
            f"support images : {ni}"
        )

        print(
            "rank "
            f"min/median/p75/p90/max : "
            f"{int(arr.min())} / "
            f"{percentile(ranks,50):.1f} / "
            f"{percentile(ranks,75):.1f} / "
            f"{percentile(ranks,90):.1f} / "
            f"{int(arr.max())}"
        )

        print(
            "within rank:"
        )

        for k in [
            1, 3, 5, 10,
            20, 50, 100,
        ]:
            pct = (
                counts[k] / n * 100
            )

            print(
                f"  Top{k:<3d}: "
                f"{counts[k]:3d}/{n:<3d} "
                f"({pct:5.1f}%)"
            )

        rem = remaining_support[key]

        print(
            "remaining-FN supports:",
            len(rem),
        )

        for r in rem:
            print(
                f"  {r['image_name']} "
                f"GT#{r['gt_index']} "
                f"rank={r['rank']} "
                f"IoU={r['iou']:.4f} "
                f"score={r['score']:.8f}"
            )

        summary_rows.append({
            "source_class":
                CLASS_NAMES[src],

            "target_class":
                CLASS_NAMES[dst],

            "support_gt":
                n,

            "support_images":
                ni,

            "rank_min":
                int(arr.min()),

            "rank_median":
                percentile(ranks, 50),

            "rank_p75":
                percentile(ranks, 75),

            "rank_p90":
                percentile(ranks, 90),

            "rank_max":
                int(arr.max()),

            "top5_support":
                counts[5],

            "top10_support":
                counts[10],

            "top20_support":
                counts[20],

            "top50_support":
                counts[50],

            "top100_support":
                counts[100],

            "remaining_fn_support":
                len(rem),
        })

    #
    # Mixed variants.
    #
    print()
    print(
        "===== ROBUST MIXED CANDIDATES ====="
    )

    mix_rows = []

    for vname in candidates:
        r = totals[vname]

        dtp = (
            r["tp"] - b["tp"]
        )

        dfn = (
            r["fn"] - b["fn"]
        )

        dfp = (
            r["fp"] - b["fp"]
        )

        denom = (
            r["tp"] + r["fn"]
        )

        recall = (
            r["tp"] / denom
        )

        print(
            f"{vname:18s} "
            f"TP={r['tp']} "
            f"FN={r['fn']} "
            f"ScoreLike={recall*100:.2f} "
            f"dTP={dtp:+d} "
            f"dFN={dfn:+d} "
            f"dFP={dfp:+d} "
            f"added={added[vname]:,}"
        )

        mix_rows.append({
            "variant": vname,
            "tp": r["tp"],
            "fp": r["fp"],
            "fn": r["fn"],
            "recall": recall,
            "score_like":
                recall * 100,
            "delta_tp": dtp,
            "delta_fp": dfp,
            "delta_fn": dfn,
            "added_boxes":
                added[vname],
        })

    #
    # Save.
    #
    with (
        args.output_dir
        / "rank_support_summary.csv"
    ).open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )
        w.writeheader()
        w.writerows(
            summary_rows
        )

    with (
        args.output_dir
        / "rank_support_detail.csv"
    ).open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(
                detail_rows[0].keys()
            ),
        )
        w.writeheader()
        w.writerows(
            detail_rows
        )

    with (
        args.output_dir
        / "mixed_candidate_summary.csv"
    ).open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(
                mix_rows[0].keys()
            ),
        )
        w.writeheader()
        w.writerows(
            mix_rows
        )

    print()
    print(
        "Saved:",
        args.output_dir,
    )


if __name__ == "__main__":
    main()
