#!/usr/bin/env python3

import argparse
import csv
import importlib.util
import json
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

CLASS_TO_ID = {
    name: i
    for i, name in enumerate(CLASS_NAMES)
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(
        name,
        path,
    )

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
        "--baseline-json",
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

    return p.parse_args()


def iou_one_to_many(
    box,
    boxes,
):
    if len(boxes) == 0:
        return np.empty(
            (0,),
            dtype=np.float32,
        )

    bx1, by1, bx2, by2 = map(
        float,
        box,
    )

    x1 = np.maximum(
        bx1,
        boxes[:, 2],
    )

    y1 = np.maximum(
        by1,
        boxes[:, 3],
    )

    x2 = np.minimum(
        bx2,
        boxes[:, 4],
    )

    y2 = np.minimum(
        by2,
        boxes[:, 5],
    )

    iw = np.maximum(
        0.0,
        x2 - x1,
    )

    ih = np.maximum(
        0.0,
        y2 - y1,
    )

    inter = iw * ih

    ba = (
        max(0.0, bx2 - bx1)
        *
        max(0.0, by2 - by1)
    )

    aa = (
        np.maximum(
            0.0,
            boxes[:, 4] - boxes[:, 2],
        )
        *
        np.maximum(
            0.0,
            boxes[:, 5] - boxes[:, 3],
        )
    )

    union = ba + aa - inter

    return np.divide(
        inter,
        union,
        out=np.zeros_like(
            inter,
            dtype=np.float32,
        ),
        where=union > 0,
    )


def percentile(vals, q):
    if not vals:
        return 0.0

    return float(
        np.percentile(
            np.asarray(
                vals,
                dtype=np.float64,
            ),
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
        Path(
            "scripts/18_final_combo_from_cache.py"
        ),
        "combo18_s33",
    )

    diag = load_module(
        Path(
            "scripts/14_fn_diagnostic.py"
        ),
        "diag14_s33",
    )

    device = torch.device(
        args.device
    )

    #
    # Exact Final Combo defaults.
    #
    old_argv = sys.argv[:]

    try:
        sys.argv = [
            "18_final_combo_from_cache.py",
            "--mode", "test",
            "--original-cache",
            str(args.original_cache),
            "--hflip-cache",
            str(args.hflip_cache),
            "--output-dir",
            str(
                args.output_dir
                / "_dummy"
            ),
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

    hclasses = {
        0, 2, 3, 4, 7
    }

    #
    # Baseline submission.
    #
    baseline = json.loads(
        args.baseline_json.read_text(
            encoding="utf-8",
        )
    )

    baseline_by_image = defaultdict(
        list
    )

    for i, row in enumerate(baseline):
        cname = row["category_name"]

        if cname not in CLASS_TO_ID:
            raise RuntimeError(
                f"Unknown class: {cname}"
            )

        bbox = row["bbox"]

        if len(bbox) != 4:
            raise RuntimeError(
                f"Bad bbox: {bbox}"
            )

        baseline_by_image[
            row["image_id"]
        ].append({
            "baseline_index": i,
            "class_id":
                CLASS_TO_ID[cname],
            "class_name":
                cname,
            "score":
                float(row["score"]),
            "bbox":
                list(map(float, bbox)),
        })

    #
    # Final Combo caches.
    #
    om = combo.load_manifest(
        args.original_cache
    )

    hm = combo.load_manifest(
        args.hflip_cache
    )

    hmap = combo.manifest_map(hm)

    if int(om["images_count"]) != int(
        hm["images_count"]
    ):
        raise RuntimeError(
            "Original/HFlip count mismatch"
        )

    detail_rows = []

    base_detection_count = 0
    stitched_count = 0

    print(
        "===== BASELINE / RAREOS TEST COMPLEMENTARITY ====="
    )

    print(
        "Images        :",
        om["images_count"],
    )

    print(
        "Baseline boxes:",
        f"{len(baseline):,}",
    )

    print()

    for idx, oitem in enumerate(
        om["items"],
        start=1,
    ):
        name = oitem[
            "image_name"
        ]

        if name not in hmap:
            raise RuntimeError(
                f"HFlip missing {name}"
            )

        hitem = hmap[name]

        onpz = np.load(
            args.original_cache
            / oitem["cache_file"]
        )

        hnpz = np.load(
            args.hflip_cache
            / hitem["cache_file"]
        )

        orig = onpz[
            "candidates"
        ].astype(
            np.float32,
            copy=False,
        )

        hflip = hnpz[
            "candidates"
        ].astype(
            np.float32,
            copy=False,
        )

        #
        # Exact Final Combo.
        #
        hsel = hflip[
            np.isin(
                hflip[
                    :, 0
                ].astype(
                    np.int32
                ),
                list(
                    hclasses
                ),
            )
        ]

        union_pre = (
            np.concatenate(
                [
                    orig,
                    hsel,
                ],
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

        final, merged_count = (
            combo.add_stitched(
                post,
                stitch_args,
                device,
            )
        )

        base_detection_count += len(
            final
        )

        stitched_count += (
            merged_count
        )

        final_cls = final[
            :, 0
        ].astype(
            np.int32
        )

        #
        # Audit Baseline boxes in this image.
        #
        for brow in baseline_by_image.get(
            name,
            [],
        ):
            cid = brow[
                "class_id"
            ]

            bbox = brow[
                "bbox"
            ]

            same = final[
                final_cls == cid
            ]

            #
            # Same-class maximum IoU.
            #
            if len(same):
                sious = (
                    iou_one_to_many(
                        bbox,
                        same,
                    )
                )

                si = int(
                    np.argmax(
                        sious
                    )
                )

                same_iou = float(
                    sious[si]
                )

                same_score = float(
                    same[
                        si, 1
                    ]
                )

            else:
                same_iou = 0.0
                same_score = 0.0

            #
            # Any-class maximum IoU.
            #
            if len(final):
                aious = (
                    iou_one_to_many(
                        bbox,
                        final,
                    )
                )

                ai = int(
                    np.argmax(
                        aious
                    )
                )

                any_iou = float(
                    aious[ai]
                )

                any_cid = int(
                    final[
                        ai, 0
                    ]
                )

                any_class = (
                    CLASS_NAMES[
                        any_cid
                    ]
                )

                any_score = float(
                    final[
                        ai, 1
                    ]
                )

            else:
                any_iou = 0.0
                any_class = ""
                any_score = 0.0

            #
            # Complementarity bucket.
            #
            if same_iou >= 0.50:
                bucket = (
                    "same_class_covered"
                )

            elif any_iou >= 0.50:
                bucket = (
                    "class_disagreement"
                )

            else:
                bucket = (
                    "independent_geometry"
                )

            detail_rows.append({
                "baseline_index":
                    brow[
                        "baseline_index"
                    ],

                "image_id":
                    name,

                "class_id":
                    cid,

                "class_name":
                    brow[
                        "class_name"
                    ],

                "baseline_score":
                    brow[
                        "score"
                    ],

                "xmin":
                    bbox[0],

                "ymin":
                    bbox[1],

                "xmax":
                    bbox[2],

                "ymax":
                    bbox[3],

                "max_same_class_iou":
                    same_iou,

                "nearest_same_class_score":
                    same_score,

                "max_any_class_iou":
                    any_iou,

                "nearest_any_class":
                    any_class,

                "nearest_any_score":
                    any_score,

                "bucket":
                    bucket,
            })

        if (
            idx % 25 == 0
            or idx == int(
                om[
                    "images_count"
                ]
            )
        ):
            print(
                f"{idx}/"
                f"{om['images_count']} "
                f"final="
                f"{base_detection_count:,} "
                f"audited="
                f"{len(detail_rows):,}"
            )

    #
    # Exact Final Combo guard.
    #
    print()
    print(
        "===== FINAL COMBO CHECK ====="
    )

    print(
        "Detections:",
        f"{base_detection_count:,}",
    )

    print(
        "Stitched  :",
        f"{stitched_count:,}",
    )

    if base_detection_count != 2403809:
        raise RuntimeError(
            "Final Combo Test count mismatch"
        )

    if len(detail_rows) != len(
        baseline
    ):
        missing = (
            len(baseline)
            - len(detail_rows)
        )

        raise RuntimeError(
            f"Baseline rows missing: {missing}"
        )

    #
    # Global distribution.
    #
    same_ious = [
        r[
            "max_same_class_iou"
        ]
        for r in detail_rows
    ]

    scores = [
        r[
            "baseline_score"
        ]
        for r in detail_rows
    ]

    print()
    print(
        "===== SAME-CLASS COVERAGE ====="
    )

    for th in [
        0.0,
        0.1,
        0.3,
        0.5,
        0.7,
        0.9,
    ]:
        if th == 0.0:
            n = sum(
                x == 0.0
                for x in same_ious
            )

            print(
                f"IoU == 0.0 : "
                f"{n:,}"
            )

        else:
            n = sum(
                x < th
                for x in same_ious
            )

            print(
                f"IoU < {th:.1f}: "
                f"{n:,} "
                f"({n/len(detail_rows)*100:.1f}%)"
            )

    print()
    print(
        "same IoU percentiles:"
    )

    for q in [
        0,
        10,
        25,
        50,
        75,
        90,
        95,
        99,
        100,
    ]:
        print(
            f"  p{q:<3d}: "
            f"{percentile(same_ious,q):.4f}"
        )

    #
    # Buckets.
    #
    bucket_counts = Counter(
        r["bucket"]
        for r in detail_rows
    )

    print()
    print(
        "===== COMPLEMENTARITY BUCKETS ====="
    )

    for k in [
        "same_class_covered",
        "class_disagreement",
        "independent_geometry",
    ]:
        n = bucket_counts[k]

        print(
            f"{k:24s} "
            f"{n:,} "
            f"({n/len(detail_rows)*100:.1f}%)"
        )

    #
    # By class.
    #
    print()
    print(
        "===== UNIQUE (<0.5 SAME-CLASS IOU) BY CLASS ====="
    )

    by_class = Counter()

    for r in detail_rows:
        if (
            r[
                "max_same_class_iou"
            ] < 0.50
        ):
            by_class[
                r["class_name"]
            ] += 1

    for cname in CLASS_NAMES:
        print(
            f"{cname:18s} "
            f"{by_class[cname]:,}"
        )

    #
    # Score-filtered complementarity.
    #
    print()
    print(
        "===== SCORE-FILTERED UNIQUE CANDIDATES ====="
    )

    score_rows = []

    for floor in [
        0.01,
        0.02,
        0.05,
        0.10,
        0.20,
        0.50,
    ]:
        eligible = [
            r
            for r in detail_rows
            if r[
                "baseline_score"
            ] >= floor
        ]

        unique05 = [
            r
            for r in eligible
            if r[
                "max_same_class_iou"
            ] < 0.50
        ]

        unique03 = [
            r
            for r in eligible
            if r[
                "max_same_class_iou"
            ] < 0.30
        ]

        disagree = [
            r
            for r in eligible
            if (
                r[
                    "max_same_class_iou"
                ] < 0.50
                and
                r[
                    "max_any_class_iou"
                ] >= 0.50
            )
        ]

        independent = [
            r
            for r in eligible
            if r[
                "max_any_class_iou"
            ] < 0.50
        ]

        print(
            f"score>={floor:<4.2f} "
            f"all={len(eligible):4d} "
            f"unique<.5={len(unique05):4d} "
            f"unique<.3={len(unique03):4d} "
            f"class_disagree={len(disagree):4d} "
            f"independent={len(independent):4d}"
        )

        score_rows.append({
            "score_floor":
                floor,

            "baseline_boxes":
                len(eligible),

            "same_iou_lt_05":
                len(unique05),

            "same_iou_lt_03":
                len(unique03),

            "class_disagreement":
                len(disagree),

            "independent_geometry":
                len(independent),
        })

    #
    # Cross-model disagreement pairs.
    #
    print()
    print(
        "===== CLASS DISAGREEMENT PAIRS ====="
    )

    pairs = Counter()

    for r in detail_rows:
        if r[
            "bucket"
        ] != "class_disagreement":
            continue

        pairs[
            (
                r[
                    "class_name"
                ],
                r[
                    "nearest_any_class"
                ],
            )
        ] += 1

    for (
        baseline_cls,
        rareos_cls
    ), n in pairs.most_common(
        30
    ):
        print(
            f"Baseline "
            f"{baseline_cls:16s} "
            f"vs RareOS "
            f"{rareos_cls:16s} "
            f": {n}"
        )

    #
    # Save detail.
    #
    detail_path = (
        args.output_dir
        / "baseline_complementarity_detail.csv"
    )

    with detail_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                detail_rows[
                    0
                ].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            detail_rows
        )

    summary_path = (
        args.output_dir
        / "score_filtered_summary.csv"
    )

    with summary_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                score_rows[
                    0
                ].keys()
            ),
        )

        w.writeheader()
        w.writerows(
            score_rows
        )

    #
    # Save the most promising candidates:
    # baseline score descending,
    # same-class IoU < .5.
    #
    promising = sorted(
        [
            r
            for r in detail_rows
            if r[
                "max_same_class_iou"
            ] < 0.50
        ],
        key=lambda r:
            -r[
                "baseline_score"
            ],
    )

    top_path = (
        args.output_dir
        / "top_unique_candidates.csv"
    )

    if promising:
        with top_path.open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as f:

            w = csv.DictWriter(
                f,
                fieldnames=list(
                    promising[
                        0
                    ].keys()
                ),
            )

            w.writeheader()
            w.writerows(
                promising
            )

    print()
    print(
        "===== DONE ====="
    )

    print(
        "Detail:",
        detail_path,
    )

    print(
        "Summary:",
        summary_path,
    )

    print(
        "Unique:",
        top_path,
    )


if __name__ == "__main__":
    main()
