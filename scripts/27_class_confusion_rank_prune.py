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


TARGETED5 = [
    (7, 0),  # yanghuatiepi -> jieba
    (6, 0),  # mamianmakeng -> jieba
    (0, 3),  # jieba -> jiaza
    (5, 4),  # huashang -> yiwuyaru
    (4, 3),  # yiwuyaru -> jiaza
]


SUPPORT4 = [
    (7, 0),
    (6, 0),
    (0, 3),
    (5, 4),
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


def select_topk(rows, k):
    if len(rows) == 0:
        return rows

    if k is None or len(rows) <= k:
        return rows

    order = np.argsort(
        -rows[:, 1],
        kind="stable",
    )

    return rows[order[:k]]


def make_duplicates(
    final,
    mappings,
    topk,
    dup_score,
):
    cls = final[:, 0].astype(np.int32)

    pieces = []
    counts = Counter()

    for src, dst in mappings:
        rows = final[cls == src]

        rows = select_topk(
            rows,
            topk,
        )

        if not len(rows):
            continue

        dup = rows.copy()

        dup[:, 0] = dst
        dup[:, 1] = dup_score

        pieces.append(dup)
        counts[(src, dst)] += len(dup)

    if not pieces:
        return np.empty(
            (0, final.shape[1]),
            dtype=np.float32,
        ), counts

    return np.concatenate(
        pieces,
        axis=0,
    ), counts


def main():
    args = parse_args()
    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    combo = load_module(
        Path("scripts/18_final_combo_from_cache.py"),
        "combo18_s27",
    )

    diag = load_module(
        Path("scripts/14_fn_diagnostic.py"),
        "diag14_s27",
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
    # Main sweep.
    #
    ks = [
        1,
        2,
        3,
        5,
        10,
        20,
        50,
        100,
        None,  # ALL
    ]

    variants = {}

    for k in ks:
        tag = (
            "all"
            if k is None
            else f"top{k}"
        )

        variants[f"targeted5_{tag}"] = (
            TARGETED5,
            k,
        )

    #
    # Also test only mappings with >= 4-image
    # Whole-Val support.
    #
    for k in [
        5,
        10,
        20,
        50,
        None,
    ]:
        tag = (
            "all"
            if k is None
            else f"top{k}"
        )

        variants[f"support4_{tag}"] = (
            SUPPORT4,
            k,
        )

    #
    # Atomic Top-K diagnostics.
    #
    for src, dst in TARGETED5:
        key = (
            f"{CLASS_NAMES[src]}"
            f"_to_"
            f"{CLASS_NAMES[dst]}"
        )

        for k in [
            1,
            3,
            5,
            10,
            20,
            50,
            None,
        ]:
            tag = (
                "all"
                if k is None
                else f"top{k}"
            )

            variants[f"{key}_{tag}"] = (
                [(src, dst)],
                k,
            )

    totals = {
        "baseline": Counter()
    }

    added = Counter()

    mapping_added = {
        v: Counter()
        for v in variants
    }

    for v in variants:
        totals[v] = Counter()

    total_stitched = 0

    print(
        "===== CLASS CONFUSION RANK PRUNE ====="
    )
    print(
        "Images   :",
        om["images_count"],
    )
    print(
        "Dup score:",
        args.dup_score,
    )
    print()

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

        final, merged_count = (
            combo.add_stitched(
                post,
                stitch_args,
                device,
            )
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
        # Baseline.
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

        #
        # Variants.
        #
        for vname, (
            maps,
            topk,
        ) in variants.items():

            dup, bymap = make_duplicates(
                final,
                maps,
                topk,
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

            for key, n in bymap.items():
                mapping_added[vname][key] += n

        if (
            idx % 25 == 0
            or idx == int(om["images_count"])
        ):
            print(
                f"{idx}/{om['images_count']} "
                f"TP={totals['baseline']['tp']} "
                f"FN={totals['baseline']['fn']}"
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

    rows = []

    print()
    print("===== MAIN RANK SWEEP =====")

    main_names = [
        f"targeted5_{'all' if k is None else f'top{k}'}"
        for k in ks
    ]

    main_names += [
        "support4_top5",
        "support4_top10",
        "support4_top20",
        "support4_top50",
        "support4_all",
    ]

    for vname in main_names:
        r = totals[vname]

        dtp = r["tp"] - b["tp"]
        dfn = r["fn"] - b["fn"]
        dfp = r["fp"] - b["fp"]

        denom = r["tp"] + r["fn"]

        recall = (
            r["tp"] / denom
            if denom
            else 0.0
        )

        print(
            f"{vname:22s} "
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

    print()
    print("===== ATOMIC RANK RETENTION =====")

    for src, dst in TARGETED5:
        key = (
            f"{CLASS_NAMES[src]}"
            f"_to_"
            f"{CLASS_NAMES[dst]}"
        )

        print()
        print(
            f"[{CLASS_NAMES[src]} "
            f"-> {CLASS_NAMES[dst]}]"
        )

        for k in [
            1,
            3,
            5,
            10,
            20,
            50,
            None,
        ]:
            tag = (
                "all"
                if k is None
                else f"top{k}"
            )

            vname = f"{key}_{tag}"
            r = totals[vname]

            dtp = (
                r["tp"]
                - b["tp"]
            )

            print(
                f"  {tag:6s} "
                f"dTP={dtp:+d} "
                f"added={added[vname]:,}"
            )

    out = (
        args.output_dir
        / "rank_prune_summary.csv"
    )

    with out.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=list(
                rows[0].keys()
            ),
        )
        w.writeheader()
        w.writerows(rows)

    print()
    print("Saved:", out)


if __name__ == "__main__":
    main()
