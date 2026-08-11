#!/usr/bin/env python3

import argparse
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


def load_module(path: Path, name: str):
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
    p.add_argument(
        "--pair-chunk",
        type=int,
        default=64,
        help="Chunk size for memory-safe Original/HFlip pairing.",
    )
    p.add_argument(
        "--baseline-only",
        action="store_true",
        help="Only verify the exact Final Combo baseline; skip fusion sweep.",
    )
    return p.parse_args()


def read_yolo_gt(label_path, width, height):
    rows = []

    if not label_path.exists():
        return np.empty((0, 5), dtype=np.float32)

    text = label_path.read_text().strip()
    if not text:
        return np.empty((0, 5), dtype=np.float32)

    for line in text.splitlines():
        x = line.split()
        if len(x) < 5:
            continue

        cls = int(float(x[0]))
        xc, yc, bw, bh = map(float, x[1:5])

        rows.append([
            cls,
            (xc - bw / 2) * width,
            (yc - bh / 2) * height,
            (xc + bw / 2) * width,
            (yc + bh / 2) * height,
        ])

    return np.asarray(rows, dtype=np.float32) if rows else \
        np.empty((0, 5), dtype=np.float32)


def image_size(item, manifest, image_name, npz):
    for wk in ("width", "image_width", "orig_width"):
        if wk in item:
            width = int(item[wk])
            break
    else:
        width = None

    for hk in ("height", "image_height", "orig_height"):
        if hk in item:
            height = int(item[hk])
            break
    else:
        height = None

    if width is not None and height is not None:
        return width, height

    root = manifest.get("images")
    if isinstance(root, str):
        p = Path(root) / image_name
        if p.exists():
            from PIL import Image
            with Image.open(p) as im:
                return im.size

    raise RuntimeError(f"Cannot determine image size: {image_name}")


def pair_iou_matrix(a, b):
    if len(a) == 0 or len(b) == 0:
        return np.empty((len(a), len(b)), dtype=np.float32)

    A = a[:, None, :]
    B = b[None, :, :]

    x1 = np.maximum(A[..., 0], B[..., 0])
    y1 = np.maximum(A[..., 1], B[..., 1])
    x2 = np.minimum(A[..., 2], B[..., 2])
    y2 = np.minimum(A[..., 3], B[..., 3])

    iw = np.maximum(0.0, x2 - x1)
    ih = np.maximum(0.0, y2 - y1)
    inter = iw * ih

    aa = (
        np.maximum(0.0, A[..., 2] - A[..., 0]) *
        np.maximum(0.0, A[..., 3] - A[..., 1])
    )
    ba = (
        np.maximum(0.0, B[..., 2] - B[..., 0]) *
        np.maximum(0.0, B[..., 3] - B[..., 1])
    )

    union = aa + ba - inter

    return np.divide(
        inter,
        union,
        out=np.zeros_like(inter, dtype=np.float32),
        where=union > 0,
    )


def build_pair_features(
    orig,
    hflip,
    cls,
    topk=3,
    chunk_size=64,
):
    """
    Memory-safe cross-view nearest-pair search.

    Previous implementation materialized full NxM feature matrices.
    This version keeps the same geometry metric but computes it in
    small chunks, so peak memory is O(chunk_size * max(N, M))
    instead of O(N * M).

    Pair selection remains completely GT-independent.
    """
    o = orig[orig[:, 0].astype(np.int32) == cls]
    h = hflip[hflip[:, 0].astype(np.int32) == cls]

    if len(o) == 0 or len(h) == 0:
        return []

    ob = o[:, 2:6].astype(np.float32, copy=False)
    hb = h[:, 2:6].astype(np.float32, copy=False)

    ow = np.maximum(ob[:, 2] - ob[:, 0], 1e-6)
    oh = np.maximum(ob[:, 3] - ob[:, 1], 1e-6)

    hw = np.maximum(hb[:, 2] - hb[:, 0], 1e-6)
    hh = np.maximum(hb[:, 3] - hb[:, 1], 1e-6)

    ocx = (ob[:, 0] + ob[:, 2]) / 2
    ocy = (ob[:, 1] + ob[:, 3]) / 2

    hcx = (hb[:, 0] + hb[:, 2]) / 2
    hcy = (hb[:, 1] + hb[:, 3]) / 2

    pairs = set()

    #
    # Pass 1:
    # for each Original candidate, retain nearest top-k HFlip candidates.
    #
    k_h = max(1, min(int(topk), len(h)))

    for start_i in range(0, len(o), chunk_size):
        end_i = min(start_i + chunk_size, len(o))

        cb = ob[start_i:end_i]
        cw = ow[start_i:end_i]
        ch = oh[start_i:end_i]
        ccx = ocx[start_i:end_i]
        ccy = ocy[start_i:end_i]

        dx = np.abs(
            ccx[:, None] - hcx[None, :]
        )
        dy = np.abs(
            ccy[:, None] - hcy[None, :]
        )

        avgw = (
            cw[:, None] + hw[None, :]
        ) / 2.0
        avgh = (
            ch[:, None] + hh[None, :]
        ) / 2.0

        dxn = dx / np.maximum(avgw, 1e-6)
        dyn = dy / np.maximum(avgh, 1e-6)

        wr = np.maximum(
            cw[:, None] / hw[None, :],
            hw[None, :] / cw[:, None],
        )

        hr = np.maximum(
            ch[:, None] / hh[None, :],
            hh[None, :] / ch[:, None],
        )

        piou = pair_iou_matrix(cb, hb)

        metric = (
            dxn
            + dyn
            + 0.15 * np.log(np.maximum(wr, 1.0))
            + 0.15 * np.log(np.maximum(hr, 1.0))
            - 0.25 * piou
        )

        inds = np.argpartition(
            metric,
            k_h - 1,
            axis=1,
        )[:, :k_h]

        for local_i in range(end_i - start_i):
            oi = start_i + local_i
            for hj in inds[local_i]:
                pairs.add((oi, int(hj)))

        # Let temporary chunk matrices be reclaimed immediately.
        del dx, dy, avgw, avgh, dxn, dyn, wr, hr, piou, metric, inds

    #
    # Pass 2:
    # symmetrical search: for each HFlip candidate,
    # retain nearest top-k Original candidates.
    #
    k_o = max(1, min(int(topk), len(o)))

    for start_j in range(0, len(h), chunk_size):
        end_j = min(start_j + chunk_size, len(h))

        cb = hb[start_j:end_j]
        cw = hw[start_j:end_j]
        ch = hh[start_j:end_j]
        ccx = hcx[start_j:end_j]
        ccy = hcy[start_j:end_j]

        dx = np.abs(
            ccx[:, None] - ocx[None, :]
        )
        dy = np.abs(
            ccy[:, None] - ocy[None, :]
        )

        avgw = (
            cw[:, None] + ow[None, :]
        ) / 2.0
        avgh = (
            ch[:, None] + oh[None, :]
        ) / 2.0

        dxn = dx / np.maximum(avgw, 1e-6)
        dyn = dy / np.maximum(avgh, 1e-6)

        wr = np.maximum(
            cw[:, None] / ow[None, :],
            ow[None, :] / cw[:, None],
        )

        hr = np.maximum(
            ch[:, None] / oh[None, :],
            oh[None, :] / ch[:, None],
        )

        piou = pair_iou_matrix(cb, ob)

        metric = (
            dxn
            + dyn
            + 0.15 * np.log(np.maximum(wr, 1.0))
            + 0.15 * np.log(np.maximum(hr, 1.0))
            - 0.25 * piou
        )

        inds = np.argpartition(
            metric,
            k_o - 1,
            axis=1,
        )[:, :k_o]

        for local_j in range(end_j - start_j):
            hj = start_j + local_j
            for oi in inds[local_j]:
                pairs.add((int(oi), hj))

        del dx, dy, avgw, avgh, dxn, dyn, wr, hr, piou, metric, inds

    #
    # Compute exact scalar features only for retained pairs.
    #
    out = []

    for i, j in pairs:
        a = ob[i]
        b = hb[j]

        wa = max(float(a[2] - a[0]), 1e-6)
        ha = max(float(a[3] - a[1]), 1e-6)

        wb = max(float(b[2] - b[0]), 1e-6)
        hb_ = max(float(b[3] - b[1]), 1e-6)

        cxa = (float(a[0]) + float(a[2])) / 2.0
        cya = (float(a[1]) + float(a[3])) / 2.0

        cxb = (float(b[0]) + float(b[2])) / 2.0
        cyb = (float(b[1]) + float(b[3])) / 2.0

        dx = abs(cxa - cxb)
        dy = abs(cya - cyb)

        dxn = dx / max((wa + wb) / 2.0, 1e-6)
        dyn = dy / max((ha + hb_) / 2.0, 1e-6)

        wr = max(wa / wb, wb / wa)
        hr = max(ha / hb_, hb_ / ha)

        aa = wa * ha
        ab = wb * hb_
        ar = max(aa / ab, ab / aa)

        piou = float(
            pair_iou_matrix(
                a.reshape(1, 4),
                b.reshape(1, 4),
            )[0, 0]
        )

        s1 = max(float(o[i, 1]), 1e-12)
        s2 = max(float(h[j, 1]), 1e-12)

        out.append({
            "cls": cls,
            "o": o[i],
            "h": h[j],
            "pair_iou": piou,
            "dxn": dxn,
            "dyn": dyn,
            "wr": wr,
            "hr": hr,
            "ar": ar,
            "score_ratio": max(s1, s2) / min(s1, s2),
        })

    return out

def gate(pair, cfg):
    if pair["cls"] not in cfg["classes"]:
        return False

    return (
        pair["pair_iou"] >= cfg["min_pair_iou"]
        and pair["dxn"] <= cfg["max_dxn"]
        and pair["dyn"] <= cfg["max_dyn"]
        and pair["wr"] <= cfg["max_wr"]
        and pair["hr"] <= cfg["max_hr"]
        and pair["ar"] <= cfg["max_ar"]
    )


def fuse_box(pair, method):
    a = pair["o"][2:6].astype(np.float32)
    b = pair["h"][2:6].astype(np.float32)

    if method == "avg50":
        return (a + b) / 2.0

    if method == "envelope":
        return np.asarray([
            min(a[0], b[0]),
            min(a[1], b[1]),
            max(a[2], b[2]),
            max(a[3], b[3]),
        ], dtype=np.float32)

    if method == "score_weighted":
        sa = max(float(pair["o"][1]), 1e-12)
        sb = max(float(pair["h"][1]), 1e-12)
        return (a * sa + b * sb) / (sa + sb)

    raise ValueError(method)


def make_fusion_rows(pairs, cfg, cols):
    out = []

    for p in pairs:
        if not gate(p, cfg):
            continue

        for method in cfg["methods"]:
            box = fuse_box(p, method)

            row = np.zeros(cols, dtype=np.float32)
            row[0] = p["cls"]

            # Always lower than either parent, so baseline detections
            # are processed first by score-based matching.
            row[1] = min(
                float(p["o"][1]),
                float(p["h"][1]),
            ) * 0.10

            row[2:6] = box
            out.append(row)

    if not out:
        return np.empty((0, cols), dtype=np.float32)

    return np.stack(out)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combo = load_module(
        Path("scripts/18_final_combo_from_cache.py"),
        "combo18_s20",
    )
    diag = load_module(
        Path("scripts/14_fn_diagnostic.py"),
        "diag14_s20",
    )

    device = torch.device(args.device)

    om = combo.load_manifest(args.original_cache)
    hm = combo.load_manifest(args.hflip_cache)

    omap = combo.manifest_map(om)
    hmap = combo.manifest_map(hm)

    hclasses = {0, 2, 3, 4, 7}

    # IMPORTANT:
    # Inherit ALL stitch defaults directly from script 18.
    # Do not hand-copy hidden/default parameters here, otherwise the
    # Final Combo baseline can drift from the validated 824/21 result.
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

    stitch_args = SimpleNamespace(**vars(combo_defaults))

    # These are the explicitly validated Final Combo values.
    stitch_args.min_aspect = 5.0
    stitch_args.x_tol = 64.0
    stitch_args.max_y_gap = 64.0
    stitch_args.min_merged_height = 1300.0

    # First-pass sweep deliberately emphasizes the four classes
    # with the strongest current fusion signal.
    main_classes = {0, 2, 3, 5}

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
            "name": "center_env",
            "classes": main_classes,
            "min_pair_iou": 0.00,
            "max_dxn": 0.60,
            "max_dyn": 0.60,
            "max_wr": 3.0,
            "max_hr": 3.0,
            "max_ar": 4.0,
            "methods": ["envelope"],
        },
        {
            "name": "center_avg",
            "classes": main_classes,
            "min_pair_iou": 0.00,
            "max_dxn": 0.60,
            "max_dyn": 0.60,
            "max_wr": 3.0,
            "max_hr": 3.0,
            "max_ar": 4.0,
            "methods": ["avg50"],
        },
        {
            "name": "center_weighted",
            "classes": main_classes,
            "min_pair_iou": 0.00,
            "max_dxn": 0.60,
            "max_dyn": 0.60,
            "max_wr": 3.0,
            "max_hr": 3.0,
            "max_ar": 4.0,
            "methods": ["score_weighted"],
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
            "name": "loose_avg",
            "classes": main_classes,
            "min_pair_iou": 0.00,
            "max_dxn": 1.00,
            "max_dyn": 1.00,
            "max_wr": 4.0,
            "max_hr": 4.0,
            "max_ar": 6.0,
            "methods": ["avg50"],
        },

        # Targeted diagnostic rules derived from the mechanism,
        # not from GT coordinates.
        {
            "name": "qilie_loose_weighted",
            "classes": {2},
            "min_pair_iou": 0.00,
            "max_dxn": 0.60,
            "max_dyn": 1.10,
            "max_wr": 3.0,
            "max_hr": 6.0,
            "max_ar": 12.0,
            "methods": ["score_weighted"],
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
        {
            "name": "jieba_wide_avg",
            "classes": {0},
            "min_pair_iou": 0.00,
            "max_dxn": 1.00,
            "max_dyn": 0.50,
            "max_wr": 2.0,
            "max_hr": 4.0,
            "max_ar": 6.0,
            "methods": ["avg50"],
        },
    ]

    stats = {
        c["name"]: {
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "added": 0,
        }
        for c in configs
    }

    base = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
    }

    print("===== CROSS-VIEW FUSION SWEEP =====")
    print("Device:", device)
    print("Images:", len(omap))
    print("Pair top-k:", args.pair_topk)
    print("Pair chunk:", args.pair_chunk)
    print()

    for idx, (name, item) in enumerate(omap.items(), 1):
        hitem = hmap[name]

        onpz = np.load(args.original_cache / item["cache_file"])
        hnpz = np.load(args.hflip_cache / hitem["cache_file"])

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

        width, height = image_size(
            item,
            om,
            name,
            onpz,
        )

        label_path = args.labels / (
            Path(name).stem + ".txt"
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

        if args.baseline_only:
            if idx % 50 == 0 or idx == len(omap):
                print(
                    f"{idx}/{len(omap)} "
                    f"baseTP={base['tp']} "
                    f"baseFN={base['fn']}"
                )
            continue

        # Build broad pair pool once per image.
        pairs = []

        target_union = set()
        for c in configs:
            target_union.update(c["classes"])

        for cls in sorted(target_union):
            pairs.extend(
                build_pair_features(
                    original,
                    hflip,
                    cls,
                    args.pair_topk,
                    args.pair_chunk,
                )
            )

        cols = final.shape[1]

        for cfg in configs:
            fused = make_fusion_rows(
                pairs,
                cfg,
                cols,
            )

            if len(fused):
                augmented = np.concatenate(
                    [final, fused],
                    axis=0,
                )
            else:
                augmented = final

            tp, fp, fn, *_ = diag.match_predictions(
                augmented,
                gt,
                0.50,
            )

            s = stats[cfg["name"]]
            s["tp"] += int(tp)
            s["fp"] += int(fp)
            s["fn"] += int(fn)
            s["added"] += len(fused)

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

    if args.baseline_only:
        print()
        print("BASELINE_EXACT_OK")
        return

    rows = []

    print()
    print("===== SWEEP RESULTS =====")

    for cfg in configs:
        name = cfg["name"]
        s = stats[name]

        denom = s["tp"] + s["fn"]
        recall = s["tp"] / denom if denom else 0

        dtp = s["tp"] - base["tp"]
        dfn = base["fn"] - s["fn"]
        dfp = s["fp"] - base["fp"]

        row = {
            "config": name,
            "tp": s["tp"],
            "fp": s["fp"],
            "fn": s["fn"],
            "recall": recall,
            "score_like": recall * 100,
            "delta_tp": dtp,
            "delta_fn": dfn,
            "delta_fp": dfp,
            "added_fusion_boxes": s["added"],
        }
        rows.append(row)

    rows.sort(
        key=lambda r: (
            -r["tp"],
            r["added_fusion_boxes"],
            r["delta_fp"],
        )
    )

    for r in rows:
        print(
            f"{r['config']:24s} "
            f"TP={r['tp']:3d} "
            f"FN={r['fn']:2d} "
            f"Recall={r['recall']:.6f} "
            f"ScoreLike={r['score_like']:.2f} "
            f"dTP={r['delta_tp']:+d} "
            f"dFP={r['delta_fp']:+d} "
            f"added={r['added_fusion_boxes']:,}"
        )

    import csv

    out_csv = args.output_dir / "sweep_summary.csv"

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
