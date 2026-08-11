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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def get_combo_defaults(combo):
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "18_final_combo_from_cache.py",
            "--mode", "val",
            "--original-cache", "_dummy_orig",
            "--hflip-cache", "_dummy_hflip",
            "--output-dir", "_dummy_out",
        ]
        return combo.parse_args()
    finally:
        sys.argv = old_argv


def parse_args(combo_defaults):
    p = argparse.ArgumentParser(
        description="Diagnose the exact remaining FNs after Final Combo."
    )

    p.add_argument(
        "--diag-script",
        type=Path,
        default=Path("scripts/14_fn_diagnostic.py"),
    )
    p.add_argument(
        "--combo-script",
        type=Path,
        default=Path("scripts/18_final_combo_from_cache.py"),
    )

    p.add_argument("--original-cache", type=Path, required=True)
    p.add_argument("--hflip-cache", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)

    p.add_argument("--global-iou", type=float, default=0.90)
    p.add_argument("--match-iou", type=float, default=0.50)
    p.add_argument(
        "--hflip-classes",
        type=int,
        nargs="+",
        default=[0, 2, 3, 4, 7],
    )

    # Keep stitch defaults exactly aligned with script 18.
    p.add_argument(
        "--min-aspect",
        type=float,
        default=getattr(combo_defaults, "min_aspect", 5.0),
    )
    p.add_argument(
        "--x-tol",
        type=float,
        default=getattr(combo_defaults, "x_tol", 64.0),
    )
    p.add_argument(
        "--max-y-gap",
        type=float,
        default=getattr(combo_defaults, "max_y_gap", 64.0),
    )
    p.add_argument(
        "--min-height",
        type=float,
        default=getattr(combo_defaults, "min_height", 0.0),
    )
    p.add_argument(
        "--min-x-overlap",
        type=float,
        default=getattr(combo_defaults, "min_x_overlap", 0.0),
    )
    p.add_argument(
        "--stride",
        type=int,
        default=getattr(combo_defaults, "stride", 768),
    )
    p.add_argument(
        "--min-rows",
        type=int,
        default=getattr(combo_defaults, "min_rows", 2),
    )
    p.add_argument(
        "--min-merged-height",
        type=float,
        default=getattr(combo_defaults, "min_merged_height", 1300.0),
    )

    p.add_argument("--device", default="cpu")
    p.add_argument("--edge-margin", type=float, default=32.0)
    p.add_argument("--nearby-iou", type=float, default=0.10)
    p.add_argument("--fusion-topk", type=int, default=5)

    return p.parse_args()


def read_yolo_gt(label_path: Path, width: int, height: int):
    rows = []

    if not label_path.exists():
        return np.empty((0, 5), dtype=np.float32)

    text = label_path.read_text().strip()
    if not text:
        return np.empty((0, 5), dtype=np.float32)

    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue

        cls = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])

        x1 = (xc - bw / 2.0) * width
        y1 = (yc - bh / 2.0) * height
        x2 = (xc + bw / 2.0) * width
        y2 = (yc + bh / 2.0) * height

        rows.append([cls, x1, y1, x2, y2])

    if not rows:
        return np.empty((0, 5), dtype=np.float32)

    return np.asarray(rows, dtype=np.float32)


def iou_one_to_many(box, boxes):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    iw = np.maximum(0.0, x2 - x1)
    ih = np.maximum(0.0, y2 - y1)
    inter = iw * ih

    area_a = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    area_b = (
        np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    )

    union = area_a + area_b - inter
    return np.divide(
        inter,
        union,
        out=np.zeros_like(inter, dtype=np.float32),
        where=union > 0,
    )


def iou_box(a, b):
    return float(iou_one_to_many(
        np.asarray(a, dtype=np.float32),
        np.asarray([b], dtype=np.float32),
    )[0])


def iou_matrix(pred_boxes, gt_boxes):
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return np.empty(
            (len(pred_boxes), len(gt_boxes)),
            dtype=np.float32,
        )

    p = pred_boxes[:, None, :]
    g = gt_boxes[None, :, :]

    x1 = np.maximum(p[..., 0], g[..., 0])
    y1 = np.maximum(p[..., 1], g[..., 1])
    x2 = np.minimum(p[..., 2], g[..., 2])
    y2 = np.minimum(p[..., 3], g[..., 3])

    iw = np.maximum(0.0, x2 - x1)
    ih = np.maximum(0.0, y2 - y1)
    inter = iw * ih

    pa = (
        np.maximum(0.0, p[..., 2] - p[..., 0])
        * np.maximum(0.0, p[..., 3] - p[..., 1])
    )
    ga = (
        np.maximum(0.0, g[..., 2] - g[..., 0])
        * np.maximum(0.0, g[..., 3] - g[..., 1])
    )

    union = pa + ga - inter

    return np.divide(
        inter,
        union,
        out=np.zeros_like(inter, dtype=np.float32),
        where=union > 0,
    )


def our_unmatched_gt(final, gt, threshold):
    """
    Score-order greedy same-class matching.

    final format:
        [class, score, x1, y1, x2, y2, ...]

    gt format:
        [class, x1, y1, x2, y2]
    """
    if len(gt) == 0:
        return []

    matched = np.zeros(len(gt), dtype=bool)

    gt_classes = gt[:, 0].astype(np.int32)
    pred_classes = (
        final[:, 0].astype(np.int32)
        if len(final)
        else np.empty((0,), dtype=np.int32)
    )

    for cls in np.unique(gt_classes):
        gi = np.where(gt_classes == cls)[0]
        pi = np.where(pred_classes == cls)[0]

        if len(pi) == 0:
            continue

        mat = iou_matrix(
            final[pi, 2:6],
            gt[gi, 1:5],
        )

        if mat.size == 0:
            continue

        # Only predictions capable of making a >= threshold match matter.
        potential = np.where(mat.max(axis=1) >= threshold)[0]

        # pi is in final prediction order; final is score-descending.
        for local_pi in potential:
            available = ~matched[gi]
            if not np.any(available):
                break

            vals = mat[local_pi].copy()
            vals[~available] = -1.0

            j = int(np.argmax(vals))
            if vals[j] >= threshold:
                matched[gi[j]] = True

    return np.where(~matched)[0].tolist()


def det_obj_to_class_box(obj):
    cls = None
    box = None

    for attr in ("class_id", "cls", "class_idx", "category_id"):
        if hasattr(obj, attr):
            try:
                cls = int(getattr(obj, attr))
                break
            except Exception:
                pass

    for attr in ("bbox", "box", "xyxy"):
        if hasattr(obj, attr):
            try:
                arr = np.asarray(getattr(obj, attr), dtype=np.float32).reshape(-1)
                if len(arr) >= 4:
                    box = arr[:4]
                    break
            except Exception:
                pass

    if box is None:
        attrs = ("x1", "y1", "x2", "y2")
        if all(hasattr(obj, x) for x in attrs):
            box = np.asarray(
                [getattr(obj, x) for x in attrs],
                dtype=np.float32,
            )

    return cls, box


def map_reference_unmatched(unmatched, gt_array):
    if unmatched is None:
        return None

    try:
        seq = list(unmatched)
    except Exception:
        return None

    if len(seq) == 0:
        return []

    # Case 1: direct GT indices.
    if all(isinstance(x, (int, np.integer)) for x in seq):
        vals = [int(x) for x in seq]
        if all(0 <= x < len(gt_array) for x in vals):
            return vals

    # Case 2: boolean mask.
    if len(seq) == len(gt_array) and all(
        isinstance(x, (bool, np.bool_)) for x in seq
    ):
        return [i for i, x in enumerate(seq) if bool(x)]

    # Case 3: Detection objects.
    out = []
    used = set()

    for obj in seq:
        cls, box = det_obj_to_class_box(obj)

        if cls is None or box is None:
            return None

        best_idx = None
        best_err = float("inf")

        for i, gt in enumerate(gt_array):
            if i in used:
                continue
            if int(gt[0]) != cls:
                continue

            err = float(np.max(np.abs(gt[1:5] - box)))
            if err < best_err:
                best_err = err
                best_idx = i

        if best_idx is None or best_err > 2.0:
            return None

        used.add(best_idx)
        out.append(best_idx)

    return out


def find_label(labels_dir: Path, image_name: str):
    p = Path(image_name)

    options = [
        labels_dir / f"{p.stem}.txt",
        labels_dir / p.with_suffix(".txt").name,
    ]

    for x in options:
        if x.exists():
            return x

    return options[0]


def image_size(item, manifest, image_name, npz):
    width_keys = ("width", "image_width", "orig_width")
    height_keys = ("height", "image_height", "orig_height")

    width = None
    height = None

    for k in width_keys:
        if k in item:
            width = int(item[k])
            break

    for k in height_keys:
        if k in item:
            height = int(item[k])
            break

    if width is None:
        for k in width_keys:
            if k in npz:
                width = int(np.asarray(npz[k]).reshape(-1)[0])
                break

    if height is None:
        for k in height_keys:
            if k in npz:
                height = int(np.asarray(npz[k]).reshape(-1)[0])
                break

    if width is not None and height is not None:
        return width, height

    # Final fallback: read original image.
    images_root = manifest.get("images")

    if isinstance(images_root, str):
        ipath = Path(images_root) / image_name

        if ipath.exists():
            from PIL import Image
            with Image.open(ipath) as im:
                return im.size

    raise RuntimeError(
        f"Could not determine image size for {image_name}. "
        f"Item keys={list(item.keys())}"
    )


def best_candidate(arr, gt_box, class_id=None):
    if arr is None or len(arr) == 0:
        return None

    if class_id is not None:
        mask = arr[:, 0].astype(np.int32) == int(class_id)
        cand = arr[mask]
    else:
        cand = arr

    if len(cand) == 0:
        return None

    ious = iou_one_to_many(gt_box, cand[:, 2:6])
    j = int(np.argmax(ious))

    return {
        "row": cand[j],
        "iou": float(ious[j]),
        "score": float(cand[j, 1]),
        "class_id": int(cand[j, 0]),
        "box": cand[j, 2:6].astype(np.float32),
    }


def top_candidates(arr, gt_box, class_id, k):
    if arr is None or len(arr) == 0:
        return []

    mask = arr[:, 0].astype(np.int32) == int(class_id)
    cand = arr[mask]

    if len(cand) == 0:
        return []

    ious = iou_one_to_many(gt_box, cand[:, 2:6])
    order = np.argsort(-ious)[:k]

    out = []
    for j in order:
        out.append({
            "row": cand[j],
            "iou": float(ious[j]),
            "score": float(cand[j, 1]),
            "box": cand[j, 2:6].astype(np.float32),
        })
    return out


def fusion_oracle(orig, hflip, gt_box, class_id, topk):
    """
    IMPORTANT:
    GT is used here only to answer:
        "Does a simple O/H fusion have theoretical rescue potential?"

    This is not a deployable test-time rule.
    """
    oa = top_candidates(orig, gt_box, class_id, topk)
    hb = top_candidates(hflip, gt_box, class_id, topk)

    if not oa or not hb:
        return None

    best = None

    for i, a in enumerate(oa):
        for j, b in enumerate(hb):
            box_a = a["box"]
            box_b = b["box"]

            candidates = []

            avg = (box_a + box_b) / 2.0
            candidates.append(("avg50", avg))

            sa = max(a["score"], 1e-12)
            sb = max(b["score"], 1e-12)
            weighted = (box_a * sa + box_b * sb) / (sa + sb)
            candidates.append(("score_weighted", weighted))

            envelope = np.asarray([
                min(box_a[0], box_b[0]),
                min(box_a[1], box_b[1]),
                max(box_a[2], box_b[2]),
                max(box_a[3], box_b[3]),
            ], dtype=np.float32)
            candidates.append(("envelope", envelope))

            for method, fused in candidates:
                fiou = iou_box(gt_box, fused)

                rec = {
                    "iou": fiou,
                    "method": method,
                    "orig_rank": i + 1,
                    "hflip_rank": j + 1,
                    "orig_iou": a["iou"],
                    "hflip_iou": b["iou"],
                    "orig_score": a["score"],
                    "hflip_score": b["score"],
                    "box": fused,
                    "orig_box": box_a,
                    "hflip_box": box_b,
                }

                if best is None or rec["iou"] > best["iou"]:
                    best = rec

    return best


def fmt_box(box):
    if box is None:
        return ""
    return ",".join(f"{float(x):.3f}" for x in box)


def safe_edge_flag(diag, result, margin):
    if result is None:
        return ""

    try:
        return int(bool(diag.candidate_edge_flag(result["row"], margin)))
    except Exception:
        return ""


def count_nearby(arr, gt_box, class_id, threshold, same_class=True):
    if arr is None or len(arr) == 0:
        return 0

    if same_class:
        arr = arr[arr[:, 0].astype(np.int32) == int(class_id)]

    if len(arr) == 0:
        return 0

    ious = iou_one_to_many(gt_box, arr[:, 2:6])
    return int(np.sum(ious >= threshold))


def classify_failure(
    cls,
    gt_box,
    best_same,
    best_any,
    fusion,
    tile_size=1280,
):
    w = float(gt_box[2] - gt_box[0])
    h = float(gt_box[3] - gt_box[1])

    long_dim = max(w, h)
    short_dim = max(1.0, min(w, h))
    elong = long_dim / short_dim

    same_iou = best_same["iou"] if best_same else 0.0
    any_iou = best_any["iou"] if best_any else 0.0
    any_cls = best_any["class_id"] if best_any else -1

    if (
        long_dim > tile_size
        and elong >= 8
        and same_iou < 0.5
        and same_iou > 0.15
    ):
        return "localization_or_tile_fragmentation"

    if (
        any_iou >= 0.50
        and any_cls != cls
        and same_iou < 0.50
    ):
        return "class_confusion"

    if same_iou >= 0.45:
        return "localization_near_miss"

    if fusion is not None and fusion["iou"] >= 0.50:
        return "fusion_oracle_rescuable"

    if same_iou > 0:
        return "localization_failure"

    return "no_same_class_candidate"


def main():
    # Load script 18 first so we can inherit its exact defaults.
    combo_path = Path("scripts/18_final_combo_from_cache.py")
    combo = load_module(combo_path, "combo18_diag_defaults")
    combo_defaults = get_combo_defaults(combo)

    args = parse_args(combo_defaults)

    # Reload requested paths if custom paths were given.
    if args.combo_script.resolve() != combo_path.resolve():
        combo = load_module(args.combo_script, "combo18_diag")

    diag = load_module(args.diag_script, "diag14_final")

    if args.device != "cpu":
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    om = combo.load_manifest(args.original_cache)
    hm = combo.load_manifest(args.hflip_cache)

    omap = combo.manifest_map(om)
    hmap = combo.manifest_map(hm)

    if len(omap) != len(hmap):
        raise RuntimeError(
            f"Original/HFlip image count mismatch: "
            f"{len(omap)} vs {len(hmap)}"
        )

    hclasses = set(args.hflip_classes)

    # Namespace passed into script 18 stitch functions.
    stitch_args = SimpleNamespace(**vars(combo_defaults))

    for key, value in vars(args).items():
        setattr(stitch_args, key, value)

    total_ref_tp = 0
    total_ref_fp = 0
    total_ref_fn = 0
    total_merged = 0

    rows = []
    backend_seen = set()
    own_mismatch_images = []

    print("===== FINAL COMBO FN DIAGNOSTIC =====")
    print("Device        :", device)
    print("Images        :", len(omap))
    print("Global IoU    :", args.global_iou)
    print("Match IoU     :", args.match_iou)
    print("HFlip classes :", sorted(hclasses))
    print()

    for idx, (name, item) in enumerate(omap.items(), 1):
        if name not in hmap:
            raise RuntimeError(f"HFlip cache missing image: {name}")

        hitem = hmap[name]

        onpz = np.load(args.original_cache / item["cache_file"])
        hnpz = np.load(args.hflip_cache / hitem["cache_file"])

        original = onpz["candidates"].astype(
            np.float32,
            copy=False,
        )
        hflip_all = hnpz["candidates"].astype(
            np.float32,
            copy=False,
        )

        hsel = hflip_all[
            np.isin(
                hflip_all[:, 0].astype(np.int32),
                list(hclasses),
            )
        ]

        if len(hsel):
            union_pre = np.concatenate(
                [original, hsel],
                axis=0,
            )
        else:
            union_pre = original

        post, backend = combo.nms(
            union_pre,
            args.global_iou,
            diag,
            device,
        )
        backend_seen.add(backend)

        # Capture raw stitch proposals for diagnostics.
        zpost = post[
            post[:, 0].astype(np.int32) == 1
        ]

        try:
            merged_raw = combo.stitch_zonglie(
                zpost,
                stitch_args,
                device,
            )
        except Exception:
            merged_raw = np.empty(
                (0, post.shape[1] if post.ndim == 2 else 6),
                dtype=np.float32,
            )

        final, merged_count = combo.add_stitched(
            post,
            stitch_args,
            device,
        )
        total_merged += merged_count

        width, height = image_size(
            item,
            om,
            name,
            onpz,
        )

        label_path = find_label(args.labels, name)

        gt_array = read_yolo_gt(
            label_path,
            width,
            height,
        )

        # Exact reference matching from script 14.
        gt_obj = diag.read_yolo_gt(
            label_path,
            width,
            height,
        )

        (
            tp_ref,
            fp_ref,
            fn_ref,
            unmatched_ref,
            matched_ref,
            extra_ref,
        ) = diag.match_predictions(
            final,
            gt_obj,
            args.match_iou,
        )

        total_ref_tp += int(tp_ref)
        total_ref_fp += int(fp_ref)
        total_ref_fn += int(fn_ref)

        # Try to map script-14 unmatched GTs directly.
        unmatched_idx = map_reference_unmatched(
            unmatched_ref,
            gt_array,
        )

        # Fallback to local matching.
        own_unmatched = our_unmatched_gt(
            final,
            gt_array,
            args.match_iou,
        )

        if unmatched_idx is None:
            unmatched_idx = own_unmatched

        if len(unmatched_idx) != int(fn_ref):
            own_mismatch_images.append(
                (
                    name,
                    int(fn_ref),
                    len(unmatched_idx),
                    len(own_unmatched),
                    type(unmatched_ref).__name__,
                )
            )

            # Prefer local matcher if it has the exact FN count.
            if len(own_unmatched) == int(fn_ref):
                unmatched_idx = own_unmatched

        if len(unmatched_idx) != int(fn_ref):
            raise RuntimeError(
                f"Could not reproduce unmatched GT indices for {name}: "
                f"reference FN={fn_ref}, "
                f"mapped={len(unmatched_idx)}, "
                f"local={len(own_unmatched)}, "
                f"unmatched_ref_type={type(unmatched_ref)}"
            )

        # Full O+H pool is diagnostic only. Final Combo itself only uses hsel.
        full_oh = (
            np.concatenate([original, hflip_all], axis=0)
            if len(hflip_all)
            else original
        )

        final_pre = union_pre

        for gt_index in unmatched_idx:
            gt = gt_array[gt_index]

            cls = int(gt[0])
            gt_box = gt[1:5].astype(np.float32)

            gw = float(gt_box[2] - gt_box[0])
            gh = float(gt_box[3] - gt_box[1])
            short_side = max(1.0, min(gw, gh))
            long_side = max(gw, gh)
            elong = long_side / short_side

            best_orig = best_candidate(
                original,
                gt_box,
                cls,
            )
            best_hflip = best_candidate(
                hflip_all,
                gt_box,
                cls,
            )
            best_final_pre = best_candidate(
                final_pre,
                gt_box,
                cls,
            )
            best_final = best_candidate(
                final,
                gt_box,
                cls,
            )
            best_any = best_candidate(
                full_oh,
                gt_box,
                None,
            )
            best_stitch = best_candidate(
                merged_raw,
                gt_box,
                1,
            ) if cls == 1 else None

            fusion = fusion_oracle(
                original,
                hflip_all,
                gt_box,
                cls,
                args.fusion_topk,
            )

            failure = classify_failure(
                cls,
                gt_box,
                best_final_pre,
                best_any,
                fusion,
            )

            full_same_count = count_nearby(
                full_oh,
                gt_box,
                cls,
                args.nearby_iou,
                True,
            )
            full_any_count = count_nearby(
                full_oh,
                gt_box,
                cls,
                args.nearby_iou,
                False,
            )

            row = {
                "image_name": name,
                "class_id": cls,
                "class_name": CLASS_NAMES[cls],
                "gt_index": gt_index,

                "image_width": width,
                "image_height": height,

                "gt_x1": float(gt_box[0]),
                "gt_y1": float(gt_box[1]),
                "gt_x2": float(gt_box[2]),
                "gt_y2": float(gt_box[3]),
                "gt_width": gw,
                "gt_height": gh,
                "gt_elongation": elong,

                "best_original_iou":
                    best_orig["iou"] if best_orig else "",
                "best_original_score":
                    best_orig["score"] if best_orig else "",
                "best_original_box":
                    fmt_box(best_orig["box"]) if best_orig else "",
                "best_original_tile_edge":
                    safe_edge_flag(
                        diag,
                        best_orig,
                        args.edge_margin,
                    ),

                "best_hflip_iou":
                    best_hflip["iou"] if best_hflip else "",
                "best_hflip_score":
                    best_hflip["score"] if best_hflip else "",
                "best_hflip_box":
                    fmt_box(best_hflip["box"]) if best_hflip else "",
                "best_hflip_tile_edge":
                    safe_edge_flag(
                        diag,
                        best_hflip,
                        args.edge_margin,
                    ),
                "hflip_currently_selected":
                    int(cls in hclasses),

                "best_final_pre_iou":
                    best_final_pre["iou"]
                    if best_final_pre else "",
                "best_final_iou":
                    best_final["iou"]
                    if best_final else "",

                "best_any_class_iou":
                    best_any["iou"] if best_any else "",
                "best_any_class_id":
                    best_any["class_id"]
                    if best_any else "",
                "best_any_class_name":
                    CLASS_NAMES[best_any["class_id"]]
                    if best_any else "",
                "best_any_class_score":
                    best_any["score"]
                    if best_any else "",

                "best_stitch_iou":
                    best_stitch["iou"]
                    if best_stitch else "",

                "nearby_same_class_count":
                    full_same_count,
                "nearby_any_class_count":
                    full_any_count,

                "fusion_oracle_iou":
                    fusion["iou"] if fusion else "",
                "fusion_oracle_method":
                    fusion["method"] if fusion else "",
                "fusion_orig_rank":
                    fusion["orig_rank"] if fusion else "",
                "fusion_hflip_rank":
                    fusion["hflip_rank"] if fusion else "",
                "fusion_orig_iou":
                    fusion["orig_iou"] if fusion else "",
                "fusion_hflip_iou":
                    fusion["hflip_iou"] if fusion else "",
                "fusion_box":
                    fmt_box(fusion["box"]) if fusion else "",

                "fusion_oracle_rescue":
                    int(
                        fusion is not None
                        and fusion["iou"] >= args.match_iou
                    ),

                "is_small_object":
                    int(max(gw, gh) <= 128),
                "is_long_object":
                    int(
                        max(gw, gh) > 1280
                        or elong >= 8
                    ),

                "failure_type": failure,
            }

            rows.append(row)

        if idx % 50 == 0 or idx == len(omap):
            print(
                f"{idx}/{len(omap)} "
                f"TP={total_ref_tp} "
                f"FN={total_ref_fn} "
                f"remaining_rows={len(rows)}"
            )

    print()
    print("===== EXACT REPRODUCTION =====")
    print("TP       :", total_ref_tp)
    print("FP       :", total_ref_fp)
    print("FN       :", total_ref_fn)
    recall = (
        total_ref_tp / (total_ref_tp + total_ref_fn)
        if total_ref_tp + total_ref_fn
        else 0.0
    )
    print("Recall   :", f"{recall:.6f}")
    print("Stitched :", total_merged)
    print("NMS      :", ", ".join(sorted(backend_seen)))

    if total_ref_tp != 824 or total_ref_fn != 21:
        raise RuntimeError(
            "Final Combo reproduction changed! "
            f"Expected TP=824 FN=21, "
            f"got TP={total_ref_tp} FN={total_ref_fn}"
        )

    if len(rows) != 21:
        raise RuntimeError(
            f"Expected 21 diagnostic rows, got {len(rows)}"
        )

    # Sort useful order:
    # 1. fusion-rescuable first
    # 2. best near-miss IoU descending
    def sort_key(r):
        rescue = int(r["fusion_oracle_rescue"])
        v = r["best_final_pre_iou"]
        try:
            iou = float(v)
        except Exception:
            iou = -1
        return (-rescue, -iou, r["class_id"], r["image_name"])

    rows.sort(key=sort_key)

    csv_path = args.output_dir / "remaining_fn_21.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)

    class_counter = Counter(
        r["class_name"] for r in rows
    )
    failure_counter = Counter(
        r["failure_type"] for r in rows
    )
    fusion_counter = Counter(
        r["class_name"]
        for r in rows
        if int(r["fusion_oracle_rescue"]) == 1
    )

    class_csv = args.output_dir / "class_summary.csv"

    with class_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "class_name",
            "remaining_fn",
            "fusion_oracle_rescuable",
        ])

        for cls_name in CLASS_NAMES:
            n = class_counter.get(cls_name, 0)
            if n:
                writer.writerow([
                    cls_name,
                    n,
                    fusion_counter.get(cls_name, 0),
                ])

    failure_csv = args.output_dir / "failure_summary.csv"

    with failure_csv.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "failure_type",
            "count",
        ])
        for k, v in failure_counter.most_common():
            writer.writerow([k, v])

    near_miss = []
    fusion_rescue = []

    for r in rows:
        try:
            biou = float(r["best_final_pre_iou"])
        except Exception:
            biou = -1

        if 0.40 <= biou < 0.50:
            near_miss.append(r)

        if int(r["fusion_oracle_rescue"]) == 1:
            fusion_rescue.append(r)

    md = []
    md.append("# Final Combo Remaining 21 FN Diagnostic")
    md.append("")
    md.append("## Exact reproduction")
    md.append("")
    md.append(f"- TP: {total_ref_tp}")
    md.append(f"- FP: {total_ref_fp}")
    md.append(f"- FN: {total_ref_fn}")
    md.append(f"- Recall: {recall:.6f}")
    md.append(f"- Stitched zonglie: {total_merged}")
    md.append("")
    md.append("## Remaining FN by class")
    md.append("")

    for cls_name in CLASS_NAMES:
        n = class_counter.get(cls_name, 0)
        if n:
            md.append(
                f"- {cls_name}: {n} "
                f"(fusion oracle rescue: "
                f"{fusion_counter.get(cls_name, 0)})"
            )

    md.append("")
    md.append("## Failure types")
    md.append("")

    for k, v in failure_counter.most_common():
        md.append(f"- {k}: {v}")

    md.append("")
    md.append("## Near-miss FN (best same-class IoU 0.40~0.50)")
    md.append("")

    for r in sorted(
        near_miss,
        key=lambda x: float(x["best_final_pre_iou"]),
        reverse=True,
    ):
        md.append(
            f"- {r['image_name']} / "
            f"{r['class_name']} / "
            f"GT#{r['gt_index']}: "
            f"preIoU={float(r['best_final_pre_iou']):.6f}, "
            f"O={float(r['best_original_iou']):.6f}"
            if r["best_original_iou"] != ""
            else
            f"- {r['image_name']} / {r['class_name']}"
        )

    md.append("")
    md.append("## Oracle O/H fusion opportunities")
    md.append("")
    md.append(
        "> Oracle means GT was used only to diagnose whether "
        "a generic Original/HFlip fusion rule may have potential. "
        "These oracle pair choices must NOT be used directly on test."
    )
    md.append("")

    if fusion_rescue:
        for r in fusion_rescue:
            md.append(
                f"- {r['image_name']} / "
                f"{r['class_name']} / GT#{r['gt_index']}: "
                f"fusion={float(r['fusion_oracle_iou']):.6f}, "
                f"method={r['fusion_oracle_method']}, "
                f"O={float(r['fusion_orig_iou']):.6f}, "
                f"H={float(r['fusion_hflip_iou']):.6f}"
            )
    else:
        md.append("- None")

    md.append("")
    md.append("## Important")
    md.append("")
    md.append(
        "Oracle fusion is diagnostic only. "
        "The next experiment must derive a GT-independent "
        "pair-selection/fusion rule and then rerun the full Val matcher."
    )

    summary_path = args.output_dir / "summary.md"
    summary_path.write_text(
        "\n".join(md) + "\n",
        encoding="utf-8",
    )

    print()
    print("===== REMAINING FN BY CLASS =====")
    for k, v in class_counter.most_common():
        print(
            f"{k:16s} FN={v:2d} "
            f"oracle_fusion={fusion_counter.get(k, 0):2d}"
        )

    print()
    print("===== FAILURE SUMMARY =====")
    for k, v in failure_counter.most_common():
        print(f"{k:36s} {v}")

    print()
    print("===== ORACLE FUSION RESCUE =====")
    print("Count:", len(fusion_rescue))

    for r in fusion_rescue:
        print(
            f"{r['class_name']:16s} "
            f"{r['image_name']} "
            f"GT#{r['gt_index']} "
            f"O={float(r['fusion_orig_iou']):.4f} "
            f"H={float(r['fusion_hflip_iou']):.4f} "
            f"F={float(r['fusion_oracle_iou']):.4f} "
            f"{r['fusion_oracle_method']}"
        )

    print()
    print("===== NEAR MISSES =====")

    near_sorted = sorted(
        near_miss,
        key=lambda x: float(x["best_final_pre_iou"]),
        reverse=True,
    )

    for r in near_sorted:
        print(
            f"{r['class_name']:16s} "
            f"{r['image_name']} "
            f"GT#{r['gt_index']} "
            f"best={float(r['best_final_pre_iou']):.6f} "
            f"O={float(r['best_original_iou']) if r['best_original_iou'] != '' else -1:.6f} "
            f"H={float(r['best_hflip_iou']) if r['best_hflip_iou'] != '' else -1:.6f} "
            f"any={float(r['best_any_class_iou']) if r['best_any_class_iou'] != '' else -1:.6f} "
            f"any_cls={r['best_any_class_name']}"
        )

    print()
    print("===== OUTPUT =====")
    print(csv_path)
    print(class_csv)
    print(failure_csv)
    print(summary_path)

    if own_mismatch_images:
        print()
        print(
            "NOTE: reference-unmatched mapping required fallback "
            f"on {len(own_mismatch_images)} images, "
            "but final FN count remained exact."
        )


if __name__ == "__main__":
    main()
