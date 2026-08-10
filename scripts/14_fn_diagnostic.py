
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

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
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

# cache array columns
COLS = [
    "class_id", "score", "xmin", "ymin", "xmax", "ymax",
    "tile_x", "tile_y", "valid_w", "valid_h",
    "local_xmin", "local_ymin", "local_xmax", "local_ymax",
]


@dataclass
class Detection:
    class_id: int
    score: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float


def parse_args():
    p = argparse.ArgumentParser(
        description="Cache Val candidates once on GPU, then analyze remaining FN on CPU."
    )
    p.add_argument(
        "--model",
        type=Path,
        default=Path(
            "runs/rareos/"
            "yolo26m_tiles1280_rareos_v1_e80_b6_seed2026/"
            "weights/best.pt"
        ),
    )
    p.add_argument(
        "--images",
        type=Path,
        default=Path("datasets/yolo_split/images/val"),
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=Path("datasets/yolo_split/labels/val"),
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("results/fn_analysis/cache"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fn_analysis"),
    )
    p.add_argument("--tile-size", type=int, default=1280)
    p.add_argument("--stride", type=int, default=768)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--conf", type=float, default=1e-5)
    p.add_argument("--tile-iou", type=float, default=0.60)
    p.add_argument("--global-iou", type=float, default=0.90)
    p.add_argument("--max-det", type=int, default=1000)
    p.add_argument("--match-iou", type=float, default=0.50)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--half", action="store_true")
    p.add_argument(
        "--build-cache",
        action="store_true",
        help="Run YOLO tiled inference and save pre-global-NMS candidates.",
    )
    p.add_argument(
        "--analyze",
        action="store_true",
        help="Analyze cached candidates. GPU is not required.",
    )
    p.add_argument(
        "--edge-margin",
        type=float,
        default=16.0,
        help="Pixels used for tile-edge diagnostics.",
    )
    return p.parse_args()


def get_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area_a = max(0.0, (box[2] - box[0]) * (box[3] - box[1]))
    area_b = np.maximum(
        0.0,
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
    )
    union = area_a + area_b - inter
    return np.divide(
        inter,
        union,
        out=np.zeros_like(inter, dtype=np.float32),
        where=union > 0,
    )


def read_yolo_gt(label_path: Path, width: int, height: int) -> list[Detection]:
    gt = []
    if not label_path.exists():
        return gt
    text = label_path.read_text(encoding="utf-8").strip()
    if not text:
        return gt

    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        class_id = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])
        xc *= width
        yc *= height
        bw *= width
        bh *= height
        gt.append(
            Detection(
                class_id=class_id,
                score=1.0,
                xmin=xc - bw / 2,
                ymin=yc - bh / 2,
                xmax=xc + bw / 2,
                ymax=yc + bh / 2,
            )
        )
    return gt


def class_aware_nms_indices(arr: np.ndarray, iou_threshold: float) -> np.ndarray:
    if len(arr) == 0:
        return np.empty((0,), dtype=np.int64)

    kept = []
    class_ids = arr[:, 0].astype(np.int32)

    for class_id in np.unique(class_ids):
        idx = np.flatnonzero(class_ids == class_id)
        scores = arr[idx, 1]
        boxes = arr[idx, 2:6]
        order = np.argsort(scores)[::-1]

        while order.size:
            current_local = int(order[0])
            kept.append(int(idx[current_local]))
            if order.size == 1:
                break

            rest = order[1:]
            ious = box_iou_one_to_many(
                boxes[current_local],
                boxes[rest],
            )
            order = rest[ious <= iou_threshold]

    return np.asarray(kept, dtype=np.int64)


def match_predictions(
    predictions: np.ndarray,
    gt: list[Detection],
    match_iou: float,
):
    matched_gt_global = set()
    matched_pred_global = set()

    gt_by_class = {}
    for gi, g in enumerate(gt):
        gt_by_class.setdefault(g.class_id, []).append(gi)

    pred_classes = (
        predictions[:, 0].astype(np.int32)
        if len(predictions)
        else np.empty((0,), dtype=np.int32)
    )

    for class_id in range(len(CLASS_NAMES)):
        pred_idx = np.flatnonzero(pred_classes == class_id)
        if len(pred_idx):
            pred_idx = pred_idx[
                np.argsort(predictions[pred_idx, 1])[::-1]
            ]
        gt_idx = gt_by_class.get(class_id, [])
        if not gt_idx:
            continue

        gt_boxes = np.asarray(
            [
                [gt[i].xmin, gt[i].ymin, gt[i].xmax, gt[i].ymax]
                for i in gt_idx
            ],
            dtype=np.float32,
        )
        matched_local = set()

        for pi in pred_idx:
            box = predictions[pi, 2:6]
            ious = box_iou_one_to_many(box, gt_boxes)
            order = np.argsort(ious)[::-1]

            for local_gi in order:
                if ious[local_gi] < match_iou:
                    break
                local_gi = int(local_gi)
                if local_gi in matched_local:
                    continue
                matched_local.add(local_gi)
                matched_gt_global.add(gt_idx[local_gi])
                matched_pred_global.add(int(pi))
                break

    unmatched_gt = [
        i for i in range(len(gt))
        if i not in matched_gt_global
    ]
    tp = len(matched_gt_global)
    fn = len(unmatched_gt)
    fp = len(predictions) - len(matched_pred_global)
    return tp, fp, fn, unmatched_gt, matched_gt_global, matched_pred_global


def fully_contained_in_any_tile(
    gt: Detection,
    width: int,
    height: int,
    tile_size: int,
    stride: int,
    margin: float,
) -> bool:
    xs = get_starts(width, tile_size, stride)
    ys = get_starts(height, tile_size, stride)

    for y in ys:
        for x in xs:
            right = min(width, x + tile_size)
            bottom = min(height, y + tile_size)

            left_m = x + (margin if x > 0 else 0.0)
            top_m = y + (margin if y > 0 else 0.0)
            right_m = right - (margin if right < width else 0.0)
            bottom_m = bottom - (margin if bottom < height else 0.0)

            if (
                gt.xmin >= left_m
                and gt.ymin >= top_m
                and gt.xmax <= right_m
                and gt.ymax <= bottom_m
            ):
                return True
    return False


def candidate_edge_flag(row: np.ndarray, margin: float) -> bool:
    valid_w = float(row[8])
    valid_h = float(row[9])
    lx1, ly1, lx2, ly2 = map(float, row[10:14])
    distances = [
        lx1,
        ly1,
        valid_w - lx2,
        valid_h - ly2,
    ]
    return min(distances) <= margin


def best_candidate(
    arr: np.ndarray,
    gt: Detection,
    same_class_only: bool,
):
    if len(arr) == 0:
        return None

    sub_idx = np.arange(len(arr))
    if same_class_only:
        sub_idx = sub_idx[arr[:, 0].astype(np.int32) == gt.class_id]
    if len(sub_idx) == 0:
        return None

    gt_box = np.asarray(
        [gt.xmin, gt.ymin, gt.xmax, gt.ymax],
        dtype=np.float32,
    )
    ious = box_iou_one_to_many(gt_box, arr[sub_idx, 2:6])
    best_local = int(np.argmax(ious))
    best_idx = int(sub_idx[best_local])
    return best_idx, float(ious[best_local]), arr[best_idx]


def build_cache(args):
    import torch
    from ultralytics import YOLO

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA 当前不可用。请切换到有卡模式后再执行 --build-cache。"
        )

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))

    image_paths = sorted(
        p for p in args.images.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError(f"No images found: {args.images}")

    manifest = {
        "model": str(args.model),
        "images": str(args.images),
        "labels": str(args.labels),
        "tile_size": args.tile_size,
        "stride": args.stride,
        "batch": args.batch,
        "conf": args.conf,
        "tile_iou": args.tile_iou,
        "max_det": args.max_det,
        "columns": COLS,
        "images_count": len(image_paths),
        "items": [],
    }

    start_time = time.time()
    total_candidates = 0

    for image_index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read: {image_path}")

        height, width = image.shape[:2]
        xs = get_starts(width, args.tile_size, args.stride)
        ys = get_starts(height, args.tile_size, args.stride)

        tiles = []
        metas = []

        for y in ys:
            for x in xs:
                crop = image[
                    y:min(height, y + args.tile_size),
                    x:min(width, x + args.tile_size),
                ]
                valid_h, valid_w = crop.shape[:2]

                if valid_h != args.tile_size or valid_w != args.tile_size:
                    padded = np.zeros(
                        (args.tile_size, args.tile_size, 3),
                        dtype=np.uint8,
                    )
                    padded[:valid_h, :valid_w] = crop
                    crop = padded

                tiles.append(crop)
                metas.append((x, y, valid_w, valid_h))

        rows = []

        for start in range(0, len(tiles), args.batch):
            batch_tiles = tiles[start:start + args.batch]
            batch_metas = metas[start:start + args.batch]

            results = model.predict(
                batch_tiles,
                imgsz=args.tile_size,
                conf=args.conf,
                iou=args.tile_iou,
                max_det=args.max_det,
                device=args.device,
                half=args.half,
                verbose=False,
            )

            for result, meta in zip(results, batch_metas):
                tile_x, tile_y, valid_w, valid_h = meta
                if result.boxes is None or len(result.boxes) == 0:
                    continue

                xyxy = result.boxes.xyxy.detach().cpu().numpy()
                cls = result.boxes.cls.detach().cpu().numpy().astype(int)
                scores = result.boxes.conf.detach().cpu().numpy()

                for box, cid, score in zip(xyxy, cls, scores):
                    x1, y1, x2, y2 = map(float, box)
                    x1 = float(np.clip(x1, 0, valid_w))
                    x2 = float(np.clip(x2, 0, valid_w))
                    y1 = float(np.clip(y1, 0, valid_h))
                    y2 = float(np.clip(y2, 0, valid_h))
                    if x2 <= x1 or y2 <= y1:
                        continue

                    rows.append([
                        int(cid),
                        float(score),
                        x1 + tile_x,
                        y1 + tile_y,
                        x2 + tile_x,
                        y2 + tile_y,
                        tile_x,
                        tile_y,
                        valid_w,
                        valid_h,
                        x1,
                        y1,
                        x2,
                        y2,
                    ])

        arr = np.asarray(rows, dtype=np.float32)
        if arr.size == 0:
            arr = np.empty((0, len(COLS)), dtype=np.float32)

        cache_path = args.cache_dir / f"{image_path.stem}.npz"
        np.savez(
            cache_path,
            candidates=arr,
            image_shape=np.asarray([height, width], dtype=np.int32),
        )

        total_candidates += len(arr)
        manifest["items"].append({
            "image_name": image_path.name,
            "cache_file": cache_path.name,
            "height": height,
            "width": width,
            "candidate_count": int(len(arr)),
        })

        if image_index % 25 == 0 or image_index == len(image_paths):
            elapsed = (time.time() - start_time) / 60
            print(
                f"{image_index}/{len(image_paths)} "
                f"candidates={total_candidates:,} "
                f"elapsed={elapsed:.2f} min"
            )

    manifest["total_candidates"] = total_candidates
    manifest["elapsed_minutes"] = (time.time() - start_time) / 60

    (args.cache_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("Cache saved:", args.cache_dir)
    print("Images     :", len(image_paths))
    print("Candidates :", f"{total_candidates:,}")


def analyze_cache(args):
    manifest_path = args.cache_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Cache manifest not found: {manifest_path}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # Prevent accidentally analyzing a cache generated with different tiling/conf.
    checks = {
        "tile_size": args.tile_size,
        "stride": args.stride,
        "conf": args.conf,
        "tile_iou": args.tile_iou,
        "max_det": args.max_det,
    }
    for key, expected in checks.items():
        actual = manifest.get(key)
        if isinstance(expected, float):
            ok = actual is not None and np.isclose(float(actual), expected)
        else:
            ok = actual == expected
        if not ok:
            raise RuntimeError(
                f"Cache mismatch for {key}: cache={actual}, requested={expected}"
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fn_rows = []
    total_tp = total_fp = total_fn = 0
    class_tp = Counter()
    class_fn = Counter()
    failure_counts = Counter()

    items = manifest["items"]
    start_time = time.time()

    for item_index, item in enumerate(items, start=1):
        image_name = item["image_name"]
        image_path = args.images / image_name
        label_path = args.labels / f"{Path(image_name).stem}.txt"

        cache = np.load(args.cache_dir / item["cache_file"])
        pre = cache["candidates"].astype(np.float32, copy=False)
        height, width = map(int, cache["image_shape"])

        gt = read_yolo_gt(label_path, width, height)
        keep_idx = class_aware_nms_indices(pre, args.global_iou)
        post = pre[keep_idx] if len(keep_idx) else np.empty((0, len(COLS)), dtype=np.float32)

        tp, fp, fn, unmatched_gt, matched_gt, _ = match_predictions(
            post,
            gt,
            args.match_iou,
        )
        total_tp += tp
        total_fp += fp
        total_fn += fn

        for gi, g in enumerate(gt):
            if gi in matched_gt:
                class_tp[g.class_id] += 1

        for gi in unmatched_gt:
            g = gt[gi]
            class_fn[g.class_id] += 1

            same_pre = best_candidate(pre, g, True)
            any_pre = best_candidate(pre, g, False)
            same_post = best_candidate(post, g, True)
            any_post = best_candidate(post, g, False)

            def unpack(best):
                if best is None:
                    return {
                        "idx": -1,
                        "iou": 0.0,
                        "class_id": -1,
                        "score": 0.0,
                        "edge": False,
                    }
                idx, iou, row = best
                return {
                    "idx": idx,
                    "iou": iou,
                    "class_id": int(row[0]),
                    "score": float(row[1]),
                    "edge": candidate_edge_flag(row, args.edge_margin),
                }

            sp = unpack(same_pre)
            ap = unpack(any_pre)
            sf = unpack(same_post)
            af = unpack(any_post)

            gt_fully_contained = fully_contained_in_any_tile(
                g,
                width,
                height,
                args.tile_size,
                args.stride,
                args.edge_margin,
            )

            if sp["idx"] < 0:
                if ap["iou"] >= args.match_iou and ap["class_id"] != g.class_id:
                    failure_type = "class_confusion"
                else:
                    failure_type = "no_same_class_candidate"
            elif sp["iou"] < args.match_iou:
                if ap["iou"] >= args.match_iou and ap["class_id"] != g.class_id:
                    failure_type = "class_confusion"
                elif (not gt_fully_contained) or sp["edge"]:
                    failure_type = "localization_or_tile_fragmentation"
                else:
                    failure_type = "localization_failure"
            elif sf["iou"] < args.match_iou:
                failure_type = "global_nms_suppression"
            else:
                failure_type = "matching_competition"

            failure_counts[failure_type] += 1

            gw = g.xmax - g.xmin
            gh = g.ymax - g.ymin
            aspect = gw / gh if gh > 0 else 0.0
            elongation = max(gw / gh, gh / gw) if gw > 0 and gh > 0 else 0.0

            fn_rows.append({
                "image_name": image_name,
                "gt_index": gi,
                "class_id": g.class_id,
                "class_name": CLASS_NAMES[g.class_id],
                "gt_xmin": g.xmin,
                "gt_ymin": g.ymin,
                "gt_xmax": g.xmax,
                "gt_ymax": g.ymax,
                "gt_width": gw,
                "gt_height": gh,
                "gt_area": gw * gh,
                "aspect_w_over_h": aspect,
                "elongation": elongation,
                "gt_fully_contained_in_tile": gt_fully_contained,
                "best_same_pre_iou": sp["iou"],
                "best_same_pre_score": sp["score"],
                "best_same_pre_near_tile_edge": sp["edge"],
                "best_any_pre_iou": ap["iou"],
                "best_any_pre_class_id": ap["class_id"],
                "best_any_pre_class_name": (
                    CLASS_NAMES[ap["class_id"]]
                    if 0 <= ap["class_id"] < len(CLASS_NAMES)
                    else ""
                ),
                "best_any_pre_score": ap["score"],
                "best_same_post_iou": sf["iou"],
                "best_same_post_score": sf["score"],
                "best_any_post_iou": af["iou"],
                "best_any_post_class_id": af["class_id"],
                "best_any_post_score": af["score"],
                "failure_type": failure_type,
            })

        if item_index % 25 == 0 or item_index == len(items):
            recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
            print(
                f"{item_index}/{len(items)} "
                f"TP={total_tp} FN={total_fn} Recall={recall:.4f}"
            )

    recall = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0.0
    precision = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0.0

    fn_csv = args.output_dir / "fn_cases.csv"
    if fn_rows:
        with fn_csv.open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(fn_rows[0].keys()))
            writer.writeheader()
            writer.writerows(fn_rows)
    else:
        fn_csv.write_text("", encoding="utf-8")

    summary_rows = []
    for class_id, class_name in enumerate(CLASS_NAMES):
        tp = class_tp[class_id]
        fn = class_fn[class_id]
        summary_rows.append({
            "class": class_name,
            "tp": tp,
            "fn": fn,
            "recall": tp / (tp + fn) if tp + fn else 0.0,
        })

    with (args.output_dir / "class_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class", "tp", "fn", "recall"],
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    with (args.output_dir / "failure_summary.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["failure_type", "count"],
        )
        writer.writeheader()
        for failure_type, count in failure_counts.most_common():
            writer.writerow({
                "failure_type": failure_type,
                "count": count,
            })

    report = [
        "# Remaining FN Diagnostic",
        "",
        f"- TP: {total_tp}",
        f"- FP: {total_fp}",
        f"- FN: {total_fn}",
        f"- Recall: {recall:.6f}",
        f"- Precision: {precision:.6f}",
        f"- ScoreLike: {recall * 100:.2f}",
        "",
        "## Failure types",
        "",
    ]
    for failure_type, count in failure_counts.most_common():
        report.append(f"- {failure_type}: {count}")

    report += ["", "## FN by class", ""]
    for row in summary_rows:
        if row["fn"]:
            report.append(
                f"- {row['class']}: FN={row['fn']}, Recall={row['recall']:.4f}"
            )

    (args.output_dir / "summary.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    elapsed = (time.time() - start_time) / 60
    print()
    print("===== Overall =====")
    print("TP       :", total_tp)
    print("FP       :", total_fp)
    print("FN       :", total_fn)
    print("Recall   :", f"{recall:.6f}")
    print("Precision:", f"{precision:.6f}")
    print("ScoreLike:", f"{recall * 100:.2f}")
    print("Elapsed  :", f"{elapsed:.2f} min")
    print("Saved    :", args.output_dir)

    if total_tp != 797 or total_fn != 48:
        print()
        print(
            "WARNING: expected yesterday's best local result TP=797, FN=48. "
            "If this differs, compare cache parameters and environment."
        )


def main():
    args = parse_args()
    if not args.build_cache and not args.analyze:
        raise SystemExit("Choose at least one: --build-cache and/or --analyze")

    if args.build_cache:
        build_cache(args)
    if args.analyze:
        analyze_cache(args)


if __name__ == "__main__":
    main()
