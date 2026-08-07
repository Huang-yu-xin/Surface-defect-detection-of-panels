from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


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


@dataclass
class Detection:
    class_id: int
    score: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model", type=Path, required=True)

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

    p.add_argument("--tile-size", type=int, default=1280)
    p.add_argument("--stride", type=int, default=1024)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--conf", type=float, default=0.0001)
    p.add_argument("--tile-iou", type=float, default=0.60)
    p.add_argument("--global-iou", type=float, default=0.80)
    p.add_argument("--max-det", type=int, default=1000)
    p.add_argument("--match-iou", type=float, default=0.50)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--half", action="store_true")

    p.add_argument(
        "--output",
        type=Path,
        default=Path("results/highrecall_val.csv"),
    )

    return p.parse_args()


def get_starts(length: int, tile_size: int, stride: int):
    if length <= tile_size:
        return [0]

    starts = list(
        range(
            0,
            length - tile_size + 1,
            stride,
        )
    )

    final_start = length - tile_size

    if starts[-1] != final_start:
        starts.append(final_start)

    return starts


def box_iou_one_to_many(box, boxes):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter_w = np.maximum(0.0, x2 - x1)
    inter_h = np.maximum(0.0, y2 - y1)
    inter = inter_w * inter_h

    area_a = max(
        0.0,
        (box[2] - box[0]) * (box[3] - box[1]),
    )

    area_b = np.maximum(
        0.0,
        (boxes[:, 2] - boxes[:, 0])
        * (boxes[:, 3] - boxes[:, 1]),
    )

    union = area_a + area_b - inter

    return np.divide(
        inter,
        union,
        out=np.zeros_like(inter),
        where=union > 0,
    )


def class_aware_nms(detections, iou_threshold):
    by_class = {}

    for det in detections:
        by_class.setdefault(det.class_id, []).append(det)

    kept = []

    for class_id, class_dets in by_class.items():
        class_dets.sort(
            key=lambda d: d.score,
            reverse=True,
        )

        boxes = np.array(
            [
                [
                    d.xmin,
                    d.ymin,
                    d.xmax,
                    d.ymax,
                ]
                for d in class_dets
            ],
            dtype=np.float32,
        )

        scores = np.array(
            [d.score for d in class_dets],
            dtype=np.float32,
        )

        order = scores.argsort()[::-1]

        while len(order):
            current = order[0]

            kept.append(class_dets[current])

            if len(order) == 1:
                break

            rest = order[1:]

            ious = box_iou_one_to_many(
                boxes[current],
                boxes[rest],
            )

            order = rest[ious <= iou_threshold]

    return kept


def read_yolo_gt(label_path, width, height):
    gt = []

    if not label_path.exists():
        return gt

    text = label_path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return gt

    for line in text.splitlines():
        parts = line.split()

        if len(parts) < 5:
            continue

        class_id = int(float(parts[0]))
        xc, yc, bw, bh = map(
            float,
            parts[1:5],
        )

        xc *= width
        yc *= height
        bw *= width
        bh *= height

        xmin = xc - bw / 2
        ymin = yc - bh / 2
        xmax = xc + bw / 2
        ymax = yc + bh / 2

        gt.append(
            Detection(
                class_id=class_id,
                score=1.0,
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )
        )

    return gt


def predict_one_image(
    model,
    image,
    tile_size,
    stride,
    batch,
    conf,
    tile_iou,
    max_det,
    device,
    half,
):
    height, width = image.shape[:2]

    xs = get_starts(
        width,
        tile_size,
        stride,
    )

    ys = get_starts(
        height,
        tile_size,
        stride,
    )

    tiles = []
    metas = []

    for y in ys:
        for x in xs:
            crop = image[
                y:y + tile_size,
                x:x + tile_size,
            ]

            h, w = crop.shape[:2]

            if h != tile_size or w != tile_size:
                padded = np.zeros(
                    (tile_size, tile_size, 3),
                    dtype=np.uint8,
                )
                padded[:h, :w] = crop
                crop = padded

            tiles.append(crop)
            metas.append((x, y, w, h))

    detections = []

    for start in range(
        0,
        len(tiles),
        batch,
    ):
        batch_tiles = tiles[
            start:start + batch
        ]

        batch_metas = metas[
            start:start + batch
        ]

        results = model.predict(
            batch_tiles,
            imgsz=tile_size,
            conf=conf,
            iou=tile_iou,
            max_det=max_det,
            device=device,
            half=half,
            verbose=False,
        )

        for result, meta in zip(
            results,
            batch_metas,
        ):
            offset_x, offset_y, valid_w, valid_h = meta

            if result.boxes is None:
                continue

            boxes = result.boxes

            xyxy = (
                boxes.xyxy
                .detach()
                .cpu()
                .numpy()
            )

            cls = (
                boxes.cls
                .detach()
                .cpu()
                .numpy()
                .astype(int)
            )

            scores = (
                boxes.conf
                .detach()
                .cpu()
                .numpy()
            )

            for box, cid, score in zip(
                xyxy,
                cls,
                scores,
            ):
                x1, y1, x2, y2 = box

                x1 = float(
                    np.clip(
                        x1,
                        0,
                        valid_w,
                    )
                )

                x2 = float(
                    np.clip(
                        x2,
                        0,
                        valid_w,
                    )
                )

                y1 = float(
                    np.clip(
                        y1,
                        0,
                        valid_h,
                    )
                )

                y2 = float(
                    np.clip(
                        y2,
                        0,
                        valid_h,
                    )
                )

                if x2 <= x1 or y2 <= y1:
                    continue

                detections.append(
                    Detection(
                        class_id=int(cid),
                        score=float(score),
                        xmin=x1 + offset_x,
                        ymin=y1 + offset_y,
                        xmax=x2 + offset_x,
                        ymax=y2 + offset_y,
                    )
                )

    return detections


def match_predictions(
    predictions,
    gt,
    match_iou,
):
    tp = 0
    fp = 0
    fn = 0

    tp_by_class = Counter()
    fp_by_class = Counter()
    fn_by_class = Counter()

    for class_id in range(len(CLASS_NAMES)):
        pred_cls = [
            p
            for p in predictions
            if p.class_id == class_id
        ]

        gt_cls = [
            g
            for g in gt
            if g.class_id == class_id
        ]

        pred_cls.sort(
            key=lambda x: x.score,
            reverse=True,
        )

        matched_gt = set()

        gt_boxes = np.array(
            [
                [
                    g.xmin,
                    g.ymin,
                    g.xmax,
                    g.ymax,
                ]
                for g in gt_cls
            ],
            dtype=np.float32,
        )

        for pred in pred_cls:
            if len(gt_boxes) == 0:
                fp += 1
                fp_by_class[class_id] += 1
                continue

            box = np.array(
                [
                    pred.xmin,
                    pred.ymin,
                    pred.xmax,
                    pred.ymax,
                ],
                dtype=np.float32,
            )

            ious = box_iou_one_to_many(
                box,
                gt_boxes,
            )

            order = np.argsort(ious)[::-1]

            matched = False

            for gt_index in order:
                if ious[gt_index] < match_iou:
                    break

                if int(gt_index) in matched_gt:
                    continue

                matched_gt.add(int(gt_index))

                tp += 1
                tp_by_class[class_id] += 1
                matched = True
                break

            if not matched:
                fp += 1
                fp_by_class[class_id] += 1

        missed = len(gt_cls) - len(matched_gt)

        fn += missed
        fn_by_class[class_id] += missed

    return (
        tp,
        fp,
        fn,
        tp_by_class,
        fp_by_class,
        fn_by_class,
    )


def main():
    args = parse_args()

    model = YOLO(str(args.model))

    image_paths = sorted(
        p
        for p in args.images.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )

    print("Images:", len(image_paths))
    print("Model :", args.model)
    print(
        "Config:",
        f"conf={args.conf}",
        f"tile_iou={args.tile_iou}",
        f"global_iou={args.global_iou}",
        f"stride={args.stride}",
        f"max_det={args.max_det}",
    )

    total_tp = 0
    total_fp = 0
    total_fn = 0

    tp_by_class = Counter()
    fp_by_class = Counter()
    fn_by_class = Counter()

    start_time = time.time()

    for i, image_path in enumerate(
        image_paths,
        start=1,
    ):
        image = cv2.imread(
            str(image_path)
        )

        if image is None:
            raise RuntimeError(
                f"Cannot read: {image_path}"
            )

        height, width = image.shape[:2]

        label_path = (
            args.labels
            / f"{image_path.stem}.txt"
        )

        gt = read_yolo_gt(
            label_path,
            width,
            height,
        )

        raw_predictions = predict_one_image(
            model=model,
            image=image,
            tile_size=args.tile_size,
            stride=args.stride,
            batch=args.batch,
            conf=args.conf,
            tile_iou=args.tile_iou,
            max_det=args.max_det,
            device=args.device,
            half=args.half,
        )

        predictions = class_aware_nms(
            raw_predictions,
            args.global_iou,
        )

        (
            tp,
            fp,
            fn,
            tp_c,
            fp_c,
            fn_c,
        ) = match_predictions(
            predictions,
            gt,
            args.match_iou,
        )

        total_tp += tp
        total_fp += fp
        total_fn += fn

        tp_by_class.update(tp_c)
        fp_by_class.update(fp_c)
        fn_by_class.update(fn_c)

        if (
            i % 25 == 0
            or i == len(image_paths)
        ):
            recall = (
                total_tp
                / (total_tp + total_fn)
                if total_tp + total_fn
                else 0.0
            )

            print(
                f"{i}/{len(image_paths)} "
                f"TP={total_tp} "
                f"FN={total_fn} "
                f"Recall={recall:.4f}"
            )

    recall = (
        total_tp
        / (total_tp + total_fn)
        if total_tp + total_fn
        else 0.0
    )

    precision = (
        total_tp
        / (total_tp + total_fp)
        if total_tp + total_fp
        else 0.0
    )

    print()
    print("===== Overall =====")
    print("TP       :", total_tp)
    print("FP       :", total_fp)
    print("FN       :", total_fn)
    print("Recall   :", f"{recall:.6f}")
    print("Precision:", f"{precision:.6f}")
    print(
        "ScoreLike:",
        f"{recall * 100:.2f}",
    )

    print()
    print("===== Per class =====")

    rows = []

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        tp = tp_by_class[class_id]
        fp = fp_by_class[class_id]
        fn = fn_by_class[class_id]

        class_recall = (
            tp / (tp + fn)
            if tp + fn
            else 0.0
        )

        print(
            f"{class_name:<18}"
            f" TP={tp:<5}"
            f" FN={fn:<5}"
            f" FP={fp:<7}"
            f" R={class_recall:.4f}"
        )

        rows.append(
            {
                "class": class_name,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "recall": class_recall,
            }
        )

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with args.output.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class",
                "tp",
                "fp",
                "fn",
                "recall",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    elapsed = (
        time.time() - start_time
    ) / 60

    print()
    print("Saved:", args.output)
    print("Elapsed:", f"{elapsed:.2f} min")


if __name__ == "__main__":
    main()
