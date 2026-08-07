from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
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

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


@dataclass
class TileMeta:
    image_name: str
    image_width: int
    image_height: int
    tile_x: int
    tile_y: int
    valid_width: int
    valid_height: int


@dataclass
class Detection:
    class_id: int
    score: float
    xmin: float
    ymin: float
    xmax: float
    ymax: float


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run tiled inference on the official steel-defect "
            "test set and generate submission.json."
        )
    )

    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/"
            "runs/baseline/"
            "yolo26m_tiles1280_e80_b6_seed2026/"
            "weights/best.pt"
        ),
    )

    parser.add_argument(
        "--test-dir",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/raw/data/test"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/"
            "submissions/"
            "baseline1_e73_conf001_nms050.json"
        ),
    )

    parser.add_argument(
        "--summary",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/"
            "submissions/"
            "baseline1_e73_conf001_nms050_summary.csv"
        ),
    )

    parser.add_argument(
        "--tile-size",
        type=int,
        default=1280,
    )

    parser.add_argument(
        "--stride",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--batch",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.01,
        help=(
            "切片级最低置信度。第一次提交使用较低值，"
            "保留 mAP 排序所需低分预测。"
        ),
    )

    parser.add_argument(
        "--tile-iou",
        type=float,
        default=0.60,
        help="YOLO 对单个切片执行 NMS 的 IoU。",
    )

    parser.add_argument(
        "--global-iou",
        type=float,
        default=0.50,
        help="映射回原图后执行 class-aware NMS 的 IoU。",
    )

    parser.add_argument(
        "--max-det",
        type=int,
        default=300,
    )

    parser.add_argument(
        "--device",
        type=str,
        default="0",
    )

    parser.add_argument(
        "--half",
        action="store_true",
        help="GPU 推理使用 FP16。",
    )

    return parser.parse_args()


def get_starts(
    length: int,
    tile_size: int,
    stride: int,
) -> list[int]:
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


def make_tile(
    image: np.ndarray,
    tile_x: int,
    tile_y: int,
    tile_size: int,
) -> tuple[np.ndarray, int, int]:
    image_height, image_width = image.shape[:2]

    right = min(
        image_width,
        tile_x + tile_size,
    )

    bottom = min(
        image_height,
        tile_y + tile_size,
    )

    crop = image[
        tile_y:bottom,
        tile_x:right,
    ]

    valid_height, valid_width = (
        crop.shape[:2]
    )

    if (
        valid_width == tile_size
        and valid_height == tile_size
    ):
        return crop.copy(), valid_width, valid_height

    padded = np.zeros(
        (tile_size, tile_size, 3),
        dtype=np.uint8,
    )

    padded[
        :valid_height,
        :valid_width,
    ] = crop

    return (
        padded,
        valid_width,
        valid_height,
    )


def iou_one_to_many(
    box: np.ndarray,
    boxes: np.ndarray,
) -> np.ndarray:
    x1 = np.maximum(
        box[0],
        boxes[:, 0],
    )

    y1 = np.maximum(
        box[1],
        boxes[:, 1],
    )

    x2 = np.minimum(
        box[2],
        boxes[:, 2],
    )

    y2 = np.minimum(
        box[3],
        boxes[:, 3],
    )

    intersection_width = np.maximum(
        0.0,
        x2 - x1,
    )

    intersection_height = np.maximum(
        0.0,
        y2 - y1,
    )

    intersection = (
        intersection_width
        * intersection_height
    )

    box_area = max(
        0.0,
        (box[2] - box[0])
        * (box[3] - box[1]),
    )

    boxes_area = np.maximum(
        0.0,
        (
            boxes[:, 2]
            - boxes[:, 0]
        )
        * (
            boxes[:, 3]
            - boxes[:, 1]
        ),
    )

    union = (
        box_area
        + boxes_area
        - intersection
    )

    return intersection / np.maximum(
        union,
        1e-12,
    )


def numpy_nms(
    detections: list[Detection],
    iou_threshold: float,
) -> list[Detection]:
    if not detections:
        return []

    boxes = np.asarray(
        [
            [
                det.xmin,
                det.ymin,
                det.xmax,
                det.ymax,
            ]
            for det in detections
        ],
        dtype=np.float32,
    )

    scores = np.asarray(
        [
            det.score
            for det in detections
        ],
        dtype=np.float32,
    )

    order = np.argsort(
        scores
    )[::-1]

    keep: list[int] = []

    while order.size > 0:
        current = int(
            order[0]
        )

        keep.append(current)

        if order.size == 1:
            break

        remaining = order[1:]

        ious = iou_one_to_many(
            boxes[current],
            boxes[remaining],
        )

        order = remaining[
            ious <= iou_threshold
        ]

    return [
        detections[index]
        for index in keep
    ]


def global_class_aware_nms(
    detections: list[Detection],
    iou_threshold: float,
) -> list[Detection]:
    by_class: dict[
        int,
        list[Detection],
    ] = defaultdict(list)

    for detection in detections:
        by_class[
            detection.class_id
        ].append(detection)

    merged: list[Detection] = []

    for class_id in sorted(
        by_class
    ):
        merged.extend(
            numpy_nms(
                by_class[class_id],
                iou_threshold,
            )
        )

    merged.sort(
        key=lambda det: det.score,
        reverse=True,
    )

    return merged


def flush_batch(
    model: YOLO,
    batch_images: list[np.ndarray],
    batch_meta: list[TileMeta],
    detections_by_image: dict[
        str,
        list[Detection],
    ],
    args,
):
    if not batch_images:
        return

    results = model.predict(
        source=batch_images,
        imgsz=args.tile_size,
        batch=args.batch,
        conf=args.conf,
        iou=args.tile_iou,
        max_det=args.max_det,
        device=args.device,
        half=args.half,
        verbose=False,
    )

    if len(results) != len(
        batch_meta
    ):
        raise RuntimeError(
            "预测结果数量与 tile metadata 数量不一致"
        )

    for result, meta in zip(
        results,
        batch_meta,
    ):
        boxes = result.boxes

        if boxes is None:
            continue

        if len(boxes) == 0:
            continue

        xyxy = (
            boxes.xyxy
            .detach()
            .cpu()
            .numpy()
        )

        scores = (
            boxes.conf
            .detach()
            .cpu()
            .numpy()
        )

        classes = (
            boxes.cls
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

        for (
            tile_box,
            score,
            class_id,
        ) in zip(
            xyxy,
            scores,
            classes,
        ):
            if (
                class_id < 0
                or class_id
                >= len(CLASS_NAMES)
            ):
                continue

            xmin = float(
                max(
                    0.0,
                    min(
                        tile_box[0],
                        meta.valid_width,
                    ),
                )
            )

            ymin = float(
                max(
                    0.0,
                    min(
                        tile_box[1],
                        meta.valid_height,
                    ),
                )
            )

            xmax = float(
                max(
                    0.0,
                    min(
                        tile_box[2],
                        meta.valid_width,
                    ),
                )
            )

            ymax = float(
                max(
                    0.0,
                    min(
                        tile_box[3],
                        meta.valid_height,
                    ),
                )
            )

            if (
                xmax <= xmin
                or ymax <= ymin
            ):
                continue

            global_xmin = (
                xmin
                + meta.tile_x
            )

            global_ymin = (
                ymin
                + meta.tile_y
            )

            global_xmax = (
                xmax
                + meta.tile_x
            )

            global_ymax = (
                ymax
                + meta.tile_y
            )

            global_xmin = max(
                0.0,
                min(
                    global_xmin,
                    meta.image_width,
                ),
            )

            global_ymin = max(
                0.0,
                min(
                    global_ymin,
                    meta.image_height,
                ),
            )

            global_xmax = max(
                0.0,
                min(
                    global_xmax,
                    meta.image_width,
                ),
            )

            global_ymax = max(
                0.0,
                min(
                    global_ymax,
                    meta.image_height,
                ),
            )

            if (
                global_xmax
                <= global_xmin
                or global_ymax
                <= global_ymin
            ):
                continue

            detections_by_image[
                meta.image_name
            ].append(
                Detection(
                    class_id=class_id,
                    score=float(score),
                    xmin=global_xmin,
                    ymin=global_ymin,
                    xmax=global_xmax,
                    ymax=global_ymax,
                )
            )


def main():
    args = parse_args()

    args.model = (
        args.model.resolve()
    )

    args.test_dir = (
        args.test_dir.resolve()
    )

    args.output = (
        args.output.resolve()
    )

    args.summary = (
        args.summary.resolve()
    )

    if not args.model.exists():
        raise FileNotFoundError(
            f"模型不存在：{args.model}"
        )

    if not args.test_dir.exists():
        raise FileNotFoundError(
            f"测试集目录不存在：{args.test_dir}"
        )

    if args.device != "cpu":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA 当前不可用。"
                "请切换到有卡模式后再执行正式推理。"
            )

    test_images = sorted(
        path
        for path
        in args.test_dir.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in IMAGE_EXTENSIONS
    )

    if not test_images:
        raise RuntimeError(
            f"测试目录中没有图片：{args.test_dir}"
        )

    print(
        "========== Baseline-1 Official Test Inference =========="
    )

    print(
        "Model:",
        args.model,
    )

    print(
        "Test dir:",
        args.test_dir,
    )

    print(
        "Test images:",
        len(test_images),
    )

    print(
        "Tile:",
        args.tile_size,
    )

    print(
        "Stride:",
        args.stride,
    )

    print(
        "Overlap:",
        args.tile_size
        - args.stride,
    )

    print(
        "Tile conf:",
        args.conf,
    )

    print(
        "Tile NMS IoU:",
        args.tile_iou,
    )

    print(
        "Global NMS IoU:",
        args.global_iou,
    )

    print(
        "Batch:",
        args.batch,
    )

    model = YOLO(
        str(args.model)
    )

    detections_by_image: dict[
        str,
        list[Detection],
    ] = defaultdict(list)

    image_shapes: dict[
        str,
        tuple[int, int],
    ] = {}

    batch_images: list[
        np.ndarray
    ] = []

    batch_meta: list[
        TileMeta
    ] = []

    total_tiles = 0

    start_time = time.time()

    for image_index, image_path in enumerate(
        test_images,
        start=1,
    ):
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise RuntimeError(
                f"无法读取测试图片：{image_path}"
            )

        image_height, image_width = (
            image.shape[:2]
        )

        image_shapes[
            image_path.name
        ] = (
            image_width,
            image_height,
        )

        x_starts = get_starts(
            image_width,
            args.tile_size,
            args.stride,
        )

        y_starts = get_starts(
            image_height,
            args.tile_size,
            args.stride,
        )

        for tile_y in y_starts:
            for tile_x in x_starts:
                (
                    tile,
                    valid_width,
                    valid_height,
                ) = make_tile(
                    image,
                    tile_x,
                    tile_y,
                    args.tile_size,
                )

                batch_images.append(
                    tile
                )

                batch_meta.append(
                    TileMeta(
                        image_name=image_path.name,
                        image_width=image_width,
                        image_height=image_height,
                        tile_x=tile_x,
                        tile_y=tile_y,
                        valid_width=valid_width,
                        valid_height=valid_height,
                    )
                )

                total_tiles += 1

                if (
                    len(batch_images)
                    >= args.batch
                ):
                    flush_batch(
                        model,
                        batch_images,
                        batch_meta,
                        detections_by_image,
                        args,
                    )

                    batch_images.clear()
                    batch_meta.clear()

        if (
            image_index % 50 == 0
            or image_index
            == len(test_images)
        ):
            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"Processed "
                f"{image_index}/"
                f"{len(test_images)} images, "
                f"tiles={total_tiles}, "
                f"elapsed="
                f"{elapsed / 60:.2f} min"
            )

    flush_batch(
        model,
        batch_images,
        batch_meta,
        detections_by_image,
        args,
    )

    batch_images.clear()
    batch_meta.clear()

    print()
    print(
        "========== Global class-aware NMS =========="
    )

    submission: list[
        dict[str, object]
    ] = []

    summary_rows: list[
        dict[str, object]
    ] = []

    class_counter: Counter[
        str
    ] = Counter()

    raw_count = 0
    final_count = 0
    images_with_detection = 0

    for image_path in test_images:
        image_name = (
            image_path.name
        )

        image_width, image_height = (
            image_shapes[
                image_name
            ]
        )

        raw_detections = (
            detections_by_image[
                image_name
            ]
        )

        raw_count += len(
            raw_detections
        )

        final_detections = (
            global_class_aware_nms(
                raw_detections,
                args.global_iou,
            )
        )

        final_count += len(
            final_detections
        )

        if final_detections:
            images_with_detection += 1

        for detection in (
            final_detections
        ):
            # 官方要求绝对像素坐标
            xmin = int(
                math.floor(
                    detection.xmin
                )
            )

            ymin = int(
                math.floor(
                    detection.ymin
                )
            )

            xmax = int(
                math.ceil(
                    detection.xmax
                )
            )

            ymax = int(
                math.ceil(
                    detection.ymax
                )
            )

            xmin = max(
                0,
                min(
                    xmin,
                    image_width - 1,
                ),
            )

            ymin = max(
                0,
                min(
                    ymin,
                    image_height - 1,
                ),
            )

            xmax = max(
                xmin + 1,
                min(
                    xmax,
                    image_width,
                ),
            )

            ymax = max(
                ymin + 1,
                min(
                    ymax,
                    image_height,
                ),
            )

            class_name = (
                CLASS_NAMES[
                    detection.class_id
                ]
            )

            score = round(
                float(
                    detection.score
                ),
                6,
            )

            submission.append({
                "image_id": (
                    image_name
                ),
                "category_name": (
                    class_name
                ),
                "bbox": [
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                ],
                "score": score,
            })

            class_counter[
                class_name
            ] += 1

        summary_rows.append({
            "image_id": image_name,
            "width": image_width,
            "height": image_height,
            "raw_detection_count": (
                len(
                    raw_detections
                )
            ),
            "final_detection_count": (
                len(
                    final_detections
                )
            ),
        })

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.summary.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(
            submission,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    with args.summary.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "image_id",
                "width",
                "height",
                "raw_detection_count",
                "final_detection_count",
            ],
        )

        writer.writeheader()
        writer.writerows(
            summary_rows
        )

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print(
        "========== Submission generated =========="
    )

    print(
        "Test images:",
        len(test_images),
    )

    print(
        "Total tiles:",
        total_tiles,
    )

    print(
        "Raw mapped detections:",
        raw_count,
    )

    print(
        "Final detections:",
        final_count,
    )

    print(
        "Images with detections:",
        images_with_detection,
    )

    print(
        "Images without detections:",
        len(test_images)
        - images_with_detection,
    )

    print()
    print(
        "Detections by class:"
    )

    for class_name in (
        CLASS_NAMES
    ):
        print(
            f"  {class_name:<18}"
            f"{class_counter[class_name]}"
        )

    print()
    print(
        "Submission:",
        args.output,
    )

    print(
        "Summary:",
        args.summary,
    )

    print(
        "JSON size:",
        f"{args.output.stat().st_size / 1024 / 1024:.2f} MB",
    )

    print(
        "Elapsed:",
        f"{elapsed / 60:.2f} min",
    )


if __name__ == "__main__":
    main()
