from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import time
from collections import Counter, defaultdict
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

LONG_CLASSES = {
    "zonglie",
    "qilie",
    "huashang",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


@dataclass(frozen=True)
class Box:
    class_id: int
    class_name: str
    xmin: float
    ymin: float
    xmax: float
    ymax: float

    @property
    def width(self) -> float:
        return self.xmax - self.xmin

    @property
    def height(self) -> float:
        return self.ymax - self.ymin

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    @property
    def aspect_ratio(self) -> float:
        short_side = max(1e-6, min(self.width, self.height))
        long_side = max(self.width, self.height)
        return long_side / short_side


@dataclass(frozen=True)
class Sample:
    split: str
    stem: str
    image_path: Path
    label_path: Path
    source_is_empty: bool


@dataclass(frozen=True)
class NegativeCandidate:
    split: str
    source_stem: str
    source_image: Path
    source_label: Path
    source_is_empty: bool
    tile_x: int
    tile_y: int
    black_ratio: float

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (
            self.split,
            self.source_stem,
            self.tile_x,
            self.tile_y,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the full 1280x1280 overlapping-tile dataset "
            "from the grouped Train/Val split."
        )
    )

    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/datasets/yolo_split"
        ),
        help="阶段 2 生成的原图级 YOLO Train/Val 数据集。",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/datasets/tiles_1280_full"
        ),
        help="全量切片数据集输出目录。",
    )

    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/metadata/tiles_1280_full"
        ),
        help="切片索引和统计结果输出目录。",
    )

    parser.add_argument(
        "--manifests-dir",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/splits"
        ),
        help="训练和验证切片清单输出目录。",
    )

    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/"
            "configs/steel_tiles_1280.yaml"
        ),
        help="Ultralytics 数据集 YAML 输出路径。",
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
        "--normal-visible-ratio",
        type=float,
        default=0.35,
        help="普通目标被切断后至少保留的面积比例。",
    )

    parser.add_argument(
        "--long-visible-ratio",
        type=float,
        default=0.20,
        help="长条目标被切断后至少保留的面积比例。",
    )

    parser.add_argument(
        "--long-aspect-ratio",
        type=float,
        default=8.0,
        help="长宽比达到该值时自动采用长条目标规则。",
    )

    parser.add_argument(
        "--min-box-pixels",
        type=float,
        default=2.0,
        help="切片中保留目标的最小宽度和高度。",
    )

    parser.add_argument(
        "--train-negative-ratio",
        type=float,
        default=0.40,
        help="Train 负切片数相对于正切片数的比例。",
    )

    parser.add_argument(
        "--val-negative-ratio",
        type=float,
        default=0.40,
        help="Val 负切片数相对于正切片数的比例。",
    )

    parser.add_argument(
        "--max-negative-candidates-per-image",
        type=int,
        default=1,
        help="每张原图最多进入候选池的干净负切片数量。",
    )

    parser.add_argument(
        "--black-pixel-threshold",
        type=int,
        default=12,
        help="灰度小于等于该值的像素视为近黑像素。",
    )

    parser.add_argument(
        "--max-black-ratio",
        type=float,
        default=0.65,
        help="无目标切片近黑像素比例超过该值时不作为负样本。",
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=92,
        help="切片 JPEG 保存质量。",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=35.0,
        help="运行前要求数据盘至少剩余的空间。",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="删除已有输出后重新生成。",
    )

    return parser.parse_args()


def prepare_directory(
    path: Path,
    overwrite: bool,
) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"输出目录非空：{path}\n"
                "确认后添加 --overwrite 重新生成。"
            )

        shutil.rmtree(path)

    path.mkdir(parents=True, exist_ok=True)


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def load_samples(
    source_root: Path,
    split: str,
) -> list[Sample]:
    images_dir = source_root / "images" / split
    labels_dir = source_root / "labels" / split

    if not images_dir.exists():
        raise FileNotFoundError(
            f"图片目录不存在：{images_dir}"
        )

    if not labels_dir.exists():
        raise FileNotFoundError(
            f"标签目录不存在：{labels_dir}"
        )

    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    samples: list[Sample] = []

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            raise FileNotFoundError(
                f"缺少同名标签：{label_path}"
            )

        source_is_empty = (
            label_path.read_text(
                encoding="utf-8"
            ).strip()
            == ""
        )

        samples.append(
            Sample(
                split=split,
                stem=image_path.stem,
                image_path=image_path,
                label_path=label_path,
                source_is_empty=source_is_empty,
            )
        )

    image_stems = {
        sample.stem for sample in samples
    }

    label_stems = {
        path.stem
        for path in labels_dir.glob("*.txt")
    }

    extra_labels = sorted(
        label_stems - image_stems
    )

    if extra_labels:
        raise RuntimeError(
            "发现没有同名图片的标签，例如："
            + ", ".join(extra_labels[:10])
        )

    return samples


def parse_yolo_label(
    label_path: Path,
    image_width: int,
    image_height: int,
) -> list[Box]:
    text = label_path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return []

    boxes: list[Box] = []

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        parts = line.strip().split()

        if len(parts) != 5:
            raise ValueError(
                f"{label_path}:{line_number} "
                f"应包含 5 列，实际为 {len(parts)}"
            )

        class_id = int(parts[0])
        center_x = float(parts[1])
        center_y = float(parts[2])
        normalized_width = float(parts[3])
        normalized_height = float(parts[4])

        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise ValueError(
                f"{label_path}:{line_number} "
                f"类别编号越界：{class_id}"
            )

        if not all(
            0.0 <= value <= 1.0
            for value in (
                center_x,
                center_y,
                normalized_width,
                normalized_height,
            )
        ):
            raise ValueError(
                f"{label_path}:{line_number} "
                "归一化坐标不在 [0, 1] 内"
            )

        if (
            normalized_width <= 0
            or normalized_height <= 0
        ):
            raise ValueError(
                f"{label_path}:{line_number} "
                "目标宽高必须大于 0"
            )

        box_width = normalized_width * image_width
        box_height = normalized_height * image_height

        xmin = center_x * image_width - box_width / 2.0
        ymin = center_y * image_height - box_height / 2.0
        xmax = center_x * image_width + box_width / 2.0
        ymax = center_y * image_height + box_height / 2.0

        xmin = max(
            0.0,
            min(xmin, float(image_width)),
        )
        ymin = max(
            0.0,
            min(ymin, float(image_height)),
        )
        xmax = max(
            0.0,
            min(xmax, float(image_width)),
        )
        ymax = max(
            0.0,
            min(ymax, float(image_height)),
        )

        if xmax <= xmin or ymax <= ymin:
            raise ValueError(
                f"{label_path}:{line_number} "
                "反归一化后目标框无效"
            )

        boxes.append(
            Box(
                class_id=class_id,
                class_name=CLASS_NAMES[class_id],
                xmin=xmin,
                ymin=ymin,
                xmax=xmax,
                ymax=ymax,
            )
        )

    return boxes


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

    last_start = length - tile_size

    if starts[-1] != last_start:
        starts.append(last_start)

    return starts


def intersect_box(
    box: Box,
    tile_x: int,
    tile_y: int,
    tile_right: int,
    tile_bottom: int,
) -> tuple[float, float, float, float] | None:
    xmin = max(box.xmin, float(tile_x))
    ymin = max(box.ymin, float(tile_y))
    xmax = min(box.xmax, float(tile_right))
    ymax = min(box.ymax, float(tile_bottom))

    if xmax <= xmin or ymax <= ymin:
        return None

    return xmin, ymin, xmax, ymax


def convert_intersection_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    tile_x: int,
    tile_y: int,
    tile_size: int,
) -> tuple[float, float, float, float]:
    local_xmin = xmin - tile_x
    local_ymin = ymin - tile_y
    local_xmax = xmax - tile_x
    local_ymax = ymax - tile_y

    width = local_xmax - local_xmin
    height = local_ymax - local_ymin

    center_x = (local_xmin + local_xmax) / 2.0
    center_y = (local_ymin + local_ymax) / 2.0

    return (
        center_x / tile_size,
        center_y / tile_size,
        width / tile_size,
        height / tile_size,
    )


def calculate_black_ratio(
    crop: np.ndarray,
    black_pixel_threshold: int,
) -> float:
    if crop.size == 0:
        return 1.0

    gray = cv2.cvtColor(
        crop,
        cv2.COLOR_BGR2GRAY,
    )

    return float(
        np.mean(gray <= black_pixel_threshold)
    )


def make_tile_name(
    source_stem: str,
    tile_x: int,
    tile_y: int,
) -> str:
    return (
        f"{source_stem}"
        f"__x{tile_x:04d}"
        f"_y{tile_y:04d}"
    )


def build_tile_image(
    source_image: np.ndarray,
    tile_x: int,
    tile_y: int,
    tile_size: int,
) -> np.ndarray:
    image_height, image_width = source_image.shape[:2]

    crop_right = min(
        tile_x + tile_size,
        image_width,
    )

    crop_bottom = min(
        tile_y + tile_size,
        image_height,
    )

    crop = source_image[
        tile_y:crop_bottom,
        tile_x:crop_right,
    ]

    if (
        crop.shape[0] == tile_size
        and crop.shape[1] == tile_size
    ):
        return crop

    tile = np.zeros(
        (tile_size, tile_size, 3),
        dtype=np.uint8,
    )

    tile[
        0:crop.shape[0],
        0:crop.shape[1],
    ] = crop

    return tile


def save_tile(
    source_image: np.ndarray,
    split: str,
    source_stem: str,
    tile_x: int,
    tile_y: int,
    tile_size: int,
    label_lines: list[str],
    output_root: Path,
    jpeg_quality: int,
) -> tuple[Path, Path, int]:
    tile_name = make_tile_name(
        source_stem,
        tile_x,
        tile_y,
    )

    image_output_path = (
        output_root
        / "images"
        / split
        / f"{tile_name}.jpg"
    )

    label_output_path = (
        output_root
        / "labels"
        / split
        / f"{tile_name}.txt"
    )

    image_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    label_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    tile = build_tile_image(
        source_image,
        tile_x,
        tile_y,
        tile_size,
    )

    success = cv2.imwrite(
        str(image_output_path),
        tile,
        [
            cv2.IMWRITE_JPEG_QUALITY,
            jpeg_quality,
        ],
    )

    if not success:
        raise RuntimeError(
            f"无法保存切片图片：{image_output_path}"
        )

    label_output_path.write_text(
        "\n".join(label_lines)
        + ("\n" if label_lines else ""),
        encoding="utf-8",
    )

    written_bytes = (
        image_output_path.stat().st_size
        + label_output_path.stat().st_size
    )

    return (
        image_output_path,
        label_output_path,
        written_bytes,
    )


def process_source_image(
    sample: Sample,
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    list[dict[str, object]],
    list[NegativeCandidate],
    dict[str, object],
    Counter[str],
    Counter[str],
    int,
]:
    source_image = cv2.imread(
        str(sample.image_path),
        cv2.IMREAD_COLOR,
    )

    if source_image is None:
        raise RuntimeError(
            f"无法读取图片：{sample.image_path}"
        )

    image_height, image_width = source_image.shape[:2]

    boxes = parse_yolo_label(
        sample.label_path,
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

    positive_tile_rows: list[
        dict[str, object]
    ] = []

    ambiguous_rows: list[
        dict[str, object]
    ] = []

    filtered_black_rows: list[
        dict[str, object]
    ] = []

    local_negative_candidates: list[
        NegativeCandidate
    ] = []

    class_tile_counter: Counter[str] = Counter()
    class_fragment_counter: Counter[str] = Counter()

    positive_tile_count = 0
    ambiguous_tile_count = 0
    black_filtered_count = 0
    ignored_tiny_intersection_tiles = 0
    written_bytes = 0

    for tile_y in y_starts:
        for tile_x in x_starts:
            tile_right = min(
                tile_x + args.tile_size,
                image_width,
            )

            tile_bottom = min(
                tile_y + args.tile_size,
                image_height,
            )

            label_lines: list[str] = []
            kept_class_names: list[str] = []

            intersected_box_count = 0
            meaningful_dropped_count = 0
            tiny_intersection_count = 0

            for object_index, box in enumerate(boxes):
                intersection = intersect_box(
                    box,
                    tile_x,
                    tile_y,
                    tile_right,
                    tile_bottom,
                )

                if intersection is None:
                    continue

                intersected_box_count += 1

                ixmin, iymin, ixmax, iymax = intersection

                intersection_width = ixmax - ixmin
                intersection_height = iymax - iymin
                intersection_area = (
                    intersection_width
                    * intersection_height
                )

                visible_ratio = (
                    intersection_area
                    / max(1e-12, box.area)
                )

                use_long_rule = (
                    box.class_name in LONG_CLASSES
                    or box.aspect_ratio
                    >= args.long_aspect_ratio
                )

                required_visible_ratio = (
                    args.long_visible_ratio
                    if use_long_rule
                    else args.normal_visible_ratio
                )

                large_enough = (
                    intersection_width
                    >= args.min_box_pixels
                    and intersection_height
                    >= args.min_box_pixels
                )

                keep_box = (
                    large_enough
                    and visible_ratio
                    >= required_visible_ratio
                )

                if not keep_box:
                    if large_enough:
                        meaningful_dropped_count += 1

                        ambiguous_rows.append({
                            "split": sample.split,
                            "source_image": sample.image_path.name,
                            "tile_x": tile_x,
                            "tile_y": tile_y,
                            "object_index": object_index,
                            "class_name": box.class_name,
                            "original_xmin": box.xmin,
                            "original_ymin": box.ymin,
                            "original_xmax": box.xmax,
                            "original_ymax": box.ymax,
                            "intersection_width": (
                                intersection_width
                            ),
                            "intersection_height": (
                                intersection_height
                            ),
                            "visible_ratio": visible_ratio,
                            "required_visible_ratio": (
                                required_visible_ratio
                            ),
                            "reason": (
                                "visible_ratio_below_threshold"
                            ),
                        })
                    else:
                        tiny_intersection_count += 1

                    continue

                (
                    yolo_x,
                    yolo_y,
                    yolo_width,
                    yolo_height,
                ) = convert_intersection_to_yolo(
                    ixmin,
                    iymin,
                    ixmax,
                    iymax,
                    tile_x,
                    tile_y,
                    args.tile_size,
                )

                normalized_values = (
                    yolo_x,
                    yolo_y,
                    yolo_width,
                    yolo_height,
                )

                if not all(
                    0.0 <= value <= 1.0
                    for value in normalized_values
                ):
                    raise RuntimeError(
                        "切片标签归一化越界："
                        f"{sample.image_path.name}, "
                        f"tile=({tile_x},{tile_y}), "
                        f"box={normalized_values}"
                    )

                label_lines.append(
                    f"{box.class_id} "
                    f"{yolo_x:.8f} "
                    f"{yolo_y:.8f} "
                    f"{yolo_width:.8f} "
                    f"{yolo_height:.8f}"
                )

                kept_class_names.append(
                    box.class_name
                )

                class_fragment_counter[
                    box.class_name
                ] += 1

            # 一个切片中只要包含尺寸不小、但可见比例不足的缺陷片段，
            # 就跳过整个切片，防止把该缺陷片段作为无标注背景训练。
            if meaningful_dropped_count > 0:
                ambiguous_tile_count += 1
                continue

            if label_lines:
                (
                    image_output_path,
                    label_output_path,
                    current_bytes,
                ) = save_tile(
                    source_image=source_image,
                    split=sample.split,
                    source_stem=sample.stem,
                    tile_x=tile_x,
                    tile_y=tile_y,
                    tile_size=args.tile_size,
                    label_lines=label_lines,
                    output_root=args.output_root,
                    jpeg_quality=args.jpeg_quality,
                )

                written_bytes += current_bytes
                positive_tile_count += 1

                unique_classes = sorted(
                    set(kept_class_names)
                )

                for class_name in unique_classes:
                    class_tile_counter[
                        class_name
                    ] += 1

                positive_tile_rows.append({
                    "split": sample.split,
                    "tile_name": image_output_path.name,
                    "label_name": label_output_path.name,
                    "source_image": sample.image_path.name,
                    "source_is_empty": (
                        sample.source_is_empty
                    ),
                    "tile_x": tile_x,
                    "tile_y": tile_y,
                    "tile_size": args.tile_size,
                    "tile_type": "positive",
                    "kept_box_count": len(label_lines),
                    "source_box_count": len(boxes),
                    "intersected_box_count": (
                        intersected_box_count
                    ),
                    "tiny_intersection_count": (
                        tiny_intersection_count
                    ),
                    "black_ratio": "",
                    "classes": ",".join(
                        unique_classes
                    ),
                })

                continue

            # 与任何目标都完全无交集，才允许进入负样本候选池。
            if intersected_box_count == 0:
                crop = source_image[
                    tile_y:tile_bottom,
                    tile_x:tile_right,
                ]

                black_ratio = calculate_black_ratio(
                    crop,
                    args.black_pixel_threshold,
                )

                if black_ratio > args.max_black_ratio:
                    black_filtered_count += 1

                    filtered_black_rows.append({
                        "split": sample.split,
                        "source_image": (
                            sample.image_path.name
                        ),
                        "source_is_empty": (
                            sample.source_is_empty
                        ),
                        "tile_x": tile_x,
                        "tile_y": tile_y,
                        "black_ratio": black_ratio,
                        "max_black_ratio": (
                            args.max_black_ratio
                        ),
                        "reason": "black_ratio_too_high",
                    })

                    continue

                local_negative_candidates.append(
                    NegativeCandidate(
                        split=sample.split,
                        source_stem=sample.stem,
                        source_image=sample.image_path,
                        source_label=sample.label_path,
                        source_is_empty=(
                            sample.source_is_empty
                        ),
                        tile_x=tile_x,
                        tile_y=tile_y,
                        black_ratio=black_ratio,
                    )
                )

            elif tiny_intersection_count > 0:
                # 只有不到 min_box_pixels 的极小边缘片段。
                # 不保存为正样本，也不当成负样本。
                ignored_tiny_intersection_tiles += 1

    # 每张图最多向全局负样本池贡献固定数量的候选，
    # 避免某些大面积空背景图片支配负样本分布。
    candidate_rng = random.Random(
        f"{args.seed}|{sample.split}|{sample.stem}"
    )

    local_negative_candidates.sort(
        key=lambda item: (
            item.tile_y,
            item.tile_x,
        )
    )

    if (
        len(local_negative_candidates)
        > args.max_negative_candidates_per_image
    ):
        local_negative_candidates = (
            candidate_rng.sample(
                local_negative_candidates,
                args.max_negative_candidates_per_image,
            )
        )

    source_summary = {
        "split": sample.split,
        "source_image": sample.image_path.name,
        "source_is_empty": sample.source_is_empty,
        "image_width": image_width,
        "image_height": image_height,
        "source_box_count": len(boxes),
        "grid_tile_count": (
            len(x_starts) * len(y_starts)
        ),
        "positive_tile_count": positive_tile_count,
        "negative_candidate_count": len(
            local_negative_candidates
        ),
        "ambiguous_tile_count": ambiguous_tile_count,
        "black_filtered_tile_count": (
            black_filtered_count
        ),
        "ignored_tiny_intersection_tiles": (
            ignored_tiny_intersection_tiles
        ),
    }

    return (
        positive_tile_rows,
        ambiguous_rows,
        filtered_black_rows,
        local_negative_candidates,
        source_summary,
        class_tile_counter,
        class_fragment_counter,
        written_bytes,
    )


def select_negative_candidates(
    candidates: list[NegativeCandidate],
    positive_tile_count: int,
    negative_ratio: float,
    seed: int,
    split: str,
) -> tuple[
    list[NegativeCandidate],
    set[tuple[str, str, int, int]],
]:
    target_count = min(
        len(candidates),
        round(positive_tile_count * negative_ratio),
    )

    ordered_candidates = sorted(
        candidates,
        key=lambda item: (
            item.source_image.name,
            item.tile_y,
            item.tile_x,
        ),
    )

    rng = random.Random(
        f"{seed}|negative-selection|{split}"
    )

    rng.shuffle(ordered_candidates)

    selected = ordered_candidates[:target_count]

    selected_keys = {
        item.key for item in selected
    }

    return selected, selected_keys


def save_selected_negative_tiles(
    selected_candidates: list[NegativeCandidate],
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, object]],
    int,
]:
    candidates_by_source: dict[
        Path,
        list[NegativeCandidate],
    ] = defaultdict(list)

    for candidate in selected_candidates:
        candidates_by_source[
            candidate.source_image
        ].append(candidate)

    tile_rows: list[dict[str, object]] = []
    written_bytes = 0

    for source_image_path in sorted(
        candidates_by_source,
        key=lambda path: path.name,
    ):
        source_image = cv2.imread(
            str(source_image_path),
            cv2.IMREAD_COLOR,
        )

        if source_image is None:
            raise RuntimeError(
                f"无法读取负样本原图："
                f"{source_image_path}"
            )

        for candidate in candidates_by_source[
            source_image_path
        ]:
            (
                image_output_path,
                label_output_path,
                current_bytes,
            ) = save_tile(
                source_image=source_image,
                split=candidate.split,
                source_stem=candidate.source_stem,
                tile_x=candidate.tile_x,
                tile_y=candidate.tile_y,
                tile_size=args.tile_size,
                label_lines=[],
                output_root=args.output_root,
                jpeg_quality=args.jpeg_quality,
            )

            written_bytes += current_bytes

            tile_rows.append({
                "split": candidate.split,
                "tile_name": image_output_path.name,
                "label_name": label_output_path.name,
                "source_image": (
                    candidate.source_image.name
                ),
                "source_is_empty": (
                    candidate.source_is_empty
                ),
                "tile_x": candidate.tile_x,
                "tile_y": candidate.tile_y,
                "tile_size": args.tile_size,
                "tile_type": "negative",
                "kept_box_count": 0,
                "source_box_count": (
                    0
                    if candidate.source_is_empty
                    else ""
                ),
                "intersected_box_count": 0,
                "tiny_intersection_count": 0,
                "black_ratio": candidate.black_ratio,
                "classes": "",
            })

    return tile_rows, written_bytes


def write_yaml(
    config_path: Path,
    output_root: Path,
) -> None:
    config_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    lines = [
        f"path: {output_root}",
        "train: images/train",
        "val: images/val",
        "",
        "names:",
    ]

    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        lines.append(
            f"  {class_id}: {class_name}"
        )

    config_path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def write_manifests(
    output_root: Path,
    manifests_dir: Path,
) -> None:
    manifests_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for split in ["train", "val"]:
        image_paths = sorted(
            (
                output_root
                / "images"
                / split
            ).glob("*.jpg")
        )

        manifest_path = (
            manifests_dir
            / f"{split}_tiles_1280.txt"
        )

        manifest_path.write_text(
            "\n".join(
                str(path.absolute())
                for path in image_paths
            )
            + ("\n" if image_paths else ""),
            encoding="utf-8",
        )


def validate_output(
    output_root: Path,
) -> dict[str, object]:
    result: dict[str, object] = {
        "splits": {},
    }

    train_tile_stems: set[str] = set()
    val_tile_stems: set[str] = set()

    for split in ["train", "val"]:
        images_dir = output_root / "images" / split
        labels_dir = output_root / "labels" / split

        image_paths = sorted(
            images_dir.glob("*.jpg")
        )

        label_paths = sorted(
            labels_dir.glob("*.txt")
        )

        image_stems = {
            path.stem for path in image_paths
        }

        label_stems = {
            path.stem for path in label_paths
        }

        missing_labels = sorted(
            image_stems - label_stems
        )

        missing_images = sorted(
            label_stems - image_stems
        )

        if missing_labels:
            raise RuntimeError(
                f"{split} 有切片缺少标签，例如："
                + ", ".join(missing_labels[:10])
            )

        if missing_images:
            raise RuntimeError(
                f"{split} 有标签缺少切片，例如："
                + ", ".join(missing_images[:10])
            )

        empty_label_count = 0
        box_count = 0
        class_box_counter: Counter[str] = Counter()

        for label_path in label_paths:
            text = label_path.read_text(
                encoding="utf-8"
            ).strip()

            if not text:
                empty_label_count += 1
                continue

            for line_number, line in enumerate(
                text.splitlines(),
                start=1,
            ):
                parts = line.split()

                if len(parts) != 5:
                    raise RuntimeError(
                        f"{label_path}:{line_number} "
                        "列数不是 5"
                    )

                class_id = int(parts[0])
                values = [
                    float(value)
                    for value in parts[1:]
                ]

                if (
                    class_id < 0
                    or class_id >= len(CLASS_NAMES)
                ):
                    raise RuntimeError(
                        f"{label_path}:{line_number} "
                        f"类别越界：{class_id}"
                    )

                if not all(
                    0.0 <= value <= 1.0
                    for value in values
                ):
                    raise RuntimeError(
                        f"{label_path}:{line_number} "
                        f"坐标越界：{values}"
                    )

                if values[2] <= 0 or values[3] <= 0:
                    raise RuntimeError(
                        f"{label_path}:{line_number} "
                        "目标宽高不大于零"
                    )

                box_count += 1
                class_box_counter[
                    CLASS_NAMES[class_id]
                ] += 1

        result["splits"][split] = {
            "image_count": len(image_paths),
            "label_count": len(label_paths),
            "empty_label_count": empty_label_count,
            "box_count": box_count,
            "class_box_counts": dict(
                class_box_counter
            ),
            "missing_label_count": len(
                missing_labels
            ),
            "missing_image_count": len(
                missing_images
            ),
        }

        if split == "train":
            train_tile_stems = image_stems
        else:
            val_tile_stems = image_stems

    tile_overlap = (
        train_tile_stems & val_tile_stems
    )

    if tile_overlap:
        raise RuntimeError(
            "Train 和 Val 存在同名切片，例如："
            + ", ".join(sorted(tile_overlap)[:10])
        )

    result["train_val_tile_overlap_count"] = 0

    return result


def format_gb(byte_count: int) -> float:
    return byte_count / (1024 ** 3)


def main() -> None:
    args = parse_args()

    args.source_root = args.source_root.resolve()
    args.output_root = args.output_root.resolve()
    args.metadata_root = args.metadata_root.resolve()
    args.manifests_dir = (
        args.manifests_dir.resolve()
    )
    args.config_path = args.config_path.resolve()

    if args.tile_size <= 0:
        raise ValueError("tile-size 必须大于零")

    if args.stride <= 0:
        raise ValueError("stride 必须大于零")

    if args.stride > args.tile_size:
        raise ValueError(
            "stride 不能大于 tile-size"
        )

    if not 0.0 <= args.train_negative_ratio <= 2.0:
        raise ValueError(
            "train-negative-ratio 不合理"
        )

    if not 0.0 <= args.val_negative_ratio <= 2.0:
        raise ValueError(
            "val-negative-ratio 不合理"
        )

    if (
        args.max_negative_candidates_per_image
        < 1
    ):
        raise ValueError(
            "max-negative-candidates-per-image "
            "必须大于等于 1"
        )

    if not 0.0 <= args.max_black_ratio <= 1.0:
        raise ValueError(
            "max-black-ratio 必须位于 [0,1]"
        )

    output_parent = args.output_root.parent
    output_parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    disk_usage = shutil.disk_usage(
        output_parent
    )

    free_gb = disk_usage.free / (1024 ** 3)

    print("========== 阶段 4：全量重叠切片 ==========")
    print("源数据：", args.source_root)
    print("输出数据：", args.output_root)
    print("切片大小：", args.tile_size)
    print("步长：", args.stride)
    print(
        "重叠像素：",
        args.tile_size - args.stride,
    )
    print(
        "重叠比例：",
        f"{(args.tile_size - args.stride) / args.tile_size:.2%}",
    )
    print(
        "普通框可见比例：",
        args.normal_visible_ratio,
    )
    print(
        "长条框可见比例：",
        args.long_visible_ratio,
    )
    print(
        "Train 负样本比例：",
        args.train_negative_ratio,
    )
    print(
        "Val 负样本比例：",
        args.val_negative_ratio,
    )
    print(
        "近黑像素阈值：",
        args.black_pixel_threshold,
    )
    print(
        "最大近黑比例：",
        args.max_black_ratio,
    )
    print(
        "当前数据盘空闲：",
        f"{free_gb:.2f} GB",
    )

    if free_gb < args.min_free_gb:
        raise RuntimeError(
            f"数据盘空闲仅 {free_gb:.2f} GB，"
            f"少于要求的 {args.min_free_gb:.2f} GB。"
            "请先清理压缩包或扩大数据盘。"
        )

    prepare_directory(
        args.output_root,
        args.overwrite,
    )

    prepare_directory(
        args.metadata_root,
        args.overwrite,
    )

    start_time = time.time()

    all_tile_rows: list[dict[str, object]] = []
    all_ambiguous_rows: list[
        dict[str, object]
    ] = []
    all_filtered_black_rows: list[
        dict[str, object]
    ] = []
    all_negative_candidate_rows: list[
        dict[str, object]
    ] = []
    all_source_summary_rows: list[
        dict[str, object]
    ] = []

    summary: dict[str, object] = {
        "parameters": {
            "tile_size": args.tile_size,
            "stride": args.stride,
            "overlap": (
                args.tile_size - args.stride
            ),
            "overlap_ratio": (
                (args.tile_size - args.stride)
                / args.tile_size
            ),
            "normal_visible_ratio": (
                args.normal_visible_ratio
            ),
            "long_visible_ratio": (
                args.long_visible_ratio
            ),
            "long_aspect_ratio": (
                args.long_aspect_ratio
            ),
            "min_box_pixels": (
                args.min_box_pixels
            ),
            "train_negative_ratio": (
                args.train_negative_ratio
            ),
            "val_negative_ratio": (
                args.val_negative_ratio
            ),
            "max_negative_candidates_per_image": (
                args.max_negative_candidates_per_image
            ),
            "black_pixel_threshold": (
                args.black_pixel_threshold
            ),
            "max_black_ratio": (
                args.max_black_ratio
            ),
            "jpeg_quality": args.jpeg_quality,
            "seed": args.seed,
        },
        "splits": {},
    }

    total_written_bytes = 0

    for split in ["train", "val"]:
        print(f"\n========== 处理 {split} ==========")

        samples = load_samples(
            args.source_root,
            split,
        )

        print(f"原图数量：{len(samples)}")
        print(
            "空标签原图：",
            sum(
                sample.source_is_empty
                for sample in samples
            ),
        )

        split_positive_rows: list[
            dict[str, object]
        ] = []

        split_negative_candidates: list[
            NegativeCandidate
        ] = []

        split_class_tile_counter: Counter[
            str
        ] = Counter()

        split_class_fragment_counter: Counter[
            str
        ] = Counter()

        split_written_bytes = 0

        for sample_index, sample in enumerate(
            samples,
            start=1,
        ):
            (
                positive_rows,
                ambiguous_rows,
                filtered_black_rows,
                negative_candidates,
                source_summary,
                class_tile_counter,
                class_fragment_counter,
                written_bytes,
            ) = process_source_image(
                sample,
                args,
            )

            split_positive_rows.extend(
                positive_rows
            )

            all_ambiguous_rows.extend(
                ambiguous_rows
            )

            all_filtered_black_rows.extend(
                filtered_black_rows
            )

            split_negative_candidates.extend(
                negative_candidates
            )

            all_source_summary_rows.append(
                source_summary
            )

            split_class_tile_counter.update(
                class_tile_counter
            )

            split_class_fragment_counter.update(
                class_fragment_counter
            )

            split_written_bytes += written_bytes

            if (
                sample_index % 100 == 0
                or sample_index == len(samples)
            ):
                elapsed = time.time() - start_time

                print(
                    f"已处理 {sample_index}/"
                    f"{len(samples)} 原图，"
                    f"当前正切片 "
                    f"{len(split_positive_rows)}，"
                    f"负候选 "
                    f"{len(split_negative_candidates)}，"
                    f"累计耗时 "
                    f"{elapsed / 60:.1f} 分钟"
                )

        positive_tile_count = len(
            split_positive_rows
        )

        negative_ratio = (
            args.train_negative_ratio
            if split == "train"
            else args.val_negative_ratio
        )

        (
            selected_negative_candidates,
            selected_negative_keys,
        ) = select_negative_candidates(
            candidates=split_negative_candidates,
            positive_tile_count=(
                positive_tile_count
            ),
            negative_ratio=negative_ratio,
            seed=args.seed,
            split=split,
        )

        print(f"\n{split} 正切片：{positive_tile_count}")
        print(
            f"{split} 干净负候选："
            f"{len(split_negative_candidates)}"
        )
        print(
            f"{split} 选中负切片："
            f"{len(selected_negative_candidates)}"
        )

        (
            negative_tile_rows,
            negative_written_bytes,
        ) = save_selected_negative_tiles(
            selected_negative_candidates,
            args,
        )

        split_written_bytes += (
            negative_written_bytes
        )

        split_tile_rows = (
            split_positive_rows
            + negative_tile_rows
        )

        all_tile_rows.extend(
            split_tile_rows
        )

        for candidate in sorted(
            split_negative_candidates,
            key=lambda item: (
                item.source_image.name,
                item.tile_y,
                item.tile_x,
            ),
        ):
            all_negative_candidate_rows.append({
                "split": candidate.split,
                "source_image": (
                    candidate.source_image.name
                ),
                "source_is_empty": (
                    candidate.source_is_empty
                ),
                "tile_x": candidate.tile_x,
                "tile_y": candidate.tile_y,
                "black_ratio": (
                    candidate.black_ratio
                ),
                "selected": (
                    candidate.key
                    in selected_negative_keys
                ),
            })

        selected_empty_source_negatives = sum(
            candidate.source_is_empty
            for candidate in (
                selected_negative_candidates
            )
        )

        selected_nonempty_source_negatives = (
            len(selected_negative_candidates)
            - selected_empty_source_negatives
        )

        split_ambiguous_fragment_count = sum(
            row["split"] == split
            for row in all_ambiguous_rows
        )

        split_black_filtered_count = sum(
            row["split"] == split
            for row in all_filtered_black_rows
        )

        split_source_summaries = [
            row
            for row in all_source_summary_rows
            if row["split"] == split
        ]

        ambiguous_tile_count = sum(
            int(row["ambiguous_tile_count"])
            for row in split_source_summaries
        )

        ignored_tiny_tile_count = sum(
            int(
                row[
                    "ignored_tiny_intersection_tiles"
                ]
            )
            for row in split_source_summaries
        )

        summary["splits"][split] = {
            "source_image_count": len(samples),
            "source_empty_image_count": sum(
                sample.source_is_empty
                for sample in samples
            ),
            "positive_tile_count": (
                positive_tile_count
            ),
            "negative_candidate_count": len(
                split_negative_candidates
            ),
            "selected_negative_tile_count": len(
                selected_negative_candidates
            ),
            "selected_empty_source_negative_count": (
                selected_empty_source_negatives
            ),
            "selected_nonempty_source_negative_count": (
                selected_nonempty_source_negatives
            ),
            "total_tile_count": len(
                split_tile_rows
            ),
            "ambiguous_tile_count": (
                ambiguous_tile_count
            ),
            "ambiguous_fragment_count": (
                split_ambiguous_fragment_count
            ),
            "black_filtered_tile_count": (
                split_black_filtered_count
            ),
            "ignored_tiny_intersection_tile_count": (
                ignored_tiny_tile_count
            ),
            "class_positive_tile_counts": {
                class_name: (
                    split_class_tile_counter[
                        class_name
                    ]
                )
                for class_name in CLASS_NAMES
            },
            "class_box_fragment_counts": {
                class_name: (
                    split_class_fragment_counter[
                        class_name
                    ]
                )
                for class_name in CLASS_NAMES
            },
            "written_gb": format_gb(
                split_written_bytes
            ),
        }

        total_written_bytes += split_written_bytes

        print(f"\n{split} 最终切片：")
        print(
            f"  正样本："
            f"{positive_tile_count}"
        )
        print(
            f"  负样本："
            f"{len(selected_negative_candidates)}"
        )
        print(
            f"  合计："
            f"{len(split_tile_rows)}"
        )
        print(
            f"  跳过模糊切片："
            f"{ambiguous_tile_count}"
        )
        print(
            f"  过滤近黑负切片："
            f"{split_black_filtered_count}"
        )
        print(
            f"  当前写入："
            f"{format_gb(split_written_bytes):.2f} GB"
        )

        print("\n  类别正切片覆盖：")

        for class_name in CLASS_NAMES:
            print(
                f"    {class_name:<18}"
                f"{split_class_tile_counter[class_name]}"
            )

        print("\n  类别框片段数：")

        for class_name in CLASS_NAMES:
            print(
                f"    {class_name:<18}"
                f"{split_class_fragment_counter[class_name]}"
            )

    print("\n========== 写入索引和配置 ==========")

    write_csv(
        args.metadata_root / "tile_index.csv",
        all_tile_rows,
        [
            "split",
            "tile_name",
            "label_name",
            "source_image",
            "source_is_empty",
            "tile_x",
            "tile_y",
            "tile_size",
            "tile_type",
            "kept_box_count",
            "source_box_count",
            "intersected_box_count",
            "tiny_intersection_count",
            "black_ratio",
            "classes",
        ],
    )

    write_csv(
        args.metadata_root
        / "ambiguous_fragments.csv",
        all_ambiguous_rows,
        [
            "split",
            "source_image",
            "tile_x",
            "tile_y",
            "object_index",
            "class_name",
            "original_xmin",
            "original_ymin",
            "original_xmax",
            "original_ymax",
            "intersection_width",
            "intersection_height",
            "visible_ratio",
            "required_visible_ratio",
            "reason",
        ],
    )

    write_csv(
        args.metadata_root
        / "filtered_black_negatives.csv",
        all_filtered_black_rows,
        [
            "split",
            "source_image",
            "source_is_empty",
            "tile_x",
            "tile_y",
            "black_ratio",
            "max_black_ratio",
            "reason",
        ],
    )

    write_csv(
        args.metadata_root
        / "negative_candidates.csv",
        all_negative_candidate_rows,
        [
            "split",
            "source_image",
            "source_is_empty",
            "tile_x",
            "tile_y",
            "black_ratio",
            "selected",
        ],
    )

    write_csv(
        args.metadata_root
        / "source_summary.csv",
        all_source_summary_rows,
        [
            "split",
            "source_image",
            "source_is_empty",
            "image_width",
            "image_height",
            "source_box_count",
            "grid_tile_count",
            "positive_tile_count",
            "negative_candidate_count",
            "ambiguous_tile_count",
            "black_filtered_tile_count",
            "ignored_tiny_intersection_tiles",
        ],
    )

    class_summary_rows: list[
        dict[str, object]
    ] = []

    for split in ["train", "val"]:
        split_summary = summary["splits"][split]

        for class_name in CLASS_NAMES:
            class_summary_rows.append({
                "split": split,
                "class_name": class_name,
                "positive_tile_count": (
                    split_summary[
                        "class_positive_tile_counts"
                    ][class_name]
                ),
                "box_fragment_count": (
                    split_summary[
                        "class_box_fragment_counts"
                    ][class_name]
                ),
            })

    write_csv(
        args.metadata_root
        / "class_summary.csv",
        class_summary_rows,
        [
            "split",
            "class_name",
            "positive_tile_count",
            "box_fragment_count",
        ],
    )

    write_yaml(
        args.config_path,
        args.output_root,
    )

    write_manifests(
        args.output_root,
        args.manifests_dir,
    )

    validation = validate_output(
        args.output_root
    )

    elapsed_seconds = time.time() - start_time

    summary["validation"] = validation
    summary["elapsed_seconds"] = elapsed_seconds
    summary["elapsed_minutes"] = (
        elapsed_seconds / 60.0
    )
    summary["written_gb"] = format_gb(
        total_written_bytes
    )

    summary_path = (
        args.metadata_root / "summary.json"
    )

    summary_path.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\n========== 输出完整性检查 ==========")

    for split in ["train", "val"]:
        split_validation = (
            validation["splits"][split]
        )

        print(f"{split}:")
        print(
            "  图片：",
            split_validation["image_count"],
        )
        print(
            "  标签：",
            split_validation["label_count"],
        )
        print(
            "  空标签：",
            split_validation[
                "empty_label_count"
            ],
        )
        print(
            "  目标框：",
            split_validation["box_count"],
        )
        print(
            "  缺标签：",
            split_validation[
                "missing_label_count"
            ],
        )
        print(
            "  缺图片：",
            split_validation[
                "missing_image_count"
            ],
        )

    print(
        "Train/Val 同名切片：",
        validation[
            "train_val_tile_overlap_count"
        ],
    )

    final_disk_usage = shutil.disk_usage(
        output_parent
    )

    final_free_gb = (
        final_disk_usage.free / (1024 ** 3)
    )

    print("\n========== 阶段 4 完成 ==========")
    print(
        "总耗时：",
        f"{elapsed_seconds / 60:.2f} 分钟",
    )
    print(
        "本次写入约：",
        f"{format_gb(total_written_bytes):.2f} GB",
    )
    print(
        "数据盘剩余：",
        f"{final_free_gb:.2f} GB",
    )
    print("切片数据集：", args.output_root)
    print("统计目录：", args.metadata_root)
    print("数据配置：", args.config_path)
    print(
        "Train 清单：",
        args.manifests_dir
        / "train_tiles_1280.txt",
    )
    print(
        "Val 清单：",
        args.manifests_dir
        / "val_tiles_1280.txt",
    )
    print(
        "总摘要：",
        summary_path,
    )
    print("\n检查通过：")
    print("  图片和标签一一对应")
    print("  所有 YOLO 标签坐标位于 [0,1]")
    print("  Train 与 Val 切片无重名")
    print("  模糊截断目标没有被作为背景保存")


if __name__ == "__main__":
    main()
