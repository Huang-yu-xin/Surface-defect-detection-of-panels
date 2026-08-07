from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


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

CLASS_COLORS = {
    0: (0, 0, 255),
    1: (0, 255, 255),
    2: (255, 0, 255),
    3: (255, 255, 0),
    4: (0, 165, 255),
    5: (255, 0, 0),
    6: (0, 255, 0),
    7: (128, 0, 255),
    8: (255, 128, 0),
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
    def min_side(self) -> float:
        return min(self.width, self.height)

    @property
    def aspect_ratio(self) -> float:
        shorter = max(1e-6, min(self.width, self.height))
        longer = max(self.width, self.height)
        return longer / shorter


@dataclass
class Sample:
    split: str
    stem: str
    image_path: Path
    label_path: Path
    image_width: int
    image_height: int
    boxes: list[Box]

    @property
    def classes(self) -> set[int]:
        return {box.class_id for box in self.boxes}

    @property
    def is_empty(self) -> bool:
        return len(self.boxes) == 0

    @property
    def smallest_box_side(self) -> float:
        if not self.boxes:
            return float("inf")
        return min(box.min_side for box in self.boxes)

    @property
    def largest_aspect_ratio(self) -> float:
        if not self.boxes:
            return 0.0
        return max(box.aspect_ratio for box in self.boxes)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a small 1280x1280 overlapping-tile trial dataset."
    )

    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/datasets/yolo_split"
        ),
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/datasets/tile_trial_1280"
        ),
    )

    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/metadata/tile_trial_1280"
        ),
    )

    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/"
            "configs/steel_tiles_trial_1280.yaml"
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
        "--normal-visible-ratio",
        type=float,
        default=0.35,
    )

    parser.add_argument(
        "--long-visible-ratio",
        type=float,
        default=0.20,
    )

    parser.add_argument(
        "--long-aspect-ratio",
        type=float,
        default=8.0,
    )

    parser.add_argument(
        "--min-box-pixels",
        type=float,
        default=2.0,
    )

    parser.add_argument(
        "--train-per-class",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--val-per-class",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--train-empty",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--val-empty",
        type=int,
        default=6,
    )

    parser.add_argument(
        "--extreme-count",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-negative-per-image",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--visualization-count",
        type=int,
        default=36,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
    )

    return parser.parse_args()


def prepare_directory(
    directory: Path,
    overwrite: bool,
) -> None:
    if directory.exists() and any(directory.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"输出目录非空：{directory}\n"
                "确认后添加 --overwrite。"
            )

        shutil.rmtree(directory)

    directory.mkdir(parents=True, exist_ok=True)


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
                f"{label_path}:{line_number} 应为 5 列"
            )

        class_id = int(parts[0])
        center_x = float(parts[1])
        center_y = float(parts[2])
        normalized_width = float(parts[3])
        normalized_height = float(parts[4])

        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise ValueError(
                f"{label_path}:{line_number} 类别编号越界"
            )

        box_width = normalized_width * image_width
        box_height = normalized_height * image_height

        xmin = center_x * image_width - box_width / 2
        ymin = center_y * image_height - box_height / 2
        xmax = center_x * image_width + box_width / 2
        ymax = center_y * image_height + box_height / 2

        xmin = max(0.0, min(xmin, float(image_width)))
        ymin = max(0.0, min(ymin, float(image_height)))
        xmax = max(0.0, min(xmax, float(image_width)))
        ymax = max(0.0, min(ymax, float(image_height)))

        if xmax <= xmin or ymax <= ymin:
            raise ValueError(
                f"{label_path}:{line_number} 存在非法框"
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


def load_split_samples(
    source_root: Path,
    split: str,
) -> list[Sample]:
    images_dir = source_root / "images" / split
    labels_dir = source_root / "labels" / split

    if not images_dir.exists():
        raise FileNotFoundError(
            f"图片目录不存在：{images_dir}"
        )

    samples: list[Sample] = []

    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    for image_path in image_paths:
        label_path = labels_dir / f"{image_path.stem}.txt"

        if not label_path.exists():
            raise FileNotFoundError(
                f"缺少标签：{label_path}"
            )

        with Image.open(image_path) as image:
            image_width, image_height = image.size

        boxes = parse_yolo_label(
            label_path,
            image_width,
            image_height,
        )

        samples.append(
            Sample(
                split=split,
                stem=image_path.stem,
                image_path=image_path,
                label_path=label_path,
                image_width=image_width,
                image_height=image_height,
                boxes=boxes,
            )
        )

    return samples


def choose_samples(
    samples: list[Sample],
    split: str,
    per_class: int,
    empty_count: int,
    extreme_count: int,
    rng: random.Random,
) -> list[Sample]:
    by_class: dict[int, list[Sample]] = defaultdict(list)

    for sample in samples:
        for class_id in sample.classes:
            by_class[class_id].append(sample)

    selected: dict[str, Sample] = {}

    rare_targets_train = {
        2: 8,  # qilie
        3: 6,  # jiaza
        5: 6,  # huashang
    }

    rare_targets_val = {
        2: 4,
        3: 5,
        5: 5,
    }

    rare_targets = (
        rare_targets_train
        if split == "train"
        else rare_targets_val
    )

    for class_id in range(len(CLASS_NAMES)):
        candidates = list(by_class[class_id])
        rng.shuffle(candidates)

        target = max(
            per_class,
            rare_targets.get(class_id, per_class),
        )

        for sample in candidates[:target]:
            selected[sample.stem] = sample

    empty_samples = [
        sample
        for sample in samples
        if sample.is_empty
    ]

    rng.shuffle(empty_samples)

    for sample in empty_samples[:empty_count]:
        selected[sample.stem] = sample

    nonempty_samples = [
        sample
        for sample in samples
        if not sample.is_empty
    ]

    smallest_samples = sorted(
        nonempty_samples,
        key=lambda item: item.smallest_box_side,
    )[:extreme_count]

    elongated_samples = sorted(
        nonempty_samples,
        key=lambda item: item.largest_aspect_ratio,
        reverse=True,
    )[:extreme_count]

    for sample in smallest_samples:
        selected[sample.stem] = sample

    for sample in elongated_samples:
        selected[sample.stem] = sample

    return sorted(
        selected.values(),
        key=lambda item: item.image_path.name,
    )


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


def convert_tile_box_to_yolo(
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

    box_width = local_xmax - local_xmin
    box_height = local_ymax - local_ymin

    center_x = (local_xmin + local_xmax) / 2
    center_y = (local_ymin + local_ymax) / 2

    return (
        center_x / tile_size,
        center_y / tile_size,
        box_width / tile_size,
        box_height / tile_size,
    )


def save_tile(
    image: np.ndarray,
    sample: Sample,
    split: str,
    tile_x: int,
    tile_y: int,
    tile_size: int,
    label_lines: list[str],
    output_root: Path,
) -> tuple[Path, Path]:
    tile_name = (
        f"{sample.stem}"
        f"__x{tile_x:04d}"
        f"_y{tile_y:04d}"
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

    image_height, image_width = image.shape[:2]

    crop_right = min(
        tile_x + tile_size,
        image_width,
    )

    crop_bottom = min(
        tile_y + tile_size,
        image_height,
    )

    crop = image[
        tile_y:crop_bottom,
        tile_x:crop_right,
    ]

    tile = np.zeros(
        (tile_size, tile_size, 3),
        dtype=np.uint8,
    )

    crop_height, crop_width = crop.shape[:2]

    tile[
        0:crop_height,
        0:crop_width,
    ] = crop

    success = cv2.imwrite(
        str(image_output_path),
        tile,
        [cv2.IMWRITE_JPEG_QUALITY, 95],
    )

    if not success:
        raise RuntimeError(
            f"无法保存切片：{image_output_path}"
        )

    label_output_path.write_text(
        "\n".join(label_lines)
        + ("\n" if label_lines else ""),
        encoding="utf-8",
    )

    return image_output_path, label_output_path


def process_sample(
    sample: Sample,
    args: argparse.Namespace,
    rng: random.Random,
) -> tuple[
    list[dict[str, object]],
    list[dict[str, object]],
    Counter[str],
]:
    image = cv2.imread(
        str(sample.image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"无法读取图片：{sample.image_path}"
        )

    image_height, image_width = image.shape[:2]

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

    positive_candidates: list[
        tuple[int, int, list[str], list[str], int]
    ] = []

    negative_candidates: list[
        tuple[int, int]
    ] = []

    ambiguous_rows: list[dict[str, object]] = []
    class_fragment_counter: Counter[str] = Counter()

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
            classes_in_tile: list[str] = []

            intersected_count = 0
            meaningful_dropped_count = 0

            for object_index, box in enumerate(
                sample.boxes
            ):
                intersection = intersect_box(
                    box,
                    tile_x,
                    tile_y,
                    tile_right,
                    tile_bottom,
                )

                if intersection is None:
                    continue

                intersected_count += 1

                ixmin, iymin, ixmax, iymax = intersection

                intersection_width = ixmax - ixmin
                intersection_height = iymax - iymin
                intersection_area = (
                    intersection_width
                    * intersection_height
                )

                visible_ratio = (
                    intersection_area
                    / max(1e-6, box.area)
                )

                use_long_rule = (
                    box.class_name in LONG_CLASSES
                    or box.aspect_ratio
                    >= args.long_aspect_ratio
                )

                threshold = (
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

                keep = (
                    visible_ratio >= threshold
                    and large_enough
                )

                if not keep:
                    if large_enough:
                        meaningful_dropped_count += 1

                        ambiguous_rows.append({
                            "split": sample.split,
                            "source_image": sample.image_path.name,
                            "tile_x": tile_x,
                            "tile_y": tile_y,
                            "object_index": object_index,
                            "class_name": box.class_name,
                            "visible_ratio": visible_ratio,
                            "intersection_width": intersection_width,
                            "intersection_height": intersection_height,
                            "required_visible_ratio": threshold,
                            "reason": "meaningful_partial_box",
                        })

                    continue

                (
                    yolo_x,
                    yolo_y,
                    yolo_width,
                    yolo_height,
                ) = convert_tile_box_to_yolo(
                    ixmin,
                    iymin,
                    ixmax,
                    iymax,
                    tile_x,
                    tile_y,
                    args.tile_size,
                )

                label_lines.append(
                    f"{box.class_id} "
                    f"{yolo_x:.8f} "
                    f"{yolo_y:.8f} "
                    f"{yolo_width:.8f} "
                    f"{yolo_height:.8f}"
                )

                classes_in_tile.append(
                    box.class_name
                )

            # 有明显缺陷片段但无法可靠标注时，整个切片跳过，
            # 防止把缺陷当作背景训练。
            if meaningful_dropped_count > 0:
                continue

            if label_lines:
                positive_candidates.append(
                    (
                        tile_x,
                        tile_y,
                        label_lines,
                        sorted(set(classes_in_tile)),
                        intersected_count,
                    )
                )

            elif intersected_count == 0:
                negative_candidates.append(
                    (tile_x, tile_y)
                )

    selected_negative_candidates = (
        negative_candidates
        if len(negative_candidates)
        <= args.max_negative_per_image
        else rng.sample(
            negative_candidates,
            args.max_negative_per_image,
        )
    )

    tile_rows: list[dict[str, object]] = []

    for (
        tile_x,
        tile_y,
        label_lines,
        classes_in_tile,
        intersected_count,
    ) in positive_candidates:
        image_path, label_path = save_tile(
            image=image,
            sample=sample,
            split=sample.split,
            tile_x=tile_x,
            tile_y=tile_y,
            tile_size=args.tile_size,
            label_lines=label_lines,
            output_root=args.output_root,
        )

        for class_name in classes_in_tile:
            class_fragment_counter[class_name] += 1

        tile_rows.append({
            "split": sample.split,
            "tile_name": image_path.name,
            "label_name": label_path.name,
            "source_image": sample.image_path.name,
            "tile_x": tile_x,
            "tile_y": tile_y,
            "tile_size": args.tile_size,
            "tile_type": "positive",
            "kept_box_count": len(label_lines),
            "source_box_count": len(sample.boxes),
            "intersected_box_count": intersected_count,
            "classes": ",".join(classes_in_tile),
        })

    for tile_x, tile_y in selected_negative_candidates:
        image_path, label_path = save_tile(
            image=image,
            sample=sample,
            split=sample.split,
            tile_x=tile_x,
            tile_y=tile_y,
            tile_size=args.tile_size,
            label_lines=[],
            output_root=args.output_root,
        )

        tile_rows.append({
            "split": sample.split,
            "tile_name": image_path.name,
            "label_name": label_path.name,
            "source_image": sample.image_path.name,
            "tile_x": tile_x,
            "tile_y": tile_y,
            "tile_size": args.tile_size,
            "tile_type": "negative",
            "kept_box_count": 0,
            "source_box_count": len(sample.boxes),
            "intersected_box_count": 0,
            "classes": "",
        })

    return (
        tile_rows,
        ambiguous_rows,
        class_fragment_counter,
    )


def write_csv(
    path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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


def read_tile_boxes(
    image_path: Path,
    label_path: Path,
) -> tuple[np.ndarray, list[Box]]:
    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_COLOR,
    )

    if image is None:
        raise RuntimeError(
            f"无法读取切片：{image_path}"
        )

    image_height, image_width = image.shape[:2]

    boxes = parse_yolo_label(
        label_path,
        image_width,
        image_height,
    )

    return image, boxes


def draw_tile_annotations(
    image: np.ndarray,
    boxes: list[Box],
) -> np.ndarray:
    output = image.copy()

    for box in boxes:
        color = CLASS_COLORS.get(
            box.class_id,
            (0, 255, 0),
        )

        xmin = int(round(box.xmin))
        ymin = int(round(box.ymin))
        xmax = int(round(box.xmax))
        ymax = int(round(box.ymax))

        cv2.rectangle(
            output,
            (xmin, ymin),
            (xmax, ymax),
            color,
            3,
        )

        label = (
            f"{box.class_name} "
            f"{box.width:.0f}x{box.height:.0f}"
        )

        cv2.putText(
            output,
            label,
            (max(2, xmin), max(25, ymin - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
            cv2.LINE_AA,
        )

    return output


def choose_visualization_rows(
    tile_rows: list[dict[str, object]],
    count: int,
    rng: random.Random,
) -> list[dict[str, object]]:
    positive_rows = [
        row
        for row in tile_rows
        if row["tile_type"] == "positive"
    ]

    negative_rows = [
        row
        for row in tile_rows
        if row["tile_type"] == "negative"
    ]

    selected: dict[str, dict[str, object]] = {}

    for class_name in CLASS_NAMES:
        candidates = [
            row
            for row in positive_rows
            if class_name
            in str(row["classes"]).split(",")
        ]

        rng.shuffle(candidates)

        for row in candidates[:2]:
            selected[str(row["tile_name"])] = row

    rng.shuffle(negative_rows)

    for row in negative_rows[:6]:
        selected[str(row["tile_name"])] = row

    remaining = [
        row
        for row in positive_rows
        if str(row["tile_name"]) not in selected
    ]

    rng.shuffle(remaining)

    for row in remaining:
        if len(selected) >= count:
            break

        selected[str(row["tile_name"])] = row

    return list(selected.values())[:count]


def create_contact_sheet(
    split: str,
    tile_rows: list[dict[str, object]],
    args: argparse.Namespace,
    rng: random.Random,
) -> None:
    selected_rows = choose_visualization_rows(
        tile_rows,
        args.visualization_count,
        rng,
    )

    if not selected_rows:
        return

    annotated_dir = (
        args.metadata_root
        / "annotated"
        / split
    )

    annotated_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = 4
    thumb_width = 480
    image_area_height = 480
    caption_height = 48
    tile_height = image_area_height + caption_height

    row_count = math.ceil(
        len(selected_rows) / columns
    )

    sheet = np.full(
        (
            row_count * tile_height,
            columns * thumb_width,
            3,
        ),
        235,
        dtype=np.uint8,
    )

    for index, row in enumerate(selected_rows):
        image_path = (
            args.output_root
            / "images"
            / split
            / str(row["tile_name"])
        )

        label_path = (
            args.output_root
            / "labels"
            / split
            / str(row["label_name"])
        )

        image, boxes = read_tile_boxes(
            image_path,
            label_path,
        )

        annotated = draw_tile_annotations(
            image,
            boxes,
        )

        annotated_path = (
            annotated_dir
            / image_path.name
        )

        cv2.imwrite(
            str(annotated_path),
            annotated,
            [cv2.IMWRITE_JPEG_QUALITY, 94],
        )

        scale = min(
            thumb_width / annotated.shape[1],
            image_area_height / annotated.shape[0],
        )

        resized_width = max(
            1,
            round(annotated.shape[1] * scale),
        )

        resized_height = max(
            1,
            round(annotated.shape[0] * scale),
        )

        resized = cv2.resize(
            annotated,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

        row_index = index // columns
        column_index = index % columns

        base_x = column_index * thumb_width
        base_y = row_index * tile_height

        offset_x = (
            base_x
            + (thumb_width - resized_width) // 2
        )

        offset_y = (
            base_y
            + (image_area_height - resized_height) // 2
        )

        sheet[
            offset_y:offset_y + resized_height,
            offset_x:offset_x + resized_width,
        ] = resized

        caption = (
            f"{row['tile_type']} | "
            f"{row['classes'] or 'background'} | "
            f"{row['source_image']}"
        )

        cv2.putText(
            sheet,
            caption[:72],
            (
                base_x + 6,
                base_y + image_area_height + 29,
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )

        cv2.rectangle(
            sheet,
            (base_x, base_y),
            (
                base_x + thumb_width - 1,
                base_y + tile_height - 1,
            ),
            (120, 120, 120),
            1,
        )

    contact_sheet_path = (
        args.metadata_root
        / f"{split}_contact_sheet.jpg"
    )

    cv2.imwrite(
        str(contact_sheet_path),
        sheet,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )


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


def main() -> None:
    args = parse_args()

    args.source_root = args.source_root.resolve()
    args.output_root = args.output_root.resolve()
    args.metadata_root = args.metadata_root.resolve()
    args.config_path = args.config_path.resolve()

    if args.stride <= 0:
        raise ValueError("stride 必须大于零")

    if args.stride > args.tile_size:
        raise ValueError(
            "stride 不应大于 tile-size，否则切片之间会出现空隙"
        )

    overlap = args.tile_size - args.stride

    print("========== 阶段 3：小规模重叠切片 ==========")
    print("源数据：", args.source_root)
    print("切片大小：", args.tile_size)
    print("步长：", args.stride)
    print("重叠像素：", overlap)
    print(
        "重叠比例：",
        f"{overlap / args.tile_size:.2%}",
    )

    prepare_directory(
        args.output_root,
        args.overwrite,
    )

    prepare_directory(
        args.metadata_root,
        args.overwrite,
    )

    rng = random.Random(args.seed)

    split_settings = {
        "train": {
            "per_class": args.train_per_class,
            "empty_count": args.train_empty,
        },
        "val": {
            "per_class": args.val_per_class,
            "empty_count": args.val_empty,
        },
    }

    all_tile_rows: list[dict[str, object]] = []
    all_ambiguous_rows: list[dict[str, object]] = []
    selected_source_rows: list[dict[str, object]] = []

    summary: dict[str, object] = {
        "tile_size": args.tile_size,
        "stride": args.stride,
        "overlap": overlap,
        "overlap_ratio": overlap / args.tile_size,
        "seed": args.seed,
        "splits": {},
    }

    for split in ["train", "val"]:
        print(f"\n========== 加载 {split} ==========")

        samples = load_split_samples(
            args.source_root,
            split,
        )

        settings = split_settings[split]

        selected_samples = choose_samples(
            samples=samples,
            split=split,
            per_class=settings["per_class"],
            empty_count=settings["empty_count"],
            extreme_count=args.extreme_count,
            rng=rng,
        )

        print(
            f"{split} 总图片：{len(samples)}"
        )

        print(
            f"{split} 试切图片：{len(selected_samples)}"
        )

        print(
            f"{split} 试切空标签图："
            f"{sum(sample.is_empty for sample in selected_samples)}"
        )

        split_tile_rows: list[dict[str, object]] = []
        split_ambiguous_rows: list[dict[str, object]] = []
        split_class_fragments: Counter[str] = Counter()

        for sample_index, sample in enumerate(
            selected_samples,
            start=1,
        ):
            (
                tile_rows,
                ambiguous_rows,
                fragment_counter,
            ) = process_sample(
                sample,
                args,
                rng,
            )

            split_tile_rows.extend(tile_rows)
            split_ambiguous_rows.extend(
                ambiguous_rows
            )

            split_class_fragments.update(
                fragment_counter
            )

            selected_source_rows.append({
                "split": split,
                "source_image": sample.image_path.name,
                "source_label": sample.label_path.name,
                "is_empty": sample.is_empty,
                "box_count": len(sample.boxes),
                "classes": ",".join(
                    CLASS_NAMES[class_id]
                    for class_id in sorted(
                        sample.classes
                    )
                ),
                "smallest_box_side": (
                    ""
                    if sample.is_empty
                    else sample.smallest_box_side
                ),
                "largest_aspect_ratio": (
                    ""
                    if sample.is_empty
                    else sample.largest_aspect_ratio
                ),
            })

            if sample_index % 10 == 0:
                print(
                    f"已处理 {sample_index}/"
                    f"{len(selected_samples)}"
                )

        positive_tiles = sum(
            row["tile_type"] == "positive"
            for row in split_tile_rows
        )

        negative_tiles = sum(
            row["tile_type"] == "negative"
            for row in split_tile_rows
        )

        kept_fragments = sum(
            int(row["kept_box_count"])
            for row in split_tile_rows
        )

        print(f"\n{split} 生成结果：")
        print(
            f"  正样本切片：{positive_tiles}"
        )
        print(
            f"  干净负样本切片：{negative_tiles}"
        )
        print(
            f"  保存框片段：{kept_fragments}"
        )
        print(
            f"  模糊边界片段记录："
            f"{len(split_ambiguous_rows)}"
        )

        print("  类别切片覆盖：")

        for class_name in CLASS_NAMES:
            print(
                f"    {class_name:<18}"
                f"{split_class_fragments[class_name]}"
            )

        summary["splits"][split] = {
            "available_source_images": len(samples),
            "selected_source_images": len(selected_samples),
            "selected_empty_source_images": sum(
                sample.is_empty
                for sample in selected_samples
            ),
            "positive_tiles": positive_tiles,
            "negative_tiles": negative_tiles,
            "saved_tiles": len(split_tile_rows),
            "kept_box_fragments": kept_fragments,
            "ambiguous_fragment_records": len(
                split_ambiguous_rows
            ),
            "class_tile_counts": dict(
                split_class_fragments
            ),
        }

        create_contact_sheet(
            split=split,
            tile_rows=split_tile_rows,
            args=args,
            rng=rng,
        )

        all_tile_rows.extend(
            split_tile_rows
        )

        all_ambiguous_rows.extend(
            split_ambiguous_rows
        )

    write_csv(
        args.metadata_root / "tile_index.csv",
        all_tile_rows,
        [
            "split",
            "tile_name",
            "label_name",
            "source_image",
            "tile_x",
            "tile_y",
            "tile_size",
            "tile_type",
            "kept_box_count",
            "source_box_count",
            "intersected_box_count",
            "classes",
        ],
    )

    write_csv(
        args.metadata_root / "ambiguous_fragments.csv",
        all_ambiguous_rows,
        [
            "split",
            "source_image",
            "tile_x",
            "tile_y",
            "object_index",
            "class_name",
            "visible_ratio",
            "intersection_width",
            "intersection_height",
            "required_visible_ratio",
            "reason",
        ],
    )

    write_csv(
        args.metadata_root / "selected_sources.csv",
        selected_source_rows,
        [
            "split",
            "source_image",
            "source_label",
            "is_empty",
            "box_count",
            "classes",
            "smallest_box_side",
            "largest_aspect_ratio",
        ],
    )

    (
        args.metadata_root / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_yaml(
        args.config_path,
        args.output_root,
    )

    print("\n========== 阶段 3 完成 ==========")
    print("试切数据：", args.output_root)
    print("统计目录：", args.metadata_root)
    print(
        "训练拼图：",
        args.metadata_root
        / "train_contact_sheet.jpg",
    )
    print(
        "验证拼图：",
        args.metadata_root
        / "val_contact_sheet.jpg",
    )
    print(
        "切片索引：",
        args.metadata_root / "tile_index.csv",
    )
    print(
        "模糊片段：",
        args.metadata_root
        / "ambiguous_fragments.csv",
    )
    print(
        "试验配置：",
        args.config_path,
    )


if __name__ == "__main__":
    main()
