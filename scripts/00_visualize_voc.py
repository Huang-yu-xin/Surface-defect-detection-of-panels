from __future__ import annotations

import argparse
import csv
import random
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# 赛题中的 9 个类别，顺序也可作为后续 YOLO 类别顺序
CLASS_ORDER = [
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

# OpenCV 使用 BGR 颜色
CLASS_COLORS = {
    "jieba": (0, 0, 255),
    "zonglie": (0, 255, 255),
    "qilie": (255, 0, 255),
    "jiaza": (255, 255, 0),
    "yiwuyaru": (0, 165, 255),
    "huashang": (255, 0, 0),
    "mamianmakeng": (0, 255, 0),
    "yanghuatiepi": (128, 0, 255),
    "gunyin": (255, 128, 0),
}

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp"]


@dataclass
class VocObject:
    name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass
class VocRecord:
    xml_path: Path
    declared_width: int | None
    declared_height: int | None
    objects: list[VocObject]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize PASCAL VOC annotations for the steel defect dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("/root/autodl-tmp/steel_defect/raw/data/train"),
        help="Directory containing paired JPG and XML files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/metadata/visualizations"
        ),
        help="Output directory.",
    )
    parser.add_argument(
        "--num-per-class",
        type=int,
        default=3,
        help="Number of images sampled for each class.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Random seed.",
    )
    parser.add_argument(
        "--contact-cols",
        type=int,
        default=3,
        help="Number of columns in the contact sheet.",
    )
    return parser.parse_args()


def parse_integer(text: str | None, field_name: str) -> int:
    if text is None:
        raise ValueError(f"Missing value for {field_name}")
    return int(round(float(text.strip())))


def parse_voc_xml(xml_path: Path) -> VocRecord:
    root = ET.parse(xml_path).getroot()

    size_node = root.find("size")
    declared_width = None
    declared_height = None

    if size_node is not None:
        width_text = size_node.findtext("width")
        height_text = size_node.findtext("height")

        if width_text:
            declared_width = parse_integer(width_text, "width")
        if height_text:
            declared_height = parse_integer(height_text, "height")

    objects: list[VocObject] = []

    for object_node in root.findall("object"):
        name = (object_node.findtext("name") or "").strip()

        bbox_node = object_node.find("bndbox")
        if not name or bbox_node is None:
            continue

        objects.append(
            VocObject(
                name=name,
                xmin=parse_integer(bbox_node.findtext("xmin"), "xmin"),
                ymin=parse_integer(bbox_node.findtext("ymin"), "ymin"),
                xmax=parse_integer(bbox_node.findtext("xmax"), "xmax"),
                ymax=parse_integer(bbox_node.findtext("ymax"), "ymax"),
            )
        )

    return VocRecord(
        xml_path=xml_path,
        declared_width=declared_width,
        declared_height=declared_height,
        objects=objects,
    )


def find_corresponding_image(xml_path: Path) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = xml_path.with_suffix(extension)
        if candidate.exists():
            return candidate

        candidate_upper = xml_path.with_suffix(extension.upper())
        if candidate_upper.exists():
            return candidate_upper

    return None


def get_color(class_name: str) -> tuple[int, int, int]:
    if class_name in CLASS_COLORS:
        return CLASS_COLORS[class_name]

    # 未知类别使用确定性颜色
    value = sum(ord(character) for character in class_name)
    return (
        64 + (value * 37) % 192,
        64 + (value * 67) % 192,
        64 + (value * 97) % 192,
    )


def draw_label(
    image: np.ndarray,
    text: str,
    x: int,
    y: int,
    color: tuple[int, int, int],
    font_scale: float,
    thickness: int,
) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    text_size, baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        max(1, thickness - 1),
    )

    text_width, text_height = text_size
    image_height, image_width = image.shape[:2]

    x = max(0, min(x, image_width - 1))
    label_top = max(0, y - text_height - baseline - 12)
    label_bottom = min(image_height - 1, y)

    label_right = min(image_width - 1, x + text_width + 12)

    cv2.rectangle(
        image,
        (x, label_top),
        (label_right, label_bottom),
        color,
        thickness=-1,
    )

    text_y = max(
        text_height,
        label_bottom - baseline - 5,
    )

    cv2.putText(
        image,
        text,
        (x + 6, text_y),
        font,
        font_scale,
        (255, 255, 255),
        max(1, thickness - 1),
        cv2.LINE_AA,
    )


def render_record(
    record: VocRecord,
    image_path: Path,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)

    if image is None:
        raise RuntimeError(f"OpenCV cannot read image: {image_path}")

    image_height, image_width = image.shape[:2]

    line_thickness = max(2, round(min(image_width, image_height) / 900))
    font_scale = max(0.6, min(image_width, image_height) / 1800)

    rows: list[dict[str, object]] = []

    for object_index, item in enumerate(record.objects):
        original_xmin = item.xmin
        original_ymin = item.ymin
        original_xmax = item.xmax
        original_ymax = item.ymax

        original_valid = (
            original_xmax > original_xmin
            and original_ymax > original_ymin
        )

        outside_image = (
            original_xmin < 0
            or original_ymin < 0
            or original_xmax > image_width
            or original_ymax > image_height
        )

        # 绘图时裁剪到实际图像范围
        xmin = max(0, min(original_xmin, image_width - 1))
        ymin = max(0, min(original_ymin, image_height - 1))
        xmax = max(0, min(original_xmax, image_width - 1))
        ymax = max(0, min(original_ymax, image_height - 1))

        clipped_valid = xmax > xmin and ymax > ymin

        rows.append(
            {
                "image": image_path.name,
                "xml": record.xml_path.name,
                "object_index": object_index,
                "class_name": item.name,
                "xmin": original_xmin,
                "ymin": original_ymin,
                "xmax": original_xmax,
                "ymax": original_ymax,
                "box_width": original_xmax - original_xmin,
                "box_height": original_ymax - original_ymin,
                "image_width": image_width,
                "image_height": image_height,
                "declared_width": record.declared_width,
                "declared_height": record.declared_height,
                "valid_box": original_valid,
                "outside_image": outside_image,
                "size_mismatch": (
                    record.declared_width is not None
                    and record.declared_height is not None
                    and (
                        record.declared_width != image_width
                        or record.declared_height != image_height
                    )
                ),
            }
        )

        if not clipped_valid:
            continue

        color = get_color(item.name)

        cv2.rectangle(
            image,
            (xmin, ymin),
            (xmax, ymax),
            color,
            thickness=line_thickness,
        )

        label = (
            f"{item.name} "
            f"{original_xmax - original_xmin}x"
            f"{original_ymax - original_ymin}"
        )

        draw_label(
            image=image,
            text=label,
            x=xmin,
            y=ymin,
            color=color,
            font_scale=font_scale,
            thickness=line_thickness,
        )

    return image, rows


def make_contact_sheet(
    rendered_images: list[tuple[Path, str]],
    output_path: Path,
    columns: int,
) -> None:
    if not rendered_images:
        return

    tile_width = 640
    image_area_height = 470
    caption_height = 44
    tile_height = image_area_height + caption_height

    rows_count = (len(rendered_images) + columns - 1) // columns

    sheet = np.full(
        (
            rows_count * tile_height,
            columns * tile_width,
            3,
        ),
        235,
        dtype=np.uint8,
    )

    for index, (image_path, caption) in enumerate(rendered_images):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        image_height, image_width = image.shape[:2]

        scale = min(
            tile_width / image_width,
            image_area_height / image_height,
        )

        resized_width = max(1, round(image_width * scale))
        resized_height = max(1, round(image_height * scale))

        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )

        row = index // columns
        column = index % columns

        tile_x = column * tile_width
        tile_y = row * tile_height

        offset_x = tile_x + (tile_width - resized_width) // 2
        offset_y = tile_y + (image_area_height - resized_height) // 2

        sheet[
            offset_y : offset_y + resized_height,
            offset_x : offset_x + resized_width,
        ] = resized

        cv2.rectangle(
            sheet,
            (tile_x, tile_y),
            (tile_x + tile_width - 1, tile_y + tile_height - 1),
            (120, 120, 120),
            thickness=1,
        )

        cv2.putText(
            sheet,
            caption[:85],
            (tile_x + 10, tile_y + image_area_height + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (30, 30, 30),
            1,
            cv2.LINE_AA,
        )

    cv2.imwrite(
        str(output_path),
        sheet,
        [cv2.IMWRITE_JPEG_QUALITY, 92],
    )


def main() -> None:
    args = parse_args()

    if not args.data_dir.exists():
        raise FileNotFoundError(
            f"Data directory does not exist: {args.data_dir}"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    annotated_dir = args.output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    xml_paths = sorted(args.data_dir.glob("*.xml"))

    if not xml_paths:
        raise RuntimeError(
            f"No XML files found in: {args.data_dir}"
        )

    records: dict[Path, VocRecord] = {}
    class_to_records: dict[str, list[VocRecord]] = defaultdict(list)
    parse_errors: list[tuple[Path, str]] = []

    print(f"Scanning {len(xml_paths)} XML files...")

    for xml_path in xml_paths:
        try:
            record = parse_voc_xml(xml_path)
            records[xml_path] = record

            class_names = {item.name for item in record.objects}
            for class_name in class_names:
                class_to_records[class_name].append(record)

        except Exception as exception:
            parse_errors.append((xml_path, repr(exception)))

    if parse_errors:
        print(f"XML parse errors: {len(parse_errors)}")
        for path, error in parse_errors[:10]:
            print(f"  {path.name}: {error}")

    discovered_classes = sorted(class_to_records)

    ordered_classes = [
        class_name
        for class_name in CLASS_ORDER
        if class_name in class_to_records
    ]

    ordered_classes.extend(
        class_name
        for class_name in discovered_classes
        if class_name not in ordered_classes
    )

    print("\nDiscovered classes:")
    for class_name in ordered_classes:
        print(
            f"  {class_name}: "
            f"{len(class_to_records[class_name])} images"
        )

    rng = random.Random(args.seed)

    selected_records: dict[Path, VocRecord] = {}

    for class_name in ordered_classes:
        candidates = list(class_to_records[class_name])
        rng.shuffle(candidates)

        for record in candidates[: args.num_per_class]:
            selected_records[record.xml_path] = record

    print(
        f"\nSelected {len(selected_records)} unique images "
        f"for visualization."
    )

    all_rows: list[dict[str, object]] = []
    rendered_images: list[tuple[Path, str]] = []
    missing_images: list[Path] = []

    for index, record in enumerate(
        sorted(selected_records.values(), key=lambda item: item.xml_path.name),
        start=1,
    ):
        image_path = find_corresponding_image(record.xml_path)

        if image_path is None:
            missing_images.append(record.xml_path)
            continue

        try:
            rendered, rows = render_record(record, image_path)
        except Exception as exception:
            print(
                f"Render failed: {image_path.name}: {exception!r}"
            )
            continue

        all_rows.extend(rows)

        output_path = annotated_dir / image_path.name

        success = cv2.imwrite(
            str(output_path),
            rendered,
            [cv2.IMWRITE_JPEG_QUALITY, 95],
        )

        if not success:
            print(f"Failed to save: {output_path}")
            continue

        classes = sorted({item.name for item in record.objects})
        caption = (
            f"{image_path.name} | "
            f"{','.join(classes)} | "
            f"{len(record.objects)} boxes"
        )

        rendered_images.append((output_path, caption))

        print(
            f"[{index:02d}/{len(selected_records):02d}] "
            f"{output_path.name}: "
            f"{len(record.objects)} boxes"
        )

    index_path = args.output_dir / "visualization_index.csv"

    fieldnames = [
        "image",
        "xml",
        "object_index",
        "class_name",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "box_width",
        "box_height",
        "image_width",
        "image_height",
        "declared_width",
        "declared_height",
        "valid_box",
        "outside_image",
        "size_mismatch",
    ]

    with index_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    contact_sheet_path = args.output_dir / "contact_sheet.jpg"

    make_contact_sheet(
        rendered_images=rendered_images,
        output_path=contact_sheet_path,
        columns=max(1, args.contact_cols),
    )

    invalid_count = sum(
        not bool(row["valid_box"])
        for row in all_rows
    )

    outside_count = sum(
        bool(row["outside_image"])
        for row in all_rows
    )

    mismatch_count = sum(
        bool(row["size_mismatch"])
        for row in all_rows
    )

    print("\n===== Visualization summary =====")
    print(f"Parsed XML files:       {len(records)}")
    print(f"XML parse errors:       {len(parse_errors)}")
    print(f"Selected images:        {len(selected_records)}")
    print(f"Rendered images:        {len(rendered_images)}")
    print(f"Missing images:         {len(missing_images)}")
    print(f"Rendered boxes:         {len(all_rows)}")
    print(f"Invalid boxes:          {invalid_count}")
    print(f"Out-of-bounds boxes:    {outside_count}")
    print(f"Image/XML size mismatch:{mismatch_count}")
    print(f"Annotated images:       {annotated_dir}")
    print(f"Contact sheet:          {contact_sheet_path}")
    print(f"Index CSV:              {index_path}")


if __name__ == "__main__":
    main()
