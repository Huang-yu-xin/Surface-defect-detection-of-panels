from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from PIL import Image


# YOLO 类别编号必须固定，之后训练和生成提交文件都使用这个顺序
CLASS_NAMES = [
    "jieba",           # 0 结疤
    "zonglie",         # 1 纵裂
    "qilie",           # 2 气裂
    "jiaza",           # 3 夹杂
    "yiwuyaru",        # 4 异物压入
    "huashang",        # 5 划伤
    "mamianmakeng",    # 6 麻面麻坑
    "yanghuatiepi",    # 7 氧化铁皮
    "gunyin",          # 8 辊印
]

CLASS_TO_ID = {
    class_name: class_id
    for class_id, class_name in enumerate(CLASS_NAMES)
}

IMAGE_EXTENSIONS = [
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".BMP",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert paired PASCAL VOC XML annotations to "
            "Ultralytics YOLO labels."
        )
    )

    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/raw/data/train"
        ),
        help="包含 JPG 和同名 XML 的原始训练目录。",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/datasets/yolo_all"
        ),
        help="转换后的 YOLO 数据根目录。",
    )

    parser.add_argument(
        "--metadata-dir",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/metadata/conversion"
        ),
        help="转换日志和统计结果目录。",
    )

    parser.add_argument(
        "--copy-images",
        action="store_true",
        help=(
            "复制图片而不是创建软链接。"
            "默认创建软链接，避免重复占用约 16GB 空间。"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已经存在的输出标签和图片链接。",
    )

    return parser.parse_args()


def find_corresponding_image(
    xml_path: Path,
) -> Path | None:
    """查找与 XML 同名的图片。"""

    for extension in IMAGE_EXTENSIONS:
        candidate = xml_path.with_suffix(extension)

        if candidate.exists():
            return candidate

    return None


def parse_integer(
    text: str | None,
    field_name: str,
) -> int:
    """兼容 XML 中可能出现的整数或浮点数字符串。"""

    if text is None:
        raise ValueError(f"缺少字段：{field_name}")

    return int(round(float(text.strip())))


def has_intersection(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: int,
    image_height: int,
) -> bool:
    """判断原始框是否与图像区域存在有效交集。"""

    return not (
        xmax <= 0
        or ymax <= 0
        or xmin >= image_width
        or ymin >= image_height
    )


def clip_bbox(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """
    将边界框裁剪到图像的连续坐标范围内。

    注意：
    - 像素索引最大为 width-1、height-1；
    - 目标检测框的右边界和下边界可以等于 width、height；
    - 因此 4096×3000 图像的 xmax 可为 4096，ymax 可为 3000。
    """

    xmin = max(
        0.0,
        min(float(xmin), float(image_width - 1)),
    )
    ymin = max(
        0.0,
        min(float(ymin), float(image_height - 1)),
    )

    xmax = max(
        xmin + 1.0,
        min(float(xmax), float(image_width)),
    )
    ymax = max(
        ymin + 1.0,
        min(float(ymax), float(image_height)),
    )

    return xmin, ymin, xmax, ymax


def voc_bbox_to_yolo(
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """将裁剪后的绝对坐标框转换为 YOLO 归一化坐标。"""

    box_width = xmax - xmin
    box_height = ymax - ymin

    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0

    normalized_center_x = center_x / image_width
    normalized_center_y = center_y / image_height
    normalized_width = box_width / image_width
    normalized_height = box_height / image_height

    return (
        normalized_center_x,
        normalized_center_y,
        normalized_width,
        normalized_height,
    )


def create_image_reference(
    source_image: Path,
    destination_image: Path,
    copy_images: bool,
    overwrite: bool,
) -> None:
    """
    默认创建相对软链接，避免复制全部原图。

    使用 --copy-images 时才真正复制图片。
    """

    destination_image.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    destination_exists = os.path.lexists(
        destination_image
    )

    if destination_exists:
        if not overwrite:
            return

        destination_image.unlink()

    if copy_images:
        shutil.copy2(
            source_image,
            destination_image,
        )
        return

    relative_target = os.path.relpath(
        source_image.resolve(),
        start=destination_image.parent.resolve(),
    )

    destination_image.symlink_to(
        relative_target
    )


def write_csv(
    output_path: Path,
    rows: list[dict[str, object]],
    fieldnames: list[str],
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open(
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


def main() -> None:
    args = parse_args()

    source_dir = args.source_dir.resolve()
    output_root = args.output_root.resolve()
    metadata_dir = args.metadata_dir.resolve()

    images_output_dir = output_root / "images"
    labels_output_dir = output_root / "labels"

    images_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    labels_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    metadata_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not source_dir.exists():
        raise FileNotFoundError(
            f"原始数据目录不存在：{source_dir}"
        )

    xml_paths = sorted(
        source_dir.glob("*.xml")
    )

    if not xml_paths:
        raise RuntimeError(
            f"没有在目录中找到 XML：{source_dir}"
        )

    correction_rows: list[dict[str, object]] = []
    error_rows: list[dict[str, object]] = []
    image_rows: list[dict[str, object]] = []

    class_box_counter: Counter[str] = Counter()

    converted_image_count = 0
    label_file_count = 0
    empty_label_count = 0

    input_box_count = 0
    written_box_count = 0
    skipped_box_count = 0

    print(f"原始目录：{source_dir}")
    print(f"XML 数量：{len(xml_paths)}")
    print(f"输出目录：{output_root}")
    print()

    for xml_index, xml_path in enumerate(
        xml_paths,
        start=1,
    ):
        image_path = find_corresponding_image(
            xml_path
        )

        if image_path is None:
            error_rows.append({
                "type": "missing_image",
                "image": "",
                "xml": xml_path.name,
                "object_index": "",
                "class_name": "",
                "details": "找不到与 XML 同名的图片",
            })
            continue

        try:
            with Image.open(image_path) as image:
                image_width, image_height = image.size
                image_mode = image.mode

        except Exception as exception:
            error_rows.append({
                "type": "unreadable_image",
                "image": image_path.name,
                "xml": xml_path.name,
                "object_index": "",
                "class_name": "",
                "details": repr(exception),
            })
            continue

        try:
            root = ET.parse(xml_path).getroot()

        except Exception as exception:
            error_rows.append({
                "type": "xml_parse_error",
                "image": image_path.name,
                "xml": xml_path.name,
                "object_index": "",
                "class_name": "",
                "details": repr(exception),
            })
            continue

        # XML 中声明的尺寸仅用于检查；
        # 转换时始终使用图片实际尺寸。
        declared_width = None
        declared_height = None

        size_node = root.find("size")

        if size_node is not None:
            try:
                declared_width = parse_integer(
                    size_node.findtext("width"),
                    "width",
                )
                declared_height = parse_integer(
                    size_node.findtext("height"),
                    "height",
                )

            except Exception as exception:
                error_rows.append({
                    "type": "invalid_declared_size",
                    "image": image_path.name,
                    "xml": xml_path.name,
                    "object_index": "",
                    "class_name": "",
                    "details": repr(exception),
                })

        if (
            declared_width is not None
            and declared_height is not None
            and (
                declared_width != image_width
                or declared_height != image_height
            )
        ):
            error_rows.append({
                "type": "image_xml_size_mismatch",
                "image": image_path.name,
                "xml": xml_path.name,
                "object_index": "",
                "class_name": "",
                "details": (
                    f"XML={declared_width}x{declared_height}, "
                    f"actual={image_width}x{image_height}"
                ),
            })

        yolo_lines: list[str] = []
        valid_classes_in_image: set[str] = set()

        object_nodes = root.findall("object")

        for object_index, object_node in enumerate(
            object_nodes
        ):
            input_box_count += 1

            class_name = (
                object_node.findtext("name") or ""
            ).strip()

            if not class_name:
                skipped_box_count += 1

                error_rows.append({
                    "type": "empty_class_name",
                    "image": image_path.name,
                    "xml": xml_path.name,
                    "object_index": object_index,
                    "class_name": "",
                    "details": "object 中缺少 name",
                })
                continue

            if class_name not in CLASS_TO_ID:
                skipped_box_count += 1

                error_rows.append({
                    "type": "unknown_class",
                    "image": image_path.name,
                    "xml": xml_path.name,
                    "object_index": object_index,
                    "class_name": class_name,
                    "details": "不在预期 9 类中",
                })
                continue

            bbox_node = object_node.find("bndbox")

            if bbox_node is None:
                skipped_box_count += 1

                error_rows.append({
                    "type": "missing_bbox",
                    "image": image_path.name,
                    "xml": xml_path.name,
                    "object_index": object_index,
                    "class_name": class_name,
                    "details": "object 中缺少 bndbox",
                })
                continue

            try:
                original_xmin = parse_integer(
                    bbox_node.findtext("xmin"),
                    "xmin",
                )
                original_ymin = parse_integer(
                    bbox_node.findtext("ymin"),
                    "ymin",
                )
                original_xmax = parse_integer(
                    bbox_node.findtext("xmax"),
                    "xmax",
                )
                original_ymax = parse_integer(
                    bbox_node.findtext("ymax"),
                    "ymax",
                )

            except Exception as exception:
                skipped_box_count += 1

                error_rows.append({
                    "type": "invalid_bbox_value",
                    "image": image_path.name,
                    "xml": xml_path.name,
                    "object_index": object_index,
                    "class_name": class_name,
                    "details": repr(exception),
                })
                continue

            if (
                original_xmax <= original_xmin
                or original_ymax <= original_ymin
            ):
                skipped_box_count += 1

                error_rows.append({
                    "type": "invalid_bbox_geometry",
                    "image": image_path.name,
                    "xml": xml_path.name,
                    "object_index": object_index,
                    "class_name": class_name,
                    "details": (
                        f"bbox=[{original_xmin},"
                        f"{original_ymin},"
                        f"{original_xmax},"
                        f"{original_ymax}]"
                    ),
                })
                continue

            if not has_intersection(
                original_xmin,
                original_ymin,
                original_xmax,
                original_ymax,
                image_width,
                image_height,
            ):
                skipped_box_count += 1

                error_rows.append({
                    "type": "bbox_no_image_intersection",
                    "image": image_path.name,
                    "xml": xml_path.name,
                    "object_index": object_index,
                    "class_name": class_name,
                    "details": (
                        f"bbox=[{original_xmin},"
                        f"{original_ymin},"
                        f"{original_xmax},"
                        f"{original_ymax}], "
                        f"image={image_width}x{image_height}"
                    ),
                })
                continue

            clipped_xmin, clipped_ymin, clipped_xmax, clipped_ymax = (
                clip_bbox(
                    original_xmin,
                    original_ymin,
                    original_xmax,
                    original_ymax,
                    image_width,
                    image_height,
                )
            )

            original_bbox = (
                float(original_xmin),
                float(original_ymin),
                float(original_xmax),
                float(original_ymax),
            )

            clipped_bbox = (
                clipped_xmin,
                clipped_ymin,
                clipped_xmax,
                clipped_ymax,
            )

            if clipped_bbox != original_bbox:
                correction_rows.append({
                    "image": image_path.name,
                    "xml": xml_path.name,
                    "object_index": object_index,
                    "class_name": class_name,
                    "image_width": image_width,
                    "image_height": image_height,
                    "original_xmin": original_xmin,
                    "original_ymin": original_ymin,
                    "original_xmax": original_xmax,
                    "original_ymax": original_ymax,
                    "clipped_xmin": clipped_xmin,
                    "clipped_ymin": clipped_ymin,
                    "clipped_xmax": clipped_xmax,
                    "clipped_ymax": clipped_ymax,
                    "reason": "bbox_outside_image",
                })

            yolo_x, yolo_y, yolo_width, yolo_height = (
                voc_bbox_to_yolo(
                    clipped_xmin,
                    clipped_ymin,
                    clipped_xmax,
                    clipped_ymax,
                    image_width,
                    image_height,
                )
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
                skipped_box_count += 1

                error_rows.append({
                    "type": "normalized_bbox_out_of_range",
                    "image": image_path.name,
                    "xml": xml_path.name,
                    "object_index": object_index,
                    "class_name": class_name,
                    "details": (
                        f"normalized={normalized_values}"
                    ),
                })
                continue

            if yolo_width <= 0 or yolo_height <= 0:
                skipped_box_count += 1

                error_rows.append({
                    "type": "zero_area_after_clipping",
                    "image": image_path.name,
                    "xml": xml_path.name,
                    "object_index": object_index,
                    "class_name": class_name,
                    "details": (
                        f"clipped_bbox={clipped_bbox}"
                    ),
                })
                continue

            class_id = CLASS_TO_ID[class_name]

            yolo_lines.append(
                f"{class_id} "
                f"{yolo_x:.8f} "
                f"{yolo_y:.8f} "
                f"{yolo_width:.8f} "
                f"{yolo_height:.8f}"
            )

            written_box_count += 1
            class_box_counter[class_name] += 1
            valid_classes_in_image.add(class_name)

        label_path = (
            labels_output_dir
            / f"{image_path.stem}.txt"
        )

        if label_path.exists() and not args.overwrite:
            pass
        else:
            label_path.write_text(
                "\n".join(yolo_lines)
                + ("\n" if yolo_lines else ""),
                encoding="utf-8",
            )

        label_file_count += 1

        if not yolo_lines:
            empty_label_count += 1

        destination_image = (
            images_output_dir
            / image_path.name
        )

        create_image_reference(
            source_image=image_path,
            destination_image=destination_image,
            copy_images=args.copy_images,
            overwrite=args.overwrite,
        )

        converted_image_count += 1

        image_rows.append({
            "image": image_path.name,
            "xml": xml_path.name,
            "width": image_width,
            "height": image_height,
            "mode": image_mode,
            "original_object_count": len(object_nodes),
            "written_box_count": len(yolo_lines),
            "is_empty_label": len(yolo_lines) == 0,
            "classes": ",".join(
                sorted(valid_classes_in_image)
            ),
            "image_reference_type": (
                "copy" if args.copy_images else "symlink"
            ),
        })

        if xml_index % 250 == 0:
            print(
                f"已处理 {xml_index}/{len(xml_paths)}"
            )

    correction_fieldnames = [
        "image",
        "xml",
        "object_index",
        "class_name",
        "image_width",
        "image_height",
        "original_xmin",
        "original_ymin",
        "original_xmax",
        "original_ymax",
        "clipped_xmin",
        "clipped_ymin",
        "clipped_xmax",
        "clipped_ymax",
        "reason",
    ]

    error_fieldnames = [
        "type",
        "image",
        "xml",
        "object_index",
        "class_name",
        "details",
    ]

    image_fieldnames = [
        "image",
        "xml",
        "width",
        "height",
        "mode",
        "original_object_count",
        "written_box_count",
        "is_empty_label",
        "classes",
        "image_reference_type",
    ]

    write_csv(
        metadata_dir / "bbox_corrections.csv",
        correction_rows,
        correction_fieldnames,
    )

    write_csv(
        metadata_dir / "conversion_errors.csv",
        error_rows,
        error_fieldnames,
    )

    write_csv(
        metadata_dir / "conversion_images.csv",
        image_rows,
        image_fieldnames,
    )

    class_map = {
        str(class_id): class_name
        for class_id, class_name in enumerate(
            CLASS_NAMES
        )
    }

    with (
        metadata_dir / "class_map.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            class_map,
            file,
            ensure_ascii=False,
            indent=2,
        )

    summary = {
        "source_dir": str(source_dir),
        "output_root": str(output_root),
        "xml_count": len(xml_paths),
        "converted_image_count": converted_image_count,
        "label_file_count": label_file_count,
        "empty_label_count": empty_label_count,
        "input_box_count": input_box_count,
        "written_box_count": written_box_count,
        "skipped_box_count": skipped_box_count,
        "bbox_correction_count": len(correction_rows),
        "error_record_count": len(error_rows),
        "class_box_counts": {
            class_name: class_box_counter.get(
                class_name,
                0,
            )
            for class_name in CLASS_NAMES
        },
        "image_storage": (
            "copy" if args.copy_images else "relative_symlink"
        ),
    }

    with (
        metadata_dir / "conversion_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print()
    print("========== VOC → YOLO 转换结果 ==========")
    print(f"转换图片数：          {converted_image_count}")
    print(f"标签文件数：          {label_file_count}")
    print(f"空标签文件数：        {empty_label_count}")
    print(f"输入目标框数：        {input_box_count}")
    print(f"写入目标框数：        {written_box_count}")
    print(f"跳过目标框数：        {skipped_box_count}")
    print(f"自动裁剪框数：        {len(correction_rows)}")
    print(f"错误记录数：          {len(error_rows)}")

    print("\n类别框数量：")
    for class_id, class_name in enumerate(
        CLASS_NAMES
    ):
        print(
            f"  {class_id}: "
            f"{class_name:<18} "
            f"{class_box_counter.get(class_name, 0)}"
        )

    print("\n图片目录：", images_output_dir)
    print("标签目录：", labels_output_dir)
    print("转换日志：", metadata_dir)
    print(
        "框修正记录：",
        metadata_dir / "bbox_corrections.csv",
    )
    print(
        "错误记录：",
        metadata_dir / "conversion_errors.csv",
    )


if __name__ == "__main__":
    main()
