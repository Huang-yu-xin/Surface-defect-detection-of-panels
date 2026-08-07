from __future__ import annotations

import csv
import json
import math
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

from PIL import Image


DATA_DIR = Path("/root/autodl-tmp/steel_defect/raw/data/train")
OUTPUT_DIR = Path("/root/autodl-tmp/steel_defect/metadata/audit")

EXPECTED_CLASSES = [
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

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]

# 用于估计“整张原图缩放训练”时，小目标会变成多少像素
SIMULATED_IMGSZ = [640, 1024, 1280, 1536]


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None

    ordered = sorted(values)

    if len(ordered) == 1:
        return float(ordered[0])

    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return float(ordered[lower])

    fraction = position - lower
    return float(
        ordered[lower] * (1 - fraction)
        + ordered[upper] * fraction
    )


def safe_mean(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.mean(values) if values else None


def safe_median(values: Iterable[float]) -> float | None:
    values = list(values)
    return statistics.median(values) if values else None


def find_image(stem: str) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = DATA_DIR / f"{stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def parse_int(text: str | None, field: str) -> int:
    if text is None:
        raise ValueError(f"缺少字段：{field}")
    return int(round(float(text.strip())))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def rounded(value: float | None, digits: int = 4):
    if value is None:
        return None
    return round(float(value), digits)


def summarize_class(
    class_name: str,
    rows: list[dict],
    images_with_class: int,
) -> dict:
    widths = [row["box_width"] for row in rows]
    heights = [row["box_height"] for row in rows]
    areas = [row["box_area"] for row in rows]
    min_sides = [row["min_side"] for row in rows]
    max_sides = [row["max_side"] for row in rows]
    aspect_ratios = [row["aspect_ratio"] for row in rows]
    area_ratios = [row["area_ratio"] for row in rows]

    result = {
        "class_name": class_name,
        "box_count": len(rows),
        "image_count": images_with_class,
        "boxes_per_image": rounded(
            len(rows) / images_with_class if images_with_class else None
        ),
        "width_min": min(widths) if widths else None,
        "width_p10": rounded(percentile(widths, 0.10)),
        "width_p25": rounded(percentile(widths, 0.25)),
        "width_median": rounded(safe_median(widths)),
        "width_p75": rounded(percentile(widths, 0.75)),
        "width_p90": rounded(percentile(widths, 0.90)),
        "width_max": max(widths) if widths else None,
        "height_min": min(heights) if heights else None,
        "height_p10": rounded(percentile(heights, 0.10)),
        "height_p25": rounded(percentile(heights, 0.25)),
        "height_median": rounded(safe_median(heights)),
        "height_p75": rounded(percentile(heights, 0.75)),
        "height_p90": rounded(percentile(heights, 0.90)),
        "height_max": max(heights) if heights else None,
        "area_median": rounded(safe_median(areas)),
        "area_p90": rounded(percentile(areas, 0.90)),
        "min_side_median": rounded(safe_median(min_sides)),
        "min_side_p10": rounded(percentile(min_sides, 0.10)),
        "max_side_median": rounded(safe_median(max_sides)),
        "aspect_ratio_median": rounded(safe_median(aspect_ratios)),
        "aspect_ratio_p90": rounded(percentile(aspect_ratios, 0.90)),
        "area_ratio_median": rounded(safe_median(area_ratios), 8),
        "tiny_original_count": sum(
            min(row["box_width"], row["box_height"]) < 16
            for row in rows
        ),
        "small_original_count": sum(
            row["box_area"] < 32 * 32
            for row in rows
        ),
        "very_elongated_count": sum(
            row["aspect_ratio"] >= 10
            for row in rows
        ),
    }

    for imgsz in SIMULATED_IMGSZ:
        key = f"min_side_at_{imgsz}_median"
        values = [row[f"min_side_at_{imgsz}"] for row in rows]
        result[key] = rounded(safe_median(values))

        key = f"under_4px_at_{imgsz}"
        result[key] = sum(value < 4 for value in values)

        key = f"under_8px_at_{imgsz}"
        result[key] = sum(value < 8 for value in values)

    return result


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    xml_paths = sorted(DATA_DIR.glob("*.xml"))
    image_paths = sorted(
        path
        for path in DATA_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    image_stems = {path.stem for path in image_paths}
    xml_stems = {path.stem for path in xml_paths}

    missing_xml = sorted(image_stems - xml_stems)
    missing_image = sorted(xml_stems - image_stems)

    anomalies: list[dict] = []
    image_rows: list[dict] = []
    bbox_rows: list[dict] = []

    class_box_counter = Counter()
    class_image_counter = Counter()
    empty_xml_count = 0
    parse_error_count = 0
    unreadable_image_count = 0

    print(f"图片数：{len(image_paths)}")
    print(f"XML 数：{len(xml_paths)}")

    for stem in missing_xml:
        anomalies.append({
            "type": "missing_xml",
            "file": stem,
            "class_name": "",
            "details": "图片缺少同名 XML",
        })

    for stem in missing_image:
        anomalies.append({
            "type": "missing_image",
            "file": stem,
            "class_name": "",
            "details": "XML 缺少同名图片",
        })

    for index, xml_path in enumerate(xml_paths, start=1):
        stem = xml_path.stem
        image_path = find_image(stem)

        try:
            root = ET.parse(xml_path).getroot()
        except Exception as exc:
            parse_error_count += 1
            anomalies.append({
                "type": "xml_parse_error",
                "file": xml_path.name,
                "class_name": "",
                "details": repr(exc),
            })
            continue

        if image_path is None:
            continue

        try:
            with Image.open(image_path) as image:
                actual_width, actual_height = image.size
                image_mode = image.mode
        except Exception as exc:
            unreadable_image_count += 1
            anomalies.append({
                "type": "unreadable_image",
                "file": image_path.name,
                "class_name": "",
                "details": repr(exc),
            })
            continue

        declared_width = None
        declared_height = None
        size_node = root.find("size")

        if size_node is not None:
            try:
                declared_width = parse_int(
                    size_node.findtext("width"), "width"
                )
                declared_height = parse_int(
                    size_node.findtext("height"), "height"
                )
            except Exception as exc:
                anomalies.append({
                    "type": "invalid_declared_size",
                    "file": xml_path.name,
                    "class_name": "",
                    "details": repr(exc),
                })

        size_mismatch = (
            declared_width is not None
            and declared_height is not None
            and (
                declared_width != actual_width
                or declared_height != actual_height
            )
        )

        if size_mismatch:
            anomalies.append({
                "type": "size_mismatch",
                "file": xml_path.name,
                "class_name": "",
                "details": (
                    f"XML={declared_width}x{declared_height}, "
                    f"image={actual_width}x{actual_height}"
                ),
            })

        objects = root.findall("object")

        if not objects:
            empty_xml_count += 1
            anomalies.append({
                "type": "empty_xml",
                "file": xml_path.name,
                "class_name": "",
                "details": "XML 中没有 object",
            })

        classes_in_image = set()
        valid_box_count = 0

        for object_index, object_node in enumerate(objects):
            class_name = (
                object_node.findtext("name") or ""
            ).strip()

            bbox_node = object_node.find("bndbox")

            if not class_name:
                anomalies.append({
                    "type": "empty_class_name",
                    "file": xml_path.name,
                    "class_name": "",
                    "details": f"object_index={object_index}",
                })
                continue

            if class_name not in EXPECTED_CLASSES:
                anomalies.append({
                    "type": "unknown_class",
                    "file": xml_path.name,
                    "class_name": class_name,
                    "details": f"object_index={object_index}",
                })

            if bbox_node is None:
                anomalies.append({
                    "type": "missing_bbox",
                    "file": xml_path.name,
                    "class_name": class_name,
                    "details": f"object_index={object_index}",
                })
                continue

            try:
                xmin = parse_int(
                    bbox_node.findtext("xmin"), "xmin"
                )
                ymin = parse_int(
                    bbox_node.findtext("ymin"), "ymin"
                )
                xmax = parse_int(
                    bbox_node.findtext("xmax"), "xmax"
                )
                ymax = parse_int(
                    bbox_node.findtext("ymax"), "ymax"
                )
            except Exception as exc:
                anomalies.append({
                    "type": "invalid_bbox_value",
                    "file": xml_path.name,
                    "class_name": class_name,
                    "details": (
                        f"object_index={object_index}, {exc!r}"
                    ),
                })
                continue

            box_width = xmax - xmin
            box_height = ymax - ymin

            valid_geometry = box_width > 0 and box_height > 0
            outside_image = (
                xmin < 0
                or ymin < 0
                or xmax > actual_width
                or ymax > actual_height
            )

            if not valid_geometry:
                anomalies.append({
                    "type": "invalid_bbox_geometry",
                    "file": xml_path.name,
                    "class_name": class_name,
                    "details": (
                        f"[{xmin}, {ymin}, {xmax}, {ymax}]"
                    ),
                })
                continue

            if outside_image:
                anomalies.append({
                    "type": "bbox_outside_image",
                    "file": xml_path.name,
                    "class_name": class_name,
                    "details": (
                        f"bbox=[{xmin},{ymin},{xmax},{ymax}], "
                        f"image={actual_width}x{actual_height}"
                    ),
                })

            box_area = box_width * box_height
            image_area = actual_width * actual_height
            min_side = min(box_width, box_height)
            max_side = max(box_width, box_height)
            aspect_ratio = max_side / max(1, min_side)

            row = {
                "image_name": image_path.name,
                "xml_name": xml_path.name,
                "image_name": image_path.name,
                "xml_name": xml_path.name,
                "object_index": object_index,
                "class_name": class_name,
                "image_width": actual_width,
                "image_height": actual_height,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "box_width": box_width,
                "box_height": box_height,
                "box_area": box_area,
                "area_ratio": box_area / image_area,
                "width_ratio": box_width / actual_width,
                "height_ratio": box_height / actual_height,
                "min_side": min_side,
                "max_side": max_side,
                "aspect_ratio": aspect_ratio,
                "outside_image": outside_image,
            }

            # 模拟原图 letterbox 到 imgsz 后目标的短边像素数
            for imgsz in SIMULATED_IMGSZ:
                scale = min(
                    imgsz / actual_width,
                    imgsz / actual_height,
                )
                row[f"box_width_at_{imgsz}"] = box_width * scale
                row[f"box_height_at_{imgsz}"] = box_height * scale
                row[f"min_side_at_{imgsz}"] = min_side * scale

            bbox_rows.append(row)

            valid_box_count += 1
            class_box_counter[class_name] += 1
            classes_in_image.add(class_name)

        for class_name in classes_in_image:
            class_image_counter[class_name] += 1

        image_rows.append({
            "image_name": image_path.name,
            "xml_name": xml_path.name,
            "width": actual_width,
            "height": actual_height,
            "mode": image_mode,
            "object_count": len(objects),
            "valid_box_count": valid_box_count,
            "classes": ",".join(sorted(classes_in_image)),
            "size_mismatch": size_mismatch,
        })

        if index % 250 == 0:
            print(f"已处理 {index}/{len(xml_paths)}")

    rows_by_class = defaultdict(list)

    for row in bbox_rows:
        rows_by_class[row["class_name"]].append(row)

    class_order = list(EXPECTED_CLASSES)
    class_order.extend(
        sorted(
            class_name
            for class_name in rows_by_class
            if class_name not in EXPECTED_CLASSES
        )
    )

    class_summary_rows = [
        summarize_class(
            class_name,
            rows_by_class.get(class_name, []),
            class_image_counter.get(class_name, 0),
        )
        for class_name in class_order
    ]

    bbox_fieldnames = [
        "image_name",
        "xml_name",
        "object_index",
        "class_name",
        "image_width",
        "image_height",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "box_width",
        "box_height",
        "box_area",
        "area_ratio",
        "width_ratio",
        "height_ratio",
        "min_side",
        "max_side",
        "aspect_ratio",
        "outside_image",
    ]

    for imgsz in SIMULATED_IMGSZ:
        bbox_fieldnames.extend([
            f"box_width_at_{imgsz}",
            f"box_height_at_{imgsz}",
            f"min_side_at_{imgsz}",
        ])

    class_summary_fieldnames = list(class_summary_rows[0].keys())

    write_csv(
        OUTPUT_DIR / "bbox_metrics.csv",
        bbox_rows,
        bbox_fieldnames,
    )

    write_csv(
        OUTPUT_DIR / "image_summary.csv",
        image_rows,
        [
            "image_name",
            "xml_name",
            "width",
            "height",
            "mode",
            "object_count",
            "valid_box_count",
            "classes",
            "size_mismatch",
        ],
    )

    write_csv(
        OUTPUT_DIR / "class_summary.csv",
        class_summary_rows,
        class_summary_fieldnames,
    )

    write_csv(
        OUTPUT_DIR / "anomalies.csv",
        anomalies,
        ["type", "file", "class_name", "details"],
    )

    anomaly_counter = Counter(
        row["type"] for row in anomalies
    )

    overall_summary = {
        "image_count": len(image_paths),
        "xml_count": len(xml_paths),
        "paired_image_count": len(image_rows),
        "bbox_count": len(bbox_rows),
        "missing_xml_count": len(missing_xml),
        "missing_image_count": len(missing_image),
        "empty_xml_count": empty_xml_count,
        "xml_parse_error_count": parse_error_count,
        "unreadable_image_count": unreadable_image_count,
        "class_box_counts": dict(class_box_counter),
        "class_image_counts": dict(class_image_counter),
        "anomaly_counts": dict(anomaly_counter),
    }

    with (
        OUTPUT_DIR / "audit_summary.json"
    ).open("w", encoding="utf-8") as file:
        json.dump(
            overall_summary,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print("\n========== 全量审计结果 ==========")
    print(f"图片数：                 {len(image_paths)}")
    print(f"XML 数：                  {len(xml_paths)}")
    print(f"成功配对并读取：          {len(image_rows)}")
    print(f"有效目标框总数：          {len(bbox_rows)}")
    print(f"图片缺 XML：              {len(missing_xml)}")
    print(f"XML 缺图片：              {len(missing_image)}")
    print(f"空 XML：                  {empty_xml_count}")
    print(f"XML 解析失败：            {parse_error_count}")
    print(f"图片读取失败：            {unreadable_image_count}")
    print(f"异常记录总数：            {len(anomalies)}")

    print("\n类别框数量：")
    for class_name in class_order:
        print(
            f"  {class_name:<18} "
            f"boxes={class_box_counter.get(class_name, 0):<5} "
            f"images={class_image_counter.get(class_name, 0)}"
        )

    if anomaly_counter:
        print("\n异常类型：")
        for name, count in sorted(anomaly_counter.items()):
            print(f"  {name}: {count}")

    print("\n输出目录：", OUTPUT_DIR)
    print("  audit_summary.json")
    print("  class_summary.csv")
    print("  bbox_metrics.csv")
    print("  image_summary.csv")
    print("  anomalies.csv")


if __name__ == "__main__":
    main()
