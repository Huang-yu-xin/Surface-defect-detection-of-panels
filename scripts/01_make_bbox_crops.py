from pathlib import Path
import xml.etree.ElementTree as ET

import cv2


DATA_DIR = Path("/root/autodl-tmp/steel_defect/raw/data/train")
VIS_DIR = Path("/root/autodl-tmp/steel_defect/metadata/visualizations")
ANNOTATED_DIR = VIS_DIR / "annotated"
OUTPUT_DIR = VIS_DIR / "bbox_crops"

CONTEXT_SCALE = 3.0
MIN_CONTEXT = 160
OUTPUT_SIZE = 640

COLORS = {
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


def parse_box(obj):
    box = obj.find("bndbox")
    return (
        int(float(box.findtext("xmin"))),
        int(float(box.findtext("ymin"))),
        int(float(box.findtext("xmax"))),
        int(float(box.findtext("ymax"))),
    )


def find_image(stem):
    for suffix in [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]:
        path = DATA_DIR / f"{stem}{suffix}"
        if path.exists():
            return path
    return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 只处理上一阶段已经抽样并生成可视化的图片
    selected_stems = {
        path.stem for path in ANNOTATED_DIR.glob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    }

    if not selected_stems:
        raise RuntimeError(f"没有在 {ANNOTATED_DIR} 找到抽样图片")

    saved = 0

    for stem in sorted(selected_stems):
        xml_path = DATA_DIR / f"{stem}.xml"
        image_path = find_image(stem)

        if not xml_path.exists() or image_path is None:
            print(f"跳过：缺少图片或 XML：{stem}")
            continue

        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"无法读取图片：{image_path}")
            continue

        height, width = image.shape[:2]
        root = ET.parse(xml_path).getroot()

        for index, obj in enumerate(root.findall("object")):
            class_name = (obj.findtext("name") or "unknown").strip()
            xmin, ymin, xmax, ymax = parse_box(obj)

            box_width = max(1, xmax - xmin)
            box_height = max(1, ymax - ymin)

            center_x = (xmin + xmax) / 2
            center_y = (ymin + ymax) / 2

            context_width = max(
                MIN_CONTEXT,
                int(box_width * CONTEXT_SCALE),
            )
            context_height = max(
                MIN_CONTEXT,
                int(box_height * CONTEXT_SCALE),
            )

            # 尽量保留长条缺陷周围的上下文
            if box_width > box_height * 4:
                context_height = max(context_height, 320)
            if box_height > box_width * 4:
                context_width = max(context_width, 320)

            crop_x1 = max(0, int(center_x - context_width / 2))
            crop_y1 = max(0, int(center_y - context_height / 2))
            crop_x2 = min(width, int(center_x + context_width / 2))
            crop_y2 = min(height, int(center_y + context_height / 2))

            if crop_x2 <= crop_x1 or crop_y2 <= crop_y1:
                continue

            crop = image[crop_y1:crop_y2, crop_x1:crop_x2].copy()

            local_x1 = max(0, xmin - crop_x1)
            local_y1 = max(0, ymin - crop_y1)
            local_x2 = min(crop.shape[1] - 1, xmax - crop_x1)
            local_y2 = min(crop.shape[0] - 1, ymax - crop_y1)

            color = COLORS.get(class_name, (0, 255, 0))
            thickness = max(2, round(min(crop.shape[:2]) / 150))

            cv2.rectangle(
                crop,
                (local_x1, local_y1),
                (local_x2, local_y2),
                color,
                thickness,
            )

            label = f"{class_name} {box_width}x{box_height}"
            cv2.putText(
                crop,
                label,
                (8, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                color,
                2,
                cv2.LINE_AA,
            )

            scale = min(
                OUTPUT_SIZE / crop.shape[1],
                OUTPUT_SIZE / crop.shape[0],
            )

            resized_width = max(1, round(crop.shape[1] * scale))
            resized_height = max(1, round(crop.shape[0] * scale))

            enlarged = cv2.resize(
                crop,
                (resized_width, resized_height),
                interpolation=(
                    cv2.INTER_CUBIC if scale > 1 else cv2.INTER_AREA
                ),
            )

            output_name = (
                f"{class_name}__{stem}__obj{index:02d}"
                f"__{box_width}x{box_height}.jpg"
            )

            output_path = OUTPUT_DIR / output_name
            cv2.imwrite(
                str(output_path),
                enlarged,
                [cv2.IMWRITE_JPEG_QUALITY, 95],
            )

            saved += 1

    print(f"已生成局部放大图：{saved}")
    print(f"输出目录：{OUTPUT_DIR}")


if __name__ == "__main__":
    main()
