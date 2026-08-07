from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from collections import Counter
from pathlib import Path


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

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an oversampled YOLO dataset using "
            "unique symlink aliases for rare-class tiles."
        )
    )

    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/"
            "datasets/tiles_1280_full"
        ),
        help="Baseline-1 全量切片数据集。",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/"
            "datasets/tiles_1280_rareos_v1"
        ),
        help="过采样后的数据集。",
    )

    parser.add_argument(
        "--metadata-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/"
            "metadata/rare_oversample_v1"
        ),
        help="过采样统计和审计文件。",
    )

    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/"
            "configs/steel_tiles_1280_rareos_v1.yaml"
        ),
        help="Baseline-2 Ultralytics YAML。",
    )

    parser.add_argument(
        "--oversample",
        type=str,
        default="qilie=4,huashang=2",
        help=(
            "目标类别总曝光倍数。"
            "例如 qilie=4 表示原始样本+3份别名。"
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有输出。",
    )

    return parser.parse_args()


def parse_oversample_spec(
    text: str,
) -> dict[str, int]:
    result: dict[str, int] = {}

    for item in text.split(","):
        item = item.strip()

        if not item:
            continue

        if "=" not in item:
            raise ValueError(
                f"非法 oversample 项：{item}"
            )

        name, multiplier_text = (
            item.split("=", 1)
        )

        name = name.strip()
        multiplier = int(
            multiplier_text.strip()
        )

        if name not in CLASS_TO_ID:
            raise ValueError(
                f"未知类别：{name}"
            )

        if multiplier < 1:
            raise ValueError(
                f"{name} 倍率必须 >= 1"
            )

        result[name] = multiplier

    if not result:
        raise ValueError(
            "没有指定任何过采样类别"
        )

    return result


def prepare_directory(
    path: Path,
    overwrite: bool,
) -> None:
    if path.exists() and any(path.iterdir()):
        if not overwrite:
            raise RuntimeError(
                f"输出目录非空：{path}\n"
                "确认后添加 --overwrite"
            )

        shutil.rmtree(path)

    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def create_symlink(
    source: Path,
    destination: Path,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if os.path.lexists(destination):
        destination.unlink()

    relative_target = os.path.relpath(
        source.resolve(),
        start=destination.parent.resolve(),
    )

    destination.symlink_to(
        relative_target
    )


def parse_label(
    label_path: Path,
) -> tuple[
    list[int],
    Counter[int],
]:
    text = label_path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return [], Counter()

    class_ids: list[int] = []
    box_counter: Counter[int] = Counter()

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        parts = line.split()

        if len(parts) != 5:
            raise RuntimeError(
                f"{label_path}:{line_number} "
                f"列数不是 5"
            )

        class_id = int(parts[0])

        if (
            class_id < 0
            or class_id >= len(CLASS_NAMES)
        ):
            raise RuntimeError(
                f"{label_path}:{line_number} "
                f"类别越界：{class_id}"
            )

        coords = [
            float(value)
            for value in parts[1:]
        ]

        if not all(
            0 <= value <= 1
            for value in coords
        ):
            raise RuntimeError(
                f"{label_path}:{line_number} "
                f"坐标越界：{coords}"
            )

        class_ids.append(class_id)
        box_counter[class_id] += 1

    return class_ids, box_counter


def find_images(
    images_dir: Path,
) -> list[Path]:
    return sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower()
        in IMAGE_EXTENSIONS
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


def write_yaml(
    path: Path,
    output_root: Path,
) -> None:
    path.parent.mkdir(
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

    path.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    source_root = (
        args.source_root.resolve()
    )

    output_root = (
        args.output_root.resolve()
    )

    metadata_root = (
        args.metadata_root.resolve()
    )

    config_path = (
        args.config_path.resolve()
    )

    multipliers = (
        parse_oversample_spec(
            args.oversample
        )
    )

    print(
        "========== 阶段 7B-1：Rare-class Oversampling =========="
    )

    print(
        "源数据：",
        source_root,
    )

    print(
        "输出数据：",
        output_root,
    )

    print("\n目标倍率：")

    for class_name, multiplier in (
        multipliers.items()
    ):
        print(
            f"  {class_name:<18}"
            f"x{multiplier}"
        )

    prepare_directory(
        output_root,
        args.overwrite,
    )

    prepare_directory(
        metadata_root,
        args.overwrite,
    )

    # -------------------------------------------------
    # 统计容器
    # -------------------------------------------------

    base_tile_counter = Counter()
    final_tile_counter = Counter()

    base_box_counter = Counter()
    final_box_counter = Counter()

    duplicate_rows: list[
        dict[str, object]
    ] = []

    train_tile_rows: list[
        dict[str, object]
    ] = []

    overlap_target_counter = Counter()

    base_train_count = 0
    final_train_count = 0

    base_val_count = 0
    final_val_count = 0

    # -------------------------------------------------
    # Train
    # -------------------------------------------------

    train_images_dir = (
        source_root
        / "images"
        / "train"
    )

    train_labels_dir = (
        source_root
        / "labels"
        / "train"
    )

    train_images = find_images(
        train_images_dir
    )

    print(
        f"\nBaseline Train tiles："
        f"{len(train_images)}"
    )

    for index, image_path in enumerate(
        train_images,
        start=1,
    ):
        label_path = (
            train_labels_dir
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            raise FileNotFoundError(
                f"缺少标签：{label_path}"
            )

        class_ids, box_counter = (
            parse_label(
                label_path
            )
        )

        unique_class_ids = sorted(
            set(class_ids)
        )

        unique_class_names = [
            CLASS_NAMES[class_id]
            for class_id
            in unique_class_ids
        ]

        # ---------------------------------------------
        # 原始样本始终保留一次
        # ---------------------------------------------

        base_image_destination = (
            output_root
            / "images"
            / "train"
            / image_path.name
        )

        base_label_destination = (
            output_root
            / "labels"
            / "train"
            / label_path.name
        )

        create_symlink(
            image_path,
            base_image_destination,
        )

        create_symlink(
            label_path,
            base_label_destination,
        )

        base_train_count += 1
        final_train_count += 1

        for class_id in unique_class_ids:
            class_name = (
                CLASS_NAMES[class_id]
            )

            base_tile_counter[
                class_name
            ] += 1

            final_tile_counter[
                class_name
            ] += 1

        for class_id, count in (
            box_counter.items()
        ):
            class_name = (
                CLASS_NAMES[class_id]
            )

            base_box_counter[
                class_name
            ] += count

            final_box_counter[
                class_name
            ] += count

        # ---------------------------------------------
        # 确定这张 tile 应有的总倍率
        #
        # 若同时包含多个目标类别：
        # 使用其中最大的 multiplier。
        #
        # 不能对同一图片删掉其他真实标签，
        # 所以其他共现类别也会随 tile 一起重复。
        # ---------------------------------------------

        matched_targets = [
            class_name
            for class_name
            in unique_class_names
            if class_name
            in multipliers
        ]

        if matched_targets:
            total_multiplier = max(
                multipliers[
                    class_name
                ]
                for class_name
                in matched_targets
            )
        else:
            total_multiplier = 1

        if len(
            matched_targets
        ) >= 2:
            combination = "+".join(
                sorted(
                    matched_targets
                )
            )

            overlap_target_counter[
                combination
            ] += 1

        duplicate_count = (
            total_multiplier - 1
        )

        train_tile_rows.append({
            "source_image": (
                image_path.name
            ),
            "classes": ",".join(
                unique_class_names
            ),
            "matched_targets": ",".join(
                matched_targets
            ),
            "total_multiplier": (
                total_multiplier
            ),
            "duplicate_count": (
                duplicate_count
            ),
            "box_count": sum(
                box_counter.values()
            ),
        })

        # ---------------------------------------------
        # 创建别名重复样本
        # ---------------------------------------------

        for duplicate_index in range(
            1,
            duplicate_count + 1,
        ):
            alias_stem = (
                f"{image_path.stem}"
                f"__rareos"
                f"_x{total_multiplier}"
                f"_{duplicate_index:02d}"
            )

            alias_image_name = (
                alias_stem
                + image_path.suffix.lower()
            )

            alias_label_name = (
                alias_stem
                + ".txt"
            )

            alias_image_path = (
                output_root
                / "images"
                / "train"
                / alias_image_name
            )

            alias_label_path = (
                output_root
                / "labels"
                / "train"
                / alias_label_name
            )

            create_symlink(
                image_path,
                alias_image_path,
            )

            create_symlink(
                label_path,
                alias_label_path,
            )

            final_train_count += 1

            for class_id in (
                unique_class_ids
            ):
                class_name = (
                    CLASS_NAMES[
                        class_id
                    ]
                )

                final_tile_counter[
                    class_name
                ] += 1

            for class_id, count in (
                box_counter.items()
            ):
                class_name = (
                    CLASS_NAMES[
                        class_id
                    ]
                )

                final_box_counter[
                    class_name
                ] += count

            duplicate_rows.append({
                "alias_image": (
                    alias_image_name
                ),
                "source_image": (
                    image_path.name
                ),
                "alias_label": (
                    alias_label_name
                ),
                "source_label": (
                    label_path.name
                ),
                "duplicate_index": (
                    duplicate_index
                ),
                "total_multiplier": (
                    total_multiplier
                ),
                "trigger_classes": (
                    ",".join(
                        matched_targets
                    )
                ),
                "all_classes": (
                    ",".join(
                        unique_class_names
                    )
                ),
                "box_count": sum(
                    box_counter.values()
                ),
            })

        if (
            index % 1000 == 0
            or index
            == len(train_images)
        ):
            print(
                f"已处理 Train "
                f"{index}/"
                f"{len(train_images)}"
            )

    # -------------------------------------------------
    # Val：严格保持原样，只创建一次软链接
    # -------------------------------------------------

    val_images_dir = (
        source_root
        / "images"
        / "val"
    )

    val_labels_dir = (
        source_root
        / "labels"
        / "val"
    )

    val_images = find_images(
        val_images_dir
    )

    print(
        f"\nBaseline Val tiles："
        f"{len(val_images)}"
    )

    for image_path in val_images:
        label_path = (
            val_labels_dir
            / f"{image_path.stem}.txt"
        )

        if not label_path.exists():
            raise FileNotFoundError(
                f"缺少 Val 标签："
                f"{label_path}"
            )

        create_symlink(
            image_path,
            output_root
            / "images"
            / "val"
            / image_path.name,
        )

        create_symlink(
            label_path,
            output_root
            / "labels"
            / "val"
            / label_path.name,
        )

        base_val_count += 1
        final_val_count += 1

    # -------------------------------------------------
    # 完整性检查
    # -------------------------------------------------

    train_output_images = find_images(
        output_root
        / "images"
        / "train"
    )

    train_output_labels = sorted(
        (
            output_root
            / "labels"
            / "train"
        ).glob("*.txt")
    )

    val_output_images = find_images(
        output_root
        / "images"
        / "val"
    )

    val_output_labels = sorted(
        (
            output_root
            / "labels"
            / "val"
        ).glob("*.txt")
    )

    train_image_stems = {
        path.stem
        for path
        in train_output_images
    }

    train_label_stems = {
        path.stem
        for path
        in train_output_labels
    }

    val_image_stems = {
        path.stem
        for path
        in val_output_images
    }

    val_label_stems = {
        path.stem
        for path
        in val_output_labels
    }

    if (
        train_image_stems
        != train_label_stems
    ):
        raise RuntimeError(
            "Train 图片/标签不一一对应"
        )

    if (
        val_image_stems
        != val_label_stems
    ):
        raise RuntimeError(
            "Val 图片/标签不一一对应"
        )

    train_val_overlap = (
        train_image_stems
        & val_image_stems
    )

    if train_val_overlap:
        raise RuntimeError(
            "Train/Val 存在同名 tile："
            + ", ".join(
                sorted(
                    train_val_overlap
                )[:10]
            )
        )

    # Val 必须与 baseline 数量完全一致
    if len(
        val_output_images
    ) != len(
        val_images
    ):
        raise RuntimeError(
            "Val 数量发生变化"
        )

    # -------------------------------------------------
    # 统计表
    # -------------------------------------------------

    class_rows: list[
        dict[str, object]
    ] = []

    for class_name in CLASS_NAMES:
        base_tiles = (
            base_tile_counter[
                class_name
            ]
        )

        final_tiles = (
            final_tile_counter[
                class_name
            ]
        )

        base_boxes = (
            base_box_counter[
                class_name
            ]
        )

        final_boxes = (
            final_box_counter[
                class_name
            ]
        )

        class_rows.append({
            "class_name": (
                class_name
            ),
            "requested_multiplier": (
                multipliers.get(
                    class_name,
                    1,
                )
            ),
            "base_positive_tile_exposure": (
                base_tiles
            ),
            "final_positive_tile_exposure": (
                final_tiles
            ),
            "actual_tile_multiplier": (
                round(
                    final_tiles
                    / base_tiles,
                    4,
                )
                if base_tiles
                else ""
            ),
            "base_box_exposure": (
                base_boxes
            ),
            "final_box_exposure": (
                final_boxes
            ),
            "actual_box_multiplier": (
                round(
                    final_boxes
                    / base_boxes,
                    4,
                )
                if base_boxes
                else ""
            ),
            "extra_tile_exposure": (
                final_tiles
                - base_tiles
            ),
            "extra_box_exposure": (
                final_boxes
                - base_boxes
            ),
        })

    write_csv(
        metadata_root
        / "class_exposure.csv",
        class_rows,
        [
            "class_name",
            "requested_multiplier",
            "base_positive_tile_exposure",
            "final_positive_tile_exposure",
            "actual_tile_multiplier",
            "base_box_exposure",
            "final_box_exposure",
            "actual_box_multiplier",
            "extra_tile_exposure",
            "extra_box_exposure",
        ],
    )

    write_csv(
        metadata_root
        / "duplicate_index.csv",
        duplicate_rows,
        [
            "alias_image",
            "source_image",
            "alias_label",
            "source_label",
            "duplicate_index",
            "total_multiplier",
            "trigger_classes",
            "all_classes",
            "box_count",
        ],
    )

    write_csv(
        metadata_root
        / "train_tile_multipliers.csv",
        train_tile_rows,
        [
            "source_image",
            "classes",
            "matched_targets",
            "total_multiplier",
            "duplicate_count",
            "box_count",
        ],
    )

    overlap_rows = [
        {
            "target_combination": (
                combination
            ),
            "tile_count": count,
        }
        for combination, count
        in sorted(
            overlap_target_counter.items()
        )
    ]

    write_csv(
        metadata_root
        / "target_overlap.csv",
        overlap_rows,
        [
            "target_combination",
            "tile_count",
        ],
    )

    summary = {
        "experiment": (
            "rare_oversample_v1"
        ),
        "source_root": str(
            source_root
        ),
        "output_root": str(
            output_root
        ),
        "requested_multipliers": (
            multipliers
        ),
        "baseline_train_tiles": (
            base_train_count
        ),
        "final_train_tiles": (
            final_train_count
        ),
        "added_train_tiles": (
            final_train_count
            - base_train_count
        ),
        "train_growth_ratio": (
            final_train_count
            / base_train_count
        ),
        "baseline_val_tiles": (
            base_val_count
        ),
        "final_val_tiles": (
            final_val_count
        ),
        "val_unchanged": (
            base_val_count
            == final_val_count
        ),
        "duplicate_alias_count": (
            len(duplicate_rows)
        ),
        "target_overlap": dict(
            overlap_target_counter
        ),
        "train_val_overlap_count": (
            len(train_val_overlap)
        ),
        "class_exposure": {
            row["class_name"]: row
            for row in class_rows
        },
    }

    (
        metadata_root
        / "summary.json"
    ).write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    write_yaml(
        config_path,
        output_root,
    )

    # -------------------------------------------------
    # 控制台输出
    # -------------------------------------------------

    print()
    print(
        "========== Baseline-2 数据生成结果 =========="
    )

    print(
        f"Baseline Train："
        f"{base_train_count}"
    )

    print(
        f"Baseline-2 Train："
        f"{final_train_count}"
    )

    print(
        f"新增训练别名："
        f"{final_train_count - base_train_count}"
    )

    print(
        "Train 增长："
        f"{(final_train_count / base_train_count - 1) * 100:.2f}%"
    )

    print()
    print(
        f"Baseline Val："
        f"{base_val_count}"
    )

    print(
        f"Baseline-2 Val："
        f"{final_val_count}"
    )

    print(
        "Val 是否保持不变：",
        base_val_count
        == final_val_count,
    )

    print()
    print(
        "类别实际曝光："
    )

    print(
        f"{'class':<18}"
        f"{'base_tile':>11}"
        f"{'final_tile':>12}"
        f"{'tile_x':>9}"
        f"{'base_box':>11}"
        f"{'final_box':>12}"
        f"{'box_x':>9}"
    )

    for row in class_rows:
        print(
            f"{row['class_name']:<18}"
            f"{row['base_positive_tile_exposure']:>11}"
            f"{row['final_positive_tile_exposure']:>12}"
            f"{str(row['actual_tile_multiplier']):>9}"
            f"{row['base_box_exposure']:>11}"
            f"{row['final_box_exposure']:>12}"
            f"{str(row['actual_box_multiplier']):>9}"
        )

    print()
    print(
        "目标类别共现："
    )

    if overlap_target_counter:
        for combination, count in (
            overlap_target_counter.items()
        ):
            print(
                f"  {combination}: "
                f"{count} tiles"
            )
    else:
        print(
            "  qilie 与 huashang "
            "没有出现在同一 tile 中"
        )

    print()
    print(
        "完整性检查："
    )

    print(
        "  Train images:",
        len(train_output_images),
    )

    print(
        "  Train labels:",
        len(train_output_labels),
    )

    print(
        "  Val images:",
        len(val_output_images),
    )

    print(
        "  Val labels:",
        len(val_output_labels),
    )

    print(
        "  Train/Val overlap:",
        len(train_val_overlap),
    )

    print()
    print(
        "输出数据集：",
        output_root,
    )

    print(
        "配置文件：",
        config_path,
    )

    print(
        "统计目录：",
        metadata_root,
    )

    print()
    print(
        "阶段 7B-1 数据准备完成。"
    )

    print(
        "未复制图片本体；"
        "新增样本通过软链接别名实现。"
    )


if __name__ == "__main__":
    main()
