from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze Baseline-1 training curves, class imbalance, "
            "tile distribution, and ambiguous fragments."
        )
    )

    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("/root/autodl-tmp/steel_defect"),
    )

    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/"
            "runs/baseline/"
            "yolo26m_tiles1280_e80_b6_seed2026"
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/"
            "metadata/baseline_analysis"
        ),
    )

    return parser.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")

    df = pd.read_csv(path)
    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


def get_nested(
    data: dict,
    *keys,
    default=None,
):
    current = data

    for key in keys:
        if not isinstance(current, dict):
            return default

        if key not in current:
            return default

        current = current[key]

    return current


def save_csv(
    path: Path,
    rows: list[dict],
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


def save_metric_curve(
    df: pd.DataFrame,
    columns: list[str],
    output_path: Path,
    title: str,
    ylabel: str,
) -> None:
    available = [
        column
        for column in columns
        if column in df.columns
    ]

    if not available:
        return

    plt.figure(figsize=(10, 6))

    for column in available:
        plt.plot(
            df["epoch"],
            df[column],
            label=column,
        )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=180,
    )
    plt.close()


def save_single_curve(
    df: pd.DataFrame,
    column: str,
    output_path: Path,
    title: str,
    ylabel: str,
    best_epoch: int | None = None,
) -> None:
    if column not in df.columns:
        return

    plt.figure(figsize=(10, 6))

    plt.plot(
        df["epoch"],
        df[column],
        label=column,
    )

    if best_epoch is not None:
        matched = df[
            df["epoch"] == best_epoch
        ]

        if not matched.empty:
            value = float(
                matched.iloc[0][column]
            )

            plt.scatter(
                [best_epoch],
                [value],
                s=70,
                label=f"best epoch={best_epoch}",
            )

    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=180,
    )
    plt.close()


def linear_slope(
    x_values: list[float],
    y_values: list[float],
) -> float | None:
    if len(x_values) < 2:
        return None

    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)

    numerator = sum(
        (x - x_mean) * (y - y_mean)
        for x, y in zip(
            x_values,
            y_values,
        )
    )

    denominator = sum(
        (x - x_mean) ** 2
        for x in x_values
    )

    if denominator == 0:
        return None

    return numerator / denominator


def main() -> None:
    args = parse_args()

    project_root = args.project_root.resolve()
    run_dir = args.run_dir.resolve()
    output_dir = args.output_dir.resolve()

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_path = (
        run_dir / "results.csv"
    )

    split_summary_path = (
        project_root
        / "splits"
        / "split_summary.json"
    )

    tile_summary_path = (
        project_root
        / "metadata"
        / "tiles_1280_full"
        / "summary.json"
    )

    ambiguous_path = (
        project_root
        / "metadata"
        / "tiles_1280_full"
        / "ambiguous_fragments.csv"
    )

    source_summary_path = (
        project_root
        / "metadata"
        / "tiles_1280_full"
        / "source_summary.csv"
    )

    print(
        "========== 阶段 7A：Baseline 离线诊断 =========="
    )

    # -------------------------------------------------
    # 1. 训练曲线
    # -------------------------------------------------

    df = safe_read_csv(
        results_path
    )

    required_columns = [
        "epoch",
        "metrics/precision(B)",
        "metrics/recall(B)",
        "metrics/mAP50(B)",
        "metrics/mAP50-95(B)",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise RuntimeError(
                f"results.csv 缺少列：{column}"
            )

    metric_column = (
        "metrics/mAP50-95(B)"
    )

    best_index = (
        df[metric_column].idxmax()
    )

    best_row = df.loc[
        best_index
    ]

    last_row = df.iloc[-1]

    best_epoch = int(
        best_row["epoch"]
    )

    last_epoch = int(
        last_row["epoch"]
    )

    best_map50 = float(
        best_row["metrics/mAP50(B)"]
    )

    best_map5095 = float(
        best_row[
            "metrics/mAP50-95(B)"
        ]
    )

    last_map50 = float(
        last_row["metrics/mAP50(B)"]
    )

    last_map5095 = float(
        last_row[
            "metrics/mAP50-95(B)"
        ]
    )

    last_10 = df.tail(
        min(10, len(df))
    )

    map5095_slope = linear_slope(
        [
            float(value)
            for value in last_10["epoch"]
        ],
        [
            float(value)
            for value
            in last_10[
                "metrics/mAP50-95(B)"
            ]
        ],
    )

    map50_slope = linear_slope(
        [
            float(value)
            for value in last_10["epoch"]
        ],
        [
            float(value)
            for value
            in last_10[
                "metrics/mAP50(B)"
            ]
        ],
    )

    top_epochs = (
        df.sort_values(
            metric_column,
            ascending=False,
        )
        .head(10)
        .copy()
    )

    top_epoch_rows = []

    for _, row in (
        top_epochs.iterrows()
    ):
        top_epoch_rows.append({
            "epoch": int(row["epoch"]),
            "precision": float(
                row[
                    "metrics/precision(B)"
                ]
            ),
            "recall": float(
                row[
                    "metrics/recall(B)"
                ]
            ),
            "mAP50": float(
                row[
                    "metrics/mAP50(B)"
                ]
            ),
            "mAP50_95": float(
                row[
                    "metrics/mAP50-95(B)"
                ]
            ),
        })

    save_csv(
        output_dir
        / "top_epochs.csv",
        top_epoch_rows,
        [
            "epoch",
            "precision",
            "recall",
            "mAP50",
            "mAP50_95",
        ],
    )

    save_single_curve(
        df,
        "metrics/mAP50-95(B)",
        output_dir
        / "map50_95_curve.png",
        "Baseline-1 mAP50-95",
        "mAP50-95",
        best_epoch=best_epoch,
    )

    save_metric_curve(
        df,
        [
            "metrics/mAP50(B)",
            "metrics/mAP50-95(B)",
        ],
        output_dir
        / "map_curves.png",
        "Baseline-1 mAP Curves",
        "mAP",
    )

    save_metric_curve(
        df,
        [
            "metrics/precision(B)",
            "metrics/recall(B)",
        ],
        output_dir
        / "precision_recall_curves.png",
        "Baseline-1 Precision / Recall",
        "Metric",
    )

    save_metric_curve(
        df,
        [
            "train/box_loss",
            "val/box_loss",
        ],
        output_dir
        / "box_loss_curves.png",
        "Box Loss",
        "Loss",
    )

    save_metric_curve(
        df,
        [
            "train/cls_loss",
            "val/cls_loss",
        ],
        output_dir
        / "cls_loss_curves.png",
        "Classification Loss",
        "Loss",
    )

    save_metric_curve(
        df,
        [
            "train/l1_loss",
            "val/l1_loss",
        ],
        output_dir
        / "l1_loss_curves.png",
        "L1 Loss",
        "Loss",
    )

    # -------------------------------------------------
    # 2. 原图 Train/Val 类别统计
    # -------------------------------------------------

    if not split_summary_path.exists():
        raise FileNotFoundError(
            f"缺少：{split_summary_path}"
        )

    split_summary = json.loads(
        split_summary_path.read_text(
            encoding="utf-8"
        )
    )

    # -------------------------------------------------
    # 3. 全量切片统计
    # -------------------------------------------------

    if not tile_summary_path.exists():
        raise FileNotFoundError(
            f"缺少：{tile_summary_path}"
        )

    tile_summary = json.loads(
        tile_summary_path.read_text(
            encoding="utf-8"
        )
    )

    # -------------------------------------------------
    # 4. 模糊截断片段
    # -------------------------------------------------

    ambiguous_counter = {
        "train": Counter(),
        "val": Counter(),
    }

    ambiguous_tile_counter = {
        "train": defaultdict(set),
        "val": defaultdict(set),
    }

    if ambiguous_path.exists():
        ambiguous_df = pd.read_csv(
            ambiguous_path,
            encoding="utf-8-sig",
        )

        ambiguous_df.columns = [
            str(column).strip()
            for column
            in ambiguous_df.columns
        ]

        for _, row in (
            ambiguous_df.iterrows()
        ):
            split = str(
                row["split"]
            ).strip()

            class_name = str(
                row["class_name"]
            ).strip()

            if split not in (
                "train",
                "val",
            ):
                continue

            ambiguous_counter[
                split
            ][class_name] += 1

            tile_key = (
                str(
                    row["source_image"]
                ),
                int(
                    row["tile_x"]
                ),
                int(
                    row["tile_y"]
                ),
            )

            ambiguous_tile_counter[
                split
            ][class_name].add(
                tile_key
            )

    # -------------------------------------------------
    # 5. 原图级切片行为
    # -------------------------------------------------

    source_stats = {
        "train": {},
        "val": {},
    }

    if source_summary_path.exists():
        source_df = pd.read_csv(
            source_summary_path,
            encoding="utf-8-sig",
        )

        source_df.columns = [
            str(column).strip()
            for column
            in source_df.columns
        ]

        for split in [
            "train",
            "val",
        ]:
            part = source_df[
                source_df["split"]
                == split
            ]

            if part.empty:
                continue

            source_stats[
                split
            ] = {
                "source_images": int(
                    len(part)
                ),
                "grid_tiles": int(
                    part[
                        "grid_tile_count"
                    ].sum()
                ),
                "positive_tiles": int(
                    part[
                        "positive_tile_count"
                    ].sum()
                ),
                "ambiguous_tiles": int(
                    part[
                        "ambiguous_tile_count"
                    ].sum()
                ),
                "black_filtered_tiles": int(
                    part[
                        "black_filtered_tile_count"
                    ].sum()
                ),
                "tiny_intersection_tiles": int(
                    part[
                        "ignored_tiny_intersection_tiles"
                    ].sum()
                ),
            }

    # -------------------------------------------------
    # 6. 合并类别诊断
    # -------------------------------------------------

    train_original_images = (
        get_nested(
            split_summary,
            "train",
            "class_images",
            default={},
        )
        or {}
    )

    val_original_images = (
        get_nested(
            split_summary,
            "val",
            "class_images",
            default={},
        )
        or {}
    )

    train_original_boxes = (
        get_nested(
            split_summary,
            "train",
            "class_boxes",
            default={},
        )
        or {}
    )

    val_original_boxes = (
        get_nested(
            split_summary,
            "val",
            "class_boxes",
            default={},
        )
        or {}
    )

    train_tile_summary = (
        get_nested(
            tile_summary,
            "splits",
            "train",
            default={},
        )
        or {}
    )

    val_tile_summary = (
        get_nested(
            tile_summary,
            "splits",
            "val",
            default={},
        )
        or {}
    )

    train_positive_tiles = (
        train_tile_summary.get(
            "class_positive_tile_counts",
            {},
        )
        or {}
    )

    val_positive_tiles = (
        val_tile_summary.get(
            "class_positive_tile_counts",
            {},
        )
        or {}
    )

    train_fragments = (
        train_tile_summary.get(
            "class_box_fragment_counts",
            {},
        )
        or {}
    )

    val_fragments = (
        val_tile_summary.get(
            "class_box_fragment_counts",
            {},
        )
        or {}
    )

    max_train_images = max(
        [
            int(
                train_original_images.get(
                    class_name,
                    0,
                )
            )
            for class_name
            in CLASS_NAMES
        ]
        or [1]
    )

    max_train_boxes = max(
        [
            int(
                train_original_boxes.get(
                    class_name,
                    0,
                )
            )
            for class_name
            in CLASS_NAMES
        ]
        or [1]
    )

    class_rows = []

    for class_name in CLASS_NAMES:
        original_train_images = int(
            train_original_images.get(
                class_name,
                0,
            )
        )

        original_val_images = int(
            val_original_images.get(
                class_name,
                0,
            )
        )

        original_train_boxes = int(
            train_original_boxes.get(
                class_name,
                0,
            )
        )

        original_val_boxes = int(
            val_original_boxes.get(
                class_name,
                0,
            )
        )

        train_tile_count = int(
            train_positive_tiles.get(
                class_name,
                0,
            )
        )

        val_tile_count = int(
            val_positive_tiles.get(
                class_name,
                0,
            )
        )

        train_fragment_count = int(
            train_fragments.get(
                class_name,
                0,
            )
        )

        val_fragment_count = int(
            val_fragments.get(
                class_name,
                0,
            )
        )

        fragment_multiplier = (
            train_fragment_count
            / original_train_boxes
            if original_train_boxes
            else math.nan
        )

        image_imbalance_ratio = (
            max_train_images
            / original_train_images
            if original_train_images
            else math.inf
        )

        box_imbalance_ratio = (
            max_train_boxes
            / original_train_boxes
            if original_train_boxes
            else math.inf
        )

        ambiguous_train = (
            ambiguous_counter[
                "train"
            ][class_name]
        )

        ambiguous_val = (
            ambiguous_counter[
                "val"
            ][class_name]
        )

        ambiguous_train_tiles = len(
            ambiguous_tile_counter[
                "train"
            ][class_name]
        )

        ambiguous_val_tiles = len(
            ambiguous_tile_counter[
                "val"
            ][class_name]
        )

        scarcity_score = 0

        if original_train_images < 50:
            scarcity_score += 3
        elif original_train_images < 100:
            scarcity_score += 2
        elif original_train_images < 200:
            scarcity_score += 1

        if original_train_boxes < 100:
            scarcity_score += 3
        elif original_train_boxes < 250:
            scarcity_score += 2
        elif original_train_boxes < 500:
            scarcity_score += 1

        if class_name in LONG_CLASSES:
            scarcity_score += 1

        if ambiguous_train >= 10:
            scarcity_score += 1

        if scarcity_score >= 6:
            priority = "very_high"
        elif scarcity_score >= 4:
            priority = "high"
        elif scarcity_score >= 2:
            priority = "medium"
        else:
            priority = "normal"

        class_rows.append({
            "class_name": class_name,
            "is_long_class": (
                class_name
                in LONG_CLASSES
            ),
            "train_original_images": (
                original_train_images
            ),
            "val_original_images": (
                original_val_images
            ),
            "train_original_boxes": (
                original_train_boxes
            ),
            "val_original_boxes": (
                original_val_boxes
            ),
            "train_positive_tiles": (
                train_tile_count
            ),
            "val_positive_tiles": (
                val_tile_count
            ),
            "train_box_fragments": (
                train_fragment_count
            ),
            "val_box_fragments": (
                val_fragment_count
            ),
            "fragment_multiplier": (
                round(
                    fragment_multiplier,
                    4,
                )
                if math.isfinite(
                    fragment_multiplier
                )
                else ""
            ),
            "image_imbalance_vs_max": (
                round(
                    image_imbalance_ratio,
                    4,
                )
                if math.isfinite(
                    image_imbalance_ratio
                )
                else ""
            ),
            "box_imbalance_vs_max": (
                round(
                    box_imbalance_ratio,
                    4,
                )
                if math.isfinite(
                    box_imbalance_ratio
                )
                else ""
            ),
            "train_ambiguous_fragments": (
                ambiguous_train
            ),
            "train_ambiguous_tiles": (
                ambiguous_train_tiles
            ),
            "val_ambiguous_fragments": (
                ambiguous_val
            ),
            "val_ambiguous_tiles": (
                ambiguous_val_tiles
            ),
            "diagnostic_priority": (
                priority
            ),
        })

    class_diagnostics_path = (
        output_dir
        / "class_diagnostics.csv"
    )

    save_csv(
        class_diagnostics_path,
        class_rows,
        list(
            class_rows[0].keys()
        ),
    )

    # -------------------------------------------------
    # 7. 类别分布图
    # -------------------------------------------------

    class_df = pd.DataFrame(
        class_rows
    )

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        class_df["class_name"],
        class_df[
            "train_original_boxes"
        ],
    )

    plt.xticks(
        rotation=35,
        ha="right",
    )

    plt.ylabel(
        "Original Train Boxes"
    )

    plt.title(
        "Original Train Class Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "original_train_box_distribution.png",
        dpi=180,
    )

    plt.close()

    plt.figure(
        figsize=(12, 6)
    )

    plt.bar(
        class_df["class_name"],
        class_df[
            "train_box_fragments"
        ],
    )

    plt.xticks(
        rotation=35,
        ha="right",
    )

    plt.ylabel(
        "Train Tile Box Fragments"
    )

    plt.title(
        "Tile Fragment Class Distribution"
    )

    plt.tight_layout()

    plt.savefig(
        output_dir
        / "tile_fragment_distribution.png",
        dpi=180,
    )

    plt.close()

    # -------------------------------------------------
    # 8. 生成 Markdown 报告
    # -------------------------------------------------

    priority_order = {
        "very_high": 0,
        "high": 1,
        "medium": 2,
        "normal": 3,
    }

    sorted_classes = sorted(
        class_rows,
        key=lambda row: (
            priority_order[
                row[
                    "diagnostic_priority"
                ]
            ],
            row[
                "train_original_images"
            ],
        ),
    )

    report_lines = []

    report_lines.append(
        "# Baseline-1 Offline Diagnostic Report"
    )

    report_lines.append("")

    report_lines.append(
        "## Training summary"
    )

    report_lines.append("")

    report_lines.append(
        f"- Epochs completed: {len(df)}"
    )

    report_lines.append(
        f"- Best epoch: {best_epoch}"
    )

    report_lines.append(
        f"- Best Precision: "
        f"{float(best_row['metrics/precision(B)']):.6f}"
    )

    report_lines.append(
        f"- Best Recall: "
        f"{float(best_row['metrics/recall(B)']):.6f}"
    )

    report_lines.append(
        f"- Best mAP50: "
        f"{best_map50:.6f}"
    )

    report_lines.append(
        f"- Best mAP50-95: "
        f"{best_map5095:.6f}"
    )

    report_lines.append(
        f"- Last epoch: {last_epoch}"
    )

    report_lines.append(
        f"- Last mAP50: "
        f"{last_map50:.6f}"
    )

    report_lines.append(
        f"- Last mAP50-95: "
        f"{last_map5095:.6f}"
    )

    report_lines.append(
        f"- Best -> last mAP50 change: "
        f"{last_map50 - best_map50:+.6f}"
    )

    report_lines.append(
        f"- Best -> last mAP50-95 change: "
        f"{last_map5095 - best_map5095:+.6f}"
    )

    if (
        map5095_slope
        is not None
    ):
        report_lines.append(
            f"- Last-10-epoch mAP50-95 slope: "
            f"{map5095_slope:+.8f} / epoch"
        )

    if (
        map50_slope
        is not None
    ):
        report_lines.append(
            f"- Last-10-epoch mAP50 slope: "
            f"{map50_slope:+.8f} / epoch"
        )

    report_lines.append("")

    report_lines.append(
        "## Dataset / tiling summary"
    )

    report_lines.append("")

    for split in [
        "train",
        "val",
    ]:
        tile_part = (
            tile_summary[
                "splits"
            ][split]
        )

        report_lines.append(
            f"### {split}"
        )

        report_lines.append("")

        report_lines.append(
            f"- Source images: "
            f"{tile_part['source_image_count']}"
        )

        report_lines.append(
            f"- Positive tiles: "
            f"{tile_part['positive_tile_count']}"
        )

        report_lines.append(
            f"- Selected negative tiles: "
            f"{tile_part['selected_negative_tile_count']}"
        )

        report_lines.append(
            f"- Total tiles: "
            f"{tile_part['total_tile_count']}"
        )

        report_lines.append(
            f"- Ambiguous tiles skipped: "
            f"{tile_part['ambiguous_tile_count']}"
        )

        report_lines.append(
            f"- Black negative tiles filtered: "
            f"{tile_part['black_filtered_tile_count']}"
        )

        report_lines.append("")

    report_lines.append(
        "## Class diagnostics"
    )

    report_lines.append("")

    report_lines.append(
        "| class | train imgs | train boxes | train tiles | fragments | fragment x | ambiguous | long | priority |"
    )

    report_lines.append(
        "|---|---:|---:|---:|---:|---:|---:|:---:|:---:|"
    )

    for row in sorted_classes:
        report_lines.append(
            "| "
            f"{row['class_name']} | "
            f"{row['train_original_images']} | "
            f"{row['train_original_boxes']} | "
            f"{row['train_positive_tiles']} | "
            f"{row['train_box_fragments']} | "
            f"{row['fragment_multiplier']} | "
            f"{row['train_ambiguous_fragments']} | "
            f"{'yes' if row['is_long_class'] else 'no'} | "
            f"{row['diagnostic_priority']} |"
        )

    report_lines.append("")

    report_lines.append(
        "## Interpretation flags"
    )

    report_lines.append("")

    high_priority = [
        row
        for row in sorted_classes
        if row[
            "diagnostic_priority"
        ] in {
            "very_high",
            "high",
        }
    ]

    if high_priority:
        report_lines.append(
            "- High-priority classes for the next experiment:"
        )

        for row in high_priority:
            report_lines.append(
                f"  - {row['class_name']}: "
                f"{row['train_original_images']} original train images, "
                f"{row['train_original_boxes']} original boxes, "
                f"{row['train_box_fragments']} tile fragments, "
                f"{row['train_ambiguous_fragments']} ambiguous fragments."
            )

    report_lines.append("")

    report_lines.append(
        "- This report is diagnostic only. "
        "It does not modify labels, sampling, augmentation, or model weights."
    )

    report_path = (
        output_dir
        / "report.md"
    )

    report_path.write_text(
        "\n".join(report_lines)
        + "\n",
        encoding="utf-8",
    )

    # -------------------------------------------------
    # 9. 控制台摘要
    # -------------------------------------------------

    print()
    print(
        "========== Baseline 训练趋势 =========="
    )

    print(
        f"训练 epoch：{len(df)}"
    )

    print(
        f"最佳 epoch：{best_epoch}"
    )

    print(
        f"Best mAP50：{best_map50:.6f}"
    )

    print(
        f"Best mAP50-95：{best_map5095:.6f}"
    )

    print(
        f"Last mAP50：{last_map50:.6f}"
    )

    print(
        f"Last mAP50-95：{last_map5095:.6f}"
    )

    print(
        "Best→Last mAP50-95："
        f"{last_map5095 - best_map5095:+.6f}"
    )

    if map5095_slope is not None:
        print(
            "最后 10 epoch mAP50-95 slope："
            f"{map5095_slope:+.8f}/epoch"
        )

    print()
    print(
        "========== 类别诊断 =========="
    )

    print(
        f"{'class':<18}"
        f"{'orig_img':>10}"
        f"{'orig_box':>10}"
        f"{'tile':>9}"
        f"{'fragment':>10}"
        f"{'frag_x':>9}"
        f"{'ambig':>8}"
        f"{'priority':>12}"
    )

    for row in sorted_classes:
        print(
            f"{row['class_name']:<18}"
            f"{row['train_original_images']:>10}"
            f"{row['train_original_boxes']:>10}"
            f"{row['train_positive_tiles']:>9}"
            f"{row['train_box_fragments']:>10}"
            f"{str(row['fragment_multiplier']):>9}"
            f"{row['train_ambiguous_fragments']:>8}"
            f"{row['diagnostic_priority']:>12}"
        )

    print()
    print(
        "========== 输出文件 =========="
    )

    print(
        "报告：",
        report_path,
    )

    print(
        "类别诊断：",
        class_diagnostics_path,
    )

    print(
        "Top epochs：",
        output_dir
        / "top_epochs.csv",
    )

    print(
        "mAP 曲线：",
        output_dir
        / "map_curves.png",
    )

    print(
        "mAP50-95 曲线：",
        output_dir
        / "map50_95_curve.png",
    )

    print(
        "Precision/Recall：",
        output_dir
        / "precision_recall_curves.png",
    )

    print(
        "Loss 曲线目录：",
        output_dir,
    )

    print()
    print(
        "阶段 7A 完成：未使用 GPU，未修改任何训练数据或权重。"
    )


if __name__ == "__main__":
    main()
