from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import re
import shutil
from collections import Counter
from dataclasses import dataclass, field
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
    class_name: class_id
    for class_id, class_name in enumerate(CLASS_NAMES)
}

DEFAULT_MIN_VAL_IMAGES = {
    "jieba": 1,
    "zonglie": 1,
    "qilie": 4,
    "jiaza": 15,
    "yiwuyaru": 1,
    "huashang": 8,
    "mamianmakeng": 1,
    "yanghuatiepi": 1,
    "gunyin": 1,
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


@dataclass
class Sample:
    stem: str
    image_path: Path
    label_path: Path
    group_id: str
    group_rule: str
    classes: set[str]
    box_counts: Counter[str]
    is_empty: bool


@dataclass
class Group:
    group_id: str
    rules: set[str] = field(default_factory=set)
    samples: list[Sample] = field(default_factory=list)
    image_count: int = 0
    empty_count: int = 0
    class_image_counts: Counter[str] = field(
        default_factory=Counter
    )
    class_box_counts: Counter[str] = field(
        default_factory=Counter
    )


@dataclass
class Metrics:
    image_count: int = 0
    empty_count: int = 0
    class_image_counts: Counter[str] = field(
        default_factory=Counter
    )
    class_box_counts: Counter[str] = field(
        default_factory=Counter
    )


@dataclass
class Targets:
    image_count: float
    empty_count: float
    class_image_counts: dict[str, float]
    class_box_counts: dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a leakage-safe grouped Train/Val split "
            "for the steel defect YOLO dataset."
        )
    )

    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/datasets/yolo_all"
        ),
        help="VOC 转 YOLO 后的完整数据目录。",
    )

    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/datasets/yolo_split"
        ),
        help="Train/Val 软链接数据集输出目录。",
    )

    parser.add_argument(
        "--splits-dir",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/splits"
        ),
        help="划分清单和统计结果目录。",
    )

    parser.add_argument(
        "--config-path",
        type=Path,
        default=Path(
            "/root/autodl-tmp/steel_defect/"
            "configs/steel_original.yaml"
        ),
        help="Ultralytics 数据配置文件。",
    )

    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.15,
        help="验证集比例，默认 0.15。",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="随机种子。",
    )

    parser.add_argument(
        "--restarts",
        type=int,
        default=120,
        help="分组划分随机重启次数。",
    )

    parser.add_argument(
        "--swap-steps",
        type=int,
        default=800,
        help="每次重启的局部交换优化次数。",
    )

    parser.add_argument(
        "--image-tolerance",
        type=int,
        default=12,
        help="验证集图片数相对目标值允许的偏差。",
    )

    parser.add_argument(
        "--c-group-level",
        choices=["specimen", "view"],
        default="specimen",
        help=(
            "C 类文件名分组方式。"
            "specimen 使用 C1294073；"
            "view 使用 C1294073_V03。"
        ),
    )

    parser.add_argument(
        "--rare-minima",
        type=str,
        default="qilie=4,huashang=8,jiaza=15",
        help=(
            "验证集稀有类别最少图片数，例如 "
            "qilie=4,huashang=8,jiaza=15。"
        ),
    )

    parser.add_argument(
        "--copy-files",
        action="store_true",
        help="复制图片和标签，而不是创建软链接。",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="覆盖已有的划分结果。",
    )

    return parser.parse_args()


def parse_rare_minima(text: str) -> dict[str, int]:
    minima = dict(DEFAULT_MIN_VAL_IMAGES)

    if not text.strip():
        return minima

    for item in text.split(","):
        item = item.strip()

        if not item:
            continue

        if "=" not in item:
            raise ValueError(
                f"非法 rare-minima 项：{item}"
            )

        class_name, value_text = item.split("=", 1)
        class_name = class_name.strip()
        value = int(value_text.strip())

        if class_name not in CLASS_TO_ID:
            raise ValueError(
                f"未知类别：{class_name}"
            )

        if value < 1:
            raise ValueError(
                f"最小验证图片数必须大于等于 1：{item}"
            )

        minima[class_name] = value

    return minima


def infer_group_id(
    stem: str,
    c_group_level: str,
) -> tuple[str, str]:
    """
    根据文件名推断关联批次。

    规则 1：
    0001388400-Raw12-f_00017
    -> group_id = 0001388400

    规则 2：
    C1294073_V03_F00005_uuid
    specimen -> C1294073
    view     -> C1294073_V03

    未匹配文件使用单图独立分组，避免错误合并。
    """

    numeric_raw_match = re.match(
        r"^(\d+)[-_]Raw\d+",
        stem,
        flags=re.IGNORECASE,
    )

    if numeric_raw_match:
        return (
            numeric_raw_match.group(1),
            "numeric_raw_prefix",
        )

    c_style_match = re.match(
        r"^(C\d+)_V(\d+)_F\d+",
        stem,
        flags=re.IGNORECASE,
    )

    if c_style_match:
        specimen_id = c_style_match.group(1).upper()
        view_id = c_style_match.group(2)

        if c_group_level == "view":
            return (
                f"{specimen_id}_V{view_id}",
                "c_specimen_view",
            )

        return (
            specimen_id,
            "c_specimen",
        )

    c_view_match = re.match(
        r"^(C\d+)_V(\d+)",
        stem,
        flags=re.IGNORECASE,
    )

    if c_view_match:
        specimen_id = c_view_match.group(1).upper()
        view_id = c_view_match.group(2)

        if c_group_level == "view":
            return (
                f"{specimen_id}_V{view_id}",
                "c_specimen_view_fallback",
            )

        return (
            specimen_id,
            "c_specimen_fallback",
        )

    return (
        f"singleton::{stem}",
        "fallback_singleton",
    )


def parse_yolo_label(
    label_path: Path,
) -> tuple[set[str], Counter[str]]:
    classes: set[str] = set()
    box_counts: Counter[str] = Counter()

    text = label_path.read_text(
        encoding="utf-8"
    ).strip()

    if not text:
        return classes, box_counts

    for line_number, line in enumerate(
        text.splitlines(),
        start=1,
    ):
        parts = line.strip().split()

        if len(parts) != 5:
            raise ValueError(
                f"{label_path.name}:{line_number} "
                f"应包含 5 列，实际为 {len(parts)}"
            )

        try:
            class_id = int(parts[0])
            values = [
                float(value)
                for value in parts[1:]
            ]
        except ValueError as exception:
            raise ValueError(
                f"{label_path.name}:{line_number} "
                f"存在非法数值"
            ) from exception

        if class_id < 0 or class_id >= len(CLASS_NAMES):
            raise ValueError(
                f"{label_path.name}:{line_number} "
                f"类别编号越界：{class_id}"
            )

        if not all(
            0.0 <= value <= 1.0
            for value in values
        ):
            raise ValueError(
                f"{label_path.name}:{line_number} "
                f"归一化坐标越界：{values}"
            )

        if values[2] <= 0 or values[3] <= 0:
            raise ValueError(
                f"{label_path.name}:{line_number} "
                "目标宽高必须大于零"
            )

        class_name = CLASS_NAMES[class_id]
        classes.add(class_name)
        box_counts[class_name] += 1

    return classes, box_counts


def load_samples(
    source_root: Path,
    c_group_level: str,
) -> list[Sample]:
    images_dir = source_root / "images"
    labels_dir = source_root / "labels"

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

    if not image_paths:
        raise RuntimeError(
            f"没有在目录中找到图片：{images_dir}"
        )

    seen_stems: set[str] = set()
    samples: list[Sample] = []

    for image_path in image_paths:
        stem = image_path.stem

        if stem in seen_stems:
            raise RuntimeError(
                f"发现重复图片 stem：{stem}"
            )

        seen_stems.add(stem)

        label_path = labels_dir / f"{stem}.txt"

        if not label_path.exists():
            raise FileNotFoundError(
                f"缺少 YOLO 标签：{label_path}"
            )

        classes, box_counts = parse_yolo_label(
            label_path
        )

        group_id, group_rule = infer_group_id(
            stem,
            c_group_level,
        )

        samples.append(
            Sample(
                stem=stem,
                image_path=image_path,
                label_path=label_path,
                group_id=group_id,
                group_rule=group_rule,
                classes=classes,
                box_counts=box_counts,
                is_empty=len(box_counts) == 0,
            )
        )

    label_stems = {
        path.stem
        for path in labels_dir.glob("*.txt")
    }

    extra_labels = sorted(
        label_stems - seen_stems
    )

    if extra_labels:
        raise RuntimeError(
            "存在没有同名图片的标签，例如："
            + ", ".join(extra_labels[:10])
        )

    return samples


def build_groups(
    samples: list[Sample],
) -> dict[str, Group]:
    groups: dict[str, Group] = {}

    for sample in samples:
        group = groups.setdefault(
            sample.group_id,
            Group(group_id=sample.group_id),
        )

        group.rules.add(sample.group_rule)
        group.samples.append(sample)
        group.image_count += 1

        if sample.is_empty:
            group.empty_count += 1

        for class_name in sample.classes:
            group.class_image_counts[class_name] += 1

        group.class_box_counts.update(
            sample.box_counts
        )

    return groups


def compute_totals(
    groups: dict[str, Group],
) -> Metrics:
    metrics = Metrics()

    for group in groups.values():
        apply_group(
            metrics,
            group,
            sign=1,
        )

    return metrics


def clone_metrics(
    metrics: Metrics,
) -> Metrics:
    return Metrics(
        image_count=metrics.image_count,
        empty_count=metrics.empty_count,
        class_image_counts=Counter(
            metrics.class_image_counts
        ),
        class_box_counts=Counter(
            metrics.class_box_counts
        ),
    )


def apply_group(
    metrics: Metrics,
    group: Group,
    sign: int,
) -> None:
    metrics.image_count += (
        sign * group.image_count
    )

    metrics.empty_count += (
        sign * group.empty_count
    )

    for class_name, count in (
        group.class_image_counts.items()
    ):
        metrics.class_image_counts[class_name] += (
            sign * count
        )

    for class_name, count in (
        group.class_box_counts.items()
    ):
        metrics.class_box_counts[class_name] += (
            sign * count
        )


def metrics_with_group(
    metrics: Metrics,
    group: Group,
    sign: int,
) -> Metrics:
    result = clone_metrics(metrics)
    apply_group(result, group, sign)
    return result


def build_targets(
    totals: Metrics,
    val_ratio: float,
) -> Targets:
    return Targets(
        image_count=totals.image_count * val_ratio,
        empty_count=totals.empty_count * val_ratio,
        class_image_counts={
            class_name: (
                totals.class_image_counts[class_name]
                * val_ratio
            )
            for class_name in CLASS_NAMES
        },
        class_box_counts={
            class_name: (
                totals.class_box_counts[class_name]
                * val_ratio
            )
            for class_name in CLASS_NAMES
        },
    )


def build_rarity_weights(
    totals: Metrics,
) -> dict[str, float]:
    maximum = max(
        totals.class_image_counts[class_name]
        for class_name in CLASS_NAMES
    )

    weights: dict[str, float] = {}

    for class_name in CLASS_NAMES:
        count = max(
            1,
            totals.class_image_counts[class_name],
        )

        weights[class_name] = min(
            4.0,
            math.sqrt(maximum / count),
        )

    return weights


def score_metrics(
    metrics: Metrics,
    targets: Targets,
    totals: Metrics,
    minima: dict[str, int],
    rarity_weights: dict[str, float],
) -> float:
    score = 0.0

    image_denominator = max(
        1.0,
        targets.image_count,
    )

    image_error = (
        metrics.image_count - targets.image_count
    ) / image_denominator

    score += 12.0 * image_error * image_error

    empty_denominator = max(
        1.0,
        targets.empty_count,
    )

    empty_error = (
        metrics.empty_count - targets.empty_count
    ) / empty_denominator

    score += 2.5 * empty_error * empty_error

    for class_name in CLASS_NAMES:
        target_class_images = max(
            1.0,
            targets.class_image_counts[class_name],
        )

        class_image_error = (
            metrics.class_image_counts[class_name]
            - targets.class_image_counts[class_name]
        ) / target_class_images

        weight = rarity_weights[class_name]

        score += (
            1.8
            * weight
            * class_image_error
            * class_image_error
        )

        target_class_boxes = max(
            1.0,
            targets.class_box_counts[class_name],
        )

        class_box_error = (
            metrics.class_box_counts[class_name]
            - targets.class_box_counts[class_name]
        ) / target_class_boxes

        score += (
            0.30
            * weight
            * class_box_error
            * class_box_error
        )

        minimum = minima[class_name]
        current = metrics.class_image_counts[class_name]

        if current < minimum:
            deficit = minimum - current

            score += (
                250.0
                * (deficit / max(1, minimum)) ** 2
            )

        train_class_images = (
            totals.class_image_counts[class_name]
            - current
        )

        if train_class_images < 1:
            score += 10000.0

    if metrics.image_count <= 0:
        score += 10000.0

    if metrics.image_count >= totals.image_count:
        score += 10000.0

    return score


def can_add_group(
    metrics: Metrics,
    group: Group,
    totals: Metrics,
) -> bool:
    for class_name, count in (
        group.class_image_counts.items()
    ):
        new_val_count = (
            metrics.class_image_counts[class_name]
            + count
        )

        remaining_train = (
            totals.class_image_counts[class_name]
            - new_val_count
        )

        if remaining_train < 1:
            return False

    return True


def can_remove_group(
    metrics: Metrics,
    group: Group,
    minima: dict[str, int],
) -> bool:
    for class_name in CLASS_NAMES:
        new_val_count = (
            metrics.class_image_counts[class_name]
            - group.class_image_counts[class_name]
        )

        if new_val_count < minima[class_name]:
            return False

    return True


def group_priority(
    group: Group,
    totals: Metrics,
    rarity_weights: dict[str, float],
) -> float:
    priority = 0.0

    for class_name in CLASS_NAMES:
        class_total = max(
            1,
            totals.class_image_counts[class_name],
        )

        priority += (
            rarity_weights[class_name]
            * group.class_image_counts[class_name]
            / class_total
        )

    if totals.empty_count:
        priority += (
            0.15
            * group.empty_count
            / totals.empty_count
        )

    return priority


def repair_minima(
    selected: set[str],
    metrics: Metrics,
    groups: dict[str, Group],
    totals: Metrics,
    targets: Targets,
    minima: dict[str, int],
    rarity_weights: dict[str, float],
) -> None:
    for class_name in CLASS_NAMES:
        while (
            metrics.class_image_counts[class_name]
            < minima[class_name]
        ):
            candidates: list[
                tuple[float, str, Metrics]
            ] = []

            for group_id, group in groups.items():
                if group_id in selected:
                    continue

                if (
                    group.class_image_counts[class_name]
                    <= 0
                ):
                    continue

                if not can_add_group(
                    metrics,
                    group,
                    totals,
                ):
                    continue

                candidate_metrics = metrics_with_group(
                    metrics,
                    group,
                    sign=1,
                )

                candidate_score = score_metrics(
                    candidate_metrics,
                    targets,
                    totals,
                    minima,
                    rarity_weights,
                )

                candidates.append(
                    (
                        candidate_score,
                        group_id,
                        candidate_metrics,
                    )
                )

            if not candidates:
                raise RuntimeError(
                    f"无法满足验证集类别 {class_name} "
                    f"最少 {minima[class_name]} 张图片。"
                    "尝试使用 --c-group-level view。"
                )

            candidates.sort(
                key=lambda item: item[0]
            )

            _, best_group_id, best_metrics = (
                candidates[0]
            )

            selected.add(best_group_id)

            metrics.image_count = (
                best_metrics.image_count
            )
            metrics.empty_count = (
                best_metrics.empty_count
            )
            metrics.class_image_counts = Counter(
                best_metrics.class_image_counts
            )
            metrics.class_box_counts = Counter(
                best_metrics.class_box_counts
            )


def fill_to_target(
    selected: set[str],
    metrics: Metrics,
    groups: dict[str, Group],
    totals: Metrics,
    targets: Targets,
    minima: dict[str, int],
    rarity_weights: dict[str, float],
    lower_bound: int,
) -> None:
    while metrics.image_count < lower_bound:
        candidates: list[
            tuple[float, int, str, Metrics]
        ] = []

        for group_id, group in groups.items():
            if group_id in selected:
                continue

            if not can_add_group(
                metrics,
                group,
                totals,
            ):
                continue

            candidate_metrics = metrics_with_group(
                metrics,
                group,
                sign=1,
            )

            candidate_score = score_metrics(
                candidate_metrics,
                targets,
                totals,
                minima,
                rarity_weights,
            )

            overshoot = max(
                0,
                candidate_metrics.image_count
                - round(targets.image_count),
            )

            candidates.append(
                (
                    candidate_score,
                    overshoot,
                    group_id,
                    candidate_metrics,
                )
            )

        if not candidates:
            break

        candidates.sort(
            key=lambda item: (
                item[0],
                item[1],
            )
        )

        _, _, best_group_id, best_metrics = (
            candidates[0]
        )

        selected.add(best_group_id)

        metrics.image_count = (
            best_metrics.image_count
        )
        metrics.empty_count = (
            best_metrics.empty_count
        )
        metrics.class_image_counts = Counter(
            best_metrics.class_image_counts
        )
        metrics.class_box_counts = Counter(
            best_metrics.class_box_counts
        )


def trim_to_target(
    selected: set[str],
    metrics: Metrics,
    groups: dict[str, Group],
    totals: Metrics,
    targets: Targets,
    minima: dict[str, int],
    rarity_weights: dict[str, float],
    upper_bound: int,
) -> None:
    while metrics.image_count > upper_bound:
        candidates: list[
            tuple[float, str, Metrics]
        ] = []

        for group_id in sorted(selected):
            group = groups[group_id]

            if not can_remove_group(
                metrics,
                group,
                minima,
            ):
                continue

            candidate_metrics = metrics_with_group(
                metrics,
                group,
                sign=-1,
            )

            candidate_score = score_metrics(
                candidate_metrics,
                targets,
                totals,
                minima,
                rarity_weights,
            )

            candidates.append(
                (
                    candidate_score,
                    group_id,
                    candidate_metrics,
                )
            )

        if not candidates:
            break

        candidates.sort(
            key=lambda item: item[0]
        )

        _, best_group_id, best_metrics = (
            candidates[0]
        )

        selected.remove(best_group_id)

        metrics.image_count = (
            best_metrics.image_count
        )
        metrics.empty_count = (
            best_metrics.empty_count
        )
        metrics.class_image_counts = Counter(
            best_metrics.class_image_counts
        )
        metrics.class_box_counts = Counter(
            best_metrics.class_box_counts
        )


def optimize_swaps(
    selected: set[str],
    metrics: Metrics,
    groups: dict[str, Group],
    totals: Metrics,
    targets: Targets,
    minima: dict[str, int],
    rarity_weights: dict[str, float],
    rng: random.Random,
    swap_steps: int,
    lower_bound: int,
    upper_bound: int,
) -> None:
    current_score = score_metrics(
        metrics,
        targets,
        totals,
        minima,
        rarity_weights,
    )

    all_group_ids = list(groups)

    for _ in range(swap_steps):
        val_ids = list(selected)
        train_ids = [
            group_id
            for group_id in all_group_ids
            if group_id not in selected
        ]

        if not val_ids or not train_ids:
            return

        remove_id = rng.choice(val_ids)
        add_id = rng.choice(train_ids)

        remove_group = groups[remove_id]
        add_group = groups[add_id]

        candidate_metrics = clone_metrics(metrics)

        apply_group(
            candidate_metrics,
            remove_group,
            sign=-1,
        )

        apply_group(
            candidate_metrics,
            add_group,
            sign=1,
        )

        if (
            candidate_metrics.image_count
            < lower_bound
            or candidate_metrics.image_count
            > upper_bound
        ):
            continue

        valid = True

        for class_name in CLASS_NAMES:
            val_count = (
                candidate_metrics
                .class_image_counts[class_name]
            )

            train_count = (
                totals.class_image_counts[class_name]
                - val_count
            )

            if val_count < minima[class_name]:
                valid = False
                break

            if train_count < 1:
                valid = False
                break

        if not valid:
            continue

        candidate_score = score_metrics(
            candidate_metrics,
            targets,
            totals,
            minima,
            rarity_weights,
        )

        if candidate_score + 1e-12 < current_score:
            selected.remove(remove_id)
            selected.add(add_id)

            metrics.image_count = (
                candidate_metrics.image_count
            )
            metrics.empty_count = (
                candidate_metrics.empty_count
            )
            metrics.class_image_counts = Counter(
                candidate_metrics
                .class_image_counts
            )
            metrics.class_box_counts = Counter(
                candidate_metrics
                .class_box_counts
            )

            current_score = candidate_score


def create_candidate_split(
    groups: dict[str, Group],
    totals: Metrics,
    targets: Targets,
    minima: dict[str, int],
    rarity_weights: dict[str, float],
    rng: random.Random,
    image_tolerance: int,
    swap_steps: int,
) -> tuple[set[str], Metrics, float]:
    target_images = round(targets.image_count)

    lower_bound = max(
        1,
        target_images - image_tolerance,
    )

    upper_bound = min(
        totals.image_count - 1,
        target_images + image_tolerance,
    )

    group_ids = list(groups)

    rng.shuffle(group_ids)

    group_ids.sort(
        key=lambda group_id: (
            group_priority(
                groups[group_id],
                totals,
                rarity_weights,
            )
            + rng.random() * 0.03
        ),
        reverse=True,
    )

    selected: set[str] = set()
    metrics = Metrics()

    for group_id in group_ids:
        group = groups[group_id]

        if not can_add_group(
            metrics,
            group,
            totals,
        ):
            continue

        candidate_metrics = metrics_with_group(
            metrics,
            group,
            sign=1,
        )

        current_score = score_metrics(
            metrics,
            targets,
            totals,
            minima,
            rarity_weights,
        )

        candidate_score = score_metrics(
            candidate_metrics,
            targets,
            totals,
            minima,
            rarity_weights,
        )

        if candidate_score < current_score:
            selected.add(group_id)
            metrics = candidate_metrics

    repair_minima(
        selected=selected,
        metrics=metrics,
        groups=groups,
        totals=totals,
        targets=targets,
        minima=minima,
        rarity_weights=rarity_weights,
    )

    fill_to_target(
        selected=selected,
        metrics=metrics,
        groups=groups,
        totals=totals,
        targets=targets,
        minima=minima,
        rarity_weights=rarity_weights,
        lower_bound=lower_bound,
    )

    trim_to_target(
        selected=selected,
        metrics=metrics,
        groups=groups,
        totals=totals,
        targets=targets,
        minima=minima,
        rarity_weights=rarity_weights,
        upper_bound=upper_bound,
    )

    optimize_swaps(
        selected=selected,
        metrics=metrics,
        groups=groups,
        totals=totals,
        targets=targets,
        minima=minima,
        rarity_weights=rarity_weights,
        rng=rng,
        swap_steps=swap_steps,
        lower_bound=lower_bound,
        upper_bound=upper_bound,
    )

    final_score = score_metrics(
        metrics,
        targets,
        totals,
        minima,
        rarity_weights,
    )

    return selected, metrics, final_score


def check_group_feasibility(
    groups: dict[str, Group],
    totals: Metrics,
    minima: dict[str, int],
) -> None:
    for class_name in CLASS_NAMES:
        class_groups = [
            group
            for group in groups.values()
            if group.class_image_counts[class_name] > 0
        ]

        if len(class_groups) < 2:
            raise RuntimeError(
                f"类别 {class_name} 只分布在 "
                f"{len(class_groups)} 个分组中，"
                "无法同时放入 Train 和 Val。"
                "尝试使用 --c-group-level view。"
            )

        if (
            totals.class_image_counts[class_name]
            <= minima[class_name]
        ):
            raise RuntimeError(
                f"类别 {class_name} 总图片数不足以满足 "
                f"Val 最少 {minima[class_name]} 张，"
                "同时保留 Train 样本。"
            )


def prepare_output_directory(
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

    path.mkdir(
        parents=True,
        exist_ok=True,
    )


def create_file_reference(
    source: Path,
    destination: Path,
    copy_files: bool,
) -> None:
    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if os.path.lexists(destination):
        destination.unlink()

    if copy_files:
        shutil.copy2(
            source.resolve(),
            destination,
        )
        return

    relative_target = os.path.relpath(
        source.resolve(),
        start=destination.parent.resolve(),
    )

    destination.symlink_to(
        relative_target
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


def create_split_files(
    samples: list[Sample],
    groups: dict[str, Group],
    val_group_ids: set[str],
    output_root: Path,
    splits_dir: Path,
    copy_files: bool,
    overwrite: bool,
) -> tuple[list[Sample], list[Sample]]:
    prepare_output_directory(
        output_root,
        overwrite=overwrite,
    )

    splits_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_samples = sorted(
        [
            sample
            for sample in samples
            if sample.group_id not in val_group_ids
        ],
        key=lambda sample: sample.image_path.name,
    )

    val_samples = sorted(
        [
            sample
            for sample in samples
            if sample.group_id in val_group_ids
        ],
        key=lambda sample: sample.image_path.name,
    )

    for split_name, split_samples in [
        ("train", train_samples),
        ("val", val_samples),
    ]:
        for sample in split_samples:
            image_destination = (
                output_root
                / "images"
                / split_name
                / sample.image_path.name
            )

            label_destination = (
                output_root
                / "labels"
                / split_name
                / sample.label_path.name
            )

            create_file_reference(
                sample.image_path,
                image_destination,
                copy_files=copy_files,
            )

            create_file_reference(
                sample.label_path,
                label_destination,
                copy_files=copy_files,
            )

    train_manifest = (
        splits_dir / "train_original.txt"
    )

    val_manifest = (
        splits_dir / "val_original.txt"
    )

    train_manifest.write_text(
        "\n".join(
            str(
                (
                    output_root
                    / "images"
                    / "train"
                    / sample.image_path.name
                ).absolute()
            )
            for sample in train_samples
        )
        + "\n",
        encoding="utf-8",
    )

    val_manifest.write_text(
        "\n".join(
            str(
                (
                    output_root
                    / "images"
                    / "val"
                    / sample.image_path.name
                ).absolute()
            )
            for sample in val_samples
        )
        + "\n",
        encoding="utf-8",
    )

    train_group_ids = sorted(
        set(groups) - val_group_ids
    )

    val_group_ids_sorted = sorted(
        val_group_ids
    )

    (
        splits_dir / "train_groups.txt"
    ).write_text(
        "\n".join(train_group_ids) + "\n",
        encoding="utf-8",
    )

    (
        splits_dir / "val_groups.txt"
    ).write_text(
        "\n".join(val_group_ids_sorted) + "\n",
        encoding="utf-8",
    )

    return train_samples, val_samples


def metrics_from_samples(
    samples: list[Sample],
) -> Metrics:
    metrics = Metrics()

    for sample in samples:
        metrics.image_count += 1

        if sample.is_empty:
            metrics.empty_count += 1

        for class_name in sample.classes:
            metrics.class_image_counts[class_name] += 1

        metrics.class_box_counts.update(
            sample.box_counts
        )

    return metrics


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


def validate_split(
    all_samples: list[Sample],
    train_samples: list[Sample],
    val_samples: list[Sample],
    minima: dict[str, int],
) -> None:
    all_stems = {
        sample.stem
        for sample in all_samples
    }

    train_stems = {
        sample.stem
        for sample in train_samples
    }

    val_stems = {
        sample.stem
        for sample in val_samples
    }

    train_groups = {
        sample.group_id
        for sample in train_samples
    }

    val_groups = {
        sample.group_id
        for sample in val_samples
    }

    errors: list[str] = []

    if train_stems & val_stems:
        errors.append(
            "Train 和 Val 存在重复图片"
        )

    if train_groups & val_groups:
        errors.append(
            "Train 和 Val 存在重复 group_id"
        )

    if train_stems | val_stems != all_stems:
        errors.append(
            "Train 与 Val 合并后未覆盖全部图片"
        )

    if (
        len(train_samples) + len(val_samples)
        != len(all_samples)
    ):
        errors.append(
            "Train 与 Val 图片数量之和不正确"
        )

    train_metrics = metrics_from_samples(
        train_samples
    )

    val_metrics = metrics_from_samples(
        val_samples
    )

    for class_name in CLASS_NAMES:
        if (
            train_metrics
            .class_image_counts[class_name]
            < 1
        ):
            errors.append(
                f"Train 缺少类别：{class_name}"
            )

        if (
            val_metrics
            .class_image_counts[class_name]
            < minima[class_name]
        ):
            errors.append(
                f"Val 类别 {class_name} 图片不足："
                f"{val_metrics.class_image_counts[class_name]}"
                f" < {minima[class_name]}"
            )

    if errors:
        raise RuntimeError(
            "划分验证失败：\n- "
            + "\n- ".join(errors)
        )


def write_reports(
    samples: list[Sample],
    groups: dict[str, Group],
    train_samples: list[Sample],
    val_samples: list[Sample],
    val_group_ids: set[str],
    splits_dir: Path,
    totals: Metrics,
    targets: Targets,
    final_score: float,
    seed: int,
    val_ratio: float,
    c_group_level: str,
) -> None:
    assignment_rows: list[
        dict[str, object]
    ] = []

    for sample in sorted(
        samples,
        key=lambda item: item.image_path.name,
    ):
        split_name = (
            "val"
            if sample.group_id in val_group_ids
            else "train"
        )

        assignment_rows.append({
            "image": sample.image_path.name,
            "label": sample.label_path.name,
            "group_id": sample.group_id,
            "group_rule": sample.group_rule,
            "split": split_name,
            "is_empty": sample.is_empty,
            "classes": ",".join(
                sorted(sample.classes)
            ),
            "box_count": sum(
                sample.box_counts.values()
            ),
        })

    write_csv(
        splits_dir / "sample_assignments.csv",
        assignment_rows,
        [
            "image",
            "label",
            "group_id",
            "group_rule",
            "split",
            "is_empty",
            "classes",
            "box_count",
        ],
    )

    group_rows: list[
        dict[str, object]
    ] = []

    group_fieldnames = [
        "group_id",
        "rules",
        "split",
        "image_count",
        "empty_count",
        "total_box_count",
    ]

    for class_name in CLASS_NAMES:
        group_fieldnames.extend([
            f"{class_name}_image_count",
            f"{class_name}_box_count",
        ])

    for group_id in sorted(groups):
        group = groups[group_id]

        row: dict[str, object] = {
            "group_id": group_id,
            "rules": ",".join(
                sorted(group.rules)
            ),
            "split": (
                "val"
                if group_id in val_group_ids
                else "train"
            ),
            "image_count": group.image_count,
            "empty_count": group.empty_count,
            "total_box_count": sum(
                group.class_box_counts.values()
            ),
        }

        for class_name in CLASS_NAMES:
            row[
                f"{class_name}_image_count"
            ] = group.class_image_counts[
                class_name
            ]

            row[
                f"{class_name}_box_count"
            ] = group.class_box_counts[
                class_name
            ]

        group_rows.append(row)

    write_csv(
        splits_dir / "group_audit.csv",
        group_rows,
        group_fieldnames,
    )

    train_metrics = metrics_from_samples(
        train_samples
    )

    val_metrics = metrics_from_samples(
        val_samples
    )

    summary_rows: list[
        dict[str, object]
    ] = []

    for split_name, metrics in [
        ("train", train_metrics),
        ("val", val_metrics),
    ]:
        split_group_count = len({
            sample.group_id
            for sample in (
                train_samples
                if split_name == "train"
                else val_samples
            )
        })

        summary_rows.append({
            "split": split_name,
            "class_name": "__all__",
            "image_count": metrics.image_count,
            "box_count": sum(
                metrics.class_box_counts.values()
            ),
            "group_count": split_group_count,
            "empty_image_count": (
                metrics.empty_count
            ),
            "image_ratio": (
                metrics.image_count
                / totals.image_count
            ),
            "class_image_ratio": "",
            "class_box_ratio": "",
        })

        summary_rows.append({
            "split": split_name,
            "class_name": "__empty__",
            "image_count": metrics.empty_count,
            "box_count": 0,
            "group_count": split_group_count,
            "empty_image_count": (
                metrics.empty_count
            ),
            "image_ratio": (
                metrics.empty_count
                / max(1, totals.empty_count)
            ),
            "class_image_ratio": "",
            "class_box_ratio": "",
        })

        for class_name in CLASS_NAMES:
            summary_rows.append({
                "split": split_name,
                "class_name": class_name,
                "image_count": (
                    metrics.class_image_counts[
                        class_name
                    ]
                ),
                "box_count": (
                    metrics.class_box_counts[
                        class_name
                    ]
                ),
                "group_count": split_group_count,
                "empty_image_count": (
                    metrics.empty_count
                ),
                "image_ratio": (
                    metrics.image_count
                    / totals.image_count
                ),
                "class_image_ratio": (
                    metrics.class_image_counts[
                        class_name
                    ]
                    / max(
                        1,
                        totals.class_image_counts[
                            class_name
                        ],
                    )
                ),
                "class_box_ratio": (
                    metrics.class_box_counts[
                        class_name
                    ]
                    / max(
                        1,
                        totals.class_box_counts[
                            class_name
                        ],
                    )
                ),
            })

    write_csv(
        splits_dir / "split_summary.csv",
        summary_rows,
        [
            "split",
            "class_name",
            "image_count",
            "box_count",
            "group_count",
            "empty_image_count",
            "image_ratio",
            "class_image_ratio",
            "class_box_ratio",
        ],
    )

    summary_json = {
        "seed": seed,
        "requested_val_ratio": val_ratio,
        "c_group_level": c_group_level,
        "objective_score": final_score,
        "total": {
            "images": totals.image_count,
            "empty_images": totals.empty_count,
            "boxes": sum(
                totals.class_box_counts.values()
            ),
            "groups": len(groups),
        },
        "target_val": {
            "images": targets.image_count,
            "empty_images": targets.empty_count,
            "class_images": (
                targets.class_image_counts
            ),
            "class_boxes": (
                targets.class_box_counts
            ),
        },
        "train": {
            "images": train_metrics.image_count,
            "empty_images": train_metrics.empty_count,
            "groups": len({
                sample.group_id
                for sample in train_samples
            }),
            "class_images": dict(
                train_metrics.class_image_counts
            ),
            "class_boxes": dict(
                train_metrics.class_box_counts
            ),
        },
        "val": {
            "images": val_metrics.image_count,
            "empty_images": val_metrics.empty_count,
            "groups": len({
                sample.group_id
                for sample in val_samples
            }),
            "class_images": dict(
                val_metrics.class_image_counts
            ),
            "class_boxes": dict(
                val_metrics.class_box_counts
            ),
        },
    }

    (
        splits_dir / "split_summary.json"
    ).write_text(
        json.dumps(
            summary_json,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    if not 0.05 <= args.val_ratio <= 0.40:
        raise ValueError(
            "--val-ratio 建议位于 0.05 到 0.40"
        )

    minima = parse_rare_minima(
        args.rare_minima
    )

    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    splits_dir = args.splits_dir.resolve()
    config_path = args.config_path.resolve()

    print("========== 加载 YOLO 数据 ==========")
    print("源目录：", source_root)
    print("C 文件分组级别：", args.c_group_level)

    samples = load_samples(
        source_root=source_root,
        c_group_level=args.c_group_level,
    )

    groups = build_groups(samples)
    totals = compute_totals(groups)

    print(f"图片总数：{len(samples)}")
    print(f"分组总数：{len(groups)}")
    print(f"空标签图片：{totals.empty_count}")
    print(
        "目标框总数：",
        sum(totals.class_box_counts.values()),
    )

    rule_counter = Counter(
        sample.group_rule
        for sample in samples
    )

    print("\n文件名分组规则：")
    for rule_name, count in sorted(
        rule_counter.items()
    ):
        print(
            f"  {rule_name:<28} {count}"
        )

    group_sizes = sorted(
        (
            group.image_count,
            group.group_id,
        )
        for group in groups.values()
    )

    print("\n最大分组：")
    for size, group_id in reversed(
        group_sizes[-10:]
    ):
        print(
            f"  {group_id:<35} {size} 张"
        )

    minima_display = ", ".join(
        f"{class_name}={minima[class_name]}"
        for class_name in CLASS_NAMES
    )

    print("\n验证集最低类别图片要求：")
    print(" ", minima_display)

    check_group_feasibility(
        groups=groups,
        totals=totals,
        minima=minima,
    )

    targets = build_targets(
        totals=totals,
        val_ratio=args.val_ratio,
    )

    rarity_weights = build_rarity_weights(
        totals
    )

    print("\n========== 优化分组划分 ==========")
    print(
        f"目标验证图片数："
        f"{targets.image_count:.1f}"
    )
    print(
        f"随机重启次数：{args.restarts}"
    )

    best_selected: set[str] | None = None
    best_metrics: Metrics | None = None
    best_score = float("inf")

    for restart_index in range(
        args.restarts
    ):
        restart_seed = (
            args.seed
            + restart_index * 1009
        )

        rng = random.Random(restart_seed)

        selected, metrics, score = (
            create_candidate_split(
                groups=groups,
                totals=totals,
                targets=targets,
                minima=minima,
                rarity_weights=rarity_weights,
                rng=rng,
                image_tolerance=(
                    args.image_tolerance
                ),
                swap_steps=args.swap_steps,
            )
        )

        if score < best_score:
            best_selected = set(selected)
            best_metrics = clone_metrics(metrics)
            best_score = score

        if (
            (restart_index + 1) % 20 == 0
            or restart_index == 0
        ):
            print(
                f"重启 {restart_index + 1:>3}/"
                f"{args.restarts}: "
                f"当前最佳 score={best_score:.6f}, "
                f"val_images="
                f"{best_metrics.image_count if best_metrics else 0}"
            )

    if best_selected is None:
        raise RuntimeError(
            "没有生成有效划分"
        )

    print("\n========== 生成软链接数据集 ==========")

    train_samples, val_samples = (
        create_split_files(
            samples=samples,
            groups=groups,
            val_group_ids=best_selected,
            output_root=output_root,
            splits_dir=splits_dir,
            copy_files=args.copy_files,
            overwrite=args.overwrite,
        )
    )

    validate_split(
        all_samples=samples,
        train_samples=train_samples,
        val_samples=val_samples,
        minima=minima,
    )

    write_yaml(
        config_path=config_path,
        output_root=output_root,
    )

    write_reports(
        samples=samples,
        groups=groups,
        train_samples=train_samples,
        val_samples=val_samples,
        val_group_ids=best_selected,
        splits_dir=splits_dir,
        totals=totals,
        targets=targets,
        final_score=best_score,
        seed=args.seed,
        val_ratio=args.val_ratio,
        c_group_level=args.c_group_level,
    )

    train_metrics = metrics_from_samples(
        train_samples
    )

    val_metrics = metrics_from_samples(
        val_samples
    )

    train_groups = {
        sample.group_id
        for sample in train_samples
    }

    val_groups = {
        sample.group_id
        for sample in val_samples
    }

    print("\n========== 最终划分结果 ==========")
    print(
        f"Train 图片："
        f"{train_metrics.image_count}"
    )
    print(
        f"Val 图片：  "
        f"{val_metrics.image_count}"
    )
    print(
        f"实际 Val 比例："
        f"{val_metrics.image_count / totals.image_count:.4f}"
    )
    print(
        f"Train 分组：{len(train_groups)}"
    )
    print(
        f"Val 分组：  {len(val_groups)}"
    )
    print(
        f"Train 空标签："
        f"{train_metrics.empty_count}"
    )
    print(
        f"Val 空标签：  "
        f"{val_metrics.empty_count}"
    )
    print(
        f"图片重复数："
        f"{len({s.stem for s in train_samples} & {s.stem for s in val_samples})}"
    )
    print(
        f"分组重复数："
        f"{len(train_groups & val_groups)}"
    )

    print("\n类别分布：")
    print(
        f"{'类别':<18}"
        f"{'Train 图':>10}"
        f"{'Val 图':>10}"
        f"{'Val比例':>10}"
        f"{'Train框':>10}"
        f"{'Val框':>10}"
    )

    for class_name in CLASS_NAMES:
        total_class_images = (
            totals.class_image_counts[
                class_name
            ]
        )

        val_class_images = (
            val_metrics.class_image_counts[
                class_name
            ]
        )

        print(
            f"{class_name:<18}"
            f"{train_metrics.class_image_counts[class_name]:>10}"
            f"{val_class_images:>10}"
            f"{val_class_images / max(1, total_class_images):>10.3f}"
            f"{train_metrics.class_box_counts[class_name]:>10}"
            f"{val_metrics.class_box_counts[class_name]:>10}"
        )

    print("\n输出内容：")
    print("数据集：", output_root)
    print("训练清单：", splits_dir / "train_original.txt")
    print("验证清单：", splits_dir / "val_original.txt")
    print("图片分配：", splits_dir / "sample_assignments.csv")
    print("分组审计：", splits_dir / "group_audit.csv")
    print("划分统计：", splits_dir / "split_summary.csv")
    print("划分 JSON：", splits_dir / "split_summary.json")
    print("Ultralytics 配置：", config_path)

    print("\n检查通过：")
    print("  Train/Val 图片无重叠")
    print("  Train/Val group_id 无重叠")
    print("  3200 张图片全部覆盖")
    print("  9 个类别均存在于 Train 和 Val")


if __name__ == "__main__":
    main()
