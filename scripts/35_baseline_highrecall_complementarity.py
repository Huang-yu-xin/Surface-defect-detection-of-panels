#!/usr/bin/env python3

from pathlib import Path
import argparse
import ast
import csv
import json
import re
from collections import Counter, defaultdict

import numpy as np


# ============================================================
# Candidate format
#
# According to the current cache:
#   col 0      = class id
#   col 1      = score
#   col 10:14  = global xyxy
#
# candidates shape = (N, 14)
# ============================================================

CLS_COL = 0
SCORE_COL = 1
GLOBAL_BOX_SLICE = slice(10, 14)


def parse_args():
    p = argparse.ArgumentParser(
        description="Audit Baseline High-Recall complementarity on Final Combo remaining FNs."
    )

    p.add_argument(
        "--fn-csv",
        type=Path,
        default=Path("results/final_combo_fn21/remaining_fn_21.csv"),
    )

    p.add_argument(
        "--baseline-cache",
        type=Path,
        default=Path("results/baseline_complementarity/cache_original"),
    )

    p.add_argument(
        "--rareos-original-cache",
        type=Path,
        default=Path("results/fn_analysis/cache"),
    )

    p.add_argument(
        "--rareos-hflip-cache",
        type=Path,
        default=Path("results/fn_analysis/cache_hflip"),
    )

    p.add_argument(
        "--dataset-yaml",
        type=Path,
        default=Path("configs/steel_tiles_1280.yaml"),
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/baseline_complementarity/audit35"),
    )

    return p.parse_args()


# ============================================================
# Generic helpers
# ============================================================

def norm_key(s):
    return re.sub(r"[^a-z0-9]+", "_", str(s).strip().lower()).strip("_")


def load_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fields = reader.fieldnames or []

    if not rows:
        raise RuntimeError(f"No rows found in {path}")

    return rows, fields


def find_field(fields, candidates):
    lookup = {norm_key(x): x for x in fields}

    for c in candidates:
        nc = norm_key(c)
        if nc in lookup:
            return lookup[nc]

    return None


def find_bbox_fields(fields):
    """
    Try common explicit GT xyxy names first, then look for fields
    sharing a prefix such as gt_box_x1, gt_box_y1, ...
    """

    explicit_sets = [
        ("gt_x1", "gt_y1", "gt_x2", "gt_y2"),
        ("gt_box_x1", "gt_box_y1", "gt_box_x2", "gt_box_y2"),
        ("gt_bbox_x1", "gt_bbox_y1", "gt_bbox_x2", "gt_bbox_y2"),
        ("x1", "y1", "x2", "y2"),
    ]

    lookup = {norm_key(x): x for x in fields}

    for names in explicit_sets:
        if all(norm_key(x) in lookup for x in names):
            return tuple(lookup[norm_key(x)] for x in names)

    # Dynamic prefix matching.
    normalized = [(f, norm_key(f)) for f in fields]

    suffix_map = defaultdict(dict)

    for original, nk in normalized:
        m = re.match(r"^(.*?)(?:_)?(x1|y1|x2|y2)$", nk)
        if not m:
            continue

        prefix = m.group(1).rstrip("_")
        coord = m.group(2)

        suffix_map[prefix][coord] = original

    # Prefer prefixes containing "gt".
    candidates = []

    for prefix, d in suffix_map.items():
        if all(k in d for k in ("x1", "y1", "x2", "y2")):
            priority = 0 if "gt" in prefix else 1
            candidates.append((priority, prefix, d))

    if candidates:
        candidates.sort(key=lambda x: (x[0], len(x[1])))
        d = candidates[0][2]

        return (
            d["x1"],
            d["y1"],
            d["x2"],
            d["y2"],
        )

    return None


def parse_box_string(v):
    try:
        x = ast.literal_eval(str(v))
        arr = np.asarray(x, dtype=float).reshape(-1)

        if len(arr) >= 4:
            return arr[:4]
    except Exception:
        pass

    nums = re.findall(
        r"[-+]?(?:\d*\.\d+|\d+\.?)(?:[eE][-+]?\d+)?",
        str(v),
    )

    if len(nums) >= 4:
        return np.asarray([float(x) for x in nums[:4]], dtype=float)

    return None


def load_class_names(path):
    if not path.exists():
        return {}, {}

    try:
        import yaml
    except Exception:
        return {}, {}

    with path.open("r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)

    names = obj.get("names") if isinstance(obj, dict) else None

    id_to_name = {}

    if isinstance(names, list):
        id_to_name = {i: str(x) for i, x in enumerate(names)}

    elif isinstance(names, dict):
        for k, v in names.items():
            try:
                k = int(k)
            except Exception:
                continue
            id_to_name[k] = str(v)

    name_to_id = {
        str(v).strip(): int(k)
        for k, v in id_to_name.items()
    }

    return id_to_name, name_to_id


def resolve_npz(cache_root, image_value):
    image_path = Path(str(image_value))
    stem = image_path.stem

    direct = cache_root / f"{stem}.npz"
    if direct.exists():
        return direct

    # Sometimes the CSV may already store a cache filename.
    if str(image_value).endswith(".npz"):
        direct2 = cache_root / Path(str(image_value)).name
        if direct2.exists():
            return direct2

    matches = list(cache_root.glob(f"{stem}.npz"))

    if len(matches) == 1:
        return matches[0]

    raise FileNotFoundError(
        f"Cannot resolve NPZ for image={image_value!r} under {cache_root}"
    )


def load_candidates(path):
    d = np.load(path, allow_pickle=False)

    if "candidates" not in d:
        raise KeyError(f"{path}: no 'candidates' key")

    c = np.asarray(d["candidates"], dtype=np.float32)

    if c.ndim != 2 or c.shape[1] < 14:
        raise ValueError(
            f"{path}: expected candidate shape (N, >=14), got {c.shape}"
        )

    return c


def iou_one_to_many(gt, boxes):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)

    gx1, gy1, gx2, gy2 = gt

    x1 = np.maximum(gx1, boxes[:, 0])
    y1 = np.maximum(gy1, boxes[:, 1])
    x2 = np.minimum(gx2, boxes[:, 2])
    y2 = np.minimum(gy2, boxes[:, 3])

    iw = np.maximum(0.0, x2 - x1)
    ih = np.maximum(0.0, y2 - y1)

    inter = iw * ih

    ga = max(0.0, gx2 - gx1) * max(0.0, gy2 - gy1)

    ba = (
        np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    )

    union = ga + ba - inter

    return np.divide(
        inter,
        union,
        out=np.zeros_like(inter, dtype=np.float32),
        where=union > 0,
    )


def best_same_class(candidates, class_id, gt):
    if len(candidates) == 0:
        return {
            "count": 0,
            "iou": 0.0,
            "score": 0.0,
            "box": None,
        }

    mask = candidates[:, CLS_COL].astype(np.int64) == int(class_id)
    same = candidates[mask]

    if len(same) == 0:
        return {
            "count": 0,
            "iou": 0.0,
            "score": 0.0,
            "box": None,
        }

    boxes = same[:, GLOBAL_BOX_SLICE]
    ious = iou_one_to_many(gt, boxes)

    j = int(np.argmax(ious))

    return {
        "count": int(len(same)),
        "iou": float(ious[j]),
        "score": float(same[j, SCORE_COL]),
        "box": boxes[j].tolist(),
    }


def bool01(v):
    return int(bool(v))


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("===== INPUT =====")
    print("FN CSV              :", args.fn_csv)
    print("Baseline cache      :", args.baseline_cache)
    print("RareOS original     :", args.rareos_original_cache)
    print("RareOS HFlip        :", args.rareos_hflip_cache)
    print()

    rows, fields = load_csv(args.fn_csv)

    print("FN rows             :", len(rows))
    print("CSV fields:")
    for f in fields:
        print("  ", f)
    print()

    image_field = find_field(
        fields,
        [
            "image",
            "image_name",
            "filename",
            "file_name",
            "img",
            "img_name",
        ],
    )

    class_id_field = find_field(
        fields,
        [
            "gt_class_id",
            "gt_cls",
            "gt_class_idx",
            "class_id",
            "cls",
            "gt_class",
        ],
    )

    class_name_field = find_field(
        fields,
        [
            "gt_class_name",
            "class_name",
            "gt_name",
            "category",
            "class",
        ],
    )

    failure_field = find_field(
        fields,
        [
            "failure_type",
            "failure",
            "fn_type",
            "reason",
            "diagnosis",
        ],
    )

    bbox_fields = find_bbox_fields(fields)

    bbox_string_field = find_field(
        fields,
        [
            "gt_xyxy",
            "gt_box_xyxy",
            "gt_bbox_xyxy",
        ],
    )

    if image_field is None:
        raise RuntimeError(
            "Could not detect image field.\n"
            f"Available fields: {fields}"
        )

    if bbox_fields is None and bbox_string_field is None:
        raise RuntimeError(
            "Could not detect GT xyxy fields.\n"
            f"Available fields: {fields}"
        )

    id_to_name, name_to_id = load_class_names(args.dataset_yaml)

    print("Detected:")
    print("  image field       :", image_field)
    print("  class id field    :", class_id_field)
    print("  class name field  :", class_name_field)
    print("  failure field     :", failure_field)
    print("  bbox fields       :", bbox_fields)
    print("  bbox string field :", bbox_string_field)
    print("  class names       :", id_to_name)
    print()

    # Check HFlip existence; script remains usable even if absent.
    use_hflip = args.rareos_hflip_cache.exists()

    if not use_hflip:
        print("WARNING: RareOS HFlip cache not found.")
        print("HFlip statistics will be set to zero.")
        print()

    out_rows = []

    for idx, row in enumerate(rows):
        image_value = row[image_field]

        # ----------------------------------------------------
        # GT class ID
        # ----------------------------------------------------

        class_id = None
        class_name = None

        if class_id_field is not None:
            raw = str(row[class_id_field]).strip()

            try:
                class_id = int(float(raw))
            except Exception:
                # Field may actually contain a class name.
                if raw in name_to_id:
                    class_id = name_to_id[raw]
                    class_name = raw

        if class_name_field is not None:
            cn = str(row[class_name_field]).strip()
            if cn:
                class_name = cn

        if class_id is None and class_name is not None:
            class_id = name_to_id.get(class_name)

        if class_id is None:
            raise RuntimeError(
                f"Row {idx}: cannot determine class id.\n"
                f"row={row}\n"
                f"class mapping={name_to_id}"
            )

        if class_name is None:
            class_name = id_to_name.get(class_id, str(class_id))

        # ----------------------------------------------------
        # GT xyxy
        # ----------------------------------------------------

        if bbox_fields is not None:
            gt = np.asarray(
                [
                    float(row[bbox_fields[0]]),
                    float(row[bbox_fields[1]]),
                    float(row[bbox_fields[2]]),
                    float(row[bbox_fields[3]]),
                ],
                dtype=np.float32,
            )
        else:
            gt = parse_box_string(row[bbox_string_field])

            if gt is None:
                raise RuntimeError(
                    f"Row {idx}: cannot parse GT box from "
                    f"{bbox_string_field}={row[bbox_string_field]!r}"
                )

        # ----------------------------------------------------
        # Load three proposal spaces
        # ----------------------------------------------------

        bpath = resolve_npz(args.baseline_cache, image_value)
        opath = resolve_npz(args.rareos_original_cache, image_value)

        baseline = load_candidates(bpath)
        rareos_o = load_candidates(opath)

        b = best_same_class(baseline, class_id, gt)
        o = best_same_class(rareos_o, class_id, gt)

        if use_hflip:
            hpath = resolve_npz(args.rareos_hflip_cache, image_value)
            rareos_h = load_candidates(hpath)
            h = best_same_class(rareos_h, class_id, gt)
        else:
            h = {
                "count": 0,
                "iou": 0.0,
                "score": 0.0,
                "box": None,
            }

        rareos_max_iou = max(o["iou"], h["iou"])

        baseline_direct_rescue = b["iou"] >= 0.50

        # Conservative definition:
        # Baseline can hit the GT, while neither RareOS Original nor
        # HFlip proposal space reaches the matching threshold.
        baseline_cache_unique_rescue = (
            b["iou"] >= 0.50
            and rareos_max_iou < 0.50
        )

        rareos_no_same_class = (
            o["count"] == 0
            and h["count"] == 0
        )

        baseline_hard_blindspot_rescue = (
            b["iou"] >= 0.50
            and rareos_no_same_class
        )

        baseline_deep_blindspot_rescue = (
            b["iou"] >= 0.50
            and rareos_max_iou < 0.30
        )

        failure = (
            str(row[failure_field]).strip()
            if failure_field is not None
            else ""
        )

        out_rows.append(
            {
                "row_id": idx,
                "image": image_value,

                "gt_class_id": class_id,
                "gt_class_name": class_name,

                "failure_type": failure,

                "gt_x1": float(gt[0]),
                "gt_y1": float(gt[1]),
                "gt_x2": float(gt[2]),
                "gt_y2": float(gt[3]),

                "baseline_same_class_count": b["count"],
                "baseline_best_iou": b["iou"],
                "baseline_best_score": b["score"],
                "baseline_best_box": json.dumps(b["box"]),

                "rareos_original_same_class_count": o["count"],
                "rareos_original_best_iou": o["iou"],
                "rareos_original_best_score": o["score"],

                "rareos_hflip_same_class_count": h["count"],
                "rareos_hflip_best_iou": h["iou"],
                "rareos_hflip_best_score": h["score"],

                "rareos_max_best_iou": rareos_max_iou,

                "baseline_iou_ge_030": bool01(b["iou"] >= 0.30),
                "baseline_iou_ge_040": bool01(b["iou"] >= 0.40),
                "baseline_iou_ge_050": bool01(b["iou"] >= 0.50),

                "baseline_direct_rescue":
                    bool01(baseline_direct_rescue),

                "baseline_cache_unique_rescue":
                    bool01(baseline_cache_unique_rescue),

                "rareos_no_same_class":
                    bool01(rareos_no_same_class),

                "baseline_hard_blindspot_rescue":
                    bool01(baseline_hard_blindspot_rescue),

                "baseline_deep_blindspot_rescue":
                    bool01(baseline_deep_blindspot_rescue),
            }
        )

    # ========================================================
    # Summary
    # ========================================================

    n = len(out_rows)

    ge030 = sum(x["baseline_iou_ge_030"] for x in out_rows)
    ge040 = sum(x["baseline_iou_ge_040"] for x in out_rows)
    ge050 = sum(x["baseline_iou_ge_050"] for x in out_rows)

    direct = sum(x["baseline_direct_rescue"] for x in out_rows)
    unique = sum(x["baseline_cache_unique_rescue"] for x in out_rows)

    hard_blind = sum(
        x["baseline_hard_blindspot_rescue"]
        for x in out_rows
    )

    deep_blind = sum(
        x["baseline_deep_blindspot_rescue"]
        for x in out_rows
    )

    # Rescue summaries.
    rescue_by_class = Counter()
    rescue_by_failure = Counter()

    blind_by_class = Counter()
    blind_by_failure = Counter()

    for x in out_rows:
        if x["baseline_direct_rescue"]:
            rescue_by_class[x["gt_class_name"]] += 1
            rescue_by_failure[x["failure_type"] or "(unknown)"] += 1

        if x["baseline_deep_blindspot_rescue"]:
            blind_by_class[x["gt_class_name"]] += 1
            blind_by_failure[x["failure_type"] or "(unknown)"] += 1

    # --------------------------------------------------------
    # Decision
    # --------------------------------------------------------

    failure_no_candidate_rescue = sum(
        1
        for x in out_rows
        if x["baseline_direct_rescue"]
        and "no_same_class_candidate" in x["failure_type"].lower()
    )

    if direct >= 3:
        decision = "GO"
        reason = (
            f"Baseline directly rescues {direct} Final Combo FN(s), "
            "meeting the >=3 independent-rescue threshold."
        )

    elif hard_blind >= 1 or failure_no_candidate_rescue >= 1:
        decision = "GO_CONDITIONAL"
        reason = (
            "Direct rescue is below 3, but Baseline reaches at least one "
            "RareOS hard blind spot / no-same-class-candidate FN."
        )

    else:
        decision = "NO_GO"
        reason = (
            f"Only {direct} direct rescue(s), with no confirmed hard "
            "RareOS proposal-space blind-spot rescue."
        )

    # ========================================================
    # Write detailed CSV
    # ========================================================

    csv_path = args.output_dir / "baseline_fn_complementarity.csv"

    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(out_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(out_rows)

    # Rescue-only CSV
    rescue_csv = args.output_dir / "baseline_rescued_fn.csv"

    rescued_rows = [
        x for x in out_rows
        if x["baseline_direct_rescue"]
    ]

    with rescue_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(out_rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rescued_rows)

    # ========================================================
    # Markdown summary
    # ========================================================

    summary_path = args.output_dir / "summary.md"

    lines = []

    lines.append("# Baseline High-Recall Complementarity Audit")
    lines.append("")

    lines.append("## Core result")
    lines.append("")
    lines.append("```text")
    lines.append(f"Final Combo remaining FN     = {n}")
    lines.append("")
    lines.append(f"Baseline best IoU >= 0.30    = {ge030}")
    lines.append(f"Baseline best IoU >= 0.40    = {ge040}")
    lines.append(f"Baseline best IoU >= 0.50    = {ge050}")
    lines.append("")
    lines.append(f"Direct Baseline rescue       = {direct}")
    lines.append(f"Cache-space unique rescue    = {unique}")
    lines.append(f"Deep blindspot rescue (<.30) = {deep_blind}")
    lines.append(f"Hard blindspot rescue        = {hard_blind}")
    lines.append("```")
    lines.append("")

    lines.append("## Rescue by class")
    lines.append("")
    lines.append("```text")

    if rescue_by_class:
        for k, v in rescue_by_class.most_common():
            lines.append(f"{k:24s} {v}")
    else:
        lines.append("(none)")

    lines.append("```")
    lines.append("")

    lines.append("## Rescue by previous failure type")
    lines.append("")
    lines.append("```text")

    if rescue_by_failure:
        for k, v in rescue_by_failure.most_common():
            lines.append(f"{k:40s} {v}")
    else:
        lines.append("(none)")

    lines.append("```")
    lines.append("")

    lines.append("## Deep blindspot rescue by class")
    lines.append("")
    lines.append("```text")

    if blind_by_class:
        for k, v in blind_by_class.most_common():
            lines.append(f"{k:24s} {v}")
    else:
        lines.append("(none)")

    lines.append("```")
    lines.append("")

    lines.append("## Directly rescued FN")
    lines.append("")

    if rescued_rows:
        lines.append(
            "| image | class | failure | "
            "Baseline IoU | score | RareOS O IoU | "
            "RareOS H IoU | RareOS max |"
        )
        lines.append(
            "|---|---|---|---:|---:|---:|---:|---:|"
        )

        for x in sorted(
            rescued_rows,
            key=lambda z: -z["baseline_best_iou"],
        ):
            lines.append(
                f"| {x['image']} "
                f"| {x['gt_class_name']} "
                f"| {x['failure_type']} "
                f"| {x['baseline_best_iou']:.4f} "
                f"| {x['baseline_best_score']:.6g} "
                f"| {x['rareos_original_best_iou']:.4f} "
                f"| {x['rareos_hflip_best_iou']:.4f} "
                f"| {x['rareos_max_best_iou']:.4f} |"
            )
    else:
        lines.append("No direct rescue.")

    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"**{decision}**")
    lines.append("")
    lines.append(reason)
    lines.append("")

    summary_path.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    # ========================================================
    # Console output
    # ========================================================

    print()
    print("=" * 68)
    print("BASELINE HIGH-RECALL COMPLEMENTARITY")
    print("=" * 68)

    print(f"Final Combo remaining FN     : {n}")
    print()

    print(f"Baseline IoU >= 0.30         : {ge030}")
    print(f"Baseline IoU >= 0.40         : {ge040}")
    print(f"Baseline IoU >= 0.50         : {ge050}")
    print()

    print(f"Direct Baseline rescue       : {direct}")
    print(f"Cache-space unique rescue    : {unique}")
    print(f"Deep blindspot rescue        : {deep_blind}")
    print(f"Hard blindspot rescue        : {hard_blind}")

    print()
    print("===== DIRECT RESCUES =====")

    if not rescued_rows:
        print("(none)")
    else:
        for x in sorted(
            rescued_rows,
            key=lambda z: -z["baseline_best_iou"],
        ):
            print(
                f"{x['image']} | "
                f"{x['gt_class_name']} | "
                f"{x['failure_type']} | "
                f"B={x['baseline_best_iou']:.4f} "
                f"(score={x['baseline_best_score']:.6g}) | "
                f"O={x['rareos_original_best_iou']:.4f} | "
                f"H={x['rareos_hflip_best_iou']:.4f}"
            )

    print()
    print("=" * 68)
    print("DECISION:", decision)
    print(reason)
    print("=" * 68)

    print()
    print("Saved:")
    print(csv_path)
    print(rescue_csv)
    print(summary_path)


if __name__ == "__main__":
    main()
