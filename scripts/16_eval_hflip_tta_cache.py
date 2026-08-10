from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from collections import Counter
from pathlib import Path
import sys

import numpy as np


def load_diag_module(path: Path):
    spec = importlib.util.spec_from_file_location("fn_diag14", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load diagnostic module: {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Evaluate Original + Horizontal-Flip cached proposals on Val, "
            "using exactly the NMS and GT matching logic from 14_fn_diagnostic.py."
        )
    )
    p.add_argument(
        "--diag-script",
        type=Path,
        default=Path("scripts/14_fn_diagnostic.py"),
    )
    p.add_argument(
        "--original-cache",
        type=Path,
        default=Path("results/fn_analysis/cache"),
    )
    p.add_argument(
        "--hflip-cache",
        type=Path,
        default=Path("results/fn_analysis/cache_hflip"),
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=Path("datasets/yolo_split/labels/val"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fn_analysis/tta_hflip"),
    )
    p.add_argument("--global-iou", type=float, default=0.90)
    p.add_argument("--match-iou", type=float, default=0.50)
    return p.parse_args()


def load_manifest(cache_dir: Path):
    path = cache_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def cache_map(manifest):
    return {
        item["image_name"]: item
        for item in manifest["items"]
    }


def eval_one(pre, gt, diag, global_iou, match_iou):
    keep = diag.class_aware_nms_indices(pre, global_iou)
    post = (
        pre[keep]
        if len(keep)
        else np.empty((0, len(diag.COLS)), dtype=np.float32)
    )
    tp, fp, fn, unmatched, matched, _ = diag.match_predictions(
        post,
        gt,
        match_iou,
    )
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "unmatched": set(unmatched),
        "matched": set(matched),
        "post": post,
    }


def main():
    args = parse_args()
    diag = load_diag_module(args.diag_script)

    om = load_manifest(args.original_cache)
    hm = load_manifest(args.hflip_cache)

    if int(om["images_count"]) != int(hm["images_count"]):
        raise RuntimeError(
            f"Image count mismatch: original={om['images_count']} "
            f"hflip={hm['images_count']}"
        )

    hmap = cache_map(hm)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    totals = {
        "original": Counter(),
        "union": Counter(),
    }
    rescue_rows = []
    regression_rows = []
    union_fn_rows = []
    rescued_by_class = Counter()
    regressed_by_class = Counter()
    union_fn_by_class = Counter()

    for idx, oitem in enumerate(om["items"], start=1):
        image_name = oitem["image_name"]
        if image_name not in hmap:
            raise RuntimeError(f"HFlip cache missing image: {image_name}")
        hitem = hmap[image_name]

        oc = np.load(args.original_cache / oitem["cache_file"])
        hc = np.load(args.hflip_cache / hitem["cache_file"])

        original_pre = oc["candidates"].astype(np.float32, copy=False)
        hflip_pre = hc["candidates"].astype(np.float32, copy=False)

        oh, ow = map(int, oc["image_shape"])
        hh, hw = map(int, hc["image_shape"])
        if (oh, ow) != (hh, hw):
            raise RuntimeError(
                f"Shape mismatch for {image_name}: "
                f"original={(oh, ow)} hflip={(hh, hw)}"
            )

        label_path = args.labels / f"{Path(image_name).stem}.txt"
        gt = diag.read_yolo_gt(label_path, ow, oh)

        original = eval_one(
            original_pre,
            gt,
            diag,
            args.global_iou,
            args.match_iou,
        )

        union_pre = np.concatenate(
            [original_pre, hflip_pre],
            axis=0,
        )
        union = eval_one(
            union_pre,
            gt,
            diag,
            args.global_iou,
            args.match_iou,
        )

        for key in ("tp", "fp", "fn"):
            totals["original"][key] += original[key]
            totals["union"][key] += union[key]

        rescued = original["unmatched"] & union["matched"]
        regressed = original["matched"] & union["unmatched"]

        for gi in sorted(rescued):
            g = gt[gi]
            rescued_by_class[g.class_id] += 1
            rescue_rows.append({
                "image_name": image_name,
                "gt_index": gi,
                "class_id": g.class_id,
                "class_name": diag.CLASS_NAMES[g.class_id],
                "gt_xmin": g.xmin,
                "gt_ymin": g.ymin,
                "gt_xmax": g.xmax,
                "gt_ymax": g.ymax,
            })

        for gi in sorted(regressed):
            g = gt[gi]
            regressed_by_class[g.class_id] += 1
            regression_rows.append({
                "image_name": image_name,
                "gt_index": gi,
                "class_id": g.class_id,
                "class_name": diag.CLASS_NAMES[g.class_id],
                "gt_xmin": g.xmin,
                "gt_ymin": g.ymin,
                "gt_xmax": g.xmax,
                "gt_ymax": g.ymax,
            })

        for gi in sorted(union["unmatched"]):
            g = gt[gi]
            union_fn_by_class[g.class_id] += 1
            union_fn_rows.append({
                "image_name": image_name,
                "gt_index": gi,
                "class_id": g.class_id,
                "class_name": diag.CLASS_NAMES[g.class_id],
                "gt_xmin": g.xmin,
                "gt_ymin": g.ymin,
                "gt_xmax": g.xmax,
                "gt_ymax": g.ymax,
            })

        if idx % 25 == 0 or idx == len(om["items"]):
            utp = totals["union"]["tp"]
            ufn = totals["union"]["fn"]
            ur = utp / (utp + ufn) if utp + ufn else 0.0
            print(
                f"{idx}/{len(om['items'])} "
                f"Union TP={utp} FN={ufn} Recall={ur:.4f} "
                f"rescued={len(rescue_rows)} regressed={len(regression_rows)}"
            )

    def write_rows(name, rows):
        path = args.output_dir / name
        if rows:
            with path.open("w", encoding="utf-8-sig", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
        else:
            path.write_text("", encoding="utf-8")
        return path

    write_rows("rescued_original_fn.csv", rescue_rows)
    write_rows("regressed_original_tp.csv", regression_rows)
    write_rows("union_remaining_fn.csv", union_fn_rows)

    ot = totals["original"]
    ut = totals["union"]

    orc = ot["tp"] / (ot["tp"] + ot["fn"]) if ot["tp"] + ot["fn"] else 0.0
    urc = ut["tp"] / (ut["tp"] + ut["fn"]) if ut["tp"] + ut["fn"] else 0.0
    up = ut["tp"] / (ut["tp"] + ut["fp"]) if ut["tp"] + ut["fp"] else 0.0

    class_rows = []
    for cid, cname in enumerate(diag.CLASS_NAMES):
        class_rows.append({
            "class": cname,
            "rescued_original_fn": rescued_by_class[cid],
            "regressed_original_tp": regressed_by_class[cid],
            "union_remaining_fn": union_fn_by_class[cid],
        })
    with (args.output_dir / "class_delta.csv").open(
        "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(class_rows[0].keys()))
        writer.writeheader()
        writer.writerows(class_rows)

    summary = [
        "# Original + HFlip TTA Evaluation",
        "",
        "## Original cache baseline",
        f"- TP: {ot['tp']}",
        f"- FP: {ot['fp']}",
        f"- FN: {ot['fn']}",
        f"- Recall: {orc:.6f}",
        f"- ScoreLike: {orc * 100:.2f}",
        "",
        "## Original + HFlip union",
        f"- TP: {ut['tp']}",
        f"- FP: {ut['fp']}",
        f"- FN: {ut['fn']}",
        f"- Recall: {urc:.6f}",
        f"- Precision: {up:.6f}",
        f"- ScoreLike: {urc * 100:.2f}",
        "",
        "## Delta",
        f"- TP delta: {ut['tp'] - ot['tp']:+d}",
        f"- FN delta: {ut['fn'] - ot['fn']:+d}",
        f"- Recall delta: {urc - orc:+.6f}",
        f"- ScoreLike delta: {(urc - orc) * 100:+.2f}",
        f"- Original FN rescued: {len(rescue_rows)}",
        f"- Original TP regressed: {len(regression_rows)}",
        "",
        "## Rescued FN by class",
    ]
    for cid, cname in enumerate(diag.CLASS_NAMES):
        if rescued_by_class[cid]:
            summary.append(f"- {cname}: {rescued_by_class[cid]}")
    if not rescue_rows:
        summary.append("- none")

    summary += ["", "## Remaining FN by class"]
    for cid, cname in enumerate(diag.CLASS_NAMES):
        if union_fn_by_class[cid]:
            summary.append(f"- {cname}: {union_fn_by_class[cid]}")

    (args.output_dir / "summary.md").write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print()
    print("===== Original =====")
    print("TP       :", ot["tp"])
    print("FP       :", ot["fp"])
    print("FN       :", ot["fn"])
    print("Recall   :", f"{orc:.6f}")
    print("ScoreLike:", f"{orc * 100:.2f}")

    print()
    print("===== Original + HFlip =====")
    print("TP       :", ut["tp"])
    print("FP       :", ut["fp"])
    print("FN       :", ut["fn"])
    print("Recall   :", f"{urc:.6f}")
    print("Precision:", f"{up:.6f}")
    print("ScoreLike:", f"{urc * 100:.2f}")

    print()
    print("Rescued original FN :", len(rescue_rows))
    print("Regressed original TP:", len(regression_rows))
    print("Saved:", args.output_dir)


if __name__ == "__main__":
    main()
