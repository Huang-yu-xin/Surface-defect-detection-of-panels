from __future__ import annotations

import argparse
import csv
import importlib.util
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ZONGLIE_ID = 1


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "GPU-accelerated zonglie cross-tile stitch sweep. "
            "Uses the cached pre-global-NMS proposals; no YOLO forward pass."
        )
    )
    p.add_argument(
        "--diag-script",
        type=Path,
        default=Path("scripts/14_fn_diagnostic.py"),
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("results/fn_analysis/cache"),
    )
    p.add_argument(
        "--labels",
        type=Path,
        default=Path("datasets/yolo_split/labels/val"),
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/fn_analysis/zonglie_stitch_gpu"),
    )
    p.add_argument("--global-iou", type=float, default=0.90)
    p.add_argument("--match-iou", type=float, default=0.50)
    p.add_argument("--stride", type=float, default=768.0)
    p.add_argument("--min-height", type=float, default=180.0)
    p.add_argument(
        "--x-tols",
        type=float,
        nargs="+",
        default=[32.0, 64.0, 96.0],
    )
    p.add_argument(
        "--max-y-gaps",
        type=float,
        nargs="+",
        default=[64.0, 256.0],
    )
    p.add_argument(
        "--min-aspects",
        type=float,
        nargs="+",
        default=[3.0, 5.0],
    )
    p.add_argument("--min-x-overlap", type=float, default=0.20)
    p.add_argument("--min-rows", type=int, default=2)
    p.add_argument("--min-merged-height", type=float, default=1300.0)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def load_diag(path: Path):
    spec = importlib.util.spec_from_file_location("fn_diag14_stitch_gpu", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_manifest(cache_dir: Path):
    path = cache_dir / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def box_iou(box, boxes):
    if len(boxes) == 0:
        return np.empty((0,), dtype=np.float32)

    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = (
        np.maximum(0.0, x2 - x1)
        * np.maximum(0.0, y2 - y1)
    )

    aa = max(
        0.0,
        (box[2] - box[0]) * (box[3] - box[1]),
    )

    ab = np.maximum(
        0.0,
        (boxes[:, 2] - boxes[:, 0])
        * (boxes[:, 3] - boxes[:, 1]),
    )

    union = aa + ab - inter

    return np.divide(
        inter,
        union,
        out=np.zeros_like(inter, dtype=np.float32),
        where=union > 0,
    )


def match_one_class(pred, gt_boxes, iou_thr):
    if len(gt_boxes) == 0 or len(pred) == 0:
        return set()

    order = np.argsort(pred[:, 1])[::-1]
    matched = set()

    for pi in order:
        ious = box_iou(pred[pi, 2:6], gt_boxes)

        for gi in np.argsort(ious)[::-1]:
            gi = int(gi)

            if ious[gi] < iou_thr:
                break

            if gi in matched:
                continue

            matched.add(gi)
            break

    return matched


class DSU:
    def __init__(self, n):
        self.p = list(range(n))
        self.sz = [1] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        a = self.find(a)
        b = self.find(b)

        if a == b:
            return

        if self.sz[a] < self.sz[b]:
            a, b = b, a

        self.p[b] = a
        self.sz[a] += self.sz[b]


def build_pair_features_gpu(
    cand_np: np.ndarray,
    *,
    stride: float,
    min_x_overlap: float,
    max_x_tol: float,
    max_y_gap: float,
    device: torch.device,
):
    """
    Build only potentially useful cross-row pairs.

    Heavy pairwise geometry runs on CUDA. Returned arrays are compact CPU
    arrays reused by every parameter configuration.
    """
    if len(cand_np) < 2:
        return None

    cand = torch.as_tensor(
        cand_np,
        dtype=torch.float32,
        device=device,
    )

    tile_y_np = cand_np[:, 7]
    unique_rows = np.unique(tile_y_np)

    edge_i = []
    edge_j = []
    edge_xdiff = []
    edge_ygap = []

    for ra_idx in range(len(unique_rows)):
        for rb_idx in range(ra_idx + 1, len(unique_rows)):
            ra = float(unique_rows[ra_idx])
            rb = float(unique_rows[rb_idx])
            row_delta = abs(rb - ra)

            if row_delta <= 1e-6 or row_delta > stride + 1e-3:
                continue

            ia_np = np.flatnonzero(np.isclose(tile_y_np, ra))
            ib_np = np.flatnonzero(np.isclose(tile_y_np, rb))

            if len(ia_np) == 0 or len(ib_np) == 0:
                continue

            ia = torch.as_tensor(ia_np, dtype=torch.long, device=device)
            ib = torch.as_tensor(ib_np, dtype=torch.long, device=device)

            a = cand[ia]
            b = cand[ib]

            acx = (a[:, 2] + a[:, 4]) * 0.5
            bcx = (b[:, 2] + b[:, 4]) * 0.5

            xdiff = torch.abs(
                acx[:, None] - bcx[None, :]
            )

            gap_ab = b[None, :, 3] - a[:, None, 5]
            gap_ba = a[:, None, 3] - b[None, :, 5]

            ygap = torch.clamp(
                torch.maximum(gap_ab, gap_ba),
                min=0.0,
            )

            inter_x = torch.clamp(
                torch.minimum(
                    a[:, None, 4],
                    b[None, :, 4],
                )
                - torch.maximum(
                    a[:, None, 2],
                    b[None, :, 2],
                ),
                min=0.0,
            )

            wa = torch.clamp(
                a[:, 4] - a[:, 2],
                min=1e-6,
            )
            wb = torch.clamp(
                b[:, 4] - b[:, 2],
                min=1e-6,
            )

            overlap_ratio = (
                inter_x
                / torch.minimum(
                    wa[:, None],
                    wb[None, :],
                )
            )

            mask = (
                (xdiff <= max_x_tol)
                & (ygap <= max_y_gap)
                & (overlap_ratio >= min_x_overlap)
            )

            rr, cc = torch.where(mask)

            if rr.numel() == 0:
                continue

            edge_i.append(
                ia[rr].detach().cpu().numpy()
            )
            edge_j.append(
                ib[cc].detach().cpu().numpy()
            )
            edge_xdiff.append(
                xdiff[rr, cc].detach().cpu().numpy()
            )
            edge_ygap.append(
                ygap[rr, cc].detach().cpu().numpy()
            )

    if not edge_i:
        return None

    return {
        "i": np.concatenate(edge_i).astype(np.int32, copy=False),
        "j": np.concatenate(edge_j).astype(np.int32, copy=False),
        "xdiff": np.concatenate(edge_xdiff).astype(np.float32, copy=False),
        "ygap": np.concatenate(edge_ygap).astype(np.float32, copy=False),
    }


def make_merged_boxes(
    cand: np.ndarray,
    aspects: np.ndarray,
    heights: np.ndarray,
    pairs,
    *,
    min_aspect: float,
    x_tol: float,
    max_y_gap: float,
    min_height: float,
    min_rows: int,
    min_merged_height: float,
):
    if len(cand) < 2 or pairs is None:
        return np.empty((0, cand.shape[1]), dtype=np.float32)

    valid_node = (
        (heights >= min_height)
        & (aspects >= min_aspect)
    )

    edge_mask = (
        (pairs["xdiff"] <= x_tol)
        & (pairs["ygap"] <= max_y_gap)
        & valid_node[pairs["i"]]
        & valid_node[pairs["j"]]
    )

    ei = pairs["i"][edge_mask]
    ej = pairs["j"][edge_mask]

    if len(ei) == 0:
        return np.empty((0, cand.shape[1]), dtype=np.float32)

    dsu = DSU(len(cand))

    for a, b in zip(ei.tolist(), ej.tolist()):
        dsu.union(int(a), int(b))

    groups = defaultdict(list)

    touched = np.unique(
        np.concatenate([ei, ej])
    )

    for i in touched.tolist():
        groups[dsu.find(int(i))].append(int(i))

    merged = []

    for ids in groups.values():
        if len(ids) < 2:
            continue

        boxes = cand[ids]
        rows = np.unique(boxes[:, 7])

        if len(rows) < min_rows:
            continue

        y1 = float(np.min(boxes[:, 3]))
        y2 = float(np.max(boxes[:, 5]))

        if y2 - y1 < min_merged_height:
            continue

        # zonglie is extremely narrow. Median x geometry avoids a very wide
        # envelope that would destroy IoU.
        x1 = float(np.median(boxes[:, 2]))
        x2 = float(np.median(boxes[:, 4]))

        if x2 <= x1:
            continue

        row = np.zeros(
            (cand.shape[1],),
            dtype=np.float32,
        )

        row[0] = ZONGLIE_ID
        row[1] = float(np.max(boxes[:, 1]))
        row[2] = x1
        row[3] = y1
        row[4] = x2
        row[5] = y2

        merged.append(row)

    if not merged:
        return np.empty((0, cand.shape[1]), dtype=np.float32)

    return np.stack(merged).astype(np.float32, copy=False)


def main():
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. This accelerated script is intended "
            "for the currently attached GPU instance."
        )

    device = torch.device(args.device)
    diag = load_diag(args.diag_script)
    manifest = load_manifest(args.cache_dir)

    configs = [
        {
            "min_aspect": float(a),
            "x_tol": float(x),
            "max_y_gap": float(g),
        }
        for a, x, g in itertools.product(
            args.min_aspects,
            args.x_tols,
            args.max_y_gaps,
        )
    ]

    stats = [
        {
            **cfg,
            "tp": 0,
            "fn": 0,
            "baseline_tp": 0,
            "baseline_fn": 0,
            "rescued": 0,
            "regressed": 0,
            "merged_boxes": 0,
        }
        for cfg in configs
    ]

    baseline_total_tp = 0
    baseline_total_fn = 0
    total_gt = 0

    max_x_tol = max(args.x_tols)
    max_y_gap = max(args.max_y_gaps)
    min_aspect_any = min(args.min_aspects)

    print("Device       :", device)
    print("Configs      :", len(configs))
    print("Images       :", len(manifest["items"]))
    print("Pair geometry: CUDA")
    print("NMS/matching : CPU exact-reference logic")
    print()

    for image_idx, item in enumerate(
        manifest["items"],
        start=1,
    ):
        npz = np.load(
            args.cache_dir / item["cache_file"]
        )

        pre = npz["candidates"].astype(
            np.float32,
            copy=False,
        )

        height, width = map(
            int,
            npz["image_shape"],
        )

        label_path = (
            args.labels
            / f"{Path(item['image_name']).stem}.txt"
        )

        gt = diag.read_yolo_gt(
            label_path,
            width,
            height,
        )

        zgt = [
            g for g in gt
            if g.class_id == ZONGLIE_ID
        ]

        gt_boxes = np.asarray(
            [
                [
                    g.xmin,
                    g.ymin,
                    g.xmax,
                    g.ymax,
                ]
                for g in zgt
            ],
            dtype=np.float32,
        )

        if gt_boxes.size == 0:
            gt_boxes = np.empty(
                (0, 4),
                dtype=np.float32,
            )

        total_gt += len(gt_boxes)

        zpre = pre[
            pre[:, 0].astype(np.int32)
            == ZONGLIE_ID
        ]

        # Exact same reference NMS as script 14, but only for zonglie and
        # only once per image.
        keep = diag.class_aware_nms_indices(
            zpre,
            args.global_iou,
        )

        zpost = (
            zpre[keep]
            if len(keep)
            else np.empty(
                (0, pre.shape[1]),
                dtype=np.float32,
            )
        )

        base_matched = match_one_class(
            zpost,
            gt_boxes,
            args.match_iou,
        )

        btp = len(base_matched)
        bfn = len(gt_boxes) - btp

        baseline_total_tp += btp
        baseline_total_fn += bfn

        widths = np.maximum(
            1e-6,
            zpost[:, 4] - zpost[:, 2],
        )

        heights = np.maximum(
            1e-6,
            zpost[:, 5] - zpost[:, 3],
        )

        aspects = heights / widths

        # Drop nodes that cannot participate in any default configuration
        # before sending pair geometry to CUDA.
        maybe = (
            (heights >= args.min_height)
            & (aspects >= min_aspect_any)
        )

        cand = zpost[maybe]
        cand_heights = heights[maybe]
        cand_aspects = aspects[maybe]

        pairs = build_pair_features_gpu(
            cand,
            stride=args.stride,
            min_x_overlap=args.min_x_overlap,
            max_x_tol=max_x_tol,
            max_y_gap=max_y_gap,
            device=device,
        )

        for s in stats:
            merged = make_merged_boxes(
                cand,
                cand_aspects,
                cand_heights,
                pairs,
                min_aspect=s["min_aspect"],
                x_tol=s["x_tol"],
                max_y_gap=s["max_y_gap"],
                min_height=args.min_height,
                min_rows=args.min_rows,
                min_merged_height=args.min_merged_height,
            )

            aug = (
                np.concatenate(
                    [zpost, merged],
                    axis=0,
                )
                if len(merged)
                else zpost
            )

            matched = match_one_class(
                aug,
                gt_boxes,
                args.match_iou,
            )

            s["baseline_tp"] += btp
            s["baseline_fn"] += bfn
            s["tp"] += len(matched)
            s["fn"] += len(gt_boxes) - len(matched)
            s["rescued"] += len(
                matched - base_matched
            )
            s["regressed"] += len(
                base_matched - matched
            )
            s["merged_boxes"] += len(merged)

        if (
            image_idx % 25 == 0
            or image_idx == len(manifest["items"])
        ):
            print(
                f"{image_idx}/{len(manifest['items'])} "
                f"baseline zonglie "
                f"TP={baseline_total_tp} "
                f"FN={baseline_total_fn}"
            )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for s in stats:
        denom = s["tp"] + s["fn"]
        s["recall"] = (
            s["tp"] / denom
            if denom
            else 0.0
        )
        s["delta_tp"] = (
            s["tp"] - s["baseline_tp"]
        )
        s["delta_fn"] = (
            s["fn"] - s["baseline_fn"]
        )

    stats.sort(
        key=lambda x: (
            -x["tp"],
            x["regressed"],
            x["merged_boxes"],
        )
    )

    fields = [
        "min_aspect",
        "x_tol",
        "max_y_gap",
        "tp",
        "fn",
        "recall",
        "delta_tp",
        "delta_fn",
        "rescued",
        "regressed",
        "merged_boxes",
        "baseline_tp",
        "baseline_fn",
    ]

    with (
        args.output_dir / "scan.csv"
    ).open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()

        for s in stats:
            writer.writerow(
                {k: s[k] for k in fields}
            )

    best = stats[0]

    summary = [
        "# Zonglie Cross-Tile Stitch GPU Sweep",
        "",
        f"- Device: {device}",
        f"- Config count: {len(configs)}",
        f"- Baseline zonglie TP: {baseline_total_tp}",
        f"- Baseline zonglie FN: {baseline_total_fn}",
        f"- Total zonglie GT: {total_gt}",
        "",
        "## Best configuration",
        f"- min_aspect: {best['min_aspect']}",
        f"- x_tol: {best['x_tol']}",
        f"- max_y_gap: {best['max_y_gap']}",
        f"- merged_boxes: {best['merged_boxes']}",
        f"- zonglie TP: {best['tp']}",
        f"- zonglie FN: {best['fn']}",
        f"- zonglie Recall: {best['recall']:.6f}",
        f"- rescued: {best['rescued']}",
        f"- regressed: {best['regressed']}",
        "",
        "## All configurations",
    ]

    for s in stats:
        summary.append(
            "- "
            f"aspect>={s['min_aspect']}, "
            f"x_tol={s['x_tol']}, "
            f"gap={s['max_y_gap']}: "
            f"TP={s['tp']} FN={s['fn']} "
            f"rescued={s['rescued']} "
            f"regressed={s['regressed']} "
            f"merged={s['merged_boxes']}"
        )

    (
        args.output_dir / "summary.md"
    ).write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print()
    print("===== Baseline zonglie =====")
    print("TP:", baseline_total_tp)
    print("FN:", baseline_total_fn)

    print()
    print("===== Best stitch =====")
    print("min_aspect :", best["min_aspect"])
    print("x_tol      :", best["x_tol"])
    print("max_y_gap  :", best["max_y_gap"])
    print("TP         :", best["tp"])
    print("FN         :", best["fn"])
    print("rescued    :", best["rescued"])
    print("regressed  :", best["regressed"])
    print("merged     :", best["merged_boxes"])
    print("Saved      :", args.output_dir)


if __name__ == "__main__":
    main()
