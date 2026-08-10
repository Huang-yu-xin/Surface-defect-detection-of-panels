from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

CLASS_NAMES = [
    "jieba", "zonglie", "qilie", "jiaza", "yiwuyaru",
    "huashang", "mamianmakeng", "yanghuatiepi", "gunyin",
]
ZONGLIE_ID = 1
DEFAULT_HFLIP_CLASSES = [0, 2, 3, 4, 7]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Build/evaluate the final cache-based combo: Original + selective "
            "HFlip + zonglie cross-tile stitching."
        )
    )
    p.add_argument("--mode", choices=["val", "test"], required=True)
    p.add_argument("--diag-script", type=Path, default=Path("scripts/14_fn_diagnostic.py"))
    p.add_argument("--original-cache", type=Path, required=True)
    p.add_argument("--hflip-cache", type=Path, required=True)
    p.add_argument("--labels", type=Path, default=Path("datasets/yolo_split/labels/val"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--submission", type=Path, default=None)
    p.add_argument("--summary-csv", type=Path, default=None)
    p.add_argument("--global-iou", type=float, default=0.90)
    p.add_argument("--match-iou", type=float, default=0.50)
    p.add_argument("--hflip-classes", type=int, nargs="+", default=DEFAULT_HFLIP_CLASSES)
    p.add_argument("--min-aspect", type=float, default=5.0)
    p.add_argument("--x-tol", type=float, default=64.0)
    p.add_argument("--max-y-gap", type=float, default=64.0)
    p.add_argument("--min-height", type=float, default=180.0)
    p.add_argument("--min-x-overlap", type=float, default=0.20)
    p.add_argument("--stride", type=float, default=768.0)
    p.add_argument("--min-rows", type=int, default=2)
    p.add_argument("--min-merged-height", type=float, default=1300.0)
    p.add_argument("--device", type=str, default="cuda:0")
    return p.parse_args()


def load_diag(path: Path):
    spec = importlib.util.spec_from_file_location("fn_diag14_final_combo", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def load_manifest(cache_dir: Path):
    p = cache_dir / "manifest.json"
    if not p.exists():
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def manifest_map(m):
    return {item["image_name"]: item for item in m["items"]}


def gpu_class_aware_nms(arr: np.ndarray, iou: float, device: torch.device):
    """Fast class-aware NMS using torchvision CUDA, with CPU fallback."""
    if len(arr) == 0:
        return arr
    try:
        from torchvision.ops import nms
    except Exception:
        return None

    kept = []
    cls_ids = arr[:, 0].astype(np.int32)
    for cid in np.unique(cls_ids):
        idx = np.flatnonzero(cls_ids == cid)
        boxes = torch.as_tensor(arr[idx, 2:6], dtype=torch.float32, device=device)
        scores = torch.as_tensor(arr[idx, 1], dtype=torch.float32, device=device)
        keep_local = nms(boxes, scores, iou).detach().cpu().numpy()
        kept.append(idx[keep_local])

    if not kept:
        return np.empty((0, arr.shape[1]), dtype=np.float32)
    inds = np.concatenate(kept)
    # Match repository behavior: final detections sorted by descending score.
    inds = inds[np.argsort(arr[inds, 1])[::-1]]
    return arr[inds]


def exact_class_aware_nms(arr: np.ndarray, iou: float, diag):
    if len(arr) == 0:
        return arr
    keep = diag.class_aware_nms_indices(arr, iou)
    return arr[keep] if len(keep) else np.empty((0, arr.shape[1]), dtype=np.float32)


def nms(arr, iou, diag, device):
    out = gpu_class_aware_nms(arr, iou, device)
    if out is not None:
        return out, "torchvision-cuda"
    return exact_class_aware_nms(arr, iou, diag), "reference-cpu"


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
        a = self.find(a); b = self.find(b)
        if a == b:
            return
        if self.sz[a] < self.sz[b]:
            a, b = b, a
        self.p[b] = a
        self.sz[a] += self.sz[b]


def build_pairs_gpu(cand: np.ndarray, args, device):
    if len(cand) < 2:
        return None

    t = torch.as_tensor(cand, dtype=torch.float32, device=device)
    tile_y = cand[:, 7]
    rows = np.unique(tile_y)
    edges = []

    for ai in range(len(rows)):
        for bi in range(ai + 1, len(rows)):
            ra, rb = float(rows[ai]), float(rows[bi])
            if abs(rb - ra) <= 1e-6 or abs(rb - ra) > args.stride + 1e-3:
                continue

            ia_np = np.flatnonzero(np.isclose(tile_y, ra))
            ib_np = np.flatnonzero(np.isclose(tile_y, rb))
            if not len(ia_np) or not len(ib_np):
                continue

            ia = torch.as_tensor(ia_np, dtype=torch.long, device=device)
            ib = torch.as_tensor(ib_np, dtype=torch.long, device=device)
            a = t[ia]; b = t[ib]

            acx = (a[:, 2] + a[:, 4]) * 0.5
            bcx = (b[:, 2] + b[:, 4]) * 0.5
            xdiff = torch.abs(acx[:, None] - bcx[None, :])

            gap_ab = b[None, :, 3] - a[:, None, 5]
            gap_ba = a[:, None, 3] - b[None, :, 5]
            ygap = torch.clamp(torch.maximum(gap_ab, gap_ba), min=0.0)

            inter_x = torch.clamp(
                torch.minimum(a[:, None, 4], b[None, :, 4])
                - torch.maximum(a[:, None, 2], b[None, :, 2]),
                min=0.0,
            )
            wa = torch.clamp(a[:, 4] - a[:, 2], min=1e-6)
            wb = torch.clamp(b[:, 4] - b[:, 2], min=1e-6)
            overlap = inter_x / torch.minimum(wa[:, None], wb[None, :])

            mask = (
                (xdiff <= args.x_tol)
                & (ygap <= args.max_y_gap)
                & (overlap >= args.min_x_overlap)
            )
            rr, cc = torch.where(mask)
            if rr.numel():
                edges.append((
                    ia[rr].detach().cpu().numpy(),
                    ib[cc].detach().cpu().numpy(),
                ))

    if not edges:
        return None
    return (
        np.concatenate([x[0] for x in edges]).astype(np.int32),
        np.concatenate([x[1] for x in edges]).astype(np.int32),
    )


def stitch_zonglie(zpost: np.ndarray, args, device):
    if len(zpost) < 2:
        return np.empty((0, zpost.shape[1]), dtype=np.float32)

    widths = np.maximum(1e-6, zpost[:, 4] - zpost[:, 2])
    heights = np.maximum(1e-6, zpost[:, 5] - zpost[:, 3])
    aspects = heights / widths
    mask = (heights >= args.min_height) & (aspects >= args.min_aspect)
    cand = zpost[mask]
    if len(cand) < 2:
        return np.empty((0, zpost.shape[1]), dtype=np.float32)

    pairs = build_pairs_gpu(cand, args, device)
    if pairs is None:
        return np.empty((0, zpost.shape[1]), dtype=np.float32)

    ei, ej = pairs
    dsu = DSU(len(cand))
    for a, b in zip(ei.tolist(), ej.tolist()):
        dsu.union(a, b)

    groups = defaultdict(list)
    touched = np.unique(np.concatenate([ei, ej]))
    for i in touched.tolist():
        groups[dsu.find(i)].append(i)

    merged = []
    for ids in groups.values():
        if len(ids) < 2:
            continue
        boxes = cand[ids]
        if len(np.unique(boxes[:, 7])) < args.min_rows:
            continue
        y1 = float(np.min(boxes[:, 3])); y2 = float(np.max(boxes[:, 5]))
        if y2 - y1 < args.min_merged_height:
            continue
        x1 = float(np.median(boxes[:, 2])); x2 = float(np.median(boxes[:, 4]))
        if x2 <= x1:
            continue
        row = np.zeros((zpost.shape[1],), dtype=np.float32)
        row[0] = ZONGLIE_ID
        row[1] = float(np.max(boxes[:, 1]))
        row[2:6] = [x1, y1, x2, y2]
        merged.append(row)

    if not merged:
        return np.empty((0, zpost.shape[1]), dtype=np.float32)
    return np.stack(merged).astype(np.float32, copy=False)


def add_stitched(post: np.ndarray, args, device):
    zpost = post[post[:, 0].astype(np.int32) == ZONGLIE_ID]
    merged = stitch_zonglie(zpost, args, device)
    if len(merged):
        return np.concatenate([post, merged], axis=0), len(merged)
    return post, 0


def submission_row(image_name, width, height, det):
    xmin = int(math.floor(float(det[2])))
    ymin = int(math.floor(float(det[3])))
    xmax = int(math.ceil(float(det[4])))
    ymax = int(math.ceil(float(det[5])))
    xmin = max(0, min(xmin, width - 1))
    ymin = max(0, min(ymin, height - 1))
    xmax = max(xmin + 1, min(xmax, width))
    ymax = max(ymin + 1, min(ymax, height))
    cid = int(det[0])
    return {
        "image_id": image_name,
        "category_name": CLASS_NAMES[cid],
        "bbox": [xmin, ymin, xmax, ymax],
        "score": round(float(det[1]), 6),
    }


def main():
    args = parse_args()
    if not torch.cuda.is_available() and args.device != "cpu":
        raise RuntimeError("CUDA unavailable")
    device = torch.device(args.device)
    diag = load_diag(args.diag_script)

    om = load_manifest(args.original_cache)
    hm = load_manifest(args.hflip_cache)
    hmap = manifest_map(hm)
    if int(om["images_count"]) != int(hm["images_count"]):
        raise RuntimeError("Original/HFlip image count mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    hclasses = set(args.hflip_classes)

    totals = Counter()
    class_stats = {name: Counter() for name in CLASS_NAMES}
    submission = []
    summary_rows = []
    total_merged = 0
    nms_backend_seen = set()

    print("Mode         :", args.mode)
    print("Device       :", device)
    print("Images       :", om["images_count"])
    print("HFlip classes:", [CLASS_NAMES[i] for i in sorted(hclasses)])
    print("Stitch       :", f"aspect>={args.min_aspect}, x_tol={args.x_tol}, gap={args.max_y_gap}")
    print()

    for idx, oitem in enumerate(om["items"], start=1):
        name = oitem["image_name"]
        if name not in hmap:
            raise RuntimeError(f"HFlip cache missing {name}")
        hitem = hmap[name]

        onpz = np.load(args.original_cache / oitem["cache_file"])
        hnpz = np.load(args.hflip_cache / hitem["cache_file"])
        orig = onpz["candidates"].astype(np.float32, copy=False)
        hflip = hnpz["candidates"].astype(np.float32, copy=False)
        height, width = map(int, onpz["image_shape"])

        hsel = hflip[np.isin(hflip[:, 0].astype(np.int32), list(hclasses))]
        union_pre = np.concatenate([orig, hsel], axis=0) if len(hsel) else orig
        post, backend = nms(union_pre, args.global_iou, diag, device)
        nms_backend_seen.add(backend)
        final, merged_count = add_stitched(post, args, device)
        total_merged += merged_count

        if args.mode == "val":
            label_path = args.labels / f"{Path(name).stem}.txt"
            gt = diag.read_yolo_gt(label_path, width, height)
            tp, fp, fn, unmatched, matched, _ = diag.match_predictions(final, gt, args.match_iou)
            totals["tp"] += tp; totals["fp"] += fp; totals["fn"] += fn

            # Per-class exact counts using the same matcher on each class.
            for cid, cname in enumerate(CLASS_NAMES):
                gtc = [g for g in gt if g.class_id == cid]
                predc = final[final[:, 0].astype(np.int32) == cid]
                ctp, cfp, cfn, *_ = diag.match_predictions(predc, gtc, args.match_iou)
                class_stats[cname]["tp"] += ctp
                class_stats[cname]["fp"] += cfp
                class_stats[cname]["fn"] += cfn
        else:
            for det in final:
                submission.append(submission_row(name, width, height, det))
            summary_rows.append({
                "image_id": name,
                "width": width,
                "height": height,
                "original_candidates": len(orig),
                "selected_hflip_candidates": len(hsel),
                "stitched_zonglie": merged_count,
                "final_detection_count": len(final),
            })

        if idx % 25 == 0 or idx == int(om["images_count"]):
            if args.mode == "val":
                denom = totals["tp"] + totals["fn"]
                rec = totals["tp"] / denom if denom else 0.0
                print(f"{idx}/{om['images_count']} TP={totals['tp']} FN={totals['fn']} Recall={rec:.4f} stitched={total_merged}")
            else:
                print(f"{idx}/{om['images_count']} final={len(submission):,} stitched={total_merged:,}")

    if args.mode == "val":
        denom = totals["tp"] + totals["fn"]
        recall = totals["tp"] / denom if denom else 0.0
        precision = totals["tp"] / (totals["tp"] + totals["fp"]) if totals["tp"] + totals["fp"] else 0.0

        rows = []
        for cname in CLASS_NAMES:
            s = class_stats[cname]
            d = s["tp"] + s["fn"]
            rows.append({
                "class": cname,
                "tp": s["tp"],
                "fp": s["fp"],
                "fn": s["fn"],
                "recall": s["tp"] / d if d else 0.0,
            })
        with (args.output_dir / "class_summary.csv").open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

        text = "\n".join([
            "# Final Combo Validation",
            "",
            f"- TP: {totals['tp']}",
            f"- FP: {totals['fp']}",
            f"- FN: {totals['fn']}",
            f"- Recall: {recall:.6f}",
            f"- Precision: {precision:.6f}",
            f"- ScoreLike: {recall * 100:.2f}",
            f"- Stitched zonglie boxes: {total_merged}",
            f"- NMS backend: {', '.join(sorted(nms_backend_seen))}",
            f"- HFlip classes: {', '.join(CLASS_NAMES[i] for i in sorted(hclasses))}",
        ]) + "\n"
        (args.output_dir / "summary.md").write_text(text, encoding="utf-8")

        print("\n===== FINAL COMBO VAL =====")
        print("TP       :", totals["tp"])
        print("FP       :", totals["fp"])
        print("FN       :", totals["fn"])
        print("Recall   :", f"{recall:.6f}")
        print("ScoreLike:", f"{recall * 100:.2f}")
        print("Stitched :", total_merged)
        print("NMS      :", ", ".join(sorted(nms_backend_seen)))
    else:
        if args.submission is None:
            args.submission = args.output_dir / "submission_final_combo.json"
        if args.summary_csv is None:
            args.summary_csv = args.output_dir / "submission_final_combo_summary.csv"
        args.submission.parent.mkdir(parents=True, exist_ok=True)
        args.summary_csv.parent.mkdir(parents=True, exist_ok=True)
        args.submission.write_text(json.dumps(submission, ensure_ascii=False, indent=2), encoding="utf-8")
        with args.summary_csv.open("w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys())); w.writeheader(); w.writerows(summary_rows)

        cc = Counter(row["category_name"] for row in submission)
        print("\n===== FINAL COMBO TEST =====")
        print("Images     :", om["images_count"])
        print("Detections :", f"{len(submission):,}")
        print("Stitched   :", f"{total_merged:,}")
        print("NMS        :", ", ".join(sorted(nms_backend_seen)))
        print("Submission :", args.submission)
        print("JSON MB    :", f"{args.submission.stat().st_size / 1024 / 1024:.2f}")
        print("By class:")
        for c in CLASS_NAMES:
            print(f"  {c:<18}{cc[c]}")


if __name__ == "__main__":
    main()
