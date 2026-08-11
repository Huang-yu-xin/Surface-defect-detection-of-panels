#!/usr/bin/env python3

import argparse
import csv
import hashlib
import importlib.util
import json
import sys
import zipfile
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


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
    name: i
    for i, name in enumerate(CLASS_NAMES)
}


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument(
        "--original-cache",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--hflip-cache",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--baseline-json",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--audit-csv",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--score-floor",
        type=float,
        default=0.10,
    )

    p.add_argument(
        "--added-score",
        type=float,
        default=1e-6,
    )

    p.add_argument(
        "--device",
        default="cpu",
    )

    return p.parse_args()


class JsonArrayWriter:
    def __init__(self, path):
        self.path = Path(path)
        self.f = None
        self.first = True
        self.count = 0

    def __enter__(self):
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.f = self.path.open(
            "w",
            encoding="utf-8",
        )

        self.f.write("[\n")
        return self

    def write(self, row):
        if not self.first:
            self.f.write(",\n")

        self.f.write(
            "  " + json.dumps(
                row,
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )

        self.first = False
        self.count += 1

    def __exit__(self, exc_type, exc, tb):
        if self.f is not None:
            if exc_type is None:
                self.f.write("\n]\n")

            self.f.close()


def sha256_file(path):
    h = hashlib.sha256()

    with Path(path).open("rb") as f:
        while True:
            block = f.read(1024 * 1024)

            if not block:
                break

            h.update(block)

    return h.hexdigest()


def make_zip(json_path, zip_path):
    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=6,
    ) as z:
        z.write(
            json_path,
            arcname="submission.json",
        )


def main():
    args = parse_args()

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    combo = load_module(
        Path("scripts/18_final_combo_from_cache.py"),
        "combo18_s34",
    )

    diag = load_module(
        Path("scripts/14_fn_diagnostic.py"),
        "diag14_s34",
    )

    device = torch.device(args.device)

    #
    # Exact Final Combo defaults.
    #
    old_argv = sys.argv[:]

    try:
        sys.argv = [
            "18_final_combo_from_cache.py",
            "--mode", "test",
            "--original-cache", "_dummy_original",
            "--hflip-cache", "_dummy_hflip",
            "--output-dir", "_dummy_output",
            "--device", args.device,
        ]

        defaults = combo.parse_args()

    finally:
        sys.argv = old_argv

    stitch_args = SimpleNamespace(
        **vars(defaults)
    )

    stitch_args.min_aspect = 5.0
    stitch_args.x_tol = 64.0
    stitch_args.max_y_gap = 64.0
    stitch_args.min_merged_height = 1300.0

    hclasses = {0, 2, 3, 4, 7}

    #
    # Baseline submission.
    #
    baseline = json.loads(
        args.baseline_json.read_text(
            encoding="utf-8",
        )
    )

    baseline_by_index = {
        i: row
        for i, row in enumerate(baseline)
    }

    #
    # Read complementarity audit.
    #
    audit_rows = list(
        csv.DictReader(
            args.audit_csv.open(
                encoding="utf-8-sig",
            )
        )
    )

    selected = []

    for r in audit_rows:
        score = float(
            r["baseline_score"]
        )

        same_iou = float(
            r["max_same_class_iou"]
        )

        if score < args.score_floor:
            continue

        if same_iou >= 0.50:
            continue

        idx = int(
            r["baseline_index"]
        )

        brow = baseline_by_index[idx]

        selected.append({
            "baseline_index": idx,
            "image_id": brow["image_id"],
            "category_name": brow["category_name"],
            "bbox": brow["bbox"],
            "original_score": float(brow["score"]),
            "bucket": r["bucket"],
            "same_iou": same_iou,
            "any_iou": float(
                r["max_any_class_iou"]
            ),
            "nearest_any_class":
                r["nearest_any_class"],
        })

    selected_by_image = {}

    for r in selected:
        selected_by_image.setdefault(
            r["image_id"],
            []
        ).append(r)

    print(
        "===== SELECTED BASELINE UNIQUE BOXES ====="
    )

    print(
        "Score floor:",
        args.score_floor,
    )

    print(
        "Selected:",
        len(selected),
    )

    buckets = Counter(
        r["bucket"]
        for r in selected
    )

    classes = Counter(
        r["category_name"]
        for r in selected
    )

    print()
    print("By bucket:")

    for k, v in buckets.items():
        print(
            f"  {k:<24} {v}"
        )

    print()
    print("By class:")

    for c in CLASS_NAMES:
        print(
            f"  {c:<18} {classes[c]}"
        )

    print()
    print("Top selected boxes:")

    for r in sorted(
        selected,
        key=lambda x: -x["original_score"],
    )[:30]:
        print(
            f"{r['original_score']:.6f} "
            f"{r['category_name']:<16} "
            f"{r['bucket']:<22} "
            f"sameIoU={r['same_iou']:.3f} "
            f"anyIoU={r['any_iou']:.3f} "
            f"{r['image_id']}"
        )

    #
    # Exact Final Combo.
    #
    om = combo.load_manifest(
        args.original_cache
    )

    hm = combo.load_manifest(
        args.hflip_cache
    )

    hmap = combo.manifest_map(hm)

    submission_path = (
        args.output_dir
        / "submission_baseline_unique_s010.json"
    )

    zip_path = (
        args.output_dir
        / "submission_baseline_unique_s010.zip"
    )

    base_count = 0
    stitched_count = 0
    added_count = 0
    zero_scores = 0

    added_bucket = Counter()
    added_class = Counter()

    with JsonArrayWriter(
        submission_path
    ) as writer:

        for idx, oitem in enumerate(
            om["items"],
            start=1,
        ):
            name = oitem["image_name"]

            hitem = hmap[name]

            onpz = np.load(
                args.original_cache
                / oitem["cache_file"]
            )

            hnpz = np.load(
                args.hflip_cache
                / hitem["cache_file"]
            )

            orig = onpz["candidates"].astype(
                np.float32,
                copy=False,
            )

            hflip = hnpz["candidates"].astype(
                np.float32,
                copy=False,
            )

            height, width = map(
                int,
                onpz["image_shape"],
            )

            hsel = hflip[
                np.isin(
                    hflip[:, 0].astype(np.int32),
                    list(hclasses),
                )
            ]

            union_pre = (
                np.concatenate(
                    [orig, hsel],
                    axis=0,
                )
                if len(hsel)
                else orig
            )

            post, _ = combo.nms(
                union_pre,
                defaults.global_iou,
                diag,
                device,
            )

            final, merged_count = combo.add_stitched(
                post,
                stitch_args,
                device,
            )

            base_count += len(final)
            stitched_count += merged_count

            #
            # Existing Final Combo predictions.
            #
            for det in final:
                row = combo.submission_row(
                    name,
                    width,
                    height,
                    det,
                )

                writer.write(row)

            #
            # Independent Baseline proposals.
            #
            for r in selected_by_image.get(
                name,
                [],
            ):
                bbox = list(
                    map(float, r["bbox"])
                )

                cid = CLASS_TO_ID[
                    r["category_name"]
                ]

                det = np.asarray(
                    [
                        cid,
                        args.added_score,
                        bbox[0],
                        bbox[1],
                        bbox[2],
                        bbox[3],
                    ],
                    dtype=np.float32,
                )

                row = combo.submission_row(
                    name,
                    width,
                    height,
                    det,
                )

                if row["score"] == 0:
                    zero_scores += 1

                writer.write(row)

                added_count += 1
                added_bucket[
                    r["bucket"]
                ] += 1
                added_class[
                    r["category_name"]
                ] += 1

            if (
                idx % 25 == 0
                or idx == int(
                    om["images_count"]
                )
            ):
                print(
                    f"{idx}/{om['images_count']} "
                    f"base={base_count:,} "
                    f"added={added_count}"
                )

    print()
    print("===== FINAL CHECK =====")

    print(
        "Base detections:",
        f"{base_count:,}",
    )

    print(
        "Stitched:",
        f"{stitched_count:,}",
    )

    print(
        "Added:",
        added_count,
    )

    print(
        "Total:",
        f"{base_count + added_count:,}",
    )

    print(
        "Zero scores:",
        zero_scores,
    )

    if base_count != 2403809:
        raise RuntimeError(
            "Final Combo count mismatch"
        )

    if added_count != len(selected):
        raise RuntimeError(
            "Selected/add count mismatch"
        )

    make_zip(
        submission_path,
        zip_path,
    )

    result = {
        "score_floor":
            args.score_floor,

        "base_detections":
            base_count,

        "added_baseline_boxes":
            added_count,

        "total_detections":
            base_count + added_count,

        "buckets":
            dict(added_bucket),

        "classes":
            dict(added_class),

        "zero_scores":
            zero_scores,

        "json":
            str(submission_path),

        "zip":
            str(zip_path),

        "json_sha256":
            sha256_file(
                submission_path
            ),

        "zip_sha256":
            sha256_file(
                zip_path
            ),
    }

    result_path = (
        args.output_dir
        / "BASELINE_UNIQUE_RESULT.json"
    )

    result_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("JSON:", submission_path)
    print("ZIP :", zip_path)

    print(
        "JSON SHA256:",
        result["json_sha256"],
    )

    print(
        "ZIP SHA256:",
        result["zip_sha256"],
    )


if __name__ == "__main__":
    main()
