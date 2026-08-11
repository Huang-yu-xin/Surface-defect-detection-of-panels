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


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(path)

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
        "--output-dir",
        type=Path,
        required=True,
    )

    p.add_argument(
        "--device",
        default="cpu",
    )

    p.add_argument(
        "--dup-score",
        type=float,
        default=1e-6,
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
            "  "
            + json.dumps(
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


def make_dup(final, rules, dup_score):
    """
    rules:
        [(src, dst, topk, min_source_score), ...]

    topk=None means score-only.
    """

    cls = final[:, 0].astype(np.int32)

    pieces = []
    by_rule = Counter()

    for rule_name, src, dst, topk, score_floor in rules:
        rows = final[cls == src]

        if not len(rows):
            continue

        order = np.argsort(
            -rows[:, 1],
            kind="stable",
        )

        if topk is None:
            rows = rows[order]
        else:
            rows = rows[order[:topk]]

        rows = rows[
            rows[:, 1] >= score_floor
        ]

        if not len(rows):
            continue

        dup = rows.copy()

        dup[:, 0] = dst
        dup[:, 1] = dup_score

        pieces.append(dup)
        by_rule[rule_name] += len(dup)

    if not pieces:
        return (
            np.empty(
                (0, final.shape[1]),
                dtype=np.float32,
            ),
            by_rule,
        )

    return (
        np.concatenate(
            pieces,
            axis=0,
        ),
        by_rule,
    )


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
        "combo18_s32",
    )

    diag = load_module(
        Path("scripts/14_fn_diagnostic.py"),
        "diag14_s32",
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

    om = combo.load_manifest(
        args.original_cache
    )

    hm = combo.load_manifest(
        args.hflip_cache
    )

    hmap = combo.manifest_map(hm)

    if int(om["images_count"]) != int(hm["images_count"]):
        raise RuntimeError(
            "Original/HFlip count mismatch"
        )

    #
    # Three variants for Test distribution audit.
    #
    variants = {
        "mid_20_5": [
            (
                "7to0",
                7, 0,
                20,
                1e-4,
            ),
            (
                "6to0",
                6, 0,
                5,
                1e-3,
            ),
        ],

        "mid_50_20": [
            (
                "7to0",
                7, 0,
                50,
                1e-4,
            ),
            (
                "6to0",
                6, 0,
                20,
                1e-3,
            ),
        ],

        "score_only": [
            (
                "7to0",
                7, 0,
                None,
                1e-4,
            ),
            (
                "6to0",
                6, 0,
                None,
                1e-3,
            ),
        ],
    }

    chosen = "mid_50_20"

    submission_path = (
        args.output_dir
        / "submission_mid_50_20.json"
    )

    zip_path = (
        args.output_dir
        / "submission_mid_50_20.zip"
    )

    summary_path = (
        args.output_dir
        / "submission_mid_50_20_summary.csv"
    )

    base_count = 0
    stitched_count = 0

    fusion_count = Counter()

    fusion_by_rule = {
        name: Counter()
        for name in variants
    }

    class_count = Counter()
    summary_rows = []

    zero_scores = 0

    print(
        "===== CLASS-CONFUSION TEST SUBMISSION ====="
    )

    print(
        "Images :",
        om["images_count"],
    )

    print(
        "Chosen :",
        chosen,
    )

    print()

    with JsonArrayWriter(
        submission_path
    ) as writer:

        for idx, oitem in enumerate(
            om["items"],
            start=1,
        ):
            name = oitem["image_name"]

            if name not in hmap:
                raise RuntimeError(
                    f"HFlip cache missing {name}"
                )

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

            #
            # Exact Final Combo reconstruction.
            #
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

            final, merged_count = (
                combo.add_stitched(
                    post,
                    stitch_args,
                    device,
                )
            )

            base_count += len(final)
            stitched_count += merged_count

            #
            # Count all three variants.
            #
            chosen_dup = None

            for vname, rules in variants.items():
                dup, by_rule = make_dup(
                    final,
                    rules,
                    args.dup_score,
                )

                fusion_count[vname] += len(dup)

                for k, v in by_rule.items():
                    fusion_by_rule[vname][k] += v

                if vname == chosen:
                    chosen_dup = dup

            if chosen_dup is None:
                raise RuntimeError(
                    "Chosen variant missing"
                )

            #
            # Write exact existing Final Combo first.
            #
            for det in final:
                row = combo.submission_row(
                    name,
                    width,
                    height,
                    det,
                )

                writer.write(row)

                class_count[
                    row["category_name"]
                ] += 1

            #
            # Then append class-confusion duplicates.
            #
            for det in chosen_dup:
                row = combo.submission_row(
                    name,
                    width,
                    height,
                    det,
                )

                if row["score"] == 0:
                    zero_scores += 1

                writer.write(row)

                class_count[
                    row["category_name"]
                ] += 1

            summary_rows.append({
                "image_id": name,
                "width": width,
                "height": height,
                "base_detections": len(final),
                "mid_20_5_added":
                    0,  # filled only in aggregate
                "mid_50_20_added":
                    len(chosen_dup),
                "stitched_zonglie":
                    merged_count,
                "submission_detections":
                    len(final)
                    + len(chosen_dup),
            })

            if (
                idx % 25 == 0
                or idx == int(
                    om["images_count"]
                )
            ):
                print(
                    f"{idx}/{om['images_count']} "
                    f"base={base_count:,} "
                    f"mid20_5={fusion_count['mid_20_5']:,} "
                    f"mid50_20={fusion_count['mid_50_20']:,} "
                    f"scoreonly={fusion_count['score_only']:,}"
                )

    #
    # Summary CSV.
    #
    with summary_path.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                summary_rows[0].keys()
            ),
        )

        w.writeheader()
        w.writerows(summary_rows)

    #
    # ZIP with exact required internal filename.
    #
    make_zip(
        submission_path,
        zip_path,
    )

    result = {
        "images":
            int(om["images_count"]),

        "base_detections":
            base_count,

        "stitched_zonglie":
            stitched_count,

        "chosen_variant":
            chosen,

        "variants": {
            vname: {
                "added_boxes":
                    fusion_count[vname],

                "by_rule":
                    dict(
                        fusion_by_rule[vname]
                    ),
            }
            for vname in variants
        },

        "submission_total":
            base_count
            + fusion_count[chosen],

        "zero_rounded_dup_scores":
            zero_scores,

        "json":
            str(submission_path),

        "zip":
            str(zip_path),

        "json_mb":
            submission_path.stat().st_size
            / 1024 / 1024,

        "zip_mb":
            zip_path.stat().st_size
            / 1024 / 1024,

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
        / "CONFUSION_SUBMISSION_RESULT.json"
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
    print(
        "===== TEST DISTRIBUTION ====="
    )

    print(
        "Base detections:",
        f"{base_count:,}",
    )

    print(
        "Stitched:",
        f"{stitched_count:,}",
    )

    for vname in variants:
        print()
        print(f"[{vname}]")

        print(
            "Added:",
            f"{fusion_count[vname]:,}",
        )

        for rule, n in (
            fusion_by_rule[vname].items()
        ):
            print(
                f"  {rule:<6} {n:,}"
            )

    print()
    print(
        "===== SUBMISSION ====="
    )

    print(
        "Chosen:",
        chosen,
    )

    print(
        "Total:",
        f"{base_count + fusion_count[chosen]:,}",
    )

    print(
        "Zero scores:",
        zero_scores,
    )

    print(
        "JSON:",
        submission_path,
    )

    print(
        "ZIP:",
        zip_path,
    )

    print(
        "JSON MB:",
        f"{result['json_mb']:.2f}",
    )

    print(
        "ZIP MB:",
        f"{result['zip_mb']:.2f}",
    )

    print(
        "JSON SHA256:",
        result["json_sha256"],
    )

    print(
        "ZIP SHA256:",
        result["zip_sha256"],
    )

    print(
        "Manifest:",
        result_path,
    )


if __name__ == "__main__":
    main()
