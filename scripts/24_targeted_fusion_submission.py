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
    p.add_argument("--original-cache", type=Path, required=True)
    p.add_argument("--hflip-cache", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--pair-topk", type=int, default=3)
    p.add_argument("--pair-chunk", type=int, default=32)
    return p.parse_args()


class JsonArrayWriter:
    """
    Memory-safe streaming JSON array writer.
    Avoids holding millions of submission dicts in RAM.
    """

    def __init__(self, path):
        self.path = Path(path)
        self.f = None
        self.first = True
        self.count = 0

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.f = self.path.open("w", encoding="utf-8")
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
        # Competition requires this exact internal name.
        z.write(
            json_path,
            arcname="submission.json",
        )


def build_rules():
    #
    # pruned_v1 = J1 + H1 + A1 + B1
    #
    J1 = {
        "classes": {0},

        "min_pair_iou": 0.00,
        "max_pair_iou": 0.12,

        "min_dxn": 0.75,
        "max_dxn": 1.05,

        "max_dyn": 0.25,

        "max_wr": 1.60,
        "max_hr": 1.60,
        "max_ar": 2.00,

        "max_score_ratio": 10.0,

        "methods": ["envelope"],
    }

    H1 = {
        "classes": {5},

        "min_pair_iou": 0.25,
        "max_pair_iou": 0.50,

        "min_dxn": 0.30,
        "max_dxn": 0.55,

        "max_dyn": 0.06,

        "min_wr": 1.45,
        "max_wr": 2.20,

        "max_hr": 1.30,

        "min_ar": 1.30,
        "max_ar": 2.30,

        "methods": ["envelope"],
    }

    A1 = {
        "classes": {3},

        "min_pair_iou": 0.25,
        "max_pair_iou": 0.40,

        "max_dxn": 0.15,

        "min_dyn": 0.25,
        "max_dyn": 0.50,

        "max_wr": 1.50,

        "min_hr": 2.00,
        "max_hr": 3.30,

        "min_ar": 2.40,
        "max_ar": 4.00,

        "max_score_ratio": 5.0,

        "methods": ["avg50"],
    }

    B1 = {
        "classes": {3},

        "min_pair_iou": 0.00,
        "max_pair_iou": 0.05,

        "min_dxn": 1.60,
        "max_dxn": 2.10,

        "min_dyn": 0.10,
        "max_dyn": 0.45,

        "max_wr": 1.70,
        "max_hr": 1.70,
        "max_ar": 2.30,

        "max_score_ratio": 6.0,

        "methods": ["envelope"],
    }

    #
    # pruned_v2_Hscore = J2 + H2 + A2 + B2
    #
    J2 = {
        "classes": {0},

        "min_pair_iou": 0.00,
        "max_pair_iou": 0.08,

        "min_dxn": 0.85,
        "max_dxn": 1.00,

        "max_dyn": 0.18,

        "max_wr": 1.40,
        "max_hr": 1.40,
        "max_ar": 1.70,

        "max_score_ratio": 6.0,

        "methods": ["envelope"],
    }

    H2 = {
        "classes": {5},

        "min_pair_iou": 0.25,
        "max_pair_iou": 0.50,

        "min_dxn": 0.30,
        "max_dxn": 0.55,

        "max_dyn": 0.06,

        "min_wr": 1.45,
        "max_wr": 2.20,

        "max_hr": 1.30,

        "min_ar": 1.30,
        "max_ar": 2.30,

        "min_score_ratio": 100.0,

        "methods": ["envelope"],
    }

    A2 = {
        "classes": {3},

        "min_pair_iou": 0.28,
        "max_pair_iou": 0.36,

        "max_dxn": 0.10,

        "min_dyn": 0.30,
        "max_dyn": 0.45,

        "max_wr": 1.35,

        "min_hr": 2.40,
        "max_hr": 3.20,

        "min_ar": 2.80,
        "max_ar": 3.70,

        "max_score_ratio": 4.0,

        "methods": ["avg50"],
    }

    B2 = {
        "classes": {3},

        "min_pair_iou": 0.00,
        "max_pair_iou": 0.02,

        "min_dxn": 1.75,
        "max_dxn": 2.05,

        "min_dyn": 0.15,
        "max_dyn": 0.40,

        "max_wr": 1.65,
        "max_hr": 1.60,
        "max_ar": 2.10,

        "max_score_ratio": 5.5,

        "methods": ["envelope"],
    }

    return {
        "pruned_v1": [
            ("J1", J1),
            ("H1", H1),
            ("A1", A1),
            ("B1", B1),
        ],

        "pruned_v2_Hscore": [
            ("J2", J2),
            ("H2", H2),
            ("A2", A2),
            ("B2", B2),
        ],
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    combo = load_module(
        Path("scripts/18_final_combo_from_cache.py"),
        "combo18_s24",
    )
    diag = load_module(
        Path("scripts/14_fn_diagnostic.py"),
        "diag14_s24",
    )
    s20 = load_module(
        Path("scripts/20_crossview_fusion_sweep.py"),
        "fusion20_s24",
    )
    s23 = load_module(
        Path("scripts/23_targeted_fusion_prune_sweep.py"),
        "fusion23_s24",
    )

    device = torch.device(args.device)

    om = combo.load_manifest(args.original_cache)
    hm = combo.load_manifest(args.hflip_cache)
    hmap = combo.manifest_map(hm)

    if int(om["images_count"]) != int(hm["images_count"]):
        raise RuntimeError(
            "Original/HFlip image count mismatch"
        )

    #
    # Exact validated Final Combo defaults.
    #
    old_argv = sys.argv[:]

    try:
        sys.argv = [
            "18_final_combo_from_cache.py",
            "--mode", "test",
            "--original-cache", "_dummy_orig",
            "--hflip-cache", "_dummy_hflip",
            "--output-dir", "_dummy_out",
        ]
        defaults = combo.parse_args()
    finally:
        sys.argv = old_argv

    stitch_args = SimpleNamespace(**vars(defaults))

    stitch_args.min_aspect = 5.0
    stitch_args.x_tol = 64.0
    stitch_args.max_y_gap = 64.0
    stitch_args.min_merged_height = 1300.0

    hclasses = {0, 2, 3, 4, 7}

    rules = build_rules()

    variant_names = [
        "pruned_v1",
        "pruned_v2_Hscore",
    ]

    json_paths = {
        v: args.output_dir / f"submission_{v}.json"
        for v in variant_names
    }

    zip_paths = {
        v: args.output_dir / f"submission_{v}.zip"
        for v in variant_names
    }

    summary_paths = {
        v: args.output_dir / f"summary_{v}.csv"
        for v in variant_names
    }

    counts = {
        v: Counter()
        for v in variant_names
    }

    class_counts = {
        v: Counter()
        for v in variant_names
    }

    fusion_counts = {
        v: Counter()
        for v in variant_names
    }

    zero_rounded_fusion_scores = Counter()

    base_detection_count = 0
    total_stitched = 0

    summary_rows = {
        v: []
        for v in variant_names
    }

    print("===== TARGETED FUSION TEST SUBMISSION =====")
    print("Device       :", device)
    print("Images       :", om["images_count"])
    print("Pair top-k   :", args.pair_topk)
    print("Pair chunk   :", args.pair_chunk)
    print("Variants     :", ", ".join(variant_names))
    print()

    with (
        JsonArrayWriter(json_paths["pruned_v1"]) as w_v1,
        JsonArrayWriter(json_paths["pruned_v2_Hscore"]) as w_v2,
    ):

        writers = {
            "pruned_v1": w_v1,
            "pruned_v2_Hscore": w_v2,
        }

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

            #
            # Match script 18 exactly.
            #
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
                0.90,
                diag,
                device,
            )

            final, merged_count = combo.add_stitched(
                post,
                stitch_args,
                device,
            )

            total_stitched += merged_count
            base_detection_count += len(final)

            #
            # Pair search only for the three targeted classes.
            #
            pairs = []

            for cls in [0, 3, 5]:
                pairs.extend(
                    s20.build_pair_features(
                        orig,
                        hflip,
                        cls,
                        args.pair_topk,
                        args.pair_chunk,
                    )
                )

            fused_by_variant = {}

            for vname in variant_names:
                pieces = []

                for rule_name, cfg in rules[vname]:
                    rows = s23.make_rows(
                        s20,
                        pairs,
                        cfg,
                        final.shape[1],
                    )

                    fusion_counts[vname][rule_name] += len(rows)

                    if len(rows):
                        pieces.append(rows)

                fused = (
                    np.concatenate(pieces, axis=0)
                    if pieces
                    else np.empty(
                        (0, final.shape[1]),
                        dtype=np.float32,
                    )
                )

                fused_by_variant[vname] = fused

                for det in fused:
                    if round(float(det[1]), 6) == 0.0:
                        zero_rounded_fusion_scores[vname] += 1

            #
            # Stream each variant directly to JSON.
            #
            for vname in variant_names:
                writer = writers[vname]
                fused = fused_by_variant[vname]

                # Existing Final Combo predictions first.
                for det in final:
                    row = combo.submission_row(
                        name,
                        width,
                        height,
                        det,
                    )

                    writer.write(row)
                    counts[vname]["base"] += 1
                    class_counts[vname][
                        row["category_name"]
                    ] += 1

                # Then low-score fusion proposals.
                for det in fused:
                    row = combo.submission_row(
                        name,
                        width,
                        height,
                        det,
                    )

                    writer.write(row)
                    counts[vname]["fusion"] += 1
                    class_counts[vname][
                        row["category_name"]
                    ] += 1

                summary_rows[vname].append({
                    "image_id": name,
                    "width": width,
                    "height": height,
                    "original_candidates": len(orig),
                    "selected_hflip_candidates": len(hsel),
                    "stitched_zonglie": merged_count,
                    "base_final_detections": len(final),
                    "fusion_detections": len(fused),
                    "total_detections": len(final) + len(fused),
                })

            if (
                idx % 25 == 0
                or idx == int(om["images_count"])
            ):
                print(
                    f"{idx}/{om['images_count']} "
                    f"base={base_detection_count:,} "
                    f"v1fusion={counts['pruned_v1']['fusion']:,} "
                    f"v2fusion={counts['pruned_v2_Hscore']['fusion']:,} "
                    f"stitched={total_stitched:,}"
                )

    #
    # Summaries + ZIP packages.
    #
    for vname in variant_names:
        rows = summary_rows[vname]

        with summary_paths[vname].open(
            "w",
            encoding="utf-8-sig",
            newline="",
        ) as f:
            w = csv.DictWriter(
                f,
                fieldnames=list(rows[0].keys()),
            )
            w.writeheader()
            w.writerows(rows)

        make_zip(
            json_paths[vname],
            zip_paths[vname],
        )

    #
    # Result manifest.
    #
    result = {
        "images": int(om["images_count"]),
        "base_detection_count": base_detection_count,
        "stitched_zonglie": total_stitched,
        "variants": {},
    }

    for vname in variant_names:
        total = (
            counts[vname]["base"]
            + counts[vname]["fusion"]
        )

        result["variants"][vname] = {
            "base_detections":
                counts[vname]["base"],

            "fusion_detections":
                counts[vname]["fusion"],

            "total_detections":
                total,

            "zero_rounded_fusion_scores":
                zero_rounded_fusion_scores[vname],

            "fusion_by_rule":
                dict(fusion_counts[vname]),

            "class_counts":
                dict(class_counts[vname]),

            "json":
                str(json_paths[vname]),

            "zip":
                str(zip_paths[vname]),

            "json_mb":
                json_paths[vname].stat().st_size
                / 1024 / 1024,

            "zip_mb":
                zip_paths[vname].stat().st_size
                / 1024 / 1024,

            "json_sha256":
                sha256_file(json_paths[vname]),

            "zip_sha256":
                sha256_file(zip_paths[vname]),
        }

    manifest_path = (
        args.output_dir
        / "TARGETED_SUBMISSION_RESULT.json"
    )

    manifest_path.write_text(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("===== TEST GENERATION COMPLETE =====")
    print("Images       :", om["images_count"])
    print("Base dets    :", f"{base_detection_count:,}")
    print("Stitched     :", f"{total_stitched:,}")

    for vname in variant_names:
        r = result["variants"][vname]

        print()
        print(f"[{vname}]")
        print(
            "Fusion dets  :",
            f"{r['fusion_detections']:,}",
        )
        print(
            "Total dets   :",
            f"{r['total_detections']:,}",
        )
        print(
            "Zero scores  :",
            r["zero_rounded_fusion_scores"],
        )

        print("Fusion by rule:")
        for k, v in r["fusion_by_rule"].items():
            print(f"  {k:<4} {v:,}")

        print(
            "JSON         :",
            r["json"],
        )
        print(
            "ZIP          :",
            r["zip"],
        )
        print(
            "JSON MB      :",
            f"{r['json_mb']:.2f}",
        )
        print(
            "ZIP MB       :",
            f"{r['zip_mb']:.2f}",
        )
        print(
            "JSON SHA256  :",
            r["json_sha256"],
        )
        print(
            "ZIP SHA256   :",
            r["zip_sha256"],
        )

    print()
    print("Manifest:", manifest_path)


if __name__ == "__main__":
    main()
