#!/usr/bin/env python3

import csv
import gc
import importlib.util
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch


# ============================================================
# Reuse ALL Final Combo logic from Script 18.
# Only submission writing is changed to streaming mode.
# ============================================================

SCRIPT18 = Path(__file__).resolve().parent / "18_final_combo_from_cache.py"

spec = importlib.util.spec_from_file_location(
    "final_combo_v18",
    SCRIPT18,
)

m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def main():
    args = m.parse_args()

    if args.mode != "test":
        raise RuntimeError(
            "Script 40 is streaming TEST-only. "
            "Use Script 18 for validation."
        )

    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    device = torch.device(args.device)
    diag = m.load_diag(args.diag_script)

    om = m.load_manifest(args.original_cache)
    hm = m.load_manifest(args.hflip_cache)
    hmap = m.manifest_map(hm)

    if int(om["images_count"]) != int(hm["images_count"]):
        raise RuntimeError("Original/HFlip image count mismatch")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.submission is None:
        args.submission = (
            args.output_dir / "submission_final_combo.json"
        )

    if args.summary_csv is None:
        args.summary_csv = (
            args.output_dir /
            "submission_final_combo_summary.csv"
        )

    args.submission.parent.mkdir(parents=True, exist_ok=True)
    args.summary_csv.parent.mkdir(parents=True, exist_ok=True)

    # Write to temporary file first.
    # Only rename after the entire submission is complete.
    tmp_submission = Path(
        str(args.submission) + ".tmp"
    )

    hclasses = set(args.hflip_classes)

    summary_rows = []
    total_merged = 0
    detection_count = 0
    nms_backend_seen = set()

    class_counts = Counter()

    zero_scores = 0
    nonfinite_scores = 0

    first_row = True

    print("Mode         : test-stream")
    print("Device       :", device)
    print("Images       :", om["images_count"])
    print(
        "HFlip classes:",
        [m.CLASS_NAMES[i] for i in sorted(hclasses)],
    )
    print(
        "Stitch       :",
        f"aspect>={args.min_aspect}, "
        f"x_tol={args.x_tol}, "
        f"gap={args.max_y_gap}",
    )
    print("Submission   :", args.submission)
    print("Temporary    :", tmp_submission)
    print()

    # ========================================================
    # Streaming JSON array
    # ========================================================

    with tmp_submission.open(
        "w",
        encoding="utf-8",
        buffering=1024 * 1024,
    ) as jf:

        jf.write("[\n")

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

            # Explicitly close NPZ files every iteration.
            with np.load(
                args.original_cache / oitem["cache_file"]
            ) as onpz:
                orig = onpz["candidates"].astype(
                    np.float32,
                    copy=False,
                )
                height, width = map(
                    int,
                    onpz["image_shape"],
                )

            with np.load(
                args.hflip_cache / hitem["cache_file"]
            ) as hnpz:
                hflip = hnpz["candidates"].astype(
                    np.float32,
                    copy=False,
                )

            hsel = hflip[
                np.isin(
                    hflip[:, 0].astype(np.int32),
                    list(hclasses),
                )
            ]

            if len(hsel):
                union_pre = np.concatenate(
                    [orig, hsel],
                    axis=0,
                )
            else:
                union_pre = orig

            post, backend = m.nms(
                union_pre,
                args.global_iou,
                diag,
                device,
            )

            nms_backend_seen.add(backend)

            final, merged_count = m.add_stitched(
                post,
                args,
                device,
            )

            total_merged += merged_count

            # ------------------------------------------------
            # Stream rows immediately.
            # Nothing is accumulated into a 2.4M-item list.
            # ------------------------------------------------

            for det in final:
                row = m.submission_row(
                    name,
                    width,
                    height,
                    det,
                )

                if not first_row:
                    jf.write(",\n")

                jf.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )

                first_row = False
                detection_count += 1

                cname = row.get("category_name")
                if cname is not None:
                    class_counts[cname] += 1

                score = None
                for key in (
                    "score",
                    "confidence",
                    "conf",
                ):
                    if key in row:
                        score = float(row[key])
                        break

                if score is not None:
                    if score == 0:
                        zero_scores += 1

                    if not math.isfinite(score):
                        nonfinite_scores += 1

            summary_rows.append(
                {
                    "image_id": name,
                    "width": width,
                    "height": height,
                    "original_candidates": len(orig),
                    "selected_hflip_candidates": len(hsel),
                    "stitched_zonglie": merged_count,
                    "final_detection_count": len(final),
                }
            )

            if idx % 25 == 0 or idx == int(om["images_count"]):
                jf.flush()

                print(
                    f"{idx}/{om['images_count']} "
                    f"final={detection_count:,} "
                    f"stitched={total_merged:,}"
                )

            # Be explicit about releasing per-image objects.
            del orig
            del hflip
            del hsel
            del union_pre
            del post
            del final

            if idx % 25 == 0:
                gc.collect()

        jf.write("\n]\n")
        jf.flush()

    # ========================================================
    # Atomic-ish finalization
    # ========================================================

    tmp_submission.replace(args.submission)

    with args.summary_csv.open(
        "w",
        encoding="utf-8-sig",
        newline="",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(summary_rows[0].keys()),
        )

        writer.writeheader()
        writer.writerows(summary_rows)

    print()
    print("===== FINAL COMBO TEST STREAM =====")
    print("Images       :", om["images_count"])
    print("Detections   :", f"{detection_count:,}")
    print("Stitched     :", f"{total_merged:,}")
    print(
        "NMS          :",
        ", ".join(sorted(nms_backend_seen)),
    )
    print("Zero scores  :", f"{zero_scores:,}")
    print("Nonfinite    :", f"{nonfinite_scores:,}")
    print("Submission   :", args.submission)
    print(
        "JSON MB      :",
        f"{args.submission.stat().st_size / 1024 / 1024:.2f}",
    )

    print("By class:")
    for c in m.CLASS_NAMES:
        print(
            f"  {c:<18}"
            f"{class_counts[c]:,}"
        )


if __name__ == "__main__":
    main()
