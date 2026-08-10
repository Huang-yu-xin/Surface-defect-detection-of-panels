from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

COLS = [
    "class_id", "score", "xmin", "ymin", "xmax", "ymax",
    "tile_x", "tile_y", "valid_w", "valid_h",
    "local_xmin", "local_ymin", "local_xmax", "local_ymax",
]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Cache horizontally-flipped Val TTA candidates. "
            "All boxes are mapped back to original-image coordinates."
        )
    )
    p.add_argument(
        "--model",
        type=Path,
        default=Path(
            "runs/rareos/"
            "yolo26m_tiles1280_rareos_v1_e80_b6_seed2026/"
            "weights/best.pt"
        ),
    )
    p.add_argument(
        "--images",
        type=Path,
        default=Path("datasets/yolo_split/images/val"),
    )
    p.add_argument(
        "--cache-dir",
        type=Path,
        default=Path("results/fn_analysis/cache_hflip"),
    )
    p.add_argument("--tile-size", type=int, default=1280)
    p.add_argument("--stride", type=int, default=768)
    p.add_argument("--batch", type=int, default=6)
    p.add_argument("--conf", type=float, default=1e-5)
    p.add_argument("--tile-iou", type=float, default=0.60)
    p.add_argument("--max-det", type=int, default=1000)
    p.add_argument("--device", type=str, default="0")
    p.add_argument("--half", action="store_true")
    return p.parse_args()


def get_starts(length: int, tile_size: int, stride: int) -> list[int]:
    if length <= tile_size:
        return [0]
    starts = list(range(0, length - tile_size + 1, stride))
    final_start = length - tile_size
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


def main():
    args = parse_args()

    import torch
    from ultralytics import YOLO

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; run this script while GPU is attached.")

    if not args.model.exists():
        raise FileNotFoundError(f"Model not found: {args.model}")
    if not args.images.exists():
        raise FileNotFoundError(f"Images dir not found: {args.images}")

    image_paths = sorted(
        p for p in args.images.iterdir()
        if p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise RuntimeError(f"No images found: {args.images}")

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.model))

    manifest = {
        "model": str(args.model),
        "images": str(args.images),
        "transform": "horizontal_flip",
        "coordinate_space": "original_image",
        "tile_size": args.tile_size,
        "stride": args.stride,
        "batch": args.batch,
        "conf": args.conf,
        "tile_iou": args.tile_iou,
        "max_det": args.max_det,
        "columns": COLS,
        "images_count": len(image_paths),
        "items": [],
    }

    start_time = time.time()
    total_candidates = 0

    for image_index, image_path in enumerate(image_paths, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Cannot read: {image_path}")

        height, width = image.shape[:2]
        flipped = cv2.flip(image, 1)

        xs = get_starts(width, args.tile_size, args.stride)
        ys = get_starts(height, args.tile_size, args.stride)

        tiles = []
        metas = []

        for y in ys:
            for x in xs:
                crop = flipped[
                    y:min(height, y + args.tile_size),
                    x:min(width, x + args.tile_size),
                ]
                valid_h, valid_w = crop.shape[:2]

                if valid_h != args.tile_size or valid_w != args.tile_size:
                    padded = np.zeros(
                        (args.tile_size, args.tile_size, 3),
                        dtype=np.uint8,
                    )
                    padded[:valid_h, :valid_w] = crop
                    crop = padded

                tiles.append(crop)
                metas.append((x, y, valid_w, valid_h))

        rows = []

        for start in range(0, len(tiles), args.batch):
            batch_tiles = tiles[start:start + args.batch]
            batch_metas = metas[start:start + args.batch]

            results = model.predict(
                batch_tiles,
                imgsz=args.tile_size,
                conf=args.conf,
                iou=args.tile_iou,
                max_det=args.max_det,
                device=args.device,
                half=args.half,
                verbose=False,
            )

            for result, meta in zip(results, batch_metas):
                flip_tile_x, tile_y, valid_w, valid_h = meta

                if result.boxes is None or len(result.boxes) == 0:
                    continue

                xyxy = result.boxes.xyxy.detach().cpu().numpy()
                cls = result.boxes.cls.detach().cpu().numpy().astype(int)
                scores = result.boxes.conf.detach().cpu().numpy()

                # This flipped-image tile maps to this x-origin in original coordinates.
                original_tile_x = width - (flip_tile_x + valid_w)

                for box, cid, score in zip(xyxy, cls, scores):
                    fx1, fy1, fx2, fy2 = map(float, box)

                    fx1 = float(np.clip(fx1, 0, valid_w))
                    fx2 = float(np.clip(fx2, 0, valid_w))
                    fy1 = float(np.clip(fy1, 0, valid_h))
                    fy2 = float(np.clip(fy2, 0, valid_h))
                    if fx2 <= fx1 or fy2 <= fy1:
                        continue

                    # Local box in the corresponding original-orientation tile.
                    lx1 = valid_w - fx2
                    lx2 = valid_w - fx1
                    ly1 = fy1
                    ly2 = fy2

                    # Global box, mapped back to original-image coordinates.
                    ox1 = original_tile_x + lx1
                    ox2 = original_tile_x + lx2
                    oy1 = tile_y + ly1
                    oy2 = tile_y + ly2

                    ox1 = float(np.clip(ox1, 0, width))
                    ox2 = float(np.clip(ox2, 0, width))
                    oy1 = float(np.clip(oy1, 0, height))
                    oy2 = float(np.clip(oy2, 0, height))
                    if ox2 <= ox1 or oy2 <= oy1:
                        continue

                    rows.append([
                        int(cid),
                        float(score),
                        ox1,
                        oy1,
                        ox2,
                        oy2,
                        original_tile_x,
                        tile_y,
                        valid_w,
                        valid_h,
                        lx1,
                        ly1,
                        lx2,
                        ly2,
                    ])

        arr = np.asarray(rows, dtype=np.float32)
        if arr.size == 0:
            arr = np.empty((0, len(COLS)), dtype=np.float32)

        cache_path = args.cache_dir / f"{image_path.stem}.npz"
        np.savez(
            cache_path,
            candidates=arr,
            image_shape=np.asarray([height, width], dtype=np.int32),
        )

        total_candidates += len(arr)
        manifest["items"].append({
            "image_name": image_path.name,
            "cache_file": cache_path.name,
            "height": height,
            "width": width,
            "candidate_count": int(len(arr)),
        })

        if image_index % 25 == 0 or image_index == len(image_paths):
            elapsed = (time.time() - start_time) / 60
            print(
                f"{image_index}/{len(image_paths)} "
                f"candidates={total_candidates:,} "
                f"elapsed={elapsed:.2f} min"
            )

    manifest["total_candidates"] = total_candidates
    manifest["elapsed_minutes"] = (time.time() - start_time) / 60

    (args.cache_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print()
    print("HFlip cache saved:", args.cache_dir)
    print("Images            :", len(image_paths))
    print("Candidates        :", f"{total_candidates:,}")
    print("Transform         : horizontal_flip")
    print("Coordinates       : mapped back to original image")


if __name__ == "__main__":
    main()
