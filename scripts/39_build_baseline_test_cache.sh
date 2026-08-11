#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/steel_defect

MODEL="runs/baseline/yolo26m_tiles1280_e80_b6_seed2026/weights/best.pt"
CACHE="results/baseline_complementarity/cache_test_original"
OUT="results/baseline_complementarity/test_analysis"

echo "===== GPU ====="
nvidia-smi

echo
echo "===== BUILD BASELINE TEST HIGH-RECALL CACHE ====="
echo "MODEL : $MODEL"
echo "IMAGES: raw/data/test"
echo "CACHE : $CACHE"
echo

python -u scripts/14_fn_diagnostic.py \
  --model "$MODEL" \
  --images raw/data/test \
  --labels datasets/yolo_split/labels/val \
  --cache-dir "$CACHE" \
  --output-dir "$OUT" \
  --tile-size 1280 \
  --stride 768 \
  --batch 6 \
  --conf 1e-5 \
  --tile-iou 0.60 \
  --global-iou 0.90 \
  --max-det 1000 \
  --match-iou 0.50 \
  --device 0 \
  --half \
  --build-cache

echo
echo "===== CACHE CHECK ====="

python - <<'PY'
import json
from pathlib import Path

p = Path(
    "results/baseline_complementarity/"
    "cache_test_original/manifest.json"
)

if not p.exists():
    raise SystemExit(f"ERROR: manifest not found: {p}")

x = json.loads(p.read_text())

for k in [
    "model",
    "images",
    "labels",
    "images_count",
    "total_candidates",
    "tile_size",
    "stride",
    "batch",
    "conf",
    "tile_iou",
    "max_det",
    "elapsed_minutes",
]:
    print(f"{k:18s}: {x.get(k)}")

npz_count = len(list(p.parent.glob("*.npz")))
print(f"{'npz files':18s}: {npz_count}")

if x.get("images_count") != 669:
    raise SystemExit(
        f"ERROR: expected images_count=669, got {x.get('images_count')}"
    )

if npz_count != 669:
    raise SystemExit(
        f"ERROR: expected 669 NPZ files, got {npz_count}"
    )

print()
print("CACHE COMPLETE.")
PY

echo
du -sh "$CACHE"
