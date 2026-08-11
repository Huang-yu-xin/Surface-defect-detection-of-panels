#!/usr/bin/env bash
set -euo pipefail

cd /root/autodl-tmp/steel_defect

MODEL="runs/baseline/yolo26m_tiles1280_e80_b6_seed2026/weights/best.pt"
CACHE="results/baseline_complementarity/cache_original"
OUT="results/baseline_complementarity/analysis"

echo "===== GPU ====="
nvidia-smi

echo
echo "===== BUILD BASELINE VAL CACHE ====="

python -u scripts/14_fn_diagnostic.py \
  --model "$MODEL" \
  --images datasets/yolo_split/images/val \
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

p = Path("results/baseline_complementarity/cache_original/manifest.json")
x = json.loads(p.read_text())

for k in [
    "model",
    "images",
    "images_count",
    "total_candidates",
    "tile_size",
    "stride",
    "conf",
    "tile_iou",
    "max_det",
    "elapsed_minutes",
]:
    print(f"{k:18s}: {x.get(k)}")

print(
    "npz files          :",
    len(list(p.parent.glob("*.npz")))
)
PY

du -sh "$CACHE"
