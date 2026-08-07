#!/usr/bin/env bash

set -euo pipefail

ROOT="/root/autodl-tmp/steel_defect"

cd "$ROOT"

export OMP_NUM_THREADS=8

echo "========================================"
echo " GPU CHECK"
echo "========================================"

nvidia-smi

python - <<'PY'
import torch

print("PyTorch:", torch.__version__)
print("CUDA:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise SystemExit("ERROR: CUDA unavailable")

print("GPU:", torch.cuda.get_device_name(0))
print("Capability:", torch.cuda.get_device_capability(0))
PY


echo
echo "========================================"
echo " STEP 1: Baseline-1 best.pt validation"
echo "========================================"

yolo detect val \
  model="$ROOT/runs/baseline/yolo26m_tiles1280_e80_b6_seed2026/weights/best.pt" \
  data="$ROOT/configs/steel_tiles_1280.yaml" \
  imgsz=1280 \
  batch=6 \
  device=0 \
  workers=8 \
  plots=True \
  project="$ROOT/runs/baseline_eval" \
  name=yolo26m_tiles1280_best_e73 \
  exist_ok=True


echo
echo "========================================"
echo " STEP 2: Baseline-2 one-epoch smoke"
echo "========================================"

yolo detect train \
  model=yolo26m.pt \
  data="$ROOT/configs/steel_tiles_1280_rareos_v1.yaml" \
  imgsz=1280 \
  epochs=1 \
  fraction=0.10 \
  batch=6 \
  device=0 \
  workers=8 \
  amp=True \
  cache=False \
  optimizer=auto \
  mosaic=0.10 \
  mixup=0.0 \
  degrees=3.0 \
  translate=0.05 \
  scale=0.15 \
  shear=0.0 \
  perspective=0.0 \
  fliplr=0.5 \
  flipud=0.5 \
  hsv_h=0.0 \
  hsv_s=0.0 \
  hsv_v=0.12 \
  close_mosaic=0 \
  seed=2026 \
  plots=True \
  project="$ROOT/runs/rareos_smoke" \
  name=yolo26m_rareos_v1_frac010_e1 \
  exist_ok=True


echo
echo "========================================"
echo " GPU PRECHECK FINISHED"
echo "========================================"

echo
echo "Baseline-1 independent validation:"
echo "$ROOT/runs/baseline_eval/yolo26m_tiles1280_best_e73"

echo
echo "Baseline-2 smoke:"
echo "$ROOT/runs/rareos_smoke/yolo26m_rareos_v1_frac010_e1"

echo
echo "DO NOT start the 80-epoch run automatically."
echo "Inspect these results first."
