#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/steel_defect
export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export CUDA_VISIBLE_DEVICES=0
mkdir -p results/vflip_factorial/logs
stage() {
  local name="$1"; shift
  local start end status
  start=$(date +%s)
  echo "START $name $(date -Is)"
  set +e
  "$@"
  status=$?
  set -e
  end=$(date +%s)
  echo "END $name STATUS=$status runtime_seconds=$((end-start)) $(date -Is)"
  return "$status"
}
trap 'status=$?; echo "RUNNER_EXIT STATUS=$status $(date -Is)"; exit "$status"' EXIT
if [ ! -f results/vflip_factorial/exact_combo/baseline/PASS.json ]; then
  stage baseline env CUDA_VISIBLE_DEVICES='' python scripts/51_reproduce_vflip_val_baseline.py
fi
stage geometry env CUDA_VISIBLE_DEVICES='' python scripts/53_test_vflip_factorial_geometry.py
for view in direction grid full; do
  stage "inference_$view" python scripts/52_cache_vflip_factorial_val.py --view "$view"
  stage "audit_$view" env CUDA_VISIBLE_DEVICES='' python scripts/52_cache_vflip_factorial_val.py --view "$view" --audit-only
done
stage proposal_and_exact env CUDA_VISIBLE_DEVICES='' python scripts/54_analyze_vflip_factorial_val.py
stage report env CUDA_VISIBLE_DEVICES='' python scripts/55_report_vflip_factorial.py
echo 'COMPLETE Test/Hidden run=NO competition submission=NO'
