#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MALLOC_ARENA_MAX=2
export CUDA_VISIBLE_DEVICES=""
stage() {
  local name="$1"
  shift
  local start end status
  start=$(date +%s)
  echo "START $name $(date -Is)"
  set +e
  "$@"
  status=$?
  set -e
  end=$(date +%s)
  echo "END $name exit_status=$status runtime_seconds=$((end-start)) $(date -Is)"
  return "$status"
}
stage build python scripts/41_build_baseline_global_complement_val.py
stage exact_val python scripts/42_eval_baseline_global_exact_final_combo.py
stage report python scripts/43_report_baseline_global_exact_combo.py
