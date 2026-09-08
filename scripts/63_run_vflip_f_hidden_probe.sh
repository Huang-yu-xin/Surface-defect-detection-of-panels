#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/steel_defect
export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2
mkdir -p results/vflip_factorial/hidden_probe_f/logs
stage() {
  local name="$1"; shift
  local start end status
  start=$(date +%s); echo "START $name $(date -Is)"
  set +e; "$@"; status=$?; set -e
  end=$(date +%s); echo "END $name STATUS=$status runtime_seconds=$((end-start)) $(date -Is)"
  return "$status"
}
trap 'status=$?; echo "RUNNER_EXIT STATUS=$status $(date -Is)"; exit "$status"' EXIT
stage baseline env CUDA_VISIBLE_DEVICES='' python scripts/46_final_combo_test_global_complement_stream.py --variant u1e3_g060
stage geometry env CUDA_VISIBLE_DEVICES='' python scripts/59_audit_vflip_f_test.py geometry
stage inference_f env CUDA_VISIBLE_DEVICES=0 python scripts/58_cache_vflip_f_test.py
stage audit_f env CUDA_VISIBLE_DEVICES='' python scripts/59_audit_vflip_f_test.py audit
stage final_combo env CUDA_VISIBLE_DEVICES='' python scripts/60_final_combo_test_vflip_f_stream.py --device cpu
stage submission_audit env CUDA_VISIBLE_DEVICES='' python scripts/61_audit_report_vflip_f_hidden_probe.py
echo 'COMPLETE Test inference=F-only Hidden/Test GT accessed=NO competition submission=NO'
