#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/steel_defect
export PATH=/root/miniconda3/bin:$PATH
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2 CUDA_VISIBLE_DEVICES=0
mkdir -p results/offset_grid_halfstride/logs
stage(){ local n="$1";shift;local s e rc;s=$(date +%s);echo "START $n $(date -Is)";set +e;"$@";rc=$?;set -e;e=$(date +%s);echo "END $n STATUS=$rc runtime_seconds=$((e-s)) $(date -Is)";return "$rc";}
trap 'rc=$?; echo "RUNNER_EXIT STATUS=$rc $(date -Is)"; exit "$rc"' EXIT
if [ ! -f results/offset_grid_halfstride/exact_combo/baseline/PASS.json ];then stage baseline env CUDA_VISIBLE_DEVICES='' python scripts/64_reproduce_offset_grid_val_baseline.py;fi
stage geometry env CUDA_VISIBLE_DEVICES='' python scripts/66_test_offset_grid_geometry.py
for grid in x y xy;do stage "inference_$grid" python scripts/65_cache_offset_grid_val.py --grid "$grid";stage "audit_$grid" env CUDA_VISIBLE_DEVICES='' python scripts/65_cache_offset_grid_val.py --grid "$grid" --audit-only;done
stage candidate_audit env CUDA_VISIBLE_DEVICES='' python scripts/67_audit_offset_grid_candidates.py
stage exact_combo env CUDA_VISIBLE_DEVICES='' python scripts/68_eval_offset_grid_exact_combo.py
stage mechanism env CUDA_VISIBLE_DEVICES='' python scripts/69_audit_offset_grid_mechanism.py
stage independent_pre env CUDA_VISIBLE_DEVICES='' python scripts/70_verify_offset_grid_artifacts.py
stage report_draft env CUDA_VISIBLE_DEVICES='' python scripts/71_report_offset_grid.py
stage independent_final env CUDA_VISIBLE_DEVICES='' python scripts/70_verify_offset_grid_artifacts.py --final
stage report_final env CUDA_VISIBLE_DEVICES='' python scripts/71_report_offset_grid.py
echo 'COMPLETE Test/Hidden run=NO competition submission=NO'
