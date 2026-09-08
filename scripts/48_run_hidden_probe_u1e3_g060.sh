#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/steel_defect
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 MALLOC_ARENA_MAX=2
export CUDA_VISIBLE_DEVICES=""
ROOT="results/baseline_complementarity/global_exact_hidden_probe"
mkdir -p "$ROOT/logs"
exec 9>"$ROOT/logs/runner.lock"
flock -n 9 || { echo "Another hidden probe runner is active"; exit 1; }

stage() {
  local name="$1"
  shift
  local start end status
  start=$(date +%s)
  echo "START $name $(date -Is)"
  set +e
  "$@" 2>&1 | tee "$ROOT/logs/$name.log"
  status=${PIPESTATUS[0]}
  set -e
  end=$(date +%s)
  echo "END $name STATUS=$status RUNTIME=$((end-start)) $(date -Is)"
  python - "$ROOT/logs/stage_metrics.json" "$name" "$start" "$end" "$status" <<'PY'
import json,sys
from pathlib import Path
path=Path(sys.argv[1])
data=json.loads(path.read_text()) if path.exists() else {}
data[sys.argv[2]]={"start_unix":int(sys.argv[3]),"end_unix":int(sys.argv[4]),
                  "runtime_seconds":int(sys.argv[4])-int(sys.argv[3]),"exit_code":int(sys.argv[5])}
tmp=path.with_suffix(".json.tmp")
tmp.write_text(json.dumps(data,indent=2)+"\n")
tmp.replace(path)
PY
  return "$status"
}

stage cache_audit python scripts/45_build_baseline_global_complement_test.py --mode audit
stage baseline_reproduction python scripts/46_final_combo_test_global_complement_stream.py --variant baseline
stage build_u1e3_g060 python scripts/45_build_baseline_global_complement_test.py --mode build
stage final_combo_u1e3_g060 python scripts/46_final_combo_test_global_complement_stream.py --variant u1e3_g060
stage submission_audit python scripts/47_audit_hidden_probe_submission.py
echo "ALL STAGES COMPLETE $(date -Is)"
