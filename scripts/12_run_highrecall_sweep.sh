#!/usr/bin/env bash

set -u
set -o pipefail

ROOT="/root/autodl-tmp/steel_defect"
cd "$ROOT" || exit 1

MODEL="runs/rareos/yolo26m_tiles1280_rareos_v1_e80_b6_seed2026/weights/best.pt"
EVAL="scripts/11_eval_highrecall_val.py"

mkdir -p \
  results/highrecall_val \
  logs

COMMON=(
  --model "$MODEL"
  --images datasets/yolo_split/images/val
  --labels datasets/yolo_split/labels/val
  --tile-size 1280
  --batch 6
  --max-det 1000
  --match-iou 0.50
  --device 0
  --half
)

run_case() {
    NAME="$1"
    LIMIT="$2"
    shift 2

    OUT="results/highrecall_val/${NAME}.csv"
    LOG="logs/val_${NAME}.log"

    echo
    echo "============================================================"
    echo "START: ${NAME}"
    echo "TIME : $(date '+%F %T')"
    echo "LIMIT: ${LIMIT}"
    echo "============================================================"

    START=$(date +%s)

    timeout "$LIMIT" \
      python "$EVAL" \
        "${COMMON[@]}" \
        "$@" \
        --output "$OUT" \
        2>&1 | tee "$LOG"

    CODE=${PIPESTATUS[0]}
    END=$(date +%s)
    ELAPSED=$((END - START))

    echo
    echo "------------------------------------------------------------"

    if [ "$CODE" -eq 0 ]; then
        echo "DONE: ${NAME}"
    elif [ "$CODE" -eq 124 ]; then
        echo "TIMEOUT: ${NAME}"
    else
        echo "FAILED: ${NAME} exit=${CODE}"
    fi

    echo "Elapsed: ${ELAPSED} sec"
    echo "Log: $LOG"
    echo "Output: $OUT"
    echo "------------------------------------------------------------"
}


# ============================================================
# A — Global NMS 0.80 -> 0.90
# 其他参数保持当前线上 93.06 配置
# ============================================================

run_case \
  "a_gnms090" \
  "20m" \
  --stride 1024 \
  --conf 0.0001 \
  --tile-iou 0.60 \
  --global-iou 0.90


# ============================================================
# B — Tile NMS 0.60 -> 0.80
# ============================================================

run_case \
  "b_tnms080" \
  "20m" \
  --stride 1024 \
  --conf 0.0001 \
  --tile-iou 0.80 \
  --global-iou 0.80


# ============================================================
# D — stride 1024 -> 896
# ============================================================

run_case \
  "d_stride896" \
  "25m" \
  --stride 896 \
  --conf 0.0001 \
  --tile-iou 0.60 \
  --global-iou 0.80


# ============================================================
# E — stride 1024 -> 768
# ============================================================

run_case \
  "e_stride768" \
  "30m" \
  --stride 768 \
  --conf 0.0001 \
  --tile-iou 0.60 \
  --global-iou 0.80


# ============================================================
# C — conf 1e-4 -> 1e-5
# 候选可能暴涨，所以最后测试
# ============================================================

run_case \
  "c_conf1e5" \
  "35m" \
  --stride 1024 \
  --conf 0.00001 \
  --tile-iou 0.60 \
  --global-iou 0.80


echo
echo "============================================================"
echo "ALL SWEEP JOBS FINISHED"
echo "TIME: $(date '+%F %T')"
echo "============================================================"

echo
echo "===== Overall result summary ====="

for LOG in \
    logs/val_v3_current.log \
    logs/val_a_gnms090.log \
    logs/val_b_tnms080.log \
    logs/val_d_stride896.log \
    logs/val_e_stride768.log \
    logs/val_c_conf1e5.log
do
    if [ -f "$LOG" ]; then
        echo
        echo "### $LOG"
        grep -E \
          "TP       :|FP       :|FN       :|Recall   :|Precision:|ScoreLike:" \
          "$LOG" || true
    fi
done

echo
echo "===== Finished ====="
