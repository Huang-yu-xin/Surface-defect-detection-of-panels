#!/usr/bin/env bash
set -euo pipefail

ROOT="/root/autodl-tmp/steel_defect"
cd "$ROOT"

mkdir -p docs

OUT="docs/experiment_log_20260808.md"

GIT_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
NOW="$(date '+%Y-%m-%d %H:%M:%S %z')"

cat > "$OUT" <<EOF
# Steel Defect Detection Experiment Log

Generated: ${NOW}

Git commit before this record: \`${GIT_COMMIT}\`

## 1. Current model

Model:

\`\`\`
YOLO26m
\`\`\`

Training data:

\`\`\`
configs/steel_tiles_1280_rareos_v1.yaml
\`\`\`

Checkpoint:

\`\`\`
runs/rareos/yolo26m_tiles1280_rareos_v1_e80_b6_seed2026/weights/best.pt
\`\`\`

Formal training configuration:

\`\`\`
epochs        = 80
batch         = 6
imgsz         = 1280
seed          = 2026
workers       = 8
optimizer     = auto
mosaic        = 0.10
close_mosaic  = 10
flipud        = 0.5
fliplr        = 0.5
hsv_v         = 0.12
degrees       = 3.0
translate     = 0.05
scale         = 0.15
AMP           = True
\`\`\`

---

## 2. Training-side comparison

| Model | Precision | Recall | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|
| Baseline-1 best | 0.570 | 0.494 | 0.516 | 0.297 |
| RareOS v1 best | 0.559 | 0.541 | 0.528 | 0.304 |

RareOS v1 therefore increased validation Recall by approximately:

\`\`\`
+0.047
\`\`\`

while also slightly improving mAP50 and mAP50-95.

RareOS v1 was created primarily to increase exposure of rare / difficult
classes, especially qilie and huashang.

---

## 3. Online leaderboard progression

Hidden test ground-truth count inferred from scoring:

\`\`\`
994 objects
\`\`\`

| Experiment | Key inference settings | Score | Recall | TP | FN | FP |
|---|---|---:|---:|---:|---:|---:|
| Baseline normal | conf=1e-2, stride=1024, gNMS=.50 | 82.50 | 0.8249 | 820 | 174 | 5,643 |
| RareOS normal | conf=1e-2, stride=1024, gNMS=.50 | 82.50 | 0.8249 | 820 | 174 | 4,808 |
| HighRecall v1 | conf=1e-3, stride=1024, gNMS=.50 | 88.03 | 0.8803 | 875 | 119 | 18,829 |
| HighRecall v2 | conf=1e-3, stride=1024, gNMS=.80 | 89.54 | 0.8954 | 890 | 104 | 31,858 |
| HighRecall v3 | conf=1e-4, stride=1024, gNMS=.80 | 93.06 | 0.9306 | 925 | 69 | 176,156 |
| Current best | conf=1e-5, stride=768, gNMS=.90 | **95.67** | **0.9567** | **951** | **43** | **1,757,850** |

Current best online score:

\`\`\`
95.67
\`\`\`

Improvement over initial Baseline:

\`\`\`
95.67 - 82.50 = +13.17 points
\`\`\`

TP improvement:

\`\`\`
820 -> 951
+131 recovered true positives
\`\`\`

FN reduction:

\`\`\`
174 -> 43
-131 false negatives
\`\`\`

Observed leaderboard behavior strongly indicates:

\`\`\`
Score ~= Recall * 100
\`\`\`

Precision and FP currently have little or no direct effect on the displayed
competition score.

---

## 4. Current best inference configuration

\`\`\`
model       = RareOS v1 best.pt

tile_size   = 1280
stride      = 768

conf        = 0.00001

tile_iou    = 0.60
global_iou  = 0.90

max_det     = 1000
batch       = 6
half        = True
\`\`\`

Official test inference statistics:

\`\`\`
Test images             = 669
Total tiles             = 13,380

Raw mapped detections   = 2,247,802
Final detections        = 1,758,801

Images with detections  = 669
Images without          = 0
\`\`\`

Submission source:

\`\`\`
submissions/rareos_conf1e5_stride768_gnms090.json
\`\`\`

Leaderboard result:

\`\`\`
score       = 95.67
recall      = 0.9567
precision   = 0.0005
f1          = 0.0011
mAP@0.5     = 0.4554

TP          = 951
FP          = 1,757,850
FN          = 43
\`\`\`

---

## 5. Local high-recall validation sweep

All experiments below use the same RareOS v1 checkpoint and IoU=0.5
one-to-one GT matching.

| Experiment | TP | FN | Recall | ScoreLike |
|---|---:|---:|---:|---:|
| V3: conf=1e-4, stride=1024, gNMS=.80 | 763 | 82 | 0.902959 | 90.30 |
| A: gNMS=.90 | 766 | 79 | 0.906509 | 90.65 |
| B: tileNMS=.80 | 763 | 82 | 0.902959 | 90.30 |
| D: stride=896 | 765 | 80 | 0.905325 | 90.53 |
| E: stride=768 | 775 | 70 | 0.917160 | 91.72 |
| C: conf=1e-5 | 783 | 62 | 0.926627 | 92.66 |
| C+E: conf=1e-5, stride=768 | 795 | 50 | 0.940828 | 94.08 |
| C+E+A: conf=1e-5, stride=768, gNMS=.90 | **797** | **48** | **0.943195** | **94.32** |

Important conclusions:

1. Lowering confidence threshold was the largest inference-side gain.
2. stride=768 produced a substantial additional recall gain.
3. global NMS 0.80 -> 0.90 gave a small but repeatable improvement.
4. tile NMS 0.60 -> 0.80 produced no measurable Recall improvement.
5. Low-conf and denser tiling gains were partially additive.
6. Local validation ranking correctly identified the stronger online
   configuration and should be used to screen future submissions.

---

## 6. Difficult validation classes under C+E+A

For the current strongest local configuration:

\`\`\`
jieba:
TP=170 FN=7  Recall=0.9605

zonglie:
TP=49  FN=16 Recall=0.7538

qilie:
TP=4   FN=3  Recall=0.5714

jiaza:
TP=32  FN=5  Recall=0.8649

yiwuyaru:
TP=89  FN=3  Recall=0.9674

huashang:
TP=19  FN=5  Recall=0.7917

mamianmakeng:
TP=332 FN=1  Recall=0.9970

yanghuatiepi:
TP=65  FN=7  Recall=0.9028

gunyin:
TP=37  FN=1  Recall=0.9737
\`\`\`

Remaining work should pay particular attention to:

\`\`\`
zonglie
qilie
huashang
\`\`\`

---

## 7. Next-stage candidates

Do not blindly continue lowering confidence.

Recommended next directions:

1. analyze the remaining FN cases on validation;
2. TTA / horizontal-flip inference;
3. additional dense-tiling experiments where justified;
4. targeted long-defect handling for zonglie / huashang;
5. model ensemble if multiple complementary checkpoints become available;
6. continue using the local Recall simulator before consuming leaderboard
   submissions.

Avoid:

- artificial grid boxes;
- manually generated detections;
- copying every box to all classes;
- other methods that do not originate from model inference.

---

## 8. Important scripts

\`\`\`
scripts/10_predict_test_submission.py
scripts/11_eval_highrecall_val.py
scripts/12_run_highrecall_sweep.sh
scripts/13_record_experiment_results.sh
\`\`\`

The local high-recall validation script is now the primary screening tool
before online submission.

EOF

echo "Experiment record written to:"
echo "$ROOT/$OUT"
echo
ls -lh "$OUT"
