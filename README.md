# 钢板表面缺陷检测

基于 **YOLO26m** 的高分辨率钢板表面缺陷检测项目。

本项目面向约 `4096 × 3000` 的工业钢板图像，完成 9 类表面缺陷目标检测，并重点研究：

- 高分辨率滑窗切片
- 长尾类别重采样
- 长条缺陷检测
- High-Recall 高召回推理
- 跨切片 Global NMS
- 本地 Recall 模拟评测
- Remaining FN 根因分析
- Horizontal Flip TTA
- 跨切片长缺陷拼接
- 可复现的单变量消融实验

截至 **2026-08-10**，项目已完成：

```text
Baseline
→ RareOS v1
→ High-Recall Inference
→ Local Recall Sweep
→ Remaining FN Diagnostic
→ Horizontal Flip TTA
→ Zonglie Cross-Tile Stitching
→ Final Selective-TTA Combo
→ Online Leaderboard Validation
```

当前最好线上成绩：

```text
Score  = 98.09
Recall = 0.9809

TP = 975
FN = 19
```

相较此前 `95.67` 版本：

```text
Score: 95.67 → 98.09
TP:       951 → 975
FN:        43 → 19

Δ Score = +2.42
Δ TP    = +24
Δ FN    = -24
```

---

## 1. 任务简介

原始图像尺寸约为：

```text
4096 × 3000
```

原始训练集：

```text
3200 images
5889 bounding boxes
1074 negative / empty-annotation images
```

共包含 9 类钢板表面缺陷：

|   ID | 类别         |
| ---: | ------------ |
|    0 | jieba        |
|    1 | zonglie      |
|    2 | qilie        |
|    3 | jiaza        |
|    4 | yiwuyaru     |
|    5 | huashang     |
|    6 | mamianmakeng |
|    7 | yanghuatiepi |
|    8 | gunyin       |

数据存在明显类别不平衡，其中 `qilie` 和 `huashang` 属于重点关注的稀有类别。

此外，`zonglie`、`qilie`、`huashang` 具有明显的细长 / 长条目标特征，对切片边界、overlap 和后处理方式较敏感。

---

## 2. 整体流程

```text
VOC Annotations
        │
        ▼
Dataset Audit
        │
        ▼
VOC → YOLO
        │
        ▼
Grouped Train / Val Split
        │
        ▼
1280 × 1280 Tiling
        │
        ▼
YOLO26m Baseline
        │
        ├── RareOS v1
        │
        ├── High-Recall Inference
        │
        ├── Dense Tiling
        │
        ├── Global NMS Sweep
        │
        ├── Local Recall Simulator
        │
        ├── Remaining FN Diagnostic
        │
        ├── Horizontal Flip TTA
        │
        └── Zonglie Cross-Tile Stitching
                         │
                         ▼
                  Final Combo
                         │
                         ▼
                  Online Score 98.09
```

实验原则是：

> 先进行单变量短实验和本地验证，确认方向有效后，再投入完整训练或有限的线上提交次数。

---

## 3. 数据划分

Train / Val 在切片之前完成，并按照原图 / 生产组进行 grouped split，避免数据泄漏。

```text
Train: 2726 images
Val:    474 images

Train / Val image overlap: 0
Train / Val group overlap: 0
```

验证集在 Baseline、RareOS、High-Recall、FN Diagnostic、TTA 和 Cross-Tile Stitching 实验中保持固定。

---

## 4. 高分辨率切片

由于原始图像约为 `4096 × 3000`，直接缩放到常规 YOLO 输入尺寸可能损失小缺陷和细长缺陷信息，因此正式训练采用滑窗切片。

基础训练切片参数：

```text
tile size = 1280
stride    = 1024
overlap   = 256
```

正式切片数据规模：

```text
Train tiles = 6476
Val tiles   = 1131
```

针对 `zonglie`、`qilie`、`huashang` 等长条目标使用更宽松的边界保留策略，以降低切片截断造成的信息损失。

High-Recall 推理阶段进一步将 stride 收缩至：

```text
stride = 768
overlap = 512
```

以提高原图覆盖密度。

---

## 5. Baseline

正式 Baseline：

```text
Model      = YOLO26m
Image size = 1280
Batch      = 6
Epochs     = 80
Seed       = 2026
Best epoch = 73
```

Baseline 独立验证结果：

| Metric    | Value |
| --------- | ----: |
| Precision | 0.570 |
| Recall    | 0.494 |
| mAP50     | 0.516 |
| mAP50-95  | 0.297 |

主要问题：

```text
qilie 数据极少
huashang 长尾 + 长条
zonglie 对切片边界敏感
```

---

## 6. RareOS v1

RareOS v1 只改变训练数据曝光频率：

```text
qilie    ×4
huashang ×2
```

其他核心训练参数、随机种子和验证集保持不变。

训练 tile 数：

```text
Baseline  = 6476
RareOS v1 = 6778
```

RareOS v1 `best.pt` 独立验证：

| Model     | Precision |    Recall |     mAP50 |  mAP50-95 |
| --------- | --------: | --------: | --------: | --------: |
| Baseline  |     0.570 |     0.494 |     0.516 |     0.297 |
| RareOS v1 |     0.559 | **0.541** | **0.528** | **0.304** |

因此：

```text
Recall:
0.494 → 0.541

Δ Recall ≈ +0.047
```

同时 mAP50 和 mAP50-95 也有小幅提升。

当前主模型：

```text
runs/rareos/yolo26m_tiles1280_rareos_v1_e80_b6_seed2026/weights/best.pt
```

---

## 7. High-Recall 推理

线上评测结果显示，当前比赛分数与 Recall 高度一致：

```text
Score ≈ Recall × 100
```

因此，在保持预测框来自模型真实推理结果的前提下，对 confidence threshold、global NMS 和 tile overlap / stride 进行了系统实验。

### 排行榜成绩演化

| 方案                              |     Score |     Recall |      TP |     FN |            FP |
| --------------------------------- | --------: | ---------: | ------: | -----: | ------------: |
| Baseline normal                   |     82.50 |     0.8249 |     820 |    174 |         5,643 |
| RareOS normal                     |     82.50 |     0.8249 |     820 |    174 |         4,808 |
| HighRecall v1                     |     88.03 |     0.8803 |     875 |    119 |        18,829 |
| HighRecall v2                     |     89.54 |     0.8954 |     890 |    104 |        31,858 |
| HighRecall v3                     |     93.06 |     0.9306 |     925 |     69 |       176,156 |
| HighRecall + Dense Tiling         |     95.67 |     0.9567 |     951 |     43 |     1,757,850 |
| **Selective HFlip + Zonglie Stitch** | **98.09** | **0.9809** | **975** | **19** | **2,402,834** |

从初始 Baseline 到当前最好结果：

```text
Score:
82.50 → 98.09

Δ = +15.59
```

漏检数量：

```text
FN:
174 → 19
```

共额外找回 155 个真实目标。

---

## 8. High-Recall 基础配置

最终方案仍以以下 High-Recall 参数作为基础：

```text
Model:
RareOS v1 best.pt

tile_size   = 1280
stride      = 768

conf        = 1e-5

tile_iou    = 0.60
global_iou  = 0.90

max_det     = 1000
batch       = 6
FP16        = True
```

此前仅使用这一基础配置的线上结果：

```text
Score      = 95.67
Recall     = 0.9567
Precision  = 0.0005
F1         = 0.0011
mAP@0.5    = 0.4554

TP = 951
FP = 1,757,850
FN = 43
```

> 注意：这一配置针对当前比赛 Recall 导向的评分机制进行优化，并不代表适用于真实工业部署场景的最佳 Precision / Recall 工作点。

---

## 9. 本地 Recall 模拟器

为了减少有限的排行榜提交次数，项目增加了：

```text
scripts/11_eval_highrecall_val.py
```

该脚本在固定验证集上模拟：

```text
Tiled Inference
      ↓
Map Boxes Back to Original Image
      ↓
Class-aware Global NMS
      ↓
IoU >= 0.5
同类别一对一匹配
      ↓
TP / FP / FN / Recall
```

High-Recall 参数实验结果：

| Config                                |      TP |     FN |       Recall | ScoreLike |
| ------------------------------------- | ------: | -----: | -----------: | --------: |
| V3                                    |     763 |     82 |     0.902959 |     90.30 |
| gNMS=.90                              |     766 |     79 |     0.906509 |     90.65 |
| tileNMS=.80                           |     763 |     82 |     0.902959 |     90.30 |
| stride=896                            |     765 |     80 |     0.905325 |     90.53 |
| stride=768                            |     775 |     70 |     0.917160 |     91.72 |
| conf=1e-5                             |     783 |     62 |     0.926627 |     92.66 |
| conf=1e-5 + stride=768                |     795 |     50 |     0.940828 |     94.08 |
| **conf=1e-5 + stride=768 + gNMS=.90** | **797** | **48** | **0.943195** | **94.32** |

本地排序与线上结果方向一致，因此后续参数筛选继续优先通过固定 Val 的 Recall 模拟完成。

---

## 10. Remaining FN Diagnostic

在 High-Recall 基础配置达到：

```text
TP = 797
FN = 48
Recall = 0.943195
```

之后，不再继续盲目降低 confidence，而是通过：

```text
scripts/14_fn_diagnostic.py
```

缓存每张 Val 图像在 Global NMS 前的候选框，并对剩余 FN 做根因分析。

### 48 个 FN 的失败类型

| Failure Type                          | Count |
| ------------------------------------- | ----: |
| localization_or_tile_fragmentation    |    22 |
| localization_failure                  |    14 |
| class_confusion                       |    12 |

即：

```text
Localization / Tile Fragmentation = 36 / 48 = 75%
Class Confusion                   = 12 / 48 = 25%
```

没有观察到主要由以下因素导致的 FN：

```text
global_nms_suppression
matching_competition
no_same_class_candidate
```

因此，继续降低 conf 或单纯调整 Global NMS 已不再是主要突破方向。

### FN 类别分布

| Class         | TP  | FN | Recall |
| ------------- | --: | -: | -----: |
| jieba         | 170 |  7 | 0.9605 |
| zonglie       |  49 | 16 | 0.7538 |
| qilie         |   4 |  3 | 0.5714 |
| jiaza         |  32 |  5 | 0.8649 |
| yiwuyaru      |  89 |  3 | 0.9674 |
| huashang      |  19 |  5 | 0.7917 |
| mamianmakeng  | 332 |  1 | 0.9970 |
| yanghuatiepi  |  65 |  7 | 0.9028 |
| gunyin        |  37 |  1 | 0.9737 |

其中 `zonglie` 成为最大的单一 FN 来源。

---

## 11. Horizontal Flip TTA

使用：

```text
scripts/15_cache_hflip_tta.py
scripts/16_eval_hflip_tta_cache.py
```

对固定 Val 集进行 Horizontal Flip TTA：

```text
Original
+
Horizontal Flip
+
Map HFlip Boxes Back to Original Coordinates
+
Unified Class-aware Global NMS
```

完整 HFlip union 结果：

```text
Original:
TP       = 797
FN       = 48
Recall   = 0.943195
ScoreLike= 94.32

Original + HFlip:
TP       = 809
FN       = 36
Recall   = 0.957396
ScoreLike= 95.74
```

即：

```text
Rescued FN = 12
Regressed TP = 0
```

HFlip 救回类别：

| Class         | Rescued FN |
| ------------- | ---------: |
| jieba         |          3 |
| qilie         |          2 |
| jiaza         |          1 |
| yiwuyaru      |          2 |
| yanghuatiepi  |          4 |

而：

```text
zonglie       = 0
huashang      = 0
mamianmakeng  = 0
gunyin        = 0
```

因此最终方案不再加入所有 HFlip proposals，而只对以下类别保留 HFlip：

```text
jieba
qilie
jiaza
yiwuyaru
yanghuatiepi
```

这样保留有效增益，同时减少无贡献的额外候选。

---

## 12. Zonglie 跨切片拼接

FN Diagnostic 发现：

```text
zonglie FN = 16
```

并且这 16 个 FN 全部属于：

```text
localization_or_tile_fragmentation
```

其典型 GT 几何形态：

```text
width  ≈ 53 ~ 129 px
height ≈ 1937 ~ 2999 px
elongation ≈ 23 ~ 53
```

绝大多数 `zonglie` 高度远大于单个：

```text
tile_size = 1280
```

因此单个 tile 产生的局部框即使定位正确，也很难与约 3000 px 高的完整 GT 达到：

```text
IoU >= 0.5
```

这说明问题的根源不是“模型完全没有检测到纵裂”，而是：

```text
同一纵裂
→ 在多个相邻 tile 中形成多个局部预测
→ Global NMS 不会把这些纵向片段合成长框
→ 与完整 GT 的 IoU 不足
```

### Cross-Tile Stitching

使用：

```text
scripts/17_eval_zonglie_cross_tile_stitch_gpu.py
```

将相邻 tile row 中：

- 类别均为 `zonglie`
- 横向位置接近
- x 区域具有重叠
- y 方向连续 / 邻近
- 具有明显纵向长条形态

的真实模型候选框进行跨切片拼接。

最终选用：

```text
min_aspect = 5
x_tol      = 64
max_y_gap  = 64
```

验证结果：

```text
Baseline zonglie:
TP = 49
FN = 16

After Cross-Tile Stitch:
TP = 64
FN = 1

Rescued   = 15
Regressed = 0
```

即：

```text
15 / 16 zonglie FN 被直接救回
```

### 参数稳定性

共测试 12 组组合：

```text
min_aspect ∈ {3, 5}
x_tol      ∈ {32, 64, 96}
max_y_gap  ∈ {64, 256}
```

所有 12 组均得到：

```text
TP = 64
FN = 1
Rescued = 15
Regressed = 0
```

说明该收益并非单一参数点过拟合，而是一个较宽的稳定平台。

---

## 13. Final Combo

最终验证脚本：

```text
scripts/18_final_combo_from_cache.py
```

最终组合：

```text
RareOS v1
+
High-Recall Base Inference
+
Selective Horizontal Flip TTA
+
Zonglie Cross-Tile Stitching
```

其中：

### Base

```text
tile_size   = 1280
stride      = 768
conf        = 1e-5
tile_iou    = 0.60
global_iou  = 0.90
max_det     = 1000
FP16        = True
```

### Selective HFlip

只加入：

```text
jieba
qilie
jiaza
yiwuyaru
yanghuatiepi
```

### Zonglie Stitch

```text
min_aspect       = 5
x_tol            = 64
max_y_gap        = 64
min_merged_height= 1300
```

### Final Combo Val

```text
TP        = 824
FP        = 1,631,208
FN        = 21
Recall    = 0.975148
ScoreLike = 97.51
```

相较 High-Recall Base：

```text
TP: 797 → 824
FN:  48 → 21

Δ TP = +27
Δ FN = -27
```

其中：

```text
HFlip rescue           = +12 TP
Zonglie stitching      = +15 TP
```

两类增益基本互补。

---

## 14. 当前最好线上成绩：98.09

2026-08-10，Final Combo 正式提交结果：

```text
Score      = 98.09
Recall     = 0.9809
Precision  = 0.0004
F1         = 0.0008
mAP@0.5    = 0.4550

TP = 975
FP = 2,402,834
FN = 19
```

相较此前 High-Recall 最佳版本：

```text
Previous:
Score = 95.67
TP    = 951
FN    = 43

Current:
Score = 98.09
TP    = 975
FN    = 19
```

提升：

```text
Δ Score = +2.42
Δ TP    = +24
Δ FN    = -24
```

从最初 Baseline 到当前版本：

```text
Score:
82.50 → 98.09

TP:
820 → 975

FN:
174 → 19
```

该结果验证了：

```text
RareOS
+
High-Recall Inference
+
FN Root-Cause Analysis
+
Selective TTA
+
Class-Specific Cross-Tile Post-processing
```

这一整套实验路线的有效性。

---

## 15. 当前主要结论

### 训练侧

RareOS v1 有效：

```text
Validation Recall
0.494 → 0.541
```

说明定向长尾重采样可以提高召回能力。

### High-Recall 推理侧

此前最有效的基础策略：

```text
1. 降低 confidence threshold
2. 增加 tile overlap
3. 放宽 global NMS
```

但在：

```text
conf = 1e-5
stride = 768
global_iou = 0.90
```

之后，继续降低 conf 已出现明显边际收益递减。

### FN 根因分析比盲目 sweep 更有效

剩余 48 FN 中：

```text
75% 与 localization / tile fragmentation 有关
```

因此后续突破来自结构化问题分析，而不是继续无方向扩大参数搜索。

### TTA 具有类别选择性

Horizontal Flip 并非对全部类别都有帮助。

有效类别：

```text
jieba
qilie
jiaza
yiwuyaru
yanghuatiepi
```

对 `zonglie` 没有救回，因此最终采用 selective TTA。

### 长缺陷需要专门的跨切片后处理

`zonglie` 的主要问题不是低置信度，而是：

```text
GT 长度 > tile_size
```

Cross-Tile Stitching 将：

```text
zonglie FN:
16 → 1
```

说明对于超长目标，单纯提高 overlap 并不能完全替代结构化跨切片拼接。

---

## 16. 下一阶段计划

当前线上：

```text
Score = 98.09
FN    = 19
```

已经从“继续扩大整体召回”进入“针对最后少量 FN 做精细优化”的阶段。

固定 Val Final Combo 下剩余 21 个 FN 主要集中于：

```text
huashang       5
jieba          4
jiaza          4
yanghuatiepi   3
zonglie        1
qilie          1
yiwuyaru       1
mamianmakeng   1
gunyin         1
```

下一阶段优先研究：

```text
Remaining 21 FN
      │
      ├── Small-object Localization
      │     └── huashang
      │
      ├── Box Fusion / Box Refinement
      │
      ├── Class-specific Proposal Union
      │
      ├── Selective Additional TTA
      │
      └── Model Ensemble
```

阶段目标：

```text
98.09
→ 98.5+
→ 尝试进一步逼近 99
```

线上提交次数有限，因此新方案仍应优先在固定 Val 和缓存候选上验证。

---

## 17. 项目结构

```text
.
├── configs/
│   ├── steel_original.yaml
│   ├── steel_tiles_1280.yaml
│   └── steel_tiles_1280_rareos_v1.yaml
│
├── scripts/
│   ├── 00_visualize_voc.py
│   ├── 01_make_bbox_crops.py
│   ├── 02_audit_dataset.py
│   ├── 03_voc_to_yolo.py
│   ├── 04_make_grouped_split.py
│   ├── 05_make_tile_trial.py
│   ├── 06_make_full_tiles.py
│   ├── 07_analyze_baseline.py
│   ├── 08_make_rare_oversample_dataset.py
│   ├── 09_run_baseline2_gpu.sh
│   ├── 10_predict_test_submission.py
│   ├── 11_eval_highrecall_val.py
│   ├── 12_run_highrecall_sweep.sh
│   ├── 13_record_experiment_results.sh
│   ├── 14_fn_diagnostic.py
│   ├── 15_cache_hflip_tta.py
│   ├── 16_eval_hflip_tta_cache.py
│   ├── 17_eval_zonglie_cross_tile_stitch_gpu.py
│   └── 18_final_combo_from_cache.py
│
├── docs/
│   └── experiment_log_20260808.md
│
├── splits/
├── .gitignore
└── README.md
```

---

## 18. 关键脚本

| Script                                          | 功能                                      |
| ----------------------------------------------- | ----------------------------------------- |
| `00_visualize_voc.py`                           | VOC 标注可视化                            |
| `01_make_bbox_crops.py`                         | 生成缺陷 bbox 裁剪                        |
| `02_audit_dataset.py`                           | 数据集审计                                |
| `03_voc_to_yolo.py`                             | VOC → YOLO                                |
| `04_make_grouped_split.py`                      | 防泄漏 Train / Val 分组                   |
| `05_make_tile_trial.py`                         | 小规模切片实验                            |
| `06_make_full_tiles.py`                         | 构建正式 tile 数据集                      |
| `07_analyze_baseline.py`                        | Baseline 分析                             |
| `08_make_rare_oversample_dataset.py`            | RareOS 数据构建                           |
| `09_run_baseline2_gpu.sh`                       | RareOS 训练                               |
| `10_predict_test_submission.py`                 | 官方 test tiled inference                 |
| `11_eval_highrecall_val.py`                     | 本地 Recall 模拟评测                      |
| `12_run_highrecall_sweep.sh`                    | High-Recall 参数扫描                      |
| `13_record_experiment_results.sh`               | 自动生成阶段实验记录                      |
| `14_fn_diagnostic.py`                           | Remaining FN 根因诊断与候选缓存           |
| `15_cache_hflip_tta.py`                         | Horizontal Flip TTA 候选缓存              |
| `16_eval_hflip_tta_cache.py`                    | Original + HFlip 离线 Val 评估            |
| `17_eval_zonglie_cross_tile_stitch_gpu.py`      | Zonglie 跨切片拼接参数验证                |
| `18_final_combo_from_cache.py`                  | Final Combo Val / Test 缓存后处理与提交生成 |

已有详细实验记录：

```text
docs/experiment_log_20260808.md
```

---

## 19. 实验原则

本项目尽量遵循：

1. **单变量消融**
2. **固定随机种子**
3. **固定 Train / Val**
4. **避免切片级数据泄漏**
5. **短实验筛选后再完整训练**
6. **线上提交前优先本地验证**
7. **区分训练收益和推理收益**
8. **保存完整实验配置和日志**
9. **先分析 FN 根因，再选择下一项实验**
10. **优先复用 inference cache，减少重复 GPU 推理**
11. **针对不同类别采用不同后处理策略**

同时明确区分：

```text
dataset problem
tiling problem
training problem
inference problem
post-processing problem
```

---

## 20. 数据、缓存与权重

由于数据、缓存和模型文件体积较大，本仓库不保存：

```text
raw/
datasets/
runs/
logs/
results/
submissions/
records/

*.pt
*.pth
*.onnx
*.engine
```

尤其不提交：

- 原始训练 / 测试数据
- 模型权重
- 百万级候选缓存
- 大型预测 JSON
- submission.zip

仓库主要用于保存：

- 数据处理代码
- 数据划分
- 实验配置
- 训练脚本
- 推理代码
- Recall 评测工具
- FN Diagnostic 工具
- TTA / Cross-Tile 后处理代码
- 实验记录
- 可复现研究流程

---

## 21. 当前状态

```text
Model:
YOLO26m + RareOS v1

RareOS Independent Validation:
Recall     = 0.541
mAP50      = 0.528
mAP50-95   = 0.304

High-Recall Base Val:
TP         = 797
FN         = 48
Recall     = 0.943195

Final Combo Val:
TP         = 824
FN         = 21
Recall     = 0.975148
ScoreLike  = 97.51

Best Leaderboard:
Score      = 98.09
Recall     = 0.9809
Precision  = 0.0004
mAP@0.5    = 0.4550

TP         = 975
FP         = 2,402,834
FN         = 19
```

当前主要路线：

```text
RareOS v1
    ↓
High-Recall Base
    ↓
Remaining FN Diagnostic
    ↓
Selective HFlip TTA
    ↓
Zonglie Cross-Tile Stitching
    ↓
Score 98.09
```

项目仍在持续开发中。

下一阶段重点：

```text
Remaining FN Analysis
Small-object Localization
Box Fusion / Refinement
Class-specific Proposal Union
Selective TTA
Model Ensemble
98.5+ / 99 Exploration
```
