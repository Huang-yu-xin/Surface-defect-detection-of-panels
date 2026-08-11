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
- Cross-View Box Fusion
- Class-Confusion Correction
- Baseline / RareOS 跨模型互补性审计
- Hidden-Test 泛化审计
- 大规模 submission 流式写盘 / OOM 工程优化
- 可复现的单变量消融实验

截至 **2026-08-12**，项目已完成：

```text
Baseline
→ RareOS v1
→ High-Recall Inference
→ Local Recall Sweep
→ Remaining FN Diagnostic
→ Horizontal Flip TTA
→ Zonglie Cross-Tile Stitching
→ Final Selective-TTA Combo
→ Online 98.09
→ Final Combo Remaining-FN Diagnostic
→ Cross-View Fusion / Targeted Fusion Pruning
→ Fusion Hidden-Test Negative
→ Class-Confusion Duplication / Rank-Score Robustness
→ Class-Confusion Hidden-Test Negative
→ Old Baseline Test Complementarity Audit
→ 68-Box Baseline Unique Probe
→ Baseline Unique Hidden-Test Negative
→ Baseline High-Recall Val Cache
→ Proposal-Level Complementarity Audit
→ Huashang Gate Robustness Sweep
→ Val 824/21 → 829/16
→ Baseline High-Recall Test Cache
→ Streaming Final-Combo Submission
→ Baseline High-Recall Hidden-Test Negative
```

当前最好线上成绩仍为：

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

2026-08-11 至 2026-08-12，在固定 Val 上继续进行 Final 21 FN 精细优化，并进行了四次 Hidden-Test 机制验证，均未新增 TP：

| Probe                                   |         Val / 本地现象 | Test 最终新增框 |      Hidden TP/FN |               Hidden FP |
| --------------------------------------- | ---------------------: | --------------: | ----------------: | ----------------------: |
| Targeted Cross-View Fusion              |      `824/21 → 828/17` |         +39,676 | `975/19 → 975/19` | `2,402,834 → 2,442,510` |
| Class-Confusion `mid_50_20`             |      `824/21 → 826/19` |         +22,047 | `975/19 → 975/19` | `2,402,834 → 2,424,881` |
| Baseline Unique `score>=0.10`           | Test-only 独立模型探针 |             +68 | `975/19 → 975/19` | `2,402,834 → 2,402,902` |
| Baseline High-Recall `huashang g065`    |      `824/21 → 829/16` |         +33,172 | `975/19 → 975/19` | `2,402,834 → 2,436,006` |

四次实验的 FP 增量都与最终新增框数量 **精确一致**，即新增框全部成为 FP。当前正式最好线上成绩因此仍为 `98.09`。

最新 Baseline High-Recall 实验尤其说明：即使引入不同训练模型的低置信 proposal source，并在固定 Val 上形成 `+5 TP` 的邻域稳定平台，也仍可能完全不覆盖 Hidden-Test 剩余 `19` FN。因此后续不再把“Val rescue 数量 + 参数稳定性”作为唯一的 Hidden-Test GO / NO-GO 标准，还必须检查 rescue 是否跨图片、跨类别、跨 failure cluster，并优先转向新的 inference view、tile geometry 或更强模型多样性。

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
                         │
                         ▼
            Final 21 FN Fine-Grained Audit
                         │
                         ▼
               Targeted Box Fusion
                         │
             Val: 824/21 → 828/17
                         │
                         ▼
              Hidden-Test Validation
                         │
                         ▼
           Online TP/FN unchanged
                         │
                         ▼
       Stop Fusion Overfitting / Change Mechanism
```

实验原则是：

> 先进行单变量短实验和本地验证，确认方向有效后，再投入完整训练或有限的线上提交次数。

进入最终少量 FN 阶段后，进一步增加：

> 固定 Val 上的增益必须经过隐藏测试泛化验证；不能把针对少数 Val FN 调出的规则直接视为可泛化结论。

---

## 3. 数据划分

Train / Val 在切片之前完成，并按照原图 / 生产组进行 grouped split，避免数据泄漏。

```text
Train: 2726 images
Val:    474 images

Train / Val image overlap: 0
Train / Val group overlap: 0
```

验证集在 Baseline、RareOS、High-Recall、FN Diagnostic、TTA、Cross-Tile Stitching 和 Cross-View Fusion 实验中保持固定。

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
stride  = 768
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

| 方案                                 |     Score |     Recall |      TP |     FN |            FP |
| ------------------------------------ | --------: | ---------: | ------: | -----: | ------------: |
| Baseline normal                      |     82.50 |     0.8249 |     820 |    174 |         5,643 |
| RareOS normal                        |     82.50 |     0.8249 |     820 |    174 |         4,808 |
| HighRecall v1                        |     88.03 |     0.8803 |     875 |    119 |        18,829 |
| HighRecall v2                        |     89.54 |     0.8954 |     890 |    104 |        31,858 |
| HighRecall v3                        |     93.06 |     0.9306 |     925 |     69 |       176,156 |
| HighRecall + Dense Tiling            |     95.67 |     0.9567 |     951 |     43 |     1,757,850 |
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

本地排序在 High-Recall 阶段多次与线上结果方向一致，因此参数筛选优先通过固定 Val 的 Recall 模拟完成。

但 2026-08-11 的 Box Fusion 负实验表明：**进入最后少量 FN 阶段后，本地小幅增益不再能自动视为隐藏测试可泛化增益。**

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

| Failure Type                       | Count |
| ---------------------------------- | ----: |
| localization_or_tile_fragmentation |    22 |
| localization_failure               |    14 |
| class_confusion                    |    12 |

即：

```text
Localization / Tile Fragmentation = 36 / 48 = 75%
Class Confusion                    = 12 / 48 = 25%
```

没有观察到主要由以下因素导致的 FN：

```text
global_nms_suppression
matching_competition
no_same_class_candidate
```

因此，继续降低 conf 或单纯调整 Global NMS 已不再是主要突破方向。

### FN 类别分布

| Class        |   TP |   FN | Recall |
| ------------ | ---: | ---: | -----: |
| jieba        |  170 |    7 | 0.9605 |
| zonglie      |   49 |   16 | 0.7538 |
| qilie        |    4 |    3 | 0.5714 |
| jiaza        |   32 |    5 | 0.8649 |
| yiwuyaru     |   89 |    3 | 0.9674 |
| huashang     |   19 |    5 | 0.7917 |
| mamianmakeng |  332 |    1 | 0.9970 |
| yanghuatiepi |   65 |    7 | 0.9028 |
| gunyin       |   37 |    1 | 0.9737 |

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
TP        = 797
FN        = 48
Recall    = 0.943195
ScoreLike = 94.32

Original + HFlip:
TP        = 809
FN        = 36
Recall    = 0.957396
ScoreLike = 95.74
```

即：

```text
Rescued FN  = 12
Regressed TP = 0
```

HFlip 救回类别：

| Class        | Rescued FN |
| ------------ | ---------: |
| jieba        |          3 |
| qilie        |          2 |
| jiaza        |          1 |
| yiwuyaru     |          2 |
| yanghuatiepi |          4 |

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
width      ≈ 53 ~ 129 px
height     ≈ 1937 ~ 2999 px
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
min_aspect        = 5
x_tol             = 64
max_y_gap         = 64
min_merged_height = 1300
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
HFlip rescue      = +12 TP
Zonglie stitching = +15 TP
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

## 15. Targeted Cross-View Fusion：本地正结果与线上负结果

2026-08-11，在无 GPU 的 CPU-only 模式下继续复用现有 Original / HFlip cache，对 Final Combo 剩余 21 FN 做更精细的诊断。

新增：

```text
scripts/19_final_combo_fn_diagnostic.py
```

首先精确复现 Final Combo：

```text
TP        = 824
FP        = 1,631,208
FN        = 21
Recall    = 0.975148
Stitched  = 4323
```

### 15.1 Final Combo 剩余 21 FN

类别分布：

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

进一步对 Original / HFlip 候选进行 oracle 几何诊断，发现 8 个 FN 理论上存在简单双视角框融合后达到 `IoU >= 0.5` 的可能。

但 oracle 只用于判断机制潜力，不作为 Test 构框规则。

### 15.2 GT-independent Cross-View Fusion

新增：

```text
scripts/20_crossview_fusion_sweep.py
scripts/21_fusion_rescue_attribution.py
```

在不使用 GT 选择测试框的前提下，通过：

```text
Original proposal
+
HFlip proposal
+
same-class geometric pairing
+
envelope / avg50 / score-weighted fusion
```

确认 4 个可独立救回的 Val FN：

```text
jieba      +1
huashang   +1
jiaza      +2
```

4 个 rescue 互不重叠。

### 15.3 Targeted Fusion

新增：

```text
scripts/22_targeted_fusion_combo.py
```

针对不同类别 / 几何机制分别设计：

```text
J: jieba
   横向分离、尺寸接近
   → envelope

H: huashang
   y 方向高度对齐、x 方向错位
   → envelope

A: jiaza
   x 中心接近、高度差异明显
   → avg50

B: jiaza
   横向分离、尺寸接近
   → envelope
```

Targeted Fusion 固定 Val：

```text
Baseline:
TP = 824
FN = 21

Targeted:
TP = 828
FN = 17

Recall    = 0.979882
ScoreLike = 97.99

Δ TP = +4
Δ FN = -4
Regression = 0
```

### 15.4 Fusion Pruning

初始 targeted 规则虽然得到 +4 TP，但会增加：

```text
1,381,340 fusion boxes
```

因此新增：

```text
scripts/23_targeted_fusion_prune_sweep.py
```

对四类几何机制增加上下界约束。

最终：

```text
pruned_v1:
TP = 828
FN = 17
added fusion boxes = 29,590

pruned_v2_Hscore:
TP = 828
FN = 17
added fusion boxes = 13,918
```

即在保持全部 +4 TP 的同时，将新增框从约 138 万压缩到数万级。

各原子规则单独依然均可保持 `+1 TP`，说明 Val 上的四类 rescue 在一定阈值邻域内是稳定的，而不是单一参数点偶然命中。

### 15.5 Hidden-Test Submission

新增：

```text
scripts/24_targeted_fusion_submission.py
```

使用已有 Test cache：

```text
Test images = 669

Original cache:
2,247,792 candidates

HFlip cache:
2,252,373 candidates
```

不重新运行模型 forward，CPU 直接生成提交。

`pruned_v1` Test：

```text
Base detections   = 2,403,809
Fusion detections = 39,676
Total detections  = 2,443,485
Zero scores       = 0
```

Fusion by rule：

```text
J1 = 33,556
H1 =  4,435
A1 =  1,296
B1 =    389
```

Test fusion 数量与 Val / Test 数据规模整体处于合理比例，因此正式提交 `pruned_v1`。

### 15.6 线上结果：没有新增 TP

原 Final Combo：

```text
Score  = 98.09
Recall = 0.9809

TP = 975
FP = 2,402,834
FN = 19
```

Targeted Fusion `pruned_v1`：

```text
Score      = 98.09
Recall     = 0.9809
Precision  = 0.0004
F1         = 0.0008
mAP@0.5    = 0.4550

TP = 975
FP = 2,442,510
FN = 19
```

精确差值：

```text
Δ TP = 0
Δ FN = 0

Δ FP =
2,442,510 - 2,402,834
= 39,676
```

这与 Test 中新增的：

```text
Fusion detections = 39,676
```

**完全一致。**

因此可以明确判断：

```text
39,676 个新增 fusion boxes
→ 0 个新增 TP
→ 39,676 个全部成为 FP
```

即：

```text
Val:
824 / 21
→ 828 / 17
(+4 TP)

Hidden Test:
975 / 19
→ 975 / 19
(+0 TP)
```

### 15.7 负实验结论

该实验说明：

1. Original / HFlip Box Fusion 在固定 Val 的最后少量 FN 上存在明显局部收益。
2. 经过 pruning 后，Val 上的 +4 TP 并非依赖海量框数量。
3. 但这四类局部几何 rescue **没有迁移到隐藏测试**。
4. 在最终少量 FN 阶段，仅依据固定 Val 的少数失败样本继续收紧规则存在明显过拟合风险。
5. 后续不再围绕 J/H/A/B 继续进行更窄的 Box Fusion 参数搜索。
6. `pruned_v2_Hscore` 是更严格的同机制子集，因此不值得消耗线上次数重复验证该机制。

这是一项正式保留的负实验结果：

> 本地验证可以用于筛选方向，但当剩余 FN 数量已经很少时，单个 Val FN 的几何修正规则必须警惕 validation overfitting；隐藏测试验证仍然不可替代。

---


## 16. Class-Confusion 与 Baseline 互补性：后续两次 Hidden-Test 负实验

Cross-View Fusion 在隐藏测试没有迁移后，继续尝试两个与“框坐标融合”不同的机制：

```text
A. Class-Confusion Correction
B. Baseline / RareOS Cross-Model Complementarity
```

这两条路线均先通过 CPU 和已有缓存完成筛选，再用有限的线上提交做机制验证。

### 16.1 Final Combo 21 FN 的类别混淆审计

在：

```text
scripts/26_class_confusion_dup_sweep.py
```

中进一步拆分 Final Combo 的 `21` 个 FN：

| Failure Type                       | Count |
| ---------------------------------- | ----: |
| class_confusion                    |     6 |
| localization_failure               |     6 |
| localization_near_miss             |     3 |
| fusion_oracle_rescuable            |     3 |
| no_same_class_candidate            |     2 |
| localization_or_tile_fragmentation |     1 |

其中 6 个 `class_confusion` 已存在 `IoU >= 0.5` 的错误类别框：

```text
jieba      <- yanghuatiepi   IoU = 0.7590
jieba      <- mamianmakeng   IoU = 0.7496
jiaza      <- jieba          IoU = 0.7199
jiaza      <- jieba          IoU = 0.6062
yiwuyaru   <- huashang       IoU = 0.5727
jiaza      <- yiwuyaru       IoU = 0.5433
```

说明这一部分失败的主要问题并不是定位，而是类别判定。

同时对整个 474-image Val 做 wrong-class support 审计，确认若干混淆关系并非只存在于单一 FN，例如：

```text
yanghuatiepi -> jieba
37 GT / 31 images

jieba -> jiaza
16 GT / 9 images

huashang -> yiwuyaru
14 GT / 14 images
```

因此继续测试“保留原框，同时复制为可能目标类别”的 GT-independent correction。

### 16.2 Class-Confusion Duplication 与裁剪

无条件复制五条 mapping：

```text
yanghuatiepi -> jieba
mamianmakeng -> jieba
jieba        -> jiaza
huashang     -> yiwuyaru
yiwuyaru     -> jiaza
```

固定 Val 一度达到：

```text
TP        = 829
FN        = 16
Recall    = 0.981065
ScoreLike = 98.11

ΔTP = +5
ΔFN = -5
```

但代价是：

```text
+995,332 duplicated boxes
```

随后通过：

```text
scripts/27_class_confusion_rank_prune.py
scripts/28_confusion_rank_support_audit.py
scripts/29_confusion_min_rescue_rank.py
scripts/30_confusion_score_gate_sweep.py
scripts/31_confusion_rank_robustness.py
```

逐步分析 source-class score rank、Whole-Val rank 分布、最小 rescue rank 和粗粒度 score gate。

最终保留的两条相对稳健机制：

```text
yanghuatiepi -> jieba
Top50
source score >= 1e-4

mamianmakeng -> jieba
Top20
source score >= 1e-3
```

其固定 Val 结果：

```text
Final Combo:
TP = 824
FN = 21

mid_50_20:
TP = 826
FN = 19

ΔTP = +2
ΔFN = -2
Added = 14,531
```

并且从：

```text
20 / 5
30 / 10
50 / 20
100 / 50
score-only
```

均保持 `+2 TP`，表明本地增益不依赖单一 Top-K 参数点。

### 16.3 Class-Confusion Hidden-Test 结果

使用：

```text
scripts/32_confusion_submission.py
```

复用已有 Test Original / HFlip cache，生成 `mid_50_20` submission。

Test：

```text
Base detections = 2,403,809
Added           = 22,047
Total           = 2,425,856
```

线上：

```text
Score  = 98.09
Recall = 0.9809

TP = 975
FN = 19
FP = 2,424,881
```

与 Final Combo 对比：

```text
ΔTP = 0
ΔFN = 0

ΔFP =
2,424,881 - 2,402,834
= 22,047
```

FP 增量与新增框完全一致：

```text
22,047 duplicated boxes
→ 0 new TP
→ 22,047 new FP
```

因此，即使该 confusion correction 在整个 Val 上表现出一定系统支持，其具体两条 rescue 机制仍未迁移到 Hidden Test。

---

### 16.4 旧 Baseline Test Submission 的跨模型互补性审计

由于暂时没有 GPU，无法立即生成 Baseline High-Recall Val cache，因此先复用历史 Baseline Test submission：

```text
submissions/baseline1_e73_conf001_nms050.json
```

其中：

```text
Baseline Test boxes = 6,463
minimum score       ≈ 0.01
```

使用：

```text
scripts/33_baseline_test_complementarity_audit.py
```

将每个 Baseline box 与当前 `98.09` Final Combo 做逐框几何比较。

结果：

```text
same_class_covered      = 5,729  (88.6%)
class_disagreement      =   178  ( 2.8%)
independent_geometry    =   556  ( 8.6%)

same-class IoU < 0.5    =   734  (11.4%)
```

说明旧 Baseline 普通推理中确实存在一小部分 RareOS Final Combo 没有同类覆盖的真实独立 proposal。

按 Baseline 原始 score 筛选：

| Baseline score floor | Unique `<0.5` | Class disagreement | Independent geometry |
| -------------------: | ------------: | -----------------: | -------------------: |
|                 0.01 |           734 |                178 |                  556 |
|                 0.02 |           366 |                 87 |                  279 |
|                 0.05 |           134 |                 40 |                   94 |
|                 0.10 |            68 |                 26 |                   42 |
|                 0.20 |            35 |                 17 |                   18 |
|                 0.50 |            11 |                  5 |                    6 |

因此选择 `score >= 0.10` 的 `68` 个高置信 unique Baseline boxes，作为高信息密度跨模型探针。

类别分布：

```text
jieba           13
zonglie          5
qilie            5
jiaza            2
yiwuyaru        14
huashang         4
mamianmakeng    15
yanghuatiepi     6
gunyin           4
```

其中：

```text
class_disagreement   = 26
independent_geometry = 42
```

### 16.5 68-Box Baseline Unique Probe：Hidden-Test 仍无增益

使用：

```text
scripts/34_baseline_unique_submission.py
```

生成：

```text
Base detections = 2,403,809
Added           = 68
Total           = 2,403,877
```

线上结果：

```text
Score  = 98.09
Recall = 0.9809

TP = 975
FN = 19
FP = 2,402,902
```

精确差值：

```text
ΔTP = 0
ΔFN = 0

ΔFP =
2,402,902 - 2,402,834
= 68
```

因此：

```text
68 high-confidence Baseline unique boxes
→ 0 new TP
→ 68 new FP
```

这一结果说明：

> 旧 Baseline 常规推理产生的高置信独立框，也没有覆盖当前 Hidden-Test 剩余 19 FN。

但该结论 **不能直接否定 Baseline / RareOS 模型互补性本身**，因为旧 submission 的最低阈值约为 `0.01`，而当前主方案依赖 `conf=1e-5` 的 High-Recall proposal space。

因此下一步不再扩大旧 Baseline submission 的 68 → 134 → 734 个框，而是等待 GPU 后生成真正对齐主方案参数的 Baseline High-Recall Val Original cache，再进行正规的 proposal-level complementarity analysis。

---

### 16.6 三次 Hidden-Test 负实验的联合结论

2026-08-11 的三次机制探针：

```text
1. Targeted Cross-View Fusion
   +39,676 boxes
   +0 TP

2. Class-Confusion Correction
   +22,047 boxes
   +0 TP

3. Baseline High-Confidence Unique Probe
   +68 boxes
   +0 TP
```

三次均满足：

```text
FP 增量 = 新增框数量
TP 不变
FN 不变
```

说明当前 Hidden-Test 剩余 `19` FN 没有被这些机制覆盖。

因此正式冻结以下原则：

> 暂停“观察 Final Combo 剩余 21 个 Val FN → 围绕这些具体 FN 继续设计局部规则 → 直接提交 Hidden Test”的实验路线。

下一阶段应优先获取新的 **High-Recall 模型级 proposal 来源**，并在固定 Val 上先验证跨模型互补性，再决定是否投入 Test inference 与线上提交。

---

## 17. 当前主要结论

### 训练侧：RareOS v1 有效

```text
Validation Recall
0.494 → 0.541
```

说明定向长尾重采样可以提高召回能力。

### High-Recall 推理有效，但存在边际收益递减

此前最有效的基础策略：

```text
1. 降低 confidence threshold
2. 增加 tile overlap
3. 放宽 global NMS
```

但在：

```text
conf       = 1e-5
stride     = 768
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

### Local Val 增益不等于 Hidden-Test 增益

2026-08-11 已有两类“Val 正增益 → Hidden Test 零增益”的直接证据：

```text
Targeted Cross-View Fusion:
Val +4 TP
Hidden Test +0 TP

Class-Confusion Correction:
Val +2 TP
Hidden Test +0 TP
```

另外，旧 Baseline 模型的 `68` 个 `score>=0.10` 高置信独立 Test proposals 也得到：

```text
Hidden Test +0 TP
```

因此在最后少量 FN 阶段：

```text
FN diagnostic
→ 针对固定 Val 设计局部规则
→ Val 提升
```

已经不足以证明规则具有数据分布层面的泛化能力。

后续优先寻找：

```text
新的 High-Recall 模型级 proposal 来源
Baseline / RareOS 正规 Val complementarity
跨模型 class evidence
```

而不是继续围绕 Final 21 Val FN 做几何修正、类别复制或更细阈值搜索。

---

## 18. 下一阶段计划

当前线上最好结果：

```text
Score = 98.09
TP    = 975
FN    = 19
FP    = 2,402,834
```

2026-08-11 已经完成三次 Hidden-Test 机制探针：

```text
Cross-View Fusion          +0 TP
Class-Confusion Correction +0 TP
Baseline Unique 68-box     +0 TP
```

因此下一阶段不再继续扩大这些已失败机制的框数量。

### Priority 1：生成 Baseline High-Recall Val Original Cache

Baseline 正式权重仍在：

```text
runs/baseline/yolo26m_tiles1280_e80_b6_seed2026/weights/best.pt
```

当前缺少的是与 RareOS 主方案 **完全同推理条件** 的 Baseline proposal cache：

```text
images      = 474 Val originals
tile_size   = 1280
stride      = 768
conf        = 1e-5
tile_iou    = 0.60
global_iou  = 0.90
max_det     = 1000
```

已经准备：

```text
scripts/25_build_baseline_val_cache.sh
```

GPU 恢复后只执行：

```text
Baseline best.pt
+
474 Val Original
+
High-Recall inference
        ↓
Baseline Val Original cache
```

第一轮 **不跑 Baseline HFlip、不跑 Test、不重新训练**。

这样 GPU 工作量保持最小；cache 生成后立即重新回到 CPU。

### Priority 2：Baseline / RareOS Proposal-Level Complementarity

Baseline High-Recall Val cache 到位后，应重新回答：

```text
1. Baseline 能直接 rescue 几个 Final Combo FN？
2. 是否存在 RareOS Original/HFlip 完全没有的 same-class proposal？
3. Baseline 是否能纠正 RareOS class confusion？
4. 这些 rescue 是否分布在多个类别 / 多张图，而不是单一 Val 样本？
5. Final Combo + Baseline candidate union 的完整 Val 净增益是多少？
```

继续推进的门槛应明显高于此前局部规则：

```text
优先：
>= 3 个独立 FN rescue

或：
数量较少，但明确解决
no_same_class_candidate / class_confusion
等不同失败机制
```

### Priority 3：只有 Val 跨模型互补成立后才生成 Baseline Test Cache

禁止直接：

```text
旧 Baseline Test submission
→ 不断降低 score floor
→ 扩大 68 / 134 / 734 个框
→ 继续线上试错
```

68-box 高置信探针已经给出 `+0 TP`。

正确流程：

```text
Baseline High-Recall Val cache
        ↓
CPU complementarity audit
        ↓
确认可解释且跨样本的独立 rescue
        ↓
才生成 Baseline High-Recall Test cache
        ↓
Model Ensemble submission
```

### Priority 4：暂时冻结已失败方向

暂不继续：

```text
J/H/A/B Cross-View Fusion threshold tuning
Class-Confusion duplication threshold tuning
旧 Baseline normal-submission score-floor expansion
针对 Final 21 Val FN 的逐样本规则
```

这些方向已经有足够 Hidden-Test 负证据。

阶段目标仍为：

```text
98.09
→ 找到真正跨分布的独立 proposal 机制
→ 再尝试 98.5+ / 99
```

重点从“继续把 Val FN 做少”转向“验证新的机制是否真正泛化”。

---

## 19. 项目结构

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
│   ├── 18_final_combo_from_cache.py
│   ├── 19_final_combo_fn_diagnostic.py
│   ├── 20_crossview_fusion_sweep.py
│   ├── 21_fusion_rescue_attribution.py
│   ├── 22_targeted_fusion_combo.py
│   ├── 23_targeted_fusion_prune_sweep.py
│   ├── 24_targeted_fusion_submission.py
│   ├── 25_build_baseline_val_cache.sh
│   ├── 26_class_confusion_dup_sweep.py
│   ├── 27_class_confusion_rank_prune.py
│   ├── 28_confusion_rank_support_audit.py
│   ├── 29_confusion_min_rescue_rank.py
│   ├── 30_confusion_score_gate_sweep.py
│   ├── 31_confusion_rank_robustness.py
│   ├── 32_confusion_submission.py
│   ├── 33_baseline_test_complementarity_audit.py
│   ├── 34_baseline_unique_submission.py
│   ├── 35_baseline_highrecall_complementarity.py
│   ├── 36_baseline_rescue_structure_audit.py
│   ├── 37_baseline_huashang_gate_sweep.py
│   ├── 38_build_baseline_huashang_complement_cache.py
│   ├── 39_build_baseline_test_cache.sh
│   └── 40_final_combo_test_stream.py
│
├── docs/
│   └── experiment_log_20260808.md
│
├── splits/
├── .gitignore
└── README.md
```

---

## 20. 关键脚本

| Script                                      | 功能                                                         |
| ------------------------------------------- | ------------------------------------------------------------ |
| `00_visualize_voc.py`                       | VOC 标注可视化                                               |
| `01_make_bbox_crops.py`                     | 生成缺陷 bbox 裁剪                                           |
| `02_audit_dataset.py`                       | 数据集审计                                                   |
| `03_voc_to_yolo.py`                         | VOC → YOLO                                                   |
| `04_make_grouped_split.py`                  | 防泄漏 Train / Val 分组                                      |
| `05_make_tile_trial.py`                     | 小规模切片实验                                               |
| `06_make_full_tiles.py`                     | 构建正式 tile 数据集                                         |
| `07_analyze_baseline.py`                    | Baseline 分析                                                |
| `08_make_rare_oversample_dataset.py`        | RareOS 数据构建                                              |
| `09_run_baseline2_gpu.sh`                   | RareOS 训练                                                  |
| `10_predict_test_submission.py`             | 官方 test tiled inference                                    |
| `11_eval_highrecall_val.py`                 | 本地 Recall 模拟评测                                         |
| `12_run_highrecall_sweep.sh`                | High-Recall 参数扫描                                         |
| `13_record_experiment_results.sh`           | 自动生成阶段实验记录                                         |
| `14_fn_diagnostic.py`                       | Remaining FN 根因诊断与候选缓存                              |
| `15_cache_hflip_tta.py`                     | Horizontal Flip TTA 候选缓存                                 |
| `16_eval_hflip_tta_cache.py`                | Original + HFlip 离线 Val 评估                               |
| `17_eval_zonglie_cross_tile_stitch_gpu.py`  | Zonglie 跨切片拼接参数验证                                   |
| `18_final_combo_from_cache.py`              | Final Combo Val / Test 缓存后处理与提交生成                  |
| `19_final_combo_fn_diagnostic.py`           | Final Combo 剩余 21 FN 精确诊断                              |
| `20_crossview_fusion_sweep.py`              | 内存安全的 Original/HFlip 跨视角融合扫描                     |
| `21_fusion_rescue_attribution.py`           | Fusion rescue / regression 归因                              |
| `22_targeted_fusion_combo.py`               | 类别特异 Targeted Fusion 组合验证                            |
| `23_targeted_fusion_prune_sweep.py`         | Targeted Fusion 几何门限裁剪                                 |
| `24_targeted_fusion_submission.py`          | CPU 流式生成 Targeted Fusion Test submission                 |
| `25_build_baseline_val_cache.sh`            | GPU 可用后生成 Baseline High-Recall Val Original cache       |
| `26_class_confusion_dup_sweep.py`           | Class-confusion duplication 与 Whole-Val wrong-class support |
| `27_class_confusion_rank_prune.py`          | Source-class Top-K rank 裁剪                                 |
| `28_confusion_rank_support_audit.py`        | Whole-Val confusion source-rank 分布审计                     |
| `29_confusion_min_rescue_rank.py`           | 最小 rescue rank 与救援 GT 归因                              |
| `30_confusion_score_gate_sweep.py`          | Coarse source-score gate 扫描                                |
| `31_confusion_rank_robustness.py`           | Class-confusion rank 稳定平台验证                            |
| `32_confusion_submission.py`                | Class-confusion Test submission 生成                         |
| `33_baseline_test_complementarity_audit.py` | 旧 Baseline Test 与 Final Combo 跨模型几何互补性审计         |
| `34_baseline_unique_submission.py`          | 高置信 Baseline unique-box Test probe 生成                   |
| `35_baseline_highrecall_complementarity.py`  | Baseline High-Recall 与 Final Combo 剩余 FN 的 proposal-level 互补性审计 |
| `36_baseline_rescue_structure_audit.py`      | Baseline rescue 的 score rank、RareOS overlap 与结构审计     |
| `37_baseline_huashang_gate_sweep.py`         | huashang Baseline complement 的 score / overlap 稳定平台扫描 |
| `38_build_baseline_huashang_complement_cache.py` | 构造 RareOS Original + 冻结 Baseline huashang complement 派生 cache |
| `39_build_baseline_test_cache.sh`            | 生成 669 Test Baseline High-Recall Original cache            |
| `40_final_combo_test_stream.py`              | 低内存流式生成百万级 Final Combo Test submission，避免 OOM   |

已有详细实验记录：

```text
docs/experiment_log_20260808.md
```

---

## 21. 实验原则

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
12. **记录负实验，而不是只保留成功实验**
13. **最终少量 FN 阶段警惕 Validation Overfitting**
14. **Hidden-Test 结果用于检验机制泛化，而不是只看 Val 最优点**
15. **同一机制 Hidden-Test 明确失败后，不通过单纯扩大框数继续重复试错**
16. **最终少量 FN 阶段优先引入新的模型级 proposal，而不是继续拟合具体 Val 样本**
17. **Val rescue 除数量外，还必须检查是否跨图片、跨类别、跨 failure cluster**
18. **大规模 submission 优先使用流式写盘，避免百万级 Python dict 列表与 `json.dumps` 峰值内存导致 OOM**

同时明确区分：

```text
dataset problem
tiling problem
training problem
inference problem
post-processing problem
model-complementarity problem
validation-overfitting problem
submission-engineering problem
```

---

## 22. 数据、缓存与权重

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
- Cross-View Fusion 实验代码
- Class-Confusion / Rank-Score 审计代码
- Baseline / RareOS 跨模型互补性审计代码
- 大规模 submission 的低内存流式生成代码
- 实验记录
- 正实验与负实验结论
- 可复现研究流程

---

## 23. 当前状态

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

2026-08-11 Hidden-Test probes：

```text
1) Targeted Cross-View Fusion
Val:
824 / 21 → 828 / 17

Test:
Added = 39,676
TP    = 975
FN    = 19
FP    = 2,442,510
ΔTP   = 0

2) Class-Confusion mid_50_20
Val:
824 / 21 → 826 / 19

Test:
Added = 22,047
TP    = 975
FN    = 19
FP    = 2,424,881
ΔTP   = 0

3) Baseline Unique score>=0.10
Test:
Added = 68
TP    = 975
FN    = 19
FP    = 2,402,902
ΔTP   = 0
```

三次均满足：

```text
ΔFP = Added Boxes
ΔTP = 0
ΔFN = 0
```

当前主方案保持：

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

已经完成并冻结的负实验路线：

```text
Final 21 Val FN
    ├── Targeted Cross-View Fusion
    │      └── Hidden Test +0 TP
    │
    ├── Class-Confusion Correction
    │      └── Hidden Test +0 TP
    │
    └── Old Baseline High-Confidence Unique Probe
           └── Hidden Test +0 TP
```

下一阶段唯一优先主线：

```text
等待 GPU
    ↓
Baseline best.pt
    ↓
High-Recall Val Original cache
(conf=1e-5, stride=768)
    ↓
CPU Baseline / RareOS
proposal-level complementarity audit
    ↓
只有 Val 跨模型独立 rescue 成立
才考虑 Baseline High-Recall Test cache
    ↓
Model Ensemble
```

项目仍在持续开发中。

---

## 24. 2026-08-12 Baseline High-Recall 跨模型互补性实验

> 本节是对第 16、18、23 节在 2026-08-12 的增量更新。原章节保留 2026-08-11 当时的实验状态和决策路径，不做删除。

旧 Baseline normal submission 的最低置信度约为：

```text
conf ≈ 0.01
```

而当前 RareOS 正式方案依赖：

```text
conf = 1e-5
```

因此 2026-08-11 的 68-box Baseline Unique Probe 只能否定“旧 Baseline 高置信独立 proposal”，不能否定真正的 Baseline High-Recall proposal space。

2026-08-12 GPU 恢复后，正式完成完整 80 epoch Baseline 的 High-Recall Val / Test proposal 互补性实验。

### 24.1 Baseline High-Recall Val Original Cache

正式权重：

```text
runs/baseline/
yolo26m_tiles1280_e80_b6_seed2026/
weights/best.pt
```

使用：

```text
scripts/25_build_baseline_val_cache.sh
```

推理条件与 RareOS Val Original 完全对齐：

```text
images      = 474 Val originals
tile_size   = 1280
stride      = 768
batch       = 6
conf        = 1e-5
tile_iou    = 0.60
global_iou  = 0.90
max_det     = 1000
half        = True
```

结果：

```text
Images       = 474
Candidates   = 1,646,484
NPZ files    = 474
Cache size   ≈ 90 MB
Elapsed      ≈ 3.49 min
```

RareOS Original Val cache 为：

```text
Candidates = 1,502,629
```

因此 Baseline 在同一 High-Recall 推理条件下产生了更多候选，说明两套模型的 proposal space 并不完全相同。

---

### 24.2 Proposal-Level Complementarity Audit

新增：

```text
scripts/35_baseline_highrecall_complementarity.py
```

对 Final Combo 剩余 `21` 个 FN 逐一比较：

```text
Baseline High-Recall
RareOS Original
RareOS HFlip
```

的 same-class candidate 最佳 IoU。

结果：

```text
Baseline IoU >= 0.30         : 4
Baseline IoU >= 0.40         : 3
Baseline IoU >= 0.50         : 3

Direct Baseline rescue       : 3
Cache-space unique rescue    : 3
Deep blindspot rescue        : 0
Hard blindspot rescue        : 0
```

这 3 个 rescue 全部来自同一张图片：

```text
C1297627_V01_F00002_08304277-d9b9-4278-992a-2fbf45ac0c3f.jpg
```

并且全部属于：

```text
huashang
```

失败类型：

```text
localization_near_miss +1
localization_failure   +2
```

因此虽然 proposal-level 上达到 `3` 个 rescue，但它们不是 3 个独立图片级 blind spot，而更像同一图片中的系统性 localization complementarity。

---

### 24.3 Baseline Rescue Structure Audit

新增：

```text
scripts/36_baseline_rescue_structure_audit.py
```

三个有效 Baseline `huashang` proposal：

| Rescue | Baseline → GT IoU | Baseline score | huashang score rank | Baseline → RareOS O/H max IoU |
| ------ | ----------------: | -------------: | -------------------: | ----------------------------: |
| #1     |            0.5407 |       1.13e-4 |            357 / 983 |                        0.4929 |
| #2     |            0.7154 |       2.30e-5 |            701 / 983 |                        0.5946 |
| #3     |            0.6118 |       1.20e-4 |            349 / 983 |                        0.5102 |

关键观察：

```text
Top50 / Top100 / Top200
均无法保留全部 3 个 rescue

Top600
仍会丢掉最强的 rescue #2
```

有效信息位于极低置信度 proposal space，因此不能通过普通 high-score / shallow Top-K 筛出。

同时：

```text
Baseline → RareOS nearest same-class IoU
≈ 0.46 ~ 0.59
```

说明 Baseline 并不是看到了 RareOS 完全没有的目标，而是在相同目标附近给出了不同的 localization hypothesis。

其中一个典型 case：

```text
RareOS Original:
score = 0.315674
GT IoU = 0.4417

Baseline:
score = 2.30e-5
GT IoU = 0.7154
```

表现为“RareOS 高置信但定位偏差，Baseline 极低置信 proposal 反而具有更好几何位置”。

---

### 24.4 Huashang GT-Independent Gate Sweep

新增：

```text
scripts/37_baseline_huashang_gate_sweep.py
```

规则族：

```text
class = huashang
min_score = 2e-5

Baseline proposal
与 RareOS Original / HFlip
最近同类 proposal overlap < gate
```

核心结果：

| min_score | upper_score | overlap_gate | added_boxes | oracle_rescues_kept |
| --------: | ----------: | -----------: | ----------: | ------------------: |
| 2e-5 | 3e-4 | 0.60 | 32,836 | 3 |
| 2e-5 | 3e-4 | 0.65 | 37,661 | 3 |
| 2e-5 | 3e-4 | 0.70 | 43,264 | 3 |
| 2e-5 | 1e-3 | 0.65 | 41,022 | 3 |

最终冻结：

```text
class        = huashang
min_score    = 2e-5
max_score    = 3e-4
overlap_gate = 0.65
```

选择 `0.65` 而不是紧贴 rescue 边界的 `0.60`，以保留一定的几何泛化余量。

---

### 24.5 Exact Final Combo Val：形成 +5 TP 稳定平台

新增：

```text
scripts/38_build_baseline_huashang_complement_cache.py
```

先构造派生 Original cache：

```text
RareOS Original
+
selected Baseline huashang proposals
```

随后继续复用原 `scripts/18_final_combo_from_cache.py` 的完整链路：

```text
Selective HFlip
+
Global NMS
+
Zonglie Cross-Tile Stitching
+
Val matching
```

邻域验证：

| Config | TP | FP | FN | Recall | ScoreLike |
| ------ | -: | -: | -: | -----: | --------: |
| Original Final Combo | 824 | 1,631,208 | 21 | 0.975148 | 97.51 |
| g060 / upper=3e-4 | 828 | 1,658,278 | 17 | 0.979882 | 97.99 |
| **g065 / upper=3e-4** | **829** | **1,662,159** | **16** | **0.981065** | **98.11** |
| g070 / upper=3e-4 | 829 | 1,666,698 | 16 | 0.981065 | 98.11 |
| g065 / upper=1e-3 | 829 | 1,664,445 | 16 | 0.981065 | 98.11 |

因此不是单一参数点：

```text
g060       +4 TP
g065       +5 TP
g070       +5 TP
g065_u1e3  +5 TP
```

最终冻结 `g065 / 3e-4`，因为它在保持 `829/16` 的配置中 FP 最少。

Val 正式结果：

```text
824 TP / 21 FN
→
829 TP / 16 FN

ΔTP = +5
ΔFN = -5
```

---

### 24.6 Baseline High-Recall Test Cache

新增：

```text
scripts/39_build_baseline_test_cache.sh
```

Test 参数继续完全对齐：

```text
images      = raw/data/test
images_count= 669
tile_size   = 1280
stride      = 768
batch       = 6
conf        = 1e-5
tile_iou    = 0.60
max_det     = 1000
half        = True
```

结果：

```text
Images       = 669
Candidates   = 2,352,507
NPZ files    = 669
Cache size   ≈ 128 MB
Elapsed      ≈ 5.08 min
```

使用冻结的 `g065` 规则后：

```text
Original candidates = 2,247,792
Selected Baseline   =    40,820
Output candidates   = 2,288,612
```

Test 阶段不再根据 candidate 数量重新调 `score` 或 `overlap_gate`。

---

### 24.7 大规模 Submission OOM 与流式写盘

原：

```text
scripts/18_final_combo_from_cache.py
```

Test 分支会执行：

```python
submission = []
```

并将约 240 万个 detection 逐个保存为 Python dict，最后调用：

```python
json.dumps(submission, ensure_ascii=False, indent=2)
```

在百万级 submission 下会产生非常高的内存峰值。

实际运行中出现：

```text
669/669 已完成
final ≈ 2.44M detections
```

但在最终 JSON 写盘阶段进程被系统终止；再次运行时甚至在约 `600/669` 因 RAM 压力提前退出。

因此新增：

```text
scripts/40_final_combo_test_stream.py
```

该脚本保持以下逻辑与 Script 18 一致：

```text
Original cache
+
Selective HFlip
+
Global NMS
+
Zonglie Stitch
+
submission_row
```

只修改 submission 输出方式：

```text
每处理一张图片
→ detection 立即流式写入 JSON
→ 不在内存中累计 240 万个 Python dict
```

同时使用 `.tmp` 文件完成后再替换正式 submission，避免留下“看似存在但未写完整”的 JSON。

最终成功得到：

```text
Images       = 669
Detections   = 2,436,981
Stitched     = 6,210
Zero scores  = 0
Nonfinite    = 0
JSON MB      = 290.37
```

类别分布：

```text
jieba             551,710
zonglie           604,905
qilie              95,636
jiaza             123,044
yiwuyaru          310,899
huashang          216,489
mamianmakeng      141,678
yanghuatiepi      225,866
gunyin            166,754
```

原 Final Combo Test：

```text
Detections = 2,403,809
Stitched   = 6,210
```

新版本：

```text
Detections = 2,436,981
Stitched   = 6,210
```

最终新增：

```text
2,436,981 - 2,403,809
= 33,172 detections
```

---

### 24.8 Hidden-Test：第四次机制级负实验

Baseline High-Recall `huashang g065` 正式线上结果：

```text
Score      = 98.09
Recall     = 0.9809
Precision  = 0.0004
F1         = 0.0008
mAP@0.5    = 0.4550

TP = 975
FP = 2,436,006
FN = 19
```

正式最好 Final Combo：

```text
TP = 975
FP = 2,402,834
FN = 19
```

精确差值：

```text
ΔTP = 0
ΔFN = 0

ΔFP =
2,436,006 - 2,402,834
= 33,172
```

这与 submission 最终新增 detection 数：

```text
33,172
```

**完全一致。**

因此可以确定：

```text
33,172 Baseline High-Recall huashang complement detections
→ 0 new TP
→ 33,172 new FP
```

即：

```text
Val:
824 / 21
→ 829 / 16
(+5 TP)

Hidden Test:
975 / 19
→ 975 / 19
(+0 TP)
```

这比此前负实验更有信息量，因为本次同时具备：

```text
不同训练模型 proposal source
+
极低置信 High-Recall proposal space
+
Val +5 TP
+
参数邻域稳定平台
```

但仍然没有覆盖 Hidden-Test 剩余 `19` FN。

---

### 24.9 四次 Hidden-Test 机制验证统一总结

截至 2026-08-12：

| Mechanism | Val / 本地结果 | Test 最终新增 detection | Hidden ΔTP | Hidden ΔFN | Hidden ΔFP |
| --------- | -------------: | ----------------------: | ---------: | ---------: | ---------: |
| Targeted Cross-View Fusion | +4 TP | 39,676 | 0 | 0 | +39,676 |
| Class-Confusion `mid_50_20` | +2 TP | 22,047 | 0 | 0 | +22,047 |
| Baseline Unique `score>=0.10` | Test-only | 68 | 0 | 0 | +68 |
| **Baseline High-Recall huashang g065** | **+5 TP** | **33,172** | **0** | **0** | **+33,172** |

四次都严格满足：

```text
ΔFP = Added final detections
ΔTP = 0
ΔFN = 0
```

因此当前阶段得到更强的实验结论：

> **固定 Val 上的 rescue 数量，甚至“不同模型 + 参数稳定平台”的正结果，都不足以单独证明最后少量 Hidden-Test FN 上的可迁移性。**

后续 Hidden-Test GO / NO-GO 不再只依据：

```text
Val ΔTP
```

而必须同时考虑：

```text
rescue 是否跨多张图片
rescue 是否跨多个类别
rescue 是否跨不同 failure cluster
新 proposal source 是否具有真正不同的 observation mechanism
full-Val combo 是否无 regression
参数是否存在稳定平台
```

尤其是：

```text
多个 rescue 全部来自同一张图片
```

即使达到 `+4 / +5 TP`，也不再自动视为足够强的 Hidden-Test 提交证据。

---

## 25. 2026-08-12 更新后的下一阶段计划

第 18 节中“等待 GPU → Baseline High-Recall complementarity”的计划已经完整执行，并得到 Hidden-Test `+0 TP` 的机制级负结果。

因此下一阶段正式从“跨模型 Baseline 补充”转向新的 observation space。

### Priority 1：RareOS Vertical Flip Val

当前 Original + Horizontal Flip 都来自同一 RareOS 主模型，但 Vertical Flip 尚未验证。

优先流程：

```text
RareOS best.pt
+
474 Val Vertical Flip cache
        ↓
Map back to original coordinates
        ↓
与 Final Combo remaining FN 做 proposal-level audit
        ↓
检查 rescue 是否跨图片 / 跨类别 / 跨 failure type
        ↓
full-Val combo
```

新的最低准入要求不再只是 `>=3 TP`，而是优先要求：

```text
rescue >= 3
至少跨 2~3 张图片
最好跨 >= 2 个类别
不全部来自同一 failure cluster
```

若 `+4 / +5 TP` 全部来自同一张图片，则默认不进入 Hidden Test。

### Priority 2：RareOS 180° Rotation

如果 Vertical Flip 能提供新 proposal，则继续验证 180° Rotation；如果 Vertical Flip 几乎完全被 O/H proposal space 覆盖，也可以用 180° 作为第二种变换做低成本验证。

### Priority 3：Multi-Scale / Different Tile Geometry

若简单几何 view 不足，考虑：

```text
不同 inference scale
不同 tile_size
不同 stride pattern
```

目标是改变目标在 tile 内的相对尺度、边界位置和上下文范围，而不是继续拟合现有候选框。

### Priority 4：真正更强的模型多样性

若 inference-view 仍不足，再考虑：

```text
不同训练 seed
不同模型尺寸
不同 model family
不同数据重采样策略
```

目标不是做平均 ensemble，而是寻找当前 RareOS / HFlip / Baseline High-Recall 都没有覆盖的 blind spot。

### 当前冻结方向

暂不继续：

```text
Targeted O/H Box Fusion
更窄 Fusion geometry gate
Class-Confusion duplication
Class-Confusion Top-K / score tuning
旧 Baseline normal score-floor expansion
Baseline High-Recall huashang g060 / g065 / g070
扩大 Baseline huashang score range
针对 Final 21 Val FN 的逐样本规则
```

原因：

```text
Hidden Test 已连续 4 次给出机制级负反馈
```

当前正式最好线上仍为：

```text
Score = 98.09
TP    = 975
FN    = 19
FP    = 2,402,834
```

距离约 `98.39` / Top 5 水平仍只需要约：

```text
+3 TP
```

但下一阶段的核心已经不再是：

```text
如何继续把固定 Val 的 21 FN 做少？
```

而是：

> **如何制造一个当前 RareOS Original / HFlip / Baseline High-Recall 都未覆盖的新观察视角，并在多个 Val 图片上证明其独立互补性？**

