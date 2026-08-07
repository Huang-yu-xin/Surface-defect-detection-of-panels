# 钢板表面缺陷检测

基于 **YOLO26m** 的高分辨率钢板表面缺陷检测项目。

本项目面向约 `4096 × 3000` 的工业钢板图像，完成 9 类表面缺陷目标检测，并重点研究：

- 高分辨率滑窗切片
- 长尾类别重采样
- 长条缺陷检测
- High-Recall 高召回推理
- 跨切片 Global NMS
- 本地 Recall 模拟评测
- 可复现的单变量消融实验

当前已完成：

```text
Baseline
→ RareOS v1
→ High-Recall Inference
→ Local Recall Sweep
→ Online Leaderboard Validation
```

当前最好线上成绩：

```text
Score  = 95.67
Recall = 0.9567

TP = 951
FN = 43
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

此外，`zonglie`、`qilie`、`huashang` 具有明显的细长 / 长条目标特征，对切片边界和 overlap 较敏感。

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
        ├── High-Recall Inference
        ├── Dense Tiling
        ├── Global NMS Sweep
        └── Local Recall Simulator
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

验证集在 Baseline、RareOS、High-Recall 和 Dense Tiling 实验中保持固定。

---

## 4. 高分辨率切片

由于原始图像约为 `4096 × 3000`，直接缩放到常规 YOLO 输入尺寸可能损失小缺陷和细长缺陷信息，因此正式训练采用滑窗切片。

基础参数：

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

| 方案             |     Score |     Recall |      TP |     FN |            FP |
| ---------------- | --------: | ---------: | ------: | -----: | ------------: |
| Baseline normal  |     82.50 |     0.8249 |     820 |    174 |         5,643 |
| RareOS normal    |     82.50 |     0.8249 |     820 |    174 |         4,808 |
| HighRecall v1    |     88.03 |     0.8803 |     875 |    119 |        18,829 |
| HighRecall v2    |     89.54 |     0.8954 |     890 |    104 |        31,858 |
| HighRecall v3    |     93.06 |     0.9306 |     925 |     69 |       176,156 |
| **Current Best** | **95.67** | **0.9567** | **951** | **43** | **1,757,850** |

从初始 Baseline 到当前最好结果：

```text
Score:
82.50 → 95.67

Δ = +13.17
```

漏检数量：

```text
FN:
174 → 43
```

共额外找回 131 个真实目标。

---

## 8. 当前最佳推理配置

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

测试集推理统计：

```text
Test images           = 669
Total tiles           = 13,380

Raw detections        = 2,247,802
Final detections      = 1,758,801

Images with detection = 669
Images without        = 0
```

对应线上结果：

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

主要实验结果：

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

本地排序与线上结果方向一致，因此后续参数筛选优先通过本地 Recall 模拟器完成。

---

## 10. 当前主要结论

### 训练侧

RareOS v1 有效：

```text
Validation Recall
0.494 → 0.541
```

说明定向长尾重采样可以提高召回能力。

### 推理侧

目前收益最大的策略是：

```text
1. 降低 confidence threshold
2. 增加 tile overlap
3. 放宽 global NMS
```

其中：

```text
tile NMS 0.60 → 0.80
```

没有带来 Recall 改善，因此暂时不作为重点方向。

### 当前主要困难类别

本地最强配置下仍需重点关注：

```text
zonglie
qilie
huashang
```

尤其需要研究：

- 长条目标被 tile 截断
- 极低样本类别
- 方向敏感目标
- 剩余 FN 的空间分布
- 不同模型之间的互补预测

---

## 11. 下一阶段计划

当前线上仍有：

```text
FN = 43
```

下一阶段不再优先采用“无限降低 conf”的方式。

计划研究：

```text
Remaining FN Analysis
        │
        ├── zonglie / qilie / huashang
        ├── TTA
        │   └── Horizontal Flip
        ├── Long-defect Tiling
        ├── Model Ensemble
        └── Additional Recall-oriented Inference
```

阶段目标：

```text
95.67
→ 97+
→ 进一步降低 FN
```

---

## 12. 项目结构

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
│   └── 13_record_experiment_results.sh
│
├── docs/
│   └── experiment_log_20260808.md
│
├── splits/
├── .gitignore
└── README.md
```

---

## 13. 关键脚本

| Script                               | 功能                      |
| ------------------------------------ | ------------------------- |
| `00_visualize_voc.py`                | VOC 标注可视化            |
| `01_make_bbox_crops.py`              | 生成缺陷 bbox 裁剪        |
| `02_audit_dataset.py`                | 数据集审计                |
| `03_voc_to_yolo.py`                  | VOC → YOLO                |
| `04_make_grouped_split.py`           | 防泄漏 Train / Val 分组   |
| `05_make_tile_trial.py`              | 小规模切片实验            |
| `06_make_full_tiles.py`              | 构建正式 tile 数据集      |
| `07_analyze_baseline.py`             | Baseline 分析             |
| `08_make_rare_oversample_dataset.py` | RareOS 数据构建           |
| `09_run_baseline2_gpu.sh`            | RareOS 训练               |
| `10_predict_test_submission.py`      | 官方 test tiled inference |
| `11_eval_highrecall_val.py`          | 本地 Recall 模拟评测      |
| `12_run_highrecall_sweep.sh`         | High-Recall 参数扫描      |
| `13_record_experiment_results.sh`    | 自动生成阶段实验记录      |

详细实验记录：

```text
docs/experiment_log_20260808.md
```

---

## 14. 实验原则

本项目尽量遵循：

1. **单变量消融**
2. **固定随机种子**
3. **固定 Train / Val**
4. **避免切片级数据泄漏**
5. **短实验筛选后再完整训练**
6. **线上提交前优先本地验证**
7. **区分训练收益和推理收益**
8. **保存完整实验配置和日志**

同时明确区分：

```text
dataset problem
tiling problem
training problem
inference problem
post-processing problem
```

---

## 15. 数据与权重

由于数据和模型文件体积较大，本仓库不保存：

```text
raw/
datasets/
runs/
logs/
submissions/

*.pt
*.pth
*.onnx
*.engine
```

尤其不提交大型预测 JSON 和模型权重。

仓库主要用于保存：

- 数据处理代码
- 数据划分
- 实验配置
- 训练脚本
- 推理代码
- Recall 评测工具
- 实验记录
- 可复现研究流程

---

## 16. 当前状态

```text
Model:
YOLO26m + RareOS v1

Validation:
Recall     = 0.541
mAP50      = 0.528
mAP50-95   = 0.304

Best Leaderboard:
Score      = 95.67
Recall     = 0.9567

TP         = 951
FN         = 43
```

项目仍在持续开发中。

下一阶段重点：

```text
Remaining FN Analysis
TTA
Long-defect Optimization
Model Ensemble
97+ Leaderboard Exploration
```