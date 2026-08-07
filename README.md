# Steel Surface Defect Detection

基于 YOLO 的钢板表面缺陷检测实验项目。

本项目面向钢板/板材表面缺陷目标检测任务，目标是在高分辨率工业图像上完成 9 类表面缺陷检测，并围绕 **高分辨率切片、长尾类别、长条缺陷以及推理后处理** 进行系统实验。

当前主要模型为 **YOLO26m**，实验重点不是一次性堆叠大量技巧，而是通过可复现的单变量消融实验分析各策略对检测性能的影响。

---

## 1. Task

原始图像尺寸约为：

```text
4096 × 3000
```

训练集：

```text
3200 images
5889 bounding boxes
1074 negative / empty-annotation images
```

共包含 9 类钢板表面缺陷：

| ID | Class |
|---:|---|
| 0 | jieba |
| 1 | zonglie |
| 2 | qilie |
| 3 | jiaza |
| 4 | yiwuyaru |
| 5 | huashang |
| 6 | mamianmakeng |
| 7 | yanghuatiepi |
| 8 | gunyin |

数据存在明显的类别不平衡问题，其中 `qilie` 和 `huashang` 属于当前重点关注的稀有类别。

---

## 2. Project Pipeline

当前数据与训练流程：

```text
VOC annotations
      │
      ▼
Dataset audit
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
      ├── Rare-class Oversampling
      │
      ├── Long-defect Tiling
      │
      └── Inference / Global NMS
```

### Grouped Split

Train / Val 划分在切片之前完成，并按照原图/生产组进行 grouped split，避免来自同一原图或同一生产组的 tile 同时进入 Train 和 Val。

当前划分：

```text
Train: 2726 images
Val:    474 images

Train/Val image overlap: 0
Train/Val group overlap: 0
```

---

## 3. High-resolution Tiling

由于原始图像约为 `4096 × 3000`，直接缩放到常规 YOLO 输入尺寸可能损失小缺陷和细长缺陷信息，因此当前正式训练采用滑窗切片。

主要参数：

```text
tile size : 1280
stride    : 1024
overlap   : 256
overlap ratio ≈ 20%
```

普通目标：

```text
visible ratio threshold = 0.35
```

长条目标：

```text
zonglie
qilie
huashang
```

使用更宽松的保留策略：

```text
visible ratio threshold = 0.20
long aspect ratio threshold = 8
```

正式切片数据规模：

```text
Train tiles: 6476
Val tiles:   1131
```

切片过程中同时处理：

- 边界截断目标
- 模糊/不确定 tile
- 背景负样本
- 大面积黑色区域
- 长条目标可见比例

---

## 4. Baseline

当前正式基准模型：

```text
Model      : YOLO26m
Image size : 1280
Batch      : 6
Epochs     : 80
Seed       : 2026
Best epoch : 73
```

Baseline validation result：

| Metric | Value |
|---|---:|
| Precision | 0.571680 |
| Recall | 0.492240 |
| mAP50 | 0.517100 |
| mAP50-95 | 0.296990 |

80 epoch 后验证指标已经出现轻微下降，因此当前不计划通过简单增加 epoch 的方式继续提升性能。

### Per-class observation

当前表现较弱的类别主要是：

```text
qilie
huashang
```

其中：

- `qilie` 的核心问题是极端数据稀缺；
- `huashang` 同时存在长尾和长条目标截断问题；
- `zonglie` 样本数量相对充足，但长条目标的切片完整性值得进一步研究。

---

## 5. Rare-class Oversampling

当前第一个正式改进实验为：

```text
RareOS v1
```

只修改训练数据曝光频率：

```text
qilie    ×4
huashang ×2
```

其他训练配置、验证集和随机种子均保持与 Baseline 一致。

训练 tile 数：

```text
Baseline : 6476
RareOS v1: 6778

Increase: +4.66%
```

验证集保持：

```text
1131 → 1131
```

该实验用于回答：

> 稀有类别的定向重采样能否显著提高其 AP，同时不明显破坏强类以及整体 mAP？

---

## 6. Long-defect Study

下一阶段重点研究长条缺陷，尤其是：

```text
zonglie
```

主要关注：

- tile overlap
- 边界截断
- long-object preservation
- visible-ratio threshold
- 跨 tile 重复预测
- global NMS

目标是区分：

```text
数据不足
≠
切片损失
≠
模型训练问题
≠
推理后处理问题
```

并通过单变量实验确定性能瓶颈。

---

## 7. Repository Structure

```text
.
├── configs/
│   ├── steel_original.yaml
│   ├── steel_tiles_1280.yaml
│   ├── steel_tiles_1280_rareos_v1.yaml
│   └── steel_tiles_trial_1280.yaml
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
│   └── 10_predict_test_submission.py
│
├── splits/
├── .gitignore
└── README.md
```

### Scripts

| Script | Purpose |
|---|---|
| `00_visualize_voc.py` | VOC 标注与样本可视化 |
| `01_make_bbox_crops.py` | 生成 bbox 缺陷裁剪样本 |
| `02_audit_dataset.py` | 原始数据集统计与标注审计 |
| `03_voc_to_yolo.py` | VOC → YOLO 标注转换 |
| `04_make_grouped_split.py` | Grouped Train / Val 划分 |
| `05_make_tile_trial.py` | 切片策略小规模试验 |
| `06_make_full_tiles.py` | 构建正式 1280 tile 数据集 |
| `07_analyze_baseline.py` | Baseline 与类别级问题诊断 |
| `08_make_rare_oversample_dataset.py` | 稀有类别重采样数据集 |
| `09_run_baseline2_gpu.sh` | RareOS GPU 训练脚本 |
| `10_predict_test_submission.py` | Test tiled inference 与提交文件生成 |

---

## 8. Environment

当前主要实验环境：

```text
GPU    : NVIDIA RTX 5090 32GB
OS     : Ubuntu 22.04
Python : 3.12
PyTorch: 2.8.0
CUDA   : 12.8
```

主要检测框架：

```text
Ultralytics YOLO
YOLO26m
```

---

## 9. Inference

正式测试集推理沿用训练阶段的切片逻辑：

```text
tile size      = 1280
stride         = 1024
overlap        = 256

tile conf      = 0.01
tile NMS IoU   = 0.60
global NMS IoU = 0.50
```

整体流程：

```text
Original test image
        │
        ▼
1280 × 1280 tiled inference
        │
        ▼
Map boxes back to original image
        │
        ▼
Class-aware global NMS
        │
        ▼
Submission
```

当前 Baseline 推理阶段暂不引入：

```text
TTA
Ensemble
Complex threshold tuning
```

以获得干净、可对比的 baseline leaderboard result。

---

## 10. Experimental Principle

本项目实验遵循以下原则：

1. **Single-variable ablation**

   每轮实验尽量只修改一个核心变量。

2. **Reproducibility**

   固定数据划分、验证集和随机种子。

3. **Strict baseline comparison**

   所有改进都与正式 Baseline 进行严格对照。

4. **Separate different error sources**

   明确区分：

   ```text
   dataset
   tiling
   training
   inference / post-processing
   ```

5. **Short screening before full training**

   新策略优先进行短训练筛选，有明确收益后再进行完整训练。

---

## 11. Current Experiment Roadmap

```text
Baseline
YOLO26m + 1280 tiles
        │
        ├── RareOS v1
        │   ├── qilie ×4
        │   └── huashang ×2
        │
        ├── Long-defect tiling
        │   └── focus: zonglie
        │
        ├── Inference optimization
        │
        └── Final combination
```

当前工作重点：

- Baseline leaderboard evaluation
- RareOS ablation
- Long-defect tiling study
- Final combination experiments

---

## 12. Data and Weights

由于数据集、训练结果和模型权重体积较大，本仓库不包含：

```text
raw/
datasets/
runs/
logs/
metadata/
submissions/

*.pt
*.pth
*.onnx
*.engine
```

仓库主要保存：

- 数据处理代码
- 数据划分信息
- 实验配置
- 训练脚本
- 推理脚本
- 可复现实验流程

---

## Status

This repository is under active development.

Current baseline:

```text
YOLO26m
mAP50    = 0.51710
mAP50-95 = 0.29699
```

Current experiment:

```text
RareOS v1
qilie ×4
huashang ×2
```
