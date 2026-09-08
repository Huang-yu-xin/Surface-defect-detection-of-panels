# Codex 实验任务书：RareOS Vertical Flip 2×2 Factorial Val-only Mechanism Study

> 项目：2026AIC「AI＋钢铁」板材表面缺陷检测  
> 仓库：`/root/autodl-tmp/steel_defect`  
> 当前正式线上最好：`98.19 / Recall=0.9819 / TP=976 / FN=18`  
> 当前正式线上方案：  
> `RareOS v1 + High-Recall + Selective HFlip + Zonglie Cross-Tile Stitching + Corrected Baseline HR Global Complement u1e3_g060`
>
> 本任务性质：**RareOS VFlip 的 474-Val-only 新 observation-space 实验。**
>
> **严格禁止运行 Test / Hidden，禁止生成 Hidden submission，禁止上传比赛平台。**

---

# 0. 当前项目状态与本任务动机

原正式线上方案：

```text
RareOS v1
+ High-Recall
+ Selective HFlip
+ Zonglie Cross-Tile Stitching

Hidden:
975 TP / 19 FN
Score = 98.09
```

之后发现历史 Baseline HR 脚本 35–38 曾误用 tile-local `10:14` 进行原图级几何比较。

修复后，corrected Baseline HR global complement：

```text
Val:
824/21 → 831/14
+7 TP / 0 regression

4 images
4 groups
3 classes
3 failure clusters
roundtrip = 831/14
```

冻结方案：

```text
u1e3_g060
min_score     = 1e-5
upper_score   = 1e-3
overlap_gate  = 0.60
global xyxy   = 2:6
HFlip active classes = {0,2,3,4,7}
```

随后 Hidden：

```text
975/19
→
976/18

Score:
98.09 → 98.19

ΔTP = +1
ΔFN = -1
```

因此：

> Corrected Baseline HR global complement 已经证明存在真实但较弱的跨分布迁移。

**禁止继续提交其它 Baseline HR 参数。**

下一优先机制正式转入：

```text
RareOS Vertical Flip
```

但本任务不能只做一个模糊的“整图 VFlip”。

历史 HFlip 的实现是：

```text
整图翻转
→ 再 tiled inference
→ 映射回原图
```

这会同时改变：

```text
1. 图像方向
2. 目标所处 tile grid / boundary / context
```

因此 VFlip 必须拆成可解释的 2×2 factorial experiment。

---

# 1. 核心科学问题

本任务要回答：

> RareOS 在垂直方向的新预测信息，究竟来自：
>
> 1. **纯方向变化（orientation）**
> 2. **纯 tile-grid / boundary / context 变化**
> 3. **方向 × grid 的组合**
> 4. 或者 VFlip 根本没有新的有效 observation space

最终不是只给出“VFlip +N TP”，而必须解释：

```text
direction effect
grid effect
direction × grid interaction
```

哪个更重要。

---

# 2. 2×2 Factorial 设计

定义两个二元因素：

```text
Factor A: Vertical orientation
0 = 原方向
1 = tile 内 vertical flip

Factor B: Y-grid
0 = Original grid
1 = Mirrored-Y grid
```

四个 cell：

| Cell | Orientation | Y-grid | 状态 |
|---|---|---|---|
| O | original | original grid | 已有 RareOS Original cache |
| D | VFlip | original grid | **新推理：direction-only** |
| G | original | mirrored-Y grid | **新推理：grid-only** |
| F | VFlip | mirrored-Y grid | **新推理：full-VFlip-equivalent** |

对应：

```text
O = current RareOS Original

D = fixed-grid tile-local VFlip
    → 只改变方向，不改变裁切窗口

G = original orientation + mirrored-Y grid
    → 只改变 tile boundary/context，不改变方向

F = mirrored-Y grid + tile-local VFlip
    → 等价于整图 VFlip 后按标准 grid tiled inference
```

---

# 3. 为什么 F 应等价于“整图 VFlip”

对图像高度 `H`，标准 grid 中某个翻转图 tile：

```text
flip-space y_start = s
valid_h            = h
```

映射回原图后的窗口 top：

```text
y_original = H - (s + h)
```

因此：

```text
整图 VFlip
→ standard grid crop
```

映射回原坐标后，等价于：

```text
原图 mirrored-Y grid crop
→ tile-local vertical flip
```

对于典型：

```text
H = 3000
tile = 1280
stride = 768
```

Original y starts：

```text
0, 768, 1536, 1720
```

Mirrored-Y mapped starts：

```text
0, 184, 952, 1720
```

但：

> **实现必须使用通用公式，不得把这四个数字硬编码。**

必须支持实际图片高度和 `valid_h`。

---

# 4. 实验严格边界

本任务：

```text
Val only
474 original Val images
RareOS model only
```

严格禁止：

```text
Test inference
Hidden inference
读取 Hidden/Test GT
生成 Test submission
比赛平台提交

训练新模型
修改 RareOS 权重
继续 Baseline gate tuning
针对当前 14 FN 写 image-specific 规则
针对某个 rescue image 修改 grid
根据 GT 选择 VFlip 类别
```

GT 只允许在：

```text
完整 prediction cache 已生成以后
```

用于：

```text
TP/FN evaluation
rescue attribution
regression attribution
class/group/failure 分析
```

---

# 5. 当前 Val baseline 已经变化

**本任务不能再以旧的 `824/21` 为正式比较基线。**

当前正式方案对应的 Val baseline 是：

```text
831 TP
14 FN
```

即：

```text
RareOS O
+
Corrected Baseline HR u1e3_g060
+
Selective HFlip
→ Global NMS
→ Zonglie Stitch
```

所以所有 VFlip exact-combo 结果必须比较：

```text
831/14
```

而不是：

```text
824/21
```

---

# 6. Step 0：必须先复现当前 831/14 baseline

任何新 GPU inference 前，先检查当前 repo / artifacts。

执行：

```bash
git rev-parse HEAD
git status --short
python --version
nvidia-smi
```

检查最近新增的 corrected Baseline HR 脚本和产物。

优先读取：

```text
results/baseline_complementarity/global_exact_final_combo/
```

以及现有：

```text
scripts/41_*
scripts/42_*
scripts/43_*
scripts/44_*
```

不要猜实际 cache 子目录。

必须通过现有正式实现重新复现：

```text
Current Val Exact Final Combo:
831 / 14
```

并同时复现：

```text
submission-roundtrip:
831 / 14
```

如果不是：

```text
831/14
```

则：

```text
STOP
```

查明差异，不得继续 VFlip 实验。

---

# 7. RareOS inference 参数必须冻结

新 D / G / F 三个 view 全部使用 RareOS v1：

```text
weights:
runs/rareos/
yolo26m_tiles1280_rareos_v1_e80_b6_seed2026/
weights/best.pt
```

参数与正式 High-Recall inference 对齐：

```text
tile_size  = 1280
stride     = 768
conf       = 1e-5
tile_iou   = 0.60
max_det    = 1000
batch      = 6（若显存允许）
half       = True
```

不得为了某个 view 改：

```text
conf
IoU
max_det
imgsz
tile size
stride
```

除非现有 RareOS 原始缓存实际 inference 参数与这里不一致。

如果不一致：

> 以已有 RareOS O / H cache manifest 和 script14/15 的真实正式参数为准，并在报告中说明。

---

# 8. 建议实现一个统一 view generator

不要分别复制三份大段 inference 代码。

建议新增：

```text
scripts/49_cache_vflip_factorial_val.py
```

核心参数：

```text
--grid-mode original|mirrored_y
--vertical-flip 0|1
--output-dir ...
```

它统一覆盖：

```text
D:
grid=original
vertical_flip=1

G:
grid=mirrored_y
vertical_flip=0

F:
grid=mirrored_y
vertical_flip=1
```

O 不重新跑，直接使用现有：

```text
results/fn_analysis/cache/
```

---

# 9. Original grid 定义

必须直接复用项目正式 `get_starts()` / tiling 逻辑。

禁止另写“差不多”的 starts。

D cell：

```text
先按 Original grid crop 原图 tile
→ 对 tile pixels 做 vertical flip
→ RareOS inference
→ 将预测 local y 坐标翻回该 tile
→ global = mapped-local + 原 tile origin
```

所以：

```text
D 的 tile origin
必须与 O 完全一致
```

这是 direction-only 的关键 invariant。

---

# 10. Mirrored-Y grid 定义

必须由标准 grid 通过通用公式映射：

```python
mirrored_y = H - (standard_y + valid_h)
```

然后：

```text
去重
排序
保证边缘合法
```

x-grid 保持 Original 不变。

G：

```text
mirrored-Y crop
不翻转 pixels
RareOS inference
正常 local→global
```

F：

```text
mirrored-Y crop
tile-local VFlip
RareOS inference
local y 反变换
local→global
```

---

# 11. Cache schema 必须与现有 14 列完全一致

D/G/F 输出：

```text
col 0      class
col 1      score
col 2:6    GLOBAL xyxy
col 6:10   tile_x,tile_y,valid_w,valid_h
col 10:14  tile-local xyxy（映射回原图方向后的 local box）
```

要求：

```text
global
=
local
+
tile origin
```

误差：

```text
<= 1e-4
```

不要改变 14 列 schema。

如需记录来源：

```text
使用 manifest / sidecar
```

不要添加第 15 列。

---

# 12. 三个输出 cache

建议：

```text
results/vflip_factorial/val_direction/
results/vflip_factorial/val_grid/
results/vflip_factorial/val_full/
```

每个都包含：

```text
474 NPZ
manifest.json
```

manifest 至少记录：

```text
view_id
factor_orientation
factor_grid
weights
repo_head
tile_size
stride
conf
tile_iou
max_det
batch
half
image_count
candidate_rows
runtime
peak GPU memory
created_at
schema
```

---

# 13. 实现正确性测试：GPU 前必须全部 PASS

建议新增：

```text
scripts/50_test_vflip_factorial_geometry.py
```

至少包括：

## 13.1 Original-grid invariant

对 D：

```text
D tile origins == O tile origins
```

必须完全一致。

## 13.2 Grid-only invariant

对 G：

```text
pixels 不发生 vertical flip
只改变 y start
```

## 13.3 Local-coordinate vertical flip roundtrip

给定 tile 高 `h` 和预测：

```text
[x1,y1,x2,y2]
```

VFlip inverse 后：

```text
y1' = h - y2
y2' = h - y1
```

synthetic test 必须 PASS。

## 13.4 Global/local invariant

非零 tile origin：

```text
global == local + origin
```

## 13.5 F 与 literal whole-image VFlip patch equivalence

无需对全部 Val 重跑两套模型。

至少选：

```text
3~5 个非 GT 驱动的 Val images
```

例如固定按文件名排序取：

```text
第 1
第 100
第 200
第 300
第 474
```

比较：

```text
literal whole-image VFlip
→ standard grid tile

vs

original image
→ mirrored-Y tile
→ tile-local VFlip
```

对应 pixel patches 应：

```text
pixel-identical
```

或明确说明边界处理导致的可解释差异。

**该检查不能根据 GT/rescue 图片选样。**

---

# 14. GPU 运行顺序

避免同时占大量缓存/显存。

顺序：

```text
D direction-only
→ cache audit
→ G grid-only
→ cache audit
→ F full VFlip
→ cache audit
```

每个 view 完成后检查：

```text
NPZ count = 474
image set == O
schema valid
coord residual <= 1e-4
NaN/Inf = 0
class valid
score valid
```

如果某个 view audit FAIL：

```text
STOP
```

不要继续后续 view。

---

# 15. GPU 长任务必须安全运行

建议主 runner：

```text
scripts/54_run_vflip_factorial_val.sh
```

GPU inference 阶段应明确选当前 GPU，例如：

```bash
CUDA_VISIBLE_DEVICES=0
```

并：

```bash
set -euo pipefail
```

推荐：

```bash
cd /root/autodl-tmp/steel_defect
mkdir -p logs

nohup env   PYTHONUNBUFFERED=1   OMP_NUM_THREADS=1   OPENBLAS_NUM_THREADS=1   MKL_NUM_THREADS=1   NUMEXPR_NUM_THREADS=1   MALLOC_ARENA_MAX=2   CUDA_VISIBLE_DEVICES=0   bash scripts/54_run_vflip_factorial_val.sh   > logs/vflip_factorial_val.log 2>&1 < /dev/null &

echo $!
```

每个阶段输出：

```text
START
END
STATUS
runtime
GPU peak memory
candidate rows
```

---

# 16. 第一层分析：proposal-space complementarity

三套 cache 全生成以后，再使用 GT。

建议新增：

```text
scripts/51_audit_vflip_factorial_complementarity.py
```

当前 baseline 剩余：

```text
14 FN
```

对每个 view D/G/F，逐 FN 计算：

```text
best same-class candidate IoU
best score
best box
```

报告：

```text
best IoU >= .30
best IoU >= .40
best IoU >= .50

direct rescue count
rescue images
rescue groups
rescue classes
rescue failure types
```

并记录当前 baseline 对同一 GT：

```text
best current Final Combo IoU
```

---

# 17. Direct rescue 的定义

仅用于科学 audit：

```text
current baseline = FN
and
new view has same-class candidate IoU >= 0.5
```

这是 oracle-style diagnostic。

**不能用它来选择 Test box。**

本任务没有 Test，所以这里允许用 Val GT 做归因。

---

# 18. 必须做 rescued-set overlap 分析

定义：

```text
R_D = direction-only direct rescued GT set
R_G = grid-only direct rescued GT set
R_F = full-VFlip direct rescued GT set
```

报告：

```text
|R_D|
|R_G|
|R_F|

D ∩ G
D ∩ F
G ∩ F
D ∩ G ∩ F

D-only
G-only
F-only
```

并给 Jaccard：

```text
J(D,G)
J(D,F)
J(G,F)
```

这是判断机制的核心。

---

# 19. 第二层：GT-independent Exact Final Combo

proposal audit 只是诊断。

最终必须把每个 view 以**统一、GT-independent、all-class**方式加入当前正式 Val pipeline。

不允许：

```text
只加入 rescue 类
只加入某些图片
只加入已知 FN 附近
```

---

# 20. 固定的三个 Exact Combo config

只测试：

```text
BASE + D
BASE + G
BASE + F
```

其中 BASE = 当前：

```text
RareOS O
+
Corrected Baseline HR u1e3_g060
+
Selective HFlip
→ Global NMS
→ Zonglie Stitch
```

新 view：

```text
all classes
all candidates
same inference threshold
```

不新增 score gate。

不新增 overlap gate。

依赖正式：

```text
class-aware global NMS
```

控制重复。

**禁止为了 Val 14 FN 再设计 VFlip-specific selector。**

---

# 21. Exact Combo pipeline 必须复用正式实现

对每个：

```text
BASE + D
BASE + G
BASE + F
```

执行：

```text
Current Original+Baseline complement
+
Selective HFlip
+
new VFlip-factorial view

→ class-aware Global NMS
→ Zonglie Cross-Tile Stitch
→ Val matcher
```

必须复用现有正式：

```text
NMS
HFlip class behavior
Zonglie stitch
submission_row / roundtrip
matcher
```

不要重写一个近似版本。

---

# 22. Stitching provenance audit 很重要

因为新 view 有不同 tile grid，纵裂 stitching 可能连接：

```text
O
H
Baseline
D/G/F
```

不同来源的 segment。

因此需要在 sidecar provenance 中追踪：

```text
view_id
tile origin
class
```

报告至少：

```text
新增 stitched boxes
由 new view 参与的 stitched boxes
mixed-source stitched boxes
```

并对最终 rescued GT 归因：

```text
pre-stitch 已可 rescue
还是
只有 stitch 后 rescue
```

不要因为 mixed-source 就禁止它。

正式 exact combo 可以允许正常 pipeline interaction。

但科学报告必须区分：

```text
direct observation gain
vs
stitch-mediated gain
```

---

# 23. 每个 Exact Combo 必须同时统计 rescue 与 regression

基准：

```text
831/14
```

每个新 config 报告：

```text
TP
FN
Recall

rescued
regressed
net_gain = rescued - regressed

rescue images
rescue groups
rescue classes
failure clusters
```

逐 GT 保存：

```text
image
group
class
failure
GT bbox
baseline status
new status
baseline best IoU
new best IoU
new source
pre/post stitch attribution
```

---

# 24. Submission-roundtrip 仍然必须做

虽然本任务不生成 Test submission，但 Val 仍要验证最终表示稳定性。

每个 config：

```text
float final detections
→ 正式 submission_row()
→ JSON serialize
→ read back
→ same matcher
```

报告：

```text
float TP/FN
roundtrip TP/FN
delta
GT status changes
```

如果 float 有收益但 roundtrip 丢失：

```text
该收益不能进入 GO
```

---

# 25. Group audit

读取：

```text
splits/sample_assignments.csv
```

使用现有 grouped-split unit。

报告：

```text
unique rescue images
unique rescue groups
rescues by group
rescues by class
rescues by failure
```

以及：

```text
remove largest rescue group
→ rescued
→ regressed
→ net_gain
```

不要把 group 夸大为已证实真实生产线 ID。

它只是本项目 grouped split 的独立性单元。

---

# 26. 机制判定规则

不仅给 GO/NO-GO，还必须分类。

## 26.1 Direction-dominant

如果：

```text
D exact net gain >= 3
且
D 的 unique rescues 明显存在
```

说明：

```text
vertical orientation
```

本身产生互补信息。

下一方向优先：

```text
180° / other orientation views
```

## 26.2 Grid-dominant

如果：

```text
G exact net gain >= 3
```

且 G 的 unique rescue 明显，

说明：

```text
tile boundary / context
```

比方向本身更关键。

下一方向优先：

```text
new Y offsets
new tile grid
multi-grid inference
```

而不是继续旋转。

## 26.3 Interaction-dominant

如果：

```text
F >= 3
但 D < 3 且 G < 3
```

或者 F 有大量：

```text
F-only rescues
```

则：

```text
orientation × grid interaction
```

成立。

下一阶段应研究：

```text
combined view design
```

而不是简单归因给“VFlip”。

## 26.4 No useful VFlip mechanism

如果：

```text
D/G/F 全部 exact net_gain < 3
```

或收益高度集中单 group / regression 明显，

则：

```text
VFlip NO-GO
```

下一优先：

```text
new tile geometry / scale
```

或：

```text
new model diversity
```

---

# 27. 预冻结 GO 标准

某个 view 要进入未来 Test-side 候选，至少满足：

```text
1. Exact Final Combo net_gain >= +3
2. Roundtrip net_gain >= +3
3. rescued >= 3 different images
4. rescued >= 3 grouped-split groups
5. 最好 >= 2 classes
6. 不全部属于单一 failure cluster
7. remove largest rescue group 后 net_gain > 0
8. regression <= 1
9. 不是只靠一个明显异常图片
10. 候选来源与当前 BASE 有明确 mechanism difference
```

注意：

> 本任务即使 GO，也**禁止跑 Test**。

---

# 28. 不做参数稳定性 sweep

VFlip 这轮没有必要继续：

```text
conf sweep
IoU sweep
Top-K sweep
class gate sweep
```

三个 D/G/F cell 本身就是机制对照。

如果某个 view 只有：

```text
+1 / +2
```

不要为了把它调到 +3 而扫参数。

接受原始结果。

---

# 29. 不要组合 D+G+F 去追求更高 Val

本轮主要目标是机制识别。

默认禁止运行：

```text
BASE + D + G
BASE + D + F
BASE + G + F
BASE + D + G + F
```

否则：

```text
解释性下降
候选量暴涨
重新进入固定 Val 拟合
```

只有在最终 report 中可以提出：

```text
未来是否值得组合
```

但本任务不执行。

---

# 30. 建议新增脚本

按当前仓库编号顺延，例如：

```text
49_cache_vflip_factorial_val.py
50_test_vflip_factorial_geometry.py
51_audit_vflip_factorial_complementarity.py
52_eval_vflip_factorial_exact_combo.py
53_report_vflip_factorial.py
54_run_vflip_factorial_val.sh
```

如果编号已占用：

```text
顺延
```

不要覆盖：

```text
41–48
```

历史实验。

---

# 31. 输出目录

统一：

```text
results/vflip_factorial/
```

建议：

```text
results/vflip_factorial/
  direction/
    cache/
    manifest.json

  grid/
    cache/
    manifest.json

  full/
    cache/
    manifest.json

  exact_combo/
    baseline/
    direction/
    grid/
    full/

  audit/
    geometry_tests.json
    direct_rescue.csv
    rescue_overlap.json
    group_audit.csv
    stitch_provenance.csv

  summary.csv
  report.md
  logs/
```

---

# 32. summary.csv 最少字段

```text
config
orientation
grid_mode

candidate_rows
runtime_seconds
peak_gpu_mem_mb

direct_rescue
direct_rescue_images
direct_rescue_groups
direct_rescue_classes

float_tp
float_fn
float_recall

roundtrip_tp
roundtrip_fn
roundtrip_recall

rescued
regressed
net_gain

rescue_images
rescue_groups
rescue_classes
rescue_failure_types

net_gain_without_largest_group

new_stitched
mixed_source_stitched
pre_stitch_rescues
stitch_only_rescues
```

---

# 33. 必须生成 rescue-set overlap 表

`report.md` 中至少：

| set | count |
|---|---:|
| D | |
| G | |
| F | |
| D-only | |
| G-only | |
| F-only | |
| D∩G | |
| D∩F | |
| G∩F | |
| D∩G∩F | |

以及：

```text
J(D,G)
J(D,F)
J(G,F)
```

---

# 34. 必须生成逐 GT attribution

至少：

| image | group | class | failure | BASE IoU | D IoU | G IoU | F IoU | rescued by exact config |
|---|---|---|---|---:|---:|---:|---:|---|

只列：

```text
当前 14 FN
+
任何 regression GT
```

不要输出无关全量 GT。

---

# 35. 最终 report.md 必须包含

## A. Execution status

```text
Repo HEAD
workspace dirty
GPU
RareOS weights
Test/Hidden run = NO
Test/Hidden GT accessed = NO
```

## B. Current baseline reproduction

```text
Expected = 831/14
Float actual
Roundtrip actual
PASS/FAIL
```

## C. Geometry implementation audit

```text
Original-grid invariant
Mirrored-Y formula
local VFlip inverse
global/local invariant
F vs literal whole-image VFlip patch equivalence
```

全部：

```text
PASS / FAIL
```

## D. Cache summary

| view | images | candidates | coord residual | runtime | GPU peak |
|---|---:|---:|---:|---:|---:|

## E. Proposal-level rescue

| view | IoU>=.30 | IoU>=.40 | direct rescue | images | groups | classes |
|---|---:|---:|---:|---:|---:|---:|

## F. Rescue overlap

D/G/F set overlap + Jaccard。

## G. Exact Combo

| config | float TP/FN | roundtrip TP/FN | rescued | regressed | net | images | groups | classes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

## H. Stitch provenance

```text
new stitched
mixed-source stitched
pre-stitch rescue
stitch-only rescue
```

## I. Group robustness

```text
largest group contribution
net after removing largest group
```

## J. Mechanism conclusion

只能选择一个主分类：

```text
DIRECTION_DOMINANT
GRID_DOMINANT
INTERACTION_DOMINANT
MIXED_DIRECTION_AND_GRID
NO_USEFUL_VFLIP_MECHANISM
```

并说明依据。

## K. Final GO / NO-GO

```text
GO
```

或：

```text
NO-GO
```

如果 GO：

```text
recommended_view_for_future_test = D | G | F
```

**但本任务不能运行 Test。**

如果多个 view 都 GO：

优先机制更简单者：

```text
D or G
```

而不是自动选候选最多的 F。

若收益相同：

```text
更少 regression
更分散 group
机制更清晰
```

优先。

---

# 36. Codex 最终聊天汇报格式

任务跑完后，不要只说“完成”。

必须汇报：

```text
1. 831/14 baseline 是否复现
2. D/G/F geometry tests 是否全部 PASS

3. D:
   candidates
   direct rescue
   exact TP/FN
   net gain
   images/groups/classes

4. G:
   candidates
   direct rescue
   exact TP/FN
   net gain
   images/groups/classes

5. F:
   candidates
   direct rescue
   exact TP/FN
   net gain
   images/groups/classes

6. D/G/F rescue overlap
7. regressions
8. stitch-only rescues
9. roundtrip consistency
10. largest-group removal result
11. runtime / GPU peak / 是否 OOM

12. mechanism classification
13. GO / NO-GO

14. 若 GO：
    recommended_view_for_future_test = D/G/F

15. 明确：
    Test/Hidden run = NO
    competition submission = NO
```

---

# 37. 如果任务中途失败

## Baseline 不能复现 831/14

```text
STOP
```

不要跑 VFlip GPU。

## Geometry test FAIL

```text
STOP
```

不要生成科学结论。

## 某 view cache 不完整

```text
STOP
```

不要拿部分 474 Val 做比较。

## GPU OOM

允许：

```text
降低 batch
```

但不得改变：

```text
tile
stride
conf
IoU
max_det
模型
view geometry
```

记录 batch 变化。

## 单个 view 完成后服务器断线

复用已完成 cache。

不要重新跑已验证 cache。

---

# 38. 最终原则

这轮不是：

> “再找一个能把 14 FN 降几项的 TTA。”

而是：

> **用严格的 2×2 factorial design，把 vertical orientation 与 tile-grid/context 的贡献分开，判断当前剩余尾部漏检究竟需要新的方向观察空间，还是新的裁切上下文。**

当前项目已经证明：

```text
更多候选 ≠ 更多 Hidden TP
Val 参数稳定 ≠ Hidden 泛化
```

因此本轮最重要的产物不是单一 TP 数，而是：

```text
可解释的新 information source
+
Exact Final Combo 净收益
+
跨 image/group 的稳健性
```

完成 `report.md` 后停止。

**禁止自行进入 Test / Hidden。**
