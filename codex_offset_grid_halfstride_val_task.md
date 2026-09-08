# Codex 实验任务书：RareOS Half-Stride Offset Grid — 474-Val-only Tile Geometry Study

> 项目：2026AIC「AI＋钢铁」板材表面缺陷检测  
> 仓库：`/root/autodl-tmp/steel_defect`  
> 当前正式线上最好：`98.19 / Recall=0.9819 / TP=976 / FN=18`  
> 当前正式线上方案：  
> `RareOS v1 + High-Recall + Selective HFlip + Zonglie Cross-Tile Stitching + Corrected Baseline HR Global Complement u1e3_g060`
>
> 本任务性质：**只研究新的 tile placement / crop context，不改变图像方向、不改变模型、不改变 tile size / stride / conf。**
>
> **严格 Val-only：禁止运行 Test / Hidden，禁止读取 Test/Hidden GT，禁止生成 Test submission，禁止上传比赛平台。**

---

# 0. 当前项目状态：必须从这里继续

## 0.1 当前正式线上最好

```text
Score      = 98.19
Recall     = 0.9819
Precision  = 0.0003
F1         = 0.0006
mAP@0.5    = 0.4514

TP = 976
FP = 3,391,975
FN = 18
```

当前正式线上方案：

```text
RareOS v1
+
High-Recall tiled inference
+
Selective HFlip {0,2,3,4,7}
+
Zonglie Cross-Tile Stitching
+
Corrected Baseline HR Global Complement u1e3_g060
```

其中 Baseline complement 已经真实跨 Hidden：

```text
98.09 / 975 TP / 19 FN
→
98.19 / 976 TP / 18 FN
```

所以该模块继续保留。

---

## 0.2 当前正式 Val baseline

当前正式 98.19 pipeline 对应 Val：

```text
TP = 831
FN = 14
```

历史已验证：

```text
float      = 831/14
roundtrip  = 831/14
```

上一次 VFlip factorial report 记录的 BASE：

```text
final detections = 2,340,806
stitched boxes   = 4,899
```

本任务开始时必须重新核对 persisted artifact / 正式实现。

如果当前代码无法复现：

```text
831/14
```

则：

```text
STOP
```

不得开始新 GPU inference。

---

# 0.3 VFlip 已正式冻结

上一阶段 2×2 factorial：

```text
D = Original grid + tile-local VFlip
G = Mirrored-Y grid + original orientation
F = Mirrored-Y grid + tile-local VFlip
```

Val：

```text
D: +2
G: +1
F: +3 float / +4 roundtrip
```

F 满足当时 GO：

```text
3 images
3 groups
3 classes
0 regression
```

但 Hidden：

```text
BASE:
976 TP / 18 FN / 3,391,975 FP

BASE + F:
976 TP / 18 FN / 5,052,528 FP

ΔTP = 0
ΔFN = 0
ΔFP = +1,660,553
ΔScore = 0
```

因此：

> **VFlip 路线正式冻结。**

禁止本任务：

```text
继续 F
运行 D/G Test
D+F
G+F
D+G+F
VFlip gate tuning
VFlip conf tuning
180° rotation
```

---

# 1. 本阶段科学问题

现在只研究：

> **同一个 RareOS 模型、同一个图像方向、同一个 tile size、同一个 stride、同一个 inference threshold，仅改变 tile 在原图中的落点，能否产生当前正式 BASE 没有的有效定位 / fragmentation recovery？**

具体拆成三个预定义 observation grids：

```text
X = 横向 half-stride phase offset
Y = 纵向 half-stride phase offset
XY = 横纵同时 half-stride phase offset
```

不翻转像素。

不改变尺度。

不改变模型。

---

# 2. 为什么优先做 offset-grid，而不是立即 multi-scale

本任务必须保持以下变量完全冻结：

```text
RareOS weights
image orientation
tile size
imgsz
stride
conf
tile NMS IoU
max_det
half precision
```

唯一改变：

```text
tile window origin
```

这样若出现收益，才能归因于：

```text
boundary placement
crop context
fragmentation pattern
```

而不是：

```text
orientation
resampling scale
更多像素分辨率
模型变化
```

multi-scale / tile-size 变化留到 offset-grid 之后。

---

# 3. 冻结的 inference 参数

RareOS：

```text
runs/rareos/
yolo26m_tiles1280_rareos_v1_e80_b6_seed2026/
weights/best.pt
```

冻结参数：

```text
tile_size = 1280
imgsz     = 1280
stride    = 768

conf      = 1e-5
tile_iou  = 0.60
max_det   = 1000

batch     = 6
half      = True
```

如果显存 OOM：

```text
只允许降低 batch
```

禁止改变其它数值参数。

---

# 4. Half-stride offset 固定定义

冻结：

```python
OFFSET = stride // 2
```

当前：

```text
stride = 768
OFFSET = 384
```

**禁止 sweep offset。**

不得试：

```text
256
320
400
512
```

只允许 384。

---

# 5. 新 grid 的通用生成公式

必须实现一个明确、GT-independent、任意图像尺寸可用的函数。

对某一维：

```text
length = L
tile   = T
stride = S
offset = O = S // 2
last   = max(0, L - T)
```

## 5.1 Original starts

必须直接复用正式 `get_starts()`。

不得重新近似实现。

## 5.2 Edge-anchored half-offset starts

定义：

```python
def get_offset_starts(length, tile, stride, offset):
    if length <= tile:
        return [0]

    last = length - tile
    starts = {0, last}

    s = offset
    while s < last:
        starts.add(s)
        s += stride

    return sorted(starts)
```

含义：

```text
保留 0 与 last 两个边界 anchor
+
内部采用 half-stride phase
```

这样：

```text
保持完整图像覆盖
同时让绝大多数内部目标落到不同 crop context
```

不要根据某张图的 GT 调整 starts。

---

# 6. 典型几何 sanity example

对于：

```text
W = 4096
H = 3000
T = 1280
S = 768
O = 384
```

正式 Original：

```text
X_O:
0, 768, 1536, 2304, 2816

Y_O:
0, 768, 1536, 1720
```

Half-offset：

```text
X_half:
0, 384, 1152, 1920, 2688, 2816

Y_half:
0, 384, 1152, 1720
```

这只是测试样例。

**禁止在正式代码中硬编码这些数字。**

---

# 7. 三个冻结 view

## X

```text
x starts = half-offset x
y starts = Original y
orientation = original
```

## Y

```text
x starts = Original x
y starts = half-offset y
orientation = original
```

## XY

```text
x starts = half-offset x
y starts = half-offset y
orientation = original
```

本轮只评估：

```text
BASE + X
BASE + Y
BASE + XY
```

---

# 8. 明确禁止的扩展

不要创建：

```text
quarter offset
negative offset
random offset
adaptive offset
FN-aware offset
aspect-aware offset
class-aware offset
```

看到 X/Y/XY 结果后也禁止追加：

```text
X+Y
X+XY
Y+XY
X+Y+XY
```

---

# 9. 实验严格边界

允许：

```text
474 Val original images
RareOS GPU inference
Val GT 在 cache 完成后的 evaluation / attribution
```

禁止：

```text
Test inference
Hidden inference
Test/Hidden GT
Test submission
competition upload

新模型训练
模型权重修改

F/D/G VFlip
180°
multi-scale
new tile_size
new stride

针对 14 FN 改 grid
image-specific rule
class-specific grid
GT-aware proposal selection
```

---

# 10. Step 0：环境与当前 baseline reproduction

先执行：

```bash
cd /root/autodl-tmp/steel_defect

git rev-parse HEAD
git status --short
python --version
nvidia-smi
```

记录：

```text
Repo HEAD
workspace dirty
Python
PyTorch
Torchvision
Ultralytics
GPU
RareOS weight path
RareOS weight SHA256
```

检查当前 scripts 编号和已有 artifact，尤其是：

```text
corrected Baseline HR
VFlip factorial Val
VFlip F Test probe
```

不要假设编号。

---

# 11. Baseline 必须复现

当前正式 Val BASE：

```text
831 / 14
```

必须重新执行或从 hash-verified persisted artifacts 验证：

```text
float = 831/14
roundtrip = 831/14
```

建议同时核对：

```text
final detections = 2,340,806
stitched boxes   = 4,899
```

若代码版本导致 detection 数略有差异但 TP/FN 一致：

```text
必须先解释并 reconcile
```

若 TP/FN 不一致：

```text
STOP
```

不得继续。

---

# 12. 新增统一 offset-grid cache generator

优先从正式 RareOS O inference generator refactor。

不要复制三份大代码。

建议新增：

```text
<next_free_number>_cache_offset_grid_val.py
```

接口：

```text
--grid x
--grid y
--grid xy
--output-dir ...
```

三个 grid 唯一差异：

```text
x starts
y starts
```

---

# 13. Cache schema 必须继续保持 14 列

```text
col 0      = class
col 1      = score
col 2:6    = GLOBAL xyxy
col 6:10   = tile_x,tile_y,valid_w,valid_h
col 10:14  = tile-local xyxy
```

必须满足：

```text
global
=
local
+
(tile_x, tile_y, tile_x, tile_y)
```

误差：

```text
<= 1e-4
```

所有原图级几何：

```text
只能使用 2:6
```

---

# 14. Geometry test suite：GPU 前必须 PASS

建议新增：

```text
<next_free_number>_test_offset_grid_geometry.py
```

至少测试：

## 14.1 Original get_starts reuse

```text
Original starts 与 Script14 正式实现完全一致
```

## 14.2 Offset formula synthetic

4096 / 3000 时：

```text
X_half:
0,384,1152,1920,2688,2816

Y_half:
0,384,1152,1720
```

## 14.3 Small image

若：

```text
L <= tile
```

必须：

```text
starts = [0]
```

## 14.4 No duplicate origins

每张图每个 view：

```text
(tile_x,tile_y)
```

不得重复。

## 14.5 Complete coverage

X/Y/XY 单独必须覆盖整张图。

## 14.6 Pixel identity

固定按 filename 排序选：

```text
1
100
200
300
474
```

验证 generator crop 与直接 numpy slicing 一致。

## 14.7 Global/local invariant

```text
global == local + origin
```

全部 PASS 才允许 GPU。

---

# 15. Tile-origin novelty audit

对每个 image / view：

```text
Original origin set O
new origin set V
```

报告：

```text
|O|
|V|
|O ∩ V|
|V - O|
origin Jaccard
new-origin fraction
tile-count ratio V/O
```

汇总：

```text
mean / p50 / p90 / max
```

目的：

> 证明新 view 真正改变了 crop placement。

---

# 16. Coverage multiplicity audit

由于 edge anchors 可能导致 X/XY 比 Original 多 tile：

必须报告：

```text
Original tiles
X tiles
Y tiles
XY tiles
```

至少：

```text
mean
median
p95
max
ratio to O
```

注意：

> 这不是严格相同计算预算试验。  
> 新 grid 改变 crop phase，同时可能增加观察窗口数。

---

# 17. 三个新 cache 输出

建议：

```text
results/offset_grid_halfstride/
  x/cache/
  y/cache/
  xy/cache/
```

每套：

```text
474 NPZ
manifest.json
```

---

# 18. Manifest 必须记录

至少：

```text
view_id
grid_mode
offset=384
offset_definition=stride//2
edge_anchor=True

weights
weights_sha256
repo_head

tile_size
imgsz
stride
conf
tile_iou
max_det
batch
half

images
candidate_rows
tile_rows
runtime
GPU allocated peak
GPU reserved peak

schema
global_box_slice=[2,6]
local_box_slice=[10,14]
```

---

# 19. GPU 执行顺序

依次：

```text
X
→ cache audit

Y
→ cache audit

XY
→ cache audit
```

每个 view 完成后验证：

```text
474 NPZ
image set exact
schema valid
coord residual <= 1e-4
NaN/Inf = 0
class valid
score valid
bbox valid
manifest counts reconcile
```

任一 FAIL：

```text
STOP
```

---

# 20. GPU 长任务安全运行

主 runner 建议：

```text
<next_free_number>_run_offset_grid_val.sh
```

必须：

```bash
set -euo pipefail
```

推荐：

```bash
cd /root/autodl-tmp/steel_defect
mkdir -p logs

nohup env   PYTHONUNBUFFERED=1   OMP_NUM_THREADS=1   OPENBLAS_NUM_THREADS=1   MKL_NUM_THREADS=1   NUMEXPR_NUM_THREADS=1   MALLOC_ARENA_MAX=2   CUDA_VISIBLE_DEVICES=0   bash <actual_runner>   > logs/offset_grid_halfstride_val.log 2>&1 < /dev/null &

echo $!
```

---

# 21. Cache-side candidate distribution audit

三套完成后，先不看 GT。

报告：

```text
candidate rows
candidate/image
candidate/tile
per-class candidate counts
per-image p50/p90/p95/p99/max
top2 concentration
```

与 Original O 比：

```text
candidate ratio
tile-count ratio
candidate/tile ratio
```

若：

```text
candidate/tile
```

出现数量级异常：

```text
调查实现
```

不要改 conf。

---

# 22. GT 使用边界

直到：

```text
X/Y/XY cache 全完成
+
cache audits PASS
+
manifest 已冻结
```

之前：

```text
不得读取 current 14 FN
不得读取 GT 做 grid 调整
```

之后才允许 GT 用于科学评估。

---

# 23. 第一层：proposal-level complementarity

对当前 BASE 剩余：

```text
14 FN
```

分别对 X/Y/XY 统计：

```text
same-class best IoU
best score
best box

IoU >= .30
IoU >= .40
IoU >= .50
```

direct rescue：

```text
BASE = FN
and
new-view raw same-class proposal IoU >= .50
```

报告：

```text
direct rescue count
images
groups
classes
failure types
```

这只是 oracle diagnostic。

不能用于 selector。

---

# 24. Proposal rescue overlap

定义：

```text
R_X
R_Y
R_XY
```

报告：

```text
X-only
Y-only
XY-only

X∩Y
X∩XY
Y∩XY
X∩Y∩XY
```

以及：

```text
J(X,Y)
J(X,XY)
J(Y,XY)
```

---

# 25. 第二层：GT-independent Exact Combo

只测试：

```text
BASE + X
BASE + Y
BASE + XY
```

BASE：

```text
RareOS O
+
corrected Baseline HR u1e3_g060
+
Selective HFlip

→ class-aware Global NMS
→ Zonglie Stitch
```

新 offset view：

```text
all classes
all candidates
```

禁止新增：

```text
score gate
class gate
overlap selector
Top-K
FN-specific filter
```

---

# 26. Input source order 必须冻结

读取当前 BASE formal source order。

统一：

```text
BASE formal active sources
→ new offset view last
```

不要因为 NMS tie / greedy behavior 改顺序。

---

# 27. Exact pipeline 必须复用正式实现

禁止另写近似：

```text
NMS
matcher
stitch
submission_row
```

每个 config：

```text
global NMS
→ Zonglie Stitch
→ float matcher
→ submission roundtrip
→ matcher
```

---

# 28. 必须统计 rescue / regression

相对：

```text
831/14
```

每个报告：

```text
TP
FN
Recall

rescued
regressed
net_gain
```

定义：

```text
rescued:
BASE FN → new TP

regressed:
BASE TP → new FN
```

保存逐 GT attribution。

---

# 29. Stitch provenance

必须追踪：

```text
new stitched boxes
new-view participation
mixed-source stitch
```

对每个 rescued GT 区分：

```text
direct raw proposal rescue
pre-stitch exact rescue
stitch-only rescue
NMS/replacement-mediated rescue
```

不能只看 raw IoU。

---

# 30. Geometric exposure audit：必须做

对于当前 14 FN，每个 GT 纯几何计算：

## 30.1 最大可见比例

对 Original / X / Y / XY：

```text
GT 与每个 tile 的 intersection area / GT area
```

取：

```text
max_visible_fraction
```

## 30.2 Tile intersection count

```text
num_intersecting_tiles
```

## 30.3 Boundary distance

计算 GT center / edges 到：

```text
nearest internal vertical tile boundary
nearest internal horizontal tile boundary
```

的距离。

## 30.4 Rescued GT 对照

报告：

```text
O max visible
new max visible
delta

O boundary distance
new boundary distance

O intersecting tiles
new intersecting tiles
```

这是 post-hoc mechanism analysis，不能用于构建 grid。

---

# 31. Roundtrip evaluation

每个 Exact Combo：

```text
float final detections
→ formal submission_row()
→ JSON serialize
→ read back
→ same matcher
```

报告：

```text
float TP/FN
roundtrip TP/FN
status changes
```

representation-only TP 单独标记。

---

# 32. Group robustness

读取：

```text
splits/sample_assignments.csv
```

报告：

```text
rescue images
rescue groups
rescue classes
failure clusters
```

移除最大收益 group：

```text
rescued
regressed
net
```

ties 取最保守结果。

---

# 33. Deterministic 3-fold robustness

由于固定 Val 已被长期观察，本轮增加预定义 dispersion check。

```python
fold = int(
    sha256(group_id.encode("utf-8")).hexdigest()[:8],
    16
) % 3
```

必须在查看结果前写死。

报告：

```text
fold0 rescued/regressed/net
fold1 rescued/regressed/net
fold2 rescued/regressed/net
```

注意：

> 这不是独立 validation，只是 gain dispersion check。

---

# 34. 为什么本轮 GO 标准提高

历史：

```text
Fusion          Val +4 → Hidden +0
Confusion       Val +2 → Hidden +0
wrong-coordinate Baseline Val +5 → Hidden +0
corrected Baseline Val +7 → Hidden +1
VFlip F         Val +3/+4 → Hidden +0
```

因此：

> `+3 / 3 groups / 0 regression` 已经证明不足以作为强 Hidden 泛化证据。

---

# 35. 新的强 GO 标准

一个 offset view 未来才允许讨论 Test，必须同时满足：

```text
1. float Exact Final Combo net_gain >= +4

2. roundtrip net_gain >= +4

3. float rescued >= 4 different images

4. float rescued >= 4 grouped-split groups

5. rescued >= 2 classes

6. rescued >= 2 failure clusters

7. regression <= 1

8. remove-largest-rescue-group 后：
   float net_gain >= +2

9. deterministic 3-fold：
   至少 2 folds 的 net_gain > 0

10. deterministic 3-fold：
    不允许任何 fold net_gain < 0

11. 收益不能全部是 representation-only

12. cache / geometry / provenance 全 PASS
```

特别：

```text
float +3
roundtrip +4
```

仍为：

```text
NO-GO
```

---

# 36. 禁止 offset sweep

即使结果：

```text
X +3
Y +2
XY +3
```

也禁止：

```text
offset=320
offset=448
offset=512
```

不要为了凑 +4 调参。

---

# 37. 禁止多 view 组合

本任务不执行：

```text
BASE + X + Y
BASE + X + XY
BASE + Y + XY
BASE + X + Y + XY
```

即使逐 GT rescue 互补。

原因：

```text
当前 14 FN 已被反复观察
组合会高度适应固定 Val
```

---

# 38. Mechanism classification

最终只能给一个主标签：

```text
X_BOUNDARY_DOMINANT
Y_BOUNDARY_DOMINANT
XY_CONTEXT_DOMINANT
MIXED_AXIS_GEOMETRY
NO_USEFUL_OFFSET_GRID
```

判断不能只看 TP 最大。

还要看：

```text
direct vs stitch-only
GT aspect / orientation
boundary distance
visible fraction
rescue overlap
```

---

# 39. 分类解释

## X_BOUNDARY_DOMINANT

横向 crop placement 明显主导。

## Y_BOUNDARY_DOMINANT

纵向 boundary / fragmentation 明显主导。

## XY_CONTEXT_DOMINANT

XY 单独明显强于 X/Y，但不要声称严格统计 interaction。

## MIXED_AXIS_GEOMETRY

X 与 Y 均有独立有效 rescue，XY 主要覆盖两者。

## NO_USEFUL_OFFSET_GRID

三个都不满足新的强 GO。

---

# 40. 建议脚本

先检查现有 scripts 编号。

使用 next free numbers，例如：

```text
*_cache_offset_grid_val.py
*_test_offset_grid_geometry.py
*_audit_offset_grid_candidates.py
*_eval_offset_grid_exact_combo.py
*_audit_offset_grid_mechanism.py
*_report_offset_grid.py
*_run_offset_grid_val.sh
```

不要覆盖历史实验。

---

# 41. 输出目录

统一：

```text
results/offset_grid_halfstride/
```

建议：

```text
x/cache/
y/cache/
xy/cache/

exact_combo/
  baseline/
  x/
  y/
  xy/

audit/
  execution_context.json
  geometry_tests.json
  tile_origin_novelty.csv
  tile_count_audit.csv
  candidate_distribution.csv
  direct_rescue.csv
  rescue_overlap.json
  gt_geometry_exposure.csv
  stitch_provenance.csv
  group_audit.json
  fold_audit.json
  independent_verification.json

summary.csv
report.md
logs/
```

---

# 42. summary.csv 最少字段

```text
config
offset_x
offset_y

images
tile_rows
tile_count_ratio_to_o

candidate_rows
candidate_per_image
candidate_per_tile

runtime_seconds
peak_gpu_allocated_mb
peak_gpu_reserved_mb

origin_jaccard_to_o
new_origin_fraction

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

largest_group_contribution
net_gain_without_largest_group

positive_folds
negative_folds

new_stitched
new_view_stitch_participation
mixed_source_stitched
stitch_only_rescues
```

---

# 43. 必须生成逐 GT 表

仅对：

```text
当前 14 FN
+
任何 regression GT
```

报告：

| image | group | class | failure | BASE IoU | X IoU | Y IoU | XY IoU | O max visible | new max visible | exact rescue source |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---|

另外保存：

```text
bbox
aspect
area
boundary distances
tile intersection counts
```

---

# 44. 必须生成 rescue overlap

| set | count |
|---|---:|
| X | |
| Y | |
| XY | |
| X-only | |
| Y-only | |
| XY-only | |
| X∩Y | |
| X∩XY | |
| Y∩XY | |
| X∩Y∩XY | |

同时给：

```text
proposal rescue overlap
exact rescue overlap
Jaccard
```

---

# 45. Independent verification

完成 report 前，另写一个只读 verifier。

验证：

```text
baseline 831/14
474×3 caches
manifest counts
global/local residual
exact metrics files
roundtrip files
source rows
stitch provenance
group mapping
fold mapping
summary/report reconciliation
```

输出：

```text
audit/independent_verification.json
```

不得只依赖生成器自报 PASS。

---

# 46. 最终 report.md 格式

## A. Execution status

```text
Repo HEAD
workspace dirty
Python/PyTorch/Ultralytics
GPU
weights + SHA

Test/Hidden run = NO
Test/Hidden GT accessed = NO
competition submission = NO
```

## B. Formal BASE reproduction

```text
Expected 831/14
Float actual
Roundtrip actual
Final detection count
Stitched
PASS / FAIL
```

## C. Grid definitions

```text
Original
OFFSET=384
edge-anchor formula
X
Y
XY
```

## D. Geometry tests

逐项 PASS / FAIL。

## E. Tile-origin novelty

| view | tiles/image | ratio to O | origin Jaccard | new-origin fraction |
|---|---:|---:|---:|---:|

## F. Cache summary

| view | images | candidates | cand/tile | coord residual | runtime | GPU peak |
|---|---:|---:|---:|---:|---:|---:|

## G. Proposal rescue

| view | IoU≥.30 | IoU≥.40 | direct | images | groups | classes |
|---|---:|---:|---:|---:|---:|---:|

## H. Exact Combo

| config | float TP/FN | roundtrip TP/FN | rescued | regressed | net | images | groups | classes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|

## I. Rescue overlap

proposal + exact overlap / Jaccard。

## J. Geometry mechanism

```text
visible fraction change
boundary distance change
tile intersection count
aspect
direct / stitch-only
```

## K. Stitch provenance

```text
new stitched
new-view participation
mixed-source
stitch-only rescues
```

## L. Group robustness

```text
largest group
contribution
net after removal
```

## M. Deterministic fold robustness

| config | fold0 net | fold1 net | fold2 net | positive folds | negative folds |
|---|---:|---:|---:|---:|---:|

## N. Mechanism conclusion

只能：

```text
X_BOUNDARY_DOMINANT
Y_BOUNDARY_DOMINANT
XY_CONTEXT_DOMINANT
MIXED_AXIS_GEOMETRY
NO_USEFUL_OFFSET_GRID
```

## O. Strong GO / NO-GO

逐条核对 12 条强标准。

如果没有 view 全 PASS：

```text
NO-GO
recommended_view_for_future_test = NONE
```

如果某一个 PASS：

```text
GO
recommended_view_for_future_test = X|Y|XY
```

但：

```text
禁止本任务运行 Test。
```

---

# 47. Codex 最终聊天汇报格式

必须返回：

```text
1. BASE：
   float / roundtrip
   detection count
   PASS?

2. Geometry：
   OFFSET
   X/Y/XY starts sanity
   tests PASS?

3. X：
   tiles
   candidates
   direct rescue
   exact TP/FN
   roundtrip
   rescued/regressed/net
   images/groups/classes
   largest-group removal
   fold nets
   stitch-only

4. Y：
   同上

5. XY：
   同上

6. proposal rescue overlap
7. exact rescue overlap

8. geometric exposure：
   rescued GT 是否改善
   visible fraction / boundary context

9. regressions

10. roundtrip-only status changes

11. candidate/tile distribution 是否异常

12. runtime / GPU peak / OOM

13. independent verification
    PASS / FAIL

14. mechanism classification

15. strong GO / NO-GO

16. recommended_view_for_future_test
    X / Y / XY / NONE

17. 明确：
    Test/Hidden run = NO
    competition submission = NO
```

---

# 48. 特别冻结解释

如果某 view 只有：

```text
+3
```

即使：

```text
3 images
3 groups
0 regression
```

仍然：

```text
NO-GO
```

不要调参补到 +4。

如果某 view：

```text
+4
```

但集中在 1~2 groups：

```text
NO-GO
```

如果 XY +4、X/Y 各 +2：

```text
可标 XY_CONTEXT_DOMINANT
```

但不组合 X+Y。

如果 X/Y/XY 都 +4：

优先：

```text
更少 regression
更多 groups
remove-largest 后更强
更多 positive folds
更少候选 / 更简单 geometry
```

---

# 49. 全部 NO-GO 时的下一阶段

如果三个都 NO-GO：

```text
停止 offset-grid。
```

下一阶段才考虑：

```text
解耦 tile_size 与 imgsz 的 multi-scale/context 实验
```

例如未来分别回答：

```text
更大 crop context 是否有帮助
vs
模型输入采样尺度是否有帮助
```

**本任务不执行。**

---

# 50. 最终原则

本轮不是：

> “围绕剩余 14 FN 再找一个能多救几项的网格。”

而是：

> **在完全冻结模型、方向、尺度、stride 和阈值的情况下，用预定义的 half-stride X/Y/XY phase shift，验证 crop placement / boundary context 是否构成真正新的、分散的 observation space。**

历史已经证明：

```text
Val +3 / 3 groups / 0 regression
仍可能 Hidden +0
```

所以本轮必须：

```text
更少调参
更强预注册
更强 group/fold 分布要求
更重机制归因
```

完成：

```text
X/Y/XY caches
Exact Combo
roundtrip
geometry exposure
group/fold robustness
independent verification
report.md
```

后立即停止。

**禁止自行进入 Test / Hidden。**
