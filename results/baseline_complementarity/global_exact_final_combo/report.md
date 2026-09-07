# GT-independent Exact Final Combo Val Report

## Scope and reproducibility

- Repo HEAD: `4dc6e34e861d10b41e1c834b45c027e8f6b12567`
- Working tree had pre-existing and task changes: YES
- GPU used: NO (`CUDA_VISIBLE_DEVICES=""`; CPU-only post-processing)
- Test/Hidden run: NO
- Hidden submission generated: NO
- Builder/evaluator separation: the builder accepts only Original, HFlip, and Baseline prediction caches plus frozen global parameters. It has no annotation, FN-list, rescue-list, failure-type, image-list, or evaluator input. Labels and retrospective CSVs are opened only by the evaluator after each cache exists.
- Cache coordinate / synthetic / active-HFlip assertions: cross_tile_synthetic=PASS, active_hflip_reference=PASS, cache_coordinate_sanity=PASS, builder_has_no_annotation_or_evaluation_argument=PASS
- Assignment schema: image=`image`, group=`group_id`. These groups are repository grouped-split units, not asserted production IDs.

## Baseline reproduction

| expected | float | roundtrip | final detections | stitched | float/roundtrip GT changes |
|---:|---:|---:|---:|---:|---:|
| 824/21 | 824/21 | 824/21 | 1632025 | 4323 | 0 |

The baseline prerequisite passed. The memory-safe CPU implementation uses the repository reference NumPy class-aware NMS and the frozen stitching logic. It reproduces the required 824 TP / 21 FN and 4,323 stitched boxes.

## Six frozen configurations

| config | added | float TP/FN | roundtrip TP/FN | rescued | regressed | net | images | groups | classes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| u3e4_g060 | 828,843 | 830/15 | 830/15 | 6 | 0 | 6 | 3 | 3 | 3 |
| u3e4_g080 | 1,262,827 | 830/15 | 830/15 | 7 | 1 | 6 | 4 | 4 | 3 |
| u1e3_g060 | 849,473 | 831/14 | 831/14 | 7 | 0 | 7 | 4 | 4 | 3 |
| u1e3_g080 | 1,312,378 | 830/15 | 830/15 | 7 | 1 | 6 | 4 | 4 | 3 |
| uinf_g060 | 858,679 | 831/14 | 831/14 | 7 | 0 | 7 | 4 | 4 | 3 |
| uinf_g080 | 1,342,904 | 830/15 | 830/15 | 7 | 1 | 6 | 4 | 4 | 3 |

- Best net gain: +7 TP.
- Float/roundtrip-stable configs: u3e4_g060, u3e4_g080, u1e3_g060, u1e3_g080, uinf_g060, uinf_g080.
- Fewest-added qualifying config: u3e4_g060.

## Attribution for selected audit config: u1e3_g060

### Rescued

| representation | image | gt index | group | class | failure | baseline → new | baseline IoU | new IoU |
|---|---|---:|---|---|---|---|---:|---:|
| float | 0001417541-Raw02-f_00011.jpg | 1 | 0001417541 | qilie | fusion_oracle_rescuable | FN → TP | 0.3022 | 0.7944 |
| float | 0001441830-Raw02-f_00002.jpg | 24 | 0001441830 | mamianmakeng | localization_failure | FN → TP | 0.3598 | 0.5950 |
| float | 0001472229_Raw13_f_00004.jpg | 4 | 0001472229 | huashang | localization_failure | FN → TP | 0.4492 | 0.5325 |
| float | C1297627_V01_F00002_08304277-d9b9-4278-992a-2fbf45ac0c3f.jpg | 2 | C1297627 | huashang | localization_near_miss | FN → TP | 0.4963 | 0.5407 |
| float | C1297627_V01_F00002_08304277-d9b9-4278-992a-2fbf45ac0c3f.jpg | 4 | C1297627 | huashang | localization_failure | FN → TP | 0.4417 | 0.7154 |
| float | C1297627_V01_F00002_08304277-d9b9-4278-992a-2fbf45ac0c3f.jpg | 6 | C1297627 | huashang | localization_failure | FN → TP | 0.3994 | 0.6118 |
| float | C1297627_V01_F00002_08304277-d9b9-4278-992a-2fbf45ac0c3f.jpg | 8 | C1297627 | huashang | fusion_oracle_rescuable | FN → TP | 0.4200 | 0.7076 |
| roundtrip | 0001417541-Raw02-f_00011.jpg | 1 | 0001417541 | qilie | fusion_oracle_rescuable | FN → TP | 0.3019 | 0.8059 |
| roundtrip | 0001441830-Raw02-f_00002.jpg | 24 | 0001441830 | mamianmakeng | localization_failure | FN → TP | 0.3583 | 0.6013 |
| roundtrip | 0001472229_Raw13_f_00004.jpg | 4 | 0001472229 | huashang | localization_failure | FN → TP | 0.4447 | 0.5300 |
| roundtrip | C1297627_V01_F00002_08304277-d9b9-4278-992a-2fbf45ac0c3f.jpg | 2 | C1297627 | huashang | localization_near_miss | FN → TP | 0.4943 | 0.5623 |
| roundtrip | C1297627_V01_F00002_08304277-d9b9-4278-992a-2fbf45ac0c3f.jpg | 4 | C1297627 | huashang | localization_failure | FN → TP | 0.4685 | 0.7576 |
| roundtrip | C1297627_V01_F00002_08304277-d9b9-4278-992a-2fbf45ac0c3f.jpg | 6 | C1297627 | huashang | localization_failure | FN → TP | 0.4274 | 0.6343 |
| roundtrip | C1297627_V01_F00002_08304277-d9b9-4278-992a-2fbf45ac0c3f.jpg | 8 | C1297627 | huashang | fusion_oracle_rescuable | FN → TP | 0.4466 | 0.7333 |

### Regressed

| representation | image | gt index | group | class | baseline → new | baseline IoU | new IoU |
|---|---|---:|---|---|---|---:|---:|
| — | — | — | — | — | — | — |

## Baseline per-class reproduction

| class | TP | FN |
|---|---:|---:|
| jieba | 173 | 4 |
| zonglie | 64 | 1 |
| qilie | 6 | 1 |
| jiaza | 33 | 4 |
| yiwuyaru | 91 | 1 |
| huashang | 19 | 5 |
| mamianmakeng | 332 | 1 |
| yanghuatiepi | 69 | 3 |
| gunyin | 37 | 1 |

## Distribution audit

- Float rescues by image: `{"0001417541-Raw02-f_00011.jpg": 1, "0001441830-Raw02-f_00002.jpg": 1, "0001472229_Raw13_f_00004.jpg": 1, "C1297627_V01_F00002_08304277-d9b9-4278-992a-2fbf45ac0c3f.jpg": 4}`
- Float rescues by group: `{"0001417541": 1, "0001441830": 1, "0001472229": 1, "C1297627": 4}`
- Float rescues by class: `{"huashang": 5, "mamianmakeng": 1, "qilie": 1}`
- Float rescues by failure: `{"fusion_oracle_rescuable": 2, "localization_failure": 4, "localization_near_miss": 1}`
- Unique rescue images/groups/classes/failure types: 4/4/3/3.
- Largest rescue-group contribution: 4.
- Net gain after removing a largest rescue group: 3.

## Submission-roundtrip consistency

- Selected config float: 831/14; roundtrip: 831/14.
- Selected config roundtrip rescued/regressed/net: 7/0/7.
- GT status changes caused solely by serialization: 0.
- Roundtrip used the formal floor(xmin/ymin), ceil(xmax/ymax), image clipping, six-decimal score, JSON write, JSON reread, and the same IoU>=0.5 matcher.

## Resources and engineering

- Builder runtime: 189.7 s.
- Evaluator runtime, baseline: 214.1 s.
- Evaluator runtime, six configs: 2469.8 s.
- Total validated pipeline runtime (builder + baseline + six configs): 47.9 min.
- Peak evaluator RSS: 98.7 MB.
- OOM/Killed: one initial unmodified PyTorch CPU replay was killed before completion; the validated memory-safe NumPy replay and all reported runs completed without OOM.
- Added scripts: `scripts/41_build_baseline_global_complement_val.py`, `scripts/42_eval_baseline_global_exact_final_combo.py`, `scripts/43_report_baseline_global_exact_combo.py`, `scripts/44_run_baseline_global_exact_combo.sh`.
- Output root: `results/baseline_complementarity/global_exact_final_combo`. Candidate counts and per-class additions are recorded in `summary.csv` and each config manifest.

## Final decision: GO

1. PASS — Exact Final Combo net gain >= +3.
2. PASS — roundtrip net gain >= +3.
3. PASS — rescued >= 3 images.
4. PASS — rescued >= 3 grouped-split groups.
5. PASS — rescued >= 2 classes.
6. PASS — rescues span >= 2 failure clusters.
7. PASS — net gain without largest group > 0.
8. PASS — adjacent frozen config within 1 TP net.

`recommended_hidden_candidate = u1e3_g060`

No Test/Hidden data was read or executed, and no Hidden submission was generated.
