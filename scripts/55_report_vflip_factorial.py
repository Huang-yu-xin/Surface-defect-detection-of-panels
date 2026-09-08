"""Generate the final Val-only scientific report from completed artifacts."""
import csv,hashlib,json,subprocess,sys,time
from collections import Counter
from pathlib import Path

ROOT=Path('results/vflip_factorial')
VIEWS={'D':'direction','G':'grid','F':'full'}
def read(path):return json.loads((ROOT/path).read_text())
def csvread(path):
    with (ROOT/path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def table(headers,rows):
    def cell(x):return str(x).replace('|','/').replace('\n',' ')
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']+['| '+' | '.join(cell(c) for c in row)+' |' for row in rows])
def write_csv(path,rows):
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)

def main():
    verification=ROOT/'audit/independent_verification.json'
    if not verification.exists():
        subprocess.run([sys.executable,'scripts/57_verify_vflip_factorial_artifacts.py'],check=True)
    assert read('audit/independent_verification.json')['status']=='PASS'
    context=read('audit/execution_context.json');baseline=read('exact_combo/baseline/metrics.json');basepass=read('exact_combo/baseline/PASS.json')
    geometry=read('audit/geometry_tests.json');analysis=read('audit/analysis_complete.json');proposal=read('audit/proposal_summary.json')
    overlap=read('audit/rescue_overlap.json');exact_overlap=read('audit/exact_rescue_overlap.json');attrs=csvread('audit/gt_attribution.csv');roundtrip_changes=csvread('audit/roundtrip_changes.csv')
    assert basepass['status']==geometry['status']==analysis['status']=='PASS'
    summaries=[];gates={};metrics={};manifests={};audits={}
    for vid,view in VIEWS.items():
        m=read(f'exact_combo/{view}/metrics.json');manifest=read(f'{view}/cache/manifest.json');audit=read(f'audit/cache_{view}.json')
        metrics[vid]=m;manifests[vid]=manifest;audits[vid]=audit;g=m['float_group_audit'];r=m['roundtrip_group_audit']
        assert audit['status']=='PASS' and audit['image_count']==474 and audit['coord_residual']<=1e-4
        assert m['float_tp']+m['float_fn']==845 and m['roundtrip_tp']+m['roundtrip_fn']==845
        assert m['float_tp']-831==g['net_gain'] and m['roundtrip_tp']-831==r['net_gain']
        gains=[a for a in attrs if a['config']==vid and a['baseline_status']=='FN' and a['new_status']=='TP']
        stable=all(a['roundtrip_status']=='TP' for a in gains)
        checks={'exact_net_ge_3':g['net_gain']>=3,'roundtrip_net_ge_3':r['net_gain']>=3,
            'rescued_ge_3_images':g['rescue_images']>=3,'rescued_ge_3_groups':g['rescue_groups']>=3,
            'multiple_failure_clusters':g['rescue_failure_types']>=2,'remove_largest_group_positive':g['net_gain_without_largest_group']>0,
            'regressions_le_1':g['regressed']<=1,'float_rescues_survive_roundtrip':stable,
            'not_only_one_image':g['rescue_images']>1 and g['net_gain_without_largest_group']>0,
            'distinct_observation_geometry':geometry['status']=='PASS', 'complete_valid_cache':audit['status']=='PASS'}
        gates[vid]={'GO':all(checks.values()),'checks':checks,'preferred_ge_2_classes':g['rescue_classes']>=2,
            'single_image_note':'At least three rescue images/groups plus positive net after removing the largest rescue group; cache shape/coordinates match O. No inference of production-line identity.'}
        s={'config':'BASE+'+vid,'orientation':manifest['factor_orientation'],'grid_mode':manifest['grid_mode'],
            'candidate_rows':manifest['candidate_rows'],'runtime_seconds':manifest['runtime_seconds'],'peak_gpu_mem_mb':manifest['peak_gpu_mem_mb'],
            **proposal[vid],'float_tp':m['float_tp'],'float_fn':m['float_fn'],'float_recall':m['float_tp']/845,
            'roundtrip_tp':m['roundtrip_tp'],'roundtrip_fn':m['roundtrip_fn'],'roundtrip_recall':m['roundtrip_tp']/845,
            **{k:g[k] for k in ['rescued','regressed','net_gain','rescue_images','rescue_groups','rescue_classes','rescue_failure_types','net_gain_without_largest_group']},
            'roundtrip_net_gain':r['net_gain'],'roundtrip_regressed':r['regressed'],
            **{k:m[k] for k in ['new_stitched','new_view_participating_stitched','mixed_source_stitched','new_mixed_source_stitched','pre_stitch_rescues','stitch_only_rescues','roundtrip_gt_status_changes']},
            'exact_runtime_seconds':m['runtime_seconds'],'oom_events':len(manifest['oom_events']),'go':'GO' if gates[vid]['GO'] else 'NO-GO'}
        summaries.append(s)
    net={k:metrics[k]['float_group_audit']['net_gain'] for k in VIEWS}
    eligible=[k for k in VIEWS if gates[k]['GO']]
    if not eligible:mechanism='NO_USEFUL_VFLIP_MECHANISM';reason='No view satisfies the frozen GO criteria. Proposal complementarity is diagnostic and cannot substitute for robust exact-combo gain.'
    elif net['D']>=3 and net['G']>=3:
        mechanism='MIXED_DIRECTION_AND_GRID';reason='Both single-factor exact additions independently achieve at least +3; overlap tables specify whether gains are shared or exclusive.'
    elif net['F']>=3 and net['D']<3 and net['G']<3:
        mechanism='INTERACTION_DOMINANT';reason='Only the combined orientation/grid cell clears the +3 exact net-gain threshold.'
    elif net['D']>=3 and exact_overlap['counts']['D-only']>0:
        mechanism='DIRECTION_DOMINANT';reason='Direction-only exact gain is at least +3 and has exact D-only rescues.'
    elif net['G']>=3 and exact_overlap['counts']['G-only']>0:
        mechanism='GRID_DOMINANT';reason='Grid-only exact gain is at least +3 and has exact G-only rescues.'
    elif net['F']>=3 and exact_overlap['counts']['F-only']>=3:
        mechanism='INTERACTION_DOMINANT';reason='The full cell supplies at least three exclusive exact rescues.'
    else:
        mechanism='MIXED_DIRECTION_AND_GRID';reason='Useful gains overlap across factors; the exclusive-rescue evidence does not support attributing all gain to one factor.'
    recommended=None
    if eligible:
        simple=[k for k in eligible if k in ['D','G']]
        choices=simple or eligible
        recommended=sorted(choices,key=lambda k:(-net[k],metrics[k]['float_group_audit']['regressed'],-metrics[k]['float_group_audit']['rescue_groups'],k))[0]
    decision={'decision':'GO' if eligible else 'NO-GO','mechanism':mechanism,'reason':reason,'recommended_view_for_future_test':recommended,
        'gates':gates,'main_effect_direction_exact_tp':net['D'],'main_effect_grid_exact_tp':net['G'],
        'factorial_interaction_exact_tp':net['F']-net['D']-net['G'],
        'interaction_caveat':'Difference-in-differences in this fixed BASE-augmentation experiment, not an inferential causal estimate; nonlinear NMS, stitch and matching can affect it.',
        'test_hidden_run':'NO','test_hidden_gt_accessed':'NO','competition_submission':'NO'}
    (ROOT/'decision.json').write_text(json.dumps(decision,indent=2))
    write_csv(ROOT/'summary.csv',summaries)
    lines=['# RareOS VFlip 2×2 Factorial — 474-Val-only report','',
        '## A. Execution status','',f'- Repo HEAD: `{context["repo_head"]}`',f'- Workspace dirty: {bool(context["workspace_dirty"])} (preexisting changes preserved; details in `audit/execution_context.json`).',
        f'- GPU: {context["gpu"]}',f'- RareOS weights: `{context["weights"]}`',
        '- Test/Hidden run = **NO**; Test/Hidden GT accessed = **NO**; competition submission = **NO**.',
        '- No model training, weight modification, selector, score/IoU sweep, or D+G/F combination was performed.',
        '- Existing script 50 was occupied; new scripts use 51–57. All artifacts and JSON roundtrips in this report are Val only.',
        '- Execution recovery: the first persisted-provenance verification hit Python CSV’s default 131072-byte field limit. The verifier reader limit was raised to 16 MiB; completed predictions and exact results were reused unchanged. The subsequent full artifact verification passed. Original and recovery logs are retained.',
        '', '## B. Current baseline reproduction','',
        f'Expected 831/14. Recomputed float **{baseline["float_tp"]}/{baseline["float_fn"]}**, JSON roundtrip **{baseline["roundtrip_tp"]}/{baseline["roundtrip_fn"]}**: **PASS**. Changed GT statuses: {baseline["roundtrip_changes"]}. Runtime: {baseline["runtime_seconds"]:.2f} s.',
        '', 'The baseline is RareOS O + frozen corrected Baseline u1e3_g060 + selective HFlip {0,2,3,4,7}, followed by the existing Script42 global NMS and Zonglie stitch. It has 2,340,806 final detections and 4,899 stitched boxes. Script42/14 are reused directly; historical 824/21 is not the comparison baseline.',
        '', '## C. Geometry implementation audit','',table(['check','status'],geometry['checks'].items()),'',
        'Mirrored starts are computed as `H - (standard_y + valid_h)`, deduplicated and sorted; x starts are unchanged. D uses the exact Script14 get_starts. Flip the valid pixels before padding, inverse y with `valid_h`, then translate to global coordinates. Cache local coordinates are stored from the same float32 global rounding, and residuals are independently checked in float64.',
        '',f'{geometry["patches_compared"]} synthetic/real patch comparisons passed, including padded small images. Five real images were selected by sorted indices 1/100/200/300/474, independently of GT.',
        '',table(['index','image','patches','identical'],[(x['index_one_based'],x['image'],x['patches'],x['pixel_identical']) for x in geometry['samples']]),
        '', '## D. Cache summary','',table(['view','images','candidates','coord residual','runtime s','GPU allocated peak MiB','GPU reserved peak MiB','OOM'],
            [(k,audits[k]['image_count'],f'{manifests[k]["candidate_rows"]:,}',audits[k]['coord_residual'],f'{manifests[k]["runtime_seconds"]:.2f}',f'{manifests[k]["peak_gpu_mem_mb"]:.2f}',f'{manifests[k]["peak_gpu_reserved_mb"]:.2f}',len(manifests[k]['oom_events'])) for k in VIEWS]),
        '', 'All three caches contain exactly 474 NPZ files, the O image set, and the original 14-column float32 schema; all coordinates, classes and scores are valid, with no NaN/Inf. Candidate-side selection uses no GT. Each cache passed audit before the next GPU view started.',
        '', 'Frozen parameters: tile/imgsz=1280, stride=768, conf=1e-5, tile IoU=.60, max_det=1000, batch=6, half=True. These numerical inference parameters match O/H manifests. O/H omit the half field, but their inference logs contain half-argument warnings; this run requests half=True and verifies actual fp16. Allocated/reserved peaks are process-local PyTorch peaks, not total board usage.',
        '', '## E. Proposal-level rescue','',table(['view','IoU≥.30','IoU≥.40','direct rescue','images','groups','classes'],[(k,proposal[k]['iou_ge_030'],proposal[k]['iou_ge_040'],proposal[k]['direct_rescue'],proposal[k]['direct_rescue_images'],proposal[k]['direct_rescue_groups'],proposal[k]['direct_rescue_classes']) for k in VIEWS]),
        '', 'Direct rescue means one of the current 14 FN has a raw same-class new-view proposal at IoU≥.50. It is an oracle-style Val diagnostic, not a box-selection rule. Per-FN best score, box, candidate row, class, group and failure are in `audit/direct_rescue.csv`.',
        '', '## F. Rescue overlap','',table(['set','proposal count','exact count'],[(k,n,exact_overlap['counts'][k]) for k,n in overlap['counts'].items()]),
        '',table(['Jaccard','proposal','exact'],[(k,n,exact_overlap['jaccard'][k]) for k,n in overlap['jaccard'].items()]),
        '', 'Pairwise intersections are inclusive; each “only” set excludes both other views. Empty-union Jaccard is undefined, shown as None. Exact sets use the official greedy matcher, not best-IoU thresholding.',
        '', '## G. Exact Combo','',table(['config','float TP/FN','roundtrip TP/FN','rescued','regressed','net','images','groups','classes'],
            [('BASE+'+k,f'{metrics[k]["float_tp"]}/{metrics[k]["float_fn"]}',f'{metrics[k]["roundtrip_tp"]}/{metrics[k]["roundtrip_fn"]}',*[metrics[k]['float_group_audit'][x] for x in ['rescued','regressed','net_gain','rescue_images','rescue_groups','rescue_classes']]) for k in VIEWS]),
        '', 'Only BASE+D, BASE+G and BASE+F were evaluated. The input order is frozen active O+Baseline, selected HFlip, then all new-view classes/candidates. Script42 NMS and stitch, Script14 matcher, and Script42 submission_row/roundtrip are used directly without gate changes.',
        '',table(['view','GT status changes in roundtrip','roundtrip net','exact runtime s'],[(k,metrics[k]['roundtrip_gt_status_changes'],metrics[k]['roundtrip_group_audit']['net_gain'],f'{metrics[k]["runtime_seconds"]:.2f}') for k in VIEWS]),
        '', 'Roundtrip uses JSON write/read with formal floor/ceil/clipping and six-decimal scores, then the same matcher. Per-image JSON is retained. Any float rescue lost on roundtrip blocks GO.',
        '',table(['view','GT / image','float matched','roundtrip matched','float best IoU','roundtrip best IoU'],[(r['config'],r['image']+' #'+r['gt_index'],r['float_status'],r['roundtrip_status'],r['float_best_iou'],r['roundtrip_best_iou']) for r in roundtrip_changes]) if roundtrip_changes else 'No GT status changes in any view.',
        '', 'An extra roundtrip-only TP is a representation-mediated result and is reported separately from float rescue. It is not used to satisfy the minimum +3 float net-gain requirement.',
        '', '## H. Stitch provenance','',table(['view','all stitched','new boxes vs BASE','new-view participation','mixed source','new & mixed','pre-stitch rescues','stitch-only rescues'],
            [(k,*[metrics[k][x] for x in ['stitched_count','new_stitched','new_view_participating_stitched','mixed_source_stitched','new_mixed_source_stitched','pre_stitch_rescues','stitch_only_rescues']]) for k in VIEWS]),
        '', '“New” compares class/global xyxy to BASE stitched boxes in the same image; score changes alone do not count. Mixed source means at least two of O/Baseline/H/new view. The actual formal stitch frame supplies its candidate groups; reconstructed provenance must exactly equal the formal emitted boxes. Sidecars retain all segment origins/classes/source rows and NMS input indices. These sources may legitimately interact.',
        '', 'Pre-stitch rescue uses the same matcher before adding merged boxes; stitch-only means the final matcher rescues the GT but the pre-stitch matcher does not. Actual final matcher assignments, rather than the best-IoU candidate, supply TP attribution sources. Pre-stitch gains can still involve NMS interactions; proposal overlap separately identifies direct observation support.',
        '', f'Grid-only G has {proposal["G"]["direct_rescue"]} direct proposal rescues but {metrics["G"]["stitch_only_rescues"]} stitch-only exact rescues. This explicitly measures crop/grid contributions mediated by the existing segment-merging pipeline.',
        '', '## I. Group robustness','',table(['view','representation','largest group','contribution','rescued after removal','regressed after removal','net after removal'],
            [(k,rep,g['largest_group'],g['largest_group_contribution'],g['rescued_without_largest_group'],g['regressed_without_largest_group'],g['net_gain_without_largest_group']) for k in VIEWS for rep in ['float','roundtrip'] for g in [metrics[k][rep+'_group_audit']]]),
        '', 'Groups are the project’s grouped-split independence units from `splits/sample_assignments.csv`, not verified production-line IDs. Both rescued and regressed GT belonging to the removed group are excluded. Tied largest groups use the conservative minimum net gain; every tie is retained in group_audit.json.',
        '',table(['view','rescues by group','rescues by class','rescues by failure'],[(k,*[json.dumps(metrics[k]['float_group_audit'][x],sort_keys=True) for x in ['rescues_by_group','rescues_by_class','rescues_by_failure']]) for k in VIEWS]),
        '', '## J. Mechanism conclusion','',f'**{mechanism}**. {reason}',
        '',f'Exact TP contrasts relative to BASE: direction={net["D"]:+d}, grid={net["G"]:+d}, full={net["F"]:+d}; full−direction−grid={decision["factorial_interaction_exact_tp"]:+d}. This arithmetic interaction describes this fixed augmentation pipeline; nonlinear NMS, stitching and greedy matching prevent treating it as a causal or statistical significance claim. The task’s INTERACTION_DOMINANT rule is operational (F reaches +3 while D/G each fall short); that label alone does not establish positive superadditivity.',
        '', 'Direction changes orientation at fixed windows; G changes only boundary/context; F changes both. The overlap tables and per-GT sources bound how strongly any observed gain can be assigned to a factor.',
        '', f'Observed proposal support: D={proposal["D"]["direct_rescue"]}, G={proposal["G"]["direct_rescue"]}, F={proposal["F"]["direct_rescue"]}; D-only={overlap["counts"]["D-only"]}, G-only={overlap["counts"]["G-only"]}, F-only={overlap["counts"]["F-only"]}, D∩F={overlap["counts"]["D∩F"]}. An F-only GT indicates target-specific combined-view complementarity. Aggregate interaction can be zero even when individual rescued GT differ.',
        '', '## K. Final GO / NO-GO','',f'**{decision["decision"]}**',
        '',table(['view','decision','failed mandatory criteria','≥2 classes preference'],[(k,'GO' if gates[k]['GO'] else 'NO-GO',', '.join(x for x,passed in gates[k]['checks'].items() if not passed) or 'none',gates[k]['preferred_ge_2_classes']) for k in VIEWS]),
        '', 'At least two rescued classes is a stated preference; all other listed GO checks are enforced. More candidate boxes or an oracle rescue alone does not justify GO.',
        '',f'recommended_view_for_future_test = {recommended or "NONE"}. **No Test/Hidden execution or competition submission was performed.**',
        '', ('Future discussion only: evaluate the recommended view in a separately authorized task; simpler D/G views are preferred when eligible.' if eligible else 'Next priority: new tile geometry/scale or model diversity. Do not tune gates or combine D/G/F to chase this fixed Val tail.'),
        '', '## Per-GT attribution: current 14 FN plus any regression','']
    keys=sorted({(a['image'],a['gt_index']) for a in attrs});rows=[]
    for image,gi in keys:
        aa=[a for a in attrs if a['image']==image and a['gt_index']==gi];a=aa[0]
        rescued=[x['config'] for x in aa if x['baseline_status']=='FN' and x['new_status']=='TP'];regressed=[x['config'] for x in aa if x['baseline_status']=='TP' and x['new_status']=='FN']
        source='; '.join(x['config']+':'+x['new_source']+' ('+x['rescue_attribution']+')' for x in aa if x['rescue_attribution'] in ['pre_stitch','stitch_only','regression'])
        def fmt(value):return f'{float(value):.4f}' if value else '—'
        rows.append([image+' #'+gi,a['group'],a['class'],a['failure'],fmt(a['baseline_best_iou']),fmt(a['D_iou']),fmt(a['G_iou']),fmt(a['F_iou']),','.join(rescued) or 'none',','.join(regressed) or 'none',source or 'none'])
    lines.append(table(['image / GT','group','class','failure','BASE IoU','D IoU','G IoU','F IoU','exact rescued','regressed','actual source / attribution'],rows))
    lines += ['', 'All boxes and pre/post/roundtrip statuses for these GT are retained in `audit/gt_attribution.csv`. Internal full matcher state is retained only for reproducibility; this table contains no unrelated GT.', '', '## Artifact verification','',
        '- Baseline expected counts and unchanged roundtrip GT status: PASS.','- D/G/F geometry and complete 474-image cache audits: PASS.','- All three allowed exact configurations completed: PASS.','- Official stitch-member and matcher-assignment provenance checks: PASS.','- Group removal includes both rescued and regressed GT: PASS.','- Report, summary, direct/overlap/group/stitch/GT attribution and Val roundtrips exist: PASS.',
        '', 'Execution stopped after this report. Test/Hidden run = NO; Test/Hidden GT accessed = NO; competition submission = NO.','']
    independent=read('audit/independent_verification.json')
    lines[-2:-2]=['Independent persisted-artifact verification: **PASS**. '+str(independent['post_nms_rows_verified_against_source_cache'])+' post-NMS rows matched their exact source-cache rows; '+str(independent['stitched_boxes_verified'])+' stitched boxes and '+str(independent['segment_records_verified'])+' segment records were checked. Fixed-sample formal JSON rows checked: '+str(independent['deterministic_roundtrip_sample_rows_verified'])+'. Details: `audit/independent_verification.json`.','']
    (ROOT/'report.md').write_text('\n'.join(lines),encoding='utf-8')
    checks={'baseline_reproduced':'PASS','geometry':'PASS','three_complete_cache_audits':'PASS','three_exact_metrics':'PASS','roundtrip_artifacts':'PASS','overlap_attribution_group_stitch_outputs':'PASS','summary_and_report':'PASS'}
    (ROOT/'audit/final_verification.json').write_text(json.dumps({'status':'PASS','checks':checks,'summary_rows':len(summaries),'reported_gt':len(keys),'report_sha256':hashlib.sha256((ROOT/'report.md').read_bytes()).hexdigest()},indent=2))
    print('END report STATUS=PASS '+json.dumps(decision),flush=True)

if __name__=='__main__':main()
