"""Proposal diagnostics and only BASE+D/G/F, using the frozen formal Val API."""
from __future__ import annotations
import csv, hashlib, importlib.util, inspect, json, sys, time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
import numpy as np

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

v=module('scripts/52_cache_vflip_factorial_val.py','vflip_analysis_views')
official=module('scripts/42_eval_baseline_global_exact_final_combo.py','vflip_official42')
diag=v.DIAG
ROOT=v.ROOT;BASE=Path('results/baseline_complementarity/global_exact_final_combo/u1e3_g060');H=Path('results/fn_analysis/cache_hflip')
CLASSES=diag.CLASS_NAMES

def write_csv(path,rows,fields=None):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(rows[0]),lineterminator='\n');writer.writeheader();writer.writerows(rows)

def key(row):return row['image'],int(row['gt_index'])
def load_array(root,item):
    with np.load(root/item['cache_file']) as z:return z['candidates'].astype(np.float32,copy=True)

def best(arr,b):
    gt=diag.Detection(b['class_id'],1.,*b['bbox'])
    found=diag.best_candidate(arr,gt,True)
    if found is None:return {'iou':0.,'score':None,'box':None,'index':None}
    index,iou,row=found
    return {'iou':iou,'score':float(row[1]),'box':row[2:6].tolist(),'index':int(index)}

def statuses_for(arr,gt):
    tp,fp,fn,unmatched,matched,pred=diag.match_predictions(arr,gt,.5)
    return tp,fp,fn,matched,pred

def actual_match_pairs(arr,gt):
    """Observe assignments in the unchanged official matcher on attribution images."""
    lines,start=inspect.getsourcelines(diag.match_predictions)
    assign_line=start+next(i for i,s in enumerate(lines) if 'matched_pred_global.add(int(pi))' in s)
    pairs={};code=diag.match_predictions.__code__
    def trace(frame,event,arg):
        if event=='call' and frame.f_code is code:return trace
        if frame.f_code is code and event=='line' and frame.f_lineno==assign_line:
            loc=frame.f_locals;pairs[int(loc['gt_idx'][loc['local_gi']])]=int(loc['pi'])
        return trace if frame.f_code is code else None
    previous=sys.gettrace();sys.settrace(trace)
    try:result=diag.match_predictions(arr,gt,.5)
    finally:sys.settrace(previous)
    assert set(pairs)==result[4] and set(pairs.values())==result[5]
    return pairs

def trace_stitch(post):
    zinds=np.flatnonzero(post[:,0].astype(np.int32)==1);zpost=post[zinds]
    captured={};code=official.stitch_zonglie.__code__
    def trace(frame,event,arg):
        if event=='call' and frame.f_code is code:
            frame.f_trace_lines=False
            return trace
        if event=='return' and frame.f_code is code:captured.update(frame.f_locals)
        return None
    previous=sys.gettrace();sys.settrace(trace)
    try:merged=official.stitch_zonglie(zpost)
    finally:sys.settrace(previous)
    if not len(merged):return merged,[]
    # Provenance is read from the actual formal function's candidate groups.
    cand=captured['cand'];groups=captured['groups']
    widths=np.maximum(1e-6,zpost[:,4]-zpost[:,2]);heights=np.maximum(1e-6,zpost[:,5]-zpost[:,3])
    candidate_indices=zinds[(heights>=180)&(heights/widths>=5)]
    assert np.array_equal(cand,post[candidate_indices])
    provenance=[];position=0
    for ids in groups.values():
        boxes=cand[ids]
        if len(ids)<2 or len(np.unique(boxes[:,7]))<2:continue
        y1=float(np.min(boxes[:,3]));y2=float(np.max(boxes[:,5]))
        if y2-y1<1300:continue
        x1=float(np.median(boxes[:,2]));x2=float(np.median(boxes[:,4]))
        if x2<=x1:continue
        expected=np.zeros(14,np.float32);expected[:6]=[1,float(np.max(boxes[:,1])),x1,y1,x2,y2]
        assert np.array_equal(expected,merged[position]), 'Formal stitch provenance mismatch'
        provenance.append(candidate_indices[np.asarray(ids,int)].tolist());position+=1
    assert position==len(merged)
    return merged,provenance

def count_sets(rescued,statuses,groups,failures):
    return {'count':len(rescued),'images':len({x[0] for x in rescued}),'groups':len({groups[x[0]] for x in rescued}),
        'classes':len({statuses[x]['class_id'] for x in rescued}),'failure_types':len({failures.get(x,'unclassified') for x in rescued})}

def group_audit(view,representation,rescued,regressed,statuses,groups,failures):
    rc=Counter(groups[k[0]] for k in rescued);gc=Counter(groups[k[0]] for k in regressed)
    byclass=Counter(CLASSES[statuses[k]['class_id']] for k in rescued);byfailure=Counter(failures.get(k,'unclassified') for k in rescued)
    largest=max(rc.values(),default=0);ties=sorted(g for g,c in rc.items() if c==largest)
    removals=[]
    for group in ties:
        r=sum(groups[k[0]]!=group for k in rescued);g=sum(groups[k[0]]!=group for k in regressed)
        removals.append({'group':group,'removed_rescues':rc[group],'removed_regressions':gc[group],'rescued':r,'regressed':g,'net_gain':r-g})
    worst=min(removals,key=lambda x:(x['net_gain'],x['group'])) if removals else {'group':'','rescued':0,'regressed':len(regressed),'net_gain':-len(regressed)}
    c=count_sets(rescued,statuses,groups,failures)
    return {'config':view,'representation':representation,'rescued':len(rescued),'regressed':len(regressed),'net_gain':len(rescued)-len(regressed),
        'rescue_images':c['images'],'rescue_groups':c['groups'],'rescue_classes':c['classes'],'rescue_failure_types':c['failure_types'],
        'rescues_by_group':dict(rc),'regressions_by_group':dict(gc),'rescues_by_class':dict(byclass),'rescues_by_failure':dict(byfailure),
        'largest_group':worst['group'],'largest_group_contribution':largest,'rescued_without_largest_group':worst['rescued'],
        'regressed_without_largest_group':worst['regressed'],'net_gain_without_largest_group':worst['net_gain'],
        'largest_group_tie_policy':'conservative minimum net gain across all tied largest groups','all_largest_group_removals':removals}

def overlap(sets):
    d,g,f=[sets[x] for x in ['D','G','F']]
    rows={'D':d,'G':g,'F':f,'D-only':d-g-f,'G-only':g-d-f,'F-only':f-d-g,'D∩G':d&g,'D∩F':d&f,'G∩F':g&f,'D∩G∩F':d&g&f}
    j={}
    for a,b in [('D','G'),('D','F'),('G','F')]:
        union=sets[a]|sets[b];j[f'J({a},{b})']=len(sets[a]&sets[b])/len(union) if union else None
    return {'counts':{k:len(s) for k,s in rows.items()},'sets':{k:[list(t) for t in sorted(s)] for k,s in rows.items()},'jaccard':j,'empty_union_jaccard':'undefined/null'}

def main():
    start=time.time()
    assert v.read_json(ROOT/'exact_combo/baseline/PASS.json')['status']=='PASS'
    # All predictions and complete audits must predate any new GT analysis.
    for view in v.VIEWS:assert v.audit(view)['status']=='PASS'
    states={key(r):r for r in v.read_json(ROOT/'exact_combo/baseline/statuses.json')}
    assert len(states)==845 and sum(not x['float'] for x in states.values())==14
    fn={k:s for k,s in states.items() if not s['float']}
    args=SimpleNamespace(assignments=Path('splits/sample_assignments.csv'),failure_csv=Path('results/final_combo_fn21/remaining_fn_21.csv'))
    groups,failures,schema=official.read_maps(args)
    assert all(k in failures for k in fn), 'Current FN must have a documented baseline failure cluster'
    with args.assignments.open(encoding='utf-8-sig',newline='') as f:assignment_rows={r['image']:r for r in csv.DictReader(f)}
    for name,_ in states:assert name in groups and groups[name] and assignment_rows[name]['split']=='val'
    om,items=v.val_items();bmap={i['image_name']:i for i in v.read_json(BASE/'manifest.json')['items']};hmap={i['image_name']:i for i in v.read_json(H/'manifest.json')['items']}
    proposals=[];proposal_by_key={};direct_sets={};direct_summary={}
    print('START proposal_analysis complete_D_G_F_audits=PASS',flush=True)
    for view,(vid,grid,orientation) in v.VIEWS.items():
        rescued=set();sub=[];directory=ROOT/view/'cache'
        for k,b in fn.items():
            item=next(i for i in items if i['image_name']==k[0]);arr=load_array(directory,item);bp=best(arr,b)
            row={'view':vid,'image':k[0],'gt_index':k[1],'group':groups[k[0]],'class':CLASSES[b['class_id']],
                'failure':failures.get(k,'unclassified'),'gt_bbox':b['bbox'],'baseline_final_iou':b['float_best']['iou'],
                'best_iou':bp['iou'],'best_score':bp['score'],'best_box':bp['box'],'candidate_row':bp['index'],'direct_rescue':bp['iou']>=.5}
            proposals.append(row);sub.append(row);proposal_by_key[(vid,k)]=row
            if bp['iou']>=.5:rescued.add(k)
        direct_sets[vid]=rescued;c=count_sets(rescued,states,groups,failures)
        direct_summary[vid]={'iou_ge_030':sum(r['best_iou']>=.3 for r in sub),'iou_ge_040':sum(r['best_iou']>=.4 for r in sub),
            'direct_rescue':c['count'],'direct_rescue_images':c['images'],'direct_rescue_groups':c['groups'],'direct_rescue_classes':c['classes'],'direct_rescue_failure_types':c['failure_types']}
    write_csv(ROOT/'audit/direct_rescue.csv',proposals)
    v.write_json(ROOT/'audit/rescue_overlap.json',overlap(direct_sets))
    v.write_json(ROOT/'audit/proposal_summary.json',direct_summary)
    print('END proposal_analysis '+json.dumps(direct_summary),flush=True)
    all_metrics=[];all_group=[];attributions=[];rtchanges=[];stitch_rows=[];exact_sets={};exact_details={}
    for view,(vid,grid,orientation) in v.VIEWS.items():
        t=time.time();out=ROOT/'exact_combo'/view;out.mkdir(parents=True,exist_ok=True)
        (out/'final_cache').mkdir(exist_ok=True);(out/'provenance').mkdir(exist_ok=True)
        directory=ROOT/view/'cache';totals=Counter();current={};nms_rows=0;total_stitched=0;new_stitched=0;new_participated=0;mixed_stitched=0;new_mixed_stitched=0
        print(f'START exact_combo BASE+{vid} images=474 all_classes=YES no_extra_gates=YES',flush=True)
        for index,item in enumerate(items,1):
            name=item['image_name'];active=load_array(BASE,bmap[name]);hf=load_array(H,hmap[name]);new=load_array(directory,item)
            oi=bmap[name]['original_count'];assert oi==item['candidate_count']
            hids=np.flatnonzero(np.isin(hf[:,0].astype(int),list(official.HFLIP_CLASSES)))
            hsel=hf[hids];union=np.concatenate([active,hsel,new])
            source=np.concatenate([np.zeros(oi,np.int8),np.ones(len(active)-oi,np.int8),np.full(len(hsel),2,np.int8),np.full(len(new),3,np.int8)])
            source_row=np.concatenate([np.arange(len(active)),hids,np.arange(len(new))]).astype(np.int64)
            observed={}
            def record_nms(arr,iou):
                keep=diag.class_aware_nms_indices(arr,iou);observed['keep']=keep;return keep
            post=official.nms(union,SimpleNamespace(class_aware_nms_indices=record_nms));keep=observed['keep']
            assert np.array_equal(post,union[keep]);post_source=source[keep];post_row=source_row[keep]
            merged,segments=trace_stitch(post)
            final=np.concatenate([post,merged]) if len(merged) else post
            with np.load(ROOT/'exact_combo/baseline/final_cache'/f'{Path(name).stem}.npz') as z:
                baseline_final=z['final'];baseline_post=int(z['post_count'])
            baseline_boxes={tuple(row[[0,2,3,4,5]].tolist()) for row in baseline_final[baseline_post:]}
            source_names=['O','Baseline','H',vid];merged_source=[];merged_new=[]
            for mi,ids in enumerate(segments):
                source_ids=sorted(set(int(x) for x in post_source[ids]));labels=[source_names[x] for x in source_ids]
                isnew=tuple(merged[mi,[0,2,3,4,5]].tolist()) not in baseline_boxes
                hasnew=3 in source_ids;ismixed=len(source_ids)>1
                members=[{'view_id':source_names[int(post_source[j])],'source_cache_row':int(post_row[j]),'post_nms_row':j,
                    'tile_x':int(post[j,6]),'tile_y':int(post[j,7]),'class':int(post[j,0]),'bbox':post[j,2:6].tolist()} for j in ids]
                stitch_rows.append({'config':vid,'image':name,'stitched_index':mi,'final_row':len(post)+mi,'box':merged[mi,2:6].tolist(),
                    'source_views':'+'.join(labels),'new_relative_to_baseline':isnew,'new_view_participates':hasnew,'mixed_source':ismixed,'segment_count':len(ids),'segments':json.dumps(members,separators=(',',':'))})
                new_stitched+=isnew;new_participated+=hasnew;mixed_stitched+=ismixed;new_mixed_stitched+=isnew and ismixed
                merged_source.append('+'.join(labels));merged_new.append(hasnew)
            np.savez_compressed(out/'final_cache'/item['cache_file'],final=final,post_count=len(post))
            np.savez_compressed(out/'provenance'/item['cache_file'],post_source_id=post_source,source_cache_row=post_row,union_row=keep)
            rt=official.roundtrip(final,name,item['width'],item['height'],out/'roundtrip_json'/f'{Path(name).stem}.json')
            gt=diag.read_yolo_gt(Path('datasets/yolo_split/labels/val')/f'{Path(name).stem}.txt',item['width'],item['height'])
            ftp,ffp,ffn,fmatch,fmp=statuses_for(final,gt);rtp,rfp,rfn,rmatch,rmp=statuses_for(rt,gt);ptp,pfp,pfn,pmatch,pmp=statuses_for(post,gt)
            needs_attribution=any((name,gi) in fn or (states[(name,gi)]['float'] and gi not in fmatch) for gi in range(len(gt)))
            matched_pairs=actual_match_pairs(final,gt) if needs_attribution else {}
            totals.update(float_tp=ftp,float_fp=ffp,float_fn=ffn,roundtrip_tp=rtp,roundtrip_fp=rfp,roundtrip_fn=rfn,pre_stitch_tp=ptp,pre_stitch_fn=pfn)
            total_stitched+=len(merged);nms_rows+=len(post)
            for gi,g in enumerate(gt):
                k=(name,gi);b=states[k];fp=best(final,b);rp=best(rt,b);pp=best(post,b)
                pi=fp['index'];source_label='none';uses_new=False
                if pi is not None:
                    if pi<len(post):source_label=source_names[int(post_source[pi])];uses_new=int(post_source[pi])==3
                    else:source_label='stitch:'+merged_source[pi-len(post)];uses_new=merged_new[pi-len(post)]
                mpi=matched_pairs.get(gi);matched_source=None
                if mpi is not None:
                    matched_source=source_names[int(post_source[mpi])] if mpi<len(post) else 'stitch:'+merged_source[mpi-len(post)]
                cur={'float':gi in fmatch,'roundtrip':gi in rmatch,'pre_stitch':gi in pmatch,'float_best':fp,'roundtrip_best':rp,'pre_stitch_best':pp,'best_iou_source':source_label,'best_iou_uses_new':uses_new,
                    'matched_final_row':mpi,'matched_source':matched_source,'matched_box':final[mpi,2:6].tolist() if mpi is not None else None}
                current[k]=cur
                if cur['float']!=cur['roundtrip']:
                    rtchanges.append({'config':vid,'image':name,'gt_index':gi,'float_status':cur['float'],'roundtrip_status':cur['roundtrip'],'float_best_iou':fp['iou'],'roundtrip_best_iou':rp['iou']})
            if index%25==0 or index==474:print(f'EXACT {vid} {index}/474 float={totals["float_tp"]}/{totals["float_fn"]} rt={totals["roundtrip_tp"]}/{totals["roundtrip_fn"]} runtime_seconds={time.time()-t:.1f}',flush=True)
        assert set(current)==set(states)
        for representation in ['float','roundtrip']:
            rescued={k for k,b in states.items() if not b[representation] and current[k][representation]}
            regressed={k for k,b in states.items() if b[representation] and not current[k][representation]}
            audit=group_audit(vid,representation,rescued,regressed,states,groups,failures);all_group.append(audit)
            if representation=='float':
                float_audit=audit;exact_sets[vid]=rescued
                for k in sorted(set(fn)|regressed):
                    b=states[k];n=current[k]
                    attributions.append({'config':vid,'image':k[0],'gt_index':k[1],'group':groups[k[0]],'class':CLASSES[b['class_id']],
                        'failure':failures.get(k,'same_class_support_lost_after_nms' if n['float_best']['iou']<.5 else 'score_order_matcher_competition'),'gt_bbox':b['bbox'],'baseline_status':'TP' if b['float'] else 'FN',
                        'new_status':'TP' if n['float'] else 'FN','roundtrip_status':'TP' if n['roundtrip'] else 'FN','baseline_best_iou':b['float_best']['iou'],
                        'new_best_iou':n['float_best']['iou'],'new_best_box':n['float_best']['box'],'new_source':n['matched_source'] or n['best_iou_source'],
                        'new_source_definition':'actual official matcher assignment for TP; best-IoU candidate source for FN','matched_box':n['matched_box'],
                        'pre_stitch_status':'TP' if n['pre_stitch'] else 'FN','pre_stitch_best_iou':n['pre_stitch_best']['iou'],
                        'rescue_attribution':('pre_stitch' if n['pre_stitch'] else 'stitch_only') if k in rescued else ('regression' if k in regressed else 'remaining_FN'),
                        'direct_new_view_candidate':proposal_by_key.get((vid,k),{}).get('direct_rescue',False),
                        'D_iou':proposal_by_key.get(('D',k),{}).get('best_iou',''),'G_iou':proposal_by_key.get(('G',k),{}).get('best_iou',''),'F_iou':proposal_by_key.get(('F',k),{}).get('best_iou','')})
            else:roundtrip_audit=audit
        rescued=exact_sets[vid];pre_rescue=sum(current[k]['pre_stitch'] for k in rescued);stitch_only=len(rescued)-pre_rescue
        metrics={'config':vid,'view':view,**dict(totals),'final_detection_count':nms_rows+total_stitched,'post_nms_count':nms_rows,
            'stitched_count':total_stitched,'new_stitched':new_stitched,'new_view_participating_stitched':new_participated,'mixed_source_stitched':mixed_stitched,
            'new_mixed_source_stitched':new_mixed_stitched,'pre_stitch_rescues':pre_rescue,'stitch_only_rescues':stitch_only,
            'roundtrip_gt_status_changes':sum(r['config']==vid for r in rtchanges),'runtime_seconds':time.time()-t,'float_group_audit':float_audit,'roundtrip_group_audit':roundtrip_audit}
        v.write_json(out/'metrics.json',metrics)
        # Internal matcher state supports reproducibility; report tables include only current FN/regressions.
        v.write_json(out/'comparison_state.json',[{'image':k[0],'gt_index':k[1],**value} for k,value in current.items()])
        v.write_json(out/'provenance/manifest.json',{'source_codes':{'0':'O','1':'Baseline','2':'H','3':vid},
            'source_cache_roots':{'O':str(BASE),'Baseline':str(BASE),'H':str(H),vid:str(directory)},
            'source_row_definition':'zero-based row in indicated cache; O and Baseline reference frozen merged active cache',
            'stitched_members':'audit/stitch_provenance.csv; read from actual Script42 stitch frame; verified exact output equality',
            'nms':'Script42.nms invoking unchanged Script14.class_aware_nms_indices; saved union index',
            'new_stitched_definition':'class/global xyxy not present among BASE stitched boxes for same image; score changes alone do not count'})
        all_metrics.append(metrics);exact_details[vid]=current
        print('END exact_combo '+json.dumps(metrics),flush=True)
    write_csv(ROOT/'audit/stitch_provenance.csv',stitch_rows)
    write_csv(ROOT/'audit/gt_attribution.csv',attributions)
    write_csv(ROOT/'audit/roundtrip_changes.csv',rtchanges,['config','image','gt_index','float_status','roundtrip_status','float_best_iou','roundtrip_best_iou'])
    v.write_json(ROOT/'audit/group_audit.json',all_group)
    group_csv=[{k:json.dumps(value,sort_keys=True) if isinstance(value,(dict,list)) else value for k,value in row.items()} for row in all_group]
    write_csv(ROOT/'audit/group_audit.csv',group_csv)
    v.write_json(ROOT/'audit/exact_rescue_overlap.json',overlap(exact_sets))
    v.write_json(ROOT/'audit/analysis_complete.json',{'status':'PASS','complete_views':list(v.VIEWS),'images_per_view':474,'gt_count':845,'runtime_seconds':time.time()-start,'assignment_schema':schema,
        'checks':{'all_caches_before_GT_analysis':'PASS','official_nms':'PASS','official_stitch_and_members_exact':'PASS','official_submission_roundtrip':'PASS','official_matcher':'PASS','all_GT_status_compared':'PASS','only_three_allowed_exact_configs':'PASS'},
        'official_code_sha256':{p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in ['scripts/14_fn_diagnostic.py','scripts/42_eval_baseline_global_exact_final_combo.py']}})

if __name__=='__main__':main()
