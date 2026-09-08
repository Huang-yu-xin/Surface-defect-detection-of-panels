"""Independent checks of the persisted exact outputs, provenance and group counts."""
import csv,hashlib,importlib.util,json,sys,time
from collections import Counter
from pathlib import Path
import numpy as np
csv.field_size_limit(16*1024*1024)

ROOT=Path('results/vflip_factorial');NAMES={'D':'direction','G':'grid','F':'full'}
def read(path):return json.loads(Path(path).read_text())
def rows(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main():
    start=time.time();base=read(ROOT/'exact_combo/baseline/statuses.json');bs={(r['image'],r['gt_index']):r for r in base}
    context=read(ROOT/'audit/execution_context.json')
    for directory,digest in context['source_manifest_sha256'].items():
        assert hashlib.sha256((Path(directory)/'manifest.json').read_bytes()).hexdigest()==digest
    weight_digest=hashlib.sha256(Path(context['weights']).read_bytes()).hexdigest()
    for view in NAMES.values():assert read(ROOT/view/'manifest.json')['weights_sha256']==weight_digest
    om=read('results/fn_analysis/cache/manifest.json');items=om['items'];assert len(items)==474
    groups={r['image']:r['group_id'] for r in rows('splits/sample_assignments.csv')}
    attr=rows(ROOT/'audit/gt_attribution.csv');direct=rows(ROOT/'audit/direct_rescue.csv')
    overlap=read(ROOT/'audit/rescue_overlap.json');summary=rows(ROOT/'summary.csv') if (ROOT/'summary.csv').exists() else []
    spec=importlib.util.spec_from_file_location('verify_official42','scripts/42_eval_baseline_global_exact_final_combo.py');official=importlib.util.module_from_spec(spec);spec.loader.exec_module(official)
    checked_post=0;checked_stitches=0;checked_segments=0;checked_json=0;exact_sets={}
    provenance_rows=rows(ROOT/'audit/stitch_provenance.csv')
    for vid,view in NAMES.items():
        out=ROOT/'exact_combo'/view;m=read(out/'metrics.json');state={(r['image'],r['gt_index']):r for r in read(out/'comparison_state.json')}
        assert set(state)==set(bs)
        assert len(list((out/'final_cache').glob('*.npz')))==474
        assert len(list((out/'roundtrip_json').glob('*.json')))==474
        assert len(list((out/'provenance').glob('*.npz')))==474
        assert sum(r['float'] for r in state.values())==m['float_tp']
        assert sum(r['roundtrip'] for r in state.values())==m['roundtrip_tp']
        assert sum(r['float']!=r['roundtrip'] for r in state.values())==m['roundtrip_gt_status_changes']
        rescues={k for k in bs if not bs[k]['float'] and state[k]['float']};regressions={k for k in bs if bs[k]['float'] and not state[k]['float']};exact_sets[vid]=rescues
        assert len(rescues)==m['float_group_audit']['rescued'] and len(regressions)==m['float_group_audit']['regressed']
        allowed={k for k in bs if not bs[k]['float']}|regressions
        assert {(a['image'],int(a['gt_index'])) for a in attr if a['config']==vid}==allowed
        for rep in ['float','roundtrip']:
            ar=m[rep+'_group_audit'];rs={k for k in bs if not bs[k][rep] and state[k][rep]};rg={k for k in bs if bs[k][rep] and not state[k][rep]}
            assert ar['rescues_by_group']==dict(Counter(groups[k[0]] for k in rs))
            for removal in ar['all_largest_group_removals']:
                assert removal['rescued']==sum(groups[k[0]]!=removal['group'] for k in rs)
                assert removal['regressed']==sum(groups[k[0]]!=removal['group'] for k in rg)
                assert removal['net_gain']==removal['rescued']-removal['regressed']
        direct_set={(r['image'],int(r['gt_index'])) for r in direct if r['view']==vid and float(r['best_iou'])>=.5}
        assert direct_set==set(map(tuple,overlap['sets'][vid]))
        source_roots=[Path('results/baseline_complementarity/global_exact_final_combo/u1e3_g060')]*2+[Path('results/fn_analysis/cache_hflip'),ROOT/view/'cache']
        source_names=['O','Baseline','H',vid]
        byimage={}
        for r in provenance_rows:
            if r['config']==vid:byimage.setdefault(r['image'],[]).append(r)
        count_final=0;count_stitched=0;count_new=0;count_newview=0;count_mixed=0
        for index,item in enumerate(items,1):
            name=item['image_name'];filename=item['cache_file']
            with np.load(out/'final_cache'/filename) as z:final=z['final'];post_count=int(z['post_count'])
            with np.load(out/'provenance'/filename) as z:source=z['post_source_id'];source_row=z['source_cache_row']
            assert len(source)==len(source_row)==post_count
            for source_id in np.unique(source):
                mask=source==source_id
                with np.load(source_roots[int(source_id)]/filename) as z:a=z['candidates']
                assert np.array_equal(final[:post_count][mask],a[source_row[mask]]), (vid,name,int(source_id))
            checked_post+=post_count;count_final+=len(final);count_stitched+=len(final)-post_count
            assert len(byimage.get(name,[]))==len(final)-post_count
            for r in byimage.get(name,[]):
                i=int(r['final_row']);assert i==post_count+int(r['stitched_index'])
                assert np.array_equal(final[i,2:6],np.asarray(json.loads(r['box']),np.float32))
                segments=json.loads(r['segments']);assert len(segments)==int(r['segment_count'])
                members=[]
                for member in segments:
                    j=member['post_nms_row'];assert member['view_id']==source_names[int(source[j])]
                    assert member['source_cache_row']==int(source_row[j])
                    assert member['class']==int(final[j,0])==1
                    assert (member['tile_x'],member['tile_y'])==tuple(final[j,6:8])
                    assert np.array_equal(np.asarray(member['bbox'],np.float32),final[j,2:6]);members.append(final[j]);checked_segments+=1
                members=np.asarray(members)
                expected=[np.median(members[:,2]),np.min(members[:,3]),np.median(members[:,4]),np.max(members[:,5])]
                assert np.array_equal(final[i,2:6],np.asarray(expected,np.float32))
                views={x['view_id'] for x in segments}
                assert (r['mixed_source']=='True')==(len(views)>1)
                assert (r['new_view_participates']=='True')==(vid in views)
                count_new+=r['new_relative_to_baseline']=='True';count_newview+=vid in views;count_mixed+=len(views)>1;checked_stitches+=1
            if index in [1,100,200,300,474]:
                rt=read(out/'roundtrip_json'/f'{Path(name).stem}.json')
                assert rt==[official.submission_row(name,item['width'],item['height'],det) for det in final]
                checked_json+=len(rt)
        assert count_final==m['final_detection_count'] and count_stitched==m['stitched_count']
        assert count_new==m['new_stitched'] and count_newview==m['new_view_participating_stitched'] and count_mixed==m['mixed_source_stitched']
        print(f'VERIFY {vid} PASS all_post_nms_source_rows={checked_post} cumulative_stitches={checked_stitches}',flush=True)
    eo=read(ROOT/'audit/exact_rescue_overlap.json')
    for vid,s in exact_sets.items():assert s==set(map(tuple,eo['sets'][vid]))
    for label,a,b in [('D∩G','D','G'),('D∩F','D','F'),('G∩F','G','F')]:assert eo['counts'][label]==len(exact_sets[a]&exact_sets[b])
    result={'status':'PASS','runtime_seconds':time.time()-start,'post_nms_rows_verified_against_source_cache':checked_post,'stitched_boxes_verified':checked_stitches,
        'segment_records_verified':checked_segments,'deterministic_roundtrip_sample_rows_verified':checked_json,
        'checks':{'frozen_source_manifests_and_weights':'PASS','474_artifacts_each_config':'PASS','recompute_counts_from_saved_GT_states':'PASS','attribution_only_FN_and_regressions':'PASS','recompute_group_removal':'PASS',
            'proposal_and_exact_overlap_sets':'PASS','every_post_nms_row_exact_source_cache_match':'PASS','every_stitched_box_and_segment_provenance':'PASS','formal_roundtrip_rows_fixed_samples':'PASS'}}
    (ROOT/'audit/independent_verification.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
