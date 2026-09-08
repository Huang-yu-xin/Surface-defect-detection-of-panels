"""Independent read-only verification of offset-grid persisted artifacts."""
import argparse,csv,hashlib,json
from collections import Counter
from pathlib import Path
import numpy as np
csv.field_size_limit(32*1024*1024)
ROOT=Path('results/offset_grid_halfstride');VIEWS={'X':'x','Y':'y','XY':'xy'}
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def main(final=False):
    checks={};base=read(ROOT/'exact_combo/baseline/statuses.json');bs={(r['image'],int(r['gt_index'])):r for r in base};bm=read(ROOT/'exact_combo/baseline/metrics.json')
    assert len(bs)==845 and sum(r['float'] for r in bs.values())==831 and sum(r['roundtrip'] for r in bs.values())==831 and (bm['float_tp'],bm['float_fn'],bm['roundtrip_tp'],bm['roundtrip_fn'])==(831,14,831,14);checks['baseline_831_14_float_roundtrip']='PASS'
    context=read(ROOT/'audit/execution_context.json');assert context['test_hidden_run']=='NO'
    for d,h in context['source_manifest_sha256'].items():assert hashlib.sha256((Path(d)/'manifest.json').read_bytes()).hexdigest()==h
    assert hashlib.sha256(Path(context['weights']).read_bytes()).hexdigest()==context['weights_sha256'];checks['source_hashes']='PASS'
    om=read('results/fn_analysis/cache/manifest.json');items=om['items'];groups={r['image']:r['group_id'] for r in rows('splits/sample_assignments.csv')};attr=rows(ROOT/'audit/gt_attribution.csv');stitches=rows(ROOT/'audit/stitch_provenance.csv');folds=read(ROOT/'audit/fold_audit.json')
    total_source=total_stitch=segments=0
    exact_sets={}
    for vid,view in VIEWS.items():
        cm=read(ROOT/view/'cache/manifest.json');ca=read(ROOT/'audit'/f'cache_{view}.json');assert cm['image_count']==474 and len(cm['items'])==474 and ca['status']=='PASS' and ca['coord_residual']<=1e-4
        assert len(list((ROOT/view/'cache').glob('*.npz')))==474;count=0;residual=0
        for item in cm['items']:
            with np.load(ROOT/view/'cache'/item['cache_file']) as z:a=z['candidates'];count+=len(a);residual=max(residual,float(np.max(np.abs(a[:,2:6].astype(float)-a[:,10:14].astype(float)-a[:,[6,7,6,7]].astype(float)))) if len(a) else 0)
        assert count==cm['candidate_rows'] and residual<=1e-4
        out=ROOT/'exact_combo'/view;m=read(out/'metrics.json');state={(r['image'],int(r['gt_index'])):r for r in read(out/'comparison_state.json')};assert set(state)==set(bs)
        assert sum(r['float'] for r in state.values())==m['float_tp'] and sum(r['roundtrip'] for r in state.values())==m['roundtrip_tp'];assert len(list((out/'final_cache').glob('*.npz')))==474 and len(list((out/'roundtrip_json').glob('*.json')))==474 and len(list((out/'provenance').glob('*.npz')))==474
        rescued={k for k in bs if not bs[k]['float'] and state[k]['float']};reg={k for k in bs if bs[k]['float'] and not state[k]['float']};exact_sets[vid]=rescued;g=m['float_group_audit'];assert (len(rescued),len(reg),len(rescued)-len(reg))==(g['rescued'],g['regressed'],g['net_gain'])
        for rem in g['all_largest_group_removals']:
            assert rem['rescued']==sum(groups[k[0]]!=rem['group'] for k in rescued);assert rem['regressed']==sum(groups[k[0]]!=rem['group'] for k in reg)
        roots=[Path('results/baseline_complementarity/global_exact_final_combo/u1e3_g060')]*2+[Path('results/fn_analysis/cache_hflip'),ROOT/view/'cache'];byimage={}
        for r in stitches:
            if r['config']==vid:byimage.setdefault(r['image'],[]).append(r)
        final_count=stitched_count=0
        for item in items:
            fn=item['cache_file'];name=item['image_name']
            with np.load(out/'final_cache'/fn) as z:arr=z['final'];post=int(z['post_count'])
            with np.load(out/'provenance'/fn) as z:source=z['post_source_id'];source_row=z['source_cache_row']
            assert len(source)==len(source_row)==post
            for sid in np.unique(source):
                mask=source==sid
                with np.load(roots[int(sid)]/fn) as z:src=z['candidates']
                assert np.array_equal(arr[:post][mask],src[source_row[mask]])
            total_source+=post;final_count+=len(arr);stitched_count+=len(arr)-post;assert len(byimage.get(name,[]))==len(arr)-post
            for r in byimage.get(name,[]):
                ii=int(r['final_row']);members=json.loads(r['segments']);boxes=np.asarray([arr[int(x['post_nms_row'])] for x in members]);expected=np.asarray([np.median(boxes[:,2]),np.min(boxes[:,3]),np.median(boxes[:,4]),np.max(boxes[:,5])],np.float32);assert np.array_equal(arr[ii,2:6],expected);segments+=len(members);total_stitch+=1
        assert final_count==m['final_detection_count'] and stitched_count==m['stitched_count']
        sf=next(r for r in folds if r['config']==vid and r['fold']=='summary');nets=[]
        for f in range(3):
            rr=sum(int(hashlib.sha256(groups[k[0]].encode()).hexdigest()[:8],16)%3==f for k in rescued);gg=sum(int(hashlib.sha256(groups[k[0]].encode()).hexdigest()[:8],16)%3==f for k in reg);nets.append(rr-gg)
        assert list(map(int,sf['fold_nets']))==nets
    checks.update(caches_474x3_and_manifests='PASS',global_local_residual='PASS',exact_metrics_and_roundtrip_files='PASS',all_post_nms_source_rows='PASS',stitch_provenance='PASS',group_and_fold_recomputed='PASS')
    if final:
        summary=rows(ROOT/'summary.csv');report=(ROOT/'report.md').read_text();assert len(summary)==3 and all(r['config'] in report for r in summary);assert '831/14' in report and 'Test/Hidden run = **NO**' in report;checks['summary_report_reconciliation']='PASS'
    result={'status':'PASS','mode':'final' if final else 'pre-report','checks':checks,'post_nms_rows_verified':total_source,'stitched_boxes_verified':total_stitch,'segment_records_verified':segments}
    target=ROOT/'audit'/('independent_verification.json' if final else 'independent_preverification.json');target.write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--final',action='store_true');a=p.parse_args();main(a.final)
