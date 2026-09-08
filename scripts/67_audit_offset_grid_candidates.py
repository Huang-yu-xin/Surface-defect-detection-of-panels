"""GT-free tile novelty, multiplicity, and candidate distribution audit."""
import csv,importlib.util,json,sys,time
from collections import Counter
from pathlib import Path
import numpy as np
spec=importlib.util.spec_from_file_location('offset65audit','scripts/65_cache_offset_grid_val.py');v=importlib.util.module_from_spec(spec);sys.modules[spec.name]=v;spec.loader.exec_module(v)
ROOT=v.v.ROOT;O=v.v.ORIGINAL
def write_csv(path,rows):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def stats(a):
    a=np.asarray(a,float);return {k:float(x) for k,x in zip(['mean','p50','p90','p95','p99','max'],[a.mean(),*np.percentile(a,[50,90,95,99]),a.max()])}
def main():
    t=time.time();om,items=v.v.val_items();novel=[];tiles=[];summary={};dist=[]
    o_counts=[];o_candidates=[]
    for item in items:
        oo=set(v.v.tile_origins(item['height'],item['width'],'original'));o_counts.append(len(oo));o_candidates.append(item['candidate_count'])
        for view in v.v.VIEWS:
            vv=set(v.tile_origins(item['height'],item['width'],view));inter=oo&vv;union=oo|vv
            novel.append({'view':view.upper(),'image':item['image_name'],'original_tiles':len(oo),'view_tiles':len(vv),'intersection':len(inter),'new_origins':len(vv-oo),'origin_jaccard':len(inter)/len(union),'new_origin_fraction':len(vv-oo)/len(vv),'tile_count_ratio':len(vv)/len(oo)})
            tiles.append({'view':view.upper(),'image':item['image_name'],'original_tiles':len(oo),'view_tiles':len(vv),'ratio_to_o':len(vv)/len(oo)})
    for view in v.v.VIEWS:
        m=json.loads((ROOT/view/'cache/manifest.json').read_text());counts=np.asarray([x['candidate_count'] for x in m['items']]);tile_counts=np.asarray([len(x['tile_origins']) for x in m['items']]);classes=Counter()
        for item in m['items']:
            with np.load(ROOT/view/'cache'/item['cache_file']) as z:a=z['candidates'];classes.update(a[:,0].astype(int).tolist())
        rows=[r for r in novel if r['view']==view.upper()];cs=stats(counts);ts=stats(tile_counts);ratios=np.asarray([r['tile_count_ratio'] for r in rows]);j=np.asarray([r['origin_jaccard'] for r in rows]);nf=np.asarray([r['new_origin_fraction'] for r in rows])
        summary[view.upper()]={'images':474,'candidate_rows':int(counts.sum()),'tile_rows':int(tile_counts.sum()),'candidate_per_image':float(counts.mean()),'candidate_per_tile':float(counts.sum()/tile_counts.sum()),'candidate_ratio_to_o':float(counts.sum()/sum(o_candidates)),'tile_count_ratio_to_o':float(tile_counts.sum()/sum(o_counts)),'candidate_per_tile_ratio_to_o':float((counts.sum()/tile_counts.sum())/(sum(o_candidates)/sum(o_counts))),'per_image':cs,'tiles_per_image':ts,'class_counts':[classes[i] for i in range(9)],'top2_image_concentration':float(np.sort(counts)[-2:].sum()/counts.sum()),'origin_jaccard':stats(j),'new_origin_fraction':stats(nf),'tile_ratio_per_image':stats(ratios)}
        dist.append({'view':view.upper(),**{k:summary[view.upper()][k] for k in ['images','candidate_rows','tile_rows','candidate_per_image','candidate_per_tile','candidate_ratio_to_o','tile_count_ratio_to_o','candidate_per_tile_ratio_to_o','top2_image_concentration']},'per_image_p50':cs['p50'],'per_image_p90':cs['p90'],'per_image_p95':cs['p95'],'per_image_p99':cs['p99'],'per_image_max':cs['max'],'class_counts':json.dumps(summary[view.upper()]['class_counts'])})
    write_csv(ROOT/'audit/tile_origin_novelty.csv',novel);write_csv(ROOT/'audit/tile_count_audit.csv',tiles);write_csv(ROOT/'audit/candidate_distribution.csv',dist)
    out={'status':'PASS','ground_truth_accessed':False,'views':summary,'checks':{'all_cache_audits_pass':all(json.loads((ROOT/'audit'/f'cache_{x}.json').read_text())['status']=='PASS' for x in v.v.VIEWS),'novel_origins_present':all(summary[x]['new_origin_fraction']['mean']>0 for x in summary),'candidate_per_tile_not_order_of_magnitude_abnormal':all(.1<=summary[x]['candidate_per_tile_ratio_to_o']<=10 for x in summary)},'runtime_seconds':time.time()-t}
    assert all(out['checks'].values());(ROOT/'audit/candidate_summary.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
