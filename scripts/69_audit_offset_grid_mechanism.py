"""Post-hoc GT geometry exposure plus group and deterministic fold robustness."""
import csv,hashlib,importlib.util,json,math,sys
from pathlib import Path
import numpy as np
ROOT=Path('results/offset_grid_halfstride');VIEWS={'X':'x','Y':'y','XY':'xy'}
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
o=module('scripts/65_cache_offset_grid_val.py','offset69_views');official=module('scripts/42_eval_baseline_global_exact_final_combo.py','offset69_official');diag=o.v.DIAG
def read(p):return json.loads(Path(p).read_text())
def rows(p):
    with Path(p).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write_csv(p,data):
    fields=list(dict.fromkeys(k for row in data for k in row))
    with Path(p).open('w',encoding='utf-8-sig',newline='') as f:w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(data)
def exposure(box,h,w,grid):
    origins=o.v.tile_origins(h,w,'original') if grid=='O' else o.tile_origins(h,w,grid.lower());x1,y1,x2,y2=map(float,box);area=max(1e-12,(x2-x1)*(y2-y1));visible=[]
    for tx,ty,tw,th in origins:visible.append(max(0,min(x2,tx+tw)-max(x1,tx))*max(0,min(y2,ty+th)-max(y1,ty))/area)
    xb=sorted({z for tx,ty,tw,th in origins for z in (tx,tx+tw) if 0<z<w});yb=sorted({z for tx,ty,tw,th in origins for z in (ty,ty+th) if 0<z<h})
    def d(vals,coords):return min((abs(a-b) for a in vals for b in coords),default=None)
    return {'max_visible_fraction':max(visible),'num_intersecting_tiles':sum(z>0 for z in visible),'center_to_vertical_boundary':d(xb,[(x1+x2)/2]),'edge_to_vertical_boundary':d(xb,[x1,x2]),'center_to_horizontal_boundary':d(yb,[(y1+y2)/2]),'edge_to_horizontal_boundary':d(yb,[y1,y2])}
def main():
    base={(r['image'],int(r['gt_index'])):r for r in read(ROOT/'exact_combo/baseline/statuses.json')};fn={k for k,v in base.items() if not v['float']};states={k:{(r['image'],int(r['gt_index'])):r for r in read(ROOT/f'exact_combo/{view}/comparison_state.json')} for k,view in VIEWS.items()}
    groups={r['image']:r['group_id'] for r in rows('splits/sample_assignments.csv')};args=type('A',(),{'assignments':Path('splits/sample_assignments.csv'),'failure_csv':Path('results/final_combo_fn21/remaining_fn_21.csv')})();_,failures,_=official.read_maps(args)
    regressions={vid:{k for k,b in base.items() if b['float'] and not states[vid][k]['float']} for vid in VIEWS};targets=fn|set().union(*regressions.values());geom=[];details=[]
    manifests=read('results/fn_analysis/cache/manifest.json');items={x['image_name']:x for x in manifests['items']}
    direct={(r['view'],r['image'],int(r['gt_index'])):r for r in rows(ROOT/'audit/direct_rescue.csv')}
    for key in sorted(targets):
        b=base[key];item=items[key[0]];box=b['bbox'];aspect=(box[2]-box[0])/max(1e-12,box[3]-box[1]);area=(box[2]-box[0])*(box[3]-box[1])
        ex={g:exposure(box,item['height'],item['width'],g) for g in ['O','X','Y','XY']}
        for g in ex:geom.append({'image':key[0],'gt_index':key[1],'group':groups[key[0]],'class_id':b['class_id'],'failure':failures.get(key,'unclassified'),'bbox':json.dumps(box),'aspect':aspect,'area':area,'grid':g,**ex[g],'is_current_base_fn':key in fn,'rescued_by_grid':g in VIEWS and (not b['float'] and states[g][key]['float']),'regressed_by_grid':g in VIEWS and (b['float'] and not states[g][key]['float'])})
        details.append({'image':key[0],'gt_index':key[1],'group':groups[key[0]],'class_id':b['class_id'],'failure':failures.get(key,'unclassified'),'bbox':json.dumps(box),'aspect':aspect,'area':area,'base_iou':b['float_best']['iou'],
          **{f'{g}_iou':float(direct.get((g,key[0],key[1]),{}).get('best_iou') or 0) for g in VIEWS},**{f'{g}_max_visible':ex[g]['max_visible_fraction'] for g in ['O','X','Y','XY']},**{f'{g}_intersecting_tiles':ex[g]['num_intersecting_tiles'] for g in ['O','X','Y','XY']},
          **{f'{g}_center_v_boundary':ex[g]['center_to_vertical_boundary'] for g in ['O','X','Y','XY']},**{f'{g}_center_h_boundary':ex[g]['center_to_horizontal_boundary'] for g in ['O','X','Y','XY']},
          'exact_rescued':','.join(g for g in VIEWS if not b['float'] and states[g][key]['float']) or 'none','regressed':','.join(g for g in VIEWS if b['float'] and not states[g][key]['float']) or 'none'})
    write_csv(ROOT/'audit/gt_geometry_exposure.csv',geom);write_csv(ROOT/'audit/gt_detailed_audit.csv',details)
    folds=[]
    for vid,view in VIEWS.items():
        rescued={k for k,b in base.items() if not b['float'] and states[vid][k]['float']};reg=regressions[vid];nets=[]
        for fold in range(3):
            r=sum(int(hashlib.sha256(groups[k[0]].encode()).hexdigest()[:8],16)%3==fold for k in rescued);g=sum(int(hashlib.sha256(groups[k[0]].encode()).hexdigest()[:8],16)%3==fold for k in reg);nets.append(r-g);folds.append({'config':vid,'fold':fold,'rescued':r,'regressed':g,'net_gain':r-g})
        folds.append({'config':vid,'fold':'summary','rescued':len(rescued),'regressed':len(reg),'net_gain':len(rescued)-len(reg),'fold_nets':nets,'positive_folds':sum(x>0 for x in nets),'negative_folds':sum(x<0 for x in nets)})
    write_csv(ROOT/'audit/fold_audit.csv',folds);(ROOT/'audit/fold_audit.json').write_text(json.dumps(folds,indent=2));print(json.dumps({'status':'PASS','targets':len(targets),'folds':folds},indent=2))
if __name__=='__main__':main()
