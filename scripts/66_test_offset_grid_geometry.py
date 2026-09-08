"""CPU-only pre-inference geometry suite for half-stride grids."""
import importlib.util,json,sys,time
from pathlib import Path
import cv2,numpy as np
spec=importlib.util.spec_from_file_location('offset65','scripts/65_cache_offset_grid_val.py');v=importlib.util.module_from_spec(spec);sys.modules[spec.name]=v;spec.loader.exec_module(v)

def covered(length,starts,tile):
    end=0
    for s in starts:
        if s>end:return False
        end=max(end,min(length,s+tile))
    return end>=length

def main():
    t=time.time();om,items=v.v.val_items();checks={};samples=[]
    assert v.v.get_starts is v.v.DIAG.get_starts;checks['original_get_starts_reused']='PASS'
    assert v.get_offset_starts(4096,1280,768,384)==[0,384,1152,1920,2688,2816]
    assert v.get_offset_starts(3000,1280,768,384)==[0,384,1152,1720];checks['offset_formula_synthetic']='PASS'
    for n in [1,1279,1280]:assert v.get_offset_starts(n,1280,768,384)==[0]
    checks['small_image_single_start']='PASS'
    for item in items:
        h,w=item['height'],item['width']
        for view in v.v.VIEWS:
            origins=v.tile_origins(h,w,view);assert len(origins)==len(set(origins))
            xs=sorted({x for x,y,vw,vh in origins});ys=sorted({y for x,y,vw,vh in origins});assert covered(w,xs,1280) and covered(h,ys,1280)
    checks['no_duplicate_origins_all_474']='PASS';checks['complete_coverage_all_474']='PASS'
    for index in [1,100,200,300,474]:
        item=sorted(items,key=lambda z:z['image_name'])[index-1];image=cv2.imread(str(v.v.IMAGES/item['image_name']));assert image is not None;count=0
        for view in v.v.VIEWS:
            for meta in v.tile_origins(*image.shape[:2],view):
                x,y,w,h=meta;assert np.array_equal(v.v.patch(image,meta,0,pad=False),image[y:y+h,x:x+w]);count+=1
        samples.append({'index_one_based':index,'image':item['image_name'],'patches':count,'pixel_identical':True})
    checks['pixel_identity_fixed_five_images']='PASS'
    boxes=np.asarray([[1,2,31,81],[0,0,100,120]],np.float32);rows=v.v.pack_rows(boxes,np.asarray([0,1]),np.asarray([.1,.2]),(384,1152,1280,1280),0)
    assert np.max(np.abs(rows[:,2:6].astype(np.float64)-rows[:,10:14].astype(np.float64)-rows[:,[6,7,6,7]].astype(np.float64)))<=1e-4;checks['global_local_invariant']='PASS'
    out={'status':'PASS','offset':384,'checks':checks,'sample_rule':'sorted filenames 1,100,200,300,474 before GT','samples':samples,'gt_accessed':False,'runtime_seconds':time.time()-t}
    v.v.write_json(v.v.ROOT/'audit/geometry_tests.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
