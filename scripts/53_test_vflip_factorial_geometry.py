"""CPU-only geometry checks; sample images fixed before reading any GT."""
import importlib.util,json,sys,time
from pathlib import Path
import cv2
import numpy as np

spec=importlib.util.spec_from_file_location('factorial_geometry','scripts/52_cache_vflip_factorial_val.py')
v=importlib.util.module_from_spec(spec);sys.modules[spec.name]=v;spec.loader.exec_module(v)

def main():
    start=time.time();om,items=v.val_items();checks={};samples=[];patches=0
    assert v.get_starts is v.DIAG.get_starts
    for h,w in [(3000,4096),(733,951),(1280,1280),(1281,1700),(2048,1399),(3599,4011)]:
        standard=[(x,y,min(1280,w-x),min(1280,h-y)) for y in v.get_starts(h,1280,768) for x in v.get_starts(w,1280,768)]
        assert v.tile_origins(h,w,'original')==standard
        mapped={(x,h-(y+vh),vw,vh) for x,y,vw,vh in standard}
        assert set(v.tile_origins(h,w,'mirrored_y'))==mapped
        rng=np.random.default_rng(h+w);image=rng.integers(0,256,(h,w,3),dtype=np.uint8)
        for meta in v.tile_origins(h,w,'mirrored_y'):
            x,y,vw,vh=meta
            assert np.array_equal(v.patch(image,meta,0,pad=False),image[y:y+vh,x:x+vw])
        flipped=cv2.flip(image,0)
        for x,s,vw,vh in standard:
            a=v.patch(flipped,(x,s,vw,vh),0)
            b=v.patch(image,(x,h-(s+vh),vw,vh),1)
            assert np.array_equal(a,b);patches+=1
        boxes=np.asarray([[0,0,vw,vh],[1,2,min(vw,31),min(vh,81)]],np.float32)
        assert np.array_equal(v.inverse_local(v.inverse_local(boxes,vh),vh),boxes)
        rows=v.pack_rows(boxes,np.asarray([0,1]),np.asarray([.1,.2]),(184,952,vw,vh),1)
        assert np.max(np.abs(rows[:,2:6].astype(np.float64)-rows[:,10:14].astype(np.float64)-rows[:,[6,7,6,7]].astype(np.float64)))<=1e-4
    checks.update(original_grid_synthetic='PASS',mirrored_y_general_formula='PASS',grid_only_pixels_unflipped='PASS',local_vertical_flip_roundtrip='PASS',global_local_nonzero_origin='PASS',synthetic_full_patch_equivalence_including_padding='PASS')
    actual_tiles=0
    for item in items:
        origins=set(v.tile_origins(item['height'],item['width'],'original'))
        with np.load(v.ORIGINAL/item['cache_file']) as z:
            a=z['candidates'];shape=z['image_shape']
            assert tuple(shape)==(item['height'],item['width'])
            assert set(map(tuple,np.unique(a[:,6:10],axis=0))).issubset(origins)
        actual_tiles+=len(origins)
    checks['D_origins_match_official_O_all_474']='PASS'
    for index in [1,100,200,300,474]:
        item=sorted(items,key=lambda x:x['image_name'])[index-1]
        image=cv2.imread(str(v.IMAGES/item['image_name']),cv2.IMREAD_COLOR);assert image is not None
        h,w=image.shape[:2];flipped=cv2.flip(image,0);count=0
        for x,s,vw,vh in v.tile_origins(h,w,'original'):
            assert np.array_equal(v.patch(flipped,(x,s,vw,vh),0),v.patch(image,(x,h-(s+vh),vw,vh),1))
            count+=1;patches+=1
        samples.append({'index_one_based':index,'image':item['image_name'],'patches':count,'pixel_identical':True})
    checks['F_literal_whole_image_vflip_five_fixed_val_images']='PASS'
    result={'status':'PASS','checks':checks,'sample_rule':'sorted filenames at 1,100,200,300,474; no GT selection','samples':samples,'patches_compared':patches,'original_tiles_all_val':actual_tiles,'gt_accessed':False,'runtime_seconds':time.time()-start}
    v.write_json(v.ROOT/'audit/geometry_tests.json',result);print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
