"""Run only BASE+X, BASE+Y and BASE+XY through unchanged formal NMS/stitch/matcher/roundtrip."""
import importlib.util,sys
from pathlib import Path
def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
o=module('scripts/65_cache_offset_grid_val.py','offset68_views');a=module('scripts/54_analyze_vflip_factorial_val.py','offset68_analysis')
a.v.ROOT=o.v.ROOT;a.ROOT=o.v.ROOT;a.v.VIEWS=o.v.VIEWS;a.v.tile_origins=o.tile_origins
def overlap(sets):
    x,y,xy=[sets[k] for k in ['X','Y','XY']];rows={'X':x,'Y':y,'XY':xy,'X-only':x-y-xy,'Y-only':y-x-xy,'XY-only':xy-x-y,'X∩Y':x&y,'X∩XY':x&xy,'Y∩XY':y&xy,'X∩Y∩XY':x&y&xy};j={}
    for p,q in [('X','Y'),('X','XY'),('Y','XY')]:
        u=sets[p]|sets[q];j[f'J({p},{q})']=len(sets[p]&sets[q])/len(u) if u else None
    return {'counts':{k:len(z) for k,z in rows.items()},'sets':{k:[list(t) for t in sorted(z)] for k,z in rows.items()},'jaccard':j,'empty_union_jaccard':'undefined/null'}
a.overlap=overlap
if __name__=='__main__':a.main()
