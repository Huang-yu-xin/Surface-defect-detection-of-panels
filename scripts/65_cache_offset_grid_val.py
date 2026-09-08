"""Frozen annotation-free X/Y/XY half-stride cache generator, reusing Script14 starts."""
import argparse,importlib.util,json,sys
from pathlib import Path

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
v=module('scripts/52_cache_vflip_factorial_val.py','offset_cache_base52')
v.ROOT=Path('results/offset_grid_halfstride')
v.VIEWS={'x':('X','x',0),'y':('Y','y',0),'xy':('XY','xy',0)}
OFFSET=v.PARAMS['stride']//2

def get_offset_starts(length,tile,stride,offset):
    if length<=tile:return [0]
    last=length-tile;starts={0,last};s=offset
    while s<last:starts.add(s);s+=stride
    return sorted(starts)

def tile_origins(height,width,grid_mode):
    ox=v.get_starts(width,v.PARAMS['tile_size'],v.PARAMS['stride']);oy=v.get_starts(height,v.PARAMS['tile_size'],v.PARAMS['stride'])
    hx=get_offset_starts(width,v.PARAMS['tile_size'],v.PARAMS['stride'],OFFSET);hy=get_offset_starts(height,v.PARAMS['tile_size'],v.PARAMS['stride'],OFFSET)
    xs=hx if grid_mode in {'x','xy'} else ox;ys=hy if grid_mode in {'y','xy'} else oy
    return [(x,y,min(v.PARAMS['tile_size'],width-x),min(v.PARAMS['tile_size'],height-y)) for y in ys for x in xs]
v.tile_origins=tile_origins

def enrich(view):
    for p in [v.ROOT/view/'cache/manifest.json',v.ROOT/view/'manifest.json']:
        m=json.loads(p.read_text());m.update(offset=OFFSET,offset_definition='stride//2',edge_anchor=True,orientation='original',offset_x=OFFSET if view in {'x','xy'} else 0,offset_y=OFFSET if view in {'y','xy'} else 0,global_box_slice=[2,6],local_box_slice=[10,14]);p.write_text(json.dumps(m,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--grid',choices=list(v.VIEWS),required=True);p.add_argument('--audit-only',action='store_true');a=p.parse_args()
    if a.audit_only:v.audit(a.grid)
    else:v.build(a.grid);enrich(a.grid)
if __name__=='__main__':main()
