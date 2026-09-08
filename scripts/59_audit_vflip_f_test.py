"""CPU geometry and cache/distribution audit for frozen Test F; no labels/GT."""
import argparse,csv,json,time
from collections import Counter
from pathlib import Path
import cv2
import numpy as np
import importlib.util,sys

def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
T=module("scripts/58_cache_vflip_f_test.py","frozen_f_test")
ROOT=T.ROOT
def stats(a):
 a=np.asarray(a,float);return {"mean":float(a.mean()),"p50":float(np.percentile(a,50)),"p90":float(np.percentile(a,90)),"p95":float(np.percentile(a,95)),"p99":float(np.percentile(a,99)),"max":float(a.max())}
def geometry():
 start=time.time();om,items=T.test_items();checks={};samples=[];patches=0
 for h,w in [(3000,4096),(733,951),(1280,1280),(1281,1700),(2048,1399),(3599,4011)]:
  std=[(x,y,min(1280,w-x),min(1280,h-y)) for y in T.V.get_starts(h,1280,768) for x in T.V.get_starts(w,1280,768)]
  assert T.V.tile_origins(h,w,"original")==std
  assert set(T.V.tile_origins(h,w,"mirrored_y"))=={(x,h-(y+vh),vw,vh) for x,y,vw,vh in std}
  rng=np.random.default_rng(h+w);im=rng.integers(0,256,(h,w,3),dtype=np.uint8);flip=cv2.flip(im,0)
  for x,y,vw,vh in std:
   assert np.array_equal(T.V.patch(flip,(x,y,vw,vh),0),T.V.patch(im,(x,h-(y+vh),vw,vh),1));patches+=1
  b=np.asarray([[0,0,vw,vh],[1,2,min(vw,31),min(vh,81)]],np.float32)
  assert np.array_equal(T.V.inverse_local(T.V.inverse_local(b,vh),vh),b)
  rows=T.V.pack_rows(b,np.asarray([0,1]),np.asarray([.1,.2]),(184,952,vw,vh),1)
  assert np.max(np.abs(rows[:,2:6].astype(float)-rows[:,10:14].astype(float)-rows[:,[6,7,6,7]].astype(float)))<=1e-4
 checks={k:"PASS" for k in ["mirrored_y_general_formula","local_vertical_flip_roundtrip","global_local_nonzero_origin","synthetic_full_patch_equivalence_including_padding"]}
 fixed=sorted(set([0,round((len(items)-1)*.25),round((len(items)-1)*.5),round((len(items)-1)*.75),len(items)-1]))
 for idx in fixed:
  item=sorted(items,key=lambda x:x["image_name"])[idx];im=cv2.imread(str(T.IMAGES/item["image_name"]));assert im is not None
  h,w=im.shape[:2];flip=cv2.flip(im,0);count=0
  for x,y,vw,vh in T.V.tile_origins(h,w,"original"):
   assert np.array_equal(T.V.patch(flip,(x,y,vw,vh),0),T.V.patch(im,(x,h-(y+vh),vw,vh),1));count+=1;patches+=1
  samples.append({"index_one_based":idx+1,"image":item["image_name"],"patches":count,"pixel_identical":True})
 checks["F_literal_whole_image_vflip_five_fixed_test_images"]="PASS"
 result={"status":"PASS","checks":checks,"sample_rule":"sorted filenames: first, quarter, half, three-quarter, last; GT-independent","samples":samples,"patches_compared":patches,"gt_accessed":False,"runtime_seconds":time.time()-start}
 T.write_json(ROOT/"audit/geometry_tests.json",result);print(json.dumps(result,indent=2))
def audit():
 start=time.time();om,items=T.test_items();m=T.read_json(ROOT/"f_cache/manifest.json");assert m["image_count"]==669
 assert [x["image_name"] for x in m["items"]]==[x["image_name"] for x in items]
 assert {p.name for p in (ROOT/"f_cache").glob("*.npz")}=={x["cache_file"] for x in items}
 counts=[];classes=Counter();residual=0.;bad=Counter()
 for item in items:
  with np.load(ROOT/"f_cache"/item["cache_file"]) as z:a=z["candidates"];shape=z["image_shape"]
  assert a.ndim==2 and a.shape[1]==14 and a.dtype==np.float32 and np.array_equal(shape,[item["height"],item["width"]]) and np.isfinite(a).all()
  assert len(a)==next(x for x in m["items"] if x["image_name"]==item["image_name"])["candidate_count"]
  if len(a):
   assert ((a[:,0]>=0)&(a[:,0]<9)&(a[:,0]==np.floor(a[:,0]))).all() and ((a[:,1]>=1e-5)&(a[:,1]<=1)).all()
   assert (a[:,4]>a[:,2]).all() and (a[:,5]>a[:,3]).all() and (a[:,2:6]>=0).all()
   assert (a[:,[2,4]]<=shape[1]).all() and (a[:,[3,5]]<=shape[0]).all()
   r=float(np.max(np.abs(a[:,2:6].astype(float)-a[:,10:14].astype(float)-a[:,[6,7,6,7]].astype(float))));residual=max(residual,r);assert r<=1e-4
   classes.update(a[:,0].astype(int).tolist())
  counts.append(len(a))
 val=T.read_json("results/vflip_factorial/full/cache/manifest.json");val_counts=[x["candidate_count"] for x in val["items"]]
 o_total=om["total_candidates"];val_o=T.read_json("results/fn_analysis/cache/manifest.json")["total_candidates"]
 ratio=(sum(counts)/669)/(sum(val_counts)/474);assert .5<=ratio<=2.0
 top=sorted(zip([x["image_name"] for x in items],counts),key=lambda x:x[1],reverse=True)
 result={"status":"PASS","image_count":669,"candidate_rows":sum(counts),"coord_residual":residual,"nan_inf_count":0,"invalid_bbox_count":0,
  "class_counts":{str(i):classes[i] for i in range(9)},"per_image":stats(counts),"top20":[{"image":n,"count":c} for n,c in top[:20]],"top2_concentration":sum(c for _,c in top[:2])/sum(counts),
  "val":{"images":474,"candidate_rows":sum(val_counts),"candidates_per_image":sum(val_counts)/474,"f_over_o":sum(val_counts)/val_o,"per_image":stats(val_counts)},
  "test":{"images":669,"candidate_rows":sum(counts),"candidates_per_image":sum(counts)/669,"f_over_o":sum(counts)/o_total,"per_image":stats(counts)},
  "test_over_val_candidates_per_image":ratio,"engineering_anomaly":False,"gt_accessed":False,"runtime_seconds":time.time()-start}
 result["test_over_val_f_over_o"]=result["test"]["f_over_o"]/result["val"]["f_over_o"]
 T.write_json(ROOT/"audit/cache_audit.json",result)
 with (ROOT/"f_cache/provenance_by_image.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.writer(f);w.writerow(["image","candidate_count"]);w.writerows(zip([x["image_name"] for x in items],counts))
 print(json.dumps(result,indent=2))
if __name__=="__main__":
 p=argparse.ArgumentParser();p.add_argument("mode",choices=["geometry","audit"]);a=p.parse_args();geometry() if a.mode=="geometry" else audit()
