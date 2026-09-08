"""Exact frozen BASE+F Test combo: active O+Baseline, selected H, then F."""
from __future__ import annotations
import csv,gc,hashlib,importlib.util,json,math,resource,subprocess,sys,time
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import torch

ROOT=Path("results/vflip_factorial/hidden_probe_f")
ACTIVE=Path("results/baseline_complementarity/global_exact_hidden_probe/u1e3_g060/cache")
HFLIP=Path("results/test_final_cache/hflip")
FCACHE=ROOT/"f_cache"
OUT=ROOT/"submission";SUB=OUT/"submission_vflip_f.json"
BASE_SHA="dfaef6806e43773dbad00ac7c4fb9bc4d7d880716a0570f113906f6e89daeca6"
HCLASSES={0,2,3,4,7}

def module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
M=module("scripts/18_final_combo_from_cache.py","formal18_vflip_f")
def sha(path):
 h=hashlib.sha256()
 with Path(path).open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def load(root,item):
 with np.load(root/item["cache_file"]) as z:return z["candidates"].astype(np.float32,copy=True),tuple(map(int,z["image_shape"]))
def nms_indices(arr,iou,device):
 from torchvision.ops import nms
 kept=[];classes=arr[:,0].astype(np.int32)
 for cid in np.unique(classes):
  idx=np.flatnonzero(classes==cid);boxes=torch.as_tensor(arr[idx,2:6],dtype=torch.float32,device=device);scores=torch.as_tensor(arr[idx,1],dtype=torch.float32,device=device)
  kept.append(idx[nms(boxes,scores,iou).detach().cpu().numpy()])
 inds=np.concatenate(kept) if kept else np.empty(0,np.int64);return inds[np.argsort(arr[inds,1])[::-1]]
def stitched_with_members(post,args,device):
 zinds=np.flatnonzero(post[:,0].astype(np.int32)==1);z=post[zinds]
 formal=M.stitch_zonglie(z,args,device)
 if len(z)<2:return formal,[]
 widths=np.maximum(1e-6,z[:,4]-z[:,2]);heights=np.maximum(1e-6,z[:,5]-z[:,3]);mask=(heights>=args.min_height)&(heights/widths>=args.min_aspect)
 cand=z[mask];candinds=zinds[mask]
 if len(cand)<2:return formal,[]
 pairs=M.build_pairs_gpu(cand,args,device)
 if pairs is None:return formal,[]
 dsu=M.DSU(len(cand));ei,ej=pairs
 for a,b in zip(ei.tolist(),ej.tolist()):dsu.union(a,b)
 groups=defaultdict(list)
 for i in np.unique(np.concatenate([ei,ej])).tolist():groups[dsu.find(i)].append(i)
 members=[];rebuilt=[]
 for ids in groups.values():
  boxes=cand[ids]
  if len(ids)<2 or len(np.unique(boxes[:,7]))<args.min_rows:continue
  y1=float(np.min(boxes[:,3]));y2=float(np.max(boxes[:,5]))
  if y2-y1<args.min_merged_height:continue
  x1=float(np.median(boxes[:,2]));x2=float(np.median(boxes[:,4]))
  if x2<=x1:continue
  row=np.zeros(z.shape[1],np.float32);row[:6]=[1,float(np.max(boxes[:,1])),x1,y1,x2,y2];rebuilt.append(row);members.append(candinds[np.asarray(ids,int)].tolist())
 check=np.stack(rebuilt).astype(np.float32) if rebuilt else np.empty((0,z.shape[1]),np.float32)
 assert np.array_equal(check,formal)
 return formal,members
def main():
 import argparse
 p=argparse.ArgumentParser();p.add_argument("--device",default="cpu");a=p.parse_args();device=torch.device(a.device)
 base=Path("results/baseline_complementarity/global_exact_hidden_probe/submission/submission_u1e3_g060.json")
 assert sha(base)==BASE_SHA
 audit=json.loads((ROOT/"audit/cache_audit.json").read_text());assert audit["status"]=="PASS" and not audit["engineering_anomaly"]
 om=M.load_manifest(ACTIVE);hm=M.load_manifest(HFLIP);fm=M.load_manifest(FCACHE);hmap=M.manifest_map(hm);fmap=M.manifest_map(fm)
 assert int(om["images_count"])==int(hm["images_count"])==int(fm["images_count"])==669
 OUT.mkdir(parents=True,exist_ok=True);(ROOT/"final_combo/stitch_provenance").mkdir(parents=True,exist_ok=True)
 args=type("Args",(),{"min_aspect":5.0,"x_tol":64.0,"max_y_gap":64.0,"min_height":180.0,"min_x_overlap":.20,"stride":768.0,"min_rows":2,"min_merged_height":1300.0})()
 tmp=Path(str(SUB)+".tmp");summary=[];stitchrows=[];cc=Counter();detections=stitched=fpart=mixed=0;first=True;start=time.time()
 with tmp.open("w",encoding="utf-8",buffering=1<<20) as jf:
  jf.write("[\n")
  for idx,oitem in enumerate(om["items"],1):
   name=oitem["image_name"];assert name in hmap and name in fmap
   active,shape=load(ACTIVE,oitem);hf,_=load(HFLIP,hmap[name]);f,_=load(FCACHE,fmap[name]);hids=np.flatnonzero(np.isin(hf[:,0].astype(int),list(HCLASSES)));hsel=hf[hids]
   union=np.concatenate([active,hsel,f]);oi=oitem["original_count"]
   sources=np.concatenate([np.zeros(oi,np.int8),np.ones(len(active)-oi,np.int8),np.full(len(hsel),2,np.int8),np.full(len(f),3,np.int8)])
   source_rows=np.concatenate([np.arange(len(active)),hids,np.arange(len(f))])
   keep=nms_indices(union,.90,device);post=union[keep];ps=sources[keep];pr=source_rows[keep]
   merged,members=stitched_with_members(post,args,device);final=np.concatenate([post,merged]) if len(merged) else post
   local_f=local_mixed=0
   for mi,ids in enumerate(members):
    labels=sorted(set(int(ps[j]) for j in ids));hasf=3 in labels;ismixed=len(labels)>1;local_f+=hasf;local_mixed+=ismixed
    stitchrows.append({"image":name,"stitched_index":mi,"final_row":len(post)+mi,"box":json.dumps(merged[mi,2:6].tolist(),separators=(",",":")),"member_sources":"+".join(["O","Baseline","H","F"][x] for x in labels),"f_participates":hasf,"mixed_source":ismixed,"segments":json.dumps([{"source":["O","Baseline","H","F"][int(ps[j])],"source_row":int(pr[j]),"post_row":j,"tile_x":int(post[j,6]),"tile_y":int(post[j,7]),"bbox":post[j,2:6].tolist()} for j in ids],separators=(",",":"))})
   h,w=shape;assert (h,w)==(oitem["height"],oitem["width"])
   for det in final:
    row=M.submission_row(name,w,h,det)
    if not first:jf.write(",\n")
    jf.write(json.dumps(row,ensure_ascii=False,separators=(",",":"),allow_nan=False));first=False;detections+=1;cc[row["category_name"]]+=1
   stitched+=len(merged);fpart+=local_f;mixed+=local_mixed
   summary.append({"image_id":name,"width":w,"height":h,"active_candidates":len(active),"selected_hflip_candidates":len(hsel),"f_candidates":len(f),"post_nms":len(post),"stitched_zonglie":len(merged),"f_participating_stitched":local_f,"mixed_source_stitched":local_mixed,"final_detection_count":len(final)})
   if idx%25==0 or idx==669:jf.flush();print(f"{idx}/669 final={detections:,} stitched={stitched:,}",flush=True)
   del active,hf,f,hsel,union,post,merged,final
   if idx%25==0:gc.collect()
  jf.write("\n]\n")
 tmp.replace(SUB)
 with (OUT/"per_image.csv").open("w",newline="",encoding="utf-8-sig") as f:
  w=csv.DictWriter(f,fieldnames=list(summary[0]));w.writeheader();w.writerows(summary)
 with (ROOT/"final_combo/stitch_provenance/stitches.csv").open("w",newline="",encoding="utf-8") as f:
  w=csv.DictWriter(f,fieldnames=list(stitchrows[0]) if stitchrows else ["image"]);w.writeheader();w.writerows(stitchrows)
 scripts=[Path("scripts")/x for x in ["18_final_combo_from_cache.py","40_final_combo_test_stream.py","46_final_combo_test_global_complement_stream.py","58_cache_vflip_f_test.py","59_audit_vflip_f_test.py","60_final_combo_test_vflip_f_stream.py"]]
 metrics={"images":669,"detections":detections,"stitched":stitched,"f_participating_stitched":fpart,"mixed_source_stitched":mixed,"class_counts":{c:cc[c] for c in M.CLASS_NAMES},"file_size_bytes":SUB.stat().st_size,"sha256":sha(SUB),"runtime_seconds":time.time()-start,"peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,"device":str(device),"streaming":True,"exit_code":0,"oom_killed":False,"source_order":["active_O_plus_Baseline","selected_HFlip","F_all_classes_all_candidates"],"nms_iou":.90,"script_sha256":{str(x):sha(x) for x in scripts},"repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"created_at":datetime.now(timezone.utc).isoformat(),"ground_truth_accessed":False,"competition_submission_performed":False}
 (OUT/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n");print(json.dumps(metrics,indent=2))
if __name__=="__main__":main()
