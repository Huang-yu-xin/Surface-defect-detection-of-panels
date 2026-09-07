from __future__ import annotations
import argparse, csv, importlib.util, json, math, resource, sys, time
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

CLASS_NAMES=["jieba","zonglie","qilie","jiaza","yiwuyaru","huashang","mamianmakeng","yanghuatiepi","gunyin"]
HFLIP_CLASSES={0,2,3,4,7}; ZONGLIE_ID=1
CONFIGS=["u3e4_g060","u3e4_g080","u1e3_g060","u1e3_g080","uinf_g060","uinf_g080"]

def parse_args():
 p=argparse.ArgumentParser()
 p.add_argument("--root",type=Path,default=Path("results/baseline_complementarity/global_exact_final_combo"))
 p.add_argument("--original-cache",type=Path,default=Path("results/fn_analysis/cache"))
 p.add_argument("--hflip-cache",type=Path,default=Path("results/fn_analysis/cache_hflip"))
 p.add_argument("--labels",type=Path,default=Path("datasets/yolo_split/labels/val"))
 p.add_argument("--assignments",type=Path,default=Path("splits/sample_assignments.csv"))
 p.add_argument("--failure-csv",type=Path,default=Path("results/final_combo_fn21/remaining_fn_21.csv"))
 p.add_argument("--diag-script",type=Path,default=Path("scripts/14_fn_diagnostic.py"))
 return p.parse_args()

def load_module(path,name):
 spec=importlib.util.spec_from_file_location(name,path);mod=importlib.util.module_from_spec(spec);sys.modules[name]=mod;spec.loader.exec_module(mod);return mod
def load_manifest(root):return json.loads((root/"manifest.json").read_text())
def nms(arr,diag):
 keep=diag.class_aware_nms_indices(arr,.90)
 return arr[keep] if len(keep) else np.empty((0,arr.shape[1]),np.float32)
class DSU:
 def __init__(self,n):self.p=list(range(n));self.sz=[1]*n
 def find(self,x):
  while self.p[x]!=x:self.p[x]=self.p[self.p[x]];x=self.p[x]
  return x
 def union(self,a,b):
  a=self.find(a);b=self.find(b)
  if a==b:return
  if self.sz[a]<self.sz[b]:a,b=b,a
  self.p[b]=a;self.sz[a]+=self.sz[b]
def stitch_zonglie(zpost):
 if len(zpost)<2:return np.empty((0,zpost.shape[1]),np.float32)
 widths=np.maximum(1e-6,zpost[:,4]-zpost[:,2]);heights=np.maximum(1e-6,zpost[:,5]-zpost[:,3]);cand=zpost[(heights>=180)&(heights/widths>=5)]
 if len(cand)<2:return np.empty((0,zpost.shape[1]),np.float32)
 tile_y=cand[:,7];rows=np.unique(tile_y);dsu=DSU(len(cand));touched=set()
 for ai in range(len(rows)):
  for bi in range(ai+1,len(rows)):
   ra,rb=float(rows[ai]),float(rows[bi])
   if abs(rb-ra)<=1e-6 or abs(rb-ra)>768.001:continue
   ia=np.flatnonzero(np.isclose(tile_y,ra));ib=np.flatnonzero(np.isclose(tile_y,rb))
   for i in ia:
    a=cand[i];acx=(a[2]+a[4])*.5;aw=max(1e-6,a[4]-a[2])
    for j in ib:
     b=cand[j];bcx=(b[2]+b[4])*.5
     if abs(acx-bcx)>64:continue
     gap=max(b[3]-a[5],a[3]-b[5],0)
     inter=max(0,min(a[4],b[4])-max(a[2],b[2]));overlap=inter/min(aw,max(1e-6,b[4]-b[2]))
     if gap<=64 and overlap>=.20:dsu.union(int(i),int(j));touched.add(int(i));touched.add(int(j))
 groups=defaultdict(list)
 for i in touched:groups[dsu.find(i)].append(i)
 merged=[]
 for ids in groups.values():
  boxes=cand[ids]
  if len(ids)<2 or len(np.unique(boxes[:,7]))<2:continue
  y1=float(np.min(boxes[:,3]));y2=float(np.max(boxes[:,5]))
  if y2-y1<1300:continue
  x1=float(np.median(boxes[:,2]));x2=float(np.median(boxes[:,4]))
  if x2<=x1:continue
  row=np.zeros(zpost.shape[1],np.float32);row[0]=1;row[1]=float(np.max(boxes[:,1]));row[2:6]=[x1,y1,x2,y2];merged.append(row)
 return np.stack(merged) if merged else np.empty((0,zpost.shape[1]),np.float32)
def final_combo(active,hflip,diag):
 hsel=hflip[np.isin(hflip[:,0].astype(np.int32),list(HFLIP_CLASSES))];union=np.concatenate([active,hsel]) if len(hsel) else active;post=nms(union,diag);merged=stitch_zonglie(post[post[:,0].astype(np.int32)==1]);return (np.concatenate([post,merged]) if len(merged) else post),len(merged)
def submission_row(name,w,h,d):
 x1=max(0,min(int(math.floor(float(d[2]))),w-1));y1=max(0,min(int(math.floor(float(d[3]))),h-1));x2=max(x1+1,min(int(math.ceil(float(d[4]))),w));y2=max(y1+1,min(int(math.ceil(float(d[5]))),h))
 return {"image_id":name,"category_name":CLASS_NAMES[int(d[0])],"bbox":[x1,y1,x2,y2],"score":round(float(d[1]),6)}
def roundtrip(final,name,w,h,path):
 rows=[submission_row(name,w,h,d) for d in final];path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(rows,separators=(",",":")));loaded=json.loads(path.read_text())
 out=np.zeros((len(loaded),6),np.float32);cmap={n:i for i,n in enumerate(CLASS_NAMES)}
 for i,r in enumerate(loaded):out[i]=[cmap[r["category_name"]],r["score"],*r["bbox"]]
 return out
def best_pred(final,g,diag):
 ids=np.flatnonzero(final[:,0].astype(np.int32)==g.class_id)
 if not len(ids):return {"iou":0.0,"score":"","box":""}
 box=np.array([g.xmin,g.ymin,g.xmax,g.ymax],np.float32);ious=diag.box_iou_one_to_many(box,final[ids,2:6]);j=ids[int(np.argmax(ious))];return {"iou":float(np.max(ious)),"score":float(final[j,1]),"box":",".join(f"{x:.3f}" for x in final[j,2:6])}
def read_maps(a):
 with a.assignments.open(encoding="utf-8-sig",newline="") as f:
  rows=list(csv.DictReader(f));cols=list(rows[0]);image_col="image" if "image" in cols else next(c for c in cols if "image" in c.lower());group_col="group_id" if "group_id" in cols else next(c for c in cols if "group" in c.lower());groups={r[image_col]:r[group_col] for r in rows}
 failures={}
 with a.failure_csv.open(encoding="utf-8-sig",newline="") as f:
  for r in csv.DictReader(f):failures[(r["image_name"],int(r["gt_index"]))]=r["failure_type"]
 return groups,failures,{"columns":cols,"image_column":image_col,"group_column":group_col}
def evaluate(cache_root,hflip_root,labels,diag,config,out_root):
 started=time.time();m=load_manifest(cache_root);hm=load_manifest(hflip_root);hmap={x["image_name"]:x for x in hm["items"]};tot=Counter();class_stats={n:Counter() for n in CLASS_NAMES};statuses={};det_count=0;stitched=0;roundtrip_changes=[]
 rtdir=out_root/config/"roundtrip_json"
 for idx,item in enumerate(m["items"],1):
  name=item["image_name"]
  with np.load(cache_root/item["cache_file"]) as z:active=z["candidates"].astype(np.float32,copy=True);shape=z["image_shape"];h,w=map(int,shape)
  with np.load(hflip_root/hmap[name]["cache_file"]) as z:hflip=z["candidates"].astype(np.float32,copy=True)
  final,mc=final_combo(active,hflip,diag);rt=roundtrip(final,name,w,h,rtdir/f"{Path(name).stem}.json");det_count+=len(final);stitched+=mc;gt=diag.read_yolo_gt(labels/f"{Path(name).stem}.txt",w,h)
  ftp,ffp,ffn,funmatched,fmatched,_=diag.match_predictions(final,gt,.5);rtp,rfp,rfn,runmatched,rmatched,_=diag.match_predictions(rt,gt,.5);tot.update(float_tp=ftp,float_fp=ffp,float_fn=ffn,roundtrip_tp=rtp,roundtrip_fp=rfp,roundtrip_fn=rfn)
  for cid,cname in enumerate(CLASS_NAMES):
   gtc=[g for g in gt if g.class_id==cid];pred=final[final[:,0].astype(np.int32)==cid];ctp,cfp,cfn,*_=diag.match_predictions(pred,gtc,.5);class_stats[cname].update(tp=ctp,fp=cfp,fn=cfn)
  for gi,g in enumerate(gt):
   bp=best_pred(final,g,diag);rp=best_pred(rt,g,diag);key=(name,gi);statuses[key]={"float":gi in fmatched,"roundtrip":gi in rmatched,"class_id":g.class_id,"bbox":[g.xmin,g.ymin,g.xmax,g.ymax],"float_best":bp,"roundtrip_best":rp}
   if (gi in fmatched)!=(gi in rmatched):roundtrip_changes.append({"config":config,"image":name,"gt_index":gi,"class":CLASS_NAMES[g.class_id],"float_status":"TP" if gi in fmatched else "FN","roundtrip_status":"TP" if gi in rmatched else "FN","float_iou":bp["iou"],"roundtrip_iou":rp["iou"]})
  if idx%25==0 or idx==len(m["items"]):print(f"EVAL {config} {idx}/{len(m['items'])} float={tot['float_tp']}/{tot['float_fn']} rt={tot['roundtrip_tp']}/{tot['roundtrip_fn']}",flush=True)
 runtime=time.time()-started;metrics=dict(tot);metrics.update(config=config,final_detection_count=det_count,stitched_count=stitched,runtime_seconds=runtime,peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,per_class={k:dict(v) for k,v in class_stats.items()},roundtrip_changes=len(roundtrip_changes))
 (out_root/config).mkdir(parents=True,exist_ok=True);(out_root/config/"metrics.json").write_text(json.dumps(metrics,indent=2))
 return metrics,statuses,roundtrip_changes
def write_csv(path,rows,fields):
 with path.open("w",encoding="utf-8-sig",newline="") as f:w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)
def main():
 a=parse_args();a.root.mkdir(parents=True,exist_ok=True);diag=load_module(a.diag_script,"diag42");groups,failures,assignment_schema=read_maps(a)
 print("START baseline reproduction",flush=True);bm,bs,changes=evaluate(a.original_cache,a.hflip_cache,a.labels,diag,"baseline_reproduction",a.root)
 if (bm["float_tp"],bm["float_fn"])!=(824,21):raise RuntimeError(f"Baseline reproduction failed: {bm['float_tp']}/{bm['float_fn']}")
 all_metrics=[bm];all_changes=changes;attr=[];audits=[]
 for cfg in CONFIGS:
  print(f"START {cfg}",flush=True);m,s,c=evaluate(a.root/cfg,a.hflip_cache,a.labels,diag,cfg,a.root);all_metrics.append(m);all_changes+=c
  for representation in ["float","roundtrip"]:
   rescued=[];regressed=[]
   for key,b in bs.items():
    n=s[key]
    if not b[representation] and n[representation]:kind="rescued";rescued.append(key)
    elif b[representation] and not n[representation]:kind="regressed";regressed.append(key)
    else:continue
    name,gi=key;cid=b["class_id"];row={"config":cfg,"representation":representation,"change":kind,"image":name,"group":groups.get(name,""),"class":CLASS_NAMES[cid],"gt_index":gi,"gt_bbox":",".join(f"{x:.3f}" for x in b["bbox"]),"failure_type":failures.get(key,""),"baseline_status":"FN" if kind=="rescued" else "TP","new_status":"TP" if kind=="rescued" else "FN","baseline_best_box":b[representation+"_best"]["box"],"baseline_best_iou":b[representation+"_best"]["iou"],"new_best_box":n[representation+"_best"]["box"],"new_best_iou":n[representation+"_best"]["iou"]};attr.append(row)
   rc=Counter(groups.get(x[0],"") for x in rescued);gc=Counter(CLASS_NAMES[bs[x]["class_id"]] for x in rescued);fc=Counter(failures.get(x,"unclassified") for x in rescued);regc=Counter(groups.get(x[0],"") for x in regressed);largest=max(rc.values(),default=0);largest_groups={g for g,v in rc.items() if v==largest};best_without=None
   for lg in largest_groups or {""}:best_without=max(best_without if best_without is not None else -10**9,sum(v for g,v in rc.items() if g!=lg)-sum(v for g,v in regc.items() if g!=lg))
   audit={"config":cfg,"representation":representation,"rescued":len(rescued),"regressed":len(regressed),"net_gain":len(rescued)-len(regressed),"rescue_images":len({x[0] for x in rescued}),"rescue_groups":len(rc),"rescue_classes":len(gc),"rescue_failure_types":len(fc),"rescues_by_group":json.dumps(rc,ensure_ascii=False,sort_keys=True),"rescues_by_class":json.dumps(gc,ensure_ascii=False,sort_keys=True),"rescues_by_failure":json.dumps(fc,ensure_ascii=False,sort_keys=True),"regressions_by_group":json.dumps(regc,ensure_ascii=False,sort_keys=True),"largest_group_rescues":largest,"net_gain_without_largest_rescue_group":best_without};audits.append(audit)
 write_csv(a.root/"rescue_regression.csv",attr,list(attr[0]) if attr else ["config","representation","change"]);write_csv(a.root/"group_audit.csv",audits,list(audits[0]));write_csv(a.root/"roundtrip_changes.csv",all_changes,list(all_changes[0]) if all_changes else ["config","image"])
 summary=[]
 amap={(x["config"],x["representation"]):x for x in audits}
 for m in all_metrics:
  cfg=m["config"]
  if cfg=="baseline_reproduction":manifest={"min_score":"","upper_score":"","overlap_gate":"","rows_baseline_considered":0,"rows_baseline_added":0,"added_by_class":{}}
  else:manifest=load_manifest(a.root/cfg)
  af=amap.get((cfg,"float"),{});ar=amap.get((cfg,"roundtrip"),{})
  row={"config":cfg,"min_score":manifest.get("min_score",""),"upper_score":manifest.get("upper_score",""),"overlap_gate":manifest.get("overlap_gate",""),"baseline_rows_considered":manifest.get("rows_baseline_considered",0),"baseline_rows_added":manifest.get("rows_baseline_added",0),"added_by_class":json.dumps(manifest.get("added_by_class",{}),ensure_ascii=False,sort_keys=True),"final_detection_count":m["final_detection_count"],"stitched_count":m["stitched_count"],"float_tp":m["float_tp"],"float_fn":m["float_fn"],"float_recall":m["float_tp"]/(m["float_tp"]+m["float_fn"]),"roundtrip_tp":m["roundtrip_tp"],"roundtrip_fn":m["roundtrip_fn"],"roundtrip_recall":m["roundtrip_tp"]/(m["roundtrip_tp"]+m["roundtrip_fn"]),"rescued":af.get("rescued",0),"regressed":af.get("regressed",0),"net_gain":af.get("net_gain",0),"roundtrip_rescued":ar.get("rescued",0),"roundtrip_regressed":ar.get("regressed",0),"roundtrip_net_gain":ar.get("net_gain",0),"rescue_images":af.get("rescue_images",0),"rescue_groups":af.get("rescue_groups",0),"rescue_classes":af.get("rescue_classes",0),"rescue_failure_types":af.get("rescue_failure_types",0),"largest_group_rescues":af.get("largest_group_rescues",0),"net_gain_without_largest_rescue_group":af.get("net_gain_without_largest_rescue_group",0),"peak_rss_mb":m["peak_rss_mb"],"runtime_seconds":m["runtime_seconds"]};summary.append(row)
 write_csv(a.root/"summary.csv",summary,list(summary[0]));(a.root/"evaluation_metadata.json").write_text(json.dumps({"assignment_schema":assignment_schema,"match_iou":.5,"global_nms_iou":.9,"hflip_classes":sorted(HFLIP_CLASSES),"submission_rounding":{"bbox":"floor x1/y1, ceil x2/y2, clipped","score_decimals":6}},indent=2))
 print("END exact evaluation",flush=True)
if __name__=="__main__":main()
