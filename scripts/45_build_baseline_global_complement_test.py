from __future__ import annotations
import argparse,csv,hashlib,importlib.util,json,platform,resource,subprocess,sys,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
ROOT=Path("results/baseline_complementarity/global_exact_hidden_probe")
SOURCES={"original":Path("results/test_final_cache/original"),"hflip":Path("results/test_final_cache/hflip"),"baseline":Path("results/baseline_complementarity/cache_test_original")}
MIN_SCORE=1e-5;UPPER_SCORE=1e-3;OVERLAP_GATE=.60;HFLIP_CLASSES={0,2,3,4,7};CLASS_NAMES=["jieba","zonglie","qilie","jiaza","yiwuyaru","huashang","mamianmakeng","yanghuatiepi","gunyin"]
def parse_args():
 p=argparse.ArgumentParser();p.add_argument("--mode",choices=["audit","build"],required=True);p.add_argument("--output-root",type=Path,default=ROOT);return p.parse_args()
def load_manifest(p):return json.loads((p/"manifest.json").read_text())
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""):h.update(b)
 return h.hexdigest()
def load_val():
 p=Path("scripts/41_build_baseline_global_complement_val.py");s=importlib.util.spec_from_file_location("val41",p);m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;s.loader.exec_module(m);return m
def audit(out):
 started=time.time()
 out.mkdir(parents=True,exist_ok=True)
 report={"global_box_slice":[2,6],"local_box_slice":[10,14],"caches":{},"image_sets_identical":False,"image_shapes_identical":False,"status":"FAIL"}
 manifests={k:load_manifest(v) for k,v in SOURCES.items()}
 sets={k:{x["image_name"] for x in m["items"]} for k,m in manifests.items()}
 report["image_sets_identical"]=sets["original"]==sets["hflip"]==sets["baseline"]
 shape_maps={}
 for label,root in SOURCES.items():
  m=manifests[label];items=m["items"];seen=set();tot=0;maxres=0.
  bad={k:0 for k in ["nonfinite","class","score","bbox","local_bbox","shape","image_shape","item_count","tile_metadata","duplicate_image","duplicate_file","manifest_schema"]}
  names=[x["image_name"] for x in items];files=[x["cache_file"] for x in items]
  bad["duplicate_image"]=len(names)-len(set(names));bad["duplicate_file"]=len(files)-len(set(files))
  expected_columns=["class_id","score","xmin","ymin","xmax","ymax","tile_x","tile_y","valid_w","valid_h","local_xmin","local_ymin","local_xmax","local_ymax"]
  bad["manifest_schema"]=int(m.get("columns")!=expected_columns)
  shape_maps[label]={}
  for item in items:
   p=root/item["cache_file"];seen.add(p.name)
   with np.load(p,allow_pickle=False) as z:
    a=z["candidates"].astype(np.float32,copy=False);shape=z["image_shape"]
   tot+=len(a)
   bad["item_count"]+=int(len(a)!=item.get("candidate_count"))
   expected_shape=(item["height"],item["width"])
   if shape.shape!=(2,) or not np.array_equal(shape,np.asarray(expected_shape)) or min(expected_shape)<=0:
    bad["image_shape"]+=1;continue
   shape_maps[label][item["image_name"]]=tuple(map(int,shape))
   if a.ndim!=2 or a.shape[1]!=14:bad["shape"]+=1;continue
   bad["nonfinite"]+=int(np.count_nonzero(~np.isfinite(a)))
   cls=a[:,0];bad["class"]+=int(np.count_nonzero((cls<0)|(cls>8)|(cls!=np.floor(cls))))
   bad["score"]+=int(np.count_nonzero((a[:,1]<0)|(a[:,1]>1)))
   bad["bbox"]+=int(np.count_nonzero((a[:,4]<=a[:,2])|(a[:,5]<=a[:,3])|(a[:,2]<0)|(a[:,3]<0)|(a[:,4]>shape[1]+1e-4)|(a[:,5]>shape[0]+1e-4)))
   bad["local_bbox"]+=int(np.count_nonzero((a[:,12]<=a[:,10])|(a[:,13]<=a[:,11])|(a[:,10]<-1e-4)|(a[:,11]<-1e-4)|(a[:,12]>a[:,8]+1e-4)|(a[:,13]>a[:,9]+1e-4)))
   bad["tile_metadata"]+=int(np.count_nonzero((a[:,6]<0)|(a[:,7]<0)|(a[:,8]<=0)|(a[:,9]<=0)|(a[:,6]+a[:,8]>shape[1]+1e-4)|(a[:,7]+a[:,9]>shape[0]+1e-4)))
   expected=a[:,10:14]+a[:,[6,7,6,7]]
   if len(a):maxres=max(maxres,float(np.max(np.abs(a[:,2:6]-expected))))
  disk={p.name for p in root.glob("*.npz")};manifest_rows=m.get("total_candidates")
  ok=(len(items)==m.get("images_count")==669 and len(disk)==669 and disk==seen and tot==manifest_rows and not any(bad.values()) and maxres<=1e-4)
  report["caches"][label]={"path":str(root),"manifest_exists":True,"manifest_sha256":sha(root/"manifest.json"),"manifest_images":m.get("images_count"),"npz_images":len(disk),"rows_manifest":manifest_rows,"rows_actual":tot,"schema_columns":m.get("columns"),"bad":bad,"max_coordinate_residual":maxres,"status":"PASS" if ok else "FAIL"}
 report["image_shapes_identical"]=shape_maps["original"]==shape_maps["hflip"]==shape_maps["baseline"]
 report["status"]="PASS" if report["image_sets_identical"] and report["image_shapes_identical"] and all(x["status"]=="PASS" for x in report["caches"].values()) else "FAIL"
 report["runtime_seconds"]=time.time()-started
 report["peak_rss_mb"]=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
 (out/"test_cache_audit.json").write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2),flush=True)
 if report["status"]!="PASS":raise RuntimeError("Test cache audit failed")

def build(root):
 audit_path=root/"audit"/"test_cache_audit.json"
 if not audit_path.exists() or json.loads(audit_path.read_text()).get("status")!="PASS":raise RuntimeError("Passing cache audit required")
 audited=json.loads(audit_path.read_text())
 for label,source in SOURCES.items():
  if audited["caches"][label]["manifest_sha256"]!=sha(source/"manifest.json"):raise RuntimeError("Source manifest changed since cache audit")
 baseline_metrics=json.loads((root/"baseline_reproduction"/"metrics.json").read_text())
 if baseline_metrics["detections"]!=2403809:raise RuntimeError("Formal baseline reproduction required before build")
 val=load_val();started=time.time();manifests={k:load_manifest(v) for k,v in SOURCES.items()};maps={k:{x["image_name"]:x for x in m["items"]} for k,m in manifests.items()};names=[x["image_name"] for x in manifests["original"]["items"]];out=root/"u1e3_g060"/"cache";out.mkdir(parents=True,exist_ok=True);items=[];prov=[];tot=scoreq=addedn=0;byclass=Counter()
 for idx,name in enumerate(names,1):
  arr={}
  for k,p in SOURCES.items():
   with np.load(p/maps[k][name]["cache_file"]) as z:
    arr[k]=z["candidates"].astype(np.float32,copy=True)
    if k=="original":shape=z["image_shape"].astype(np.int32,copy=True)
  b=arr["baseline"];cls=b[:,0].astype(np.int32);score=(b[:,1]>=MIN_SCORE)&(b[:,1]<=UPPER_SCORE);maxiou=np.zeros(len(b),np.float32)
  for cid in range(9):
   ids=np.flatnonzero(score&(cls==cid))
   if len(ids):maxiou[ids]=val.max_iou_chunked(b[ids,2:6],val.active_reference(arr["original"],arr["hflip"],cid),256,1024)
  keep=score&(maxiou<OVERLAP_GATE);added=b[keep];combo=np.concatenate([arr["original"],added]) if len(added) else arr["original"];fn=maps["original"][name]["cache_file"];np.savez_compressed(out/fn,candidates=combo,image_shape=shape);cnt=Counter(cls[keep].tolist());tot+=len(b);scoreq+=int(np.count_nonzero(score));addedn+=len(added);byclass.update(cnt)
  items.append({"image_name":name,"cache_file":fn,"height":int(shape[0]),"width":int(shape[1]),"candidate_count":len(combo),"original_count":len(arr["original"]),"baseline_added":len(added)})
  row={"image":name,"baseline_total":len(b),"score_qualified":int(np.count_nonzero(score)),"overlap_qualified":len(added),"added_total":len(added)};row.update({f"added_class_{i}":cnt[i] for i in range(9)});prov.append(row)
  if idx%25==0 or idx==len(names):print(f"BUILD {idx}/{len(names)}",flush=True)
 runtime=time.time()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;manifest={"config":"u1e3_g060","min_score":MIN_SCORE,"upper_score":UPPER_SCORE,"overlap_gate":OVERLAP_GATE,"hflip_classes":sorted(HFLIP_CLASSES),"global_box_slice":[2,6],"local_box_slice":[10,14],"source_original_cache":str(SOURCES["original"]),"source_hflip_cache":str(SOURCES["hflip"]),"source_baseline_cache":str(SOURCES["baseline"]),"source_manifest_sha256":{k:sha(v/"manifest.json") for k,v in SOURCES.items()},"image_count":len(names),"images_count":len(names),"baseline_candidates_before":tot,"baseline_candidates_score_qualified":scoreq,"baseline_candidates_added":addedn,"added_by_class":{CLASS_NAMES[i]:byclass[i] for i in range(9)},"rows_total_after_merge":sum(x["candidate_count"] for x in items),"repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),"created_at":datetime.now(timezone.utc).isoformat(),"runtime_seconds":runtime,"peak_rss_mb":peak,"python_version":platform.python_version(),"numpy_version":np.__version__,"items":items}
 manifest["total_candidates"]=manifest["rows_total_after_merge"]
 manifest["columns"]=manifests["original"]["columns"]
 manifest["selector_script_sha256"]={str(p):sha(p) for p in [Path("scripts/41_build_baseline_global_complement_val.py"),Path(__file__)]}
 (root/"u1e3_g060"/"manifest.json").write_text(json.dumps(manifest,indent=2))
 (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
 with (root/"u1e3_g060"/"provenance_by_image.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(prov[0]),lineterminator="\n");w.writeheader();w.writerows(prov)
 print(f"END build added={addedn} runtime_seconds={runtime:.3f} peak_rss_mb={peak:.3f}",flush=True)
def main():
 a=parse_args();a.output_root.mkdir(parents=True,exist_ok=True)
 if a.mode=="audit":audit(a.output_root/"audit")
 else:build(a.output_root)
if __name__=="__main__":main()
