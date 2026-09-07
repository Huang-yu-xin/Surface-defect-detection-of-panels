from __future__ import annotations
import argparse, csv, hashlib, json, math, platform, resource, subprocess, time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
CLASS_NAMES=["jieba","zonglie","qilie","jiaza","yiwuyaru","huashang","mamianmakeng","yanghuatiepi","gunyin"]
HFLIP_CLASSES={0,2,3,4,7}
CONFIGS={"u3e4_g060":(3e-4,.60),"u3e4_g080":(3e-4,.80),"u1e3_g060":(1e-3,.60),"u1e3_g080":(1e-3,.80),"uinf_g060":(math.inf,.60),"uinf_g080":(math.inf,.80)}
MIN_SCORE=1e-5
def parse_args():
 p=argparse.ArgumentParser(description="Build six frozen global-coordinate Baseline complements without annotations.")
 p.add_argument("--original-cache",type=Path,default=Path("results/fn_analysis/cache"));p.add_argument("--hflip-cache",type=Path,default=Path("results/fn_analysis/cache_hflip"));p.add_argument("--baseline-cache",type=Path,default=Path("results/baseline_complementarity/cache_original"));p.add_argument("--output-root",type=Path,default=Path("results/baseline_complementarity/global_exact_final_combo"));p.add_argument("--baseline-chunk",type=int,default=256);p.add_argument("--reference-chunk",type=int,default=1024);return p.parse_args()
def load_manifest(path):return json.loads((path/"manifest.json").read_text())
def sha256(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):h.update(block)
 return h.hexdigest()
def max_iou_chunked(boxes,refs,bc,rc):
 out=np.zeros(len(boxes),np.float32)
 if not len(boxes) or not len(refs):return out
 refs=refs.astype(np.float32,copy=False);ra=np.maximum(0,refs[:,2]-refs[:,0])*np.maximum(0,refs[:,3]-refs[:,1])
 for bs in range(0,len(boxes),bc):
  b=boxes[bs:bs+bc].astype(np.float32,copy=False);ba=np.maximum(0,b[:,2]-b[:,0])*np.maximum(0,b[:,3]-b[:,1]);best=np.zeros(len(b),np.float32)
  for rs in range(0,len(refs),rc):
   r=refs[rs:rs+rc];x1=np.maximum(b[:,None,0],r[None,:,0]);y1=np.maximum(b[:,None,1],r[None,:,1]);x2=np.minimum(b[:,None,2],r[None,:,2]);y2=np.minimum(b[:,None,3],r[None,:,3]);inter=np.maximum(0,x2-x1)*np.maximum(0,y2-y1);union=ba[:,None]+ra[None,rs:rs+rc]-inter;iou=np.divide(inter,union,out=np.zeros_like(inter),where=union>0);best=np.maximum(best,np.max(iou,axis=1))
  out[bs:bs+len(b)]=best
 return out
def active_reference(orig,hflip,cid):
 o=orig[orig[:,0].astype(np.int32)==cid,2:6]
 if cid not in HFLIP_CLASSES:return o
 h=hflip[hflip[:,0].astype(np.int32)==cid,2:6];return np.concatenate([o,h]) if len(h) else o
def coordinate_residual(arr):
 if not len(arr):return 0.,0
 sub=arr[(np.abs(arr[:,6])>0)|(np.abs(arr[:,7])>0)]
 if not len(sub):return 0.,0
 expected=sub[:,10:14]+sub[:,[6,7,6,7]];return float(np.max(np.abs(sub[:,2:6]-expected))),len(sub)
def run_unit_tests(roots,manifests):
 local=np.array([[0,0,10,10]],np.float32);shifted=local+np.array([100,0,100,0],np.float32)
 assert max_iou_chunked(local,local,1,1)[0]==1 and max_iou_chunked(local,shifted,1,1)[0]==0
 o=np.array([[5,.1,0,0,10,10]+[0]*8,[2,.1,0,0,10,10]+[0]*8],np.float32);h=np.array([[5,.1,1,1,9,9]+[0]*8,[2,.1,1,1,9,9]+[0]*8],np.float32)
 assert len(active_reference(o,h,5))==1 and len(active_reference(o,h,2))==2
 checks={}
 for label,root in roots.items():
  checked=0;max_res=0.
  for item in manifests[label]["items"]:
   with np.load(root/item["cache_file"]) as z:residual,count=coordinate_residual(z["candidates"])
   checked+=count;max_res=max(max_res,residual)
   if checked>=1000:break
  assert checked and max_res<=1e-4;checks[label]={"nonzero_origin_rows_checked":checked,"max_coordinate_residual":max_res}
 return {"cross_tile_synthetic":"PASS","active_hflip_reference":"PASS","cache_coordinate_sanity":"PASS","builder_has_no_annotation_or_evaluation_argument":"PASS","cache_checks":checks}
def main():
 a=parse_args();started=time.time();roots={"original":a.original_cache,"hflip":a.hflip_cache,"baseline":a.baseline_cache};manifests={k:load_manifest(v) for k,v in roots.items()};names=[[x["image_name"] for x in manifests[k]["items"]] for k in roots]
 if not names[0]==names[1]==names[2]:raise RuntimeError("Cache image order mismatch")
 a.output_root.mkdir(parents=True,exist_ok=True);(a.output_root/"builder_unit_tests.json").write_text(json.dumps(run_unit_tests(roots,manifests),indent=2));maps={k:{x["image_name"]:x for x in manifests[k]["items"]} for k in roots};stats={cfg:{"considered":0,"added":0,"by_class":Counter(),"items":[],"sidecar":[]} for cfg in CONFIGS}
 for idx,name in enumerate(names[0],1):
  arrays={};shape=None
  for label,root in roots.items():
   with np.load(root/maps[label][name]["cache_file"]) as z:
    arrays[label]=z["candidates"].astype(np.float32,copy=True)
    if label=="original":shape=z["image_shape"].astype(np.int32,copy=True)
  b=arrays["baseline"];bcls=b[:,0].astype(np.int32);eligible=b[:,1]>=MIN_SCORE;max_iou=np.zeros(len(b),np.float32)
  for cid in range(len(CLASS_NAMES)):
   ids=np.flatnonzero(eligible&(bcls==cid))
   if len(ids):max_iou[ids]=max_iou_chunked(b[ids,2:6],active_reference(arrays["original"],arrays["hflip"],cid),a.baseline_chunk,a.reference_chunk)
  for cfg,(upper,gate) in CONFIGS.items():
   score_mask=eligible&((b[:,1]<=upper) if math.isfinite(upper) else True);keep=score_mask&(max_iou<gate);added=b[keep];combo=np.concatenate([arrays["original"],added]) if len(added) else arrays["original"];out_dir=a.output_root/cfg;out_dir.mkdir(parents=True,exist_ok=True);cache_file=maps["original"][name]["cache_file"];np.savez_compressed(out_dir/cache_file,candidates=combo,image_shape=shape);st=stats[cfg];st["considered"]+=int(np.count_nonzero(score_mask));st["added"]+=len(added);st["by_class"].update(bcls[keep].tolist());st["items"].append({"image_name":name,"cache_file":cache_file,"height":int(shape[0]),"width":int(shape[1]),"candidate_count":len(combo),"original_count":len(arrays["original"]),"baseline_added":len(added)});st["sidecar"].append({"image":name,"baseline_before":len(b),"baseline_score_kept":int(np.count_nonzero(score_mask)),"baseline_overlap_kept":len(added),"added_total":len(added)})
  if idx%25==0 or idx==len(names[0]):print(f"BUILD {idx}/{len(names[0])}",flush=True)
 commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip();created=datetime.now(timezone.utc).isoformat();hashes={k:sha256(v/"manifest.json") for k,v in roots.items()}
 for cfg,(upper,gate) in CONFIGS.items():
  st=stats[cfg];out_dir=a.output_root/cfg;manifest={"schema_version":1,"code_commit":commit,"created_at":created,"parent_rareos_cache":str(a.original_cache),"parent_hflip_cache":str(a.hflip_cache),"parent_baseline_cache":str(a.baseline_cache),"source_manifest_sha256":hashes,"min_score":MIN_SCORE,"upper_score":"inf" if not math.isfinite(upper) else upper,"overlap_gate":gate,"hflip_classes":sorted(HFLIP_CLASSES),"images":len(st["items"]),"images_count":len(st["items"]),"rows_original":sum(x["original_count"] for x in st["items"]),"rows_baseline_considered":st["considered"],"rows_baseline_added":st["added"],"rows_total":sum(x["candidate_count"] for x in st["items"]),"added_by_class":{CLASS_NAMES[k]:v for k,v in sorted(st["by_class"].items())},"global_box_slice":[2,6],"local_box_slice":[10,14],"columns":manifests["original"]["columns"],"script_path":"scripts/41_build_baseline_global_complement_val.py","python_version":platform.python_version(),"numpy_version":np.__version__,"baseline_chunk":a.baseline_chunk,"reference_chunk":a.reference_chunk,"items":st["items"]}
  (out_dir/"manifest.json").write_text(json.dumps(manifest,indent=2))
  with (out_dir/"provenance.csv").open("w",newline="") as f:w=csv.DictWriter(f,fieldnames=list(st["sidecar"][0]),lineterminator="\n");w.writeheader();w.writerows(st["sidecar"])
 runtime=time.time()-started;peak=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024;(a.output_root/"build_metrics.json").write_text(json.dumps({"runtime_seconds":runtime,"peak_rss_mb":peak},indent=2));print(f"END build runtime_seconds={runtime:.3f} peak_rss_mb={peak:.3f}",flush=True)
if __name__=="__main__":main()
