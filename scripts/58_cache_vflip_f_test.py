"""Frozen F-only Test inference; no labels or GT are read."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import cv2
import numpy as np

ROOT = Path("results/vflip_factorial/hidden_probe_f")
ORIGINAL = Path("results/test_final_cache/original")
IMAGES = Path("raw/data/test")
WEIGHTS = Path("runs/rareos/yolo26m_tiles1280_rareos_v1_e80_b6_seed2026/weights/best.pt")
PARAMS = {"tile_size":1280,"stride":768,"conf":1e-5,"tile_iou":.60,"max_det":1000,"batch":6,"half":True}
EXPECTED_BASE_SHA = "dfaef6806e43773dbad00ac7c4fb9bc4d7d880716a0570f113906f6e89daeca6"

def module(path, name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

V = module("scripts/52_cache_vflip_factorial_val.py", "vflip_f_test_geometry")

def read_json(path): return json.loads(Path(path).read_text())
def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+".tmp");tmp.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n");tmp.replace(path)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def test_items():
    m=read_json(ORIGINAL/"manifest.json");items=m["items"]
    for k in ["tile_size","stride","conf","tile_iou","max_det","batch"]: assert m[k]==PARAMS[k],(k,m[k])
    assert Path(m["model"])==WEIGHTS
    names=[x["image_name"] for x in items]
    actual=sorted(p.name for p in IMAGES.iterdir() if p.suffix.lower() in V.DIAG.IMAGE_EXTENSIONS)
    assert len(items)==669 and len(set(names))==669 and sorted(names)==actual
    return m,items

def validate_file(path,item):
    with np.load(path) as z:a=z["candidates"];shape=z["image_shape"]
    assert a.ndim==2 and a.shape[1]==14 and a.dtype==np.float32 and np.isfinite(a).all()
    assert np.array_equal(shape,[item["height"],item["width"]])
    if len(a):
        assert ((a[:,0]>=0)&(a[:,0]<9)&(a[:,0]==np.floor(a[:,0]))).all()
        assert ((a[:,1]>=PARAMS["conf"])&(a[:,1]<=1)).all()
        assert (a[:,4]>a[:,2]).all() and (a[:,5]>a[:,3]).all()
        r=float(np.max(np.abs(a[:,2:6].astype(np.float64)-a[:,10:14].astype(np.float64)-a[:,[6,7,6,7]].astype(np.float64))))
        assert r<=1e-4
    return len(a)

def main():
    p=argparse.ArgumentParser();p.add_argument("--batch",type=int,default=6);a=p.parse_args()
    assert 1<=a.batch<=6
    bm=read_json("results/baseline_complementarity/global_exact_hidden_probe/submission/metrics.json")
    assert bm["detections"]==3392951 and bm["sha256"]==EXPECTED_BASE_SHA
    assert sha("results/baseline_complementarity/global_exact_hidden_probe/submission/submission_u1e3_g060.json")==EXPECTED_BASE_SHA
    geo=read_json(ROOT/"audit/geometry_tests.json");assert geo["status"]=="PASS"
    import torch
    from ultralytics import YOLO
    assert torch.cuda.is_available()
    om,items=test_items();target=ROOT/"f_cache";target.mkdir(parents=True,exist_ok=True)
    start=time.time();torch.cuda.reset_peak_memory_stats();batch=a.batch;oom=[];model=YOLO(str(WEIGHTS))
    by_name={x["image_name"]:x for x in items};counts={};reused=0
    for index,item in enumerate(items,1):
        out=target/item["cache_file"]
        if out.exists():
            try: counts[item["image_name"]]=validate_file(out,item);reused+=1;continue
            except Exception: out.unlink()
        image=cv2.imread(str(IMAGES/item["image_name"]),cv2.IMREAD_COLOR);assert image is not None
        h,w=image.shape[:2];assert (h,w)==(item["height"],item["width"])
        metas=V.tile_origins(h,w,"mirrored_y");tiles=[V.patch(image,t,1) for t in metas];rows=[];begin=0
        while begin<len(tiles):
            count=min(batch,len(tiles)-begin)
            try:
                predictions=model.predict(tiles[begin:begin+count],imgsz=1280,conf=1e-5,iou=.60,max_det=1000,device="0",half=True,verbose=False)
            except torch.cuda.OutOfMemoryError:
                if batch==1: raise
                old=batch;batch=max(1,batch//2);oom.append({"image_index":index,"from_batch":old,"to_batch":batch})
                torch.cuda.empty_cache();print("OOM lowering batch "+json.dumps(oom[-1]),flush=True);continue
            for pred,meta in zip(predictions,metas[begin:begin+count]):
                if pred.boxes is not None and len(pred.boxes):
                    rows.append(V.pack_rows(pred.boxes.xyxy.detach().cpu().numpy(),pred.boxes.cls.detach().cpu().numpy(),pred.boxes.conf.detach().cpu().numpy(),meta,1))
            begin+=count
        arr=np.concatenate(rows) if rows else np.empty((0,14),np.float32)
        np.savez_compressed(out,candidates=arr,image_shape=np.asarray([h,w],np.int32));counts[item["image_name"]]=len(arr)
        if index%10==0 or index==669: print(f"PROGRESS F {index}/669 rows={sum(counts.values())} runtime={time.time()-start:.2f}s",flush=True)
    torch.cuda.synchronize()
    manifest={"view_id":"F","definition":"mirrored-Y grid + tile-local vertical flip","factor_orientation":1,"factor_grid":1,"grid_mode":"mirrored_y",
      "weights":str(WEIGHTS),"weights_sha256":sha(WEIGHTS),"repo_head":subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip(),**PARAMS,
      "batch_final":batch,"half_actual":bool(model.predictor.model.fp16),"image_count":669,"images_count":669,"candidate_rows":sum(counts.values()),
      "schema":V.DIAG.COLS,"coordinate_space":"original_image","source_order_position":"after active O+Baseline and selected HFlip",
      "items":[{"image_name":i["image_name"],"cache_file":i["cache_file"],"height":i["height"],"width":i["width"],"candidate_count":counts[i["image_name"]],
                "tile_origins":[list(t) for t in V.tile_origins(i["height"],i["width"],"mirrored_y")]} for i in items],
      "runtime_seconds":time.time()-start,"peak_gpu_allocated_mb":torch.cuda.max_memory_allocated()/2**20,"peak_gpu_reserved_mb":torch.cuda.max_memory_reserved()/2**20,
      "resume_reused_images":reused,"oom_events":oom,"ground_truth_accessed":False,"competition_submission_performed":False,"created_at":datetime.now(timezone.utc).isoformat()}
    assert manifest["half_actual"]
    write_json(target/"manifest.json",manifest);write_json(ROOT/"manifest.json",manifest)
    print(json.dumps({k:manifest[k] for k in ["image_count","candidate_rows","runtime_seconds","peak_gpu_allocated_mb","peak_gpu_reserved_mb","resume_reused_images","oom_events"]},indent=2),flush=True)

if __name__=="__main__":main()
