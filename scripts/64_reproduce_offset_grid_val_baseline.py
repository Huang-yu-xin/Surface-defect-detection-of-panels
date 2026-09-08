"""Reproduce the frozen 831/14 Val BASE using the existing formal implementation."""
import hashlib, importlib.util, json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path('results/offset_grid_halfstride')
BASE=Path('results/baseline_complementarity/global_exact_final_combo/u1e3_g060')
O=Path('results/fn_analysis/cache'); H=Path('results/fn_analysis/cache_hflip')

def module(path,name):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def main():
    started=time.time();out=ROOT/'exact_combo/baseline';out.mkdir(parents=True,exist_ok=True);(out/'final_cache').mkdir(exist_ok=True);(ROOT/'audit').mkdir(exist_ok=True)
    manifests=[json.loads((p/'manifest.json').read_text()) for p in [O,H,BASE]]
    names=[[i['image_name'] for i in m['items']] for m in manifests]
    assert len(names[0])==474 and names[0]==names[1]==names[2]
    bm=manifests[2];assert (bm['min_score'],bm['upper_score'],bm['overlap_gate'])==(1e-5,1e-3,.60);assert set(bm['hflip_classes'])=={0,2,3,4,7}
    for m in manifests[:2]:
        for k,v in {'tile_size':1280,'stride':768,'conf':1e-5,'tile_iou':.60,'max_det':1000}.items():assert m[k]==v,(k,m[k],v)
    weight=Path(manifests[0]['model'])
    import torch,torchvision,ultralytics
    context={'created_at':datetime.now(timezone.utc).isoformat(),'repo_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
      'workspace_dirty':subprocess.check_output(['git','status','--short'],text=True),'python':sys.version,'torch':torch.__version__,'torchvision':torchvision.__version__,'ultralytics':ultralytics.__version__,
      'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip(),'weights':str(weight),'weights_sha256':hashlib.sha256(weight.read_bytes()).hexdigest(),
      'task_sha256':hashlib.sha256(Path('codex_offset_grid_halfstride_val_task.md').read_bytes()).hexdigest(),'source_manifest_sha256':{str(p):hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest() for p in [O,H,BASE]},
      'test_hidden_run':'NO','test_hidden_gt_accessed':'NO','competition_submission':'NO','nms_backend':'unchanged Script42/Script14','stitch_backend':'unchanged Script42'}
    (ROOT/'audit/execution_context.json').write_text(json.dumps(context,indent=2))
    official=module('scripts/42_eval_baseline_global_exact_final_combo.py','official42_offset_base');diag=official.load_module(Path('scripts/14_fn_diagnostic.py'),'diag14_offset_base')
    real=official.final_combo;idx=0
    def save(active,hflip,diagnostic):
        nonlocal idx
        final,count=real(active,hflip,diagnostic);np.savez_compressed(out/'final_cache'/f'{Path(names[0][idx]).stem}.npz',final=final,post_count=len(final)-count);idx+=1;return final,count
    official.final_combo=save
    print('START baseline_reproduction expected=831/14',flush=True)
    metrics,statuses,changes=official.evaluate(BASE,H,Path('datasets/yolo_split/labels/val'),diag,'baseline',ROOT/'exact_combo')
    assert idx==474 and len(statuses)==845 and (metrics['float_tp'],metrics['float_fn'])==(831,14) and (metrics['roundtrip_tp'],metrics['roundtrip_fn'])==(831,14) and not changes
    (out/'statuses.json').write_text(json.dumps([{'image':k[0],'gt_index':k[1],**v} for k,v in statuses.items()],indent=2));(out/'roundtrip_changes.json').write_text(json.dumps(changes,indent=2))
    checks={'status':'PASS','float_831_14':'PASS','roundtrip_831_14':'PASS','same_gt_status':'PASS','complete_474_images':'PASS','official_implementation':'PASS','runtime_seconds':time.time()-started}
    (out/'PASS.json').write_text(json.dumps(checks,indent=2));print('END baseline_reproduction '+json.dumps(metrics),flush=True)
if __name__=='__main__':main()
