"""Reproduce only frozen u1e3_g060 Val through the existing Script42 API."""
import hashlib, importlib.util, json, subprocess, sys, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path('results/vflip_factorial')
BASE = Path('results/baseline_complementarity/global_exact_final_combo/u1e3_g060')
O = Path('results/fn_analysis/cache')
H = Path('results/fn_analysis/cache_hflip')

def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m

def main():
    started = time.time()
    out = ROOT / 'exact_combo' / 'baseline'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'final_cache').mkdir(exist_ok=True)
    (ROOT / 'audit').mkdir(exist_ok=True)
    assert not (out / 'PASS.json').exists(), 'Already reproduced; reuse verified baseline'
    manifests = [json.loads((p / 'manifest.json').read_text()) for p in [O, H, BASE]]
    names = [[i['image_name'] for i in m['items']] for m in manifests]
    assert len(names[0]) == 474 and names[0] == names[1] == names[2]
    bm = manifests[2]
    assert bm['min_score'] == 1e-5 and bm['upper_score'] == 1e-3 and bm['overlap_gate'] == .60
    assert set(bm['hflip_classes']) == {0,2,3,4,7}
    for m in manifests[:2]:
        for k,v in {'tile_size':1280, 'stride':768, 'conf':1e-5, 'tile_iou':.60, 'max_det':1000}.items():
            assert m[k] == v, (k,m[k],v)
    context = {'created_at':datetime.now(timezone.utc).isoformat(),
        'repo_head':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
        'workspace_dirty':subprocess.check_output(['git','status','--short'],text=True),
        'python':sys.version, 'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True).strip(),
        'weights':manifests[0]['model'], 'test_hidden_run':'NO', 'test_hidden_gt_accessed':'NO', 'competition_submission':'NO',
        'task_sha256':hashlib.sha256(Path('codex_rareos_vflip_factorial_val_task.md').read_bytes()).hexdigest(),
        'source_manifest_sha256':{str(p):hashlib.sha256((p/'manifest.json').read_bytes()).hexdigest() for p in [O,H,BASE]},
        'half_note':'O/H manifests omit half; use frozen task half=True; inspect original logs separately.',
        'nms_backend':'Script42 reference NumPy, Script14 class_aware_nms_indices',
        'stitch_backend':'Script42 stitch_zonglie; unchanged existing corrected Val implementation'}
    (ROOT/'audit'/'execution_context.json').write_text(json.dumps(context,indent=2))
    official = module('scripts/42_eval_baseline_global_exact_final_combo.py','official42_vflip_base')
    diag = official.load_module(Path('scripts/14_fn_diagnostic.py'),'diag14_vflip_base')
    real_final = official.final_combo
    image_index = 0
    def save_final(active,hflip,diagnostic):
        nonlocal image_index
        final, count = real_final(active,hflip,diagnostic)
        np.savez_compressed(out/'final_cache'/f'{Path(names[0][image_index]).stem}.npz', final=final, post_count=len(final)-count)
        image_index += 1
        return final,count
    official.final_combo = save_final
    print('START baseline_reproduction images=474 expected_float=831/14 expected_roundtrip=831/14',flush=True)
    metrics,statuses,changes = official.evaluate(BASE,H,Path('datasets/yolo_split/labels/val'),diag,'baseline',ROOT/'exact_combo')
    assert image_index == 474 and len(statuses) == 845
    (out/'statuses.json').write_text(json.dumps([{'image':key[0],'gt_index':key[1],**value} for key,value in statuses.items()],indent=2))
    (out/'roundtrip_changes.json').write_text(json.dumps(changes,indent=2))
    assert (metrics['float_tp'],metrics['float_fn']) == (831,14), metrics
    assert (metrics['roundtrip_tp'],metrics['roundtrip_fn']) == (831,14), metrics
    assert len(changes) == 0, changes
    checks = {'float_831_14':'PASS','roundtrip_831_14':'PASS','same_gt_status':'PASS','complete_474_images':'PASS','frozen_selector':'PASS','official_implementation':'PASS','status':'PASS','runtime_seconds':time.time()-started}
    (out/'PASS.json').write_text(json.dumps(checks,indent=2))
    print('END baseline_reproduction STATUS=PASS '+json.dumps(metrics),flush=True)

if __name__ == '__main__': main()
