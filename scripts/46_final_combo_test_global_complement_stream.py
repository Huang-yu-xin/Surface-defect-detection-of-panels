from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import resource
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("results/baseline_complementarity/global_exact_hidden_probe")
ORIGINAL = Path("results/test_final_cache/original")
HFLIP = Path("results/test_final_cache/hflip")
FROZEN = {"min_score": 1e-5, "upper_score": 1e-3, "overlap_gate": 0.60,
          "hflip_classes": [0, 2, 3, 4, 7], "global_box_slice": [2, 6],
          "local_box_slice": [10, 14]}
CLASS_NAMES = ["jieba", "zonglie", "qilie", "jiaza", "yiwuyaru",
               "huashang", "mamianmakeng", "yanghuatiepi", "gunyin"]

def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()

def main():
    parser = argparse.ArgumentParser(description="Run the unchanged official Script 40 on CPU.")
    parser.add_argument("--variant", choices=["baseline", "u1e3_g060"], required=True)
    args = parser.parse_args()
    audit = json.loads((ROOT / "audit/test_cache_audit.json").read_text())
    if audit["status"] != "PASS" or not audit["image_sets_identical"]:
        raise RuntimeError("Test cache audit prerequisite failed")
    if args.variant == "baseline":
        active = ORIGINAL
        out = ROOT / "baseline_reproduction"
        submission = out / "submission.json"
    else:
        baseline = json.loads((ROOT / "baseline_reproduction/metrics.json").read_text())
        baseline_file = ROOT / "baseline_reproduction/submission.json"
        if baseline["detections"] != 2403809 or baseline["sha256"] != sha(baseline_file):
            raise RuntimeError("Formal baseline checksum prerequisite failed")
        cm = json.loads((ROOT / "u1e3_g060/manifest.json").read_text())
        if cm["config"] != "u1e3_g060" or any(cm[k] != v for k, v in FROZEN.items()):
            raise RuntimeError("Frozen selector prerequisite failed")
        active = ROOT / "u1e3_g060/cache"
        out = ROOT / "submission"
        submission = out / "submission_u1e3_g060.json"
    out.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "scripts/40_final_combo_test_stream.py",
               "--mode", "test", "--device", "cpu",
               "--original-cache", str(active), "--hflip-cache", str(HFLIP),
               "--output-dir", str(out), "--submission", str(submission),
               "--summary-csv", str(out / "per_image.csv"),
               "--global-iou", "0.90", "--hflip-classes", "0", "2", "3", "4", "7",
               "--min-aspect", "5.0", "--x-tol", "64.0", "--max-y-gap", "64.0",
               "--min-height", "180.0", "--min-x-overlap", "0.20",
               "--stride", "768.0", "--min-rows", "2", "--min-merged-height", "1300.0"]
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
               MKL_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1", MALLOC_ARENA_MAX="2",
               PYTHONUNBUFFERED="1")
    started = time.time()
    print("Official command:", json.dumps(command), flush=True)
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               text=True, env=env)
    log_lines = []
    with (out / "official_stream.log").open("w") as log:
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
            log_lines.append(line)
    status = process.wait()
    if status:
        raise RuntimeError(f"Official streaming process failed: returncode={status}")
    with (out / "per_image.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    detections = sum(int(row["final_detection_count"]) for row in rows)
    stitched = sum(int(row["stitched_zonglie"]) for row in rows)
    counts = {}
    for line in log_lines:
        fields = line.split()
        if len(fields) == 2 and fields[0] in CLASS_NAMES:
            counts[fields[0]] = int(fields[1].replace(",", ""))
    if sum(counts.values()) != detections or len(counts) != 9 or len(rows) != 669:
        raise RuntimeError("Official output count/image/class reconciliation failed")
    if args.variant == "baseline" and detections != 2403809:
        raise RuntimeError(f"Formal baseline checksum failed: {detections}")
    if not any("NMS          : torchvision-cuda" in line for line in log_lines):
        raise RuntimeError("Official NMS did not use torchvision; investigate fallback")
    script_paths = [Path("scripts") / name for name in [
        "18_final_combo_from_cache.py", "40_final_combo_test_stream.py",
        "41_build_baseline_global_complement_val.py", "45_build_baseline_global_complement_test.py",
        "46_final_combo_test_global_complement_stream.py"]]
    runtime = time.time() - started
    peak = max(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
               resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss) / 1024
    metrics = {"variant": args.variant, "active_cache": str(active),
               "hflip_cache": str(HFLIP), "device": "cpu", "gpu_used": False,
               "nms_backend": "torchvision-cpu",
               "official_backend_label": "torchvision-cuda",
               "backend_note": "Script18 legacy label says cuda; command and tensors use CPU.",
               "images": len(rows), "detections": detections, "stitched": stitched,
               "class_counts": counts, "file_size_bytes": submission.stat().st_size,
               "sha256": sha(submission), "runtime_seconds": runtime, "peak_rss_mb": peak,
               "streaming": True, "exit_code": status, "oom_killed": False,
               "command": command, "script_sha256": {str(p): sha(p) for p in script_paths},
               "repo_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
               "created_at": datetime.now(timezone.utc).isoformat(),
               "competition_submission_performed": False}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    if args.variant == "baseline":
        (out / "manifest.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)

if __name__ == "__main__":
    main()
