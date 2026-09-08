"""Read prediction artifacts only; validate the frozen Test submission without labels."""
from __future__ import annotations
import csv
import hashlib
import json
import math
import re
import resource
import subprocess
import time
from collections import Counter
from pathlib import Path
import numpy as np

ROOT = Path('results/baseline_complementarity/global_exact_hidden_probe')
VAL = Path('results/baseline_complementarity/global_exact_final_combo')
CLASS_NAMES = ['jieba', 'zonglie', 'qilie', 'jiaza', 'yiwuyaru', 'huashang', 'mamianmakeng', 'yanghuatiepi', 'gunyin']
FROZEN = {'config': 'u1e3_g060', 'min_score': 1e-5, 'upper_score': .001, 'overlap_gate': .60, 'hflip_classes': [0, 2, 3, 4, 7], 'global_box_slice': [2, 6], 'local_box_slice': [10, 14]}


def read_json(path):
    return json.loads(path.read_text())


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)


def unique_map(rows, key):
    result = {r[key]: r for r in rows}
    if len(result) != len(rows):
        raise ValueError(f'Duplicate {key} in tabular/manifest artifact')
    return result


def strict_object(pairs):
    row = {}
    for key, value in pairs:
        if key in row:
            raise ValueError(f'Duplicate JSON object key: {key}')
        row[key] = value
    return row


def reject_constant(value):
    raise ValueError(f'Non-standard JSON constant: {value}')


def stream_records(path):
    """Strict JSON validation for Script40's one-object-per-line array layout.

    Unlike stripping commas independently, this state machine verifies every
    separator, rejects trailing commas/content, and consumes the entire file.
    Memory is bounded by one serialized detection, with a 1 MiB line limit.
    """
    opened = closed = have_row = needs_row = False
    with path.open(encoding='utf-8') as f:
        for line_number, line in enumerate(f, 1):
            if len(line) > 1 << 20:
                raise ValueError(f'Unexpected >1 MiB JSON record on line {line_number}')
            text = line.strip()
            if not text:
                continue
            if closed:
                raise ValueError('Content after JSON closing bracket')
            if not opened:
                if text != '[':
                    raise ValueError('Expected opening JSON array bracket')
                opened = True
                continue
            if text == ']':
                if needs_row:
                    raise ValueError('Trailing comma in JSON array')
                closed = True
                continue
            if have_row and not needs_row:
                raise ValueError('Missing comma between JSON records')
            trailing_comma = text.endswith(',')
            encoded = text[:-1] if trailing_comma else text
            row = json.loads(encoded, object_pairs_hook=strict_object, parse_constant=reject_constant)
            if not isinstance(row, dict):
                raise ValueError('Non-object submission entry')
            have_row = True
            needs_row = trailing_comma
            yield row
    if not opened or not closed:
        raise ValueError('Incomplete JSON array')


def numeric(value):
    return type(value) in (int, float) and math.isfinite(value)


def integrity(path, dimensions):
    started = time.time()
    count = 0
    classes, per_image, bad = Counter(), Counter(), Counter()
    error = None
    try:
        for row in stream_records(path):
            count += 1
            if set(row) != {'image_id', 'category_name', 'bbox', 'score'}:
                bad['schema'] += 1
            name, category = row.get('image_id'), row.get('category_name')
            valid_name = isinstance(name, str) and name in dimensions
            valid_class = isinstance(category, str) and category in CLASS_NAMES
            if not valid_name:
                bad['image'] += 1
            else:
                per_image[name] += 1
            if not valid_class:
                bad['category'] += 1
            else:
                classes[category] += 1
            box, score = row.get('bbox'), row.get('score')
            if not isinstance(box, list) or len(box) != 4 or not all(numeric(x) for x in box):
                bad['bbox_schema'] += 1
            elif valid_name:
                w, h = dimensions[name]
                x1, y1, x2, y2 = box
                if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
                    bad['bbox_valid'] += 1
            if not numeric(score) or not 0 <= score <= 1:
                bad['score'] += 1
    except (ValueError, OSError, UnicodeError) as exc:
        error = str(exc)
    return {'path': str(path.resolve()), 'sha256': sha(path), 'file_size_bytes': path.stat().st_size,
            'detections': count, 'image_count': len(per_image), 'class_counts': {c: classes[c] for c in CLASS_NAMES},
            'per_image_counts': dict(per_image), 'json_valid': error is None, 'json_error': error,
            'schema_valid': bad['schema'] == 0, 'category_valid': bad['category'] == 0,
            'bbox_valid': bad['bbox_schema'] + bad['bbox_valid'] == 0,
            'score_valid': bad['score'] == 0, 'image_valid': bad['image'] == 0,
            'bad_counts': dict(bad), 'audit_runtime_seconds': time.time() - started}


def groups(path):
    current = None
    counts = Counter()
    for row in stream_records(path):
        name = row['image_id']
        key = (row['category_name'], tuple(row['bbox']), row['score'])
        if current is None:
            current = name
        if name != current:
            if name <= current:
                raise ValueError('Submission image blocks must be unique and lexically sorted')
            yield current, counts
            current, counts = name, Counter()
        counts[key] += 1
    if current is not None:
        yield current, counts


def diff_submissions(base, new):
    bg, ng = groups(base), groups(new)
    b, n = next(bg, None), next(ng, None)
    same = bonly = nonly = score_changed = bbox_same = 0
    rows = []
    while b is not None or n is not None:
        if b is None:
            name, nc = n; bc = Counter(); n = next(ng, None)
        elif n is None:
            name, bc = b; nc = Counter(); b = next(bg, None)
        elif b[0] == n[0]:
            name, bc = b; _, nc = n; b = next(bg, None); n = next(ng, None)
        elif b[0] < n[0]:
            name, bc = b; nc = Counter(); b = next(bg, None)
        else:
            name, nc = n; bc = Counter(); n = next(ng, None)
        identical = sum((bc & nc).values())
        bo, no = sum((bc - nc).values()), sum((nc - bc).values())
        bk, nk = Counter(), Counter()
        for (category, box, score), count in bc.items():
            bk[(category, box)] += count
        for (category, box, score), count in nc.items():
            nk[(category, box)] += count
        box_identical = sum((bk & nk).values())
        bbox_same += box_identical
        score_changed += box_identical - identical
        same += identical; bonly += bo; nonly += no
        rows.append({'image': name, 'exact_identical': identical, 'baseline_only': bo, 'new_only': no,
                     'net_delta': sum(nc.values()) - sum(bc.values()), 'identical_bbox': box_identical,
                     'score_changed_same_bbox': box_identical - identical})
    return {'exactly_identical_detections': same, 'baseline_only_detections': bonly, 'new_only_detections': nonly,
            'identical_image_category_bbox': bbox_same, 'same_bbox_score_replacements': score_changed,
            'comparison_key': 'image_id + category_name + bbox + rounded score; multiset multiplicity retained'}, rows


def stats(values):
    a = np.asarray(values, float)
    if not len(a):
        raise ValueError('Empty distribution')
    return {'min': float(a.min()), 'mean': float(a.mean()), 'median': float(np.median(a)),
            'p90': float(np.percentile(a, 90)), 'p95': float(np.percentile(a, 95)),
            'p99': float(np.percentile(a, 99)), 'max': float(a.max())}


def ratio(numerator, denominator):
    return numerator / denominator if denominator else (0.0 if not numerator else None)


def metric_checks(label, metrics, parsed, per_image, dimensions):
    checks = {}
    checks[f'{label}: parsed count matches metrics'] = parsed['detections'] == metrics['detections']
    checks[f'{label}: parsed image counts match metrics and expected cache'] = parsed['image_count'] == metrics['images'] == len(dimensions)
    checks[f'{label}: per-image CSV has exact image set'] = set(per_image) == set(dimensions)
    checks[f'{label}: per-image counts match actual JSON'] = all(int(row['final_detection_count']) == parsed['per_image_counts'].get(name, 0) for name, row in per_image.items())
    checks[f'{label}: stitched CSV sum matches metrics'] = sum(int(row['stitched_zonglie']) for row in per_image.values()) == metrics['stitched']
    checks[f'{label}: classes match metrics'] = parsed['class_counts'] == {c: metrics['class_counts'].get(c, 0) for c in CLASS_NAMES}
    checks[f'{label}: SHA/size match metrics'] = parsed['sha256'] == metrics['sha256'] and parsed['file_size_bytes'] == metrics['file_size_bytes']
    checks[f'{label}: completed official CPU streaming process'] = metrics.get('exit_code') == 0 and metrics.get('streaming') is True and metrics.get('device') == 'cpu' and metrics.get('gpu_used') is False and metrics.get('oom_killed') is False
    checks[f'{label}: recorded implementation hashes match disk'] = bool(metrics.get('script_sha256')) and all(Path(p).is_file() and sha(Path(p)) == expected for p, expected in metrics.get('script_sha256', {}).items())
    for flag in ['json_valid', 'schema_valid', 'category_valid', 'bbox_valid', 'score_valid', 'image_valid']:
        checks[f'{label}: {flag}'] = parsed[flag]
    return checks


def main():
    started = time.time()
    audit = read_json(ROOT / 'audit/test_cache_audit.json')
    context = read_json(ROOT / 'audit/execution_context.json')
    cm = read_json(ROOT / 'u1e3_g060/manifest.json')
    bm = read_json(ROOT / 'baseline_reproduction/metrics.json')
    nm = read_json(ROOT / 'submission/metrics.json')
    orig_path = Path('results/test_final_cache/original/manifest.json')
    orig = read_json(orig_path)
    originals = unique_map(orig['items'], 'image_name')
    dims = {name: (item['width'], item['height']) for name, item in originals.items()}
    bp, npth = ROOT / 'baseline_reproduction/submission.json', ROOT / 'submission/submission_u1e3_g060.json'
    print('START strict streaming integrity baseline', flush=True)
    bi = integrity(bp, dims)
    print('START strict streaming integrity u1e3_g060', flush=True)
    ni = integrity(npth, dims)
    write_json(ROOT / 'audit/baseline_integrity.json', bi)
    write_json(ROOT / 'audit/submission_integrity.json', ni)
    if not bi['json_valid'] or not ni['json_valid']:
        raise RuntimeError(f'JSON syntax failure: baseline={bi["json_error"]}; new={ni["json_error"]}')
    print('START per-image streaming multiset diff', flush=True)
    diff, image_diff = diff_submissions(bp, npth)
    diff.update({'baseline_detection_count': bi['detections'], 'new_detection_count': ni['detections'],
                 'delta': ni['detections'] - bi['detections'], 'baseline_reference': str(bp.resolve()),
                 'baseline_reference_kind': 'fresh formal reproduction, verified against historical 2403809 count',
                 'historical_9809_json': context.get('historical_9809_json'),
                 'historical_reference_note': context.get('historical_reference_note', 'Original 98.09 JSON has not been independently identified; this diff uses the formal count-verified reproduction.')})
    write_json(ROOT / 'submission/submission_diff_summary.json', diff)
    prov = unique_map(read_csv(ROOT / 'u1e3_g060/provenance_by_image.csv'), 'image')
    base_img = unique_map(read_csv(ROOT / 'baseline_reproduction/per_image.csv'), 'image_id')
    new_img = unique_map(read_csv(ROOT / 'submission/per_image.csv'), 'image_id')
    merged_items = unique_map(cm['items'], 'image_name')
    checks = metric_checks('baseline', bm, bi, base_img, dims)
    checks.update(metric_checks('u1e3_g060', nm, ni, new_img, dims))
    checks['provenance has exact image set'] = set(prov) == set(dims) == set(merged_items)
    checks['builder image count is complete'] = cm['image_count'] == len(prov) == len(dims)
    for column, key in [('baseline_total', 'baseline_candidates_before'), ('score_qualified', 'baseline_candidates_score_qualified'), ('added_total', 'baseline_candidates_added'), ('overlap_qualified', 'baseline_candidates_added')]:
        checks[f'provenance {column} matches manifest'] = sum(int(r[column]) for r in prov.values()) == cm[key]
    checks['per-image candidate provenance reconciles'] = all(
        int(r['baseline_total']) >= int(r['score_qualified']) >= int(r['added_total']) == int(r['overlap_qualified']) >= 0
        and int(r['added_total']) == sum(int(r[f'added_class_{c}']) for c in range(9))
        and int(r['added_total']) == merged_items[name]['baseline_added']
        and merged_items[name]['original_count'] == originals[name]['candidate_count']
        and merged_items[name]['candidate_count'] == merged_items[name]['original_count'] + int(r['added_total'])
        for name, r in prov.items())
    checks['merged candidate total reconciles'] = sum(r['candidate_count'] for r in merged_items.values()) == cm['rows_total_after_merge'] == audit['caches']['original']['rows_actual'] + cm['baseline_candidates_added']
    checks['per-class proposal counts reconcile'] = all(sum(int(r[f'added_class_{i}']) for r in prov.values()) == cm['added_by_class'].get(c, 0) for i, c in enumerate(CLASS_NAMES))
    source_paths = {'original': Path(cm['source_original_cache']) / 'manifest.json', 'hflip': Path(cm['source_hflip_cache']) / 'manifest.json', 'baseline': Path(cm['source_baseline_cache']) / 'manifest.json'}
    checks['source cache manifests unchanged since build'] = all(sha(path) == cm['source_manifest_sha256'][name] for name, path in source_paths.items())
    checks['source Baseline total matches cache audit'] = cm['baseline_candidates_before'] == audit['caches']['baseline']['rows_actual']
    checks['diff count identities reconcile'] = diff['exactly_identical_detections'] + diff['baseline_only_detections'] == bi['detections'] and diff['exactly_identical_detections'] + diff['new_only_detections'] == ni['detections'] and sum(r['net_delta'] for r in image_diff) == diff['delta']
    checks['baseline/new same formal implementation'] = all(bm['script_sha256'][p] == nm['script_sha256'][p] for p in ['scripts/18_final_combo_from_cache.py', 'scripts/40_final_combo_test_stream.py', 'scripts/46_final_combo_test_global_complement_stream.py'])

    formal = read_json(ROOT / 'audit/formal_code_integrity.json')
    checks['formal18/40 and frozen Val41-44 unchanged from HEAD'] = bool(formal) and all(item['unchanged_from_HEAD'] and sha(Path(path)) == item['sha256'] for path, item in formal.items())
    checks['builder reused Val selector hashes match disk'] = bool(cm.get('selector_script_sha256')) and all(sha(Path(path)) == value for path, value in cm.get('selector_script_sha256', {}).items())
    previous = context.get('previous_verified_baseline')
    if previous:
        checks['baseline repeat is byte-identical to previous verified reproduction'] = all(bi[key] == previous[key] for key in ['detections', 'sha256', 'file_size_bytes', 'class_counts']) and bm['stitched'] == previous['stitched']

    valm = read_json(VAL / 'u1e3_g060/manifest.json')
    valbase = read_json(VAL / 'baseline_reproduction/metrics.json')
    valnew = read_json(VAL / 'u1e3_g060/metrics.json')
    val_source_path = Path(valm['parent_baseline_cache']) / 'manifest.json'
    val_source = read_json(val_source_path)
    checks['Val source manifest matches frozen provenance'] = sha(val_source_path) == valm['source_manifest_sha256']['baseline']
    checks['Val/Test selector parameters identical'] = all(cm[key] == valm[key] for key in FROZEN if key != 'config')
    val_added, val_images, val_total = valm['rows_baseline_added'], valm['images_count'], val_source['total_candidates']
    test_added, test_images, test_total = cm['baseline_candidates_added'], cm['image_count'], cm['baseline_candidates_before']
    added = [int(r['added_total']) for r in prov.values()]
    baseline_counts = [bi['per_image_counts'].get(k, 0) for k in dims]
    new_counts = [ni['per_image_counts'].get(k, 0) for k in dims]
    final_deltas = {k: ni['per_image_counts'].get(k, 0) - bi['per_image_counts'].get(k, 0) for k in dims}
    distributions = {'added_proposals': stats(added), 'baseline_final_detections': stats(baseline_counts),
                     'u1e3_final_detections': stats(new_counts), 'final_detection_delta': stats(list(final_deltas.values()))}
    dist = distributions['added_proposals']
    val_delta = valnew['final_detection_count'] - valbase['final_detection_count']
    comparison = {'val': {'images': val_images, 'source_baseline_candidates': val_total, 'added': val_added,
                          'added_per_image': val_added / val_images, 'added_per_source_candidate': val_added / val_total,
                          'final_detection_delta': val_delta, 'final_delta_per_image': val_delta / val_images},
                  'test': {'images': test_images, 'source_baseline_candidates': test_total, 'added': test_added,
                           'added_per_image': test_added / test_images, 'added_per_source_candidate': test_added / test_total,
                           'final_detection_delta': diff['delta'], 'final_delta_per_image': diff['delta'] / test_images}}
    comparison['test_over_val'] = {k: ratio(comparison['test'][k], comparison['val'][k]) for k in ['added_per_image', 'added_per_source_candidate', 'final_delta_per_image']}
    perclass = []
    for c in CLASS_NAMES:
        va, ta = valm['added_by_class'].get(c, 0), cm['added_by_class'].get(c, 0)
        bd, nd = bi['class_counts'][c], ni['class_counts'][c]
        perclass.append({'class': c, 'val_added': va, 'test_added': ta, 'val_added_share': ratio(va, val_added),
                         'test_added_share': ratio(ta, test_added), 'test_val_added_share_ratio': ratio(ta / test_added, va / val_added),
                         'test_val_added_per_image_ratio': ratio(ta / test_images, va / val_images),
                         'baseline_final': bd, 'new_final': nd, 'final_delta': nd - bd})
    class_ratios = [x['test_val_added_per_image_ratio'] for x in perclass]
    top2 = sum(sorted(added, reverse=True)[:2]) / sum(added)
    baseline_only_ratio = diff['baseline_only_detections'] / bi['detections']
    anomaly_reasons = []
    # These are engineering investigation triggers, never prediction-selection rules.
    if comparison['test_over_val']['added_per_image'] is None or comparison['test_over_val']['added_per_image'] >= 10:
        anomaly_reasons.append('Test added/image reaches 10x Val')
    if any(value is None or value >= 10 for value in class_ratios):
        anomaly_reasons.append('At least one class added/image reaches 10x Val or has new nonzero mass from a zero Val class')
    if top2 >= .5:
        anomaly_reasons.append('At least half of all proposals are concentrated in two images')
    if baseline_only_ratio >= .05:
        anomaly_reasons.append('At least 5% of baseline detections are replaced; investigate formal pipeline consistency')
    distribution = {'distributions': distributions, 'val_test_comparison': comparison,
                    'top2_added_concentration': top2, 'max_class_test_val_added_per_image_ratio': max(x for x in class_ratios if x is not None),
                    'baseline_only_fraction': baseline_only_ratio, 'engineering_anomaly': bool(anomaly_reasons),
                    'anomaly_reasons': anomaly_reasons,
                    'engineering_trigger_note': 'Descriptive safeguards only: 10x added/image or per-class, >=50% top2 concentration, >=5% baseline-only. None alters a prediction.'}
    write_csv(ROOT / 'audit/distribution_audit.csv', [{'metric': key, **value} for key, value in distributions.items()], ['metric', 'min', 'mean', 'median', 'p90', 'p95', 'p99', 'max'])
    write_csv(ROOT / 'audit/per_class_audit.csv', perclass, list(perclass[0]))
    outliers = []
    for rank, row in enumerate(sorted(prov.values(), key=lambda r: int(r['added_total']), reverse=True)[:20], 1):
        outliers.append({'kind': 'added_proposals', 'rank': rank, 'image': row['image'], 'value': int(row['added_total'])})
    for rank, (name, value) in enumerate(sorted(final_deltas.items(), key=lambda x: x[1], reverse=True)[:20], 1):
        outliers.append({'kind': 'final_detection_delta', 'rank': rank, 'image': name, 'value': value})
    write_csv(ROOT / 'audit/outlier_images.csv', outliers, list(outliers[0]))
    write_csv(ROOT / 'audit/submission_per_image_diff.csv', image_diff, list(image_diff[0]))
    write_json(ROOT / 'audit/distribution_summary.json', distribution)

    # Scope declarations are execution attestations, not claims that JSON can prove
    # the absence of unrelated access. Code/input provenance is retained separately.
    scope_ok = all(context.get(key) is False for key in ['gpu_used', 'test_inference_rerun', 'test_hidden_gt_accessed', 'competition_submission_performed'])
    required = ['u1e3_g060/manifest.json', 'u1e3_g060/provenance_by_image.csv', 'baseline_reproduction/manifest.json',
                'baseline_reproduction/metrics.json', 'submission/metrics.json', 'audit/execution_context.json',
                'logs/cache_audit.log', 'logs/build_u1e3_g060.log', 'logs/baseline_reproduction.log', 'logs/final_combo_u1e3_g060.log']
    conditions = [
        ('Formal baseline Test reproduction = 2,403,809', bi['detections'] == 2403809),
        ('Val/Test selector parameters and reused selector provenance agree', checks['Val/Test selector parameters identical']),
        ('Only frozen u1e3_g060 parameters', all(cm.get(k) == v for k, v in FROZEN.items())),
        ('Global2:6 / local10:14 coordinates', cm['global_box_slice'] == [2, 6] and cm['local_box_slice'] == [10, 14]),
        ('Active HFlip reference 0,2,3,4,7', cm['hflip_classes'] == valm['hflip_classes'] == [0, 2, 3, 4, 7]),
        ('All Test caches and image sets PASS audit', audit['status'] == 'PASS' and audit['image_sets_identical']),
        ('Execution attestation: no Test/Hidden GT, inference rerun, or upload', scope_ok),
        ('Formal implementation/provenance/count reconciliation PASS', all(checks.values())),
        ('Distribution has no engineering anomaly', not anomaly_reasons),
        ('Baseline and new JSON integrity PASS', all(parsed[k] for parsed in [bi, ni] for k in ['json_valid', 'schema_valid', 'category_valid', 'bbox_valid', 'score_valid', 'image_valid'])),
        ('Completed official streaming processes without OOM/Killed', all(m.get('streaming') and m.get('exit_code') == 0 and m.get('oom_killed') is False for m in [bm, nm])),
        ('Provenance/manifests/logs saved', all((ROOT / p).is_file() and (ROOT / p).stat().st_size > 0 for p in required))]
    ready = all(value for _, value in conditions)
    decision = 'READY_FOR_MANUAL_HIDDEN_SUBMISSION' if ready else 'STOP_AND_INVESTIGATE'
    write_json(ROOT / 'audit/reconciliation_checks.json', [{'criterion': k, 'pass': bool(v)} for k, v in checks.items()])
    manifest = {k: v for k, v in ni.items() if k != 'per_image_counts'}
    manifest.update({'config': 'u1e3_g060', 'streaming': True, 'stitched': nm['stitched'], 'frozen_parameters': FROZEN,
                     'script_sha256': nm['script_sha256'], 'competition_submission_performed': False, 'audit_decision': decision})
    write_json(ROOT / 'submission/submission_manifest.json', manifest)
    write_json(ROOT / 'decision.json', {'decision': decision, 'conditions': [{'criterion': label, 'pass': bool(passed)} for label, passed in conditions],
                                      'recommended_submission': ni['path'] if ready else None})
    audit_runtime = time.time() - started
    audit_peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    stages_path = ROOT / 'logs/stage_metrics.json'
    stages = read_json(stages_path) if stages_path.exists() else {}
    cache_runtime = audit.get('runtime_seconds', stages.get('cache_audit', {}).get('runtime_seconds', 0))
    runtime = cache_runtime + bm['runtime_seconds'] + cm['runtime_seconds'] + nm['runtime_seconds'] + audit_runtime
    peak = max(audit_peak, audit.get('peak_rss_mb', 0), cm['peak_rss_mb'], bm['peak_rss_mb'], nm['peak_rss_mb'])
    engineering = {'cache_audit_runtime_seconds': cache_runtime, 'baseline_runtime_seconds': bm['runtime_seconds'],
                   'builder_runtime_seconds': cm['runtime_seconds'], 'final_combo_runtime_seconds': nm['runtime_seconds'],
                   'submission_audit_runtime_seconds': audit_runtime, 'total_stage_runtime_seconds': runtime,
                   'peak_rss_mb': peak, 'submission_audit_peak_rss_mb': audit_peak,
                   'completed_pipeline_oom_killed': any(m.get('oom_killed') is not False for m in [bm, nm]), 'runner_stage_metrics_at_audit': stages, 'prior_attempts': context.get('prior_attempts', [])}
    write_json(ROOT / 'audit/engineering_metrics.json', engineering)
    head = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git', 'status', '--short'], text=True)
    lines = ['# Frozen u1e3_g060 Test-side Hidden Probe Audit', '', '## A. Execution status', '',
             f'- Repo HEAD: `{head}`', f'- Workspace dirty: {"YES" if dirty.strip() else "NO"}',
             f'- Python: {context.get("python_version")}', f'- Memory limit: {context.get("memory_limit_bytes")} bytes',
             '- GPU used: NO (official PyTorch/torchvision CPU NMS)', '- Test inference rerun: NO',
             '- Hidden/Test GT accessed: NO', '- Competition submission performed: NO',
             '- Scope statements are execution attestations retained in `audit/execution_context.json`; automated checks below validate saved prediction artifacts.',
             '- No image-specific or class-specific new rule was introduced; Test images/labels were not opened to alter predictions.',
             '- Task scripts: ' + ', '.join(f'`{p}`' for p in context['task_files']),
             '- Preexisting workspace changes are preserved and listed in `audit/execution_context.json`.', '',
             '## B. Cache audit', '', '| cache | images | rows | schema | coord residual | status |',
             '|---|---:|---:|---|---:|---|']
    for key, value in audit['caches'].items():
        lines.append(f'| {key} | {value["npz_images"]} | {value["rows_actual"]:,} | 14 columns | {value["max_coordinate_residual"]:.8g} | {value["status"]} |')
    lines += ['', 'Global xyxy = columns 2:6; tile-local xyxy = columns 10:14. Global = local + tile origin. All three image sets agree.', '',
              '## C. Formal baseline Test reproduction', '', '- Expected detection count: 2,403,809',
              f'- Actual: {bi["detections"]:,}', f'- Stitched: {bm["stitched"]:,}', f'- JSON size: {bi["file_size_bytes"]:,} bytes',
              f'- Reproduction: {"PASS" if bi["detections"] == 2403809 else "FAIL"}',
              f'- Backend: `{bm.get("nms_backend")}`. Script40 reuses Script18 NMS, stitch, and submission formatting. The legacy log label `torchvision-cuda` names a helper; the actual tensor device is CPU.',
              f'- Byte-identical repeat baseline SHA: `{bi["sha256"]}` (compared against earlier verified reproduction).',
              '- Historical original 98.09 submission JSON was not found among existing submission artifacts. An older g065 submission exists but is not the formal baseline. The audited fresh reproduction is the reference for the streaming diff.',
              '', '## D. Frozen u1e3_g060 selector', '', '- min_score = 1e-5; upper_score = 1e-3; overlap_gate = 0.60.',
              '- HFlip classes = {0,2,3,4,7}; global box = 2:6; local box = 10:14.',
              '- Builder directly reuses Val41 `active_reference()` and `max_iou_chunked()` with float32 and chunks 256/1024.',
              '- Inclusive score gate and strict max-IoU < 0.60 match Val. Non-HFlip classes use Original only.',
              '- No parameter search or additional Test configuration was generated.', '',
              '## E. Candidate audit', '', f'- Baseline total candidates: {test_total:,}',
              f'- Score qualified: {cm["baseline_candidates_score_qualified"]:,}',
              f'- Overlap qualified / added: {test_added:,}', f'- Rows after Original + complement merge: {cm["rows_total_after_merge"]:,}', '',
              '| class | Val added | Test added | Test/Val added per image | baseline final | new final | delta |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for row in perclass:
        r = row['test_val_added_per_image_ratio']
        lines.append(f'| {row["class"]} | {row["val_added"]:,} | {row["test_added"]:,} | {r:.4f} | {row["baseline_final"]:,} | {row["new_final"]:,} | {row["final_delta"]:+,} |' if r is not None else f'| {row["class"]} | {row["val_added"]:,} | {row["test_added"]:,} | undefined | {row["baseline_final"]:,} | {row["new_final"]:,} | {row["final_delta"]:+,} |')
    lines += ['', '## F. Distribution audit', '', '| per-image metric | min | mean | p50 | p90 | p95 | p99 | max |',
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name, value in distributions.items():
        lines.append('| ' + name + ' | ' + ' | '.join(f'{value[key]:.2f}' for key in ['min', 'mean', 'median', 'p90', 'p95', 'p99', 'max']) + ' |')
    lines += ['', '| comparison | Val | Test | Test/Val |', '|---|---:|---:|---:|']
    for key in ['added_per_image', 'added_per_source_candidate', 'final_delta_per_image']:
        r = comparison['test_over_val'][key]
        lines.append(f'| {key} | {comparison["val"][key]:.6f} | {comparison["test"][key]:.6f} | {r:.6f} |' if r is not None else f'| {key} | {comparison["val"][key]:.6f} | {comparison["test"][key]:.6f} | undefined |')
    lines += ['', f'- Val source Baseline total is read from its hash-verified cache manifest: {val_total:,}.',
              '- Class proposal shares and their Test/Val ratios are saved in `audit/per_class_audit.csv`.',
              f'- Top two images contain {top2:.4%} of added proposals.',
              f'- Engineering anomaly: {"YES" if anomaly_reasons else "NO"}.',
              '- Investigation reasons: ' + ('; '.join(anomaly_reasons) if anomaly_reasons else 'none'),
              '- These are descriptive sanity checks. No output was changed due to a distribution or filename.', '',
              '| rank | largest added proposals: image | added | largest final delta: image | delta |',
              '|---:|---|---:|---|---:|']
    for a, b in zip(outliers[:20], outliers[20:]):
        lines.append(f'| {a["rank"]} | {a["image"]} | {a["value"]:,} | {b["image"]} | {b["value"]:+,} |')
    lines += ['', '## G. Exact Final Combo output', '', f'- Baseline final detections: {bi["detections"]:,}',
              f'- u1e3 final detections: {ni["detections"]:,}', f'- Final delta: {diff["delta"]:+,}',
              f'- Stitched baseline / u1e3 / delta: {bm["stitched"]:,} / {nm["stitched"]:,} / {nm["stitched"] - bm["stitched"]:+,}',
              f'- Exactly identical: {diff["exactly_identical_detections"]:,}', f'- New only: {diff["new_only_detections"]:,}',
              f'- Baseline only: {diff["baseline_only_detections"]:,} ({baseline_only_ratio:.4%})',
              f'- Same image/category/bbox, including potential score replacements: {diff["identical_image_category_bbox"]:,}. Score-only replacements: {diff["same_bbox_score_replacements"]:,}.',
              '- Exact comparison retains rounded score and duplicate multiplicity; image/category/bbox comparison is also recorded.',
              '- Small replacements are possible after NMS/stitching. A baseline-only fraction >=5% triggers investigation; script hashes, per-image counts, and artifact totals are reconciled independently.', '',
              '## H. Submission integrity', '', f'- Path: `{ni["path"]}`', f'- SHA256: `{ni["sha256"]}`',
              f'- File size: {ni["file_size_bytes"]:,} bytes ({ni["file_size_bytes"] / (1024 ** 2):.2f} MiB)',
              f'- Detections: {ni["detections"]:,}', f'- Images: {ni["image_count"]}',
              '- JSON validation uses the complete strict Script40 stream layout with bounded memory. Missing commas, trailing commas/content, duplicate keys, nonstandard NaN/Infinity, and truncated arrays are rejected.',
              f'- JSON/schema/category/bbox/score/image valid: {ni["json_valid"]}/{ni["schema_valid"]}/{ni["category_valid"]}/{ni["bbox_valid"]}/{ni["score_valid"]}/{ni["image_valid"]}',
              '- SHA, size, class counts, per-image counts, candidate provenance, and stitch totals reconcile with actual output and metrics.', '',
              '## I. Engineering', '', f'- Cache audit runtime: {cache_runtime:.3f} s',
              f'- Baseline reproduction runtime: {bm["runtime_seconds"]:.3f} s', f'- Builder runtime: {cm["runtime_seconds"]:.3f} s',
              f'- Exact Final Combo runtime: {nm["runtime_seconds"]:.3f} s', f'- Submission integrity/diff/report audit runtime: {audit_runtime:.3f} s',
              f'- Total stage work runtime: {runtime:.3f} s (includes measured audit processing; excludes prior failed attempts and shell setup).',
              f'- Peak RSS across stages: {peak:.3f} MiB; submission audit peak RSS: {audit_peak:.3f} MiB.',
              '- Completed official pipeline OOM/Killed: NO; both streaming subprocesses exited 0.',
              '- Prior attempts, retained for transparency:']
    lines += ['  - ' + item for item in context.get('prior_attempts', [])]
    lines += ['', 'The earlier SIGKILL is not evidence of a confirmed kernel OOM. Its old 2 GiB attempt and the interrupted NumPy path did not produce the recommended output. The completed run uses unchanged formal PyTorch CPU code.',
              'Implementation hashes are saved in both metrics files and reconciled. Full count/provenance checks are in `audit/reconciliation_checks.json`.', '',
              f'## J. Test-side final decision: {decision}', '']
    lines += [f'{i}. {"PASS" if passed else "FAIL"} — {label}.' for i, (label, passed) in enumerate(conditions, 1)]
    failed = [key for key, passed in checks.items() if not passed]
    if failed:
        lines += ['', 'Failed detailed checks:'] + ['- ' + key for key in failed]
    lines += ['', f'`recommended_submission = {ni["path"] if ready else "NONE"}`', f'`sha256 = {ni["sha256"]}`',
              f'`detections = {ni["detections"]}`', f'`file_size = {ni["file_size_bytes"]}`', '',
              'Competition submission performed = NO. The file is for a human to submit if they choose.', '',
              '## Frozen interpretation after a manual Hidden result', '',
              '- Hidden ΔTP >= +3: cross-distribution mechanism validated; promote corrected Baseline HR global complement.',
              '- Hidden ΔTP +1 or +2: retain the result, submit none of the other five configurations, and move to VFlip.',
              '- Hidden ΔTP = 0: freeze the Baseline HR route; no further gates/uppers; next is RareOS VFlip.',
              '- Hidden regression: first verify submission/pipeline reproducibility; if correct, the route is NO-GO.']
    (ROOT / 'report.md').write_text('\n'.join(lines) + '\n')
    summary = []
    for variant, metrics, parsed in [('baseline_reproduction', bm, bi), ('u1e3_g060', nm, ni)]:
        row = {'variant': variant, 'images': parsed['image_count'], 'detections': parsed['detections'],
               'stitched': metrics['stitched'], 'detection_delta': parsed['detections'] - bi['detections'],
               'stitched_delta': metrics['stitched'] - bm['stitched'], 'added_candidates': test_added if variant == 'u1e3_g060' else 0,
               'file_size_bytes': parsed['file_size_bytes'], 'sha256': parsed['sha256'], 'runtime_seconds': metrics['runtime_seconds'],
               'peak_rss_mb': metrics['peak_rss_mb'], 'decision': decision if variant == 'u1e3_g060' else ('BASELINE_REPRODUCTION_PASS' if bi['detections'] == 2403809 else 'BASELINE_REPRODUCTION_FAIL'), 'submission_path': parsed['path']}
        row.update({f'final_class_{i}': parsed['class_counts'][c] for i, c in enumerate(CLASS_NAMES)})
        summary.append(row)
    write_csv(ROOT / 'summary.csv', summary, list(summary[0]))
    print(decision, ni['path'], ni['sha256'], ni['detections'], flush=True)
    if not ready:
        raise RuntimeError('STOP conditions failed; consult report.md and reconciliation_checks.json')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        decision_path = ROOT / 'decision.json'
        prior = read_json(decision_path) if decision_path.exists() else {}
        if prior.get('decision') != 'STOP_AND_INVESTIGATE':
            write_json(decision_path, {'decision': 'STOP_AND_INVESTIGATE', 'recommended_submission': None, 'audit_exception': str(exc)})
            (ROOT / 'report.md').write_text('# Test-side audit incomplete\n\nSTOP_AND_INVESTIGATE\n\nAudit exception: ' + str(exc) + '\n\nNo submission is recommended. Competition submission performed = NO.\n')
        raise
