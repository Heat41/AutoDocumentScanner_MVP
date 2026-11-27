"""Local PR9.5 paired evidence comparison on unchanged PR9.4 proposals.

Explicit null annotations denote negative scenes; omitted entries do not.
Only anonymous measurements are exported. No identity or crop decisions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from autodocscanner.core.pr95_region_evidence import audit_region_evidence
from autodocscanner.tools.pr94_validation import evaluate_photo as evaluate_parent, summarize as summarize_parent
from autodocscanner.tools.pr9_holdout_validation import EXTENSIONS, _stats

FIELDS=('multiscale_sides','centered_side_count','multiscale_boundary_supported',
        'boundary_and_structure','partial_boundary_and_structure','frame_limited','group_id','structured_inner_groups','structured_outer_groups')
EVIDENCE=('boundary_supported','combined_evidence','multiscale_boundary_supported',
          'boundary_and_structure','partial_boundary_and_structure')


def evaluate_photo(image, annotation, *, detectors=None, audit=audit_region_evidence, negative=False):
    captured=[]
    def discover(im):
        result=audit(im);captured.append(result);return result
    row=evaluate_parent(image,annotation,detectors=detectors,discover=discover,negative=negative)
    if row['status'] not in ('evaluated','negative_scene'): return row
    result=captured[0]
    for c, original in zip(row['candidates'],result['candidates'],strict=True):
        c.update({k:original[k] for k in FIELDS})
    row['groups']=result['groups']
    row['competing_evidence_groups']=result['competing_evidence_groups']
    return row


def summarize(rows,role):
    out=summarize_parent(rows,role)
    positives=[r for r in rows if r['status']=='evaluated']
    negatives=[r for r in rows if r['status']=='negative_scene']
    flat=[c for r in positives for c in r['candidates']]
    def good(c): return c['error']['mean']<=.02
    metrics={}
    for key in EVIDENCE:
        supported=[c for c in flat if c[key]]
        group_count=0; accurate_groups=0; severe_groups=0
        for r in positives:
            for group in r['groups']:
                members=[r['candidates'][i] for i in group['members'] if r['candidates'][i][key]]
                if members:
                    group_count+=1
                    accurate_groups+=any(good(c) for c in members)
                    severe_groups+=any(c['error']['mean']>.10 for c in members)
        metrics[key]={'supported_candidates':len(supported),'accurate_supported':sum(good(c) for c in supported),
                      'wrong_supported':sum(not good(c) for c in supported),
                      'severe_supported':sum(c['error']['mean']>.10 for c in supported),
                      'accurate_unsupported':sum(good(c) and not c[key] for c in flat),
                      'photos_with_accurate_supported':sum(any(c[key] and good(c) for c in r['candidates']) for r in positives),
                      'supported_groups':group_count,'groups_with_accurate_supported_member':accurate_groups,
                      'groups_with_severe_supported_member':severe_groups,
                      'supported_error':_stats([c['error']['mean'] for c in supported]),
                      'negative_photos_with_evidence':sum(any(c[key] for c in r['candidates']) for r in negatives)}
    out['paired_evidence_metrics']=metrics
    out['geometry_groups']={'positive_count':sum(len(r['groups']) for r in positives),
                            'negative_count':sum(len(r['groups']) for r in negatives),
                            'competing_evidence_photos':sum(r['competing_evidence_groups'] for r in positives),
                            'competing_negative_photos':sum(r['competing_evidence_groups'] for r in negatives)}
    states={}
    for c in flat:
        for s in c['multiscale_sides']:
            key=s['state'];states[key]=states.get(key,0)+1
    out['side_state_counts']=states
    out['note']='Unchanged PR9.4 proposals. Evidence is advisory; partial evidence is not complete boundary support. Groups are bounded geometric duplicates, not independent votes. Oracle is retrospective.'
    return out


def evaluate_folder(folder,annotations,role):
    if not isinstance(annotations,dict): raise ValueError('invalid annotations')
    rows=[]
    for path in sorted(Path(folder).iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS: continue
        try: im=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        except (OSError,ValueError,cv2.error): im=None
        rows.append(evaluate_photo(im,annotations.get(path.name),negative=path.name in annotations and annotations[path.name] is None))
    return summarize(rows,role)


def main(argv=None):
    parser=argparse.ArgumentParser(description='PR9.5 physical/region evidence audit, local only')
    parser.add_argument('folder',type=Path);parser.add_argument('--annotations',type=Path,required=True)
    parser.add_argument('--dataset-role',choices=('development','holdout'),required=True)
    args=parser.parse_args(argv)
    if not args.folder.is_dir(): parser.error('Folder foto tidak ditemukan')
    if not args.annotations.is_file(): parser.error('File anotasi tidak ditemukan')
    try: result=evaluate_folder(args.folder,json.loads(args.annotations.read_text(encoding='utf-8-sig')),args.dataset_role)
    except (OSError,ValueError,TypeError,cv2.error): parser.error('Dataset, anotasi atau kontrak evaluasi tidak valid')
    print(json.dumps(result,indent=2,ensure_ascii=False,allow_nan=False))
    return 0 if result['evaluated_photo_count'] or result['negative_scene_metrics']['photo_count'] else 2


if __name__=='__main__':
    raise SystemExit(main())
