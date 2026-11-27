"""PR9.4 local-only retrospective evaluation of independent region proposals.

Annotations map names to four original corners; explicit null means a scene
without the target document. Missing names remain unannotated, not negatives.
No coordinate, filename or image data in the report.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from autodocscanner.core.pr94_region_boundary import discover_regions,FAMILIES
from autodocscanner.core.pr93_independent_boundary import _image_ok
from autodocscanner.tools.pr9_holdout_validation import SOURCES,EXTENSIONS,_quad,_errors,_stats,_default_detectors
from autodocscanner.core.pr92_boundary_recovery import inspect_geometry


def evaluate_photo(image,annotation,*,detectors=None,discover=discover_regions,negative=False):
    if not _image_ok(image): return {'status':'invalid_image'}
    reference=_quad(annotation,image.shape)
    if reference is None and not negative: return {'status':'invalid_annotation'}
    result=discover(image.copy())
    if result.get('auto_crop_allowed') is not False or result.get('auto_select','missing') is not None:
        raise ValueError('unsafe discovery contract')
    def measure(corners):
        q=_quad(corners,image.shape)
        if q is None or negative: return None
        mean,worst=_errors(q,reference,image.shape)
        return {'mean':mean,'max':worst}
    rows=[]
    for c in result['candidates']:
        e=measure(c['corners'])
        if _quad(c['corners'],image.shape) is None: raise ValueError('invalid region geometry')
        # Explicit allowlist, no coordinates, filenames or pixel data.
        rows.append({'error':e,**{k:c[k] for k in ('family','geometry','boundary_supported','side_states','structure','matching_families','contains_supported_region','inside_supported_region','contains_structured_supported_region','combined_evidence')}})
    if negative:
        return {'status':'negative_scene','candidates':rows,'family_counts':result['family_counts'],'competing_combined_geometry':result['competing_combined_geometry']}
    baselines={}
    if detectors is None: detectors=_default_detectors()
    for s in SOURCES:
        try: corners,_=detectors[s].detect(image.copy())
        except (KeyError,AttributeError,cv2.error,ValueError,TypeError,ArithmeticError):
            baselines[s]={'status':'detector_error','error':None};continue
        geometry,_=inspect_geometry(corners,image.shape)
        baselines[s]={'status':geometry['status'],'error':measure(corners)}
    return {'status':'evaluated','baselines':baselines,'candidates':rows,'family_counts':result['family_counts'],'competing_combined_geometry':result['competing_combined_geometry']}


def summarize(rows,role):
    if role not in ('development','holdout'): raise ValueError('invalid role')
    evaluated=[r for r in rows if r['status']=='evaluated']
    def good(e): return e is not None and e['mean']<=.02
    metrics={}
    for s in SOURCES:
        es=[r['baselines'][s]['error'] for r in evaluated]; valid=[e for e in es if e is not None]
        metrics[s]={'measured':len(valid),'within_2pct':sum(good(e) for e in es),'mean_corner_error':_stats([e['mean'] for e in valid]),'worst_corner_error':_stats([e['max'] for e in valid]),'severe_over_10pct':sum(e['mean']>.10 for e in valid)}
    for family in (*FAMILIES,'all_families'):
        sets=[[c for c in r['candidates'] if family=='all_families' or c['family']==family] for r in evaluated]
        def best(cs): return min((c['error']['mean'] for c in cs),default=None)
        all_best=[best(cs) for cs in sets]; flat=[c for cs in sets for c in cs]
        entry={'candidate_count':len(flat),'photos_with_proposals':sum(bool(cs) for cs in sets),'oracle_photos_within_2pct':sum(v is not None and v<=.02 for v in all_best),'oracle_error':_stats([v for v in all_best if v is not None]),'additional_photos_with_candidate_within_2pct':sum(v is not None and v<=.02 and not any(good(r['baselines'][s]['error']) for s in SOURCES) for v,r in zip(all_best,evaluated))}
        for key in ('boundary_supported','combined_evidence'):
            supported=[c for c in flat if c[key]]
            entry[key]={'count':len(supported),'within_2pct':sum(good(c['error']) for c in supported),'over_2pct':sum(not good(c['error']) for c in supported),'severe_over_10pct':sum(c['error']['mean']>.10 for c in supported),'accurate_but_unsupported':sum(good(c['error']) and not c[key] for c in flat),'photos_with_accurate_supported':sum(any(c[key] and good(c['error']) for c in cs) for cs in sets)}
        metrics[family]=entry
    negatives=[r for r in rows if r['status']=='negative_scene']
    negative_metrics={'photo_count':len(negatives),'photos_with_boundary_support':sum(any(c['boundary_supported'] for c in r['candidates']) for r in negatives),'photos_with_combined_evidence':sum(any(c['combined_evidence'] for c in r['candidates']) for r in negatives),
                      'by_family':{f:{k:sum(any(c['family']==f and c[k] for c in r['candidates']) for r in negatives) for k in ('boundary_supported','combined_evidence')} for f in FAMILIES}}
    return {'negative_scene_metrics':negative_metrics,'competing_combined_photos':sum(r['competing_combined_geometry'] for r in evaluated),'dataset_role':role,'photo_count':len(rows),'evaluated_photo_count':len(evaluated),'source_metrics':metrics,'auto_select':None,'auto_crop_allowed':False,'error_unit':'fraction_of_original_image_diagonal','note':'Oracle is retrospective. Families may be correlated; repeated geometries are counted per family. Combined evidence is advisory, not KTP identification.','photos':[{'case_id':f'D{i:04d}',**r} for i,r in enumerate(rows,1)]}


def evaluate_folder(folder,annotations,role):
    if not isinstance(annotations,dict): raise ValueError('invalid annotations')
    rows=[]
    for path in sorted(Path(folder).iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS: continue
        try: image=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        except (OSError,ValueError,cv2.error): image=None
        rows.append(evaluate_photo(image,annotations.get(path.name),negative=path.name in annotations and annotations[path.name] is None))
    return summarize(rows,role)


def main(argv=None):
    parser=argparse.ArgumentParser(description='PR9.4 region-aware validation, local only')
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
