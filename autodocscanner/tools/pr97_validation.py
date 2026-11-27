"""Positive-only local root-cause audit; no KTP identity or crop decisions."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import cv2
import numpy as np
from autodocscanner.core.pr97_boundary_diagnostics import inspect_quad,audit_transforms,draw_sampling_overlay,preview_sampling_limits
from autodocscanner.core.pr96_region_verification import verify_regions
from autodocscanner.tools.pr96_validation import evaluate_photo as evaluate_parent,summarize
from autodocscanner.tools.pr9_holdout_validation import _quad,EXTENSIONS


def evaluate_photo(image,annotation,*,detectors=None,audit=verify_regions,overlay=None):
    captured=[]
    def run(im):
        result=audit(im);captured.append(result);return result
    row=evaluate_parent(image,annotation,detectors=detectors,audit=run,negative=False)
    if row['status']!='evaluated':return row
    ref=_quad(annotation,image.shape)
    base=inspect_quad(image,ref)
    row['root_cause_reference']={'original':base,'transforms':audit_transforms(image,ref,base),'default_preview_limits':preview_sampling_limits(image.shape)}
    # Compare exactly the same candidates, never change their states or rank.
    for c,original in zip(row['candidates'],captured[0]['candidates'],strict=True):
        q=_quad(original['corners'],image.shape)
        c['root_cause_sampling']=inspect_quad(image,q)
    if overlay is not None:
        overlay('reference',draw_sampling_overlay(image,ref,base))
        if row['candidates']:
            # Oracle overlay is explicitly retrospective; never a suggestion.
            i=min(range(len(row['candidates'])),key=lambda i:row['candidates'][i]['error']['mean'])
            q=_quad(captured[0]['candidates'][i]['corners'],image.shape)
            overlay('oracle',draw_sampling_overlay(image,q,row['candidates'][i]['root_cause_sampling']))
    return row


def evaluate_folder(folder,annotations,role,*,overlay_dir=None,evaluator=evaluate_photo):
    if not isinstance(annotations,dict):raise ValueError('invalid annotations')
    if overlay_dir is not None:
        overlay_dir=Path(overlay_dir)
        # Prevent overwriting an existing diagnostic directory or inputs.
        overlay_dir.mkdir(parents=True,exist_ok=False)
    rows=[]
    for path in sorted(Path(folder).iterdir()):
        if not path.is_file() or path.suffix.lower() not in EXTENSIONS:continue
        try: image=cv2.imdecode(np.frombuffer(path.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        except (OSError,ValueError,cv2.error):image=None
        case=f'D{len(rows)+1:04d}'
        def save(kind,im):
            ok,encoded=cv2.imencode('.png',im)
            if not ok:raise ValueError('overlay encode failed')
            (overlay_dir/f'{case}_{kind}.png').write_bytes(encoded.tobytes())
        rows.append(evaluator(image,annotations.get(path.name),overlay=save if overlay_dir is not None else None))
    report=summarize(rows,role)
    controls=[r['root_cause_reference'] for r in rows if r['status']=='evaluated']
    transforms={}
    for ctrl in controls:
        for v in ctrl['transforms']:
            m=transforms.setdefault(v['transform'],{'compared_photos':0,'excluded_geometry':0,'pr95_state_changes':0,'pr96_state_changes':0})
            if v['state_changes'] is None:m['excluded_geometry']+=1;continue
            m['compared_photos']+=1
            for method in ('pr95','pr96'):m[method+'_state_changes']+=v['state_changes'][method]
    report['root_cause_summary']={'reference_photos':len(controls),'transforms':transforms,
        'excluded_photos':{k:sum(r['status']==k for r in rows) for k in ('invalid_image','invalid_annotation')}}
    report['mode']='positive_only_root_cause_audit' 
    report['note']='PR9.5/9.6 unchanged. Reference, oracle and transform controls are retrospective; center grids are diagnostic only. No identity, selection or crop. Optional overlays contain private original pixels and stay local.'
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description='PR9.7 positive-only boundary sampling audit')
    p.add_argument('folder',type=Path);p.add_argument('--annotations',type=Path,required=True)
    p.add_argument('--dataset-role',choices=('development','holdout'),required=True)
    p.add_argument('--overlay-dir',type=Path,help='Optional NEW local folder; private image overlays, do not upload')
    a=p.parse_args(argv)
    if not a.folder.is_dir():p.error('Folder foto tidak ditemukan')
    if not a.annotations.is_file():p.error('File anotasi tidak ditemukan')
    try:r=evaluate_folder(a.folder,json.loads(a.annotations.read_text(encoding='utf-8-sig')),a.dataset_role,overlay_dir=a.overlay_dir)
    except (OSError,ValueError,TypeError,cv2.error):p.error('Dataset, anotasi, atau folder overlay tidak valid; gunakan folder overlay baru')
    print(json.dumps(r,indent=2,allow_nan=False,ensure_ascii=False))
    return 0 if r['evaluated_photo_count'] else 2


if __name__=='__main__':raise SystemExit(main())
