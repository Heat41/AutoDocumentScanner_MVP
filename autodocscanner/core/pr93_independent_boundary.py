"""PR9.3 full-image Lab line hypotheses; independent of all parent detectors.

Line segments across different channels and gaps can describe the same side.
Reconstruct possible document boundaries from intersections, then measure
persistent region transitions (PR9.2) on the joined quad. No GT, V2, OCR,
network or file writes. A supported rectangle is not proof of a document.
"""
from __future__ import annotations

import itertools
import math
import cv2
import numpy as np
from autodocscanner.core.pr92_boundary_recovery import inspect_geometry, _measure


def _image_ok(image):
    return (isinstance(image, np.ndarray) and image.dtype == np.uint8 and
            image.ndim == 3 and image.shape[2] == 3 and min(image.shape[:2]) >= 75)


def _angle(u, v):
    return math.acos(float(np.clip(abs(u @ v), 0., 1.)))


def _order(q):
    q = np.asarray(q, dtype=np.float64)
    c = q.mean(axis=0)
    return q[np.argsort(np.arctan2(q[:, 1]-c[1], q[:, 0]-c[0]))]


def _distance(a, b, diag):
    return min(float(np.linalg.norm(a-np.roll(b, k, axis=0), axis=1).mean())
               for k in range(4))/diag


def _line(segments):
    # Weight each segment by its length via sampling, not channel count.
    pts = np.concatenate([np.linspace(a, b, max(2, min(20, int(np.linalg.norm(b-a)/8))))
                          for a, b in segments]).astype(np.float32)
    vx, vy, x, y = cv2.fitLine(pts, cv2.DIST_L2, 0, .01, .01).reshape(-1)
    u = np.array([vx, vy], dtype=float)
    u /= np.linalg.norm(u)
    p = np.array([x, y], dtype=float)
    intervals = sorted(tuple(sorted(((a-p) @ u, (b-p) @ u))) for a, b in segments)
    merged = []
    for lo, hi in intervals:
        if merged and lo <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], hi)
        else:
            merged.append([lo, hi])
    return {'p': p, 'u': u, 'intervals': merged,
            'length': sum(b-a for a,b in merged), 'segments': segments}


def _segments(lab):
    h,w = lab.shape[:2]
    minlen = max(12., .09*min(h,w))
    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    raw=[]
    for i in range(3):
        channel = np.ascontiguousarray(lab[:,:,i])
        detected = lsd.detect(channel)[0]
        if detected is None:
            continue
        for coords in detected.reshape(-1,4):
            a,b=coords[:2].astype(float),coords[2:].astype(float)
            length=float(np.linalg.norm(b-a))
            if length>=minlen:
                raw.append((length,a,b))
    raw.sort(key=lambda r:-r[0])
    lines=[]
    tolerance=max(1.8,.004*min(h,w))
    for length,a,b in raw[:180]:
        u=(b-a)/length
        matches=[]
        for idx,line in enumerate(lines):
            n=np.array([-line['u'][1],line['u'][0]])
            deviation=max(abs((a-line['p'])@n),abs((b-line['p'])@n))
            if _angle(u,line['u'])<=math.radians(6) and deviation<=tolerance:
                matches.append((deviation,idx))
        if matches:
            _,idx=min(matches)
            if len(lines[idx]['segments'])<24:
                lines[idx]=_line(lines[idx]['segments']+[(a,b)])
        elif len(lines)<60:
            lines.append(_line([(a,b)]))
    lines.sort(key=lambda l:-l['length'])
    # Limit combinations while keeping orientation diversity via round robin.
    buckets=[[] for _ in range(12)]
    for line in lines:
        angle=math.atan2(line['u'][1],line['u'][0])%math.pi
        buckets[min(11,int(angle/math.pi*12))].append(line)
    result=[]
    while len(result)<32 and any(buckets):
        for bucket in buckets:
            if bucket and len(result)<32:
                result.append(bucket.pop(0))
    return result


def _cross(a,b):
    u,v=a['u'],b['u']
    det=float(u[0]*v[1]-u[1]*v[0])
    if abs(det)<.33:
        return None
    delta=b['p']-a['p']
    return a['p']+u*((delta[0]*v[1]-delta[1]*v[0])/det)


def _coverage(line,a,b):
    lo,hi=sorted(((a-line['p'])@line['u'],(b-line['p'])@line['u']))
    return sum(max(0.,min(hi,end)-max(lo,start)) for start,end in line['intervals'])/max(hi-lo,1e-8)


def _line_quads(lab, max_candidates):
    h,w=lab.shape[:2]; diag=float(np.hypot(h,w)); lines=_segments(lab)
    pairs=[]
    for i,j in itertools.combinations(range(len(lines)),2):
        a,b=lines[i],lines[j]
        # Perspective allows opposite sides to converge. Don't assume parallel.
        if _angle(a['u'],b['u'])>math.radians(43):
            continue
        d=b['p']-a['p']
        na=np.array([-a['u'][1],a['u'][0]])
        nb=np.array([-b['u'][1],b['u'][0]])
        if min(abs(d@na),abs(d@nb))<.055*min(h,w):
            continue
        pairs.append((min(a['length'],b['length']),i,j))
    pairs.sort(reverse=True)
    # Bounded combinatorial search; no score is a probability or identity test.
    candidates=[]
    for (_,i,j),(_,k,l) in itertools.combinations(pairs[:64],2):
        if len({i,j,k,l})!=4:
            continue
        a,b,c,d=[lines[t] for t in (i,j,k,l)]
        corners=[_cross(a,c),_cross(a,d),_cross(b,d),_cross(b,c)]
        if any(p is None for p in corners):
            continue
        q=np.asarray(corners)
        if (not np.isfinite(q).all() or np.any(q<0) or
            np.any(q[:,0]>w-1) or np.any(q[:,1]>h-1)):
            continue
        if not cv2.isContourConvex(q.astype(np.float32)):
            continue
        area=abs(float(cv2.contourArea(q.astype(np.float32))))/(h*w)
        lengths=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
        if not .015<=area<=.94 or min(lengths)<.055*min(h,w):
            continue
        coverage=[_coverage(line,q[idx],q[(idx+1)%4]) for idx,line in enumerate((a,d,b,c))]
        if min(coverage)<.42 or np.mean(coverage)<.65:
            continue
        candidates.append((min(coverage)+.25*np.mean(coverage),_order(q),coverage))
    candidates.sort(key=lambda r:-r[0])
    # Dedup using all cyclic correspondences, including rotated documents.
    out=[]
    for _,q,cov in candidates:
        if any(_distance(q,old[0],diag)<.009 for old in out):
            continue
        out.append((q,cov))
        if len(out)>=max_candidates:
            break
    return out,len(lines)


def discover_boundaries(image, *, max_candidates=16, max_longest=960):
    """Full-frame search + original-image physical evidence, manual-only.

    Bounded line support orders exploration only. No preferred candidate.
    Geometric alternatives with weak evidence remain visible for evaluation.
    """
    out={'status':'invalid_image','candidates':[],'line_count':0,
         'auto_select':None,'auto_crop_allowed':False,'requires_manual_confirmation':True,
         'supported_geometry_count':0,'competing_supported_geometry':False}
    if not _image_ok(image):
        return out
    if type(max_candidates) is not int or not 1<=max_candidates<=32 or type(max_longest) is not int or not 160<=max_longest<=1600:
        raise ValueError('invalid search bounds')
    h,w=image.shape[:2]
    scale=min(1.,max_longest/max(h,w))
    sw,sh=max(2,round(w*scale)),max(2,round(h*scale))
    small=cv2.resize(image,(sw,sh),interpolation=cv2.INTER_AREA) if scale<1 else image
    lab=cv2.GaussianBlur(cv2.cvtColor(small,cv2.COLOR_BGR2LAB),(3,3),0)
    raw,out['line_count']=_line_quads(lab,max_candidates)
    if not raw:
        out['status']='no_line_hypothesis';return out
    # Evidence and corner coordinates measured in original image space.
    full_lab=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2LAB),(3,3),0)
    # Match OpenCV pixel-center resize convention (separate rounded scales).
    factor=np.array([w/sw,h/sh])
    for q,coverage in raw:
        q=(q+.5)*factor-.5
        geom,p=inspect_geometry(q,image.shape)
        if geom['status'] not in ('in_frame','near_frame'):
            continue
        sides=[s for s,_ in _measure(full_lab,p)]
        centered=all(s['state']=='centered' for s in sides)
        out['candidates'].append({'corners':p.tolist(),'geometry':geom['status'],
             'line_coverage_min':float(min(coverage)), 'side_evidence':sides,
             'four_centered_transitions':centered,
             'boundary_supported':centered and geom['status']=='in_frame',
             'nested_in_supported_region':False})
    # Nesting is ambiguity, not proof the larger or smaller rectangle is KTP.
    for row in out['candidates']:
        q=np.asarray(row['corners'],np.float32); area=abs(cv2.contourArea(q))
        for other in out['candidates']:
            if other is row or not other['boundary_supported']:
                continue
            p=np.asarray(other['corners'],np.float32)
            if abs(cv2.contourArea(p))>1.2*area and all(cv2.pointPolygonTest(p,tuple(map(float,pt)),False)>=0 for pt in q):
                row['nested_in_supported_region']=True;break
    out['supported_geometry_count']=sum(row['boundary_supported'] for row in out['candidates'])
    out['competing_supported_geometry']=out['supported_geometry_count']>1
    out['status']='hypotheses_for_manual_review' if out['candidates'] else 'no_valid_hypothesis'
    return out
