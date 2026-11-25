"""Experimental region-aware proposals and advisory document structure evidence.

Region and chroma searches are independent of baseline and line hypotheses.
Structure is not KTP identity: books, other cards and packaging can match.
No OCR, ML, ground truth, V2, selection, output crop or file writes.
"""
from __future__ import annotations
import cv2
import numpy as np
from autodocscanner.core.pr93_independent_boundary import discover_boundaries, _image_ok, _order, _distance
from autodocscanner.core.pr92_boundary_recovery import inspect_geometry, _measure

FAMILIES = ('independent_lines', 'outer_region', 'chroma')


def _region_proposals(image, limit=12):
    """Bounded global Lab component and chroma contour hypotheses.

    Develop the earlier V3 region/chroma concepts with separate provenance,
    polarity diversity, size-band budgets and correct resize coordinates.
    """
    h,w=image.shape[:2]
    scale=min(1.,800/max(h,w)); sw,sh=round(w*scale),round(h*scale)
    small=cv2.resize(image,(sw,sh),interpolation=cv2.INTER_AREA) if scale<1 else image
    lab=cv2.GaussianBlur(cv2.cvtColor(small,cv2.COLOR_BGR2LAB),(5,5),0)
    m=min(sh,sw); diag=np.hypot(h,w)
    kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(max(3,round(.012*m)|1),)*2)
    buckets={f:[[],[],[]] for f in FAMILIES[1:]}
    def collect(mask,family,mode):
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
        contours,_=cv2.findContours(mask,mode,cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(contours,key=cv2.contourArea,reverse=True)[:24]:
            area=abs(cv2.contourArea(c)); fraction=area/(sh*sw)
            if not .015<=fraction<=.94: continue
            hull=cv2.convexHull(c); perimeter=cv2.arcLength(hull,True)
            for eps in (.012,.025,.04):
                p=cv2.approxPolyDP(hull,eps*perimeter,True)
                if len(p)!=4: continue
                q=_order(p.reshape(4,2)); qa=abs(cv2.contourArea(q.astype(np.float32)))
                if qa<=0 or area/qa<.65: continue
                q=(q+.5)*np.array([w/sw,h/sh])-.5
                geom,q=inspect_geometry(q,image.shape)
                if geom['status'] not in ('in_frame','near_frame'): continue
                band=0 if fraction<.08 else 1 if fraction<.32 else 2
                buckets[family][band].append((min(1.,area/qa),q));break
    for k in range(3):
        channel=np.ascontiguousarray(lab[:,:,k])
        if np.std(channel)<2.: continue
        t,_=cv2.threshold(channel,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)
        for threshold in sorted(set([int(t),int(np.percentile(channel,35)),int(np.percentile(channel,65))])):
            for mode in (cv2.CMP_GT,cv2.CMP_LE):
                collect(cv2.compare(channel,threshold,mode),'outer_region',cv2.RETR_EXTERNAL)
        if k:
            collect(cv2.Canny(channel,20,65),'chroma',cv2.RETR_LIST)
    out={}
    for f,bands in buckets.items():
        for b in bands: b.sort(key=lambda x:-x[0])
        out[f]=[]
        while len(out[f])<limit and any(bands):
            for b in bands:
                while b and len(out[f])<limit:
                    _,q=b.pop(0)
                    if all(_distance(q,old,diag)>=.008 for old in out[f]):
                        out[f].append(q);break
    return out


def _rectify(image,q):
    # Use the longer opposing sides as horizontal; evaluate both axes below.
    lengths=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
    if lengths[::2].mean()<lengths[1::2].mean(): q=np.roll(q,1,axis=0)
    dst=np.array([[0,0],[319,0],[319,199],[0,199]],np.float32)
    t=cv2.getPerspectiveTransform(q.astype(np.float32),dst)
    return cv2.warpPerspective(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),t,(320,200))


def _stroke_groups(gray):
    """Look for rows of separated compact strokes, not long lines or noise.

    No character recognition, words, layout template or fixed card position.
    """
    h,w=gray.shape; groups=[]; ratios=[]
    smooth=cv2.GaussianBlur(gray,(3,3),0)
    for polarity in (cv2.THRESH_BINARY_INV,cv2.THRESH_BINARY):
        mask=cv2.adaptiveThreshold(smooth,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,polarity,21,7 if polarity==cv2.THRESH_BINARY_INV else -7)
        n,_,stats,centers=cv2.connectedComponentsWithStats(mask,8)
        glyphs=[]
        for i in range(1,n):
            x,y,ww,hh,a=stats[i]
            if (x<1 or y<1 or x+ww>=w-1 or y+hh>=h-1 or not 4<=hh<=.14*h
                    or not 2<=ww<=.10*w or not .12<=ww/hh<=2.6 or a<6): continue
            glyphs.append((float(centers[i,0]),float(centers[i,1]),float(hh),float(ww)))
        # Limit pathological textures; dense components are not text proof.
        if len(glyphs)>350: continue
        for x,y,hh,ww in glyphs:
            neighbors=[(xx,yy) for xx,yy,h2,w2 in glyphs if abs(yy-y)<=.45*max(hh,h2)
                       and .55<=h2/hh<=1.8 and abs(xx-x)<=5*hh]
            if len(neighbors)>=4 and max(xx for xx,yy in neighbors)-min(xx for xx,yy in neighbors)>=2*hh:
                groups.append((x,y)); ratios.append(ww/hh)
    # Distribution across several interior zones, not just one logo/portrait.
    bins={(min(3,int(x/w*4)),min(3,int(y/h*4))) for x,y in groups}
    rows={min(7,int(y/h*8)) for x,y in groups}
    return {'aligned_stroke_centers':len(set(groups)), 'stroke_tiles':len(bins), 'stroke_bands':len(rows),
            'stroke_aspect_iqr':float(np.percentile(ratios,75)-np.percentile(ratios,25)) if ratios else 0.}


def inspect_structure(image,q):
    gray=_rectify(image,q)[16:-16,24:-24]
    # Both axes, independent of 90-degree/180-degree orientation.
    options=[_stroke_groups(gray),_stroke_groups(np.ascontiguousarray(gray.T))]
    cue=max(options,key=lambda a:(a['stroke_tiles'],a['stroke_bands'],a['aligned_stroke_centers']))
    cue['distributed_stroke_pattern']=bool(cue['stroke_tiles']>=4 and cue['stroke_bands']>=3 and cue['aligned_stroke_centers']>=12)
    return cue


def discover_regions(image, *, line_discover=discover_boundaries, region_discover=_region_proposals):
    """Report independent proposal families and region/structure cues only.

    Agreement can be correlated and text-like strokes can occur on non-KTPs.
    Keep alternatives including near-frame, nested and unsupported hypotheses.
    """
    out={'status':'invalid_image','candidates':[],'family_counts':{},'auto_select':None,
         'auto_crop_allowed':False,'requires_manual_confirmation':True}
    if not _image_ok(image): return out
    lines=line_discover(image.copy())
    if lines.get('auto_select','missing') is not None or lines.get('auto_crop_allowed') is not False:
        raise ValueError('unsafe line discovery contract')
    regions=region_discover(image.copy())
    sources={'independent_lines':[c['corners'] for c in lines['candidates']],**regions}
    lab=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2LAB),(3,3),0)
    diag=np.hypot(*image.shape[:2]); rows=[]
    for family in FAMILIES:
        out['family_counts'][family]=0
        for corners in sources.get(family,[]):
            geom,q=inspect_geometry(corners,image.shape)
            if geom['status'] not in ('in_frame','near_frame'): continue
            out['family_counts'][family]+=1
            sides=[s for s,_ in _measure(lab,q)]
            boundary=geom['status']=='in_frame' and all(s['state']=='centered' for s in sides)
            rows.append({'corners':q.tolist(),'family':family,'geometry':geom['status'],
                         'boundary_supported':boundary,'side_states':[s['state'] for s in sides],
                         'structure':inspect_structure(image,q),'matching_families':[],
                         'contains_supported_region':False,'inside_supported_region':False,'contains_structured_supported_region':False})
    for row in rows:
        q=np.asarray(row['corners'],np.float32); area=abs(cv2.contourArea(q)); matches=set()
        for other in rows:
            if other is row: continue
            p=np.asarray(other['corners'],np.float32); a=abs(cv2.contourArea(p))
            if other['family']!=row['family'] and _distance(q,p,diag)<.012:
                matches.add(other['family'])
            if not other['boundary_supported']: continue
            intersection,_=cv2.intersectConvexConvex(q,p)
            if a>1.3*area and intersection>=.95*area: row['inside_supported_region']=True
            if area>1.3*a and intersection>=.95*a:
                row['contains_supported_region']=True
                if other['structure']['distributed_stroke_pattern']: row['contains_structured_supported_region']=True
        row['matching_families']=sorted(matches)
        # Advisory conjunction, not a hard acceptance gate. A nested actual
        # card remains a candidate; the enclosing surface is not preferred.
        row['combined_evidence']=bool(row['boundary_supported'] and row['matching_families'] and row['structure']['distributed_stroke_pattern'])
    out['candidates']=rows
    combined=[r for r in rows if r['combined_evidence']]
    out['competing_combined_geometry']=any(_distance(np.asarray(a['corners']),np.asarray(b['corners']),diag)>=.02
                                          for i,a in enumerate(combined) for b in combined[i+1:])
    out['status']='hypotheses_for_manual_review' if rows else 'no_hypothesis'
    return out
