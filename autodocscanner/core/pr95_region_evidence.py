"""PR9.5 shadow evidence audit, not document identity or candidate selection.

Keep PR9.4 hypotheses unchanged. Separate measurable border, weak contrast,
competing transitions and frame censorship. Compare at multiple distances
with local noise estimates; do not count correlated detector families as
independent votes. No annotations, OCR, network, file IO or automatic crop.
"""
from __future__ import annotations

import cv2
import numpy as np
from autodocscanner.core.pr94_region_boundary import discover_regions, FAMILIES
from autodocscanner.core.pr93_independent_boundary import _image_ok, _distance
from autodocscanner.core.pr92_boundary_recovery import inspect_geometry


def _side(lab, a, b, center, unit):
    """Sample paired strips around a side, with visibility tracked explicitly."""
    tangent = (b-a)/np.linalg.norm(b-a)
    normal = np.array([-tangent[1], tangent[0]])
    if normal @ (center-(a+b)/2) < 0:
        normal = -normal
    points = a+np.linspace(.12,.88,48)[:,None]*(b-a)
    offsets = np.linspace(-2*unit,2*unit,17)
    h,w = lab.shape[:2]
    scales = []
    for scale in (.6,1.2,2.4):
        # Three samples per strip, avoiding the edge itself. A thin line
        # cannot explain a persistent step across separated strips.
        dist = unit*scale*np.array([-1.6,-1.2,-.8,.8,1.2,1.6])
        xy = points[:,None,None,:]+(offsets[None,:,None,None]+dist[None,None,:,None])*normal
        visible = ((xy[...,0]>=0)&(xy[...,0]<=w-1)&(xy[...,1]>=0)&(xy[...,1]<=h-1)).all(axis=2)
        pixels = cv2.remap(lab,xy[...,0].reshape(48,-1).astype(np.float32),
                           xy[...,1].reshape(48,-1).astype(np.float32),cv2.INTER_LINEAR).reshape(48,17,6,3).astype(float)
        outside,inside = pixels[:,:,:3],pixels[:,:,3:]
        left,right = np.median(outside,axis=2),np.median(inside,axis=2)
        delta = right-left
        strength = np.linalg.norm(delta,axis=2)
        noise = np.maximum(np.linalg.norm(outside-left[:,:,None,:],axis=3).mean(axis=2),
                           np.linalg.norm(inside-right[:,:,None,:],axis=3).mean(axis=2))
        # A local contrast/noise ratio plus an absolute quantization floor.
        support = visible & (strength>=np.maximum(4.,3.*noise+2.))
        scales.append((visible,support,delta,strength))
    # Require narrow and medium scale agreement AND direction consistency.
    # Large-scale evidence alone could reward illumination gradients.
    v0,s0,d0,n0=scales[0]; v1,s1,d1,n1=scales[1]; v2,s2,d2,n2=scales[2]
    coh01=(d0*d1).sum(axis=2)>=.7*n0*n1
    coh12=(d1*d2).sum(axis=2)>=.7*n1*n2
    visible=v0&v1&v2
    support=visible&s0&s1&s2&coh01&coh12&(n2>=.55*n1)&(n0>=.50*n2)
    coverage=support.mean(axis=0)
    bands=(support.reshape(4,12,17).mean(axis=1)>=.5).sum(axis=0)
    scores=np.where(support,np.minimum(n0,n1),0.).mean(axis=0)
    scores[(coverage<.65)|(bands<3)]=0
    # Check that an edge is localized rather than broad diffuse shading.
    peak=float(scores.max()); center_visible=float(visible[:,8].mean())
    result={'state':'weak_or_unresolved','visible_fraction':center_visible,
            'center_support_fraction':float(coverage[8]),'best_support_fraction':float(coverage.max()),
            'offset_fraction_of_unit':None,'supported_bands':int(bands.max())}
    if center_visible<.75:
        result['state']='frame_limited';return result
    if peak<=0: return result
    strong=np.flatnonzero(scores>=.85*peak)
    groups=np.split(strong,np.where(np.diff(strong)>1)[0]+1)
    if len(groups)!=1 or offsets[strong[-1]]-offsets[strong[0]]>1.5*unit:
        result['state']='ambiguous_or_diffuse';return result
    delta=float(np.average(offsets[strong],weights=scores[strong]))/unit
    result['offset_fraction_of_unit']=delta
    result['state']='search_limit' if abs(delta)>=1.8 else 'centered' if abs(delta)<=.65 else 'displaced'
    return result


def audit_region_evidence(image, *, discover=discover_regions):
    """Preserve all PR9.4 proposals; attach independent evidence dimensions.

    Geometric groups use complete-link matching so transitive chains cannot
    combine distinct rectangles. Groups are not ranked, accepted or averaged.
    """
    out={'status':'invalid_image','candidates':[],'groups':[],
         'auto_select':None,'auto_crop_allowed':False,'requires_manual_confirmation':True}
    if not _image_ok(image): return out
    parent=discover(image.copy())
    if parent.get('auto_select','missing') is not None or parent.get('auto_crop_allowed') is not False:
        raise ValueError('unsafe region discovery contract')
    lab=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2LAB),(3,3),0)
    diag=float(np.hypot(*image.shape[:2])); quads=[]
    for c in parent['candidates']:
        g,q=inspect_geometry(c['corners'],image.shape)
        if g['status'] not in ('in_frame','near_frame') or c['family'] not in FAMILIES:
            raise ValueError('invalid region hypothesis')
        shortest=float(np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1).min())
        unit=max(1.5,min(.007*min(image.shape[:2]),.025*shortest))
        sides=[_side(lab,q[i],q[(i+1)%4],q.mean(axis=0),unit) for i in range(4)]
        states=[s['state'] for s in sides]
        count=states.count('centered'); structured=bool(c['structure']['distributed_stroke_pattern'])
        # A complete boundary remains a claim about region geometry, not KTP.
        complete=count==4 and g['status']=='in_frame'
        row={**c,'multiscale_sides':sides,'centered_side_count':count,
             'multiscale_boundary_supported':complete,
             'boundary_and_structure':complete and structured,
             'partial_boundary_and_structure':0<count<4 and structured,
             'frame_limited':g['status']=='near_frame' or 'frame_limited' in states,
             'group_id':None}
        out['candidates'].append(row);quads.append(q)
    for i,q in enumerate(quads):
        eligible=[g for g in out['groups'] if all(_distance(q,quads[j],diag)<=.008 for j in g['members'])]
        if eligible: group=eligible[0]
        else:
            group={'id':len(out['groups'])+1,'members':[]};out['groups'].append(group)
        group['members'].append(i);out['candidates'][i]['group_id']=group['id']
    # Compare structured regions even when their outer boundary is weak.
    # Containment is a risk/context flag, never a rule to prefer size.
    for i, row in enumerate(out['candidates']):
        row['structured_inner_groups']=[]
        row['structured_outer_groups']=[]
        q=quads[i].astype(np.float32); area=abs(cv2.contourArea(q))
        inner=set(); outer=set()
        for j, other in enumerate(out['candidates']):
            if i==j or row['group_id']==other['group_id'] or not other['structure']['distributed_stroke_pattern']:
                continue
            p=quads[j].astype(np.float32); a=abs(cv2.contourArea(p))
            if max(a,area)<1.3*min(a,area): continue
            intersection,_=cv2.intersectConvexConvex(q,p)
            if intersection>=.95*min(a,area):
                (inner if area>a else outer).add(other['group_id'])
        row['structured_inner_groups']=sorted(inner)
        row['structured_outer_groups']=sorted(outer)
    for group in out['groups']:
        rows=[out['candidates'][i] for i in group['members']]
        group['families']=sorted({r['family'] for r in rows})
        group['boundary_and_structure']=any(r['boundary_and_structure'] for r in rows)
        group['partial_boundary_and_structure']=any(r['partial_boundary_and_structure'] for r in rows)
        group['contains_structured_supported_region']=any(r['contains_structured_supported_region'] for r in rows)
        group['inside_supported_region']=any(r['inside_supported_region'] for r in rows)
        group['structured_inner_groups']=sorted({g for r in rows for g in r['structured_inner_groups']})
        group['structured_outer_groups']=sorted({g for r in rows for g in r['structured_outer_groups']})
    out['family_counts']={f:sum(c['family']==f for c in out['candidates']) for f in FAMILIES}
    out['competing_combined_geometry']=bool(parent.get('competing_combined_geometry',False))
    out['competing_evidence_groups']=sum(g['boundary_and_structure'] for g in out['groups'])>1
    out['status']='manual_review_only' if quads else 'no_hypothesis'
    return out
