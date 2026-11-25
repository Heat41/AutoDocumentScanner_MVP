"""PR9.6: regional change evidence, independent of PR9.5 pixel profiles.

Compare distributions in adjacent strips, with within-region controls and
multiple widths. Texture changes are evidence of a region transition, not
proof of a KTP. Preserve every parent hypothesis and never select or crop.
"""
from __future__ import annotations
import cv2
import numpy as np
from autodocscanner.core.pr95_region_evidence import audit_region_evidence
from autodocscanner.core.pr93_independent_boundary import _image_ok
from autodocscanner.core.pr92_boundary_recovery import inspect_geometry


def _region_side(lab, a, b, center, unit):
    """Compare four equal-width strips over 12 longitudinal blocks.

    Cross-boundary change must exceed both within-region changes, at two
    widths, with consistent direction. A gradient or isolated thin stroke
    should not count as a regional step. Missing samples are never padded
    into evidence. All offsets and widths scale with image/card geometry.
    """
    a,b=np.asarray(a,float),np.asarray(b,float)
    tangent=(b-a)/np.linalg.norm(b-a)
    normal=np.array([-tangent[1],tangent[0]])
    if normal @ (center-(a+b)/2)<0: normal=-normal
    points=a+np.linspace(.10,.90,96).reshape(12,8,1)*(b-a)
    offsets=np.linspace(-3.,3.,25)
    h,w=lab.shape[:2]
    results=[]
    for width in (1.,1.8):
        depths=unit*width*(np.array([-2.25,-.75,.75,2.25])[:,None]+np.linspace(-.58,.58,5))
        xy=points[:,None,:,None,None,:]+(offsets[None,:,None,None,None,None]*unit+depths[None,None,None,:,:,None])*normal
        valid=((xy[...,0]>=0)&(xy[...,0]<=w-1)&(xy[...,1]>=0)&(xy[...,1]<=h-1)).all(axis=(2,3,4))
        pix=cv2.remap(lab,xy[...,0].reshape(12,-1).astype(np.float32),xy[...,1].reshape(12,-1).astype(np.float32),cv2.INTER_LINEAR).reshape(12,25,8,4,5,3).astype(np.float32)
        # Robust central color plus dispersion and tangential texture.
        med=np.median(pix,axis=(2,4))
        spread=np.median(np.linalg.norm(pix-med[:,:,None,:,None,:],axis=-1),axis=(2,4))
        texture=np.mean(np.minimum(np.linalg.norm(np.diff(pix,axis=2),axis=-1),50.),axis=(2,4))
        # Color and texture are measured separately to expose their roles.
        features=np.concatenate((med,spread[...,None],texture[...,None]),axis=-1)
        steps=np.diff(features,axis=2)
        change=steps[:,:,1,:]
        strength=np.linalg.norm(change,axis=-1)
        control=np.maximum(np.linalg.norm(steps[:,:,0,:],axis=-1),np.linalg.norm(steps[:,:,2,:],axis=-1))
        support=valid&(strength>=4.)&(strength>=1.8*control+1.)
        results.append((valid,support,change,strength))
    v0,s0,d0,n0=results[0];v1,s1,d1,n1=results[1]
    valid=v0&v1
    support=valid&s0&s1&((d0*d1).sum(axis=-1)>=.65*n0*n1)&(n0>=.5*n1)&(n1>=.5*n0)
    coverage=support.mean(axis=0)
    bands=(support.reshape(4,3,25).sum(axis=1)>=2).sum(axis=0)
    score=np.where(support,np.minimum(n0,n1),0.).mean(axis=0)
    score[(coverage<2/3)|(bands<3)]=0.
    mid=len(offsets)//2
    out={'state':'weak_or_unresolved','visible_fraction':float(valid[:,mid].mean()),
         'center_support_fraction':float(coverage[mid]),'best_support_fraction':float(coverage.max()),
         'offset_fraction_of_unit':None,'offset_span_fraction_of_unit':None,
         'color_dominant_blocks':0,'texture_dominant_blocks':0,'supported_bands':int(bands.max())}
    if out['visible_fraction']<.75:
        out['state']='frame_limited';return out
    peak=float(score.max())
    if peak<=0:return out
    best=np.flatnonzero(score>=.85*peak)
    span=float(offsets[best[-1]]-offsets[best[0]])
    out['offset_span_fraction_of_unit']=span
    if np.any(np.diff(best)>1) or span>1.5:
        out['state']='ambiguous_or_diffuse';return out
    offset=float(np.average(offsets[best],weights=score[best]))
    out['offset_fraction_of_unit']=offset
    idx=int(np.argmax(score)); mask=support[:,idx]
    color=np.linalg.norm(d0[:,idx,:3],axis=-1)
    tex=np.linalg.norm(d0[:,idx,3:],axis=-1)
    out['color_dominant_blocks']=int((mask&(color>=tex)).sum())
    out['texture_dominant_blocks']=int((mask&(tex>color)).sum())
    out['state']='search_limit' if abs(offset)>=2.75 else 'centered' if abs(offset)<=.65 else 'displaced'
    return out


def inspect_region_boundary(lab, q):
    """Original-image coordinates; geometry must be validated by caller."""
    shortest=float(np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1).min())
    unit=max(1.5,min(.007*min(lab.shape[:2]),.025*shortest))
    return [_region_side(lab,q[i],q[(i+1)%4],q.mean(axis=0),unit) for i in range(4)]


def prepare_lab(image):
    return cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2LAB),(3,3),0)


def verify_regions(image, *, audit=audit_region_evidence):
    """Attach regional evidence to unchanged PR9.5 candidates and groups."""
    out={'status':'invalid_image','candidates':[],'groups':[], 'auto_select':None,
         'auto_crop_allowed':False,'requires_manual_confirmation':True}
    if not _image_ok(image):return out
    parent=audit(image.copy())
    if parent.get('auto_select','missing') is not None or parent.get('auto_crop_allowed') is not False:
        raise ValueError('unsafe evidence contract')
    lab=prepare_lab(image)
    rows=[]
    for c in parent['candidates']:
        geom,q=inspect_geometry(c['corners'],image.shape)
        if geom['status'] not in ('in_frame','near_frame'):raise ValueError('invalid parent geometry')
        sides=inspect_region_boundary(lab,q)
        count=sum(s['state']=='centered' for s in sides)
        complete=count==4 and geom['status']=='in_frame'
        structured=bool(c['structure']['distributed_stroke_pattern'])
        rows.append({**c,'region_sides':sides,'region_centered_sides':count,
                     'region_boundary_supported':complete,'region_and_structure':complete and structured,
                     'either_complete_with_structure':bool(c['boundary_and_structure'] or (complete and structured)),
                     'both_complete_with_structure':bool(c['boundary_and_structure'] and complete and structured)})
    # Neither union nor intersection is an acceptance rule. Do not pool sides
    # across different quads or count two methods as independent probabilities.
    return {**parent,'candidates':rows,'requires_manual_confirmation':True}
