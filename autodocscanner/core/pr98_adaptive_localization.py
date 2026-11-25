"""PR9.8 shadow: adaptive local boundary tracks and bounded reconstruction.

No target coordinates, annotation, identity model, ranking or auto crop.
All original proposals remain available, even if a recovery is generated.
"""
from __future__ import annotations
import cv2
import numpy as np
from autodocscanner.core.pr92_boundary_recovery import inspect_geometry, _intersect
from autodocscanner.core.pr93_independent_boundary import _image_ok
from autodocscanner.core.pr94_region_boundary import inspect_structure
from autodocscanner.core.pr96_region_verification import verify_regions, prepare_lab


def _profiles(lab, points, normal, offsets, unit):
    """Use the widest available adjacent scale pair at each location.

    Visibility is decided before contrast; never choose a scale just because
    it produces a stronger response. Missing samples cannot be evidence.
    """
    h,w=lab.shape[:2]; scales=[]
    for width in (.5,1.,2.):
        depth=unit*width*np.array([-1.6,-1.2,-.8,.8,1.2,1.6])
        xy=points[:,None,None,:]+(offsets[None,:,None,None]+depth[None,None,:,None])*normal
        valid=((xy[...,0]>=0)&(xy[...,0]<=w-1)&(xy[...,1]>=0)&(xy[...,1]<=h-1)).all(axis=2)
        pix=cv2.remap(lab,xy[...,0].reshape(len(points),-1).astype(np.float32),
                      xy[...,1].reshape(len(points),-1).astype(np.float32),cv2.INTER_LINEAR).reshape(len(points),len(offsets),6,3).astype(float)
        left,right=np.median(pix[:,:,:3],axis=2),np.median(pix[:,:,3:],axis=2)
        delta=right-left; strength=np.linalg.norm(delta,axis=-1)
        noise=np.maximum(np.linalg.norm(pix[:,:,:3]-left[:,:,None,:],axis=-1).mean(axis=2),
                         np.linalg.norm(pix[:,:,3:]-right[:,:,None,:],axis=-1).mean(axis=2))
        support=valid&(strength>=np.maximum(4.,3.*noise+2.))
        scales.append((valid,support,delta,strength))
    score=np.zeros((len(points),len(offsets))); pair=np.zeros_like(score,dtype=np.uint8)
    for i in (0,1):
        v,s,d,n=scales[i]; vv,ss,dd,nn=scales[i+1]
        available=v&vv
        good=available&s&ss&((d*dd).sum(axis=-1)>=.7*n*nn)&(n>=.60*nn)&(nn>=.60*n)
        # Prefer the narrow-pair response for localization when it agrees.
        previous=score.copy()
        score[available]=np.where(good,np.where(previous>0,previous,np.minimum(n,nn)),0)[available]
        pair[available]=i+1
    return score,pair


def _track(lab,a,b,center,unit,radius):
    length=float(np.linalg.norm(b-a)); tangent=(b-a)/length
    normal=np.array([-tangent[1],tangent[0]])
    if normal@(center-(a+b)/2)<0:normal=-normal
    t=np.linspace(.12,.88,48); points=a+t[:,None]*(b-a)
    offsets=np.linspace(-radius,radius,81); step=offsets[1]-offsets[0]
    scores,pairs=_profiles(lab,points,normal,offsets,unit)
    result={'state':'weak_or_unresolved','visible_fraction':float((pairs[:,40]>0).mean()),
            'track_coverage':0.,'supported_bands':0,'offset_px':None,'end_offsets_px':None,
            'narrow_pair_fraction':None,'residual_px':None}
    # Find distinct local peaks per longitudinal position, not a single
    # parallel offset for the entire side. Keep up to four alternatives.
    peaks=np.full((48,4),np.inf)
    for i,row in enumerate(scores):
        active=np.flatnonzero(row>0)
        groups=np.split(active,np.where(np.diff(active)>1)[0]+1)
        groups=sorted((g for g in groups if len(g)),key=lambda g:float(row[g].max()),reverse=True)
        for j,g in enumerate(groups[:4]):
            strong=g[row[g]>=.85*row[g].max()]
            if offsets[strong[-1]]-offsets[strong[0]]>3*unit:continue
            peaks[i,j]=float(np.average(offsets[strong],weights=row[strong]))
    tol=max(1.25,unit*.65,step*1.2)
    # Bounded straight-line hypotheses in normal coordinates (offset + slope).
    max_slope=min(2*radius,.20*length)
    mids=np.linspace(-radius,radius,41); slopes=np.linspace(-max_slope,max_slope,41)
    m,s=np.meshgrid(mids,slopes); m=m.ravel();s=s.ravel()
    legal=(np.abs(m-s/2)<=radius)&(np.abs(m+s/2)<=radius)
    m,s=m[legal],s[legal]
    prediction=m[:,None]+s[:,None]*(t-.5)
    residual=np.min(np.abs(prediction[:,:,None]-peaks[None,:,:]),axis=2)
    hit=residual<=tol; count=hit.sum(axis=1)
    bands=(hit.reshape(-1,4,12).sum(axis=2)>=6).sum(axis=1)
    fitcost=np.minimum(residual,tol*2).sum(axis=1)
    eligible=(count>=32)&(bands>=3)
    if not eligible.any():
        result.update(track_coverage=float(count.max()/48),supported_bands=int(bands.max()))
        if result['visible_fraction']<.75:result['state']='frame_limited'
        return result,None
    order=np.lexsort((fitcost,-count)); best=next(i for i in order if eligible[i])
    # Comparable, geometrically distinct tracks are ambiguous, not a winner.
    alternative=eligible&(count>=count[best]-3)&((np.abs(prediction-prediction[best])>2*tol).mean(axis=1)>=.5)
    result.update(track_coverage=float(count[best]/48),supported_bands=int(bands[best]))
    if alternative.any():result['state']='ambiguous_tracks';return result,None
    mid,slope=float(m[best]),float(s[best])
    for _ in range(2):
        pred=mid+slope*(t-.5); dist=np.abs(pred[:,None]-peaks)
        idx=dist.argmin(axis=1); ok=dist[np.arange(48),idx]<=tol
        if ok.sum()<32 or (ok.reshape(4,12).sum(axis=1)>=6).sum()<3:return result,None
        mid,slope=np.linalg.lstsq(np.column_stack((np.ones(ok.sum()),t[ok]-.5)),peaks[np.arange(48),idx][ok],rcond=None)[0]
    pred=mid+slope*(t-.5); dist=np.abs(pred[:,None]-peaks);idx=dist.argmin(axis=1)
    nearest=dist[np.arange(48),idx];ok=nearest<=tol
    ends=np.array([mid-slope/2,mid+slope/2]);bands=int((ok.reshape(4,12).sum(axis=1)>=6).sum())
    if ok.sum()<32 or bands<3:return result,None
    chosen=np.rint((peaks[np.arange(48),idx][ok]+radius)/step).astype(int).clip(0,80)
    result.update(track_coverage=float(ok.mean()),supported_bands=bands,offset_px=float(mid),
                  end_offsets_px=ends.tolist(),residual_px=float(np.median(nearest[ok])),
                  narrow_pair_fraction=float((pairs[np.flatnonzero(ok),chosen]==1).mean()))
    if np.abs(ends).max()>=.90*radius or abs(slope)>.20*length:
        result['state']='search_limit';return result,None
    result['state']='centered' if np.abs(ends).max()<=tol else 'displaced'
    start=a+normal*ends[0]; vector=b+normal*ends[1]-start
    return result,(start,vector)


def inspect_tracks(lab,q):
    shortest=float(np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1).min())
    unit=max(1.5,min(.007*min(lab.shape[:2]),.025*shortest))
    radius=max(4.,min(.035*np.hypot(*lab.shape[:2]),.15*shortest))
    return [_track(lab,q[i],q[(i+1)%4],q.mean(axis=0),unit,radius) for i in range(4)]


def recover_tracks(image,corners,*,lab=None):
    out={'status':'invalid_image','sides':[],'recovered_sides':[],'hypothesis':None,
         'boundary_supported':False,'recovered_structure':None,'auto_select':None,'auto_crop_allowed':False}
    if not _image_ok(image):return out
    geom,q=inspect_geometry(corners,image.shape)
    if geom['status'] not in ('in_frame','near_frame'):out['status']=geom['status'];return out
    lab=prepare_lab(image) if lab is None else lab
    tracks=inspect_tracks(lab,q);out['sides']=[s for s,_ in tracks]
    if all(s['state']=='centered' for s,_ in tracks):
        out.update(status='centered_tracks',boundary_supported=True);return out
    if any(line is None for _,line in tracks):out['status']='incomplete_or_ambiguous_tracks';return out
    lines=[line for _,line in tracks]
    vertices=[_intersect(*lines[i-1],*lines[i]) for i in range(4)]
    if any(v is None for v in vertices):out['status']='intersection_rejected';return out
    p=np.asarray(vertices);g,_=inspect_geometry(p,image.shape)
    area=lambda x:abs(float(cv2.contourArea(x.astype(np.float32))))
    if (g['status'] not in ('in_frame','near_frame') or not .70<=area(p)/area(q)<=1.40 or
            np.linalg.norm(p-q,axis=1).max()>.045*np.hypot(*image.shape[:2])):
        out['status']='recovery_geometry_rejected';return out
    after=inspect_tracks(lab,p);out['recovered_sides']=[s for s,_ in after]
    if not all(s['state']=='centered' for s,_ in after):out['status']='remeasurement_rejected';return out
    out.update(status='recovered_hypothesis',hypothesis=p.tolist(),recovered_structure=inspect_structure(image,p))
    return out


def audit_recovery(image,*,discover=verify_regions):
    parent=discover(image.copy())
    if parent.get('auto_select','missing') is not None or parent.get('auto_crop_allowed') is not False:
        raise ValueError('unsafe parent contract')
    if not _image_ok(image):return parent
    lab=prepare_lab(image)
    return {**parent,'candidates':[{**c,'adaptive_recovery':recover_tracks(image,c['corners'],lab=lab)}
                                  for c in parent['candidates']]}
