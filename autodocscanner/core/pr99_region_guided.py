"""PR9.9 local seeded region segmentation, not KTP identity or selection.

No annotation, OCR, trained model, fixed color, or source modification.
Two appearance views share pixels; agreement is diagnostic, not independent proof.
"""
from __future__ import annotations
import cv2
import numpy as np
from autodocscanner.core.pr92_boundary_recovery import inspect_geometry, _intersect

W,H,P=320,200,48
SIZE=(W+2*P,H+2*P)
TARGET=np.float32([[P,P],[P+W-1,P],[P+W-1,P+H-1],[P,P+H-1]])


def _patch(image,q):
    lengths=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
    if lengths[::2].mean()<lengths[1::2].mean():q=np.roll(q,1,axis=0)
    t=cv2.getPerspectiveTransform(q.astype(np.float32),TARGET)
    # Mask invalid samples; never interpret padding as background.
    bgr=cv2.warpPerspective(image,t,SIZE)
    valid=cv2.warpPerspective(np.ones(image.shape[:2],np.float32),t,SIZE)>=.999
    valid=cv2.erode(valid.astype(np.uint8),np.ones((13,13),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    return bgr,valid,np.linalg.inv(t),q


def _features(bgr):
    lab=cv2.cvtColor(bgr,cv2.COLOR_BGR2LAB)
    smooth=cv2.GaussianBlur(lab,(7,7),0)
    g=lab[:,:,0].astype(np.float32)
    texture=cv2.GaussianBlur(abs(g-cv2.GaussianBlur(g,(7,7),0)),(7,7),0)
    return np.dstack((smooth,np.clip(32*np.log1p(texture),0,255).astype(np.uint8)))


def _segment(features,valid,channels):
    yy,xx=np.indices(valid.shape)
    inner=(xx>=P+30)&(xx<P+W-30)&(yy>=P+24)&(yy<P+H-24)&valid
    outer=((xx<P-18)|(xx>=P+W+18)|(yy<P-18)|(yy>=P+H+18))&valid
    out={'status':'insufficient_seeds','inner_seed_fraction':None,'outer_seed_fraction':None,'core_retained_fraction':None}
    if inner.sum()<500 or outer.sum()<500:return out,None
    # Smoothed empirical feature distributions. These are relative appearance
    # cues, not a learned classifier or calibrated posterior probabilities.
    score=np.zeros(valid.shape,float)
    for k in channels:
        bins=features[:,:,k]//8
        def hist(mask):
            counts=np.bincount(bins[mask],minlength=32).astype(float)
            counts=np.convolve(np.pad(counts,(1,1),mode='edge'),[.25,.5,.25],mode='valid')+1
            return counts/counts.sum()
        a,b=hist(inner),hist(outer)
        score+=np.clip(np.log(a[bins]/b[bins]),-3,3)
    mask=(score>0)&valid
    fi,fo=float(mask[inner].mean()),float(mask[outer].mean())
    out.update(inner_seed_fraction=fi,outer_seed_fraction=fo,status='appearance_not_separated')
    if fi<.55 or fo>.45 or fi-fo<.25:return out,None
    binary=cv2.morphologyEx(mask.astype(np.uint8),cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))
    binary=cv2.morphologyEx(binary,cv2.MORPH_OPEN,np.ones((3,3),np.uint8));binary[~valid]=0
    n,labels,_,_=cv2.connectedComponentsWithStats(binary,8)
    overlap=np.bincount(labels[inner],minlength=n);overlap[0]=0
    if n<2:return out,None
    order=np.argsort(overlap)[::-1];best=order[0]
    if overlap[best]<.45*inner.sum() or (n>2 and overlap[order[1]]>.65*overlap[best]):
        out['status']='fragmented_or_competing_region';return out,None
    component=(labels==best).astype(np.uint8)
    contours,_=cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    filled=np.zeros_like(component);cv2.drawContours(filled,contours,-1,1,cv2.FILLED);filled[~valid]=0
    out.update(status='region_segmented',core_retained_fraction=float(filled[inner].mean()))
    return out,filled.astype(bool)


def _side(mask,valid,a,b):
    # Fit region occupancy transitions, not strongest gradient. A transition
    # must have observed exterior AND interior pixels; padding is never proof.
    tangent=b-a;normal=np.array([-tangent[1],tangent[0]],float);normal/=np.linalg.norm(normal)
    ts=np.linspace(.12,.88,32);offsets=np.arange(-36,37)
    xy=a+ts[:,None,None]*tangent+offsets[None,:,None]*normal
    x,y=np.rint(xy).astype(int).transpose(2,0,1)
    m,v=mask[y,x],valid[y,x];hits=[]
    for i in range(32):
        choices=[]
        for j in range(4,len(offsets)-5):
            if v[i,j-4:j+6].all() and m[i,j:j+6].mean()>=.83 and m[i,j-4:j].mean()<=.25 and m[i,j] and not m[i,j-1]:
                choices.append(offsets[j]-.5)
        # Multiple disjoint transitions do not establish a unique local edge.
        if len(choices)==1:hits.append((i,choices[0]))
    out={'state':'insufficient_region_transition','coverage':len(hits)/32,'supported_bands':0,'residual_px':None}
    if len(hits)<22:return out,None
    idx,z=np.array(hits).T;idx=idx.astype(int);t=ts[idx]
    # Deterministic robust line fit, followed by inlier distribution check.
    line=cv2.fitLine(np.column_stack((t*100,z)).astype(np.float32),cv2.DIST_HUBER,0,.01,.01).ravel()
    vx,vy,x0,y0=map(float,line)
    if abs(vx)<.01:return out,None
    slope=100*vy/vx;intercept=y0-vy/vx*x0
    res=abs(z-(intercept+slope*t));keep=res<=3.5
    bands=len(set(idx[keep]//8));out.update(coverage=float(keep.sum()/32),supported_bands=bands,residual_px=float(np.median(res)))
    if keep.sum()<22 or bands<3:return out,None
    slope,intercept=np.polyfit(t[keep],z[keep],1);ends=np.array([intercept,intercept+slope])
    if abs(ends).max()>34:out['state']='search_limit';return out,None
    out['state']='localized_region_transition'
    start=a+normal*ends[0];return out,(start,b+normal*ends[1]-start)


def _localize(mask,valid):
    rows=[_side(mask,valid,TARGET[i],TARGET[(i+1)%4]) for i in range(4)]
    sides=[r for r,_ in rows];lines=[l for _,l in rows]
    if any(l is None for l in lines):return sides,None
    points=[_intersect(*lines[i-1],*lines[i]) for i in range(4)]
    if any(p is None for p in points):return sides,None
    return sides,np.asarray(points)


def inspect_region(image,corners):
    out={'status':'invalid_image','views':[],'hypotheses':[],'auto_select':None,'auto_crop_allowed':False}
    if not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3 or image.shape[2]!=3 or min(image.shape[:2])<64:return out
    g,q=inspect_geometry(corners,image.shape)
    if g['status'] not in ('in_frame','near_frame'):out['status']=g['status'];return out
    patch,valid,inv,q=_patch(image,q);features=_features(patch)
    for name,channels in [('color',(0,1,2)),('color_texture',(0,1,2,3))]:
        row,mask=_segment(features,valid,channels);row.update(view=name,sides=[]);out['views'].append(row)
        if mask is None:continue
        sides,points=_localize(mask,valid);row['sides']=sides
        if points is None:row['status']='incomplete_region_boundary';continue
        p=cv2.perspectiveTransform(points.reshape(1,4,2).astype(np.float64),inv).reshape(4,2)
        geom,_=inspect_geometry(p,image.shape)
        area=lambda x:abs(cv2.contourArea(x.astype(np.float32)))
        if geom['status'] not in ('in_frame','near_frame') or not .65<=area(p)/area(q)<=1.5 or np.linalg.norm(p-q,axis=1).max()>.075*np.hypot(*image.shape[:2]):
            row['status']='geometry_rejected';continue
        # Compare fitted polygon against observed connected region, including
        # exterior leakage. No claims for unobserved pixels.
        poly=np.zeros(valid.shape,np.uint8);cv2.fillConvexPoly(poly,np.rint(points).astype(np.int32),1)
        inside=(poly>0)&valid;union=(inside|mask)&valid;inter=inside&mask
        iou=float(inter.sum()/max(1,union.sum()));row['region_iou']=iou
        if iou<.72:row['status']='region_shape_inconsistent';continue
        row['status']='region_hypothesis';out['hypotheses'].append({'view':name,'corners':p.tolist(),'region_iou':iou})
    out['status']='hypotheses_for_manual_review' if out['hypotheses'] else 'region_unresolved'
    return out


def audit_regions(image,*,discover=None,structure=None):
    if discover is None:
        from autodocscanner.core.pr98_adaptive_localization import audit_recovery as discover
    if structure is None:
        from autodocscanner.core.pr94_region_boundary import inspect_structure as structure
    parent=discover(image.copy() if isinstance(image,np.ndarray) else image)
    if parent.get('auto_select','missing') is not None or parent.get('auto_crop_allowed') is not False:raise ValueError('unsafe parent contract')
    rows=[]
    for c in parent['candidates']:
        region=inspect_region(image,c['corners'])
        for h in region['hypotheses']:
            h['structure']=structure(image,np.asarray(h['corners']))
        rows.append({**c,'region_guided':region})
    return {**parent,'candidates':rows}
