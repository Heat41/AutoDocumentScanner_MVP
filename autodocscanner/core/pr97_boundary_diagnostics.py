"""PR9.7 retrospective sampling diagnostics. No new detector or crop rule.

The sampling coordinates below mirror the zero-offset grids of PR9.5/9.6.
Actual method states come from the unmodified methods, not these summaries.
All output measurements are anonymous; drawing is explicit and local only.
"""
from __future__ import annotations
import cv2
import numpy as np
from autodocscanner.core.pr95_region_evidence import _side
from autodocscanner.core.pr96_region_verification import _region_side, prepare_lab
from autodocscanner.core.pr92_boundary_recovery import inspect_geometry


def sampling_unit(shape, q):
    shortest = float(np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1).min())
    return max(1.5,min(.007*min(shape[:2]),.025*shortest))


def center_grids(a,b,center,unit):
    """Zero-offset sampling points exactly as in existing side methods."""
    a,b=np.asarray(a,float),np.asarray(b,float)
    t=(b-a)/np.linalg.norm(b-a); n=np.array([-t[1],t[0]])
    if n @ (center-(a+b)/2)<0: n=-n
    p=a+np.linspace(.12,.88,48)[:,None]*(b-a)
    r=a+np.linspace(.10,.90,96).reshape(12,8,1)*(b-a)
    grids={}
    for scale in (.6,1.2,2.4):
        depths=unit*scale*np.array([-1.6,-1.2,-.8,.8,1.2,1.6])
        grids[f'pr95_{scale:g}']=p[:,None,:]+depths[None,:,None]*n
    for scale in (1.,1.8):
        depths=unit*scale*(np.array([-2.25,-.75,.75,2.25])[:,None]+np.linspace(-.58,.58,5))
        grids[f'pr96_{scale:g}']=r[:,:,None,None,:]+depths[None,None,:,:,None]*n
    return grids


def _grid_summary(lab,xy):
    h,w=lab.shape[:2]
    valid=(xy[...,0]>=0)&(xy[...,0]<=w-1)&(xy[...,1]>=0)&(xy[...,1]<=h-1)
    block=valid.reshape(len(xy),-1).all(axis=1)
    flat=xy.reshape(len(xy),-1,2)
    pix=cv2.remap(lab,flat[...,0].astype(np.float32),flat[...,1].astype(np.float32),cv2.INTER_LINEAR).astype(float)
    pix=pix.reshape(xy.shape[:-1]+(3,))
    if xy.ndim==3:
        outer=np.median(pix[:,:3],axis=1);inner=np.median(pix[:,3:],axis=1)
    else:
        med=np.median(pix,axis=(1,3));outer=med[:,1];inner=med[:,2]
    contrast=np.linalg.norm(inner-outer,axis=-1)
    return {'visible_sample_fraction':float(valid.mean()),'fully_visible_block_fraction':float(block.mean()),
            'median_color_step':float(np.median(contrast[block])) if block.any() else None}


def preview_sampling_limits(shape, width=960, height=680, header=54):
    """Default annotation preview: theoretical sampling limits, not measured click error."""
    h,w=shape[:2];out=[]
    for turns in (0,1):
        rw,rh=(h,w) if turns else (w,h)
        scale=min(width/rw,(height-header)/rh)
        dw,dh=max(1,round(rw*scale)),max(1,round(rh*scale))
        sx,sy=rw/dw,rh/dh
        out.append({"quarter_turn":turns,"source_pixels_per_preview_pixel_max":max(sx,sy),
                    "pixel_center_convention_delta_max_px":.5*max(abs(sx-1),abs(sy-1))})
    return out


def inspect_quad(image,q):
    """Evaluate actual side states and independent center-grid visibility."""
    geom,p=inspect_geometry(q,image.shape)
    if geom['status'] not in ('in_frame','near_frame'):
        return {'geometry':geom['status'],'sides':[]}
    # Preserve supplied cyclic order: the rotation/resize audit compares the
    # same physical side, not a new arbitrary starting vertex.
    q=np.asarray(q,float)
    if not cv2.isContourConvex(q.astype(np.float32)): q=p
    lab=prepare_lab(image);unit=sampling_unit(image.shape,q)
    sides=[]
    for i in range(4):
        a,b=q[i],q[(i+1)%4];center=q.mean(axis=0)
        old=_side(lab,a,b,center,unit);new=_region_side(lab,a,b,center,unit)
        grids=center_grids(a,b,center,unit)
        sides.append({'side':i+1,'length_over_diagonal':float(np.linalg.norm(b-a)/np.hypot(*image.shape[:2])),
                      'pr95':old,'pr96':new,'center_grids':{k:_grid_summary(lab,g) for k,g in grids.items()}})
    return {'geometry':geom['status'],'unit_px':unit,'sides':sides}


def resized_points(q,old_shape,new_shape):
    """OpenCV pixel-center resize mapping (no clipping)."""
    scale=np.array([new_shape[1]/old_shape[1],new_shape[0]/old_shape[0]])
    return (np.asarray(q,float)+.5)*scale-.5


def rotated_points(q,shape):
    q=np.asarray(q,float)
    return np.column_stack((shape[0]-1-q[:,1],q[:,0]))


def _compare(base,variant):
    if len(variant['sides'])!=4:return None
    return {m:sum(a[m]['state']!=b[m]['state'] for a,b in zip(base['sides'],variant['sides'])) for m in ('pr95','pr96')}


def audit_transforms(image,q,base):
    """Fixed perturbations for diagnosis only; do not choose best variant."""
    if len(base['sides'])!=4:return []
    rot=cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)
    variants=[('rotate90',rot,rotated_points(q,image.shape))]
    h,w=image.shape[:2]
    for factor in (.5,.25):
        size=(round(w*factor),round(h*factor))
        if min(size)<64:continue
        im=cv2.resize(image,size,interpolation=cv2.INTER_AREA)
        variants.append((f'resize_{factor:g}',im,resized_points(q,image.shape,im.shape)))
    out=[]
    for name,im,p in variants:
        result=inspect_quad(im,p)
        out.append({'transform':name,'state_changes':_compare(base,result),'audit':result})
    return out


def draw_sampling_overlay(image,q,audit):
    """Private local overlay: only center grids, no inferred physical edge."""
    canvas=image.copy();q=np.asarray(q,float)
    if len(audit['sides'])!=4:return canvas
    radius=max(1,round(min(image.shape[:2])/500))
    for i,s in enumerate(audit['sides']):
        a,b=q[i],q[(i+1)%4]
        grids=center_grids(a,b,q.mean(axis=0),audit['unit_px'])
        # Cyan/yellow indicate PR9.5/9.6 sampling locations, not confidence.
        for name,color in [('pr95_2.4',(255,255,0)),('pr96_1.8',(0,210,255))]:
            for x,y in grids[name].reshape(-1,2):
                if 0<=x<image.shape[1] and 0<=y<image.shape[0]:
                    cv2.circle(canvas,(round(x),round(y)),radius,color,-1)
        cv2.line(canvas,tuple(np.rint(a).astype(int)),tuple(np.rint(b).astype(int)),(255,0,255),max(1,radius))
        mid=np.rint((a+b)/2).astype(int)
        cv2.putText(canvas,str(i+1),tuple(mid),cv2.FONT_HERSHEY_SIMPLEX,max(.5,radius*.5),(255,255,255),max(1,radius))
    return canvas
