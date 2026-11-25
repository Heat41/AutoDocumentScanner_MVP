"""PR9.2 bounded, annotation-free boundary translation hypotheses.

A persistent Lab transition is evidence of a region boundary, NOT proof of a
physical document. A solid nested rectangle may still satisfy this test.
No candidate selection, crop, file writes, or V2 dependency. Experimental.
"""
from __future__ import annotations

import cv2
import numpy as np
from autodocscanner.core.quad_validation import validate_quad


def inspect_geometry(corners, shape):
    """Expose invalid vs out-of-frame causes without clipping coordinates."""
    try:
        p = np.asarray(corners, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        p = np.empty((0, 2))
    if p.shape != (4, 2) or not np.isfinite(p).all():
        return {'status': 'invalid_points', 'reasons': ['invalid_points'],
                'outside_vertex_count': None, 'outside_excess_fraction': None}, None
    h, w = shape[:2]
    excess = np.maximum.reduce([np.zeros(4), -p[:, 0], -p[:, 1],
                                p[:, 0]-(w-1), p[:, 1]-(h-1)])
    # Reject huge coordinates before float32/OpenCV conversion.
    if np.abs(p).max() > max(h, w)*4:
        return {'status': 'invalid_geometry', 'reasons': ['outside_image'],
                'outside_vertex_count': int(np.count_nonzero(excess)),
                'outside_excess_fraction': float(excess.max()/np.hypot(h,w))}, None
    valid, reasons, metrics = validate_quad(p, shape)
    outside = int(np.count_nonzero(excess))
    status = ('invalid_geometry' if not valid else 'off_photo' if outside else
              'near_frame' if metrics.get('near_frame_vertex_count') else 'in_frame')
    if outside and 'outside_image' not in reasons:
        reasons = list(reasons) + ['outside_image_pixels']
    center = p.mean(axis=0)
    p = p[np.argsort(np.arctan2(p[:,1]-center[1], p[:,0]-center[0]))]
    return {'status': status, 'reasons': reasons,
            'outside_vertex_count': outside,
            'outside_excess_fraction': float(excess.max()/np.hypot(h,w))}, p if valid else None


def _side(lab, a, b, center, radius, band):
    tangent = (b-a)/np.linalg.norm(b-a)
    normal = np.array([-tangent[1], tangent[0]])
    if normal @ (center-(a+b)/2) < 0:
        normal = -normal
    offsets = np.linspace(-radius, radius, 25)
    positions = a + np.linspace(.15,.85,40)[:,None]*(b-a)
    # A near transition must persist across two wider scales. Thin printed
    # strokes often have opposing transitions and matching far neighborhoods.
    near = max(.8, .32*band)
    distances = np.array([-2.5*band,-band,-near,near,band,2.5*band])
    xy = positions[:,None,None,:]+(offsets[None,:,None,None]+distances[None,None,:,None])*normal
    h,w = lab.shape[:2]
    valid = ((xy[...,0]>=1)&(xy[...,0]<=w-2)&(xy[...,1]>=1)&(xy[...,1]<=h-2)).all(axis=2)
    pixels = cv2.remap(lab, xy[...,0].reshape(40,-1).astype(np.float32),
                       xy[...,1].reshape(40,-1).astype(np.float32), cv2.INTER_LINEAR).reshape(40,25,6,3).astype(float)
    close = pixels[:,:,3]-pixels[:,:,2]
    mid = pixels[:,:,4]-pixels[:,:,1]
    far = pixels[:,:,5]-pixels[:,:,0]
    nc,nm,nf = (np.linalg.norm(v,axis=2) for v in (close,mid,far))
    coherent = ((close*mid).sum(axis=2)>=.6*nc*nm)&((mid*far).sum(axis=2)>=.6*nm*nf)
    # Noise floor in OpenCV Lab units; provisional, not calibrated.
    support = valid & coherent & (nc>=5) & (nm>=10) & (nf>=10) & (nf>=.55*nm)
    # Continuity in at least three of four longitudinal bands, not just
    # one strong isolated edge fragment.
    coverage = support.mean(axis=0)
    bands = (support.reshape(4,10,25).mean(axis=1)>=.6).sum(axis=0)
    strengths = np.where(support, np.minimum(nc,np.minimum(nm,nf)),0.)
    scores = strengths.mean(axis=0)
    eligible = (coverage>=.70)&(bands>=3)
    scores[~eligible] = 0
    peak = float(scores.max())
    if peak<=0:
        return {'state':'insufficient_persistent_transition','offset_px':None,
                'coverage':float(coverage.max()), 'center_coverage':float(coverage[12]),
                'supported_bands':int(bands.max())}, normal
    strong = np.flatnonzero(scores>=.85*peak)
    groups = np.split(strong,np.where(np.diff(strong)>1)[0]+1)
    if len(groups)!=1 or offsets[strong[-1]]-offsets[strong[0]]>max(3.,.50*radius):
        return {'state':'ambiguous_transition','offset_px':None,
                'coverage':float(coverage.max()), 'center_coverage':float(coverage[12]),
                'supported_bands':int(bands.max())}, normal
    best = int(strong[np.argmax(scores[strong])])
    # Weighted center of a single narrow peak avoids arbitrary argmax tie.
    delta = float(np.average(offsets[strong],weights=scores[strong]))
    state = 'centered' if abs(delta)<=max(1.2,.14*radius) else 'displaced'
    if abs(delta)>=.92*radius:
        state = 'search_limit'
    return {'state':state,'offset_px':delta,'coverage':float(coverage[best]),
            'center_coverage':float(coverage[12]),'supported_bands':int(bands[best])}, normal


def _measure(lab,p):
    h,w=lab.shape[:2]
    shortest=float(np.linalg.norm(np.roll(p,-1,axis=0)-p,axis=1).min())
    radius=max(3.,min(.035*min(h,w),.12*shortest))
    band=max(1.5,min(.009*min(h,w),.035*shortest))
    return [_side(lab,p[i],p[(i+1)%4],p.mean(axis=0),radius,band) for i in range(4)]


def _intersect(a,u,b,v):
    matrix=np.column_stack((u,-v))
    if abs(np.linalg.det(matrix))<.08*np.linalg.norm(u)*np.linalg.norm(v):
        return None
    return a+u*np.linalg.solve(matrix,b-a)[0]


def recover_boundary(image,corners):
    """Return diagnostics and at most one bounded hypothesis, never a crop.

    Hypotheses require all sides to show persistent, unambiguous transitions,
    a valid in-frame joined quad, and fresh centered evidence after joining.
    Off-frame/near-frame candidates are diagnosed but not recovered here.
    """
    out={'geometry':None,'sides':[],'recovered_sides':[], 'hypothesis':None,
         'status':'invalid_image','auto_select':None,'auto_crop_allowed':False}
    if (not isinstance(image,np.ndarray) or image.dtype!=np.uint8 or image.ndim!=3
            or image.shape[2]!=3 or min(image.shape[:2])<64):
        return out
    geometry,p=inspect_geometry(corners,image.shape)
    out['geometry']=geometry
    if geometry['status']!='in_frame':
        out['status']=geometry['status']
        return out
    lab=cv2.GaussianBlur(cv2.cvtColor(image,cv2.COLOR_BGR2LAB),(3,3),0)
    evidence=_measure(lab,p)
    out['sides']=[s for s,_ in evidence]
    states=[s['state'] for s,_ in evidence]
    if all(s=='centered' for s in states):
        out['status']='persistent_centered_transitions'
        return out
    if not all(s in ('centered','displaced') for s in states):
        out['status']='insufficient_or_ambiguous_transitions'
        return out
    shifted=[p[i]+n*s['offset_px'] for i,(s,n) in enumerate(evidence)]
    vectors=np.roll(p,-1,axis=0)-p
    vertices=[_intersect(shifted[i-1],vectors[i-1],shifted[i],vectors[i]) for i in range(4)]
    if any(x is None for x in vertices):
        out['status']='intersection_rejected'; return out
    candidate=np.asarray(vertices)
    checked,_=inspect_geometry(candidate,image.shape)
    area=lambda q:abs(float(cv2.contourArea(q.astype(np.float32))))
    if (checked['status']!='in_frame' or not .80<=area(candidate)/area(p)<=1.25 or
            np.linalg.norm(candidate-p,axis=1).max()>.045*np.hypot(*image.shape[:2])):
        out['status']='recovery_geometry_rejected'; return out
    after=_measure(lab,candidate)
    out['recovered_sides']=[s for s,_ in after]
    if not all(s['state']=='centered' for s,_ in after):
        out['status']='recovery_remeasurement_rejected'; return out
    out['hypothesis']=candidate.tolist()
    out['status']='hypothesis_for_manual_review'
    return out
