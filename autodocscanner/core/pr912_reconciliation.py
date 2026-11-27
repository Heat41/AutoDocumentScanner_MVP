"""PR9.12: advisory region/structure/boundary reconciliation, no selection.

Inputs are existing hypotheses and PR9.11 local stroke groups, never labels.
All alternatives survive. Geometry agreement is not independent evidence.
"""
from __future__ import annotations
import cv2
import numpy as np


def _ordered(points):
    q=np.asarray(points,dtype=np.float32)
    if q.shape!=(4,2) or not np.isfinite(q).all():
        raise ValueError('invalid quadrilateral')
    d=q-q.mean(axis=0);q=q[np.argsort(np.arctan2(d[:,1],d[:,0]))]
    if not cv2.isContourConvex(q) or cv2.contourArea(q)<=0:
        raise ValueError('invalid quadrilateral')
    return q


def _patch(image,q):
    # Map to a padded normalized view; no replicated pixels beyond the image.
    lengths=np.linalg.norm(np.roll(q,-1,axis=0)-q,axis=1)
    if lengths[::2].mean()<lengths[1::2].mean():q=np.roll(q,1,axis=0)
    dst=np.float32([[48,48],[368,48],[368,248],[48,248]])
    t=cv2.getPerspectiveTransform(q,dst)
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB)
    patch=cv2.warpPerspective(lab,t,(417,297),flags=cv2.INTER_LINEAR)
    valid=cv2.warpPerspective(np.ones(image.shape[:2],np.uint8),t,(417,297),flags=cv2.INTER_LINEAR)>0
    valid=cv2.erode(valid.astype(np.uint8),np.ones((3,3),np.uint8),borderType=cv2.BORDER_CONSTANT,borderValue=0)>0
    return patch.astype(np.float32),valid,t


def _side(patch,valid,a,b,center):
    """Compare robust inner/outer bands at center and neighboring offsets.

    Separate regional change from narrow ink/edge response; mark displaced
    or ambiguous transitions rather than inventing a physical boundary.
    Four longitudinal bands, two widths, normalized coordinates.
    """
    direction=b-a;length=float(np.linalg.norm(direction));u=direction/length
    n=np.array([-u[1],u[0]])
    if np.dot(center-(a+b)/2,n)<0:n=-n
    tangential=np.linspace(.10,.90,48)
    positions=a+tangential[:,None]*direction
    offsets=np.arange(-18,19,3,dtype=float)
    results=[]
    for offset in offsets:
        votes=np.zeros(48,bool);seen=np.zeros(48,bool);strength=np.zeros(48,float)
        for width in (4,8):
            distances=np.arange(2,width+2,dtype=float)
            sides=[];available=[]
            for sign in (-1,1):
                coords=np.rint(positions[:,None,:]+(offset+sign*distances)[None,:,None]*n).astype(int)
                x,y=coords.transpose(2,0,1)
                ok=(x>=0)&(x<patch.shape[1])&(y>=0)&(y<patch.shape[0]);x=x.clip(0,patch.shape[1]-1);y=y.clip(0,patch.shape[0]-1)
                available.append((ok&valid[y,x]).all(axis=1));sides.append(patch[y,x])
            good=available[0]&available[1]
            med=[np.median(s,axis=1) for s in sides]
            noise=[np.median(np.linalg.norm(s-m[:,None,:],axis=2),axis=1) for s,m in zip(sides,med)]
            gap=np.linalg.norm(med[0]-med[1],axis=1)
            # A transition must exceed local within-band variation and a
            # modest absolute floor in OpenCV Lab units at both widths.
            response=gap/(4.+noise[0]+noise[1]);supported=good&(gap>=7)&(response>=1.4)
            if width==4:votes=supported;seen=good;strength=response
            else:votes&=supported;seen&=good;strength=np.minimum(strength,response)
        coverage=float(votes.mean());bands=sum(int(votes[i:i+12].sum()>=6) for i in range(0,48,12))
        results.append(dict(offset=float(offset),visible=float(seen.mean()),coverage=coverage,bands=bands,
                            strength=float(np.median(strength[votes])) if votes.any() else 0.))
    middle=results[len(results)//2];eligible=[r for r in results if r['coverage']>=.60 and r['bands']>=3]
    best=max(eligible,key=lambda r:(r['coverage'],r['strength'],-abs(r['offset']))) if eligible else None
    if middle['visible']<.75:state='frame_limited'
    elif best is None:state='unresolved'
    elif abs(best['offset'])>6 and (middle['coverage']<.60 or best['coverage']>=middle['coverage']+.20):state='displaced_transition'
    elif middle['coverage']>=.60 and middle['bands']>=3:state='centered_transition'
    else:state='ambiguous_transition'
    return dict(state=state,visible_fraction=middle['visible'],center_coverage=middle['coverage'],center_bands=middle['bands'],
                best_offset_normalized=None if best is None else best['offset'],best_coverage=None if best is None else best['coverage'])


def _seed_evidence(q,seeds,transform):
    inside=[];crossing=[];outside=[]
    occupied=np.zeros((201,321),np.uint8)
    tiles=set()
    for seed in seeds:
        p=np.asarray(seed['points'],np.float32)
        if p.ndim!=2 or p.shape[1]!=2 or not len(p) or not np.isfinite(p).all():raise ValueError('invalid seed')
        frac=float(np.mean([cv2.pointPolygonTest(q,tuple(pt),False)>=0 for pt in p]))
        if frac>=.90:
            inside.append(seed['seed_id'])
            # Union the mapped seed hulls, so overlapping groups do not
            # artificially multiply the spatial footprint. Not KTP coverage.
            mapped=cv2.perspectiveTransform(p[None],transform)[0]-[48,48]
            finite=np.isfinite(mapped).all(axis=1)
            mapped=mapped[finite]
            if len(mapped)>=3:
                hull=cv2.convexHull(np.rint(mapped.clip(-10000,10000)).astype(np.int32))
                cv2.fillConvexPoly(occupied,hull,1)
            for x,y in mapped:
                if 0<=x<=320 and 0<=y<=200:tiles.add((min(3,int(x/80)),min(3,int(y/50))))
        elif frac>=.15:crossing.append(seed['seed_id'])
        else:outside.append(seed['seed_id'])
    return dict(enclosed_seed_ids=inside,crossing_seed_ids=crossing,outside_seed_ids=outside,
                enclosed_seed_hull_union_fraction=float(occupied.mean()),enclosed_seed_tiles=len(tiles))


def inspect_candidate(image,corners,seeds,structure):
    q=_ordered(corners);patch,valid,transform=_patch(image,q)
    # Side numbering is normalized view order, not original-image corner order.
    rect=np.float32([[48,48],[368,48],[368,248],[48,248]])
    sides=[_side(patch,valid,rect[i],rect[(i+1)%4],rect.mean(axis=0)) for i in range(4)]
    seed=_seed_evidence(q,seeds,transform);centered=sum(s['state']=='centered_transition' for s in sides)
    pattern=bool(structure['distributed_stroke_pattern'])
    # Deliberately separate each cue; crossing is advisory and may be texture.
    return dict(sides=sides,centered_sides=centered,structure_supported=pattern,**seed,
                surface_boundary_supported=centered==4,
                region_structure_supported=pattern and bool(seed['enclosed_seed_ids']),
                partial_joint_supported=centered>=2 and pattern and bool(seed['enclosed_seed_ids']),
                joint_supported=centered==4 and pattern and bool(seed['enclosed_seed_ids']),
                joint_without_crossing=centered==4 and pattern and bool(seed['enclosed_seed_ids']) and not seed['crossing_seed_ids'])


def _distance(a,b,diag):
    return min(float(np.linalg.norm(a-np.roll(b,i,axis=0),axis=1).mean()/diag) for i in range(4))


def reconcile(image,candidates,seeds):
    """Preserve candidates; complete-link near-duplicate groups and conflicts.

    A group never chains via transitive near-duplicates. Nested hypotheses
    remain distinct. No representative is preferred and no auto selection.
    """
    if not isinstance(image,np.ndarray) or image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8 or min(image.shape[:2])<32:
        raise ValueError('invalid image')
    diag=float(np.hypot(*image.shape[:2]));rows=[];quads=[];groups=[]
    lab_image=image  # no mutation or writes
    for c in candidates:
        q=_ordered(c['corners']);i=len(rows);quads.append(q)
        evidence=inspect_candidate(lab_image,q,seeds,c['structure'])
        group=next((g for g in groups if all(_distance(q,quads[j],diag)<=.006 for j in g['members'])),None)
        if group is None:
            group=dict(group_id=len(groups)+1,members=[],families=[]);groups.append(group)
        group['members'].append(i)
        if c['family'] not in group['families']:group['families'].append(c['family'])
        rows.append(dict(candidate_id=i+1,family=c['family'],group_id=group['group_id'],**evidence))
    relations=[];nested=[]
    # Compare every member pair, not a selected representative.
    for i,a in enumerate(quads):
        aa=float(cv2.contourArea(a))
        for j in range(i+1,len(quads)):
            if rows[i]['group_id']==rows[j]['group_id']:continue
            b=quads[j];bb=float(cv2.contourArea(b));overlap=max(0.,float(cv2.intersectConvexConvex(a,b)[0]));small=min(aa,bb);large=max(aa,bb)
            if overlap>=.90*small and large>=1.18*small:
                kind='nested';inner=i+1 if aa<bb else j+1
            elif overlap/max(1.,aa+bb-overlap)>=.35:kind='overlapping_alternatives';inner=None
            elif overlap<=.05*small:kind='separate_alternatives';inner=None
            else:kind='partial_overlap';inner=None
            relation=dict(a=i+1,b=j+1,kind=kind,inner_candidate_id=inner)
            if kind=='nested':nested.append(relation)
            if rows[i]['joint_supported'] and rows[j]['joint_supported']:relations.append(relation)
    return dict(candidates=rows,groups=groups,joint_conflicts=relations,nested_relations=nested,auto_select=None,auto_crop_allowed=False,requires_manual_confirmation=True)
