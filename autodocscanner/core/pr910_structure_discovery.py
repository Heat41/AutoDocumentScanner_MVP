"""PR9.10: full-image structure seeds, surface expansion, boundary hypotheses.

Classical CV only. No pre-existing quadrilateral, target annotation, OCR,
fixed document layout, identity classification, or automatic selection.
Regular textures and other printed objects can still create seeds.
"""
from __future__ import annotations
import cv2
import numpy as np
from autodocscanner.core.pr92_boundary_recovery import inspect_geometry
from autodocscanner.core.pr93_independent_boundary import _image_ok, _distance

MAX_SEEDS = 12
MAX_PROPOSALS = 24
MAX_COMPONENTS = 1800


def _groups(gray):
    """Compact stroke components with local collinear neighbors, any angle.

    Keep separate scale/polarity views; skip overloaded views rather than
    silently treating the first scan-order components as the document.
    """
    h,w=gray.shape; m=min(h,w); groups=[]; views=[]
    for block in sorted({max(9,round(m*f)|1) for f in (.025,.055,.10)}):
        for polarity in (cv2.THRESH_BINARY_INV,cv2.THRESH_BINARY):
            mask=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       polarity,block,5 if polarity==cv2.THRESH_BINARY_INV else -5)
            n,_,stats,centers=cv2.connectedComponentsWithStats(mask,8)
            x,y,ww,hh,a=stats[1:].T
            size=np.maximum(ww,hh); short=np.minimum(ww,hh)
            good=(x>0)&(y>0)&(x+ww<w)&(y+hh<h)&(short>=2)&(size>=4)&(size<=.075*m)&(size<=block*1.25)&(short>=.18*size)&(a>=5)&(a<=.95*ww*hh)
            pts=centers[1:][good]; sizes=size[good].astype(float)
            view={'block':block,'polarity':'dark' if polarity==cv2.THRESH_BINARY_INV else 'light','components':len(pts),'groups':0,'status':'measured'};views.append(view)
            if len(pts)>MAX_COMPONENTS:view['status']='component_budget_exceeded';continue
            if len(pts)<12:continue
            delta=pts[None,:,:]-pts[:,None,:];dist=np.linalg.norm(delta,axis=2)
            ratio=np.minimum(sizes[:,None],sizes[None,:])/np.maximum(sizes[:,None],sizes[None,:])
            near=(dist>=.45*np.maximum(sizes[:,None],sizes[None,:]))&(dist<=3.8*np.minimum(sizes[:,None],sizes[None,:]))&(ratio>=.45)
            aligned=np.zeros(len(pts),bool)
            for i in range(len(pts)):
                ids=np.flatnonzero(near[i]);ids=ids[np.argsort(dist[i,ids])[:12]]
                if len(ids)<2:continue
                vectors=delta[i,ids]/dist[i,ids,None]
                aligned[i]=bool((vectors@vectors.T<-.88).any())
            # Include neighboring endpoints of aligned runs; group nearby rows
            # without imposing horizontal image coordinates or a KTP layout.
            active=aligned | (near[:,aligned].sum(axis=1)>=2)
            visited=np.zeros(len(pts),bool)
            for i in np.flatnonzero(active):
                if visited[i]:continue
                todo=[int(i)];visited[i]=True;ids=[]
                while todo:
                    j=todo.pop();ids.append(j)
                    neighbors=np.flatnonzero(near[j]&active&~visited)
                    visited[neighbors]=True;todo.extend(neighbors.tolist())
                if len(ids)<12 or aligned[ids].sum()<6:continue
                p=pts[ids];s=float(np.median(sizes[ids]));rect=cv2.minAreaRect(p.astype(np.float32));lo,hi=sorted(rect[1])
                if lo<2.5*s or hi<5*s:continue
                hull=cv2.convexHull(p.astype(np.float32));area=cv2.contourArea(hull)
                if area<8*s*s:continue
                groups.append({'points':p,'size':s,'hull':hull,'area':float(area),'aligned':int(aligned[ids].sum()),'view':len(views)-1});view['groups']+=1
    # Deduplicate nearly identical seed footprints across views. No ordering
    # based on image location or presumed card color. Size diversity budget.
    bins=[[],[],[]]
    for g in sorted(groups,key=lambda g:-g['aligned']):
        if any(cv2.intersectConvexConvex(g['hull'],a['hull'])[0]>=.8*min(g['area'],a['area']) and .6<=g['area']/a['area']<=1.67 for b in bins for a in b):continue
        f=g['area']/(h*w);bins[0 if f<.015 else 1 if f<.08 else 2].append(g)
    selected=[]
    while len(selected)<MAX_SEEDS and any(bins):
        for b in bins:
            if b and len(selected)<MAX_SEEDS:selected.append(b.pop(0))
    return selected,views,len(groups)


def _expand(lab,g):
    """Estimate surface from neighborhoods of stroke centers and grow it.

    Surface masks come from seed-derived Lab distances, not an expanded
    rectangle accepted as a boundary. A real observed component is required.
    """
    h,w=lab.shape[:2];p=g['points'];s=g['size'];lo=p.min(axis=0);hi=p.max(axis=0)
    pad=max(8*s,1.6*max(hi-lo));x0,y0=np.maximum(0,np.floor(lo-pad)).astype(int);x1,y1=np.minimum([w,h],np.ceil(hi+pad+1)).astype(int)
    local=lab[y0:y1,x0:x1];pts=p-[x0,y0];lh,lw=local.shape[:2]
    core=np.zeros((lh,lw),np.uint8)
    for x,y in np.rint(pts).astype(int):cv2.circle(core,(x,y),max(2,round(s*1.4)),1,-1)
    for x,y in np.rint(pts).astype(int):cv2.circle(core,(x,y),max(1,round(s*.45)),0,-1)
    # Smooth in proportion to stroke size to reduce influence of ink.
    k=min(31,max(3,round(s*1.2)|1));surface=cv2.GaussianBlur(local,(k,k),0).astype(np.float32)
    sample=surface[core>0]
    if len(sample)<20:return [],[{'status':'insufficient_surface_samples'}]
    center=np.median(sample,axis=0);d=np.linalg.norm(surface-center,axis=2)
    spread=float(np.percentile(d[core>0],75));base=max(6.,spread)
    rows=[];out=[];kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(min(31,max(3,round(s*.9)|1)),)*2)
    for factor in (1.25,2.,3.):
        threshold=min(60.,base*factor)
        mask=cv2.morphologyEx((d<=threshold).astype(np.uint8),cv2.MORPH_CLOSE,kernel)
        n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
        px,py=np.rint(pts).astype(int).T
        # The ink itself need not belong to the surface. Count a seed once
        # per component if its observed neighborhood reaches that component.
        angles=np.arange(8)*np.pi/4
        samples=np.rint(pts[:,None,:]+s*.85*np.column_stack((np.cos(angles),np.sin(angles)))[None,:,:]).astype(int)
        sx,sy=samples.transpose(2,0,1);valid=(sx>=0)&(sx<lw)&(sy>=0)&(sy<lh)
        memberships=np.where(valid,labels[sy.clip(0,lh-1),sx.clip(0,lw-1)],0)
        pair=np.unique(np.arange(len(p))[:,None]*n+memberships)
        counts=np.bincount(pair%n,minlength=n);counts[0]=0
        row={'status':'no_seed_component','surface_threshold_lab':threshold,'seed_fraction':0.};rows.append(row)
        if n<2:continue
        best=int(counts.argmax());fraction=float(counts[best]/len(p));row['seed_fraction']=fraction
        if fraction<.70:continue
        component=(labels==best).astype(np.uint8)
        # A crop/image border is not an observed physical boundary.
        if component[0].any() or component[-1].any() or component[:,0].any() or component[:,-1].any():row['status']='component_touches_search_or_image_border';continue
        contours,_=cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if not contours:continue
        contour=max(contours,key=cv2.contourArea);hull=cv2.convexHull(contour);area=cv2.contourArea(contour)
        if area<1.25*g['area']:row['status']='insufficient_surface_expansion';continue
        filled=np.zeros_like(component);cv2.drawContours(filled,[contour],-1,1,-1)
        row['status']='no_quadrilateral'
        for eps in (.012,.022,.035):
            q=cv2.approxPolyDP(hull,eps*cv2.arcLength(hull,True),True).reshape(-1,2)
            if len(q)!=4:continue
            poly=np.zeros_like(component);cv2.fillConvexPoly(poly,q,1)
            iou=float((poly&filled).sum()/max(1,(poly|filled).sum()))
            coverage=float(poly[py,px].mean())
            if iou<.78 or coverage<.90:continue
            row.update(status='region_hypothesis',region_iou=iou,enclosed_seed_fraction=coverage)
            out.append({'corners':q.astype(float)+[x0,y0],'region_iou':iou,'enclosed_seed_fraction':coverage,'surface_threshold_lab':threshold});break
    return out,rows


def discover_structure_regions(image,*,structure=None):
    out={'status':'invalid_image','seed_count':0,'raw_group_count':0,'views':[],'seeds':[],'candidates':[],
         'auto_select':None,'auto_crop_allowed':False,'requires_manual_confirmation':True}
    if not _image_ok(image):return out
    if structure is None:
        from autodocscanner.core.pr94_region_boundary import inspect_structure as structure
    h,w=image.shape[:2];diag=np.hypot(h,w);rows=[]
    # Two working resolutions help separate small strokes from dense texture.
    # Each view has its own bounded group search; neither uses previous corners.
    scales=[min(1.,1200/max(h,w))]
    if min(h,w)*scales[0]*.65>=75:scales.append(scales[0]*.65)
    for scale in scales:
        sw,sh=round(w*scale),round(h*scale)
        small=cv2.resize(image,(sw,sh),interpolation=cv2.INTER_AREA) if scale<1 else image
        gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY);lab=cv2.cvtColor(small,cv2.COLOR_BGR2LAB)
        groups,views,raw=_groups(gray);offset=len(out['views'])
        out['views'].extend({**v,'working_scale':scale} for v in views);out['raw_group_count']+=raw
        for g in groups:
            sid=len(out['seeds'])+1;proposals,attempts=_expand(lab,g)
            out['seeds'].append({'seed_id':sid,'stroke_count':len(g['points']),'aligned_count':g['aligned'],'area_fraction':g['area']/(sw*sh),'view_index':offset+g['view'],'attempts':attempts})
            for p in proposals:
                q=(p['corners']+.5)*[w/sw,h/sh]-.5;geom,q=inspect_geometry(q,image.shape)
                if geom['status'] not in ('in_frame','near_frame'):continue
                match=next((c for c in rows if _distance(q,np.asarray(c['corners']),diag)<.006),None)
                if match is not None:
                    if sid not in match['seed_ids']:match['seed_ids'].append(sid)
                    match['supporting_expansions']+=1;continue
                if len(rows)>=MAX_PROPOSALS:continue
                rows.append({**p,'corners':q.tolist(),'family':'structure_guided','geometry':geom['status'],'seed_ids':[sid],
                             'supporting_expansions':1,'structure':structure(image,q)})
    out['seed_count']=len(out['seeds']);out['candidates']=rows
    out['status']='hypotheses_for_manual_review' if rows else ('no_surface_hypothesis' if out['seeds'] else 'no_structure_seed')
    return out
