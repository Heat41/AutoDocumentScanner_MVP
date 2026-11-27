"""PR9.11: bounded local structure groups and competing surface hypotheses.

No annotation or preceding quad enters discovery. Local line alignment and
surface similarity are advisory, not KTP identity. Preserve alternatives.
"""
from __future__ import annotations
import cv2
import numpy as np
from autodocscanner.core.pr92_boundary_recovery import inspect_geometry
from autodocscanner.core.pr93_independent_boundary import _image_ok, _distance

MAX_TILE_COMPONENTS = 480
MAX_SEEDS = 16  # per working scale
MAX_PROPOSALS = 24


def _tiles(h, w):
    """Overlapping local windows. Limit graph size without skipping a view."""
    for size in sorted({max(48, round(min(h,w)*f)) for f in (.32,.58)}):
        step=max(1,size//2)
        for y in sorted(set(list(range(0,max(1,h-size+1),step))+[max(0,h-size)])):
            for x in sorted(set(list(range(0,max(1,w-size+1),step))+[max(0,w-size)])):
                yield x,y,min(w,x+size),min(h,y+size)


def _local_groups(points, sizes, colors):
    """Line coherence with local surface-compatible links, any orientation.

    Reject multi-directional lattice centers as seeds, not as proof of
    background. Bounded quadratic work only inside a local window.
    """
    if len(points)<12:return []
    delta=points[None,:,:]-points[:,None,:]
    dist=np.linalg.norm(delta,axis=2)
    ratio=np.minimum(sizes[:,None],sizes[None,:])/np.maximum(sizes[:,None],sizes[None,:])
    color_gap=np.linalg.norm(colors[None,:,:]-colors[:,None,:],axis=2)
    near=(dist>=.45*np.maximum(sizes[:,None],sizes[None,:]))&(dist<=3.8*np.minimum(sizes[:,None],sizes[None,:]))&(ratio>=.45)&(color_gap<=28)
    aligned=np.zeros(len(points),bool);ambiguous=np.zeros(len(points),bool)
    directions=np.zeros((len(points),2),float)
    for i in range(len(points)):
        ids=np.flatnonzero(near[i]);ids=ids[np.argsort(dist[i,ids],kind='stable')[:10]]
        if len(ids)<2:continue
        v=delta[i,ids]/dist[i,ids,None]
        a,b=np.where(np.triu(v@v.T<-.93,1))
        if not len(a):continue
        axes=v[a]; ref=axes[0]
        if np.any(np.abs(axes@ref)<.55):ambiguous[i]=True;continue
        aligned[i]=True;directions[i]=ref
    # Different local row orientations do not automatically form one group.
    compatible=np.abs(directions@directions.T)>=.80
    links=near&compatible&aligned[:,None]&aligned[None,:]
    # Link parallel nearby rows even where no glyph lies exactly between them.
    visited=np.zeros(len(points),bool);out=[]
    for i in np.flatnonzero(aligned):
        if visited[i]:continue
        todo=[int(i)];visited[i]=True;ids=[]
        while todo:
            j=todo.pop();ids.append(j)
            neighbors=np.flatnonzero(links[j]&~visited);visited[neighbors]=True;todo.extend(neighbors.tolist())
        # Include neighboring endpoints but not lattice centers.
        ids=np.unique(np.r_[ids,np.flatnonzero((near[:,ids].sum(axis=1)>=2)&~ambiguous)])
        if len(ids)<12 or aligned[ids].sum()<6:continue
        p=points[ids];s=float(np.median(sizes[ids]));hull=cv2.convexHull(p.astype(np.float32));area=float(cv2.contourArea(hull))
        if area<8*s*s:continue
        lo,hi=sorted(cv2.minAreaRect(hull)[1])
        if lo<2*s or hi<5*s:continue
        out.append(dict(points=p,size=s,hull=hull,area=area,aligned=int(aligned[ids].sum()),ambiguous_fraction=float(ambiguous.mean())))
    return out


def _groups(gray,lab):
    h,w=gray.shape;m=min(h,w);groups=[];views=[];tiles=list(_tiles(h,w))
    # Smooth the surface surrounding strokes; compare local colors, not a
    # fixed KTP color. Separate threshold/polarity views are correlated.
    for block in sorted({max(9,round(m*f)|1) for f in (.025,.055,.10)}):
        for polarity in (cv2.THRESH_BINARY_INV,cv2.THRESH_BINARY):
            mask=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,polarity,block,5 if polarity==cv2.THRESH_BINARY_INV else -5)
            _,_,stats,centers=cv2.connectedComponentsWithStats(mask,8)
            x,y,ww,hh,a=stats[1:].T;size=np.maximum(ww,hh);short=np.minimum(ww,hh)
            good=(x>0)&(y>0)&(x+ww<w)&(y+hh<h)&(short>=2)&(size>=4)&(size<=.075*m)&(size<=1.25*block)&(short>=.18*size)&(a>=5)&(a<=.95*ww*hh)
            p=centers[1:][good];s=size[good].astype(float)
            view=dict(block=block,polarity='dark' if polarity==cv2.THRESH_BINARY_INV else 'light',components=len(p),tiles=len(tiles),overloaded_tiles=0,measured_tiles=0,groups=0)
            views.append(view)
            if not len(p):continue
            blur=cv2.GaussianBlur(lab,(max(3,round(float(np.median(s))*2)|1),)*2,0)
            px,py=np.rint(p).astype(int).T;colors=blur[py,px].astype(float)
            for x0,y0,x1,y1 in tiles:
                take=(p[:,0]>=x0)&(p[:,0]<x1)&(p[:,1]>=y0)&(p[:,1]<y1)
                if take.sum()>MAX_TILE_COMPONENTS:view['overloaded_tiles']+=1;continue
                view['measured_tiles']+=1
                for g in _local_groups(p[take],s[take],colors[take]):
                    g['view']=len(views)-1;groups.append(g);view['groups']+=1
    # Prefer coherent groups, retain size diversity, do not merge through
    # transitive links between windows. Near-identical hulls are redundant.
    bins=[[],[],[]]
    for g in sorted(groups,key=lambda g:(g["ambiguous_fraction"],-g["aligned"])):
        f=g['area']/(h*w);bucket=bins[0 if f<.008 else 1 if f<.035 else 2]
        if len(bucket)>=2*MAX_SEEDS:continue
        if any(.55<=g['area']/a['area']<=1.82 and cv2.intersectConvexConvex(g['hull'],a['hull'])[0]>=.8*min(g['area'],a['area']) for b in bins for a in b):continue
        bucket.append(g)
        # Bound dedup cost in highly textured scenes.
        if all(len(b)>=MAX_SEEDS for b in bins):break
    selected=[]
    while len(selected)<MAX_SEEDS and any(bins):
        for b in bins:
            if b and len(selected)<MAX_SEEDS:selected.append(b.pop(0))
    return selected,views,len(groups)


def _expand(lab,g):
    """Grow from several local surface samples; no artificial ROI border.

    Full working image, seed-derived color distances, distinct quantile-based
    thresholds. A physical image-border contact is unresolved, never a side.
    """
    h,w=lab.shape[:2];p=g['points'];s=g['size'];px,py=np.rint(p).astype(int).T
    angles=np.arange(8)*np.pi/4
    samples=np.rint(p[:,None,:]+s*1.15*np.column_stack((np.cos(angles),np.sin(angles)))[None,:,:]).astype(int)
    sx,sy=samples.transpose(2,0,1);valid=(sx>=0)&(sx<w)&(sy>=0)&(sy<h)
    if valid.sum()<20:return [],[{'status':'insufficient_surface_samples'}]
    sx=sx.clip(0,w-1);sy=sy.clip(0,h-1)
    k=min(31,max(3,round(s*1.2)|1));surface=cv2.GaussianBlur(lab,(k,k),0).astype(np.float32)
    sample=surface[sy,sx][valid];center=np.median(sample,axis=0)
    out=[];rows=[]
    for mode,weights in [('lab',np.array([1.,1.,1.],np.float32)),('chroma_weighted',np.array([.45,1.,1.],np.float32))]:
        d=np.linalg.norm((surface-center)*weights,axis=2)
        ds=d[sy,sx][valid];base=float(np.percentile(ds,50))
        thresholds=sorted(set(round(float(np.clip(v,5,55)),2) for v in (base,float(np.percentile(ds,75)),float(np.percentile(ds,90))+4)))
        for threshold in thresholds:
            kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(min(17,max(3,round(s*.7)|1)),)*2)
            mask=cv2.morphologyEx((d<=threshold).astype(np.uint8),cv2.MORPH_CLOSE,kernel)
            n,labels,stats,_=cv2.connectedComponentsWithStats(mask,8)
            membership=np.where(valid,labels[sy,sx],0);pairs=np.unique(np.arange(len(p))[:,None]*n+membership)
            counts=np.bincount(pairs%n,minlength=n);counts[0]=0
            best=int(counts.argmax());fraction=float(counts[best]/len(p))
            row=dict(status='no_seed_component',mode=mode,threshold=threshold,seed_fraction=fraction);rows.append(row)
            if best==0 or fraction<.70:continue
            x,y,ww,hh,_=stats[best]
            if x==0 or y==0 or x+ww==w or y+hh==h:row['status']='physical_image_border_contact';continue
            component=(labels==best).astype(np.uint8);contours,_=cv2.findContours(component,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if not contours:continue
            contour=max(contours,key=cv2.contourArea);hull=cv2.convexHull(contour);area=cv2.contourArea(contour)
            if area<1.25*g['area']:row['status']='insufficient_surface_expansion';continue
            filled=np.zeros_like(component);cv2.drawContours(filled,[contour],-1,1,-1);row['status']='no_quadrilateral'
            for eps in (.012,.022,.035):
                q=cv2.approxPolyDP(hull,eps*cv2.arcLength(hull,True),True).reshape(-1,2)
                if len(q)!=4:continue
                poly=np.zeros_like(component);cv2.fillConvexPoly(poly,q,1)
                iou=float((poly&filled).sum()/max(1,(poly|filled).sum()));coverage=float(poly[py,px].mean())
                if iou<.78 or coverage<.90:continue
                row.update(status='region_hypothesis',region_iou=iou,enclosed_seed_fraction=coverage)
                out.append(dict(corners=q.astype(float),region_iou=iou,enclosed_seed_fraction=coverage,mode=mode,threshold=threshold));break
    return out,rows


def discover_competing_regions(image,*,structure=None):
    out=dict(status='invalid_image',seed_count=0,raw_group_count=0,views=[],seeds=[],candidates=[],auto_select=None,auto_crop_allowed=False,requires_manual_confirmation=True)
    if not _image_ok(image):return out
    if structure is None:
        from autodocscanner.core.pr94_region_boundary import inspect_structure as structure
    h,w=image.shape[:2];diag=np.hypot(h,w);scales=[min(1.,1200/max(h,w))]
    if min(h,w)*scales[0]*.65>=75:scales.append(scales[0]*.65)
    for scale in scales:
        sw,sh=round(w*scale),round(h*scale);small=cv2.resize(image,(sw,sh),interpolation=cv2.INTER_AREA) if scale<1 else image
        gray=cv2.cvtColor(small,cv2.COLOR_BGR2GRAY);lab=cv2.cvtColor(small,cv2.COLOR_BGR2LAB)
        groups,views,raw=_groups(gray,lab);offset=len(out['views']);out['raw_group_count']+=raw
        out['views'].extend({**v,'working_scale':scale} for v in views)
        for g in groups:
            sid=len(out['seeds'])+1;proposals,attempts=_expand(lab,g)
            points=(g['points']+.5)*[w/sw,h/sh]-.5
            out['seeds'].append(dict(seed_id=sid,stroke_count=len(points),aligned_count=g['aligned'],area_fraction=g['area']/(sw*sh),ambiguous_fraction=g['ambiguous_fraction'],view_index=offset+g['view'],attempts=attempts,points=points.tolist()))
            for p in proposals:
                q=(p['corners']+.5)*[w/sw,h/sh]-.5;geom,q=inspect_geometry(q,image.shape)
                if geom['status'] not in ('in_frame','near_frame'):continue
                match=next((c for c in out['candidates'] if _distance(q,np.asarray(c['corners']),diag)<.006),None)
                if match is not None:
                    if sid not in match['seed_ids']:match['seed_ids'].append(sid)
                    match['supporting_expansions']+=1;continue
                if len(out['candidates'])>=MAX_PROPOSALS:continue
                out['candidates'].append({**p,'corners':q.tolist(),'family':'local_structure_guided','geometry':geom['status'],'seed_ids':[sid],'supporting_expansions':1,'structure':structure(image,q)})
    out['seed_count']=len(out['seeds']);out['status']='hypotheses_for_manual_review' if out['candidates'] else ('no_surface_hypothesis' if out['seeds'] else 'no_structure_seed')
    return out
