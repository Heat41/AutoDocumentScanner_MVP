"""General synthetic conditions; no real photos, identities or tuned examples."""
import json
import cv2
import numpy as np
import pytest
from autodocscanner.core.pr93_independent_boundary import discover_boundaries
from autodocscanner.tools.pr93_validation import evaluate_photo, summarize


def scene(angle=0, size=(360,480), fg=(210,225,230), bg=(40,40,40)):
    h,w=size
    im=np.full((h,w,3),bg,np.uint8)
    q=cv2.boxPoints(((w*.5,h*.5),(min(h,w)*.62,min(h,w)*.40),angle)).round().astype(np.int32)
    cv2.fillConvexPoly(im,q,fg)
    return im,q


def error(c,q,shape):
    def cyclic(p):
        p=np.asarray(p);center=p.mean(axis=0)
        return p[np.argsort(np.arctan2(p[:,1]-center[1],p[:,0]-center[0]))]
    a,b=cyclic(c),cyclic(q)
    return min(np.linalg.norm(a-np.roll(b,i,axis=0),axis=1).mean() for i in range(4))/np.hypot(*shape[:2])


def good(result,q,shape,limit=.008):
    return [c for c in result['candidates'] if error(c['corners'],q,shape)<limit]


@pytest.mark.parametrize('angle',[0,19,47,83])
@pytest.mark.parametrize('size',[(240,320),(480,640),(1020,1360)])
def test_rotations_and_scales(angle,size):
    im,q=scene(angle,size)
    original=im.copy()
    r=discover_boundaries(im)
    assert any(c['boundary_supported'] for c in good(r,q,im.shape))
    assert r['auto_crop_allowed'] is False and r['auto_select'] is None
    assert np.array_equal(im,original)


@pytest.mark.parametrize('q',[
    [[75,55],[330,85],[290,225],[55,195]],
    [[130,45],[340,95],[360,250],[40,220]],
    [[85,55],[270,25],[335,245],[50,230]],
])
def test_perspective(q):
    im=np.full((300,400,3),40,np.uint8);q=np.asarray(q,np.int32)
    cv2.fillConvexPoly(im,q,(210,225,230))
    r=discover_boundaries(im)
    assert any(c['boundary_supported'] for c in good(r,q,im.shape))


def test_chroma_only_and_gradient():
    im,q=scene(17,fg=(255,0,255),bg=(105,105,105))
    r=discover_boundaries(im)
    assert any(c['boundary_supported'] for c in good(r,q,im.shape))
    im,q=scene(31)
    im=np.clip(im.astype(float)+np.linspace(-35,35,im.shape[1])[None,:,None],0,255).astype(np.uint8)
    assert any(c['boundary_supported'] for c in good(discover_boundaries(im),q,im.shape))


def test_fragmented_side_reconstructed_without_claiming_full_support():
    im=np.full((300,400,3),40,np.uint8);q=np.array([[65,55],[335,55],[335,245],[65,245]])
    cv2.fillConvexPoly(im,q,(210,225,230));cv2.rectangle(im,(175,35),(225,77),(40,40,40),-1)
    r=discover_boundaries(im)
    near=good(r,q,im.shape)
    assert near and all(not c['boundary_supported'] for c in near)
    assert r['auto_crop_allowed'] is False


def test_nested_solid_rectangle_remains_ambiguous():
    im=np.full((300,400,3),40,np.uint8);q=np.array([[65,55],[335,55],[335,245],[65,245]])
    cv2.fillConvexPoly(im,q,(210,225,230));cv2.rectangle(im,(125,105),(270,200),(40,70,90),-1)
    r=discover_boundaries(im)
    assert any(c['boundary_supported'] for c in good(r,q,im.shape))
    assert r['competing_supported_geometry']
    assert any(c['boundary_supported'] and c['nested_in_supported_region'] for c in r['candidates'])
    assert r['auto_select'] is None


def test_thin_internal_border_not_supported():
    im=np.full((300,400,3),40,np.uint8);q=np.array([[65,55],[335,55],[335,245],[65,245]])
    cv2.fillConvexPoly(im,q,(210,225,230));cv2.rectangle(im,(125,105),(270,200),(40,70,90),1)
    r=discover_boundaries(im)
    internal=np.array([[125,105],[270,105],[270,200],[125,200]])
    near=good(r,internal,im.shape)
    assert near and all(not c['boundary_supported'] for c in near)


@pytest.mark.parametrize('im',[None,np.zeros((20,20,3),np.uint8),np.full((300,400,3),100,np.uint8)])
def test_blank_invalid_safe(im):
    r=discover_boundaries(im)
    assert r['candidates']==[] and not r['auto_crop_allowed']


def test_evaluation_isolated_from_baselines_and_annotations():
    im,q=scene(19)
    events=[]
    class Missing:
        def detect(self,img):
            events.append('baseline');return None,{}
    class Forbidden:
        def detect(self,img):raise AssertionError('V2 called')
    def discovery(img):
        events.append('independent');return discover_boundaries(img)
    detectors={'stable':Missing(),'adaptive':Missing(),'v2':Forbidden()}
    a=evaluate_photo(im,q,detectors=detectors,discover=discovery)
    b=evaluate_photo(im,q+[10,0],detectors=detectors,discover=discovery)
    assert events==['independent','baseline','baseline']*2
    assert a['independent_candidates'] and len(a['independent_candidates'])==len(b['independent_candidates'])
    for x,y in zip(a['independent_candidates'],b['independent_candidates']):
        assert {k:v for k,v in x.items() if k!='error'}=={k:v for k,v in y.items() if k!='error'}
        assert x['error']!=y['error']
    out=summarize([a],'development');m=out['source_metrics']['independent_lines']
    assert m['additional_photos_with_candidate_within_2pct']==1
    assert out['auto_select'] is None and not out['auto_crop_allowed']
    assert '"corners"' not in json.dumps(out,allow_nan=False)


def test_bounded_candidates():
    im,q=scene(0)
    assert len(discover_boundaries(im,max_candidates=1)['candidates'])<=1
    with pytest.raises(ValueError):discover_boundaries(im,max_candidates=1000)


def stress_scenes():
    """Fixed seed, varied placement, size, orientation and capture conditions.

    Synthetic stress fixture, NOT independent real-world holdout data.
    """
    rng=np.random.default_rng(9301)
    for group in ('clean','gradient','blur_noise','texture','lowcontrast','occluded'):
        for i,(h,w) in enumerate(((320,480),(480,320),(560,740),(740,560))):
            m=min(h,w)
            rect=cv2.boxPoints(((w*(.36+.22*rng.random()),h*(.36+.22*rng.random())),
                                (m*(.48+.18*rng.random()),m*(.28+.11*rng.random())),
                                float(rng.uniform(-80,80))))
            rect+=rng.uniform(-m*.025,m*.025,size=(4,2))
            q=np.round(rect).astype(np.int32)
            fg,bg=(210,225,230),(45,50,55)
            if group=='lowcontrast':
                fg,bg=(113,117,122),(100,101,104)
            im=np.full((h,w,3),bg,np.uint8)
            if group=='texture':
                yy,xx=np.indices((h,w))
                im=np.clip(im.astype(float)+(18*np.sin(xx/7)*np.cos(yy/9))[...,None],0,255).astype(np.uint8)
            cv2.fillConvexPoly(im,q,fg)
            if group=='gradient':
                im=np.clip(im.astype(float)+np.linspace(-40,40,w)[None,:,None],0,255).astype(np.uint8)
            if group=='blur_noise':
                im=np.clip(cv2.GaussianBlur(im,(5,5),1.1).astype(float)+rng.normal(0,4,im.shape),0,255).astype(np.uint8)
            if group=='occluded':
                a,b=q[0],q[1]; mid=(a+b)/2
                cv2.circle(im,tuple(mid.astype(int)),max(3,int(np.linalg.norm(b-a)*.16)),bg,-1)
            yield group, i, im, q


def test_stress_matrix_recall_and_safety():
    # Review-only proposal recall, not automatic selection accuracy.
    # Keep the complete matrix in the test rather than cherry-pick successes.
    counts={}
    for group,i,im,q in stress_scenes():
        r=discover_boundaries(im)
        near=good(r,q,im.shape,limit=.02)
        count=counts.setdefault(group,{'photos':0,'oracle_within_2pct':0,'supported_within_2pct':0,'supported_over_2pct':0})
        count['photos']+=1
        count['oracle_within_2pct']+=bool(near)
        count['supported_within_2pct']+=any(c['boundary_supported'] for c in near)
        count['supported_over_2pct']+=sum(int(c['boundary_supported'] and error(c['corners'],q,im.shape)>.02) for c in r['candidates'])
        assert r['auto_select'] is None and r['auto_crop_allowed'] is False
    print(json.dumps(counts,sort_keys=True))
    assert all(c['oracle_within_2pct']>=3 for c in counts.values())
    assert counts['occluded']['supported_within_2pct']==0
