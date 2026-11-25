import cv2
import numpy as np
import pytest
from autodocscanner.core.pr99_region_guided import inspect_region,audit_regions,_segment,_features
from autodocscanner.tools.pr99_validation import region_metrics


def fixture(clutter=False):
    im=np.full((420,640,3),50,np.uint8)
    if clutter:
        rng=np.random.default_rng(99);im=np.clip(im.astype(float)+rng.normal(0,18,im.shape),0,255).astype(np.uint8)
        cv2.rectangle(im,(5,5),(108,85),(210,120,60),-1)
        cv2.rectangle(im,(535,105),(630,250),(80,110,170),-1)
    q=np.array([[130,110],[490,90],[510,320],[115,330]],np.int32)
    cv2.fillConvexPoly(im,q,(180,210,220))
    for y in range(150,300,25):cv2.putText(im,'ABCD EFGH',(165,y),cv2.FONT_HERSHEY_SIMPLEX,.6,(30,30,30),1)
    seed=q+np.array([[12,8],[-10,10],[-12,-9],[12,-8]])
    return im,q,seed


def err(p,q,shape):
    p=np.asarray(p);q=np.asarray(q)
    return min(np.linalg.norm(np.roll(p,k,axis=0)-q,axis=1).mean() for k in range(4))/np.hypot(*shape[:2])


@pytest.mark.parametrize('clutter',[False,True])
@pytest.mark.parametrize('scale',[.5,1.])
@pytest.mark.parametrize('rotation',[0,1,2,3])
def test_recovery(clutter,scale,rotation):
    im,q,seed=fixture(clutter);im=cv2.resize(im,None,fx=scale,fy=scale);q=(q+.5)*scale-.5;seed=(seed+.5)*scale-.5
    for _ in range(rotation):
        h,w=im.shape[:2];im=cv2.rotate(im,cv2.ROTATE_90_CLOCKWISE)
        q=np.column_stack((h-1-q[:,1],q[:,0]));seed=np.column_stack((h-1-seed[:,1],seed[:,0]))
    copy=im.copy();r=inspect_region(im,seed)
    assert r['hypotheses']
    assert min(err(h['corners'],q,im.shape) for h in r['hypotheses'])<.008
    assert all(err(h['corners'],q,im.shape)<err(seed,q,im.shape) for h in r['hypotheses'])
    assert r['auto_select'] is None and r['auto_crop_allowed'] is False
    assert np.array_equal(im,copy)


def test_identical_appearance():
    im=np.full((420,640,3),100,np.uint8);_,q,_=fixture()
    assert not inspect_region(im,q)['hypotheses']


def test_frame_is_not_background():
    im=np.full((300,500,3),30,np.uint8);q=np.array([[0,30],[400,30],[400,270],[0,270]])
    cv2.fillConvexPoly(im,q.astype(np.int32),(200,200,200))
    assert not inspect_region(im,q)['hypotheses']


@pytest.mark.parametrize('bad',[None,[[0,0]]*4,[[0,0],[100,0],[100,100],[-1,100]]])
def test_invalid_geometry(bad):
    im,_,_=fixture();assert not inspect_region(im,bad)['hypotheses']


def test_competing_printed_object_is_not_identity():
    im,q,seed=fixture(True);cv2.rectangle(im,(20,340),(230,410),(180,210,220),-1)
    other=np.array([[20,340],[230,340],[230,410],[20,410]])
    # Two targets can both be geometrically plausible. Never select either.
    r=inspect_region(im,other);assert r['auto_select'] is None and not r['auto_crop_allowed']
    assert inspect_region(im,seed)['hypotheses']


def test_preserve_parent_and_rec():
    im,q,seed=fixture();parent={'auto_select':None,'auto_crop_allowed':False,'candidates':[{'corners':seed.tolist(),'adaptive_recovery':{'hypothesis':q.tolist()}}]}
    r=audit_regions(im,discover=lambda _:parent,structure=lambda *_:{'distributed_stroke_pattern':True})
    assert r['candidates'][0]['corners']==parent['candidates'][0]['corners']
    assert r['candidates'][0]['adaptive_recovery']==parent['candidates'][0]['adaptive_recovery']
    assert 'region_guided' not in parent['candidates'][0]
    assert all(h['structure']['distributed_stroke_pattern'] for h in r['candidates'][0]['region_guided']['hypotheses'])
    with pytest.raises(ValueError):audit_regions(im,discover=lambda _:{**parent,'auto_crop_allowed':True},structure=lambda *_:{})


def test_metrics_preserve_previous_oracle_and_count_wrong_support():
    def c(e,rec,hs):return {'error':{'mean':e},'adaptive_recovery':{'error':None if rec is None else {'mean':rec}},'region_guided':{'views':[],'hypotheses':[{'error':{'mean':x,'max':x*2},'mean_error_change':x-e,'structure':{'distributed_stroke_pattern':True}} for x in hs]}}
    rows=[{'status':'evaluated','candidates':[c(.03,None,[.01,.012]),c(.30,None,[.25])]}, {'status':'evaluated','candidates':[c(.01,.008,[.025])]}, {'status':'invalid_annotation'}]
    m=region_metrics(rows)
    assert m['region_hypotheses']==4 and m['newly_accurate']==2 and m['lost_accuracy']==1
    assert m['previous_oracle_photos_within_2pct']==1 and m['augmented_oracle_photos_within_2pct']==2
    assert m['photos_with_improved_oracle']==1 and m['severe_region_with_structure']==1
    assert region_metrics([])['paired_before_error']['mean'] is None


def test_report_uses_annotation_only_after_discovery(monkeypatch):
    import sys,types,json
    from autodocscanner.tools.pr99_validation import evaluate_photo
    im,q,seed=fixture()
    # Integration contract of PR9.8, injected because this isolated test does
    # not need the full app or baseline engines.
    raw={'auto_select':None,'auto_crop_allowed':False,'candidates':[{'corners':seed.tolist(),'structure':{'distributed_stroke_pattern':True},'adaptive_recovery':{'hypothesis':None}}]}
    def audit(i):return audit_regions(i,discover=lambda _:raw,structure=lambda *_:{'distributed_stroke_pattern':True})
    def parent(i,annotation,*,detectors,audit):
        a=audit(i);return {'status':'evaluated','candidates':[{'error':{'mean':err(seed,annotation,i.shape),'max':.03},'structure':a['candidates'][0]['structure'],'adaptive_recovery':{'error':None}}]}
    pm=types.ModuleType('autodocscanner.tools.pr98_validation');pm.evaluate_photo=parent
    gm=types.ModuleType('autodocscanner.tools.pr9_holdout_validation');gm._quad=lambda p,s:np.array(p);gm._errors=lambda p,r,s:(err(p,r,s),err(p,r,s))
    monkeypatch.setitem(sys.modules,pm.__name__,pm);monkeypatch.setitem(sys.modules,gm.__name__,gm)
    a=evaluate_photo(im,q,audit=audit);b=evaluate_photo(im,q+5,audit=audit)
    ra=a['candidates'][0]['region_guided'];rb=b['candidates'][0]['region_guided']
    assert ra['views']==rb['views']
    assert ra['hypotheses'][0]['error']!=rb['hypotheses'][0]['error']
    assert 'corners' not in json.dumps(a,allow_nan=False)
    assert 'hypothesis' not in raw['candidates'][0].get('region_guided',{})


def test_two_printed_regions_can_both_pass_geometry():
    im=np.full((620,900,3),45,np.uint8)
    targets=[np.array([[75,80],[410,80],[410,280],[75,280]]),np.array([[490,320],[825,320],[825,520],[490,520]])]
    for q in targets:
        cv2.fillConvexPoly(im,q.astype(np.int32),(200,220,210))
        x,y=q[0]
        for j in range(5):cv2.putText(im,'ABCD EFGH',(int(x+25),int(y+35+27*j)),cv2.FONT_HERSHEY_SIMPLEX,.65,(20,20,20),1)
    for q in targets:
        r=inspect_region(im,q)
        assert r['hypotheses'] and all(h['region_iou']>.9 for h in r['hypotheses'])
        assert r['auto_select'] is None and not r['auto_crop_allowed']
