import json
import cv2
import numpy as np
import pytest
from autodocscanner.core.pr98_adaptive_localization import recover_tracks,audit_recovery
from autodocscanner.core.pr96_region_verification import verify_regions
from autodocscanner.tools.pr98_validation import evaluate_photo,summarize_recovery


def scene(left=60):
    q=np.array([[left,60],[350,60],[350,240],[left,240]],float)
    im=np.full((300,420,3),50,np.uint8)
    cv2.fillConvexPoly(im,q.astype('int32'),(190,210,220))
    return im,q


@pytest.mark.parametrize('shift',[np.array([5,5]),np.array([[6,5],[-5,9],[-7,-3],[4,-7]])])
def test_recover_translation_and_angle(shift):
    im,q=scene();p=q+shift;before=im.copy();r=recover_tracks(im,p)
    assert r['status']=='recovered_hypothesis'
    assert np.linalg.norm(np.asarray(r['hypothesis'])-q,axis=1).max()<2
    assert all(s['state']=='centered' for s in r['recovered_sides'])
    assert np.array_equal(im,before) and not r['auto_crop_allowed'] and r['auto_select'] is None


@pytest.mark.parametrize('left',[60,3])
def test_centered_and_adaptive_visibility(left):
    im,q=scene(left);r=recover_tracks(im,q)
    assert r['status']=='centered_tracks' and r['hypothesis'] is None
    if left==3:assert r['sides'][3]['narrow_pair_fraction']>.9


def test_no_pixels_outside_photo():
    im,q=scene(0);r=recover_tracks(im,q)
    assert r['sides'][3]['state']=='frame_limited' and not r['boundary_supported'] and r['hypothesis'] is None


@pytest.mark.parametrize('kind',['blank','gradient','thin_line'])
def test_no_complete_support_without_regional_step(kind):
    _,q=scene();im=np.full((300,420,3),50,np.uint8)
    if kind=='gradient':im[:]=np.linspace(20,210,420,dtype=np.uint8)[None,:,None]
    if kind=='thin_line':cv2.polylines(im,[q.astype('int32')],True,(220,220,220),1)
    r=recover_tracks(im,q);assert not r['boundary_supported'] and r['hypothesis'] is None


def test_competing_nested_boundaries():
    im,q=scene();cv2.fillConvexPoly(im,q.astype('int32'),(130,130,130));inset=np.array([[8,8],[-8,8],[-8,-8],[8,-8]])
    cv2.fillConvexPoly(im,(q+inset).astype('int32'),(220,220,220))
    r=recover_tracks(im,q+inset/2);assert any(s['state']=='ambiguous_tracks' for s in r['sides']) and r['hypothesis'] is None


@pytest.mark.parametrize('scale',[1.,.5,.25])
def test_rotation_and_scale(scale):
    im,q=scene();h,w=im.shape[:2];im=np.rot90(im,-1).copy();q=np.column_stack((h-1-q[:,1],q[:,0]));im=cv2.resize(im,None,fx=scale,fy=scale,interpolation=cv2.INTER_AREA);q=(q+.5)*scale-.5
    r=recover_tracks(im,q);assert r['boundary_supported']


def test_positive_competing_objects_and_report():
    im,q=scene();w=im.shape[1];im=np.hstack([im,im]);parent=verify_regions(im)
    out=audit_recovery(im,discover=lambda _:parent)
    assert len(out['candidates'])==len(parent['candidates'])>1
    assert [{k:v for k,v in c.items() if k!='adaptive_recovery'} for c in out['candidates']]==parent['candidates']
    class D:
        def detect(self,im):return q,None
    kw={'detectors':{'stable':D(),'adaptive':D()},'audit':lambda _:out}
    a=evaluate_photo(im,q,**kw);b=evaluate_photo(im,q+[w,0],**kw)
    def evidence(row):return [{k:v for k,v in c['adaptive_recovery'].items() if k not in ('error','mean_error_change')} for c in row['candidates']]
    assert evidence(a)==evidence(b)
    assert evaluate_photo(im,None,**kw)['status']=='invalid_annotation'
    report=summarize_recovery([a],'development');raw=json.dumps(report,allow_nan=False)
    assert '"corners"' not in raw and '"hypothesis"' not in raw
    assert report['auto_select'] is None and report['auto_crop_allowed'] is False
    assert report['adaptive_recovery_metrics']['augmented_oracle_photos_within_2pct']>=report['adaptive_recovery_metrics']['original_oracle_photos_within_2pct']
