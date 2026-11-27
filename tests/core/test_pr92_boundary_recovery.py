"""Synthetic geometry regressions. Not a substitute for unseen real photos."""
import json
import cv2
import numpy as np
import pytest
from autodocscanner.core.pr92_boundary_recovery import recover_boundary, inspect_geometry
from autodocscanner.tools.pr92_validation import evaluate_photo, summarize, main


def scene(w=600,h=400,angle=0,foreground=(210,210,210),background=(40,40,40)):
    q=cv2.boxPoints(((w*.5,h*.5),(min(w,h)*.68,min(w,h)*.42),angle)).astype(float)
    im=np.full((h,w,3),background,np.uint8)
    cv2.fillConvexPoly(im,np.round(q).astype(np.int32),foreground)
    return im,q


def error(a,b):
    a=np.asarray(a); b=np.asarray(b)
    return min(np.linalg.norm(a-np.roll(b,i,axis=0),axis=1).mean() for i in range(4))


@pytest.mark.parametrize('size',[(320,240),(640,480),(1200,800),(480,640)])
@pytest.mark.parametrize('angle',[0,17,41,78])
def test_scale_rotation_recovery(size,angle):
    w,h=size; im,q=scene(w,h,angle)
    original=q+[min(w,h)*.012,0]
    before=im.copy()
    r=recover_boundary(im,original)
    assert r['status']=='hypothesis_for_manual_review'
    assert error(r['hypothesis'],q)<error(original,q)*.6
    assert all(s['state']=='centered' for s in r['recovered_sides'])
    assert r['auto_select'] is None and r['auto_crop_allowed'] is False
    assert np.array_equal(im,before)


def test_perspective_and_smooth_lighting():
    im=np.full((480,640,3),45,np.uint8)
    q=np.array([[130,100],[500,75],[530,350],[100,380]],float)
    cv2.fillConvexPoly(im,q.astype(np.int32),(190,200,210))
    im=np.clip(im.astype(float)+np.linspace(-25,25,640)[None,:,None],0,255).astype(np.uint8)
    r=recover_boundary(im,q+[6,0])
    assert r['hypothesis'] is not None
    assert error(r['hypothesis'],q)<3


def test_chroma_boundary():
    im,q=scene(foreground=(255,0,255),background=(105,105,105))
    r=recover_boundary(im,q+[5,0])
    assert r['hypothesis'] is not None
    assert error(r['hypothesis'],q)<2


def test_clean_candidate_not_needlessly_shifted():
    im,q=scene()
    r=recover_boundary(im,q)
    assert r['status']=='persistent_centered_transitions' and r['hypothesis'] is None


def test_thin_printed_border_not_persistent():
    im,q=scene(foreground=(200,200,200),background=(200,200,200))
    cv2.polylines(im,[q.astype(np.int32)],True,(20,20,20),2)
    r=recover_boundary(im,q)
    assert r['status']=='insufficient_or_ambiguous_transitions'
    assert r['hypothesis'] is None


def test_solid_nested_rectangle_is_not_document_proof():
    im,q=scene()
    inner=(q-q.mean(axis=0))*.5+q.mean(axis=0)
    cv2.fillConvexPoly(im,inner.astype(np.int32),(20,20,20))
    r=recover_boundary(im,inner)
    assert r['status']=='persistent_centered_transitions'  # known limitation
    assert r['auto_crop_allowed'] is False and r['auto_select'] is None


def test_flat_and_large_shift_cannot_be_recovered():
    im,q=scene()
    assert recover_boundary(np.full_like(im,80),q)['hypothesis'] is None
    assert recover_boundary(im,q+[60,0])['hypothesis'] is None


def test_geometry_distinguishes_extrapolation_and_invalid():
    im,q=scene(); h,w=im.shape[:2]
    for pts,state in [([[-1,70],[200,70],[200,300],[-1,300]],'off_photo'),
                      ([[-100,70],[200,70],[200,300],[-100,300]],'invalid_geometry'),
                      ([[2,70],[200,70],[200,300],[2,300]],'near_frame'),
                      ([[60,70],[200,70],[200,300],[200,300]],'invalid_geometry')]:
        r=recover_boundary(im,pts)
        assert r['status']==state and r['hypothesis'] is None
    assert inspect_geometry([[0,0]],im.shape)[0]['status']=='invalid_points'


def test_evaluation_is_annotation_independent_and_anonymous():
    im,q=scene()
    calls=[]
    class D:
        def __init__(self,name):self.name=name
        def detect(self,image):calls.append(self.name); return q+[5,0],{}
    ds={s:D(s) for s in ('stable','adaptive','v2')}
    a=evaluate_photo(im,q,detectors=ds); b=evaluate_photo(im,q+[15,0],detectors=ds)
    assert calls==['stable','adaptive']*2
    for source in ('stable','adaptive'):
        aa,bb=a['candidates'][source],b['candidates'][source]
        assert aa['sides']==bb['sides'] and aa['recovered_sides']==bb['recovered_sides']
        assert aa['original_error']!=bb['original_error']
    out=summarize([a],'development')
    assert out['source_metrics']['stable']['improved']==1
    assert out['source_metrics']['stable']['paired_recovery_count']==1
    assert 'hypothesis' not in json.dumps(out).replace('hypothesis_for_manual_review','')
    assert 'corners' not in json.dumps(out)


def test_cli_missing_directory_is_specific(tmp_path,capsys):
    with pytest.raises(SystemExit) as e:
        main([str(tmp_path/'no-folder'),'--annotations','no.json','--dataset-role','development'])
    assert e.value.code==2 and 'Folder foto tidak ditemukan' in capsys.readouterr().err


def test_two_parallel_persistent_transitions_remain_ambiguous():
    im=np.full((400,600,3),30,np.uint8)
    q=np.array([[100,80],[500,80],[500,320],[100,320]],float)
    im[80:321,100:501]=120
    im[87:314,107:494]=210
    r=recover_boundary(im,q+[3,3])
    assert r['hypothesis'] is None
    assert all(s['state']=='ambiguous_transition' for s in r['sides'])


def test_long_occlusion_does_not_invent_missing_edge():
    im=np.full((400,600,3),35,np.uint8)
    q=np.array([[100,80],[500,80],[500,320],[100,320]],float)
    cv2.fillConvexPoly(im,q.astype(np.int32),(200,200,200))
    im[65:95,160:440]=35
    r=recover_boundary(im,q+[5,0])
    assert r['hypothesis'] is None and r['auto_crop_allowed'] is False
