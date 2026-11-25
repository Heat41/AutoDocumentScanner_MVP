"""Compare diagnostic grids to actual remap calls and test positive-only runner."""
import json
import cv2
import numpy as np
import pytest
from autodocscanner.core.pr97_boundary_diagnostics import (center_grids,inspect_quad,audit_transforms,
    rotated_points,resized_points,preview_sampling_limits,draw_sampling_overlay)
from autodocscanner.core.pr95_region_evidence import _side
from autodocscanner.core.pr96_region_verification import _region_side,prepare_lab,verify_regions
from autodocscanner.tools.pr97_validation import evaluate_photo,evaluate_folder,main
from test_pr94_region_boundary import card_scene


def scene(x=60):
    im=np.full((300,420,3),80,np.uint8)
    q=np.array([[x,50],[360,50],[360,250],[x,250]],float)
    cv2.fillConvexPoly(im,q.astype('int32'),(190,190,190))
    return im,q


@pytest.mark.parametrize('angle',[0,29,89,145])
def test_exact_sampling_grids(monkeypatch,angle):
    im=np.full((300,420,3),100,np.uint8);lab=prepare_lab(im)
    a=np.array([200.,150.]);theta=np.deg2rad(angle);t=np.array([np.cos(theta),np.sin(theta)])
    b=a+75*t;center=(a+b)/2+20*np.array([-t[1],t[0]])
    grids=center_grids(a,b,center,2.3);calls=[];orig=cv2.remap
    def remap(src,x,y,*args,**kw):
        calls.append(np.stack((x,y),axis=-1));return orig(src,x,y,*args,**kw)
    monkeypatch.setattr(cv2,'remap',remap)
    _side(lab,a,b,center,2.3);_region_side(lab,a,b,center,2.3)
    actual=[c.reshape(48,17,6,2)[:,8] for c in calls[:3]]+[c.reshape(12,25,8,4,5,2)[:,12] for c in calls[3:]]
    assert len(actual)==len(grids)==5
    for a,b in zip(actual,grids.values()):np.testing.assert_allclose(a,b,atol=3e-5,rtol=0)


def test_rotation_and_resize():
    im,q=scene();a=inspect_quad(im,q);vs=audit_transforms(im,q,a)
    assert all(s['pr95']['state']==s['pr96']['state']=='centered' for s in a['sides'])
    assert len(vs)==3 and all(v['state_changes']=={'pr95':0,'pr96':0} for v in vs)
    p=rotated_points(q,im.shape)
    np.testing.assert_allclose(rotated_points(rotated_points(rotated_points(p,(420,300)),(300,420)),(420,300)),q)
    np.testing.assert_allclose(resized_points(resized_points(q,im.shape,(151,209)),(151,209),im.shape),q)


def test_wider_grid_loses_visibility():
    im,q=scene(9);a=inspect_quad(im,q);s=a['sides'][3]
    assert s['center_grids']['pr95_2.4']['fully_visible_block_fraction']==1
    assert s['center_grids']['pr96_1.8']['fully_visible_block_fraction']==0
    assert s['pr95']['state']=='centered' and s['pr96']['state']=='frame_limited'


def test_preview_limits():
    a=preview_sampling_limits((3000,4000));assert len(a)==2
    assert all(x['source_pixels_per_preview_pixel_max']>1 for x in a)
    assert all(x['pixel_center_convention_delta_max_px']==.5*(x['source_pixels_per_preview_pixel_max']-1) for x in a)


def test_overlay_no_mutation():
    im,q=scene();before=im.copy();a=inspect_quad(im,q);o=draw_sampling_overlay(im,q,a)
    assert np.array_equal(im,before) and not np.array_equal(im,o)


def test_positive_only_report_preserves_candidates(tmp_path):
    im,q=card_scene();r=verify_regions(im)
    class D:
        def detect(self,im):return q,None
    kw={'detectors':{'stable':D(),'adaptive':D()},'audit':lambda _:r}
    row=evaluate_photo(im,q,**kw)
    assert row['status']=='evaluated' and len(row['candidates'])==len(r['candidates'])
    assert all(c['region_sides']==[s['pr96'] for s in c['root_cause_sampling']['sides']] for c in row['candidates'])
    assert [s['pr96']['state'] for s in row['root_cause_reference']['original']['sides']]==row['reference_control']['pr96_side_states']
    assert evaluate_photo(im,None,**kw)['status']=='invalid_annotation'
    folder=tmp_path/'photos';folder.mkdir();cv2.imwrite(str(folder/'private.png'),im)
    out=tmp_path/'overlays'
    report=evaluate_folder(folder,{'private.png':q.tolist()},'development',overlay_dir=out,evaluator=lambda *a,**k:evaluate_photo(*a,**kw,**k))
    assert report['auto_select'] is None and report['auto_crop_allowed'] is False
    assert report['evaluated_photo_count']==1 and report['negative_scene_metrics']['photo_count']==0
    assert report['root_cause_summary']['reference_photos']==1
    assert set(x.name for x in out.iterdir())=={'D0001_reference.png','D0001_oracle.png'}
    raw=json.dumps(report,allow_nan=False);assert 'private.png' not in raw and '"corners"' not in raw
    with pytest.raises(FileExistsError):evaluate_folder(folder,{},'development',overlay_dir=out)


def test_missing_annotation_positive_only(tmp_path,capsys):
    im,q=scene();cv2.imwrite(str(tmp_path/'photo.png'),im);a=tmp_path/'ann.json';a.write_text('{"photo.png": null}')
    assert main([str(tmp_path),'--annotations',str(a),'--dataset-role','development'])==2
    r=json.loads(capsys.readouterr().out);assert r['root_cause_summary']['excluded_photos']['invalid_annotation']==1
    assert r['negative_scene_metrics']['photo_count']==0


def test_complex_positive_competing_printed_regions():
    im,q=card_scene(17);w=im.shape[1];im=np.hstack([im,im]);r=verify_regions(im)
    class D:
        def detect(self,im):return q,None
    kw={'detectors':{'stable':D(),'adaptive':D()},'audit':lambda _:r}
    a=evaluate_photo(im,q,**kw);b=evaluate_photo(im,q+[w,0],**kw)
    assert a['status']==b['status']=='evaluated'
    assert len(a['candidates'])>1
    assert any(c['error']['mean']<=.02 for c in a['candidates'])
    assert any(c['error']['mean']>.10 for c in a['candidates'])
    assert [{k:v for k,v in c.items() if k!='error'} for c in a['candidates']]==[{k:v for k,v in c.items() if k!='error'} for c in b['candidates']]
