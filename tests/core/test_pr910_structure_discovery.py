"""Synthetic positive scenes, no personal document content."""
import json
import sys
import types
import cv2
import numpy as np
import pytest
from autodocscanner.core.pr910_structure_discovery import discover_structure_regions, MAX_PROPOSALS, MAX_SEEDS
from autodocscanner.core.pr93_independent_boundary import _distance, _order
from autodocscanner.tools.pr910_validation import structure_metrics, evaluate_photo


def cue(image,q):
    # Independent discovery does not rely on the downstream advisory cue.
    return {'distributed_stroke_pattern':True}


def draw(im,q,scale=.65):
    cv2.fillConvexPoly(im,q.astype(np.int32),(200,220,215))
    x,y=q[0].astype(int);width,height=(q[2]-q[0]).astype(int)
    for dy in range(35,height-25,28):
        cv2.putText(im,'ABCD EFGH 1234',(x+25,y+dy),cv2.FONT_HERSHEY_SIMPLEX,scale,(20,20,20),1,cv2.LINE_AA)


def scene(texture=False):
    im=np.full((500,800,3),45,np.uint8)
    if texture:
        rng=np.random.default_rng(9)
        im=np.clip(im.astype(float)+rng.normal(0,9,im.shape[:2])[:,:,None],0,255).astype(np.uint8)
    q=np.float32([[160,120],[640,120],[640,400],[160,400]]);draw(im,q)
    return im,q


def error(c,q,shape):return _distance(_order(np.array(c['corners'])),_order(q),np.hypot(*shape[:2]))


@pytest.mark.parametrize('texture',[False,True])
@pytest.mark.parametrize('angle',[0,25,90,157])
@pytest.mark.parametrize('scale',[1.,.65])
def test_independent_rotation_scale_texture(texture,angle,scale):
    im,q=scene(texture);mat=cv2.getRotationMatrix2D((400,250),angle,scale);mat[:,2]+=[100,250]
    im=cv2.warpAffine(im,mat,(1000,1000),borderValue=(45,45,45));q=cv2.transform(q[None],mat)[0];copy=im.copy()
    r=discover_structure_regions(im,structure=cue)
    assert r['seed_count']>0 and min(error(c,q,im.shape) for c in r['candidates'])<=.02
    assert r['auto_select'] is None and r['auto_crop_allowed'] is False and r['requires_manual_confirmation']
    assert len(r['candidates'])<=MAX_PROPOSALS and r['seed_count']<=2*MAX_SEEDS
    assert np.array_equal(im,copy)
    assert all(c['seed_ids'] and c['enclosed_seed_fraction']>=.90 for c in r['candidates'])
    json.dumps(r,allow_nan=False)


def test_perspective():
    im,q=scene();target=np.float32([[110,95],[675,140],[615,425],[195,370]])
    mat=cv2.getPerspectiveTransform(q,target);im=cv2.warpPerspective(im,mat,(800,500),borderValue=(45,45,45))
    r=discover_structure_regions(im,structure=cue)
    assert r['candidates'] and min(error(c,target,im.shape) for c in r['candidates'])<=.02


def test_two_printed_surfaces_remain_competing():
    im=np.full((650,1000,3),45,np.uint8)
    targets=[np.float32([[65,65],[470,65],[470,290],[65,290]]),np.float32([[530,355],[935,355],[935,580],[530,580]])]
    for q in targets:draw(im,q)
    r=discover_structure_regions(im,structure=cue)
    for q in targets:assert min(error(c,q,im.shape) for c in r['candidates'])<=.02
    assert r['auto_select'] is None and not r['auto_crop_allowed']


def test_frame_contact_not_boundary():
    im=np.full((500,800,3),45,np.uint8);q=np.float32([[0,90],[500,90],[500,390],[0,390]]);draw(im,q)
    r=discover_structure_regions(im,structure=cue)
    assert not r['candidates']
    assert any(a['status']=='component_touches_search_or_image_border' for s in r['seeds'] for a in s['attempts'])


@pytest.mark.parametrize('bad',[None,np.zeros((50,60,3),np.uint8),np.zeros((100,100),np.uint8)])
def test_invalid(bad):assert discover_structure_regions(bad,structure=cue)['status']=='invalid_image'


def test_no_structure():
    im=np.full((400,600,3),100,np.uint8)
    assert discover_structure_regions(im,structure=cue)['status']=='no_structure_seed'


def test_metrics_previous_preserved():
    def row(old,new):return {'status':'evaluated','candidates':[{'error':{'mean':old},'adaptive_recovery':{'error':None},'region_guided':{'hypotheses':[]}}], 'structure_discovery':{'seed_count':2,'seeds':[],'status':'hypotheses_for_manual_review','candidates':[{'error':{'mean':e,'max':2*e},'structure':{'distributed_stroke_pattern':True}} for e in new]}}
    m=structure_metrics([row(.04,[.01,.30]),row(.005,[.03]),{'status':'invalid_annotation'}])
    assert m['previous_oracle_photos_within_2pct']==1 and m['augmented_oracle_photos_within_2pct']==2
    assert m['additional_accurate_photos']==1 and m['photos_with_improved_oracle']==1
    assert m['wrong_structure_supported']==2 and m['severe_structure_supported']==1
    assert structure_metrics([])['independent_oracle_error']['mean'] is None


def test_annotation_independence_and_anonymous_report(monkeypatch):
    im,q=scene();calls=[]
    def discover(i):
        r=discover_structure_regions(i,structure=cue);calls.append(r);return r
    def parent(i,a,*,detectors):return {'status':'evaluated','candidates':[]}
    gm=types.ModuleType('autodocscanner.tools.pr9_holdout_validation');gm._quad=lambda p,s:_order(np.array(p));gm._errors=lambda p,r,s:(_distance(p,r,np.hypot(*s[:2])),.01)
    monkeypatch.setitem(sys.modules,gm.__name__,gm)
    a=evaluate_photo(im,q,parent=parent,discover=discover);b=evaluate_photo(im,q+10,parent=parent,discover=discover)
    assert calls[0]==calls[1]
    assert a['structure_discovery']['candidates'][0]['error']!=b['structure_discovery']['candidates'][0]['error']
    assert 'corners' not in json.dumps(a,allow_nan=False)
    with pytest.raises(ValueError):evaluate_photo(im,q,parent=parent,discover=lambda i:{**calls[0],'auto_crop_allowed':True})
