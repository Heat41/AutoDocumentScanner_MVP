"""Synthetic regional transitions, parent regression, and evaluation privacy."""
import json
import cv2
import numpy as np
import pytest
from autodocscanner.core.pr96_region_verification import _region_side,prepare_lab,verify_regions
from autodocscanner.core.pr95_region_evidence import _side,audit_region_evidence
from autodocscanner.tools.pr96_validation import evaluate_photo,summarize,main
from test_pr94_region_boundary import card_scene,err


def side_scene(kind):
    im=np.full((240,320,3),100,np.uint8);x=160
    if kind=='step':im[:,x:]=200
    if kind=='weak':im[:,x:]=108
    if kind=='gradient':im[:]=np.linspace(60,220,320)[None,:,None].astype(np.uint8)
    if kind=='line':im[:,x-1:x+2]=200
    if kind=='frame':x=2;im[:,x:]=200
    if kind=='shift':im[:,x+5:]=200
    if kind=='occluded':im[:60,x:]=200
    if kind=='texture':im[:,x:]=100+(np.indices((240,160))[0]//4%2*2-1)[:,:,None]*35
    return prepare_lab(im),np.array([x,30]),np.array([x,210]),np.array([230,120]),3.


@pytest.mark.parametrize('kind,expected',[('step','centered'),('weak','centered'),('gradient','weak_or_unresolved'),('line','weak_or_unresolved'),('frame','frame_limited'),('shift','displaced'),('blank','weak_or_unresolved'),('occluded','weak_or_unresolved'),('texture','centered')])
def test_regional_states(kind,expected):
    assert _region_side(*side_scene(kind))['state']==expected


def test_texture_without_mean_step():
    args=side_scene('texture');r=_region_side(*args)
    assert r['texture_dominant_blocks']>=8 and r['color_dominant_blocks']==0
    assert _side(*args)['state']=='centered'  # Both methods detect this texture step.


@pytest.mark.parametrize('angle',[0,37,82,143])
@pytest.mark.parametrize('size',[(300,420),(600,800)])
def test_preserve_real_proposals(angle,size):
    im,q=card_scene(angle,size);before=im.copy();parent=audit_region_evidence(im)
    result=verify_regions(im,audit=lambda _:parent)
    assert np.array_equal(before,im)
    assert result['auto_select'] is None and result['auto_crop_allowed'] is False
    assert result['requires_manual_confirmation']
    assert result['groups']==parent['groups']
    assert len(result['candidates'])==len(parent['candidates'])
    assert all(all(c[k]==v for k,v in p.items()) for c,p in zip(result['candidates'],parent['candidates']))
    assert any(c['either_complete_with_structure'] and err(c['corners'],q,im.shape)<=.02 for c in result['candidates'])


def test_invalid_and_blank():
    assert verify_regions(None)['status']=='invalid_image'
    assert not verify_regions(np.full((240,320,3),100,np.uint8))['candidates']


def test_unsafe_contract():
    im,_=card_scene()
    with pytest.raises(ValueError):verify_regions(im,audit=lambda _: {'auto_select':None,'auto_crop_allowed':True})


def test_reference_control_isolation():
    im,q=card_scene();result=verify_regions(im);seen=[]
    class Detector:
        def detect(self,image):return q,None
    def run(image):seen.append(image.copy());return result
    kw={'detectors':{'stable':Detector(),'adaptive':Detector()},'audit':run}
    a=evaluate_photo(im,q,**kw);b=evaluate_photo(im,q+[12,0],**kw)
    assert np.array_equal(*seen)
    assert a['groups']==b['groups']
    assert [{k:v for k,v in c.items() if k!='error'} for c in a['candidates']]==[{k:v for k,v in c.items() if k!='error'} for c in b['candidates']]
    assert a['reference_control']!=b['reference_control']
    report=summarize([a],'development');raw=json.dumps(report,allow_nan=False)
    assert '"corners"' not in raw and 'filename' not in raw
    assert report['reference_control_metrics']['photo_count']==1
    assert report['source_metrics']['all_families']['oracle_photos_within_2pct']==1


def test_printed_negative_not_ktp_identity():
    im,_=card_scene();row=evaluate_photo(im,None,negative=True);r=summarize([row],'development')
    assert 'reference_control' not in row
    assert r['negative_scene_metrics']['photo_count']==1
    assert r['paired_evidence_metrics']['either_complete_with_structure']['negative_photos_with_evidence']==1
    assert r['auto_select'] is None and r['auto_crop_allowed'] is False


def test_negative_folder_and_missing(tmp_path,capsys):
    im=np.full((240,320,3),100,np.uint8);pos=tmp_path/'pos';neg=tmp_path/'neg';pos.mkdir();neg.mkdir()
    cv2.imwrite(str(pos/'missing.png'),im);cv2.imwrite(str(neg/'negative.png'),im)
    ann=tmp_path/'annotations.json';ann.write_text('{}')
    assert main([str(pos),'--annotations',str(ann),'--dataset-role','development','--negative-folder',str(neg)])==0
    raw=capsys.readouterr().out;r=json.loads(raw)
    assert r['photo_count']==2 and r['negative_scene_metrics']['photo_count']==1
    assert r['evaluated_photo_count']==0 and 'negative.png' not in raw and 'missing.png' not in raw
