"""Generalized shadow evidence tests: synthetic geometry, no private inputs."""
import json
import cv2
import numpy as np
import pytest
from autodocscanner.core.pr95_region_evidence import _side,audit_region_evidence
from autodocscanner.core.pr94_region_boundary import discover_regions
from autodocscanner.tools.pr95_validation import evaluate_photo,summarize,main
from test_pr94_region_boundary import card_scene,err


def measure(kind):
    im=np.full((240,320,3),100,np.uint8);x=160
    if kind=='step': im[:,x:]=200
    if kind=='weak': im[:,x:]=108
    if kind=='gradient': im[:]=np.linspace(60,220,320)[None,:,None].astype(np.uint8)
    if kind=='line': im[:,x-1:x+2]=200
    if kind=='frame': x=2;im[:,x:]=200
    if kind=='shift': im[:,x+4:]=200
    if kind=='occluded': im[:60,x:]=200
    lab=cv2.GaussianBlur(cv2.cvtColor(im,cv2.COLOR_BGR2LAB),(3,3),0)
    return _side(lab,np.array([x,30]),np.array([x,210]),np.array([230,120]),3.)


@pytest.mark.parametrize('kind,expected',[
    ('step','centered'),('weak','centered'),('gradient','weak_or_unresolved'),
    ('line','weak_or_unresolved'),('frame','frame_limited'),('shift','displaced'),
    ('occluded','weak_or_unresolved'),('blank','weak_or_unresolved')])
def test_side_states(kind,expected):
    assert measure(kind)['state']==expected


@pytest.mark.parametrize('angle',[0,37,82,143])
@pytest.mark.parametrize('size',[(300,420),(600,800)])
def test_real_generators_preserved(angle,size):
    im,q=card_scene(angle,size); before=im.copy();parent=discover_regions(im)
    out=audit_region_evidence(im,discover=lambda _:parent)
    assert np.array_equal(im,before)
    assert out['auto_select'] is None and out['auto_crop_allowed'] is False
    assert out['requires_manual_confirmation']
    assert len(out['candidates'])==len(parent['candidates'])
    assert all(all(c[k]==v for k,v in p.items()) for c,p in zip(out['candidates'],parent['candidates']))
    assert any(c['boundary_and_structure'] and err(c['corners'],q,im.shape)<=.02 for c in out['candidates'])
    assert sum(len(g['members']) for g in out['groups'])==len(parent['candidates'])


def parent_with(quads,structured=True):
    return {'auto_select':None,'auto_crop_allowed':False,'candidates':[
        {'corners':np.asarray(q).tolist(),'family':f,'structure':{'distributed_stroke_pattern':structured},
         'contains_structured_supported_region':False,'inside_supported_region':False}
        for f,q in zip(('independent_lines','outer_region','chroma'),quads)]}


def test_complete_link_no_transitive_chain():
    im,q=card_scene();p=parent_with([q,q+[5,0],q+[10,0]])
    r=audit_region_evidence(im,discover=lambda _:p)
    assert [len(g['members']) for g in r['groups']]==[2,1]
    assert r['groups'][0]['families']==['independent_lines','outer_region']


def test_nested_structured_regions_not_eliminated():
    im,q=card_scene();center=q.mean(axis=0);p=parent_with([q,center+(q-center)*1.5])
    r=audit_region_evidence(im,discover=lambda _:p)
    assert len(r['candidates'])==2
    assert r['candidates'][0]['structured_outer_groups']==[2]
    assert r['candidates'][1]['structured_inner_groups']==[1]


def test_near_frame_never_complete_even_if_structured():
    im=np.full((240,320,3),80,np.uint8);q=np.array([[2,50],[270,50],[270,190],[2,190]])
    cv2.fillConvexPoly(im,q,(190,190,190))
    r=audit_region_evidence(im,discover=lambda _:parent_with([q]))['candidates'][0]
    assert r['frame_limited'] and not r['multiscale_boundary_supported']
    assert any(s['state']=='frame_limited' for s in r['multiscale_sides'])


def test_no_identity_claim_on_printed_negative():
    im,q=card_scene();r=evaluate_photo(im,None,negative=True)
    report=summarize([r],'development')
    assert report['negative_scene_metrics']['photo_count']==1
    assert report['paired_evidence_metrics']['boundary_and_structure']['negative_photos_with_evidence']==1
    assert report['auto_select'] is None and report['auto_crop_allowed'] is False


def test_two_printed_objects_conflict():
    im,_=card_scene();r=audit_region_evidence(np.hstack([im,im]))
    assert r['competing_evidence_groups']
    assert r['auto_select'] is None


def test_blank_negative():
    im=np.full((240,320,3),100,np.uint8);r=evaluate_photo(im,None,negative=True)
    assert not r['candidates'] and not r['groups']
    assert not r['competing_evidence_groups']


def test_unsafe_contract():
    im,q=card_scene()
    with pytest.raises(ValueError): audit_region_evidence(im,discover=lambda _:{'auto_select':q.tolist(),'auto_crop_allowed':False})


def test_annotation_and_report_contract():
    im,q=card_scene();p=audit_region_evidence(im)
    class Detector:
        def detect(self,image): return q,None
    kw=dict(detectors={'stable':Detector(),'adaptive':Detector()},audit=lambda _:p)
    a=evaluate_photo(im,q,**kw);b=evaluate_photo(im,q+[8,0],**kw)
    assert a['groups']==b['groups']
    assert [{k:v for k,v in c.items() if k!='error'} for c in a['candidates']]==[{k:v for k,v in c.items() if k!='error'} for c in b['candidates']]
    report=summarize([a],'development');raw=json.dumps(report,allow_nan=False)
    assert '"corners"' not in raw and '"hypothesis"' not in raw
    assert report['source_metrics']['all_families']['oracle_photos_within_2pct']==1
    assert report['paired_evidence_metrics']['boundary_and_structure']['photos_with_accurate_supported']==1
    assert report['paired_evidence_metrics']['boundary_and_structure']['supported_groups']<=report['paired_evidence_metrics']['boundary_and_structure']['supported_candidates']


def test_negative_cli_missing_not_negative(tmp_path,capsys):
    im=np.full((240,320,3),100,np.uint8)
    for name in ('negative.png','missing.png'): cv2.imwrite(str(tmp_path/name),im)
    ann=tmp_path/'annotations.json';ann.write_text('{"negative.png":null}')
    assert main([str(tmp_path),'--annotations',str(ann),'--dataset-role','development'])==0
    raw=capsys.readouterr().out;report=json.loads(raw)
    assert report['photo_count']==2 and report['negative_scene_metrics']['photo_count']==1
    assert 'negative.png' not in raw and 'missing.png' not in raw
