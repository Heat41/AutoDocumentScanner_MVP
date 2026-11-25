"""Synthetic documents and distractors; no real KTP or identifiers."""
import json
import cv2
import numpy as np
import pytest
from autodocscanner.core.pr94_region_boundary import discover_regions, inspect_structure, _region_proposals
from autodocscanner.tools.pr94_validation import evaluate_photo,summarize
from autodocscanner.core.pr93_independent_boundary import _order


def card_scene(angle=0, size=(400,600), fg=(185,220,230), bg=(35,45,55), printed=True):
    h,w=size
    patch=np.full((200,320,3),fg,np.uint8)
    if printed:
        for i in range(6):
            cv2.putText(patch,'ABCD EFGH  XY',(22,35+i*25),cv2.FONT_HERSHEY_SIMPLEX,.53,(30,35,40),1,cv2.LINE_AA)
        cv2.rectangle(patch,(245,50),(295,145),(110,140,150),-1)
    q=cv2.boxPoints(((w*.50,h*.50),(min(h,w)*.78,min(h,w)*.49),angle)).astype(np.float32)
    q=_order(q).astype(np.float32)
    if np.linalg.norm(q[1]-q[0])<np.linalg.norm(q[2]-q[1]): q=np.roll(q,1,axis=0)
    t=cv2.getPerspectiveTransform(np.array([[0,0],[319,0],[319,199],[0,199]],np.float32),q)
    im=np.full((h,w,3),bg,np.uint8)
    layer=cv2.warpPerspective(patch,t,(w,h));mask=cv2.warpPerspective(np.full((200,320),255,np.uint8),t,(w,h))
    im[mask>127]=layer[mask>127]
    return im,q


def err(q,p,shape):
    a,b=_order(q),_order(p)
    return min(np.linalg.norm(a-np.roll(b,k,axis=0),axis=1).mean() for k in range(4))/np.hypot(*shape[:2])


@pytest.mark.parametrize('angle',[0,28,76,145])
@pytest.mark.parametrize('size',[(300,420),(600,800)])
def test_structure_rotation_and_scale(angle,size):
    im,q=card_scene(angle,size)
    assert inspect_structure(im,_order(q))['distributed_stroke_pattern']


@pytest.mark.parametrize('angle',[0,37,82])
def test_real_generators_and_safe_contract(angle):
    im,q=card_scene(angle);original=im.copy();r=discover_regions(im)
    assert r['auto_select'] is None and r['auto_crop_allowed'] is False
    assert np.array_equal(im,original)
    near=[c for c in r['candidates'] if err(c['corners'],q,im.shape)<=.02]
    assert any(c['family']=='outer_region' for c in near)
    assert any(c['combined_evidence'] for c in near)


def test_blank_rectangle_no_structure():
    im,q=card_scene(printed=False);r=discover_regions(im)
    assert any(c['boundary_supported'] for c in r['candidates'])
    assert not any(c['combined_evidence'] for c in r['candidates'])


def test_chroma_proposal_not_seeded_by_lines():
    im,q=card_scene(19,fg=(255,0,255),bg=(105,105,105),printed=False)
    r=discover_regions(im,line_discover=lambda _: {'candidates':[],'auto_select':None,'auto_crop_allowed':False})
    assert any(c['family']=='chroma' and err(c['corners'],q,im.shape)<=.02 for c in r['candidates'])


def test_nested_actual_card_not_discarded():
    im,q=card_scene();outer=np.array([[50,35],[550,35],[550,365],[50,365]],np.int32)
    mask=np.zeros(im.shape[:2],np.uint8);cv2.fillConvexPoly(mask,outer,255);cv2.fillConvexPoly(mask,q.astype(np.int32),0)
    im[mask>0]=(100,100,100)
    r=discover_regions(im)
    near=[c for c in r['candidates'] if err(c['corners'],q,im.shape)<=.02]
    assert near and any(c['inside_supported_region'] for c in near)
    assert any(c['combined_evidence'] for c in near)
    assert r['auto_select'] is None


def test_printed_non_ktp_can_match_no_identity_claim():
    im,q=card_scene();r=discover_regions(im)
    assert any(c['combined_evidence'] for c in r['candidates'])
    # Synthetic print could equally be a book or another card.
    assert r['requires_manual_confirmation'] and r['auto_select'] is None


def test_blank_and_invalid():
    assert discover_regions(None)['status']=='invalid_image'
    assert not discover_regions(np.full((200,300,3),100,np.uint8))['candidates']


def test_provenance_and_annotation_isolation():
    im,q=card_scene();seen=[]
    class Detector:
        def detect(self,image): return q,None
    def search(image):
        seen.append(image.copy());return discover_regions(image)
    a=evaluate_photo(im,q,detectors={'stable':Detector(),'adaptive':Detector()},discover=search)
    b=evaluate_photo(im,q+[5,5],detectors={'stable':Detector(),'adaptive':Detector()},discover=search)
    assert np.array_equal(*seen)
    assert [{k:v for k,v in c.items() if k!='error'} for c in a['candidates']]==[{k:v for k,v in c.items() if k!='error'} for c in b['candidates']]
    report=summarize([a],'development');s=json.dumps(report,allow_nan=False)
    assert '"corners"' not in s and 'filename' not in s
    assert report['source_metrics']['all_families']['oracle_photos_within_2pct']==1


def test_reject_unsafe_discovery():
    im,q=card_scene()
    with pytest.raises(ValueError):
        discover_regions(im,line_discover=lambda _: {'candidates':[],'auto_select':None,'auto_crop_allowed':True})


def test_negative_scene_metrics_and_missing_annotation():
    im,_=card_scene()
    # Explicitly no target: other printed rectangle has document-like cues.
    row=evaluate_photo(im,None,negative=True)
    assert row['status']=='negative_scene'
    assert all(c['error'] is None for c in row['candidates'])
    assert evaluate_photo(im,None)['status']=='invalid_annotation'
    report=summarize([row],'development')
    assert report['negative_scene_metrics']['photos_with_combined_evidence']==1
    assert report['evaluated_photo_count']==0


def test_two_printed_objects_ambiguity():
    im,_=card_scene();im=np.hstack([im,im])
    r=discover_regions(im)
    assert r['competing_combined_geometry'] and r['auto_select'] is None


def test_regular_texture_not_proof():
    im,q=card_scene(printed=False)
    for y in range(115,290,16):
        for x in range(155,445,16): cv2.circle(im,(x,y),3,(30,30,30),-1)
    r=discover_regions(im)
    assert any(c['combined_evidence'] for c in r['candidates'])
    # Deliberate failure characterization: periodic texture can mimic strokes.
    assert r['requires_manual_confirmation'] and r['auto_select'] is None


def test_gradient():
    im,q=card_scene(28)
    im=np.clip(im.astype(float)+np.linspace(-40,40,im.shape[1])[None,:,None],0,255).astype(np.uint8)
    r=discover_regions(im)
    assert any(c['combined_evidence'] and err(c['corners'],q,im.shape)<=.02 for c in r['candidates'])


def test_negative_folder_cli(tmp_path,capsys):
    from autodocscanner.tools.pr94_validation import main
    im,_=card_scene(printed=False)
    assert cv2.imwrite(str(tmp_path/'negative.png'),im)
    assert cv2.imwrite(str(tmp_path/'unannotated.png'),im)
    annotation=tmp_path/'annotation.json';annotation.write_text('{"negative.png": null}')
    assert main([str(tmp_path),'--annotations',str(annotation),'--dataset-role','development'])==0
    raw=capsys.readouterr().out;report=json.loads(raw)
    assert report['negative_scene_metrics']['photo_count']==1
    assert report['photo_count']==2
    assert report['negative_scene_metrics']['photos_with_combined_evidence']==0
    assert 'negative.png' not in raw and 'unannotated.png' not in raw
