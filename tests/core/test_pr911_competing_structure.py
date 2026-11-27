"""Synthetic positive scenes and safety contracts, no private documents."""
import json
import cv2
import numpy as np
import pytest
from autodocscanner.core.pr911_competing_structure import discover_competing_regions, _local_groups, _expand, MAX_SEEDS, MAX_PROPOSALS
from autodocscanner.core.pr93_independent_boundary import _order,_distance
from autodocscanner.tools.pr911_validation import evaluate_photo,metrics


def draw(im,q):
    cv2.fillConvexPoly(im,q.astype(int),(200,220,215))
    x,y=q[0].astype(int);w,h=(q[2]-q[0]).astype(int)
    for dy in range(35,h-25,28):cv2.putText(im,'ABCD EFGH 1234',(x+25,y+dy),cv2.FONT_HERSHEY_SIMPLEX,.65,(20,20,20),1,cv2.LINE_AA)


def scene(texture=False):
    im=np.full((500,800,3),45,np.uint8)
    if texture:
        for y in range(8,500,12):
            for x in range(8,800,12):cv2.ellipse(im,(x,y),(3,4),25,0,360,(105,110,120),-1)
    q=np.float32([[160,120],[640,120],[640,400],[160,400]]);draw(im,q)
    return im,q


def error(c,q,shape):return _distance(_order(c['corners']),_order(q),np.hypot(*shape[:2]))


@pytest.mark.parametrize('texture',[False,True])
@pytest.mark.parametrize('angle',[0,25,90,157])
def test_local_discovery(texture,angle):
    im,q=scene(texture);mat=cv2.getRotationMatrix2D((400,250),angle,1);mat[:,2]+=[100,250]
    im=cv2.warpAffine(im,mat,(1000,1000),borderValue=(45,45,45));q=cv2.transform(q[None],mat)[0];copy=im.copy()
    r=discover_competing_regions(im)
    assert r['candidates'] and min(error(c,q,im.shape) for c in r['candidates'])<=.02
    assert r['auto_select'] is None and not r['auto_crop_allowed'] and r['requires_manual_confirmation']
    assert r['seed_count']<=2*MAX_SEEDS and len(r['candidates'])<=MAX_PROPOSALS
    assert np.array_equal(im,copy)
    assert all(s['area_fraction']<=.58**2+.01 for s in r['seeds'])
    json.dumps(r,allow_nan=False)


def test_lattice_centers_not_row_evidence():
    p=np.float64([(x*12,y*12) for y in range(10) for x in range(10)])
    g=_local_groups(p,np.full(len(p),8.),np.full((len(p),3),100.))
    assert all(x['aligned']<len(p)/2 for x in g)


def test_frame_contact_not_physical_boundary():
    im=np.full((500,800,3),45,np.uint8);q=np.float32([[0,90],[500,90],[500,390],[0,390]]);draw(im,q)
    r=discover_competing_regions(im)
    assert not r['candidates']
    assert any(a['status']=='physical_image_border_contact' for s in r['seeds'] for a in s['attempts'])


def test_two_printed_surfaces_remain_alternatives():
    im=np.full((650,1000,3),45,np.uint8)
    targets=[np.float32([[65,65],[470,65],[470,290],[65,290]]),np.float32([[530,355],[935,355],[935,580],[530,580]])]
    for q in targets:draw(im,q)
    r=discover_competing_regions(im)
    for q in targets:assert min(error(c,q,im.shape) for c in r['candidates'])<=.02
    assert r['auto_select'] is None


@pytest.mark.parametrize('bad',[None,np.zeros((50,60,3),np.uint8),np.zeros((100,100),np.uint8)])
def test_invalid(bad):assert discover_competing_regions(bad)['status']=='invalid_image'


def test_no_structure():assert discover_competing_regions(np.full((400,600,3),100,np.uint8))['status']=='no_structure_seed'


def test_annotation_independent_and_anonymous():
    im,q=scene();calls=[]
    def discover(i):
        r=discover_competing_regions(i);calls.append(r);return r
    def parent(i,a,*,detectors):return dict(status='evaluated',candidates=[],structure_discovery={'candidates':[]})
    a=evaluate_photo(im,q,parent=parent,discover=discover);b=evaluate_photo(im,q+80,parent=parent,discover=discover)
    assert calls[0]==calls[1]
    assert a['competing_structure']['candidates'][0]['error']!=b['competing_structure']['candidates'][0]['error']
    assert a['competing_structure']['seeds'][0]['reference_point_fraction']!=b['competing_structure']['seeds'][0]['reference_point_fraction']
    assert 'corners' not in json.dumps(a,allow_nan=False) and '"points"' not in json.dumps(a)
    with pytest.raises(ValueError):evaluate_photo(im,q,parent=parent,discover=lambda i:{**calls[0],'auto_crop_allowed':True})
    m=metrics([a,b]);assert m['independent_candidates']>=2 and m['additional_accurate_photos']==1
    assert metrics([])['independent_oracle_error']['mean'] is None
