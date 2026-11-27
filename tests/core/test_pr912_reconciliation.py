"""Synthetic documents and contract checks, no real personal data."""
import json
import sys
import types
import cv2
import numpy as np
import pytest
from autodocscanner.core.pr912_reconciliation import reconcile,inspect_candidate
from autodocscanner.tools.pr912_validation import _collect,evaluate_photo,metrics


def scene(texture=False):
    im=np.full((500,800,3),45,np.uint8)
    if texture:
        rng=np.random.default_rng(9);im=np.clip(im.astype(float)+rng.normal(0,14,im.shape[:2])[:,:,None],0,255).astype(np.uint8)
    q=np.float32([[160,100],[640,100],[640,400],[160,400]])
    cv2.fillConvexPoly(im,q.astype(int),(200,220,215))
    for y in range(150,350,28):cv2.putText(im,'ABCD EFGH 1234',(200,y),cv2.FONT_HERSHEY_SIMPLEX,.8,(20,20,20),1,cv2.LINE_AA)
    p=np.float32([[x,y] for x in range(205,380,12) for y in range(145,340,28)])
    return im,q,p


def candidate(q,family='synthetic',pattern=True):return dict(corners=q.tolist(),family=family,structure={'distributed_stroke_pattern':pattern})
def seeds(p):return [dict(seed_id=1,points=p.tolist())]


@pytest.mark.parametrize('angle',[0,33,90,161])
@pytest.mark.parametrize('texture',[False,True])
def test_rotation_texture(angle,texture):
    im,q,p=scene(texture);m=cv2.getRotationMatrix2D((400,250),angle,.82);m[:,2]+=[100,250]
    im=cv2.warpAffine(im,m,(1000,1000),borderValue=(45,45,45));q=cv2.transform(q[None],m)[0];p=cv2.transform(p[None],m)[0];copy=im.copy()
    r=reconcile(im,[candidate(q)],seeds(p));c=r['candidates'][0]
    assert c['joint_supported'] and c['centered_sides']==4 and c['enclosed_seed_ids']==[1]
    assert r['auto_select'] is None and not r['auto_crop_allowed'] and r['requires_manual_confirmation']
    assert np.array_equal(im,copy);json.dumps(r,allow_nan=False)


def test_perspective_and_order():
    im,q,p=scene();target=np.float32([[120,80],[690,145],[610,435],[195,370]]);m=cv2.getPerspectiveTransform(q,target)
    im=cv2.warpPerspective(im,m,(800,500),borderValue=(45,45,45));p=cv2.perspectiveTransform(p[None],m)[0]
    r=reconcile(im,[candidate(target[[2,0,3,1]])],seeds(p));assert r['candidates'][0]['joint_supported']


def test_inner_outer_and_duplicate_preservation():
    im,q,p=scene();offset=np.float32([[25,25],[-25,25],[-25,-25],[25,-25]])
    rows=[candidate(q),candidate(q+3,'other'),candidate(q+offset),candidate(q-offset)]
    r=reconcile(im,rows,seeds(p));assert len(r['candidates'])==4 and len(r['groups'])==3
    assert r['groups'][0]['members']==[0,1] and r['groups'][0]['families']==['synthetic','other']
    assert [c['joint_supported'] for c in r['candidates']]==[True,True,False,False]
    assert r['nested_relations']


def test_grouping_not_transitive():
    im,q,p=scene();r=reconcile(im,[candidate(q),candidate(q+[5,0]),candidate(q+[10,0])],seeds(p))
    assert [g['members'] for g in r['groups']]==[[0,1],[2]]


def test_two_documents_no_selection():
    im=np.full((500,900,3),35,np.uint8);quads=[np.float32([[35,80],[415,80],[415,330],[35,330]]),np.float32([[480,150],[865,150],[865,410],[480,410]])]
    ss=[]
    for i,q in enumerate(quads):
        cv2.fillConvexPoly(im,q.astype(int),(180,220,210));x,y=q[0]
        ss.append(dict(seed_id=i+1,points=[[float(x+dx),float(y+dy)] for dx in range(30,170,10) for dy in range(40,200,25)]))
    r=reconcile(im,[candidate(q) for q in quads],ss)
    assert all(c['joint_supported'] for c in r['candidates']) and r['joint_conflicts'][0]['kind']=='separate_alternatives'
    assert r['auto_select'] is None


def test_cut_seed_advisory():
    im,q,p=scene();ss=seeds(p)+[dict(seed_id=2,points=[[x,180] for x in range(135,185,5)])]
    c=reconcile(im,[candidate(q)],ss)['candidates'][0]
    assert c['joint_supported'] and c['crossing_seed_ids']==[2] and not c['joint_without_crossing']


def test_frame_no_artificial_side():
    im=np.full((500,800,3),45,np.uint8);q=np.float32([[0,90],[500,90],[500,390],[0,390]])
    cv2.fillConvexPoly(im,q.astype(int),(200,220,215));p=np.float32([[100,200],[150,220]])
    c=reconcile(im,[candidate(q)],seeds(p))['candidates'][0]
    assert 'frame_limited' in [s['state'] for s in c['sides']] and not c['joint_supported']


def test_internal_thin_printed_rectangle_is_not_region_boundary():
    im,q,p=scene();inner=np.float32([[195,128],[600,128],[600,370],[195,370]])
    cv2.polylines(im,[inner.astype(int)],True,(10,10,10),2)
    c=reconcile(im,[candidate(inner)],seeds(p))['candidates'][0]
    assert not c['surface_boundary_supported']


def test_collection_order():
    _,q,_=scene();c=candidate(q);a={**c,'adaptive_recovery':{'hypothesis':q.tolist(),'recovered_structure':c['structure']},'region_guided':{'hypotheses':[c,c]}}
    r=_collect({'candidates':[a]},{'candidates':[candidate(q,'pr910')]},{'candidates':[candidate(q,'pr911')]})
    assert [x['family'] for x in r]==['synthetic','adaptive_recovery','region_guided','region_guided','pr910','pr911']


def test_validation_independence_and_privacy(monkeypatch):
    im,q,p=scene();raw=[candidate(q)];calls=[]
    # Isolate runner from unavailable predecessor modules; verify contract.
    gm=types.ModuleType('autodocscanner.tools.pr9_holdout_validation')
    gm._quad=lambda a,s:np.asarray(a,np.float32)
    gm._errors=lambda a,b,s:(float(np.linalg.norm(a-b,axis=1).mean()/np.hypot(*s[:2])),float(np.linalg.norm(a-b,axis=1).max()/np.hypot(*s[:2])))
    monkeypatch.setitem(sys.modules,gm.__name__,gm)
    structure=types.ModuleType('autodocscanner.core.pr94_region_boundary');structure.inspect_structure=lambda i,q:raw[0]['structure'];monkeypatch.setitem(sys.modules,structure.__name__,structure)
    def parent(i,ref,detectors):
        mean,worst=gm._errors(q,ref,i.shape)
        return dict(status='evaluated',candidates=[dict(error=dict(mean=mean,max=worst),adaptive_recovery={'error':None},region_guided={'hypotheses':[]})],structure_discovery={'candidates':[]},competing_structure={'candidates':[]}),raw,seeds(p)
    def audit(i,c,s):
        r=reconcile(i,c,s);calls.append(json.dumps(r));return r
    a=evaluate_photo(im,q,parent=parent,audit=audit);b=evaluate_photo(im,q+20,parent=parent,audit=audit)
    assert calls[0]==calls[1] and a['reconciliation']['candidates'][0]['error']!=b['reconciliation']['candidates'][0]['error']
    report=json.dumps(a,allow_nan=False);assert 'corners' not in report and '"points"' not in report
    assert metrics([a])['oracle_photos_within_2pct']==1 and metrics([a])['evidence']['joint_supported']['accurate_supported']==1
    with pytest.raises(ValueError):evaluate_photo(im,q,parent=parent,audit=lambda i,c,s:{**reconcile(i,c,s),'auto_crop_allowed':True})
    with pytest.raises(ValueError):evaluate_photo(im,q,parent=parent,audit=lambda i,c,s:{**reconcile(i,c,s),'candidates':[]})


def test_empty_and_invalid():
    im,q,p=scene();r=reconcile(im,[],seeds(p));assert not r['groups'] and not r['candidates']
    with pytest.raises(ValueError):reconcile(None,[],[])
    with pytest.raises(ValueError):reconcile(im,[candidate(q*0)],[])
    assert metrics([])['oracle_error']['mean'] is None


def test_seed_spatial_union_not_double_counted():
    im,q,p=scene();a=inspect_candidate(im,q,seeds(p),candidate(q)['structure']);b=inspect_candidate(im,q,seeds(p)+[dict(seed_id=2,points=p.tolist())],candidate(q)['structure'])
    assert a['enclosed_seed_hull_union_fraction']>0 and a['enclosed_seed_hull_union_fraction']==b['enclosed_seed_hull_union_fraction']
    assert a['enclosed_seed_tiles']==b['enclosed_seed_tiles']
    c=inspect_candidate(im,q,[],candidate(q)['structure']);assert c['surface_boundary_supported'] and not c['joint_supported']


def test_parent_capture_chain(monkeypatch):
    from autodocscanner.tools.pr912_validation import _parent_capture
    im,q,p=scene();c=candidate(q)
    original={'candidates':[{**c,'adaptive_recovery':{'hypothesis':q.tolist(),'recovered_structure':c['structure']},'region_guided':{'hypotheses':[c]}}]}
    independent={'candidates':[candidate(q,'pr910')]};competing={'candidates':[candidate(q,'pr911')],'seeds':seeds(p)}
    order=[]
    def module(name,**fields):
        obj=types.ModuleType(name);obj.__dict__.update(fields);monkeypatch.setitem(sys.modules,name,obj)
    def discovery(key,result):
        def f(i):
            assert np.array_equal(i,im);order.append(key);return result
        return f
    module('autodocscanner.core.pr99_region_guided',audit_regions=discovery('original',original))
    module('autodocscanner.core.pr910_structure_discovery',discover_structure_regions=discovery('independent',independent))
    module('autodocscanner.core.pr911_competing_structure',discover_competing_regions=discovery('competing',competing))
    def p99(i,ref,*,detectors,audit):
        audit(i);return {'status':'evaluated'}
    def next_parent(i,ref,*,detectors,discover,parent):
        r=parent(i,ref,detectors=detectors);discover(i);return r
    module('autodocscanner.tools.pr99_validation',evaluate_photo=p99)
    module('autodocscanner.tools.pr910_validation',evaluate_photo=next_parent)
    module('autodocscanner.tools.pr911_validation',evaluate_photo=next_parent)
    row,raw,ss=_parent_capture(im,q,None)
    assert row['status']=='evaluated' and order==['original','independent','competing']
    assert [x['family'] for x in raw]==['synthetic','adaptive_recovery','region_guided','pr910','pr911'] and ss==seeds(p)
