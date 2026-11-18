import json
import numpy as np
from PIL import Image
import torch

from robot.data.indooruav_dataset import (
    cumulative_local_targets,cumulative_targets_to_world,letterbox_rgb,
    letterbox_depth,letterbox_layout,IndoorUAVVLADataset)
from experiments.uavflow_predictor_idm.rgbwarp_geometry import action_camera_transform


def test_cumulative_targets_keep_anchor_axes_during_turn():
    poses=np.array([[0,0,0,0],[0,-1,0,90],[1,-1,0,90],[1,-2,1,180.]])
    actions=cumulative_local_targets(poses,0)
    np.testing.assert_allclose(actions[:,:3],[[1,0,0],[1,1,0],[2,1,1],[2,1,1],[2,1,1]])
    np.testing.assert_allclose(actions[:2,3],np.pi/2,atol=1e-6)
    targets=cumulative_targets_to_world(poses[0],actions)
    np.testing.assert_allclose(targets[:,:3],poses[[1,2,3,3,3],:3],atol=1e-6)
    np.testing.assert_allclose(targets[:,3],poses[[1,2,3,3,3],3],atol=1e-5)


def test_terminal_clamps_pose_instead_of_zeroing_remaining_targets():
    poses=np.array([[0,0,0,0],[0,-1,0,0],[0,-2,0,0.]])
    np.testing.assert_allclose(cumulative_local_targets(poses,0)[:,0],[1,2,2,2,2])
    np.testing.assert_allclose(cumulative_local_targets(poses,1)[:,0],[1,1,1,1,1])
    assert np.count_nonzero(cumulative_local_targets(poses,2))==0


def test_metric_depth_uses_metres_equal_thirds_and_consistent_warp_units():
    from experiments.indooruav_warp.prepare import configure_metric_depth
    from experiments.uavflow_predictor_idm.rgbwarp_objectives import depth_loss
    cfg={'model':{'rgbwarp_correction':{'depth_scale_meters':100.}},'loss':{
        'depth_fixed_scale_meters':100.,'depth_linear_weight':1/3,
        'depth_log_weight':1/3,'depth_grad_weight':1/3,
        'depth_log_epsilon_normalized':1e-6}}
    # Nonuniform physical errors exercise both value and spatial-gradient loss.
    target=torch.tensor([[[1.,2.,4.],[2.,3.,5.],[3.,5.,8.]]])
    prediction=target+torch.tensor([[[.1,.2,-.1],[.3,-.2,.4],[.5,.1,-.3]]])
    valid=torch.ones_like(target,dtype=torch.bool)
    configure_metric_depth(cfg)
    loss=cfg['loss']
    new=depth_loss(prediction,target,valid,loss['depth_fixed_scale_meters'],
        linear_weight=loss['depth_linear_weight'],log_weight=loss['depth_log_weight'],
        grad_weight=loss['depth_grad_weight'],
        log_epsilon_normalized=loss['depth_log_epsilon_normalized'])
    torch.testing.assert_close(new['total'],(new['l1']+new['log_l1']+new['grad'])/3)
    assert loss['depth_linear_weight']==loss['depth_log_weight']==loss['depth_grad_weight']==1/3
    assert cfg['model']['rgbwarp_correction']['depth_scale_meters']==1.
    # A 2-metre prediction reaches projection as 2 metres, with no x100.
    assert 2.*cfg['model']['rgbwarp_correction']['depth_scale_meters']==2.


def test_yaw_wrap_and_rotated_current_heading():
    poses=np.array([[10,20,1,90],[11,20,2,100.]])
    action=cumulative_local_targets(poses,0)[0]
    np.testing.assert_allclose(action,[1,0,1,np.deg2rad(10)],atol=1e-6)
    poses=np.array([[0,0,0,359],[0,0,0,1.]])
    np.testing.assert_allclose(cumulative_local_targets(poses,0)[:,3],np.deg2rad(2),atol=1e-6)


def test_cumulative_warp_uses_only_last_pose_and_has_endpoint_gradients():
    actions=torch.zeros(1,5,4,requires_grad=True)
    shifted=actions+torch.tensor([.2,.1,0,.02])
    cumulative=action_camera_transform(shifted,action_representation='chunk_anchor_local')
    endpoint=action_camera_transform(shifted[:,-1:])
    torch.testing.assert_close(cumulative,endpoint)
    assert not torch.allclose(cumulative,action_camera_transform(shifted))
    cumulative.sum().backward()
    assert actions.grad[:,:4].count_nonzero()==0
    assert actions.grad[:,-1].abs().sum()>0


def test_letterbox_intrinsics_depth_scale_and_padding():
    rgb=np.full((720,1280,3),200,np.uint8)
    image,content=letterbox_rgb(rgb)
    layout,k,mask=letterbox_layout((720,1280))
    assert layout==(126,224,49,0)
    np.testing.assert_allclose(k,[[112,0,111.5],[0,112,111.5],[0,0,1]])
    assert image[:,:49].count_nonzero()==0 and image[:,175:].count_nonzero()==0
    assert content.sum()==126*224
    depth,valid,k2,_=letterbox_depth(np.full((720,1280),2000,np.uint16),(224,224),.001,'z')
    torch.testing.assert_close(depth[valid],torch.full_like(depth[valid],2.))
    assert depth[~valid].count_nonzero()==0
    np.testing.assert_array_equal(valid.numpy(),mask)
    np.testing.assert_array_equal(k2.numpy(),k)


def test_rgb_depth_padding_pixels_never_become_valid_supervision():
    raw=np.zeros((8,16),np.uint16);raw[2:6,4:12]=1000;raw[3,6]=65535
    depth,valid,_,content=letterbox_depth(raw,(224,224),.001)
    assert not valid[~content].any()
    assert torch.isfinite(depth).all() and depth.max()==1


def test_dataset_does_not_read_beyond_subtask_or_use_future_as_current(tmp_path):
    trajectory='hm3d_1/test/traj_1'
    for i in (11,12,13):
        for kind,folder in [('rgb','screenshots'),('depth','depth_screenshots')]:
            root=tmp_path/kind/trajectory/folder;root.mkdir(parents=True,exist_ok=True)
            data=np.full((8,16,3),i,np.uint8) if kind=='rgb' else np.full((8,16),1000,np.uint16)
            Image.fromarray(data).save(root/f'{i}.png')
    key='/hm3d_1/test/traj_1/vla_ins_1.json'
    manifest={'splits':{'train':[key],'val':[key]},'episodes':{key:{'trajectory':trajectory,
               'source':[11,13],'instruction':'Move forward.',
               'poses':[[0,0,0,0],[0,-1,0,0],[0,-2,0,0]]}}}
    path=tmp_path/'manifest.json';path.write_text(json.dumps(manifest))
    ds=IndoorUAVVLADataset(path,tmp_path,depth_unit_scale_m=.001,depth_kind='z')
    assert len(ds)==3
    first,last=ds[0],ds[2]
    np.testing.assert_allclose(first['actions'][0,:,0],[1,2,2,2,2])
    assert first['frame_indices'].tolist()==[0,2]
    assert first['action_real_target_mask'][0].tolist()==[True,True,False,False,False]
    assert last['frame_indices'].tolist()==[2,2]
    assert last['actions'].count_nonzero()==0
    assert not last['action_real_target_mask'].any()
    assert first['all_view_images'].shape==(2,1,3,224,224)
    assert first['all_view_images'][0].max()<first['all_view_images'][1].max()


def test_grouped_prepared_split_and_normalization_use_only_training():
    from pathlib import Path
    path=Path(__file__).resolve().parents[1]/'prepared/indooruav_vla/manifest.json'
    if not path.exists():return
    data=json.loads(path.read_text());train=set(data['splits']['train']);val=set(data['splits']['val'])
    assert not train&val and not (train|val)&set(data['official_test_excluded'])
    groups=lambda ids:{data['episodes'][key]['group'] for key in ids}
    assert not groups(train)&groups(val)
    stats=json.loads((path.parent/'action_stats/indooruav_vla.json').read_text())
    assert stats['source_split']=='train' and stats['action_representation']=='chunk_anchor_local'


def test_observation_only_inference_converts_independent_targets_and_yaw_units():
    from types import SimpleNamespace
    from experiments.indooruav_warp.inference import IndoorUAVWarpPolicy
    class Policy:
        def __init__(self):
            self.model=torch.nn.Linear(1,1)
            self.model.rgbwarp_correction=SimpleNamespace(config={'action_representation':'chunk_anchor_local'})
        def predict(self,images,instructions,**kwargs):
            self.inputs=kwargs
            assert images.shape==(1,1,3,224,224)
            actions=torch.zeros(1,5,4);actions[0,:,0]=torch.arange(1,6)
            return {'actions_meters_radians':actions}
    policy=Policy();adapter=IndoorUAVWarpPolicy(policy)
    output=adapter.predict(np.zeros((120,160,3),np.uint8),'Move forward.',
              first_pose=[10,20,1,90],current_pose=[10,20,1,90],is_start=True)
    targets=output['world_target_poses_degrees']
    np.testing.assert_allclose(targets[:,0],[11,12,13,14,15])
    np.testing.assert_allclose(targets[:,1:3],np.tile([20,1],(5,1)),atol=1e-6)
    np.testing.assert_allclose(output['simulator_target_poses'][:,3],np.pi/2)
    assert policy.inputs['current_prompt_pose'].tolist()==[[0.,0.,0.,0.,1.]]
    assert not any('depth' in key for key in policy.inputs)


def test_action_statistics_match_original_concatenation():
    ds = IndoorUAVVLADataset.__new__(IndoorUAVVLADataset)
    ds.chunk_size = 5
    ds.action_stats_key = 'indooruav_vla'
    ds.is_eval = False
    ds.episodes = {
        'a': {'poses': [[0, 0, 0, 0], [0, -1, 0, 30], [1, -1, 1, 90]]},
        'b': {'poses': [[2, 3, 1, -20], [1, 4, 1, 45]]},
    }
    ds.samples = [('a', 0), ('b', 0), ('a', 2), ('a', 1), ('b', 1)]
    original = np.concatenate([cumulative_local_targets(ds.episodes[key]['poses'], step)
                               for key, step in ds.samples], axis=0)
    result = ds.compute_action_statistics()['indooruav_vla']
    for key, expected in [('mean', original.mean(0)), ('std', original.std(0)),
                          ('min', original.min(0)), ('max', original.max(0)),
                          ('q01', np.quantile(original, .01, axis=0)),
                          ('q99', np.quantile(original, .99, axis=0))]:
        np.testing.assert_array_equal(result[key], expected)
