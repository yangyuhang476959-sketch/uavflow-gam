"""Observation-only progress state, legacy compatibility and conditioning."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from robot.data.uavflow_dataset import episode_pose_progress,episode_yaw4d_state
from experiments.uavflow_support_aware.inference import canonical_progress_state
from experiments.uavflow_predictor_idm.data import EpisodePoseNormalizer
from experiments.uavflow_predictor_idm.objectives import prepare_observed_pose_history


def history():
    raw=np.zeros((5,6));raw[:,4]=[0,90,180,-90,0]
    raw[:,:3]=[[0,0,0],[100,0,0],[100,100,0],[0,100,0],[0,0,0]]
    return raw


def test_full_turn_and_return_to_start_preserve_progress():
    states=episode_pose_progress(history())
    np.testing.assert_allclose(states[0,:5],states[-1,:5],atol=1e-6)
    np.testing.assert_allclose(states[-1,5:],[2*np.pi,4.],rtol=1e-6)
    np.testing.assert_array_equal(states[0],[0,0,0,0,1,0,0])
    reverse=history();reverse[:,4]*=-1
    assert episode_pose_progress(reverse)[-1,5]==pytest.approx(-2*np.pi)


def test_prefix_causality_clamped_absorbing_and_heading_units():
    raw=history();raw[:,4]+=179;raw[:,3]=10;raw[:,5]=-12
    states=episode_pose_progress(raw)
    np.testing.assert_array_equal(states[:3],episode_pose_progress(raw[:3]))
    repeated=np.concatenate((raw,np.repeat(raw[-1:],5,axis=0)))
    np.testing.assert_array_equal(episode_pose_progress(repeated)[-1],states[-1])
    for index in range(len(raw)):
        old=episode_yaw4d_state(raw[0],raw[index])
        np.testing.assert_allclose(states[index,:3],old[:3]*.01,atol=1e-6)
        np.testing.assert_allclose(states[index,3:5],[np.sin(np.deg2rad(old[3])),np.cos(np.deg2rad(old[3]))],atol=1e-6)
    raw=np.zeros((2,6));raw[:,4]=[179,-179];raw[1,2]=100
    state=episode_pose_progress(raw)[-1]
    assert state[5]==pytest.approx(np.deg2rad(2)) and state[6]==1


def test_online_ragged_history_matches_training_and_rejects_invented_progress():
    images=torch.zeros(2,1,3,2,2);raw=history()
    result=canonical_progress_state(images,['a','b'],None,[raw,raw[:2]])
    torch.testing.assert_close(result,torch.tensor(np.stack((episode_pose_progress(raw)[-1],episode_pose_progress(raw[:2])[-1]))))
    with pytest.raises(ValueError,match='two endpoints'):
        canonical_progress_state(images,['a','b'],None,None,np.stack([raw[0]]*2),np.stack([raw[-1]]*2))
    with pytest.raises(ValueError,match='endpoints'):
        canonical_progress_state(images,['a','b'],None,[raw,raw[:2]],current_sim_pose=np.zeros((2,6)))
    with pytest.raises(ValueError,match='nonnegative'):
        bad=result.clone();bad[0,-1]=-1
        canonical_progress_state(images,['a','b'],bad,None)


def test_baseline_appends_scaled_progress_without_changing_legacy_normalizer():
    norm=EpisodePoseNormalizer(dict(q01=[-1]*5,q99=[1]*5))
    states=torch.tensor(episode_pose_progress(history()))[None]
    batch=dict(episode_pose=states[...,:5],episode_pose_progress=states)
    model=SimpleNamespace(use_pose_history=True,pose_progress_enabled=True,training=False)
    pose=prepare_observed_pose_history(model=model,pose_normalizer=norm,batch=batch,context_len=5)
    torch.testing.assert_close(pose[...,:5],norm.normalize(states[...,:5]))
    torch.testing.assert_close(pose[0,-1,5:],torch.tensor([1.,.4]))
    model.pose_progress_enabled=False
    assert prepare_observed_pose_history(model=model,pose_normalizer=norm,batch=batch,context_len=5).shape[-1]==5


def test_numeric_seven_dimensional_state_reaches_qwen_slots_and_gradients():
    from experiments.uavflow_predictor_idm.vlm_conditioning import FrozenQwen35Conditioner
    class Encoder(nn.Module):
        prompt_mode='current_image_action_question'
        def __init__(self):
            super().__init__();self.embedding=nn.Embedding(8,16).requires_grad_(False)
            self.qwen=SimpleNamespace(get_input_embeddings=lambda:self.embedding)
        def forward(self,images,texts,*,omit_prompt_pose,state_placeholder):
            assert omit_prompt_pose and state_placeholder
            values=self.embedding(torch.tensor([[7,1,2,3,4,5]]))
            return dict(joint_layers=(values[:,1:]+values[:,:1])[:,None],joint_mask=torch.ones(1,5,dtype=torch.bool))
    cond=FrozenQwen35Conditioner.__new__(FrozenQwen35Conditioner);nn.Module.__init__(cond)
    cond.encoder=Encoder();cond.numeric_pose_token=True;cond.pose_progress_enabled=True;cond.state_token_id=7
    cond.register_buffer('state_scale',torch.tensor([10.,10.,10.,1.,1.,2*torch.pi,10.]))
    cond.state_encoder=nn.Sequential(nn.Linear(7,256),nn.GELU(),nn.Linear(256,16))
    states=torch.tensor(episode_pose_progress(history()))
    kwargs=dict(images=torch.zeros(1,1,1,3,2,2))
    a=cond.encode_tokens(['Turn'],current_pose=states[:1],**kwargs)['last_hidden_state']
    b=cond.encode_tokens(['Turn'],current_pose=states[-1:],**kwargs)['last_hidden_state']
    assert not torch.allclose(a,b)
    b.sum().backward()
    assert cond.state_encoder[0].weight.grad[:,5:].abs().sum()>0
    assert cond.encoder.embedding.weight.grad is None
    assert not cond.encoder.embedding._forward_hooks


def test_real_dataset_anchors_collate_and_absorbing_keep_prefix_state(tmp_path):
    import io
    import json
    import pyarrow as pa
    import pyarrow.parquet as pq
    from PIL import Image
    from torch.utils.data import DataLoader
    from robot.data.uavflow_dataset import UAVFlowParquetDataset
    raw=np.zeros((12,6));raw[:,0]=np.arange(12)*20;raw[:,4]=np.arange(12)*40
    image=io.BytesIO();Image.new('RGB',(4,4)).save(image,format='PNG')
    log=json.dumps(dict(raw_logs=raw.tolist(),instruction='Pass the tree on the right'))
    pq.write_table(pa.Table.from_pylist([
        dict(id='episode',frame_idx=i,log=log,image=dict(bytes=image.getvalue()))
        for i in range(len(raw))]),tmp_path/'train-00000.parquet')
    ds=UAVFlowParquetDataset(tmp_path,future_steps=1,chunk_size=5,visual_anchor_stride=5,
        action_convention='yaw4d',action_delta_mode='openvla_yaw_only',translation_scale=.01,
        include_episodes=['episode'],openvla_uav_compatible=True,terminal_zero_action=True,
        endpoint_absorbing_window_count=5,openvla_prompt_pose_mode='evaluator_yaw4d',
        pose_progress_enabled=True)
    expected=torch.tensor(episode_pose_progress(raw))
    for sample in ds:
        torch.testing.assert_close(sample['episode_pose_progress'],expected[sample['frame_indices']])
        torch.testing.assert_close(sample['episode_pose'],sample['episode_pose_progress'][:,:5])
    assert any(ds[i]['frame_indices'][-1]==11 for i in range(len(ds)))
    batch=next(iter(DataLoader(ds,batch_size=2)))
    assert batch['episode_pose_progress'].shape==(2,2,7)


def test_gam_predictor_accepts_seven_channel_history_and_f0(monkeypatch):
    from experiments.uavflow_predictor_idm.model import UAVFlowPredictorIDM
    from experiments.uavflow_remote_ablation.test_geometry_architectures import TinyDA3
    import robot.modeling.future_predictor as predictor
    monkeypatch.setattr(predictor,'_HAS_FLEX',False)
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        model=UAVFlowPredictorIDM(da3=TinyDA3(),idm=None,action_dim=4,action_chunk_size=5,
            d_model=256,depth=1,num_heads=4,language_dim=16,language_len=2,
            compute_idm_branch=False,direct_action_enabled=True,deep_action_enabled=True,
            gradient_checkpointing=False,use_pose_history=True,pose_progress_enabled=True,
            use_fixed_first_frame=True).eval()
        # Predictor LayerScale intentionally starts at zero; open the test
        # gate to verify the conditioning path after this gate has learned.
        with torch.no_grad():
            model.predictor.blocks[0].ls_self.gamma.fill_(.1)
        observed=torch.rand(1,1,1,259,16)
        pose=torch.randn(1,1,7,requires_grad=True)
        predicted,_=model.rollout_shallow(observed,reference_shallow=observed,
            observed_pose_history=pose,observed_action_history=None,reference_pose=torch.zeros(1,1,7),
            observed_action_history_valid_mask=None,
            lang_feats=torch.rand(1,2,16),lang_padding_mask=torch.ones(1,2,dtype=torch.bool))
        predicted.square().sum().backward()
        assert pose.grad[...,5:].abs().sum()>0
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('raw',[[],np.zeros((0,6)),np.zeros((2,4)),np.full((1,6),np.nan)])
def test_invalid_history_rejected(raw):
    with pytest.raises(ValueError,match='finite nonempty'):
        episode_pose_progress(raw)
