import pytest
import torch

from experiments.uavflow_predictor_idm.alpha_warp import composite_alpha_points
from experiments.uavflow_predictor_idm.rgbwarp_geometry import forward_alpha_splat,action_camera_transform
from experiments.uavflow_predictor_idm.patch_transport import nominal_intrinsics
from scripts.compare_official_softsplat_small import project_flow


def test_training_projection_matches_tested_probe_rgb_and_action_gradients():
    torch.manual_seed(42)
    rgb=torch.rand(1,3,28,28);depth=torch.rand(1,28,28)+2
    valid=torch.ones_like(depth,dtype=torch.bool);valid[:,:2]=False
    actions=torch.randn(1,5,4)*.01;actions.requires_grad_()
    k=nominal_intrinsics(28,28)
    production=forward_alpha_splat(rgb,depth,action_camera_transform(actions),k,valid=valid)
    flow,z=project_flow(depth,valid,actions,k)
    reference=composite_alpha_points(rgb,flow,z,radius=1.5,points_per_pixel=8)
    torch.testing.assert_close(production['rgb'],reference['rgb'])
    torch.testing.assert_close(production['opacity'],reference['opacity'])
    signs=torch.randn_like(rgb)
    a=torch.autograd.grad((production['rgb']*signs).sum(),actions,retain_graph=True)[0]
    b=torch.autograd.grad((reference['rgb']*signs).sum(),actions)[0]
    torch.testing.assert_close(a,b)
    assert a.abs().sum()>0 and torch.isfinite(a).all()


def test_empty_and_invalid_sources_are_black_with_no_correspondence():
    rgb=torch.rand(2,3,28,28)
    depth=torch.zeros(2,28,28,requires_grad=True)
    actions=torch.zeros(2,5,4,requires_grad=True)
    output=forward_alpha_splat(rgb,depth,action_camera_transform(actions),nominal_intrinsics(28,28))
    assert output['rgb'].eq(0).all() and output['opacity'].eq(0).all()
    assert not output['valid'].any() and not output['correspondence'].any()
    grads=torch.autograd.grad(output['rgb'].sum(),(actions,depth))
    assert all(torch.isfinite(g).all() and g.eq(0).all() for g in grads)


def test_partial_opacity_does_not_normalize_weak_color_and_stays_differentiable():
    rgb=torch.ones(1,3,1,1)
    flow=torch.tensor([[[[.99]],[[0.]]]],requires_grad=True)
    result=composite_alpha_points(rgb,flow,torch.ones(1,1,1,1),radius=1.,patch_size=1)
    torch.testing.assert_close(result['rgb'],torch.full_like(rgb,1-.99**2))
    torch.testing.assert_close(result['opacity'],result['rgb'][:,:1])
    assert 0<float(result['opacity'].detach())<.1
    assert result['valid'].all() and result['correspondence'].all()
    gradient=torch.autograd.grad(result['rgb'].sum(),flow)[0]
    assert gradient.abs().max()<6


def test_correspondence_tracks_alpha_footprint_and_is_batch_isolated():
    rgb=torch.ones(2,3,28,28);depth=torch.zeros(2,28,28)
    depth[0,5,13]=2;depth[1,5,14]=2
    output=forward_alpha_splat(rgb,depth,torch.eye(4).expand(2,-1,-1),nominal_intrinsics(28,28))
    corr=output['correspondence']
    assert corr[0,1,0] and not corr[0,:,1:].any()
    assert corr[1,0,1] and not corr[1,:,0].any()
    assert not corr.requires_grad and not output['valid'].requires_grad


def test_fully_occluded_point_is_not_a_correspondence_contributor():
    rgb=torch.zeros(1,3,14,14)
    rgb[0,:,0,0]=.2;rgb[0,:,0,13]=.8
    flow=torch.full((1,2,14,14),-1000.)
    flow[0,:,0,0]=torch.tensor([0.,0.]);flow[0,:,0,13]=torch.tensor([-13.,0.])
    z=torch.ones(1,1,14,14)*5;z[0,0,0,0]=2
    result=composite_alpha_points(rgb,flow,z,patch_size=1)
    torch.testing.assert_close(result['rgb'][0,:,0,0],torch.full((3,),.2))
    assert result['correspondence'][0,0,0] and not result['correspondence'][0,0,13]


def test_alpha_projection_is_fp32_under_outer_amp():
    rgb=torch.rand(1,3,28,28);depth=torch.ones(1,28,28)*2
    with torch.autocast('cpu',dtype=torch.bfloat16):
        output=forward_alpha_splat(rgb,depth,torch.eye(4)[None],nominal_intrinsics(28,28))
    assert output['rgb'].dtype==torch.float32 and output['opacity'].dtype==torch.float32


@pytest.mark.parametrize('radius,k',[(0.,8),(float('nan'),8),(1.5,0)])
def test_invalid_alpha_settings_rejected(radius,k):
    with pytest.raises(ValueError,match='Positive'):
        composite_alpha_points(torch.ones(1,3,1,1),torch.zeros(1,2,1,1),torch.ones(1,1,1,1),radius=radius,points_per_pixel=k)
