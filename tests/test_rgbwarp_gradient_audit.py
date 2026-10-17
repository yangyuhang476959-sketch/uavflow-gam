import torch

from experiments.uavflow_predictor_idm.rgbwarp_geometry import action_camera_transform,forward_splat


def test_exact_integer_identity_can_have_zero_action_gradient_despite_textured_rgb():
    rgb=torch.rand(1,3,28,28);depth=torch.ones(1,28,28)*2
    actions=torch.zeros(1,5,4,requires_grad=True)
    result=forward_splat(rgb,depth,action_camera_transform(actions),torch.eye(3),return_diagnostics=True)
    torch.testing.assert_close(result['rgb'],rgb)
    assert result['diagnostics']['contributors'].eq(1).all()
    assert result['diagnostics']['coordinate_jacobian_norm'].eq(0).all()
    gradient=torch.autograd.grad((result['rgb']*torch.randn_like(rgb)).sum(),actions)[0]
    assert gradient.eq(0).all()


def test_fractional_shift_recovers_texture_gradient_and_diagnostic_is_opt_in():
    rgb=torch.rand(1,3,28,28);depth=torch.ones(1,28,28)*2
    actions=torch.zeros(1,5,4);actions[0,-1,1]=.4;actions.requires_grad_()
    transform=action_camera_transform(actions)
    result=forward_splat(rgb,depth,transform,torch.eye(3),return_diagnostics=True)
    plain=forward_splat(rgb,depth,transform,torch.eye(3))
    assert 'diagnostics' not in plain
    for key in plain:
        torch.testing.assert_close(plain[key],result[key])
    assert result['diagnostics']['contributors'].max()==2
    assert (result['diagnostics']['coordinate_jacobian_norm']>0).float().mean()>.9
    gradient=torch.autograd.grad((result['rgb']*torch.randn_like(rgb)).sum(),actions)[0]
    assert gradient.abs().sum()>0
    flat=forward_splat(torch.ones_like(rgb)*.5,depth,transform,torch.eye(3),return_diagnostics=True)
    assert flat['diagnostics']['coordinate_jacobian_norm'].eq(0).all()


def test_official_comparison_flow_has_same_camera_axes_and_invalid_sources_outside():
    from scripts.compare_official_softsplat_small import project_flow
    from experiments.uavflow_predictor_idm.patch_transport import nominal_intrinsics
    depth=torch.ones(1,28,28)*2;valid=torch.ones_like(depth,dtype=torch.bool)
    valid[:,0,0]=False
    actions=torch.zeros(1,5,4);actions[0,-1,1]=.4;actions.requires_grad_()
    flow,z=project_flow(depth,valid,actions,nominal_intrinsics(28,28))
    torch.testing.assert_close(flow[0,0,1:,1:],torch.full((27,27),-2.8),atol=1e-5,rtol=1e-5)
    torch.testing.assert_close(flow[0,1,1:,1:],torch.zeros(27,27),atol=1e-5,rtol=1e-5)
    assert flow[0,:,0,0].eq(-10000).all()
    assert torch.isfinite(flow).all() and z[0,0,1,1]==2
    gradient=torch.autograd.grad(flow[0,0,1,1],actions)[0]
    assert gradient[0,-1,1]==-7
