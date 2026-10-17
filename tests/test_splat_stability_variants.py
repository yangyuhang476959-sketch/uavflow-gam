import torch

from scripts.splat_stability_variants import log_kernel_splat,alpha_point_splat


def test_log_kernel_matches_normalized_bilinear_and_local_derivatives():
    rgb=torch.tensor([[[[.1,.9]],[[.2,.7]],[[.3,.6]]]],dtype=torch.float64)
    flow=torch.tensor([[[[.2,-.2]],[[.1,.1]]]],dtype=torch.float64,requires_grad=True)
    z=torch.ones(1,1,1,2,dtype=torch.float64)
    result=log_kernel_splat(rgb,flow,z,epsilon=0)
    # Both points at x=.2/.8, hence mass=.9*(.8+.2)=.9 at each target.
    expected=torch.stack((.8*rgb[0,:,0,0]+.2*rgb[0,:,0,1],
                          .2*rgb[0,:,0,0]+.8*rgb[0,:,0,1]),dim=-1)[None,:,None]
    torch.testing.assert_close(result,expected)
    assert torch.autograd.gradcheck(lambda f:log_kernel_splat(rgb,f,z,epsilon=0),flow)


def test_log_kernel_normalization_does_not_bound_true_low_mass_derivative():
    rgb=torch.tensor([[[[0.,1.]]]],dtype=torch.float64)
    # Both source points only barely reach target x=1, but by different amounts.
    flow=torch.tensor([[[[.0001,-.9998]],[[0.,0.]]]],dtype=torch.float64,requires_grad=True)
    result=log_kernel_splat(rgb,flow,torch.ones(1,1,1,2,dtype=torch.float64),epsilon=0)
    gradient=torch.autograd.grad(result[0,0,0,1],flow)[0]
    assert gradient.abs().max()>1000


def test_alpha_single_point_weak_coverage_stays_weak_and_keeps_flow_gradient():
    rgb=torch.tensor([[[[1.]]]],dtype=torch.float64)
    flow=torch.tensor([[[[.9]],[[0.]]]],dtype=torch.float64,requires_grad=True)
    result=alpha_point_splat(rgb,flow,torch.ones(1,1,1,1,dtype=torch.float64),radius=1.)
    torch.testing.assert_close(result,torch.full_like(result,.19))
    gradient=torch.autograd.grad(result.sum(),flow)[0]
    torch.testing.assert_close(gradient[0,0,0,0],torch.tensor(-1.8,dtype=torch.float64))


def test_alpha_uses_depth_order_and_does_not_amplify_colors():
    rgb=torch.tensor([[[[.2,.8]]]],dtype=torch.float64)
    flow=torch.tensor([[[[0.,-1.]],[[0.,0.]]]],dtype=torch.float64)
    # Both points coincide at target 0, both fully opaque. Nearer point wins.
    z=torch.tensor([[[[2.,1.]]]],dtype=torch.float64)
    result=alpha_point_splat(rgb,flow,z,radius=.9)
    torch.testing.assert_close(result[0,0,0,0],torch.tensor(.8,dtype=torch.float64))
    torch.testing.assert_close(result[0,0,0,1],torch.tensor(0.,dtype=torch.float64))
