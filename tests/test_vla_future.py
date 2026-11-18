import torch

from test_rgbwarp_correction import make_model
from test_paper_current_only import objective_fixture


def test_future_branch_predicts_actions_and_future_depth(monkeypatch):
    torch.set_num_threads(1)
    model = make_model(corr=False, mask=False, output_mode='direct', branch_mode='future_only', current_final_only=True)
    slots = torch.rand(2, 5, 16, requires_grad=True)
    output = model(torch.rand(2, 3, 224, 224), reference_shallow=None,
                   lang_feats=slots, lang_padding_mask=None, action_denormalizer=lambda x: x)
    assert output['current_depth_output'] is None
    assert output['direct_actions_norm'] is None
    (output['actions_norm'].square().mean() + output['deep_joint_features']['depth'].square().mean()).backward()
    assert slots.grad.abs().sum() > 0
    assert not [name for name, p in model.named_parameters() if p.requires_grad and p.grad is None]


def test_future_objective_uses_future_depth(monkeypatch):
    obj, model, batch, args, _ = objective_fixture(monkeypatch)
    model.rgbwarp_correction.config['branch_mode'] = 'future_only'
    model.rgbwarp_correction.config['current_final_only'] = True
    model.parallel_action_head.requires_grad_(False)
    batch['gt_depth_meters'][:, 0] = 10
    batch['gt_depth_meters'][:, 1] = 40
    values = obj.forward_rgbwarp_batch(**args)
    assert 'depth_future_total' in values
    assert 'depth_current_total' not in values
    before = values['depth'].detach().clone()
    batch['all_view_images'][:, 1].fill_(float('nan'))
    batch['gt_depth_meters'][:, 0] = 80
    torch.testing.assert_close(obj.forward_rgbwarp_batch(**args)['depth'], before)
    batch['gt_depth_meters'][:, 1] = 70
    assert not torch.isclose(obj.forward_rgbwarp_batch(**args)['depth'], before)
