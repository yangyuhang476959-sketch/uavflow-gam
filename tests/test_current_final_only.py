import torch
from test_rgbwarp_correction import make_model
from test_paper_current_only import objective_fixture

def test_no_proposal_forward_final_current_gradients(monkeypatch):
 torch.set_num_threads(1)
 net=make_model(corr=False,mask=False,output_mode='direct',branch_mode='current_only',current_final_only=True)
 def forbidden(*a,**kw):raise AssertionError('Unused proposal head called')
 monkeypatch.setattr(net.parallel_action_head,'forward',forbidden)
 assert not any(p.requires_grad for p in net.parallel_action_head.parameters())
 slots=torch.rand(2,5,16,requires_grad=True)
 out=net(torch.rand(2,3,224,224),reference_shallow=None,lang_feats=slots,lang_padding_mask=None,action_denormalizer=lambda x:x)
 assert out['direct_actions_norm'] is None and out['deep_joint_features'] is None
 assert net.da3.dpt_head.counts==[256]
 (out['actions_norm'].square().mean()+out['current_depth_output']['depth'].square().mean()).backward()
 assert slots.grad.abs().sum()>0
 assert not [n for n,p in net.named_parameters() if p.requires_grad and p.grad is None]

def test_final_only_loss_and_global_logging(monkeypatch):
 obj,net,batch,args,_=objective_fixture(monkeypatch)
 net.rgbwarp_correction.config['current_final_only']=True;net.parallel_action_head.requires_grad_(False)
 values=obj.forward_rgbwarp_batch(**args)
 torch.testing.assert_close(values['action'],values['refine_action'])
 assert values['direct_action']==0
 from experiments.uavflow_predictor_idm.global_loss_logging import GlobalLossWindow
 log=GlobalLossWindow(current_only=True,final_only=True);loss=3*values['action']+3*values['depth'];log.add(values,loss);report,_,_=log.finish()
 torch.testing.assert_close(report['action'],values['refine_action'].double())
 assert 'rgbwarp_proposal_loss' not in report
 assert not any(k.startswith('depth_future_') for k in report)
