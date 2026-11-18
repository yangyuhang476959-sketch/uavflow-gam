"""Strict, CPU-light contracts for action-to-joint continuation."""
def validate_start_joint(old,new,epoch,batch,stage_steps,step,updates_per_epoch):
    allowed={'action_only','joint_epochs','matched_joint_stop','common_stop_init_file','self_pair_end_count'}
    if any(old.get(k)!=new.get(k) for k in set(old)|set(new) if k not in allowed):
        raise ValueError('Joint transition cannot change action budget, data, initialization, LR, batch, Pose or architecture')
    if not old.get('action_only') or old.get('joint_epochs')!=0 or new.get('action_only') or new.get('joint_epochs',0)<1 or not new.get('matched_joint_stop'):
        raise ValueError('Expected action-only -> explicitly enabled matched joint Stop')
    if not new.get('matched_warp_config') or old.get('self_pair_end_count')!=0 or new.get('self_pair_end_count')!=5:
        raise ValueError('This transition adds exactly5 Stop-only terminal self-pairs; action samples are retained')
    expected=old['action_epochs']*updates_per_epoch
    if epoch!=old['action_epochs'] or batch!=0 or list(stage_steps)!=[expected,0] or step!=expected:
        raise ValueError('Joint transition requires completed action epoch boundary and exact cursor')
    if new['scheduler']!='constant' or new['warmup_ratio']!=0 or new['min_lr_ratio']!=1:
        raise ValueError('Joint continuation requires unchanged constant LR')


def verify_optimizer_state(expected,actual):
    import torch
    if expected['param_groups']!=actual['param_groups'] or expected['state'].keys()!=actual['state'].keys():
        raise AssertionError('Optimizer groups or state IDs differ after restore')
    tensors=0
    for key,values in expected['state'].items():
        if values.keys()!=actual['state'][key].keys():raise AssertionError('Optimizer state fields differ')
        for name,value in values.items():
            restored=actual['state'][key][name]
            if isinstance(value,torch.Tensor):
                if not isinstance(restored,torch.Tensor) or not torch.equal(value.detach().cpu(),restored.detach().cpu()):raise AssertionError('Optimizer tensor not restored exactly')
                tensors+=1
            elif value!=restored:raise AssertionError('Optimizer scalar not restored exactly')
    return dict(states=len(expected['state']),tensors=tensors,exact=True)
