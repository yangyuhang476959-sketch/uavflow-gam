"""Explicit epoch-boundary migration from action/depth Warp to joint Stop."""
import copy
import torch
import torch.nn.functional as F


STOP_MIGRATION_PATHS={
    'model.stop_head_enabled','model.stop_head_mode','model.stop_model_dim',
    'training.stop_head_lr','model.stop_init_file',
    'training.joint_stop_start_epoch','training.joint_stop_start_step',
    'training.joint_stop_terminal_repeat_count',
}


def validate_stop_activation(checkpoint,current_cfg):
    old=checkpoint['config'];new=current_cfg
    if old['model'].get('stop_head_enabled',False) or checkpoint.get('stop_head') is not None:
        raise ValueError('Stop activation requires an action/depth-only checkpoint')
    if not old['model'].get('dual_stream_rgbwarp_enabled') or not new['model'].get('dual_stream_rgbwarp_enabled'):
        raise ValueError('Stop activation is restricted to RGB Warp')
    if not new['model'].get('stop_head_enabled') or new['model'].get('stop_head_mode')!='action_hidden_pose':
        raise ValueError('Expected the existing action-hidden/pose MLP Stop')
    if not new['stage1'].get('qwen_pose5_enabled'):
        raise ValueError('Joint Warp Stop requires the same canonical5D Pose')
    if new['dataset'].get('stop_target_mode')!='current' or new['dataset'].get('stop_soft_target_fraction',0)!=0:
        raise ValueError('Expected hard current-frame Stop targets')
    boundary=int(new['training'].get('joint_stop_start_epoch',0));world=int(checkpoint['world_size'])
    batch=int(new['training']['batch_size']);accum=int(new['training'].get('gradient_accumulation_steps',1))
    import math
    updates=math.ceil(checkpoint['train_size']/world)//batch//accum
    if boundary!=10 or int(checkpoint['step'])!=updates*boundary:
        raise ValueError('Activate Stop only at the exact end of epoch10')
    if (int(checkpoint['epoch']),int(checkpoint['batch'])) not in ((boundary,0),(boundary-1,updates)):
        raise ValueError('Stop activation cursor is not an epoch boundary')
    if int(old['training']['max_epochs']) not in (10,20) or int(new['training']['max_epochs'])!=20 or float(new['training']['stop_head_lr'])<=0 or float(new['loss']['stop_weight'])<=0:
        raise ValueError('Expected10 action/depth epochs followed by10 joint epochs')
    # Every inherited setting is immutable except the explicitly new Stop fields.
    allowed=STOP_MIGRATION_PATHS|{'loss.stop_weight','training.joint_stop_start_epoch','training.max_epochs','training.results_dir','training.joint_stop_start_step','training.joint_stop_terminal_repeat_count'}
    repeat=int(new['training'].get('joint_stop_terminal_repeat_count',0))
    if repeat not in (0,5):raise ValueError('Only the Pure-matched5 Stop-only terminal pairs are supported')
    if repeat and int(new['training'].get('joint_stop_start_step',0))!=int(checkpoint['step']):raise ValueError('Stop phase start step must match the source checkpoint')
    if repeat and not new['model'].get('stop_init_file'):
        raise ValueError('Pure-matched Stop repeats require the shared Stop initialization')
    def leaves(value,prefix=''):
        result={}
        for key,item in value.items():
            path=prefix+'.'+key if prefix else key
            if isinstance(item,dict):result.update(leaves(item,path))
            else:result[path]=item
        return result
    before,after=leaves(old),leaves(new)
    changed=[k for k in before.keys()|after.keys() if before.get(k)!=after.get(k) and k not in allowed]
    if changed:raise ValueError('Stop activation cannot change inherited settings: '+', '.join(sorted(changed)))
    return dict(step=int(checkpoint['step']),epoch=boundary,batch=0,stage='joint_action_depth_stop')


def load_optimizer_with_new_stop_group(optimizer,saved):
    current=optimizer.state_dict();groups=saved['param_groups']
    if len(current['param_groups'])!=len(groups)+1:
        raise ValueError('Exactly one new Stop optimizer group is required')
    for old,new in zip(groups,current['param_groups']):
        if old['params']!=new['params']:
            raise ValueError('Inherited optimizer parameter ordering changed')
    state=dict(state=saved['state'],param_groups=[*groups,current['param_groups'][-1]])
    optimizer.load_state_dict(state)


def stop_values(logits,targets,pos_weight=5.):
    targets=targets.float()
    if logits.shape!=targets.shape or not ((targets==0)|(targets==1)).all():
        raise ValueError('Stop requires shape-matched hard0/1 current-frame labels')
    loss=F.binary_cross_entropy_with_logits(logits.float(),targets,pos_weight=logits.new_tensor(float(pos_weight)))
    p=logits.detach().float().sigmoid();positive=targets.ge(.5);pred=p.ge(.5)
    pc=positive.float().sum();nc=(~positive).float().sum();ps=p[positive].sum();ns=p[~positive].sum()
    counts=torch.stack(((pred&positive).sum(),(pred&~positive).sum(),(~pred&positive).sum(),(~pred&~positive).sum())).to(p)
    stats=torch.cat((torch.stack((loss.detach()*targets.numel(),p.new_tensor(targets.numel()))),counts,torch.stack((ps,ns))))
    return dict(stop=loss,stop_probabilities=p,stop_targets=targets.detach(),_global_log_stop_stats=stats,
                stop_accuracy=(pred==positive).float().mean(),
                stop_recall=(pred&positive).float().sum()/pc.clamp_min(1),stop_rate=positive.float().mean(),
                stop_positive_probability_sum=ps,stop_negative_probability_sum=ns,
                stop_positive_count=pc,stop_negative_count=nc,
                stop_positive_probability=ps/pc.clamp_min(1),stop_negative_probability=ns/nc.clamp_min(1))

class StopOnlyTerminalRepeats(torch.utils.data.Dataset):
    """Pure-matched sample order; added terminal pairs supervise Stop only."""
    def __init__(self, source, count=5):
        from experiments.uavflow_pure_stop_probe.data import variant_view
        if count != 5:raise ValueError('Matched Stop recipe requires exactly5 terminal pairs')
        self.native_size=len(source)
        self.source=variant_view(source,2,'A',count)
        self.added_windows=len(self.source)-self.native_size
    def __len__(self):return len(self.source)
    def __getattr__(self,name):
        source=self.__dict__.get('source')
        if source is None:raise AttributeError(name)
        return getattr(source,name)
    def __getitem__(self,index):
        row=self.source[index]
        if index>=self.native_size:
            if not bool(row['is_self_pair']):raise AssertionError('Expected artificial terminal pair')
            row['action_loss_mask']=torch.zeros_like(row['action_loss_mask'])
            for key in ['gt_depth_mask','gt_depth_semantic_mask']:
                if key in row:row[key]=torch.zeros_like(row[key])
            if not torch.all(row['stop_target']==1):raise AssertionError('Terminal Stop pair must be hard positive')
        return row


def joint_step_budget(config,steps_per_epoch):
    t=config['training'];epochs=int(t['max_epochs'])
    if int(t.get('joint_stop_terminal_repeat_count',0)):
        if epochs!=20 or int(t.get('joint_stop_start_epoch',0))!=10 or int(t.get('joint_stop_start_step',0))!=128810:
            raise ValueError('Expected the matched10+10 Stop schedule')
        return 128810+(epochs-10)*steps_per_epoch
    return epochs*steps_per_epoch if epochs>0 else int(t['max_steps'])
