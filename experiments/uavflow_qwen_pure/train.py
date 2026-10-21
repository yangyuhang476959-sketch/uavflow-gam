"""Standalone Pure-Qwen five-action-slot training with evaluator yaw4d Pose Token."""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from experiments.uavflow_predictor_idm.eval_sampler import evaluation_sampler

from experiments.uavflow_predictor_idm.runtime import (
    amp_dtype, bootstrap_accelerator_plugin, configure_accelerator,
    distributed_info, manual_seed_all, optimizer_batches,
)
bootstrap_accelerator_plugin()

from experiments.uavflow_predictor_idm.metric_train import build_config as metric_config
from experiments.uavflow_qwen_pure.model import PureQwenActionPolicy
from experiments.uavflow_qwen_pure.utils import CursorSampler, atomic_save, masked_l1
from robot.data.dataset import ActionNormalizer, build_robot_dataset


def _cfg(max_trajectories=None):
    cfg=metric_config('gam')
    cfg.dataset.parquet_root=os.environ.get('UAVFLOW_SIM_ROOT',str(cfg.dataset.parquet_root))
    cfg.dataset.type='uavflow'
    cfg.dataset.dataset_name='uavflow_sim'
    cfg.dataset.action_stats_key='uavflow_sim'
    cfg.dataset.openvla_prompt_pose_mode='evaluator_yaw4d'
    cfg.dataset.pose_progress_enabled=True
    cfg.dataset.future_steps=1
    cfg.dataset.visual_anchor_stride=5
    cfg.dataset.chunk_size=5
    cfg.dataset.action_delta_mode='openvla_yaw_only'
    cfg.dataset.gt_depth_root=None
    cfg.dataset.gt_depth_required=False
    cfg.dataset.gt_depth_fallback_roots=[]
    cfg.dataset.episode_split_mode='stratified_instruction'
    if max_trajectories is not None: cfg.dataset.max_trajectories=int(max_trajectories)
    cfg.stage1.action_stats_dir=os.environ.get('ACTION_STATS_DIR',str(cfg.stage1.action_stats_dir))
    model_root=Path(os.environ.get('MODEL_ROOT','checkpoints'))
    cfg.stage1.qwen_model=os.environ.get('QWEN_MODEL',str(model_root/'qwen3.5-2b'))
    cfg.model.pose_mode='numeric_token'
    cfg.model.pose_progress_enabled=True
    cfg.stage1.qwen_numeric_pose_token=True
    cfg.stage1.qwen_pose_progress_enabled=True
    cfg.training.batch_size=4
    cfg.training.gradient_accumulation_steps=1
    cfg.training.max_epochs=10
    cfg.training.lr_schedule='constant'
    cfg.training.warmup_steps=0
    cfg.training.lora_lr=1e-5
    cfg.training.head_lr=5e-4
    cfg.training.state_lr=1e-5
    cfg.training.num_workers=4
    cfg.training.log_every=25
    cfg.training.save_every=1000
    cfg.training.eval_every=5000
    cfg.training.eval_every_epoch=True
    cfg.training.eval_max_batches=0
    cfg.training.results_dir=str(Path(os.environ.get('OUTPUT_ROOT','results/metric_release'))/'01_pure_qwen_yaw4d')
    return cfg


def _move_sample(batch,device):
    return {k:(v.to(device,non_blocking=True) if isinstance(v,torch.Tensor) else v) for k,v in batch.items()}


def _forward(model,batch,normalizer,device,amp_enabled,amp_type):
    batch=_move_sample(batch,device)
    images=batch['all_view_images'][:,:1,:1]
    instructions=list(batch['task_description'])
    reference=model.module if hasattr(model,'module') else model
    state_key='episode_pose_progress' if getattr(reference,'pose_progress_enabled',False) else 'openvla_prompt_pose'
    states=batch[state_key][:,0]
    raw=batch['actions'].float()
    keys=list(batch['action_stats_key'])
    target=normalizer.normalize(raw,stats_keys=keys)[:,0]
    mask=batch.get('action_loss_mask')
    mask=torch.ones_like(raw,dtype=torch.bool) if mask is None else mask.bool()
    mask=mask[:,0]
    with torch.autocast(device_type=device.type,dtype=amp_type,enabled=amp_enabled and device.type in {'cuda','npu'}):
        prediction=model(images,instructions,states)[:,0]
        loss=masked_l1(prediction,target,mask)
    return loss,prediction.detach(),target.detach(),mask


def _eval(model,loader,normalizer,device,amp_type,limit):
    model=model.module if hasattr(model,'module') else model
    model.eval(); model.encoder.qwen.eval()
    sums=torch.zeros(2,device=device)
    with torch.no_grad():
        for index,batch in enumerate(loader):
            if limit>0 and index>=limit:break
            loss,pred,target,mask=_forward(model,batch,normalizer,device,True,amp_type)
            sums[0]+=(pred.float()-target.float()).abs().mul(mask).sum()
            sums[1]+=mask.sum()
    if dist.is_initialized():dist.all_reduce(sums)
    model.train(); model.encoder.qwen.eval()
    return float((sums[0]/sums[1].clamp_min(1)).item())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume',default='')
    parser.add_argument('--max-trajectories',type=int)
    parser.add_argument('--print-config',action='store_true')
    parser.add_argument('--set',action='append',default=[])
    args=parser.parse_args()
    cfg=_cfg(args.max_trajectories)
    if args.set: cfg=OmegaConf.merge(cfg,OmegaConf.from_dotlist(args.set))
    if str(cfg.training.lr_schedule).lower()!='constant' or int(cfg.training.warmup_steps)!=0:
        raise ValueError('Pure-Qwen handoff uses constant LR and zero warmup; cosine is not supported here')
    accumulation_steps=int(cfg.training.get('gradient_accumulation_steps',1))
    if accumulation_steps < 1:
        raise ValueError('gradient_accumulation_steps must be positive')
    if args.print_config:
        print(OmegaConf.to_yaml(cfg,resolve=True)); return
    distributed,rank,local_rank,world=distributed_info()
    accelerator=configure_accelerator(local_rank=local_rank,distributed=distributed)
    device=accelerator.device
    seed=int(cfg.seed)+rank
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    manual_seed_all(accelerator,seed)

    dataset_cfg=OmegaConf.to_container(cfg.dataset,resolve=True)
    train_set=build_robot_dataset(dataset_cfg,is_eval=False)
    eval_set=build_robot_dataset(dataset_cfg,is_eval=True)
    train_set_size=len(train_set)
    norm_key=str(cfg.dataset.action_stats_key)
    normalizer=ActionNormalizer.from_stats_dir(str(cfg.stage1.action_stats_dir),[norm_key])
    model_root=Path(os.environ.get('MODEL_ROOT','checkpoints'))
    qwen_path=os.environ.get('QWEN_MODEL',str(model_root/'qwen3.5-2b'))
    pose_progress_enabled=bool(cfg.get('model',{}).get('pose_progress_enabled',False))
    model=PureQwenActionPolicy(qwen_path,pose_mode='numeric_token',pose_progress_enabled=pose_progress_enabled).to(device)
    model.encoder.qwen.eval()
    trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    optimizer_groups=[
        {'params':[p for p in model.encoder.parameters() if p.requires_grad],'lr':float(cfg.training.lora_lr)},
        {'params':list(model.projection.parameters())+list(model.head.parameters())+list(model.state_encoder.parameters()),'lr':float(cfg.training.head_lr)},
    ]
    if pose_progress_enabled:
        optimizer_groups[1]['params']=list(model.projection.parameters())+list(model.head.parameters())
        optimizer_groups.append({'params':list(model.state_encoder.parameters()),'lr':float(cfg.training.get('state_lr',cfg.training.lora_lr))})
    optimizer=torch.optim.AdamW(optimizer_groups,weight_decay=float(cfg.training.weight_decay))

    start_epoch=0;start_batch=0;step=0
    output=Path(str(cfg.training.results_dir));output.mkdir(parents=True,exist_ok=True)
    if args.resume:
        checkpoint=torch.load(args.resume,map_location='cpu',weights_only=False)
        if checkpoint.get('format')!='uavflow_pure_qwen_yaw4d_v1':
            raise ValueError('Resume expects a Pure-Qwen yaw4d numeric-token checkpoint')
        if checkpoint['train_size']!=train_set_size or checkpoint['world_size']!=world or checkpoint['batch_size']!=int(cfg.training.batch_size) or int(checkpoint.get('gradient_accumulation_steps',1))!=accumulation_steps:
            raise ValueError('Exact resume requires same train size, world size, per-device batch size and accumulation steps')
        if bool(checkpoint.get('pose_progress_enabled',False))!=pose_progress_enabled:
            raise ValueError('Cannot exactly resume between legacy 4D pose and 7D pose-progress; start a new run')
        model.load_trainable_state(checkpoint['trainable_state'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        step=int(checkpoint['step']);start_epoch=int(checkpoint['epoch']);start_batch=int(checkpoint['batch'])

    train_sampler=CursorSampler(train_set,num_replicas=world,rank=rank,shuffle=True,seed=int(cfg.seed),
        batch_size=int(cfg.training.batch_size)*accumulation_steps,resume_epoch=start_epoch,resume_batch=start_batch)
    val_sampler=evaluation_sampler(eval_set,world=world,rank=rank,
        max_batches=int(cfg.training.eval_max_batches),shuffle=False,seed=int(cfg.seed))
    workers=int(cfg.training.num_workers)
    train_loader=DataLoader(train_set,batch_size=int(cfg.training.batch_size),sampler=train_sampler,
        shuffle=False,num_workers=workers,pin_memory=accelerator.kind=='cuda',drop_last=False)
    eval_loader=DataLoader(eval_set,batch_size=int(cfg.training.batch_size),sampler=val_sampler,
        shuffle=False,num_workers=workers,pin_memory=accelerator.kind=='cuda',drop_last=False)
    if distributed:
        model=torch.nn.parallel.DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank,
            find_unused_parameters=False,gradient_as_bucket_view=True)
    raw_model=model.module if distributed else model
    amp_type=amp_dtype('bf16',device.type)
    if rank==0:
        (output/'resolved_config.yaml').write_text(OmegaConf.to_yaml(cfg,resolve=True))
        (output/'meta.json').write_text(json.dumps({'world_size':world,'per_device_batch':int(cfg.training.batch_size),
            'global_batch':world*int(cfg.training.batch_size)*accumulation_steps,'gradient_accumulation_steps':accumulation_steps,'train_size':len(train_set),'eval_size':len(eval_set),
            'trainable_parameters':trainable,'pose_mode':'numeric_token/evaluator_yaw4d','pose_progress_enabled':pose_progress_enabled,'pose_dimensions':7 if pose_progress_enabled else 4,
            'lr_schedule':'constant','warmup_steps':0},indent=2))
        print(f'Pure-Qwen yaw4d train={len(train_set)} eval={len(eval_set)} world={world} '
              f'per_device_batch={cfg.training.batch_size} accumulation={accumulation_steps} global_batch={world*int(cfg.training.batch_size)*accumulation_steps} trainable={trainable/1e6:.2f}M lr=constant '
              f'lora_lr={cfg.training.lora_lr} head_lr={cfg.training.head_lr} '
              f'pose_dimensions={7 if pose_progress_enabled else 4} '
              f'state_lr={cfg.training.get("state_lr",cfg.training.lora_lr) if pose_progress_enabled else cfg.training.head_lr}',flush=True)

    max_epochs=int(cfg.training.max_epochs)
    last_eval_step=None
    last_eval_value=None
    def validate(epoch_number):
        nonlocal last_eval_step,last_eval_value
        if last_eval_step==step:
            return last_eval_value
        val=_eval(model,eval_loader,normalizer,device,amp_type,int(cfg.training.eval_max_batches))
        last_eval_step,last_eval_value=step,val
        if rank==0:
            with (output/'train.log').open('a') as log:
                log.write(f'epoch={epoch_number} step={step} val_action_l1={val:.8f}\n')
            print(f'[step={step:07d}] epoch={epoch_number} val_action_l1={val:.6f} '
                  f'full_validation={int(cfg.training.eval_max_batches)<=0} '
                  f'validation_windows={len(eval_set) if int(cfg.training.eval_max_batches)<=0 else "bounded"}',flush=True)
        return val
    for epoch in range(start_epoch,max_epochs):
        train_sampler.set_epoch(epoch)
        if hasattr(train_set,'set_epoch'):train_set.set_epoch(epoch)
        raw_model.train();raw_model.encoder.qwen.eval()
        logical_batch=start_batch if epoch==start_epoch else 0
        running=0.0
        running_updates=0
        for microbatches in optimizer_batches(train_loader,accumulation_steps,drop_incomplete=False):
            optimizer.zero_grad(set_to_none=True)
            # Preserve Pure's partial last batch and weight unequal microbatches
            # by their actual valid action-element counts.
            counts=[int(b['action_loss_mask'][:,0].sum()) if 'action_loss_mask' in b
                    else int(b['actions'][:,0].numel()) for b in microbatches]
            denominator=max(1,sum(counts))
            average=None
            for batch,count in zip(microbatches,counts):
                micro_loss,_,_,_=_forward(model,batch,normalizer,device,True,amp_type)
                weighted=micro_loss*(count/denominator)
                weighted.backward()
                average=weighted.detach() if average is None else average+weighted.detach()
            loss=average
            running_updates+=1
            torch.nn.utils.clip_grad_norm_([p for p in raw_model.parameters() if p.requires_grad],1.0)
            optimizer.step();step+=1;logical_batch+=1;running+=float(loss.detach())
            if rank==0 and step%int(cfg.training.log_every)==0:
                print(f'[step={step:07d}] epoch={epoch+1}/{max_epochs} action={float(loss.detach()):.6f} '
                      f'epoch_mean_so_far={running/max(1,running_updates):.6f}',flush=True)
            eval_every=int(cfg.training.get('eval_every',0))
            if eval_every>0 and step%eval_every==0:
                validate(epoch+1)
            if step%int(cfg.training.save_every)==0:
                if rank==0:
                    atomic_save({'format':'uavflow_pure_qwen_yaw4d_v1','trainable_state':raw_model.trainable_state(),
                        'optimizer':optimizer.state_dict(),'step':step,'epoch':epoch,'batch':logical_batch,
                        'world_size':world,'batch_size':int(cfg.training.batch_size),'gradient_accumulation_steps':accumulation_steps,'pose_progress_enabled':pose_progress_enabled,'train_size':train_set_size,
                        'action_normalizer':normalizer.state_dict(),'config':OmegaConf.to_container(cfg,resolve=True)},output/'last.pt')
                if distributed:dist.barrier()
        val=validate(epoch+1)
        if rank==0:
            atomic_save({'format':'uavflow_pure_qwen_yaw4d_v1','trainable_state':raw_model.trainable_state(),
                'optimizer':optimizer.state_dict(),'step':step,'epoch':epoch+1,'batch':0,
                'world_size':world,'batch_size':int(cfg.training.batch_size),'gradient_accumulation_steps':accumulation_steps,'pose_progress_enabled':pose_progress_enabled,'train_size':train_set_size,
                'action_normalizer':normalizer.state_dict(),'config':OmegaConf.to_container(cfg,resolve=True),
                'val_action_l1':val},output/f'epoch_{epoch+1:02d}.pt')
            atomic_save({'format':'uavflow_pure_qwen_yaw4d_v1','trainable_state':raw_model.trainable_state(),
                'optimizer':optimizer.state_dict(),'step':step,'epoch':epoch+1,'batch':0,
                'world_size':world,'batch_size':int(cfg.training.batch_size),'gradient_accumulation_steps':accumulation_steps,'pose_progress_enabled':pose_progress_enabled,'train_size':train_set_size,
                'action_normalizer':normalizer.state_dict(),'config':OmegaConf.to_container(cfg,resolve=True),
                'val_action_l1':val},output/'last.pt')
        if distributed:dist.barrier()
        start_batch=0
    if distributed:dist.destroy_process_group()


if __name__=='__main__':main()
