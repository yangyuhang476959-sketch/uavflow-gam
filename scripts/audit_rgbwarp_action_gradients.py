"""CPU audit of the EXACT training splat: geometric coverage is not gradient coverage.

Uses observed RGB/GT current depth and GT/zero/tiny-jitter five-step actions.
Random VJPs estimate RGB Jacobian column RMS, not a DA3 depth-loss gradient.
No weights, UE, GPU, training or dataset mutations are needed.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from robot.data.dataset import build_robot_dataset
from robot.data.uavflow_dataset import _instruction_task_class
from experiments.uavflow_predictor_idm.patch_transport import nominal_intrinsics
from experiments.uavflow_predictor_idm.rgbwarp_geometry import action_camera_transform,forward_splat


def measure(rgb,depth,valid,actions,*,probes=4,seed=42):
    generator=torch.Generator().manual_seed(seed)
    value=torch.as_tensor(actions,dtype=torch.float32)[None].clone().requires_grad_(True)
    intrinsics=nominal_intrinsics(*depth.shape[-2:])
    def warp(a,diagnostics=False):
        return forward_splat(rgb,depth,action_camera_transform(a),intrinsics,
                             valid=valid,return_diagnostics=diagnostics)
    output=warp(value,True)
    mask=output['valid'][:,0]
    diagnostic=output['diagnostics'];count=diagnostic['contributors']
    jacobian=diagnostic['coordinate_jacobian_norm']
    n=int(mask.sum());total=mask.numel()
    gradients=[]
    for _ in range(probes):
        signs=torch.randint(0,2,output['rgb'].shape,generator=generator).float()*2-1
        loss=(output['rgb']*signs).sum()/np.sqrt(output['rgb'].numel())
        gradients.append(torch.autograd.grad(loss,value,retain_graph=True)[0].detach()[0])
    # E[(J^T r)^2]/pixels estimates per-action RGB Jacobian column mean square.
    rms=torch.stack(gradients).square().mean(0).sqrt()
    scale=value.new_tensor([.01,.01,.01,np.deg2rad(1.)])
    direction=(torch.randint(0,2,value.shape,generator=generator).float()*2-1)*scale
    signs=torch.randint(0,2,output['rgb'].shape,generator=generator).float()*2-1
    scalar=(output['rgb']*signs).sum()/np.sqrt(output['rgb'].numel())
    ad=float((torch.autograd.grad(scalar,value)[0]*direction).sum())
    with torch.no_grad():
        epsilon=.01
        plus=warp(value+epsilon*direction);minus=warp(value-epsilon*direction)
        fd=float(((plus['rgb']-minus['rgb'])*signs).sum()/np.sqrt(output['rgb'].numel())/(2*epsilon))
        topology_change=float((plus['valid']!=minus['valid']).float().mean())
    return dict(coverage=n/total,
        tiny_mass_fraction_of_covered=float(((output['mass']<.001)&mask).sum())/max(n,1),
        singleton_fraction_of_covered=float(((count==1)&mask).sum())/max(n,1),
        multi_source_fraction_of_covered=float(((count>=2)&mask).sum())/max(n,1),
        active_coordinate_fraction_of_covered=float(((jacobian>1e-6)&mask).sum())/max(n,1),
        active_coordinate_fraction_of_image=float((jacobian>1e-6).sum())/total,
        rgb_action_jacobian_column_rms=rms.tolist(),
        rgb_action_jacobian_scaled_column_rms=(rms*scale).tolist(),
        any_action_vjp_nonzero=bool(rms.max()>1e-7),
        autograd_random_direction=ad,finite_difference_random_direction=fd,
        finite_difference_coverage_change=topology_change),output


def measure_da3(model,rgb,depth,valid,actions,future_depth,future_valid):
    """Isolate actual frozen DA3 depth-loss -> warped RGB -> numeric actions.

    No Qwen, token-value shortcut, trained LoRA or GT scale fit. This is an
    autograd-path check using pretrained DA3, not a trained policy evaluation.
    """
    from experiments.uavflow_predictor_idm.support_aware_transport import decode_levels
    from experiments.uavflow_predictor_idm.rgbwarp_objectives import depth_loss
    device=next(model.parameters()).device
    rgb,depth,valid,future_depth,future_valid=[v.to(device) for v in (rgb,depth,valid,future_depth,future_valid)]
    value=torch.as_tensor(actions,dtype=torch.float32,device=device)[None].clone().requires_grad_(True)
    warp=forward_splat(rgb,depth,action_camera_transform(value),nominal_intrinsics(*depth.shape[-2:],device=device),valid=valid)
    with torch.autocast(device_type=device.type,dtype=torch.bfloat16,enabled=device.type=='cuda'):
        shallow=model.encode_shallow_rgb_grad(warp['rgb'],gradient_checkpointing=True)
        deep=model.propagate_shallow_visual_slots_grad(shallow,gradient_checkpointing=True)
        prediction=decode_levels(model,deep['deep_levels'],1+model.num_register_tokens)['depth'].reshape_as(future_depth)
        loss=depth_loss(prediction,future_depth,future_valid,100.)['total']
    image_gradient,action_gradient=torch.autograd.grad(loss,(warp['rgb'],value),retain_graph=True)
    tiny=warp['mass'].detach()<.001
    tiny_loss=(warp['rgb']*image_gradient.detach()*tiny[:,None]).sum()
    tiny_grad=torch.autograd.grad(tiny_loss,value,retain_graph=True)[0]
    other_grad=torch.autograd.grad((warp['rgb']*image_gradient.detach()*(~tiny)[:,None]).sum(),value)[0]
    return dict(loss=float(loss.detach()),image_gradient_rms=float(image_gradient.square().mean().sqrt()),
        action_gradient=action_gradient.detach()[0].cpu().tolist(),
        tiny_mass_action_gradient=tiny_grad.detach()[0].cpu().tolist(),
        other_mass_action_gradient=other_grad.detach()[0].cpu().tolist(),
        nonzero=bool(action_gradient.detach().abs().max()>1e-10),
        finite=bool(torch.isfinite(action_gradient).all()),
        note='Frozen pretrained DA3; GT current depth; fixed future GT/100; no trained correction/Qwen.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset-config',type=Path,required=True)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--limit',type=int,default=100)
    parser.add_argument('--random-probes',type=int,default=4)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--threads',type=int,default=2)
    parser.add_argument('--da3-checkpoint',type=Path)
    parser.add_argument('--da3-device',default='cuda:0')
    parser.add_argument('--da3-memory-budget-gib',type=float,default=8.)
    args=parser.parse_args()
    if min(args.limit,args.random_probes,args.threads)<1:
        parser.error('limit, random-probes and threads must be positive')
    torch.set_num_threads(args.threads)
    selected=json.loads(args.selection.read_text())
    pairs=(selected['pairs'] if isinstance(selected,dict) else selected)[:args.limit]
    cfg=dict(OmegaConf.to_container(OmegaConf.load(args.dataset_config).dataset,resolve=True))
    cfg.pop('split_file',None)
    cfg.update(include_episodes=sorted({p['episode_id'] for p in pairs}),max_trajectories=None,
        pose_progress_enabled=False,future_steps=1,chunk_size=5,visual_anchor_stride=5)
    dataset=build_robot_dataset(cfg,is_eval=False)
    args.output.mkdir(parents=True,exist_ok=True)
    da3=None;da3_tasks=set()
    if args.da3_checkpoint:
        device=torch.device(args.da3_device)
        if device.type=='cuda':
            free,total=torch.cuda.mem_get_info(device)
            budget=min(args.da3_memory_budget_gib*1024**3,free-2*1024**3)
            if budget<5*1024**3:
                raise RuntimeError('Less than 5 GiB safe DA3 budget after reserving 2 GiB; no GPU probe started')
            torch.cuda.set_per_process_memory_fraction(budget/total,device)
        from robot.modeling.da3_giant_encoder import DA3GiantEncoder
        da3=DA3GiantEncoder(ckpt_path=str(args.da3_checkpoint),encoder_input_size=224,
            freeze_backbone=True,n_action_steps=0,views_per_timestep=1,use_temporal_embed=False)
        da3.requires_grad_(False);da3=da3.to(device).eval()
    rows=[];grouped=defaultdict(list)
    for number,pair in enumerate(pairs):
        tid=pair['episode_id'];start=int(pair['start']);end=int(pair['future'])
        loc=dataset.traj[tid]
        if end!=start+5 or any(loc[start+i][0]!=loc[start][0]+i for i in range(6)):
            raise ValueError(f'Not actual consecutive stride5: {tid} F{start}->{end}')
        raw,instruction=dataset.logs[tid]
        rgb=dataset._image(loc[start])[None]
        depth,valid,_=dataset._depth(tid,loc[start][0]);depth=depth[None];valid=valid[None]
        gt=np.stack([dataset._scaled_delta_at(raw,start+j) for j in range(5)])
        rng=np.random.default_rng(args.seed+number)
        jitter=rng.normal(size=(5,4))*np.array([.001,.001,.001,np.deg2rad(.1)])
        row=dict(episode_id=tid,start=start,future=end,category=_instruction_task_class(instruction),
                 current_valid_depth_quantiles=np.quantile(depth[valid].numpy(),[0,.01,.5,.99,1]).tolist(),
                 instruction=instruction,gt_actions_m_rad=gt.tolist(),modes={})
        probe_da3=da3 is not None and row['category'] in {'Approach','Land','Pass'} and row['category'] not in da3_tasks
        for mode,actions in [('gt',gt),('zero',np.zeros_like(gt)),('tiny_jitter',jitter)]:
            metrics,_=measure(rgb,depth,valid,actions,probes=args.random_probes,seed=args.seed+number)
            if probe_da3:
                future_depth,future_valid,_=dataset._depth(tid,loc[end][0])
                metrics['da3']=measure_da3(da3,rgb,depth,valid,actions,future_depth[None],future_valid[None])
                print('DA3',row['category'],mode,metrics['da3'],flush=True)
            row['modes'][mode]=metrics
            for category in ('ALL',row['category']):
                grouped[(category,mode)].append(metrics)
        rows.append(row)
        if probe_da3:
            da3_tasks.add(row['category'])
        print(f'[{number+1}/{len(pairs)}] {row["category"]} {tid} '+
              ' '.join(f'{name}:active={m["active_coordinate_fraction_of_covered"]:.3f},grad={m["any_action_vjp_nonzero"]}'
                       for name,m in row['modes'].items()),flush=True)
    summary=[]
    for (category,mode),values in sorted(grouped.items()):
        entry=dict(category=category,mode=mode,samples=len(values),
            nonzero_vjp_samples=sum(v['any_action_vjp_nonzero'] for v in values))
        for key in ('coverage','singleton_fraction_of_covered','multi_source_fraction_of_covered',
                    'active_coordinate_fraction_of_covered','active_coordinate_fraction_of_image','tiny_mass_fraction_of_covered'):
            entry[key]=float(np.mean([v[key] for v in values]))
        entry['median_scaled_action_jacobian_by_axis']=np.median(
            [np.array(v['rgb_action_jacobian_scaled_column_rms']).mean(0) for v in values],axis=0).tolist()
        summary.append(entry)
    result=dict(samples=len(rows),seed=args.seed,random_probes=args.random_probes,summary=summary,rows=rows,
        limitation='Oracle current GT depth; not predicted model depth/proposals, not DA3 depth-loss gradients. '
        'Active coordinate Jacobian is a local necessary condition, not proof of useful control supervision. '
        'Random VJP estimates can miss null directions. FD may cross discrete raster/visibility boundaries. '
        'Endpoint warp constrains net 4DoF motion, not all 20 chunk components independently.')
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('Saved',args.output/'summary.json',flush=True)


if __name__=='__main__':main()
