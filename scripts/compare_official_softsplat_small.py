"""Small CUDA-only official SoftSplat comparison; no DA3, policy or training.

Official avg/soft modes do NOT match our hard per-destination z buffer.
Soft importance is explicitly -target_z/median_current_z (clipped -20..0),
an offline diagnostic choice, not a claimed optimal occlusion model.
"""
import argparse
import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from omegaconf import OmegaConf
from PIL import Image,ImageDraw
import torch

from robot.data.dataset import build_robot_dataset
from experiments.uavflow_predictor_idm.patch_transport import nominal_intrinsics
from experiments.uavflow_predictor_idm.rgbwarp_geometry import action_camera_transform,forward_splat


def project_flow(depth,valid,actions,k):
    b,h,w=depth.shape
    y,x=torch.meshgrid(torch.arange(h,device=depth.device),torch.arange(w,device=depth.device),indexing='ij')
    pixels=torch.stack((x,y,torch.ones_like(x)),-1).float().flatten(0,1)
    good=valid&torch.isfinite(depth)&(depth>0)
    points=(pixels[None]@torch.linalg.inv(k).T)*torch.where(good,depth,0).flatten(1)[...,None]
    t=action_camera_transform(actions)
    target=points@t[:,:3,:3].transpose(-1,-2)+t[:,None,:3,3]
    uvw=target@k.T
    uv=uvw[...,:2]/uvw[...,2:].clamp_min(1e-6)
    good=good.flatten(1)&torch.isfinite(target).all(-1)&(target[...,2]>1e-6)
    flow=torch.where(good[...,None],uv-pixels[None,:,:2],-10000.)
    return flow.reshape(b,h,w,2).permute(0,3,1,2).contiguous(),target[...,2].reshape(b,1,h,w)


def run_case(rgb,depth,valid,actions,official,*,k=None,stability_variants=False,selected_methods=None):
    k=nominal_intrinsics(*depth.shape[-2:],device=depth.device) if k is None else k
    methods=('ours_zbuffer','official_avg','official_soft_depth',
             'official_avg_floor1e-3','official_soft_floor1e-3',
             'ours_drop1e-3','official_avg_drop1e-3')
    if stability_variants:
        methods=('ours_zbuffer','official_avg','official_soft_depth','ours_drop1e-3',
                 'stable_log_kernel_avg','stable_log_kernel_soft','alpha_r1p1','alpha_r1p5')
    if selected_methods is not None:
        methods=tuple(selected_methods)
    metrics={};images={}
    for method in methods:
        value=actions.clone().requires_grad_(True)
        if method.startswith('ours'):
            out=forward_splat(rgb,depth,action_camera_transform(value),k,valid=valid)
            result=out['rgb']
            if 'drop' in method:
                result=result*(out['mass'].detach()>=.001)[:,None]
        else:
            flow,z=project_flow(depth,valid,value,k)
            metric=None
            if method.startswith('official_soft') or method=='stable_log_kernel_soft':
                scale=depth[valid].median().clamp_min(.001).detach()
                metric=(-z/scale).clamp(-20,0).contiguous()
            if method.startswith('stable_log'):
                from scripts.splat_stability_variants import log_kernel_splat
                result=log_kernel_splat(rgb,flow,z,importance=metric)
            elif method.startswith('alpha_'):
                from scripts.splat_stability_variants import alpha_point_splat
                result=alpha_point_splat(rgb,flow,z,radius=1.1 if method=='alpha_r1p1' else 1.5)
            elif 'floor' in method or 'drop' in method:
                # Official CUDA accumulation/backward, with an explicitly
                # different offline normalization floor (NOT vanilla code).
                importance=torch.ones_like(depth[:,None]) if metric is None else metric.exp()
                packed=torch.cat((rgb*importance,importance),dim=1).contiguous()
                summed=official.softsplat(tenIn=packed,tenFlow=flow,tenMetric=None,strMode='sum')
                result=summed[:,:3]/summed[:,3:].clamp_min(.001)
                if 'drop' in method:
                    result=result*(summed[:,3:].detach()>=.001)
            else:
                result=official.softsplat(tenIn=rgb.contiguous(),tenFlow=flow,tenMetric=metric,
                                         strMode='avg' if metric is None else 'soft')
        generator=torch.Generator(device=rgb.device).manual_seed(42)
        gradients=[]
        for _ in range(4):
            signs=(torch.randint(0,2,result.shape,generator=generator,device=rgb.device).float()*2-1)/np.sqrt(result.numel())
            gradients.append(torch.autograd.grad((result*signs).sum(),value,retain_graph=True)[0].detach())
        gradient=torch.stack(gradients).square().mean(0).sqrt()[0]
        metrics[method]=dict(random_vjp_rms_by_step_axis=gradient.cpu().tolist(),
            max_abs_random_vjp=float(torch.stack(gradients).abs().max()),
            finite=bool(torch.isfinite(gradient).all()),nonzero=bool(gradient.max()>1e-7))
        images[method]=result.detach().cpu()[0]
        del result,value,gradients
    for method in methods[1:]:
        metrics[method]['rgb_mae_vs_ours']=float((images[method]-images[methods[0]]).abs().mean())
    return metrics,images


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-config',type=Path,required=True)
    p.add_argument('--source-summary',type=Path,required=True)
    p.add_argument('--official-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--memory-limit-mib',type=int,default=512)
    p.add_argument('--stability-variants',action='store_true')
    args=p.parse_args();torch.set_num_threads(2)
    device=torch.device(args.device)
    if device.type!='cuda':raise ValueError('Official implementation requires CUDA')
    torch.cuda.set_device(device)
    free,total=torch.cuda.mem_get_info(device)
    if args.memory_limit_mib*1024**2+1024**3>free:
        raise RuntimeError('Insufficient free VRAM after reserving 1 GiB for other processes')
    torch.cuda.set_per_process_memory_fraction(args.memory_limit_mib*1024**2/total,device)
    import cupy
    cupy.get_default_memory_pool().set_limit(size=64*1024**2)
    sys.path.insert(0,str(args.official_dir.resolve()))
    official=importlib.import_module('softsplat')
    commit=subprocess.check_output(['git','-C',str(args.official_dir),'rev-parse','HEAD'],text=True).strip()
    wanted={'Approach','Pass','Land'};pairs=[]
    for row in json.loads(args.source_summary.read_text())['rows']:
        if row['category'] in wanted:
            wanted.remove(row['category']);pairs.append(row)
    if wanted:raise ValueError(f'Missing requested task samples: {wanted}')
    cfg=dict(OmegaConf.to_container(OmegaConf.load(args.dataset_config).dataset,resolve=True))
    cfg.pop('split_file',None)
    cfg.update(include_episodes=[r['episode_id'] for r in pairs],max_trajectories=None,pose_progress_enabled=False)
    dataset=build_robot_dataset(cfg,is_eval=False)
    args.output.mkdir(parents=True,exist_ok=True)
    rows=[]
    # Exact integer identity: isolates zero-weight backward without occlusion.
    torch.manual_seed(42)
    synthetic_rgb=torch.rand(1,3,28,28,device=device)
    synthetic_depth=torch.ones(1,28,28,device=device)*2
    metrics,_=run_case(synthetic_rgb,synthetic_depth,torch.ones_like(synthetic_depth,dtype=torch.bool),
        torch.zeros(1,5,4,device=device),official,k=torch.eye(3,device=device),stability_variants=args.stability_variants)
    rows.append(dict(category='synthetic_integer_identity',mode='zero',metrics=metrics))
    print('SYNTHETIC',json.dumps(metrics),flush=True)
    for pair in pairs:
        tid=pair['episode_id'];start=pair['start'];end=pair['future']
        loc=dataset.traj[tid];raw,_=dataset.logs[tid]
        if end-start!=5 or any(loc[start+j][0]!=loc[start][0]+j for j in range(6)):
            raise ValueError('Require actual physical stride5')
        rgb=dataset._image(loc[start])[None].to(device)
        future=dataset._image(loc[end])
        depth,valid,_=dataset._depth(tid,loc[start][0])
        depth,valid=depth[None].to(device),valid[None].to(device)
        gt=torch.tensor(np.stack([dataset._scaled_delta_at(raw,start+j) for j in range(5)]),device=device)[None].float()
        jitter=torch.tensor(np.random.default_rng(42).normal(size=(1,5,4)),device=device).float()*gt.new_tensor([.001,.001,.001,np.deg2rad(.1)])
        for mode,actions in [('gt',gt),('zero',torch.zeros_like(gt)),('tiny_jitter',jitter)]:
            metrics,images=run_case(rgb,depth,valid,actions,official,stability_variants=args.stability_variants)
            rows.append(dict(episode_id=tid,category=pair['category'],start=start,future=end,mode=mode,metrics=metrics))
            canvas=Image.new('RGB',(224*(len(images)+2),250),'white');draw=ImageDraw.Draw(canvas)
            tiles={'current':rgb.detach().cpu()[0],'real_future':future,**images}
            for i,(name,image) in enumerate(tiles.items()):
                array=(image.permute(1,2,0).clamp(0,1).numpy()*255).round().astype(np.uint8)
                canvas.paste(Image.fromarray(array),(i*224,26));draw.text((i*224+3,4),name,fill='black')
            canvas.save(args.output/f'{pair["category"]}_{tid}_{mode}.png')
            print(pair['category'],mode,json.dumps(metrics),flush=True)
    report=dict(official_commit=commit,cupy=cupy.__version__,torch=torch.__version__,
        device=str(device),torch_peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2,
        torch_peak_reserved_mib=torch.cuda.max_memory_reserved(device)/1024**2,rows=rows,
        note='No DA3. Random upstream gradient, not task loss. Official modes lack our hard z-buffer. '
             'Soft depth importance=-z/median(current valid z), clipped [-20,0]. '
             'floor1e-3 variants use official sum primitive with OUR denominator clamp, not vanilla official normalization. '
             'drop1e-3 variants additionally zero low-mass RGB/gradient with a detached threshold; diagnostic only. '
             'stable_log_kernel variants reproduce paper formulas in PyTorch, NOT official author code. '
             'Alpha variants use pixel disks r=1.1/1.5, top K=8, gamma=1; NOT exact SynSin/PyTorch3D reproduction. '
             'Integer-grid boundary gradient conventions differ. Not an equivalence or performance claim.')
    (args.output/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print('SAVED',args.output/'summary.json','peak reserved MiB',report['torch_peak_reserved_mib'],flush=True)


if __name__=='__main__':main()
