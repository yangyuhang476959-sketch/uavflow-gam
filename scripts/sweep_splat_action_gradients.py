"""Matched warp-only action sweep; no DA3 or production model changes.

GT scaling, GT-centered offsets, sign reversal, pure lateral/yaw and random
perturbations. Each unique chunk remains FIVE local [dx,dy,dz,dyaw] steps.
Random upstream probes are identical across actions/methods, not depth loss.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf
from PIL import Image,ImageDraw
import torch

from robot.data.dataset import build_robot_dataset
from scripts.compare_official_softsplat_small import run_case


METHODS=('ours_zbuffer','stable_log_kernel_soft','alpha_r1p5','ours_drop1e-3')


def action_candidates(gt,seed=42):
    gt=np.asarray(gt,dtype=np.float32)
    if gt.shape!=(5,4) or not np.isfinite(gt).all():
        raise ValueError('Expected finite five-step physical actions [5,4]')
    rows=[];seen={}
    def add(name,family,value,**extra):
        value=np.array(value,dtype=np.float32,copy=True)
        value[value==0]=0  # Canonicalize signed zero before deduplication.
        key=value.tobytes()
        if key in seen:
            seen[key]['aliases'].append(name)
            if 'scale' in extra:seen[key].setdefault('scale',extra['scale'])
            return
        row=dict(name=name,family=family,actions=value.copy(),aliases=[name],**extra)
        seen[key]=row;rows.append(row)
    # GT first, ensuring exact baseline and no misleading duplicates.
    add('gt','scaled_gt',gt,scale=1.)
    for scale in [-1.,0.,1e-5,1e-4,.001,.01,.05,.1,.25,.5,.75,.9,.95,.99,1.01,1.05,1.1,1.25,1.5,2.]:
        add(f'gt_scale_{scale:g}','scaled_gt',gt*scale,scale=scale)
    for scale in [-1.,0.,.5,1.5]:
        value=gt.copy();value[:,:3]*=scale
        add(f'translation_scale_{scale:g}','translation_scale',value)
        value=gt.copy();value[:,3]*=scale
        add(f'yaw_scale_{scale:g}','yaw_scale',value)
    for axis,label in enumerate(('dx','dy','dz')):
        for delta in [-.5,-.1,-.01,.01,.1,.5]:
            value=gt.copy();value[-1,axis]+=delta
            add(f'gt_last_{label}_{delta:+g}m','gt_offset',value)
    for degrees in [-15.,-5.,-1.,-.1,.1,1.,5.,15.]:
        value=gt.copy();value[-1,3]+=np.deg2rad(degrees)
        add(f'gt_last_yaw_{degrees:+g}deg','gt_offset',value)
    for delta in [-1.,-.5,-.1,-.01,.01,.1,.5,1.]:
        value=np.zeros_like(gt);value[-1,1]=delta
        add(f'pure_last_dy_{delta:+g}m','pure_lateral',value)
    for degrees in [-15.,-5.,-1.,-.1,.1,1.,5.,15.]:
        value=np.zeros_like(gt);value[-1,3]=np.deg2rad(degrees)
        add(f'pure_last_yaw_{degrees:+g}deg','pure_yaw',value)
    rng=np.random.default_rng(seed)
    for meters,degrees in [(.01,.1),(.05,1.),(.2,5.)]:
        noise=rng.normal(size=(5,4))*[meters,meters,meters,np.deg2rad(degrees)]
        for sign in [-1,1]:
            add(f'gt_random_{meters:g}m_{degrees:g}deg_{sign:+d}','random_gt',gt+sign*noise)
    return rows


def plot_sweep(rows,path,title):
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for axis,(ax,label,scale) in enumerate(zip(axes.flat,['dx','dy','dz','dyaw'],[.01,.01,.01,np.deg2rad(1.)])):
        for method in METHODS:
            points=sorted((r['scale'],r['metrics'][method]['mean_rms_by_axis'][axis]*scale)
                          for r in rows if 'scale' in r)
            ax.plot([p[0] for p in points],[p[1] for p in points],'.-',label=method)
        ax.set_xscale('symlog',linthresh=.01);ax.set_yscale('symlog',linthresh=1e-5)
        ax.axvline(1.,color='gray',linestyle='--');ax.set_title(label)
        ax.set_xlabel('GT chunk multiplier (1=GT, 0=zero)')
        ax.set_ylabel('Random VJP RMS x (1cm or 1degree)');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7)
    fig.suptitle(title+'; warp-only, NOT depth-loss/control quality')
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset-config',type=Path,required=True)
    p.add_argument('--source-summary',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',default='cuda:0')
    p.add_argument('--memory-limit-mib',type=int,default=512)
    args=p.parse_args();torch.set_num_threads(2)
    device=torch.device(args.device)
    if device.type=='cuda':
        torch.cuda.set_device(device);free,total=torch.cuda.mem_get_info(device)
        if free<args.memory_limit_mib*1024**2+1024**3:
            raise RuntimeError('Insufficient safe GPU headroom; reserve 1GiB for other processes')
        torch.cuda.set_per_process_memory_fraction(args.memory_limit_mib*1024**2/total,device)
    wanted={'Approach','Pass','Land'};pairs=[]
    for row in json.loads(args.source_summary.read_text())['rows']:
        if row['category'] in wanted:
            wanted.remove(row['category']);pairs.append(row)
    if wanted:raise ValueError(f'Missing categories: {wanted}')
    cfg=dict(OmegaConf.to_container(OmegaConf.load(args.dataset_config).dataset,resolve=True))
    cfg.pop('split_file',None)
    cfg.update(include_episodes=[r['episode_id'] for r in pairs],max_trajectories=None,pose_progress_enabled=False)
    dataset=build_robot_dataset(cfg,is_eval=False)
    args.output.mkdir(parents=True,exist_ok=True)
    all_rows=[];groups=defaultdict(list)
    for pair in pairs:
        tid=pair['episode_id'];start=pair['start'];end=pair['future'];loc=dataset.traj[tid]
        if end-start!=5 or any(loc[start+j][0]!=loc[start][0]+j for j in range(6)):
            raise ValueError('Require actual physical stride5')
        rgb=dataset._image(loc[start])[None].to(device)
        future=dataset._image(loc[end]);raw,_=dataset.logs[tid]
        depth,valid,_=dataset._depth(tid,loc[start][0]);depth,valid=depth[None].to(device),valid[None].to(device)
        gt=np.stack([dataset._scaled_delta_at(raw,start+j) for j in range(5)])
        candidates=action_candidates(gt);rows=[]
        for number,candidate in enumerate(candidates):
            value=torch.tensor(candidate['actions'],device=device)[None]
            metrics,images=run_case(rgb,depth,valid,value,None,selected_methods=METHODS)
            for method,m in metrics.items():
                grad=np.array(m['random_vjp_rms_by_step_axis'])
                m['mean_rms_by_axis']=grad.mean(0).tolist()
                m['physical_scaled_rms']=float(np.sqrt(np.mean((grad*[.01,.01,.01,np.deg2rad(1.)])**2)))
                groups[(method,candidate['family'])].append(m['physical_scaled_rms'])
            row=dict(episode_id=tid,category=pair['category'],start=start,future=end,
                **{k:v for k,v in candidate.items() if k!='actions'},actions_m_rad=candidate['actions'].tolist(),metrics=metrics)
            rows.append(row);all_rows.append(row)
            if candidate['name'] in {'gt','gt_scale_0','gt_scale_0.5','gt_scale_2','gt_last_dy_+0.5m','gt_last_yaw_+15deg'}:
                tiles={'current':rgb.cpu()[0],'real_future':future,**images}
                canvas=Image.new('RGB',(224*len(tiles),250),'white');draw=ImageDraw.Draw(canvas)
                for i,(name,image) in enumerate(tiles.items()):
                    array=(image.permute(1,2,0).clamp(0,1).numpy()*255).round().astype(np.uint8)
                    canvas.paste(Image.fromarray(array),(224*i,26));draw.text((224*i+3,4),name,fill='black')
                canvas.save(args.output/f'{pair["category"]}_{candidate["name"]}.png')
            if number%10==0 or number==len(candidates)-1:
                print(pair['category'],f'{number+1}/{len(candidates)}',candidate['name'],flush=True)
        plot_sweep(rows,args.output/f'{pair["category"]}_GT_scale_sweep.png',f'{pair["category"]} {tid} F{start}->{end}')
        (args.output/f'{pair["category"]}_rows.json').write_text(json.dumps(rows,indent=2,allow_nan=False))
    summary=[dict(method=m,family=f,samples=len(v),median=float(np.median(v)),p90=float(np.quantile(v,.9)),maximum=max(v))
             for (m,f),v in sorted(groups.items())]
    result=dict(pairs=pairs,unique_action_cases=len(all_rows),methods=METHODS,summary=summary,rows=all_rows,
        note='Four matched random upstream probes per action; NOT DA3/depth-loss gradients. '
             'One frame pair per task, same scenes as previous test, not a dataset-wide claim. '
             'Stable log-kernel/SynSin-inspired alpha are formula probes, not author-code reproductions. '
             'Different fusion/footprints can change images; smaller gradient is not proof of better control.',
        peak_allocated_mib=torch.cuda.max_memory_allocated(device)/1024**2 if device.type=='cuda' else None,
        peak_reserved_mib=torch.cuda.max_memory_reserved(device)/1024**2 if device.type=='cuda' else None)
    (args.output/'summary.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print('Saved',args.output/'summary.json','cases',len(all_rows),flush=True)


if __name__=='__main__':main()
