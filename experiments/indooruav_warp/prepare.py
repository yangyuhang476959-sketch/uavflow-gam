"""Prepare a reviewable config/split/stats; never launch training."""
import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import yaml

from robot.data.indooruav_dataset import (IndoorUAVVLADataset, read_json,
                                         cumulative_local_targets)

REPO=Path(__file__).resolve().parents[2]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def configure_metric_depth(cfg):
    """Predict metres with equal thirds, as explicitly requested by the user."""
    old_scale=float(cfg['loss']['depth_fixed_scale_meters'])
    cfg['model']['rgbwarp_correction']['depth_scale_meters']=1.
    cfg['loss']['depth_fixed_scale_meters']=1.
    for key in ('depth_linear_weight','depth_grad_weight',
                'depth_semantic_linear_weight','depth_semantic_grad_weight'):
        if key in cfg['loss']:
            cfg['loss'][key]=1/3
    cfg['loss']['depth_log_weight']=1/3
    cfg['loss']['depth_log_epsilon_normalized']*=old_scale


def prepare(root, output, ratio=.05, seed=42, variant="vla_future"):
    root,output=Path(root).resolve(),Path(output).resolve()
    output.mkdir(parents=True,exist_ok=True)
    snapshot=read_json(REPO/'docs/experiment_baselines/indooruav/split_ids.json')
    expected_tasks={
        'train':['vla_ins'+key for key in snapshot['splits']['train']+snapshot['splits']['val']],
        'test_seen':['vla_ins'+key for key in snapshot['official_test_excluded']],
        'test_unseen':[],
    }
    official_train={r['traj_path'].strip('/') for r in csv.DictReader(open(root/'train.csv'))}
    episodes={};groups=collections.defaultdict(list);pose_cache={}
    for name in sorted(expected_tasks['train']):
        path=root/name;data=read_json(path)
        trajectory=str(Path(name).parent.relative_to('vla_ins'))
        if trajectory not in official_train:raise ValueError('Task outside official training split')
        if trajectory not in pose_cache:
            pose_cache[trajectory]=read_json(root/'without_screenshot'/trajectory/'posture.json')
        start,end=map(int,data['source']);poses=pose_cache[trajectory][start-1:end]
        if len(poses)!=end-start+1:raise ValueError(f'Incomplete poses: {name}')
        # Keep all sub-instructions of a long path AND its reversed recording
        # together, avoiding overlapping frames/geometry across train and val.
        parent=Path(trajectory)
        group=str(parent.parent/f'traj_{abs(int(parent.name.removeprefix("traj_")))}')
        key='/'+str(Path(name).relative_to('vla_ins'))
        episodes[key]=dict(trajectory=trajectory,instruction=data['instruction'],
                           source=[start,end],poses=poses,group=group)
        groups[group].append(key)
    target=round(len(episodes)*ratio)
    rng=np.random.default_rng(seed)
    order=list(np.array(sorted(groups),dtype=object)[rng.permutation(len(groups))])
    val_groups=set();count=0
    for group in order:
        proposed=count+len(groups[group])
        if abs(proposed-target)<abs(count-target):
            val_groups.add(group);count=proposed
        if count==target:break
    val=sorted(key for key,e in episodes.items() if e['group'] in val_groups)
    train=sorted(set(episodes)-set(val))
    official_test={'/'+str(Path(p).relative_to('vla_ins'))
                   for split in ('test_seen','test_unseen') for p in expected_tasks[split]}
    if official_test.intersection(train+val):raise ValueError('Official test leakage')
    if {episodes[k]['group'] for k in train}&{episodes[k]['group'] for k in val}:
        raise ValueError('Grouped train/val leakage')
    manifest=dict(format='indooruav_grouped_vla_v1',seed=seed,eval_ratio_requested=ratio,
                  grouping='long trajectory + reversed recording; all sub-instructions together',
                  action_representation='chunk_anchor_local',splits=dict(train=train,val=val),
                  official_test_excluded=sorted(official_test),episodes=episodes,
                  sources={name:sha(root/name) for name in ['train.csv','test_seen.csv','test_unseen.csv']})
    if manifest['splits'] != snapshot['splits'] or manifest['official_test_excluded'] != snapshot['official_test_excluded'] or manifest['sources'] != snapshot['sources']:
        raise ValueError('IndoorUAV split or official source hashes differ from the recorded experiment')
    manifest_path=output/'manifest.json'
    if manifest_path.exists() and read_json(manifest_path)!=manifest:
        raise ValueError('Refusing to replace a different prepared split')
    manifest_path.write_text(json.dumps(manifest,ensure_ascii=False))
    cfg=yaml.safe_load((REPO/'experiments/uavflow_paper_ablation/configs'/f'{variant}.yaml').read_text())
    cfg['stage1'].update(da3_checkpoint=str(REPO/'checkpoints/track4world_da3.pth'),
                         qwen_model=str(REPO/'checkpoints/qwen3.5-2b'),
                         t5_model=str(REPO/'checkpoints/t5-base'),
                         action_stats_dir=str(output/'action_stats'))
    cfg['dataset']=dict(type='indooruav_vla',data_root=str(root),manifest=str(manifest_path),
                        manifest_sha256=sha(manifest_path),
                        image_size=[224,224],future_steps=1,visual_anchor_stride=5,chunk_size=5,
                        action_delta_mode='chunk_anchor_local',action_convention='yaw4d',
                        action_stats_key='indooruav_vla',openvla_prompt_pose_mode='evaluator_yaw4d',
                        depth_unit_scale_m=.001,depth_kind='z',hfov_degrees=90.,
                        terminal_zero_action=True,endpoint_absorbing_window_count=1,
                        endpoint_absorbing_max_starts=5,endpoint_repeat_count=0,
                        eval_ratio=ratio,episode_split_seed=seed,max_trajectories=None)
    cfg['model']['rgbwarp_correction'].update(action_representation='chunk_anchor_local')
    cfg['model']['stop_head_enabled']=False
    cfg['loss']['stop_weight']=0.
    cfg['loss']['depth_target_source']='habitat_gt'
    configure_metric_depth(cfg)
    cfg['training'].update(results_dir=str(REPO/'results'/f'indooruav_{variant}'),
                           batch_size=8,gradient_accumulation_steps=1,num_workers=4,
                           max_epochs=20,max_steps=0,amp=True,amp_dtype='bf16',
                           epoch_weights_only=False,preserve_warp24_microgroups=True,
                           full_eval_unique=True,eval_batch_size=8,eval_every=5000,
                           eval_every_epoch=True,save_every_epochs=1)
    # Ignore any dated orchestration/provenance initializer from the old run.
    for key in ('matched_manifest','matched_common_init',
                'instruction_map','instruction_map_sha256'):
        cfg.pop(key,None)
    for key in ('joint_stop_start_epoch','joint_stop_terminal_repeat_count','preserve_two_rank_microgroups'):
        cfg['training'].pop(key,None)
    val_windows=sum(len(episodes[k]['poses']) for k in val)
    # DDP broadcasts buffers on each eval forward. Preserve unique validation
    # windows while choosing a batch size with equal forward counts per rank.
    eval_batch=next((size for size in range(8,0,-1)
        if len({math.ceil(len(range(r,val_windows,3))/size) for r in range(3)})==1),None)
    if eval_batch is None:raise ValueError('No equal-forward 3-rank validation batching found')
    cfg['training']['eval_batch_size']=eval_batch
    dataset=IndoorUAVVLADataset(**cfg['dataset'])
    stats=dataset.compute_action_statistics()
    (output/'action_stats').mkdir(exist_ok=True)
    stats['indooruav_vla']['manifest_sha256']=sha(manifest_path)
    (output/'action_stats/indooruav_vla.json').write_text(json.dumps(stats['indooruav_vla'],indent=2))
    config_path=output/f'{variant}.yaml'
    config_path.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True))
    windows={split:sum(len(episodes[k]['poses']) for k in ids)
             for split,ids in [('train',train),('val',val)]}
    batches=[math.ceil(len(range(r,windows['val'],3))/eval_batch) for r in range(3)]
    report=dict(train_tasks=len(train),val_tasks=len(val),val_task_ratio=len(val)/len(episodes),
                windows=windows,updates_per_epoch=windows['train']//24,
                updates_20_epochs=(windows['train']//24)*20,
                eval_rank_batches=batches,eval_forward_counts_equal=len(set(batches))==1,
                eval_batch_size_per_rank=eval_batch,
                data_root=str(root),manifest=str(manifest_path),manifest_sha256=sha(manifest_path),
                config=str(config_path),training_started=False,formal_training_started=False,
                image='1280x720 -> 224x126 + 49 black rows above/below; bilinear RGB, nearest depth',
                intrinsics=[[112.,0.,111.5],[0.,112.,111.5],[0.,0.,1.]],
                pose='F0-heading-relative cm/deg + is_start; original shared MLP/scaling',
                action='5 cumulative future targets in CURRENT heading axes, m/rad; terminal clamp',
                depth='uint16 * 0.001 metres; camera-axis z; zero/saturated/padding invalid',
                depth_prediction='direct metres; training/warp scale=1; linear/log/gradient weights '
                                 'each 1/3, log epsilon converted to metres',
                depth_evidence='Geometric reprojection: audit/depth_geometry_hypotheses.json. '
                               'Exporter not published: author issue23; NOT source-code confirmed.')
    (output/'preparation_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data-root',default='/root/autodl-tmp')
    parser.add_argument('--output',default=str(REPO/'prepared/indooruav_vla'))
    parser.add_argument('--eval-ratio',type=float,default=.05)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--variant',choices=('vla_future','vla_gam'),default='vla_future')
    args=parser.parse_args()
    prepare(args.data_root,args.output,args.eval_ratio,args.seed,args.variant)
