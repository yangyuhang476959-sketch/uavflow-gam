"""Opt-in RGB-warp correction; reuse the existing DDP/cursor/AMP trainer."""
import argparse
import os
from pathlib import Path
import sys
import tempfile
from omegaconf import OmegaConf
from .correction_train import build_config as correction_config


def build_config(extra=()):
    cfg=correction_config('pure')
    cfg.model.correction_mode='legacy'
    cfg.model.provenance_correction={}
    cfg.model.dual_stream_rgbwarp_enabled=True
    cfg.model.rgbwarp_correction={
        'use_pixel_binary_mask':True,'use_correspondence_attention':True,
        'correspondence_layers':[19,27,33,39],'correspondence_width':256,'correspondence_heads':8,
        'depth_scale_meters':100.,'fov_degrees':90.}
    cfg.model.action_residual_gate_init=.1
    cfg.model.train_deep_backbone=False
    cfg.model.deep_lora.enabled=False
    cfg.model.da3_lora={'enabled':True,'rank':64,'alpha':128.,'dropout':0.,'target_modules':['all-linear']}
    cfg.model.use_fixed_first_frame=False
    cfg.model.use_reference_type_embedding=False
    cfg.stage1.qwen_numeric_pose_token=True
    cfg.stage1.qwen_pose_progress_enabled=True
    cfg.dataset.pose_progress_enabled=True
    cfg.model.pose_progress_enabled=True
    cfg.dataset.openvla_prompt_pose_mode='evaluator_yaw4d'
    cfg.loss.feature_weight=0.
    cfg.loss.deep_feature_weight=0.
    cfg.loss.action_weight=3.
    cfg.loss.action_direct_weight=.5
    cfg.loss.action_refine_weight=.5
    cfg.loss.depth_weight=3.
    # Match GAM linear L1 + finite-difference L1 gradient. Action/depth
    # branches share each outer weight equally instead of doubling it.
    cfg.loss.depth_reconstruction='l1'
    cfg.loss.depth_linear_weight=1.
    cfg.loss.depth_log_weight=0.
    cfg.loss.depth_semantic_weight=1.
    cfg.loss.depth_semantic_separate_weight=0.
    cfg.training.results_dir=str(Path(os.environ.get('OUTPUT_ROOT','results'))/'dual_stream_rgbwarp')
    return OmegaConf.merge(cfg,OmegaConf.from_dotlist(list(extra)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['resume','init-checkpoint','proposal-checkpoint']:p.add_argument('--'+name,default='')
    p.add_argument('--eval-only',action='store_true')
    p.add_argument('--print-config',action='store_true')
    p.add_argument('--max-trajectories',type=int)
    p.add_argument('--set',action='append',default=[])
    args=p.parse_args();extra=list(args.set)
    if args.max_trajectories is not None:extra.append(f'dataset.max_trajectories={args.max_trajectories}')
    cfg=build_config(extra)
    if args.print_config:print(OmegaConf.to_yaml(cfg,resolve=True));return
    if args.eval_only and not args.resume:p.error('--eval-only requires --resume')
    from .train import main as train
    with tempfile.TemporaryDirectory(prefix='rgbwarp-config-') as folder:
        path=Path(folder)/'config.yaml';OmegaConf.save(cfg,path)
        old=sys.argv;sys.argv=[old[0],'--config',str(path)]
        for flag in ['resume','init_checkpoint','proposal_checkpoint']:
            if getattr(args,flag):sys.argv+=['--'+flag.replace('_','-'),getattr(args,flag)]
        if args.eval_only:sys.argv+=['--eval-only']
        try:train()
        finally:sys.argv=old


if __name__=='__main__':main()
