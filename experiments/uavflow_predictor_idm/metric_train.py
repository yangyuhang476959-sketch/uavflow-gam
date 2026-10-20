"""Three maintained metric-depth experiments on the existing GAM trainer."""
import argparse
import os
from pathlib import Path
import sys
import tempfile

from omegaconf import OmegaConf

from .correction_train import build_config as correction_config
from .rgbwarp_train import build_config as rgbwarp_config
from experiments.uavflow_remote_ablation.matrix_v2 import overrides


VARIANTS=('gam','current_bank','rgbwarp','rgbwarp_no_action')


def build_config(variant,extra=()):
    if variant not in VARIANTS:
        raise ValueError(f'Choose one of {VARIANTS}')
    if variant in {'rgbwarp','rgbwarp_no_action'}:
        cfg=rgbwarp_config()
        if variant=='rgbwarp_no_action':
            cfg.model.rgbwarp_correction.inject_proposal_action_tokens=False
            cfg.model.rgbwarp_correction.visual_residual_heads=8
            cfg.training.results_dir=str(Path(os.environ.get('OUTPUT_ROOT','results/metric_release'))/variant)
    else:
        cfg=correction_config('pure')
        cfg.model.provenance_correction={}
        cfg.model.train_deep_backbone=False
        cfg.model.deep_lora={'enabled':True,'start_block':13,'rank':64,'alpha':128.,'dropout':0.}
        cfg.model.deep_train_start_block=13
        if variant=='gam':
            # Preserve the local T5/pose/F0/H1/chunk5 baseline, except that
            # metric D/100 replaces window point-norm depth supervision.
            cfg=OmegaConf.merge(cfg,OmegaConf.from_dotlist(list(overrides('G1'))))
            cfg.model.correction_mode='legacy'
            cfg.model.train_deep_backbone=False
            cfg.model.deep_lora.enabled=True
            cfg.model.parallel_current_depth_enabled=False
            cfg.model.parallel_current_geometry_read_enabled=False
            cfg.stage1.qwen_numeric_pose_token=False
            cfg.stage1.qwen_lora_enabled=False
            cfg.stage1.qwen_action_placeholder_count=0
            cfg.stage1.qwen_token_selection='text_after_image'
            cfg.stage1.qwen_prompt_mode='current_image_instruction'
        else:
            # Original VLA proposal -> GAM future Predictor -> DA3 correction.
            # Proposal is decoded from native Qwen slots, not predicted GAM slots.
            cfg.model.correction_mode='pure'
            cfg.model.action_residual_gate_init=.1
            cfg.stage1.qwen_numeric_pose_token=True
            cfg.model.parallel_current_depth_enabled=True
            cfg.model.parallel_current_geometry_read_enabled=True
            cfg.model.current_geometry_bank_mode='output_current'
            cfg.model.current_geometry_read_mode='per_layer'
            cfg.loss.feature_weight=0.
            cfg.loss.action_direct_weight=.5
            cfg.loss.action_refine_weight=.5
        cfg.loss.depth_scale_mode='fixed_metric'
        cfg.loss.depth_fixed_scale_meters=100.
        cfg.training.max_epochs=5
    # Correction graphs can leave disabled/optional heads unused on any backend.
    cfg.training.find_unused_parameters = variant in {'current_bank', 'rgbwarp', 'rgbwarp_no_action'}
    cfg.training.gradient_accumulation_steps = 1
    # Split the outer depth budget3 equally, like proposal/final action.
    cfg.loss.depth_linear_weight=1./3.
    cfg.loss.depth_log_weight=1./3.
    cfg.loss.depth_grad_weight=1./3.
    cfg.loss.depth_semantic_linear_weight=1./3.
    cfg.loss.depth_semantic_log_weight=1./3.
    cfg.loss.depth_semantic_grad_weight=1./3.
    # All new pose-token runs use the online yaw4d convention explicitly.
    cfg.dataset.openvla_prompt_pose_mode='evaluator_yaw4d'
    cfg.dataset.pose_progress_enabled=True
    cfg.model.pose_progress_enabled=True
    cfg.stage1.qwen_pose_progress_enabled=variant!='gam'
    root=Path(os.environ.get('MODEL_ROOT',Path(__file__).resolve().parents[2]/'checkpoints'))
    cfg.stage1.t5_model=os.environ.get('T5_MODEL',str(root/'t5-base'))
    cfg.training.results_dir=str(Path(os.environ.get('OUTPUT_ROOT','results/metric_release'))/variant)
    return OmegaConf.merge(cfg,OmegaConf.from_dotlist(list(extra)))


def main(default_variant=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=VARIANTS,default=default_variant,required=default_variant is None)
    for name in ('resume','init-checkpoint','proposal-checkpoint'):
        parser.add_argument('--'+name,default='')
    parser.add_argument('--max-trajectories',type=int)
    parser.add_argument('--eval-only',action='store_true')
    parser.add_argument('--print-config',action='store_true')
    parser.add_argument('--set',action='append',default=[])
    args=parser.parse_args();extra=list(args.set)
    if args.max_trajectories is not None:
        extra.append(f'dataset.max_trajectories={args.max_trajectories}')
    cfg=build_config(args.variant,extra)
    if args.print_config:
        print(OmegaConf.to_yaml(cfg,resolve=True));return
    if args.eval_only and not args.resume:
        parser.error('--eval-only requires --resume')
    from .train import main as train
    with tempfile.TemporaryDirectory(prefix='metric-gam-') as directory:
        path=Path(directory)/'config.yaml';OmegaConf.save(cfg,path)
        old=sys.argv;sys.argv=[old[0],'--config',str(path)]
        for flag in ('resume','init_checkpoint','proposal_checkpoint'):
            if getattr(args,flag):sys.argv+=['--'+flag.replace('_','-'),getattr(args,flag)]
        if args.eval_only:sys.argv+=['--eval-only']
        try:train()
        finally:sys.argv=old


if __name__=='__main__':main()
