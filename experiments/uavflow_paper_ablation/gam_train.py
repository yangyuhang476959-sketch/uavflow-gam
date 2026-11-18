"""Explicit matched UAV-GAM / VLA-GAM entry; does not affect legacy GAM jobs."""
import argparse
from pathlib import Path
import sys
import tempfile
from omegaconf import OmegaConf

CONFIGS=Path(__file__).resolve().parent/'configs'


def build_config(variant,extra=(),*,match_gam_slot_embeddings=False):
    if variant not in {'uav_gam','vla_gam'}:raise ValueError('Expected uav_gam or vla_gam')
    cfg=OmegaConf.merge(OmegaConf.load(CONFIGS/(variant+'.yaml')),OmegaConf.from_dotlist(list(extra)))
    if match_gam_slot_embeddings:
        correction=cfg.model.rgbwarp_correction
        if variant!='vla_gam' or not correction.native_gam_path:
            raise ValueError('Matching corrected UAV-GAM slot embeddings requires native VLA-GAM')
        correction.gam_action_slot_position_encoding=True
        correction.gam_action_slot_init_seed=48
    return cfg


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant',choices=['uav_gam','vla_gam'],required=True)
    parser.add_argument('--print-config',action='store_true')
    parser.add_argument('--resume',default='')
    parser.add_argument('--eval-only',action='store_true')
    parser.add_argument('--output',default='')
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--match-gam-slot-embeddings',action='store_true',
                        help='Use the corrected UAV-GAM five-slot embeddings with seed 48 (fresh VLA-GAM runs)')
    parser.add_argument('--set',action='append',default=[])
    args=parser.parse_args()
    if args.match_gam_slot_embeddings and args.resume:
        parser.error('For resume, use the run configuration with its original slot embedding settings')
    try:
        cfg=build_config(args.variant,args.set,match_gam_slot_embeddings=args.match_gam_slot_embeddings)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output:cfg.training.results_dir=args.output
    if args.epochs is not None:cfg.training.max_epochs=args.epochs
    if cfg.model.rgbwarp_correction.branch_mode!=args.variant:parser.error('Variant and branch_mode disagree')
    if args.print_config:
        print(OmegaConf.to_yaml(cfg,resolve=True));return
    if args.eval_only and not args.resume:parser.error('--eval-only requires --resume')
    if not args.resume:
        output=Path(cfg.training.results_dir)
        if output.exists() and any(output.iterdir()):parser.error('Output is nonempty; choose a new output or explicitly --resume')
    from experiments.uavflow_predictor_idm.train import main as train
    with tempfile.TemporaryDirectory(prefix='matched-gam-') as folder:
        path=Path(folder)/'config.yaml';OmegaConf.save(cfg,path)
        previous=sys.argv;sys.argv=[previous[0],'--config',str(path)]
        if args.resume:sys.argv+=['--resume',args.resume]
        if args.eval_only:sys.argv+=['--eval-only']
        try:train()
        finally:sys.argv=previous


if __name__=='__main__':main()
