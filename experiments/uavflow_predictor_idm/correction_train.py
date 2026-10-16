"""Three switches on the existing cursor-aware trainer, not three trainers."""
import argparse
import os
from pathlib import Path
import sys
import tempfile
from omegaconf import OmegaConf
from experiments.uavflow_support_aware.train import build_config as asset_config


def build_config(mode='pure', extra=()):
    if mode not in {'pure', 'support_aware', 'uncertain'}:
        raise ValueError('Unknown correction mode')
    cfg = asset_config(phase='geometry')
    cfg.model.support_aware_transport = {}
    cfg.model.correction_mode = mode
    cfg.model.provenance_correction = {
        'uncertainty_layers': [27, 39], 'uncertainty_width': 64,
        'uncertainty_tau': 1., 'uncertainty_weight': 1.,
        'depth_scale_meters': 100., 'fov_degrees': 90.,
        'cleanup_kernel': 5, 'min_hole_area': 128,
        'support_detach_action': True, 'support_detach_depth': True,
        'support_routing': {'action_to_geometry': True, 'geometry_to_action': True, 'cross_action_branch': False},
        'residual_merge': {'type': 'support_mass'}}
    # Current is only a depth pass, never a second future-view decoder/bank CA.
    cfg.model.parallel_current_geometry_read_enabled = False
    cfg.training.results_dir = str(Path(os.environ.get('OUTPUT_ROOT', 'results/provenance_correction')) / mode)
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(extra)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--correction-mode', choices=['pure', 'support_aware', 'uncertain'], default='pure')
    parser.add_argument('--init-checkpoint', default='')
    parser.add_argument('--resume', default='')
    parser.add_argument('--proposal-checkpoint', default='')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--print-config', action='store_true')
    parser.add_argument('--max-trajectories', type=int)
    parser.add_argument('--set', action='append', default=[])
    args = parser.parse_args()
    extra = list(args.set)
    if args.max_trajectories is not None:
        extra.append(f'dataset.max_trajectories={args.max_trajectories}')
    cfg = build_config(args.correction_mode, extra)
    if args.print_config:
        print(OmegaConf.to_yaml(cfg, resolve=True)); return
    if args.eval_only and not args.resume:
        parser.error('--eval-only requires --resume')
    from .train import main as train
    with tempfile.TemporaryDirectory(prefix='uavflow-correction-') as folder:
        path = Path(folder)/'config.yaml'
        OmegaConf.save(cfg, path)
        old = sys.argv
        sys.argv = [old[0], '--config', str(path)]
        for flag in ['init_checkpoint', 'resume', 'proposal_checkpoint']:
            if getattr(args, flag):
                sys.argv += ['--'+flag.replace('_', '-'), getattr(args, flag)]
        if args.eval_only:
            sys.argv += ['--eval-only']
        try:
            train()
        finally:
            sys.argv = old


if __name__ == '__main__':
    main()
