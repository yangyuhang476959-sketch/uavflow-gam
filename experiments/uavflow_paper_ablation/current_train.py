"""Current-only ablation, using the existing full-checkpoint Warp trainer."""
import argparse
from pathlib import Path
import sys
import tempfile
from omegaconf import OmegaConf

CONFIG = Path(__file__).resolve().parent / 'configs/current_only.yaml'


def build_config(extra=()):
    return OmegaConf.merge(OmegaConf.load(CONFIG), OmegaConf.from_dotlist(list(extra)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--print-config', action='store_true')
    parser.add_argument('--resume', default='')
    parser.add_argument('--eval-only', action='store_true')
    parser.add_argument('--output', default='')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--set', action='append', default=[])
    args = parser.parse_args()
    cfg = build_config(args.set)
    if args.output:
        cfg.training.results_dir = args.output
    if args.epochs is not None:
        cfg.training.max_epochs = args.epochs
    if args.print_config:
        print(OmegaConf.to_yaml(cfg, resolve=True))
        return
    if args.eval_only and not args.resume:
        parser.error('--eval-only requires --resume')
    if cfg.model.rgbwarp_correction.branch_mode != 'current_only':
        parser.error('This entry point is only for current_only')
    if not args.resume:
        output = Path(cfg.training.results_dir)
        if output.exists() and any(output.iterdir()):
            parser.error('Output directory is nonempty; choose a new output or explicitly --resume')
    from experiments.uavflow_predictor_idm.train import main as train
    with tempfile.TemporaryDirectory(prefix='uavflow-current-') as folder:
        path = Path(folder) / 'config.yaml'
        OmegaConf.save(cfg, path)
        previous = sys.argv
        sys.argv = [previous[0], '--config', str(path)]
        if args.resume:
            sys.argv += ['--resume', args.resume]
        if args.eval_only:
            sys.argv += ['--eval-only']
        try:
            train()
        finally:
            sys.argv = previous


if __name__ == '__main__':
    main()
