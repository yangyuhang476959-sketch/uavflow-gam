"""Candidate 4: RGB warp with GAM-style random learned Future DA3 action slots."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.uavflow_predictor_idm.metric_train import main


if __name__ == '__main__':
    main('rgbwarp_no_action')
