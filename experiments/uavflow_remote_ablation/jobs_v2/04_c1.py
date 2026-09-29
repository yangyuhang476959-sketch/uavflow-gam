#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from experiments.uavflow_remote_ablation.run_experiment import main
sys.argv[1:1] = ["C1"]
main()
