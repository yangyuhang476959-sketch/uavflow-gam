"""Consecutive-update health guard for real RGB-warp training."""
import math


class WarpHealthGuard:
    def __init__(self, patience=100, minimum_coverage=0.001, collapse_fraction=0.99):
        self.patience = int(patience)
        if self.patience < 1:
            raise ValueError('Health patience must be positive')
        self.minimum_coverage = float(minimum_coverage)
        self.collapse_fraction = float(collapse_fraction)
        self.empty_steps = 0
        self.collapsed_steps = 0

    def update(self, *, coverage, zero_fraction, depth_mean_m, finite=True):
        metrics = (coverage, zero_fraction, depth_mean_m)
        if not finite or not all(math.isfinite(v) for v in metrics):
            return 'nonfinite'
        self.empty_steps = self.empty_steps + 1 if coverage < self.minimum_coverage else 0
        self.collapsed_steps = self.collapsed_steps + 1 if zero_fraction > self.collapse_fraction else 0
        if self.collapsed_steps >= self.patience:
            return 'current_depth_collapse'
        if self.empty_steps >= self.patience:
            return 'persistent_empty_warp'
        return None

    def state_dict(self):
        return dict(empty_steps=self.empty_steps, collapsed_steps=self.collapsed_steps)

    def load_state_dict(self, state):
        self.empty_steps = max(0, int(state.get('empty_steps', 0)))
        self.collapsed_steps = max(0, int(state.get('collapsed_steps', 0)))
