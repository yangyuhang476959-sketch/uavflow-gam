from experiments.uavflow_predictor_idm.training_health import WarpHealthGuard


def feed(guard, coverage=0.0, zero_fraction=0.0, finite=True):
    return guard.update(coverage=coverage, zero_fraction=zero_fraction, depth_mean_m=10.0, finite=finite)


def test_exactly_one_hundred_consecutive_empty_updates_required():
    guard = WarpHealthGuard(patience=100)
    for _ in range(99):
        assert feed(guard) is None
    assert feed(guard) == 'persistent_empty_warp'


def test_recovery_resets_both_counters_independently():
    guard = WarpHealthGuard(patience=3)
    assert feed(guard, zero_fraction=1.0) is None
    assert feed(guard, zero_fraction=1.0) is None
    assert feed(guard, coverage=0.5, zero_fraction=1.0) == 'current_depth_collapse'
    guard = WarpHealthGuard(patience=3)
    feed(guard); feed(guard)
    assert feed(guard, coverage=0.5) is None
    assert guard.empty_steps == 0
    assert feed(guard) is None
    assert feed(guard) is None
    assert feed(guard) == 'persistent_empty_warp'


def test_nonfinite_stops_immediately_and_thresholds_are_strict():
    guard = WarpHealthGuard()
    assert feed(guard, finite=False) == 'nonfinite'
    assert feed(guard, coverage=float('nan')) == 'nonfinite'
    for _ in range(101):
        assert feed(guard, coverage=0.001, zero_fraction=0.99) is None


def test_resume_preserves_consecutive_count():
    first = WarpHealthGuard(patience=100)
    for _ in range(99):
        assert feed(first) is None
    resumed = WarpHealthGuard(patience=100)
    resumed.load_state_dict(first.state_dict())
    assert feed(resumed) == 'persistent_empty_warp'
