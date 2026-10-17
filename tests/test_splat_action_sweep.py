import numpy as np

from scripts.sweep_splat_action_gradients import action_candidates


def test_action_sweep_keeps_gt_and_uses_physical_offsets_without_mutating_input():
    gt=np.array([[.1,.2,.3,.04]]*5,dtype=np.float32);before=gt.copy()
    rows=action_candidates(gt)
    np.testing.assert_array_equal(gt,before)
    names={r['name']:r for r in rows}
    assert len(rows)>60
    np.testing.assert_array_equal(names['gt']['actions'],gt)
    np.testing.assert_allclose(names['gt_scale_0.5']['actions'],gt*.5)
    np.testing.assert_allclose(names['gt_scale_-1']['actions'],-gt)
    assert names['gt']['scale']==1
    offset=names['gt_last_dy_+0.5m']['actions']-gt
    assert np.count_nonzero(offset)==1 and abs(offset[-1,1]-.5)<1e-6
    offset=names['gt_last_yaw_+15deg']['actions']-gt
    assert np.count_nonzero(offset)==1 and abs(offset[-1,3]-np.deg2rad(15))<1e-6
    pure=names['pure_last_dy_+1m']['actions']
    assert np.count_nonzero(pure)==1 and pure[-1,1]==1
    assert len({r['actions'].tobytes() for r in rows})==len(rows)
    repeated=action_candidates(gt)
    assert [r['actions'].tobytes() for r in repeated]==[r['actions'].tobytes() for r in rows]


def test_zero_gt_sweep_deduplicates_identical_chunks():
    rows=action_candidates(np.zeros((5,4),dtype=np.float32))
    assert rows[0]['name']=='gt' and rows[0]['scale']==1
    assert 'gt_scale_0' in rows[0]['aliases']
    assert len({r['actions'].tobytes() for r in rows})==len(rows)
