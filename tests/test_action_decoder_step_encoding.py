import pytest
import torch

from robot.modeling.action_head_v2 import ActionHeadV2
from experiments.uavflow_predictor_idm.semantic_geometry_action import ParallelContinuousActionHead
from robot.modeling.future_predictor import GAMFuturePredictor


def test_predictor_decoder_only_removes_extra_slot_embedding():
    kwargs = dict(d_da3=8, d_model=16, depth=1, num_heads=2,
                  num_patches_per_view=4, use_language=False, num_action_slots=5)
    legacy = GAMFuturePredictor(**kwargs)
    decoder_only = GAMFuturePredictor(**kwargs, action_slot_position_encoding=False)
    assert legacy.action_slot_embed is not None
    assert decoder_only.action_slot_embed is None
    assert "action_slot_embed" not in decoder_only.state_dict()
    assert decoder_only.num_action_slots == 5


def test_shared_decoder_matches_gam_for_identical_slots():
    head = ActionHeadV2(input_dim=8, hidden_dim=8, n_views=1,
                        n_dims=4, chunk_size=5, chunk_position_encoding="learned")
    hidden = torch.randn(2, 3, 1, 8)
    torch.testing.assert_close(head(hidden), head.forward_slots(hidden.expand(-1, -1, 5, -1)))


def test_decoder_only_preserves_independent_slot_gradients():
    head = ParallelContinuousActionHead(8, 4, chunk_size=5, decoder_step_embedding=True)
    slots = torch.randn(2, 1, 5, 8, requires_grad=True)
    output = head(slots)
    assert output.shape == (2, 1, 5, 4)
    output.square().sum().backward()
    assert (slots.grad.abs().sum(dim=(0, 1, 3)) > 0).all()
    assert head.decoder.chunk_pos_embed.grad is not None
    with pytest.raises(ValueError):
        head(slots[:, :, :4])


def test_legacy_head_keeps_checkpoint_keys():
    head = ParallelContinuousActionHead(8, 4)
    assert all(name.startswith("model.") for name in head.state_dict())


def test_new_matrix_vla_cells_use_decoder_only_and_all_linear():
    from experiments.uavflow_remote_ablation.matrix_v2 import overrides
    for cell in ("Q0", "S0", "S1", "S2", "R1"):
        flags = dict(item.split("=", 1) for item in overrides(cell))
        assert flags["model.parallel_action_position_mode"] == "decoder_only"
        assert flags["stage1.qwen_lora_scope"] == "all_linear"
    for cell in ("G0", "G1", "C0", "C1"):
        flags = dict(item.split("=", 1) for item in overrides(cell))
        assert "model.parallel_action_position_mode" not in flags


def test_decoder_only_residual_output_is_zero_initialized():
    from experiments.uavflow_remote_ablation.test_geometry_architectures import TinyParallelDA3
    from experiments.uavflow_predictor_idm.model import UAVFlowPredictorIDM
    from experiments.uavflow_predictor_idm.geometry_architectures import (
        architecture_state, load_architecture_state,
    )
    net = UAVFlowPredictorIDM(
        da3=TinyParallelDA3(), idm=None, action_dim=4, action_chunk_size=5,
        d_model=256, depth=1, num_heads=4, language_dim=16, language_len=8,
        direct_action_enabled=True, deep_action_enabled=True, compute_idm_branch=False,
        geometry_architecture="dual_vla_gfm", parallel_vla_gfm_enabled=True,
        parallel_vla_gfm_mode="external_query", parallel_action_decode_mode="geometry_residual",
        parallel_action_position_mode="decoder_only", parallel_vla_gfm_width=32,
        parallel_vla_gfm_heads=4,
    )
    assert net.deep_action_step_embed is None
    assert net.predictor.action_slot_embed is None
    final = net.parallel_action_correction_head.decoder.output
    assert torch.count_nonzero(final.weight) == 0
    assert torch.count_nonzero(final.bias) == 0
    state = architecture_state(net)
    assert state["parallel_action_position_mode"] == "decoder_only"
    state["parallel_action_position_mode"] = "legacy"
    with pytest.raises(ValueError, match="position mode changed"):
        load_architecture_state(net, {"geometry_architecture_state": state})
