import copy
import types

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from experiments.uavflow_predictor_idm.patch_transport import (
    actions_to_camera_transform, bilinear_patch_transport, nominal_intrinsics,
)
from experiments.uavflow_predictor_idm.support_aware_transport import (
    FrozenDeepTeacher, GeometryRead, transport_losses,
)
from experiments.uavflow_predictor_idm.model import UAVFlowPredictorIDM
from experiments.uavflow_predictor_idm.geometry_architectures import architecture_state, load_architecture_state


@pytest.fixture(autouse=True)
def one_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def test_camera_motion_matches_cpu_reference_and_path_composition():
    actions = torch.tensor([[[.2, .1, -.3, .4], [1., -.2, .1, -.1]]])
    actual = actions_to_camera_transform(actions)[0]
    angle = .4
    rotation_first = np.array([[np.cos(angle), -np.sin(angle), 0],
                               [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    translation = np.array([.2, .1, -.3]) + rotation_first @ np.array([1., -.2, .1])
    rotation_total = np.array([[np.cos(.3), -np.sin(.3), 0],
                               [np.sin(.3), np.cos(.3), 0], [0, 0, 1]])
    motion = np.eye(4)
    motion[:3, :3], motion[:3, 3] = rotation_total, translation
    extrinsic = np.array([[0., 0., 1., 0.], [1., 0., 0., 0.],
                          [0., -1., 0., 0.], [0., 0., 0., 1.]])
    np.testing.assert_allclose(actual.numpy(), np.linalg.inv(extrinsic) @ np.linalg.inv(motion) @ extrinsic, atol=1e-6)
    assert not torch.allclose(actual, actions_to_camera_transform(actions.sum(1, keepdim=True))[0])


def test_identity_transport_is_patch_identity_not_feature_expansion():
    depth = torch.full((2, 28, 28), 4.)
    warp = bilinear_patch_transport(depth, nominal_intrinsics(28, 28), torch.eye(4))
    torch.testing.assert_close(warp["matrix"], torch.eye(4).expand(2, -1, -1), atol=2e-6, rtol=0.)
    torch.testing.assert_close(warp["rho"], torch.ones(2, 4))
    geometry = torch.randn(2, 4, 1536, requires_grad=True)
    (warp["matrix"] @ geometry).sum().backward()
    torch.testing.assert_close(geometry.grad, torch.ones_like(geometry))


def test_floating_bilinear_point_has_four_covered_pixels():
    depth = torch.zeros(1, 28, 28)
    depth[0, 14, 14] = 4.
    transform = torch.eye(4)
    transform[0, 3], transform[1, 3] = 4*.3/14, 4*.7/14
    warp = bilinear_patch_transport(depth, nominal_intrinsics(28, 28), transform)
    assert warp["pixel_support"].sum() == 4
    assert warp["rho_raw"].sum() == pytest.approx(4/196)
    assert warp["matrix"].shape == (1, 4, 4)


def test_empty_depth_and_cleanup_cannot_manufacture_feature_support():
    warp = bilinear_patch_transport(torch.zeros(1, 28, 28), nominal_intrinsics(28, 28),
                                    torch.eye(4), cleanup_kernel=5, min_hole_area=128)
    assert not warp["matrix"].any()
    assert not warp["rho"].any()
    assert torch.isfinite(warp["matrix"]).all()


def test_invalid_depth_and_near_competition_are_finite():
    depth = torch.zeros(1, 28, 28)
    depth[0, 14, 14] = 2.
    depth[0, 14, 15] = 4.
    depth[0, 0, 0], depth[0, 0, 1] = float("nan"), float("inf")
    transform = torch.eye(4)
    transform[0, 3] = 4/14
    warp = bilinear_patch_transport(depth, nominal_intrinsics(28, 28), transform,
                                    z_absolute_tolerance=0., z_relative_tolerance=0.)
    assert warp["pixel_support"].sum() == 1
    assert torch.isfinite(warp["matrix"]).all()


def test_reader_empty_and_completion_value_scaling_not_softmax_rescaling():
    reader = GeometryRead(16, 16, width=16, heads=4)
    q, memory, support = torch.randn(2, 5, 16), torch.randn(2, 7, 16), torch.ones(2, 7)
    full = reader(q, memory, support, torch.ones_like(support))
    quarter = reader(q, memory, support, torch.full_like(support, .25))
    torch.testing.assert_close(quarter, full*.25)
    assert not reader(q, memory, support*0).any()
    assert not reader(q, memory, support, support*0).any()


class TinyTransportDA3(nn.Module):
    embed_dim = 16
    num_register_tokens = 0
    out_layers = [19, 27, 33, 39]

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.))
        self.dpt_head = nn.Identity()

    def encode_shallow_visual_slots(self, images, T, V):
        value = images.reshape(images.shape[0], T, V, -1).mean(-1)
        return {"visual_tokens": value[..., None, None].expand(-1, -1, -1, 257, 16)}

    def propagate_shallow_visual_slots_grad(self, visual, **kwargs):
        deep = torch.cat((visual, visual*self.scale), -1)
        return {"deep_levels": [deep]*4, "shallow": visual}

    def propagate_shallow_with_actions_grad(self, visual, actions, **kwargs):
        assert kwargs["return_multi_level"] and not kwargs["decode_visuals"]
        shifted = visual + actions.mean(-2).unsqueeze(-2)*.01
        deep = torch.cat((shifted, shifted*self.scale), -1)
        return {"action_tokens": actions.reshape(visual.shape[0], 5, 16),
                "level_feats": [(deep[:, 0, :, 1:], deep[:, 0, :, 0])]*4}

    def decode_depth_full(self, pairs, **kwargs):
        patches = pairs[-1][0]
        depth = patches.mean(-1).reshape(patches.shape[0], 1, 16, 16).abs()+.04
        return {"depth": torch.nn.functional.interpolate(depth, size=(224, 224), mode="nearest")}


def build(uncertainty="none", source="proposal"):
    net = UAVFlowPredictorIDM(
        da3=TinyTransportDA3(), idm=None, action_dim=4, action_chunk_size=5,
        d_model=256, depth=1, num_heads=4, language_dim=16, language_len=8,
        direct_action_enabled=True, deep_action_enabled=True, compute_idm_branch=False,
        geometry_architecture="dual_vla_gfm", parallel_vla_gfm_enabled=True,
        parallel_vla_gfm_mode="qwen_tokens", parallel_action_decode_mode="geometry_residual",
        parallel_action_position_mode="decoder_only", parallel_vla_gfm_width=32,
        parallel_vla_gfm_heads=4, parallel_current_geometry_read_enabled=False,
        depth_decode_enabled=True, use_fixed_first_frame=True,
        use_reference_type_embedding=True, gradient_checkpointing=False,
        support_aware_transport={"enabled": True, "uncertainty": uncertainty,
                                 "geometry_action_source": source, "cleanup_kernel": 1,
                                 "min_hole_area": 0, "read_width": 16, "read_heads": 4,
                                 "readout_width": 16},
    )
    return net


def inputs():
    observed = torch.randn(2, 1, 1, 257, 16)*.1
    metric = torch.full((2, 224, 224), 10.)
    metric[:, :, :70] = 0.
    return observed, {"reference_shallow": observed.clone(), "lang_feats": torch.randn(2, 5, 16),
                      "lang_padding_mask": torch.ones(2, 5, dtype=torch.bool),
                      "action_denormalizer": lambda x: x*.01,
                      "current_depth_meters": metric}


def test_real_predictor_candidate_b_forward_backward_and_no_uncertainty_by_default():
    net = build()
    observed, kwargs = inputs()
    result = net(observed, **kwargs)
    assert result["actions_norm"].shape == (2, 1, 5, 4)
    torch.testing.assert_close(result["actions_norm"], result["direct_actions_norm"])
    assert result["transport"]["completion_uncertainty"] is None
    teacher = net.da3.propagate_shallow_visual_slots_grad(observed)["deep_levels"]
    losses = transport_losses(result, teacher, prefix=1, visual_dim=16)
    (losses["geometry"] + result["actions_norm"].square().mean()).backward()
    assert net.transport_policy.motion_projection[0].weight.grad is not None
    assert net.transport_policy.geometry_projection[-1].weight.grad is not None
    assert net.predictor.blocks[0].cross_attn.q_proj.weight.grad is not None
    assert net.transport_policy.residual_heads[0].delta[-1].weight.grad is not None
    assert net.transport_policy.completion_heads[0].delta[-1].weight.grad is not None
    assert result["deep_joint_features"]["depth"].shape == (2, 1, 224, 224)


def test_gt_action_pretraining_does_not_change_output_actions_with_gt():
    net = build(source="gt")
    observed, kwargs = inputs()
    first = net(observed, action_targets_norm=torch.zeros(2, 1, 5, 4), **kwargs)
    second = net(observed, action_targets_norm=torch.ones(2, 1, 5, 4), **kwargs)
    torch.testing.assert_close(first["actions_norm"], second["actions_norm"])
    assert not first["geometry_action_residual_norm"].any()
    assert not torch.equal(first["transport"]["geometry_actions_raw"], second["transport"]["geometry_actions_raw"])
    net.eval()
    a = net(observed, action_targets_norm=torch.zeros(2, 1, 5, 4), **kwargs)
    b = net(observed, action_targets_norm=torch.ones(2, 1, 5, 4), **kwargs)
    torch.testing.assert_close(a["actions_norm"], b["actions_norm"])
    torch.testing.assert_close(a["transport"]["geometry_actions_raw"], a["transport"]["proposal_actions_raw"])


def test_scalar_error_head_has_no_action_gradients_but_has_calibration_gradients():
    net = build("scalar")
    nn.init.normal_(net.parallel_action_correction_head.decoder.output.weight, std=.01)
    observed, kwargs = inputs()
    result = net(observed, **kwargs)
    result["actions_norm"].square().mean().backward(retain_graph=True)
    assert all(p.grad is None for p in net.transport_policy.error_head.parameters())
    teacher = net.da3.propagate_shallow_visual_slots_grad(observed)["deep_levels"]
    calibration = transport_losses(result, teacher, prefix=1, visual_dim=16)["uncertainty"]
    net.zero_grad(set_to_none=True)
    calibration.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in net.transport_policy.error_head.parameters())
    assert all(p.grad is None for p in net.transport_policy.completion_heads.parameters())
    assert all(p.grad is None for p in net.predictor.parameters())


def test_future_action_hidden_cannot_bypass_completion_reader():
    net = build("scalar").eval()
    nn.init.normal_(net.parallel_action_correction_head.decoder.output.weight, std=.01)
    observed, kwargs = inputs()
    before = net(observed, **kwargs)["actions_norm"]
    original = net.da3.propagate_shallow_with_actions_grad
    def changed(*args, **options):
        result = original(*args, **options)
        result["action_tokens"] = result["action_tokens"] + 1e6
        return result
    net.da3.propagate_shallow_with_actions_grad = changed
    after = net(observed, **kwargs)["actions_norm"]
    torch.testing.assert_close(before, after)


def test_inference_checkpoint_load_roundtrip(monkeypatch, tmp_path):
    from experiments.uavflow_support_aware.train import build_config
    from experiments.uavflow_support_aware.inference import load_policy, SupportAwarePolicy
    from omegaconf import OmegaConf
    import robot.modeling.da3_giant_encoder as da3_module
    import experiments.uavflow_predictor_idm.vlm_conditioning as condition_module
    net = build("scalar").eval()
    with torch.no_grad():
        net.da3.scale.fill_(1.8)
        nn.init.normal_(net.parallel_action_correction_head.decoder.output.weight, std=.01)
    cfg = build_config(uncertainty="scalar", extra=(
        "model.d_model=256", "model.depth=1", "model.num_heads=4", "model.language_len=8",
        "model.parallel_vla_gfm_width=32", "model.parallel_vla_gfm_heads=4",
        "model.train_deep_backbone=false", "training.amp=false", "model.gradient_checkpointing=false",
        "model.residual_prediction=true"))
    OmegaConf.update(cfg, "model.support_aware_transport", net.transport_policy.config, merge=False)
    checkpoint = {"config": OmegaConf.to_container(cfg),
                  "geometry_architecture_state": architecture_state(net),
                  "predictor": net.predictor.state_dict(),
                  "semantic_geometry_action": net.semantic_geometry_action.state_dict(),
                  "parallel_action_head": net.parallel_action_head.state_dict(),
                  "parallel_action_correction_head": net.parallel_action_correction_head.state_dict(),
                  "action_normalizer": normalizer().state_dict(),
                  "da3_trainable": {"scale": net.da3.scale.detach()},
                  "reference_step_embed": net.reference_step_embed.detach(),
                  "residual_gate_logit": net.residual_gate_logit.detach()}
    path = tmp_path/"policy.pt"
    torch.save(checkpoint, path)
    monkeypatch.setattr(da3_module, "DA3GiantEncoder", lambda **kwargs: TinyTransportDA3())
    monkeypatch.setattr(condition_module, "build_stage2_conditioner", lambda *args: TinyConditioner())
    loaded = load_policy(path, device="cpu", da3_checkpoint="new/base.pth", qwen_model="new/qwen")
    reference, current = torch.rand(2, 1, 3, 224, 224), torch.rand(2, 1, 3, 224, 224)
    previous = SupportAwarePolicy(net, TinyConditioner(), normalizer(), amp=False)
    a = previous.predict(current, ["pass", "land"], reference_images=reference)
    b = loaded.predict(current, ["pass", "land"], reference_images=reference)
    torch.testing.assert_close(a["actions_meters_radians"], b["actions_meters_radians"])
    torch.testing.assert_close(a["transport"]["completion_confidence"], b["transport"]["completion_confidence"])


def test_pure_prompt_proposal_transfer_is_exact_and_rejects_numeric_architectures():
    from experiments.uavflow_predictor_idm.support_aware_transport import initialize_pure_qwen_proposal
    net = build().eval()
    # A small source with the exact existing Pure-Qwen projection/head shape.
    projection = nn.Sequential(nn.LayerNorm(16), nn.Linear(16, 16))
    from robot.modeling.action_head_v2 import ActionHeadV2
    head = ActionHeadV2(input_dim=16, hidden_dim=16, n_views=1, n_dims=4,
                        chunk_size=5, chunk_position_encoding="learned")
    state = {"projection."+k: v for k, v in projection.state_dict().items()}
    state.update({"head."+k: v for k, v in head.state_dict().items()})
    checkpoint = {"format": "uavflow_qwen_pure_v1", "config": {"model": {"pose_mode": "prompt"}},
                  "trainable_state": state,
                  "action_stats": {"q01": [-.1]*4, "q99": [.1]*4, "mask": [True]*4}}
    initialize_pure_qwen_proposal(net, TinyConditioner(), normalizer(), checkpoint)
    observed, kwargs = inputs()
    expected = head.forward_slots(projection(kwargs["lang_feats"])[:, None])
    actual = net(observed, **kwargs)["direct_actions_norm"]
    torch.testing.assert_close(actual, expected)
    for mode in ("numeric_head", "numeric_token"):
        checkpoint["config"]["model"]["pose_mode"] = mode
        with pytest.raises(ValueError, match="state-injection"):
            initialize_pure_qwen_proposal(net, TinyConditioner(), normalizer(), checkpoint)


def test_checkpoint_roundtrip_and_explicit_stage_transfer():
    net = build()
    clone = build()
    state = {"geometry_architecture_state": architecture_state(net)}
    load_architecture_state(clone, state)
    for key, value in net.transport_policy.state_dict().items():
        torch.testing.assert_close(clone.transport_policy.state_dict()[key], value)
    changed = build(source="gt")
    with pytest.raises(ValueError, match="configuration changed"):
        load_architecture_state(changed, state)
    load_architecture_state(changed, state, allow_new_transport=True)
    old = copy.deepcopy(state)
    old["geometry_architecture_state"].pop("support_aware_transport")
    with pytest.raises(KeyError, match="resume checkpoint"):
        load_architecture_state(clone, old)
    load_architecture_state(clone, old, allow_new_transport=True)
    scalar = build("scalar")
    load_architecture_state(scalar, state, allow_new_transport=True)
    torch.testing.assert_close(scalar.transport_policy.proposal_projection[-1].weight,
                               net.transport_policy.proposal_projection[-1].weight)


def test_frozen_teacher_retains_base_after_student_changes_and_preserves_graph():
    da3 = TinyTransportDA3()
    teacher = FrozenDeepTeacher(da3)
    shallow = torch.randn(1, 1, 1, 257, 16, requires_grad=True)
    baseline = teacher(shallow)["deep_levels"][-1]
    with torch.no_grad():
        da3.scale.fill_(3.)
    student = da3.propagate_shallow_visual_slots_grad(shallow)["deep_levels"][-1]
    torch.testing.assert_close(teacher(shallow)["deep_levels"][-1], baseline)
    student.sum().backward()
    assert da3.scale.grad is not None
    assert da3.scale.item() == 3.


def test_bad_proposal_suppresses_all_future_geometry_supervision():
    net = build("scalar")
    observed, kwargs = inputs()
    # Denormalizer preserves physical units in this test.
    kwargs["action_denormalizer"] = lambda x: x
    result = net(observed, action_targets_norm=torch.full((2, 1, 5, 4), 20.), **kwargs)
    assert not result["transport"]["supervision_gate"].any()
    teacher = net.da3.propagate_shallow_visual_slots_grad(observed)["deep_levels"]
    loss = transport_losses(result, teacher, prefix=1, visual_dim=16)
    assert loss["geometry"].item() == 0.
    assert loss["uncertainty"].item() == 0.


class TinyConditioner(nn.Module):
    hidden_size = 16
    is_multimodal = True
    variable_length_tokens = True
    use_reference_image = False

    def encode_tokens(self, texts, **kwargs):
        b = len(texts)
        device = kwargs["images"].device
        return {"last_hidden_state": torch.ones(b, 5, 16, device=device),
                "attention_mask": torch.ones(b, 5, device=device, dtype=torch.bool)}


def normalizer():
    from robot.data.dataset import ActionNormalizer
    return ActionNormalizer({"sim": {"q01": [-.1]*4, "q99": [.1]*4, "mask": [True]*4}}, default_key="sim")


def batch():
    return {"all_view_images": torch.rand(2, 2, 1, 3, 224, 224),
            "episode_first_image": torch.rand(2, 1, 3, 224, 224),
            "task_description": ["pass the car from the right", "land beside the tree"],
            "actions": torch.zeros(2, 2, 5, 4), "action_stats_key": ["sim"]*2,
            "action_loss_mask": torch.ones(2, 2, 5, 4, dtype=torch.bool),
            "gt_depth_meters": torch.full((2, 2, 1, 224, 224), 10.),
            "gt_depth_mask": torch.ones(2, 2, 1, 224, 224, dtype=torch.bool),
            "start_t": torch.zeros(2, dtype=torch.long)}


def test_forward_batch_full_depth_losses_and_proposal_gate_no_future_gt_inputs():
    from experiments.uavflow_predictor_idm.objectives import forward_batch
    net = build("scalar")
    net.da3._transport_teacher = FrozenDeepTeacher(net.da3)
    data = batch()
    args = dict(model=net, da3=net.da3, text=TinyConditioner(), normalizer=normalizer(),
                pose_normalizer=None, batch=data, context_len=1, rollout_steps=1,
                feature_horizon_weights=[1.], feature_patch_weight=1., feature_cls_weight=0.,
                feature_register_weight=0., deep_feature_enabled=False,
                deep_feature_patch_weight=1., deep_feature_cls_weight=0., stop_pos_weight=5., amp=False,
                depth_target_mode="both", depth_scale_mode="fixed_metric", depth_gradient_mode="gam_l1")
    values = forward_batch(**args)
    assert "depth_current_total" in values and "depth_future_total" in values
    assert torch.isfinite(values["transport_geometry"]) and torch.isfinite(values["completion_uncertainty_loss"])
    (values["action"]+values["feature"]+values["depth"]+values["transport_geometry"]+
     values["completion_uncertainty_loss"]).backward()
    assert net.transport_policy.error_head[-1].weight.grad is not None
    captured = []
    original_forward = net.forward
    def record(*inputs, **options):
        out = original_forward(*inputs, **options)
        captured.append(out["actions_norm"].detach().clone())
        return out
    net.forward = record
    forward_batch(**args)
    data["all_view_images"][:, 1] = 1-data["all_view_images"][:, 1]
    data["gt_depth_meters"][:, 1] *= 2
    data["gt_depth_mask"][:, 1] = False
    forward_batch(**args)
    torch.testing.assert_close(captured[0], captured[1])
    # Deliberately wrong endpoints: current supervision survives, all future
    # targets are gated (feature, branch deep features, depth, calibration).
    data["actions"] = torch.full_like(data["actions"], 20.)
    values = forward_batch(**args)
    assert values["transport_matched_fraction"] == 0
    assert values["feature"] == 0
    assert values["transport_geometry"] == 0
    assert values["completion_uncertainty_loss"] == 0
    assert values["depth_future_total"] == 0
    assert values["depth_current_total"] > 0
    assert values["action"] > 0


def test_inference_uses_no_future_images_depth_actions_or_gt_pose():
    from experiments.uavflow_support_aware.inference import SupportAwarePolicy
    net = build(source="gt")
    policy = SupportAwarePolicy(net, TinyConditioner(), normalizer(), amp=False)
    result = policy.predict(torch.rand(2, 1, 3, 224, 224), ["pass right", "land"],
                            reference_images=torch.rand(2, 1, 3, 224, 224))
    assert result["actions_meters_radians"].shape == (2, 5, 4)
    torch.testing.assert_close(result["transport"]["geometry_actions_raw"],
                               result["transport"]["proposal_actions_raw"])


def test_config_horizon_and_optional_c_and_selected_paths(monkeypatch, tmp_path):
    from experiments.uavflow_support_aware.train import build_config
    monkeypatch.setenv("UAVFLOW_DEPTH_ROOT", str(tmp_path/"depth"))
    monkeypatch.setenv("OUTPUT_ROOT", str(tmp_path/"output"))
    cfg = build_config()
    assert cfg.model.support_aware_transport.uncertainty == "none"
    assert cfg.dataset.visual_anchor_stride*cfg.model.feature_target_offset == cfg.model.action_chunk_size == 5
    assert cfg.loss.depth_target_mode == "both" and cfg.loss.depth_scale_mode == "fixed_metric"
    assert cfg.stage1.qwen_action_placeholder_count == 5
    assert cfg.stage1.qwen_lora_scope == "all_linear"
    assert cfg.dataset.gt_depth_root == str(tmp_path/"depth/hybrid")
    assert cfg.dataset.gt_depth_fallback_roots == [str(tmp_path/"depth/replay")]
    c = build_config(phase="correction", uncertainty="scalar")
    assert c.model.support_aware_transport.geometry_action_source == "proposal"


def _ddp_worker(rank, rendezvous):
    from torch.nn.parallel import DistributedDataParallel
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method="file://"+rendezvous, rank=rank, world_size=2)
    try:
        for source in ("gt", "proposal"):
            net = build("scalar", source)
            # Match the real trainer's frozen teacher capture before DDP.
            net.da3._transport_teacher = FrozenDeepTeacher(net.da3)
            ddp = DistributedDataParallel(net, find_unused_parameters=False)
            optimizer = torch.optim.AdamW((p for p in net.parameters() if p.requires_grad), lr=1e-4)
            observed, kwargs = inputs()
            teacher = net.da3._transport_teacher(observed)["deep_levels"]
            # Second forward is essential: DDP detects previous unused grads
            # only on the following iteration.
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                out = ddp(observed, action_targets_norm=torch.zeros(2, 1, 5, 4), **kwargs)
                geom = transport_losses(out, teacher, prefix=1, visual_dim=16)
                loss = (out["actions_norm"].square().mean() + geom["geometry"]
                        + geom["uncertainty"] + out["current_depth_output"]["depth"].mean())
                loss.backward()
                optimizer.step()
            del ddp
            dist.barrier()
    finally:
        dist.destroy_process_group()


def test_two_rank_ddp_gt_and_proposal_phases_have_no_unused_parameter_failure(tmp_path):
    mp.spawn(_ddp_worker, args=(str(tmp_path/"ddp_init"),), nprocs=2, join=True)
