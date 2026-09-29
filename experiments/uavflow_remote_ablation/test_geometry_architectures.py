"""CPU tests for A/B/D/E/F routing, objectives, masks and checkpoint transfer."""
import io
import types
import unittest
import csv
import subprocess
from pathlib import Path
from unittest.mock import patch
import torch
from torch import nn
from experiments.uavflow_predictor_idm.model import UAVFlowPredictorIDM
from experiments.uavflow_predictor_idm.geometry_architectures import (
    DualActionFusion, architecture_state, load_architecture_state, select_dual_view,
)
from experiments.uavflow_direct_visual_probe.qwen35_semantic import (
    _replace_action_placeholder_embeddings,
)
from experiments.uavflow_predictor_idm.objectives import architecture_feature_loss, forward_batch
from robot.modeling.dual_geometry_attention import dual_allow_mask, run_dual_masked_block
from experiments.uavflow_predictor_idm.semantic_geometry_action import (
    OFTDimensionActionTokenizer,
    ParallelContinuousActionHead,
    QwenInternalActionProjector,
    QwenParallelActionTokenizer,
    SemanticActionInitializer,
)
from experiments.uavflow_direct_visual_probe.qwen35_semantic import (
    _qwen35_text_forward_with_action_block,
)


MODES = ["current_prediction", "direct_current", "dual_observed", "dual_predicted", "dual_action_bridge"]


class TinyDA3(nn.Module):
    embed_dim = 16
    num_register_tokens = 2

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.dpt_head = nn.Identity()
        self.calls = 0
        self.last_mask = None

    def encode_shallow_visual_slots(self, images, T, V):
        b = images.shape[0]
        content = images.reshape(b, T, V, -1).mean(-1)
        return {"visual_tokens": content[..., None, None].expand(b, T, V, 7, 16)}

    def propagate_shallow_with_actions_grad(self, visual, actions, **kwargs):
        self.calls += 1
        self.last_mask = kwargs.get("dual_state_attention", "full")
        b, h, v = visual.shape[:3]
        depth = (visual.mean((-1, -2)) * self.scale).reshape(b, h * v, 1, 1).expand(b, h * v, 2, 2)
        return {"action_tokens": ((actions + visual.mean(-2)) * self.scale).reshape(b * h * v, -1),
                "depth": depth}

    def propagate_shallow_visual_slots_grad(self, *args, **kwargs):
        raise AssertionError("Architecture should reuse its joint/current decode, not run a second current pass")


class TinyParallelDA3(TinyDA3):
    def propagate_shallow_visual_slots_grad(self, visual, **kwargs):
        deep = torch.cat([visual, visual], dim=-1) * self.scale
        return {
            "shallow": visual, "deep_levels": [deep, deep, deep, deep],
            "layer_patches": {index: deep[..., 3:, :] for index in range(4)},
        }

    def propagate_shallow_with_actions_grad(self, visual, actions, **kwargs):
        self.calls += 1
        if actions.ndim != 5:
            raise AssertionError(f"Expected parallel actions, got {tuple(actions.shape)}")
        b, h, v, k, _ = actions.shape
        mixed = actions + visual.mean(-2).unsqueeze(-2)
        depth = (visual.mean((-1, -2)) * self.scale).reshape(
            b, h * v, 1, 1
        ).expand(b, h * v, 2, 2)
        return {"action_tokens": (mixed * self.scale).reshape(b * h * v, k, -1),
                "depth": depth}


def build(mode, stop=False, stop_mode="legacy_action_token"):
    net = UAVFlowPredictorIDM(
        da3=TinyDA3(), idm=None, action_dim=4, action_chunk_size=5,
        d_model=256, depth=1, num_heads=4, language_dim=16, language_len=2,
        direct_action_enabled=True, deep_action_enabled=True, compute_idm_branch=False,
        gradient_checkpointing=False, geometry_architecture=mode,
        depth_decode_enabled=True, stop_head_enabled=stop,
        stop_head_mode=stop_mode,
        use_fixed_first_frame=True, use_reference_type_embedding=True,
    )
    def rollout(self, observed, **kwargs):
        shift = 0.1 + (kwargs["prediction_role"].mean() if "prediction_role" in kwargs else 0.0)
        return observed + shift, observed[..., 0, :] + shift
    net.rollout_shallow = types.MethodType(rollout, net)
    return net


class Attention(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_heads = 2
        self.qkv = nn.Linear(8, 24)
        self.proj = nn.Linear(8, 8)
        self.q_norm = self.k_norm = self.proj_drop = self.attn_drop = nn.Identity()
        self.rope = None


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(8)
        self.norm2 = nn.LayerNorm(8)
        self.attn = Attention()
        self.ls1 = self.ls2 = nn.Identity()
        self.mlp = nn.Sequential(nn.Linear(8, 16), nn.GELU(), nn.Linear(16, 8))


class IdentityNormalizer:
    def normalize(self, value, **kwargs):
        return value

    denormalize = normalize


class Tests(unittest.TestCase):
    def test_action_placeholder_embeddings_are_ordered_and_trainable(self):
        base = torch.zeros(1, 5, 3)
        token_ids = torch.tensor([[4, 101, 9, 102, 4]])
        action_embeddings = torch.nn.Parameter(torch.tensor([
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
        ]))
        output = _replace_action_placeholder_embeddings(
            base, token_ids, (101, 102), action_embeddings
        )
        self.assertTrue(torch.equal(output[0, 1], action_embeddings[0]))
        self.assertTrue(torch.equal(output[0, 3], action_embeddings[1]))
        self.assertTrue(torch.equal(output[0, 0], torch.zeros(3)))
        output.sum().backward()
        self.assertTrue(torch.equal(
            action_embeddings.grad, torch.ones_like(action_embeddings)
        ))

    def test_dual_action_fusion_starts_as_mean_and_learns(self):
        fusion = DualActionFusion(16)
        current = torch.randn(2, 3, 16, requires_grad=True)
        future = torch.randn(2, 3, 16, requires_grad=True)
        fused, gate = fusion(current, future)
        torch.testing.assert_close(gate, torch.full_like(gate, 0.5))
        torch.testing.assert_close(fused, 0.5 * (current + future))
        fused.square().mean().backward()
        self.assertGreater(fusion.gate[-1].weight.grad.abs().sum().item(), 0)

    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

    def test_future_predictor_has_five_real_action_slots(self):
        import robot.modeling.future_predictor as predictor_module
        predictor = predictor_module.GAMFuturePredictor(
            d_da3=16, d_model=32, depth=2, num_heads=4,
            num_patches_per_view=4, num_register_tokens=2,
            use_language=False, proprio_dim=5, action_dim=4,
            action_chunk_size=1, use_proprio_input=False,
            num_action_slots=5,
        )
        visual = torch.randn(2, 1, 1, 7, 16)
        history = torch.zeros(2, 1, 1, 4)
        seed = torch.randn(2, 5, 32, requires_grad=True)
        with patch("robot.modeling.future_predictor._HAS_FLEX", False):
            output = predictor(
                visual, past_action_history=history, action_slot_seed=seed,
            )
        self.assertEqual(output["predicted_action_tokens"].shape, (2, 1, 1, 5, 16))
        output["predicted_action_tokens"].square().mean().backward()
        self.assertGreater(seed.grad.abs().sum().item(), 0)
        legacy = predictor_module.GAMFuturePredictor(
            d_da3=16, d_model=32, depth=1, num_heads=4,
            num_patches_per_view=4, num_register_tokens=2,
            use_language=False, proprio_dim=5, action_dim=4,
            action_chunk_size=1, use_proprio_input=False,
        )
        self.assertNotIn("action_slot_embed", legacy.state_dict())

    def test_stop_only_pose_does_not_change_actions(self):
        for mode, steps in (("legacy", 3), ("dual_observed", 1)):
            with self.subTest(mode=mode):
                net = build(mode, stop=True, stop_mode="action_hidden_pose")
                self.assertFalse(net.use_pose_history)
                x = torch.randn(2, steps, 1, 7, 16)
                kwargs = dict(
                    reference_shallow=x[:, :1],
                    lang_feats=torch.randn(2, 2, 16),
                    lang_padding_mask=torch.ones(2, 2, dtype=torch.bool),
                )
                pose_a = torch.zeros(2, steps, 5)
                pose_b = pose_a.clone()
                pose_b[..., 0] = 2.0
                first = net(x, stop_pose=pose_a, **kwargs)
                second = net(x, stop_pose=pose_b, **kwargs)
                self.assertEqual(first["stop_logits"].shape, (2, steps))
                self.assertTrue(torch.equal(first["actions_norm"], second["actions_norm"]))
                self.assertFalse(torch.equal(first["stop_logits"], second["stop_logits"]))
                second["stop_logits"].sum().backward()
                self.assertGreater(net.stop_head.pose_proj[0].weight.grad.abs().sum().item(), 0)
                with self.assertRaisesRegex(ValueError, "stop-only current pose"):
                    net(x, **kwargs)

    def test_stop_only_pose_uses_current_anchor(self):
        net = build("legacy", stop=True, stop_mode="action_hidden_pose")
        raw_pose = torch.tensor([
            [[1., 2., 3., 0., 1.], [8., 9., 10., 0., 1.]],
            [[4., 5., 6., 0., 1.], [11., 12., 13., 0., 1.]],
        ])
        batch = {
            "all_view_images": torch.ones(2, 2, 1, 3, 2, 2),
            "episode_first_image": torch.ones(2, 1, 3, 2, 2),
            "task_description": ["move", "turn"],
            "action_stats_key": ["sim", "sim"],
            "actions": torch.zeros(2, 1, 5, 4),
            "action_loss_mask": torch.ones(2, 1, 5, 4, dtype=torch.bool),
            "stop_target": torch.zeros(2, 1),
            "episode_pose": raw_pose,
            "gt_depth_meters": torch.full((2, 2, 1, 2, 2), 3.),
            "gt_depth_mask": torch.ones(2, 2, 1, 2, 2, dtype=torch.bool),
        }
        captured = []
        hook = net.stop_head.register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[1].detach().clone())
        )
        class ShiftPoseNormalizer:
            def normalize(self, value):
                return value + 10.
        try:
            with patch(
                "experiments.uavflow_predictor_idm.objectives.encode_stage2_condition",
                return_value={
                    "last_hidden_state": torch.ones(2, 3, 16),
                    "attention_mask": torch.ones(2, 3, dtype=torch.bool),
                },
            ):
                forward_batch(
                    model=net, da3=net.da3, text=None,
                    normalizer=IdentityNormalizer(),
                    pose_normalizer=ShiftPoseNormalizer(), batch=batch,
                    context_len=1, rollout_steps=1,
                    feature_horizon_weights=[1], feature_patch_weight=1,
                    feature_cls_weight=0, feature_register_weight=0,
                    deep_feature_enabled=False, deep_feature_patch_weight=1,
                    deep_feature_cls_weight=0, stop_pos_weight=5, amp=False,
                )
        finally:
            hook.remove()
        torch.testing.assert_close(captured[0], raw_pose[:, :1] + 10.)

    def test_real_predictor_two_roles(self):
        net = build("dual_predicted")
        # Exercise the real Predictor, not the fast routing stub used below.
        del net.rollout_shallow
        x = torch.randn(1, 1, 1, 259, 16)
        # CPU uses the Predictor's existing dense causal-mask fallback.
        with patch("robot.modeling.future_predictor._HAS_FLEX", False):
            out = net(x, reference_shallow=x, lang_feats=torch.randn(1, 2, 16),
                      lang_padding_mask=torch.ones(1, 2, dtype=torch.bool))
        self.assertEqual(out["predicted_current_shallow"].shape, x.shape)
        loss = out["future_shallow"].square().mean() + out["predicted_current_shallow"].square().mean()
        loss.backward()
        self.assertTrue((net.prediction_roles.grad.abs().sum(1) > 0).all())

    def test_matrix_matches_executable_modes(self):
        from experiments.uavflow_remote_ablation.matrix_v2 import (
            EXPERIMENTS, overrides,
        )
        root = Path(__file__).parent
        with (root / "compact_matrix.tsv").open() as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual([row["id"] for row in rows], list(EXPERIMENTS))
        for row in rows:
            flags = dict(value.split("=", 1) for value in overrides(row["id"]))
            self.assertEqual(
                flags["model.parallel_action_decode_mode"],
                "geometry_residual" if row["id"] == "R1" else "full",
            )
            if row["assistant_slots"] == "5":
                self.assertEqual(
                    flags["stage1.qwen_token_selection"], "action_placeholders"
                )
                self.assertEqual(flags["stage1.qwen_action_placeholder_count"], "5")

    def test_real_encoder_loop_checkpoint_mask(self):
        from robot.modeling.da3_giant_encoder import DA3GiantEncoder
        trans = nn.Module()
        trans.blocks = nn.ModuleList([Block() for _ in range(4)])
        trans.num_register_tokens = 2
        trans.alt_start = 0
        trans.rope_start = 0
        trans.rope = None
        trans.norm = nn.LayerNorm(8)
        trans._prepare_rope = lambda *args: (None, None)
        backbone = nn.Module()
        backbone.pretrained = trans
        encoder = types.SimpleNamespace(
            backbone=backbone, embed_dim=8, PATCH_SIZE=14, temporal_embed=None,
            out_layers=[0, 1, 2, 3], _deep_prefix_lengths=lambda **kw: None,
            _build_camera_tokens=lambda b, v, dev, dtype: torch.zeros(b, v, 8, device=dev, dtype=dtype),
        )
        x = torch.randn(2, 1, 2, 7, 8, requires_grad=True)
        a = torch.randn(2, 1, 2, 8, requires_grad=True)
        run = DA3GiantEncoder._propagate_shallow_with_actions_impl
        normal = run(encoder, x, a, decode_visuals=False, return_multi_level=True,
                     dual_state_attention="action_bridge")
        checkpointed = run(encoder, x, a, decode_visuals=False, return_multi_level=True,
                           dual_state_attention="action_bridge", gradient_checkpointing=True)
        torch.testing.assert_close(normal["action_tokens"], checkpointed["action_tokens"])
        self.assertEqual(len(checkpointed["level_feats"]), 4)
        cur = checkpointed["level_feats"][-1][0][:, 0].square().mean()
        gx, ga = torch.autograd.grad(cur, (x, a), retain_graph=True)
        self.assertEqual(gx[:, :, 1].abs().sum().item(), 0)
        self.assertEqual(ga.abs().sum().item(), 0)
        g1 = torch.autograd.grad(normal["action_tokens"].square().mean(), x, retain_graph=True)[0]
        g2 = torch.autograd.grad(checkpointed["action_tokens"].square().mean(), x)[0]
        torch.testing.assert_close(g1, g2)

    def test_parallel_semantic_geometry_tokens_and_da3_insertion(self):
        bridge = SemanticActionInitializer(
            language_dim=12, output_dim=8,
            chunk_size=5, width=16, heads=2,
        )
        language = torch.randn(2, 6, 12, requires_grad=True)
        language_mask = torch.tensor([
            [1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 1],
        ], dtype=torch.bool)
        seeds = bridge(language, language_mask)
        self.assertEqual(seeds.shape, (2, 5, 8))

        trans = nn.Module()
        trans.blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])
        trans.num_register_tokens, trans.alt_start, trans.rope_start = 2, 0, 0
        trans.rope, trans.cat_token = None, True
        trans.norm = nn.LayerNorm(8)
        trans._prepare_rope = lambda *args: (None, None)
        trans.process_attention = lambda x, blk, **kw: x + blk(x.mean(2, keepdim=True))
        backbone = nn.Module(); backbone.pretrained = trans
        enc = types.SimpleNamespace(
            backbone=backbone, embed_dim=8, PATCH_SIZE=14, temporal_embed=None,
            out_layers=[0, 1, 2, 3], _deep_prefix_lengths=lambda **kw: None,
            _build_camera_tokens=lambda b, v, dev, dtype: torch.zeros(
                b, v, 8, device=dev, dtype=dtype
            ),
        )
        visual = torch.randn(2, 1, 1, 7, 8, requires_grad=True)
        run = __import__(
            "robot.modeling.da3_giant_encoder", fromlist=["DA3GiantEncoder"]
        ).DA3GiantEncoder._propagate_shallow_with_actions_impl
        result = run(
            enc, visual, seeds[:, None, None], decode_visuals=False,
            return_multi_level=True,
        )
        self.assertEqual(result["action_tokens"].shape, (2, 5, 8))
        self.assertEqual(result["level_feats"][-1][0].shape[-2], 4)
        head = ParallelContinuousActionHead(8, 4)
        actions = head(result["action_tokens"].reshape(2, 1, 1, 5, 8))
        self.assertEqual(actions.shape, (2, 1, 5, 4))
        actions.square().mean().backward()
        self.assertGreater(language.grad[:, :3].abs().sum().item(), 0)
        self.assertGreater(visual.grad.abs().sum().item(), 0)

        qwen = QwenParallelActionTokenizer(
            language_dim=12, output_dim=8, chunk_size=5, width=16, heads=2,
        )
        self.assertEqual(qwen(language.detach(), language_mask).shape, (2, 5, 8))
        oft = OFTDimensionActionTokenizer(
            language_dim=12, output_dim=8, chunk_size=5, action_dim=4,
            width=16, heads=2,
        )(language.detach(), language_mask)
        self.assertEqual(oft["dimension_tokens"].shape, (2, 20, 16))
        self.assertEqual(oft["plan_tokens"].shape, (2, 5, 8))
        self.assertEqual(oft["direct_actions_norm"].shape, (2, 5, 4))

    def test_five_token_post_qwen_bidirectional_mixer(self):
        plain = QwenInternalActionProjector(
            language_dim=12, output_dim=8, chunk_size=5, layers=0,
        )
        mixed = QwenInternalActionProjector(
            language_dim=12, output_dim=8, chunk_size=5,
            width=16, heads=2, layers=2,
        )
        tokens = torch.randn(2, 5, 12)
        changed = tokens.clone()
        changed[:, -1] += torch.linspace(-2.0, 3.0, 12)
        # Point-wise projection cannot route the final slot into the first.
        torch.testing.assert_close(plain(tokens)[:, 0], plain(changed)[:, 0])
        # Full self-attention can route information in both directions.
        self.assertFalse(torch.allclose(mixed(tokens)[:, 0], mixed(changed)[:, 0]))

    def test_qwen_full_attention_action_block_mask(self):
        class Layer(nn.Module):
            def __init__(self):
                super().__init__(); self.seen = None
            def forward(self, hidden_states, attention_mask=None, **kwargs):
                self.seen = attention_mask
                return hidden_states
        module = nn.Module()
        module.embed_tokens = nn.Embedding(8, 4)
        module.config = types.SimpleNamespace(
            num_hidden_layers=2,
            layer_types=["full_attention", "linear_attention"],
        )
        module.layers = nn.ModuleList([Layer(), Layer()])
        module.norm = nn.Identity()
        module.rotary_emb = lambda hidden, positions: (None, None)
        module._update_linear_attn_mask = lambda mask, cache: mask
        module._uav_action_token_mask = torch.tensor(
            [[False, False, False, True, True]]
        )
        padding = torch.ones(1, 5, dtype=torch.long)
        _qwen35_text_forward_with_action_block(
            module, input_ids=torch.arange(5).view(1, 5),
            attention_mask=padding, use_cache=False,
        )
        full = module.layers[0].seen[0, 0]
        self.assertTrue(full[3, 4] and full[4, 3])
        self.assertFalse(full[0, 3] or full[0, 4])
        self.assertEqual(module.layers[1].seen.shape, (1, 5))

    def test_complete_dual_vla_gfm_routing(self):
        for mode in ("external_query", "qwen_tokens", "oft_gfm", "oft_direct"):
            with self.subTest(mode=mode):
                net = UAVFlowPredictorIDM(
                    da3=TinyParallelDA3(), idm=None, action_dim=4, action_chunk_size=5,
                    d_model=256, depth=1, num_heads=4, language_dim=16, language_len=8,
                    direct_action_enabled=True, deep_action_enabled=True,
                    compute_idm_branch=False, gradient_checkpointing=False,
                    geometry_architecture="dual_vla_gfm",
                    parallel_vla_gfm_enabled=True, parallel_vla_gfm_mode=mode,
                    parallel_vla_gfm_width=32, parallel_vla_gfm_heads=4,
                    depth_decode_enabled=True, use_fixed_first_frame=True,
                    use_reference_type_embedding=True,
                )
                def rollout(self, observed, **kwargs):
                    seed = kwargs["action_slot_seed"]
                    self._test_seed = seed
                    projected = self.predictor.action_proj(
                        self.predictor.out_action_norm(seed)
                    )
                    return observed + 0.1, projected[:, None, None]
                net.rollout_shallow = types.MethodType(rollout, net)
                visual = torch.randn(2, 1, 1, 7, 16, requires_grad=True)
                language_length = (
                    5 if mode == "qwen_tokens"
                    else 20 if mode in {"oft_gfm", "oft_direct"}
                    else 6
                )
                language = torch.randn(2, language_length, 16, requires_grad=True)
                language_mask = torch.ones(2, language_length, dtype=torch.bool)
                if mode == "external_query":
                    language_mask[0, 3:] = False
                output = net(
                    visual, reference_shallow=visual,
                    lang_feats=language,
                    lang_padding_mask=language_mask,
                )
                self.assertIsNone(net.causal_action_decoder)
                self.assertEqual(net._test_seed.shape, (2, 5, 256))
                self.assertEqual(output["actions_norm"].shape, (2, 1, 5, 4))
                self.assertEqual(output["refine_actions_norm"].shape, (2, 1, 5, 4))
                self.assertEqual(output["deep_joint_features"]["action_tokens"].shape, (2, 5, 16))
                self.assertIsNotNone(output["current_geometry_features"])
                if mode == "oft_direct":
                    torch.testing.assert_close(
                        output["actions_norm"], output["direct_actions_norm"]
                    )
                output["actions_norm"].square().mean().backward()
                self.assertGreater(language.grad.abs().sum().item(), 0)
                if mode != "oft_direct":
                    self.assertGreater(visual.grad.abs().sum().item(), 0)

    def test_geometry_residual_starts_from_vla_base(self):
        net = UAVFlowPredictorIDM(
            da3=TinyParallelDA3(), idm=None, action_dim=4, action_chunk_size=5,
            d_model=256, depth=1, num_heads=4, language_dim=16, language_len=8,
            direct_action_enabled=True, deep_action_enabled=True,
            compute_idm_branch=False, gradient_checkpointing=False,
            geometry_architecture="dual_vla_gfm",
            parallel_vla_gfm_enabled=True,
            parallel_vla_gfm_mode="external_query",
            parallel_action_decode_mode="geometry_residual",
            parallel_vla_gfm_width=32, parallel_vla_gfm_heads=4,
            depth_decode_enabled=True, use_fixed_first_frame=True,
            use_reference_type_embedding=True,
        )

        def rollout(self, observed, **kwargs):
            projected = self.predictor.action_proj(
                self.predictor.out_action_norm(kwargs["action_slot_seed"])
            )
            return observed + 0.1, projected[:, None, None]

        net.rollout_shallow = types.MethodType(rollout, net)
        visual = torch.randn(2, 1, 1, 7, 16)
        language = torch.randn(2, 6, 16)
        output = net(
            visual, reference_shallow=visual,
            lang_feats=language,
            lang_padding_mask=torch.ones(2, 6, dtype=torch.bool),
        )
        torch.testing.assert_close(
            output["refine_actions_norm"], output["direct_actions_norm"]
        )
        torch.testing.assert_close(
            output["geometry_action_residual_norm"],
            torch.zeros_like(output["geometry_action_residual_norm"]),
        )
        output["refine_actions_norm"].square().mean().backward()
        final = net.parallel_action_correction_head.model[-1]
        self.assertGreater(final.weight.grad.abs().sum().item(), 0)

    def test_mask_truth_table(self):
        n = 5
        m = dual_allow_mask(n)
        current, actions, future = [0, 2, 3, 4], [1, 6], [5, 7, 8, 9]
        self.assertTrue(m[actions].all())
        self.assertFalse(m[current][:, actions + future].any())
        self.assertTrue(m[current][:, current].all())
        self.assertFalse(m[future][:, current].any())
        self.assertTrue(m[future][:, actions + future].all())
        self.assertTrue(m.any(-1).all())
        local = dual_allow_mask(n, local=True)
        self.assertTrue(torch.equal(local[0], m[:n, :n]))
        self.assertTrue(torch.equal(local[1], m[n:, n:]))

    def test_per_layer_ca_affects_next_visual_layer(self):
        from robot.modeling.da3_giant_encoder import DA3GiantEncoder
        from experiments.uavflow_predictor_idm.current_geometry import CurrentGeometryRead
        trans = nn.Module()
        trans.blocks = nn.ModuleList([nn.Linear(8, 8) for _ in range(4)])
        trans.num_register_tokens, trans.alt_start, trans.rope_start = 2, 0, 0
        trans.rope, trans.cat_token = None, True
        trans.norm = nn.LayerNorm(8)
        trans._prepare_rope = lambda *args: (None, None)
        # Deterministic small token mixer exercising the actual encoder loops.
        trans.process_attention = lambda x, blk, **kw: x + blk(x.mean(2, keepdim=True))
        backbone = nn.Module()
        backbone.pretrained = trans
        enc = types.SimpleNamespace(
            backbone=backbone, embed_dim=8, PATCH_SIZE=14, temporal_embed=None,
            out_layers=[0, 1, 2, 3], _deep_prefix_lengths=lambda **kw: None,
            _build_camera_tokens=lambda b, v, dev, dtype: torch.zeros(b, v, 8, device=dev, dtype=dtype),
        )
        current = torch.randn(2, 1, 1, 7, 8, requires_grad=True)
        visual = torch.randn_like(current, requires_grad=True)
        action = torch.randn(2, 1, 1, 8, requires_grad=True)
        memories = DA3GiantEncoder.propagate_shallow_visual_slots_grad(
            enc, current, gradient_checkpointing=True, return_layer_patches=True)
        self.assertEqual(set(memories["layer_patches"]), {0, 1, 2, 3})
        sparse_current = DA3GiantEncoder.propagate_shallow_visual_slots_grad(
            enc, current, gradient_checkpointing=True, return_layer_patches=True,
            layer_patch_indices=[1, 3], layer_patch_mode="current",
        )
        self.assertEqual(set(sparse_current["layer_patches"]), {1, 3})
        self.assertEqual(sparse_current["layer_patches"][1].shape[-1], 8)
        read = CurrentGeometryRead(8, width=16, heads=2)
        args = dict(decode_visuals=False, return_multi_level=True,
                    current_geometry_by_layer=memories["layer_patches"], current_geometry_read=read)
        run = DA3GiantEncoder._propagate_shallow_with_actions_impl
        plain = run(enc, visual, action, **args)
        ckpt = run(enc, visual, action, gradient_checkpointing=True, **args)
        torch.testing.assert_close(plain["action_tokens"], ckpt["action_tokens"])
        early_grad = torch.autograd.grad(ckpt["level_feats"][0][0].sum(), current,
                                         retain_graph=True, allow_unused=True)[0]
        self.assertTrue(early_grad is None or early_grad.abs().sum() == 0)
        late_loss = ckpt["level_feats"][-1][0].square().mean()
        late_grad = torch.autograd.grad(late_loss, current, retain_graph=True)[0]
        self.assertGreater(late_grad.abs().sum().item(), 0)
        plain_grad = torch.autograd.grad(plain["level_feats"][-1][0].square().mean(), current)[0]
        torch.testing.assert_close(late_grad, plain_grad)
        sparse_read = CurrentGeometryRead(8, width=16, heads=2, memory_dim=8)
        sparse = run(
            enc, visual, action, decode_visuals=False,
            current_geometry_by_layer=sparse_current["layer_patches"],
            current_geometry_read=sparse_read,
        )
        self.assertEqual(sparse["action_tokens"].shape, (2, 8))

    def test_current_never_receives_future_or_actions_across_layers(self):
        block = Block()
        x = torch.randn(2, 2, 5, 8, requires_grad=True)
        altered = x.detach().clone()
        altered[:, 1] += torch.randn_like(altered[:, 1]) * 5
        altered[:, 0, 1] += torch.randn_like(altered[:, 0, 1]) * 5
        a, b = x, altered
        for global_attention in [False, True, False, True]:
            a = run_dual_masked_block(a, block, None, global_attention=global_attention)
            b = run_dual_masked_block(b, block, None, global_attention=global_attention)
        indices = [0, 2, 3, 4]
        torch.testing.assert_close(a[:, 0, indices], b[:, 0, indices], rtol=0, atol=0)
        grad = torch.autograd.grad(a[:, 0, indices].square().sum(), x, retain_graph=True)[0]
        self.assertEqual(grad[:, 1].abs().sum().item(), 0)
        self.assertEqual(grad[:, 0, 1].abs().sum().item(), 0)
        action_grad = torch.autograd.grad(a[:, :, 1].square().sum(), x)[0]
        self.assertGreater(action_grad[:, 0, indices].abs().sum().item(), 0)
        self.assertGreater(action_grad[:, 1, indices].abs().sum().item(), 0)

    def test_all_architectures_one_deep_pass_and_stop_transfer(self):
        x = torch.randn(2, 1, 1, 7, 16)
        for mode in MODES:
            with self.subTest(mode=mode):
                net = build(mode)
                kwargs = dict(reference_shallow=x, lang_feats=torch.randn(2, 3, 16),
                              lang_padding_mask=torch.ones(2, 3, dtype=torch.bool))
                result = net(x, **kwargs)
                self.assertEqual(net.da3.calls, 1)
                self.assertEqual(result["actions_norm"].shape, (2, 1, 5, 4))
                self.assertEqual(result["current_depth_output"]["depth"].shape, (2, 1, 2, 2))
                self.assertEqual(net.da3.last_mask, "action_bridge" if mode == "dual_action_bridge" else "full")
                (result["actions_norm"].square().mean() + result["current_depth_output"]["depth"].mean()).backward()
                self.assertIsNotNone(net.da3.scale.grad)
                if mode == "direct_current":
                    self.assertTrue(all(not p.requires_grad for p in net.predictor.parameters()))
                    self.assertGreater(net.direct_current_seed.output.weight.grad.abs().sum().item(), 0)
                if mode == "dual_predicted":
                    self.assertTrue((net.prediction_roles.grad.abs().sum(1) > 0).all())
                buf = io.BytesIO()
                torch.save({"geometry_architecture_state": architecture_state(net)}, buf)
                buf.seek(0)
                target = build(mode, stop=True)
                load_architecture_state(target, torch.load(buf, weights_only=False))
                output = target(x, **kwargs)
                self.assertEqual(output["stop_logits"].shape, (2, 1))
                with self.assertRaises(KeyError):
                    load_architecture_state(target, {})

    def test_feature_targets_and_dual_average(self):
        current = torch.ones(1, 1, 1, 4, 2, requires_grad=True)
        future = torch.full_like(current, 3, requires_grad=True)
        output = {"future_shallow": current.detach().clone().requires_grad_(),
                  "predicted_current_shallow": torch.full_like(current, 2, requires_grad=True)}
        kwargs = dict(patch_start=1, horizon_weights=[1], patch_weight=1, cls_weight=0, register_weight=0)
        self.assertEqual(architecture_feature_loss("current_prediction", output, current, future, **kwargs)["total"].item(), 0)
        self.assertEqual(architecture_feature_loss("direct_current", output, current, future, **kwargs)["total"].item(), 0)
        self.assertEqual(architecture_feature_loss("dual_observed", output, current, future, **kwargs)["total"].item(), 4)
        loss = architecture_feature_loss("dual_predicted", output, current, future, **kwargs)["total"]
        self.assertEqual(loss.item(), 2.5)
        loss.backward()
        self.assertIsNone(current.grad)
        self.assertIsNone(future.grad)

    def test_forward_batch_reuses_decoded_depth_and_selects_gt(self):
        b = 2
        batch = {"all_view_images": torch.ones(b, 2, 1, 3, 2, 2),
                 "episode_first_image": torch.ones(b, 1, 3, 2, 2),
                 "task_description": ["move", "turn"], "action_stats_key": ["sim"] * b,
                 "actions": torch.zeros(b, 1, 5, 4), "action_loss_mask": torch.ones(b, 1, 5, 4, dtype=torch.bool),
                 "gt_depth_meters": torch.stack([torch.full((b, 1, 2, 2), 2.), torch.full((b, 1, 2, 2), 7.)], 1),
                 "gt_depth_mask": torch.ones(b, 2, 1, 2, 2, dtype=torch.bool)}
        for mode in MODES:
            with self.subTest(mode=mode):
                net = build(mode)
                targets = []
                def depth_loss(prediction, target, *args, **kwargs):
                    targets.append(float(target.mean()))
                    loss = (prediction - target).abs().mean()
                    return dict(total=loss, l1=loss, grad=loss * 0, valid_ratio=loss * 0 + 1)
                with patch("experiments.uavflow_predictor_idm.objectives.encode_stage2_condition",
                           return_value={"last_hidden_state": torch.ones(b, 3, 16), "attention_mask": torch.ones(b, 3, dtype=torch.bool)}), \
                     patch("experiments.uavflow_predictor_idm.objectives.gam_window_pointnorm_depth_loss", side_effect=depth_loss):
                    losses = forward_batch(
                        model=net, da3=net.da3, text=None, normalizer=IdentityNormalizer(), pose_normalizer=None,
                        batch=batch, context_len=1, rollout_steps=1, feature_horizon_weights=[1],
                        feature_patch_weight=1, feature_cls_weight=0, feature_register_weight=0,
                        deep_feature_enabled=False, deep_feature_patch_weight=1, deep_feature_cls_weight=0,
                        stop_pos_weight=5, amp=False, depth_scale_mode="fixed_metric",
                        depth_target_mode="both" if mode.startswith("dual_") else "current",
                    )
                self.assertEqual(net.da3.calls, 1)
                self.assertEqual(targets, [7., 2.] if mode.startswith("dual_") else [2.])
                self.assertTrue(torch.isfinite(losses["depth"]))

    def test_view_split_layouts(self):
        for b in [1, 2, 3]:
            values = torch.arange(b * 2 * 4.).reshape(b, 2, 2, 2)
            for view in [0, 1]:
                batched = select_dual_view({"depth": values}, b, view)["depth"]
                flat = select_dual_view({"depth": values.flatten(0, 1)}, b, view)["depth"]
                torch.testing.assert_close(batched[:, 0], flat)


if __name__ == "__main__":
    unittest.main()
