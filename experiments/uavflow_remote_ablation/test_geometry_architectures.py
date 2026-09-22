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
    architecture_state, load_architecture_state, select_dual_view,
)
from experiments.uavflow_predictor_idm.objectives import architecture_feature_loss, forward_batch
from robot.modeling.dual_geometry_attention import dual_allow_mask, run_dual_masked_block


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


def build(mode, stop=False):
    net = UAVFlowPredictorIDM(
        da3=TinyDA3(), idm=None, action_dim=4, action_chunk_size=5,
        d_model=256, depth=1, num_heads=4, language_dim=16, language_len=2,
        direct_action_enabled=True, deep_action_enabled=True, compute_idm_branch=False,
        gradient_checkpointing=False, geometry_architecture=mode,
        depth_decode_enabled=True, stop_head_enabled=stop,
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
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)

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
        root = Path(__file__).parent
        script = (root / "run_remote.sh").read_text()
        variants = script.split("variant_overrides() {", 1)[1].split("latest_checkpoint()", 1)[0]
        variants = "variant_overrides() {" + variants
        outputs = "output_id_for() {" + script.split("output_id_for() {", 1)[1].split("run_stage()", 1)[0]
        with (root / "compact_matrix.tsv").open() as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        for row in rows:
            result = subprocess.check_output(["bash", "-c", variants + '\nvariant_overrides "$1"', "test", row["id"]], text=True)
            flags = dict(line.split("=", 1) for line in result.splitlines())
            self.assertEqual(flags.get("model.geometry_architecture", "legacy"), row["geometry_architecture"])
        for run_id in ["H0", "HB", "C2_D2HB", "C3_W3HB", "C5_F10HB"]:
            name = subprocess.check_output(["bash", "-c", outputs + '\noutput_id_for "$1"', "test", run_id], text=True).strip()
            self.assertEqual(name, run_id + "_g2")

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
