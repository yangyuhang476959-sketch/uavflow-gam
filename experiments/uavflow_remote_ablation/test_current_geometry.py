"""CPU-only regression tests; no pretrained weights, data, UE or GPUs needed.

PYTHONPATH=src:. python -m unittest experiments.uavflow_remote_ablation.test_current_geometry -v
"""
import csv
import io
import types
import unittest
from pathlib import Path
import torch
from torch import nn
from experiments.uavflow_predictor_idm.current_geometry import CurrentGeometryRead, load_current_geometry_read
from experiments.uavflow_predictor_idm.model import UAVFlowPredictorIDM


class FakeDA3(nn.Module):
    embed_dim = 16
    num_register_tokens = 2

    def __init__(self):
        super().__init__()
        self.dpt_head = nn.Identity()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.current_calls = 0

    def propagate_shallow_with_actions_grad(self, future, action, **kwargs):
        if "current_geometry_read" in kwargs:
            for patches in kwargs["current_geometry_by_layer"].values():
                action = kwargs["current_geometry_read"](action, patches)
        return {"action_tokens": action * self.scale}

    def propagate_shallow_visual_slots_grad(self, observed, **kwargs):
        self.current_calls += 1
        level = torch.cat([observed, observed], -1) * self.scale
        return {"deep_levels": [level], "layer_patches": {0: level[..., 3:, :], 1: level[..., 3:, :]}}


def model(enabled):
    result = UAVFlowPredictorIDM(
        da3=FakeDA3(), idm=None, action_dim=4, action_chunk_size=5,
        d_model=256, depth=1, num_heads=4, language_dim=16,
        direct_action_enabled=True, deep_action_enabled=True,
        compute_idm_branch=False, gradient_checkpointing=False,
        current_geometry_action_enabled=enabled,
    )
    # Isolate deep/action wiring from Predictor internals.
    def rollout(self, observed, **kwargs):
        return observed + 0.1, observed[..., 0, :]
    result.rollout_shallow = types.MethodType(rollout, result)
    return result


class Tests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        torch.set_num_threads(2)

    def test_per_layer_routing_and_checkpoint_rejection(self):
        net = model(True)
        net.current_geometry_read_mode = "per_layer"
        x = torch.randn(2, 1, 1, 7, 16, requires_grad=True)
        out = net(x, reference_shallow=x, lang_feats=torch.zeros(2, 2, 16), lang_padding_mask=None)
        self.assertEqual(net.da3.current_calls, 1)
        self.assertIsNotNone(out["current_geometry_features"])
        out["actions_norm"].square().mean().backward()
        self.assertGreater(net.current_geometry_read.gate.grad.abs().item(), 0)
        with self.assertRaisesRegex(ValueError, "read mode changed"):
            load_current_geometry_read(net, {"current_geometry_read": net.current_geometry_read.state_dict()}, required=True)
        load_current_geometry_read(net, {
            "current_geometry_read_mode": "per_layer",
            "current_geometry_read": net.current_geometry_read.state_dict(),
        }, required=True)

    def test_read_shape_gradients_and_identity(self):
        read = CurrentGeometryRead(16, width=32, heads=4)
        actions = torch.randn(2, 1, 1, 16, requires_grad=True)
        patches = torch.randn(2, 1, 1, 7, 32, requires_grad=True)
        result = read(actions, patches)
        result.square().mean().backward()
        self.assertGreater(patches.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in read.parameters()))
        with torch.no_grad():
            read.gate.zero_()
        self.assertTrue(torch.equal(read(actions, patches), actions))

    def test_current_only_memory_is_half_width(self):
        read = CurrentGeometryRead(16, width=32, heads=4, memory_dim=16)
        actions = torch.randn(2, 1, 1, 5, 16, requires_grad=True)
        patches = torch.randn(2, 1, 1, 7, 16, requires_grad=True)
        result = read(actions, patches, layer_index=19)
        self.assertEqual(result.shape, actions.shape)
        result.square().mean().backward()
        self.assertGreater(patches.grad.abs().sum().item(), 0)

    def test_forward_cache_and_unchanged_old_parameters(self):
        torch.manual_seed(7)
        baseline = model(False)
        torch.manual_seed(7)
        changed = model(True)
        for name, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, changed.state_dict()[name]), name)
        x = torch.randn(2, 1, 1, 7, 16, requires_grad=True)
        kwargs = dict(reference_shallow=x, lang_feats=torch.zeros(2, 2, 16), lang_padding_mask=None)
        base = baseline(x, **kwargs)
        output = changed(x, **kwargs)
        self.assertEqual(baseline.da3.current_calls, 0)
        self.assertEqual(changed.da3.current_calls, 1)
        self.assertIsNone(base["current_geometry_features"])
        self.assertEqual(output["actions_norm"].shape, (2, 1, 5, 4))
        self.assertEqual(output["current_geometry_features"]["deep_levels"][-1].shape, (2, 1, 1, 7, 32))
        output["actions_norm"].square().mean().backward()
        self.assertGreater(changed.current_geometry_read.gate.grad.abs().item(), 0)
        self.assertGreater(changed.da3.scale.grad.abs().item(), 0)
        with self.assertRaisesRegex(ValueError, "H=1"):
            changed(x.expand(-1, 2, -1, -1, -1), **kwargs)

    def test_checkpoint_transfer_and_optimizer_resume(self):
        first = model(True)
        optimizer = torch.optim.AdamW(first.current_geometry_read.parameters(), lr=1e-4)
        a = torch.randn(1, 1, 1, 16)
        p = torch.randn(1, 1, 1, 7, 32)
        first.current_geometry_read(a, p).sum().backward()
        optimizer.step()
        buffer = io.BytesIO()
        torch.save({"current_geometry_read": first.current_geometry_read.state_dict(),
                    "current_geometry_bank_mode": first.current_geometry_bank_mode,
                    "optimizer": optimizer.state_dict()}, buffer)
        buffer.seek(0)
        ckpt = torch.load(buffer, weights_only=False)
        second = model(True)
        load_current_geometry_read(second, ckpt, required=True)
        resumed = torch.optim.AdamW(second.current_geometry_read.parameters(), lr=1e-4)
        resumed.load_state_dict(ckpt["optimizer"])
        self.assertTrue(torch.equal(first.current_geometry_read(a, p), second.current_geometry_read(a, p)))
        self.assertEqual(len(optimizer.state), len(resumed.state))
        with self.assertRaises(KeyError):
            load_current_geometry_read(second, {}, required=True)
        load_current_geometry_read(model(False), {}, required=True)

    def test_v2_matrix_is_exact_and_residual_is_matched(self):
        root = Path(__file__).parent
        with (root / "compact_matrix.tsv").open() as stream:
            rows = list(csv.DictReader(stream, delimiter="\t"))
        self.assertEqual(
            [row["id"] for row in rows],
            ["G0", "G1", "C0", "C1", "Q0", "S0", "S1", "S2", "R1", "DV"],
        )
        s2 = next(row for row in rows if row["id"] == "S2")
        r1 = next(row for row in rows if row["id"] == "R1")
        for key in s2.keys() - {"id", "purpose", "action_decode"}:
            self.assertEqual(s2[key], r1[key])
        self.assertEqual(r1["action_decode"], "geometry_residual")


if __name__ == "__main__":
    unittest.main()
