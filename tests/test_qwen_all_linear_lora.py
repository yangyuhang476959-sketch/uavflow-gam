import torch
from torch import nn
from experiments.uavflow_direct_visual_probe.qwen_lora import inject_qwen_lora
from robot.modeling.lora import LoRALinear


class MiniQwen(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.language_model = nn.ModuleDict({
            name: nn.Linear(4, 4) for name in (
                'q_proj', 'gate_proj', 'in_proj_qkv', 'in_proj_z',
                'in_proj_a', 'in_proj_b', 'out_proj',
            )
        })
        self.model.visual = nn.ModuleDict({
            'qkv': nn.Linear(4, 4), 'linear_fc1': nn.Linear(4, 4),
            'linear_fc2': nn.Linear(4, 4), 'norm': nn.LayerNorm(4),
            'patch': nn.Conv2d(3, 4, 1),
        })
        self.embed_tokens = nn.Embedding(10, 4)
        self.lm_head = nn.Linear(4, 10)
        self.requires_grad_(False)

    def get_output_embeddings(self):
        return self.lm_head


def test_all_linear_covers_visual_adapter_and_linear_attention():
    qwen = MiniQwen()
    x = torch.randn(3, 4)
    expected = {n: m(x).detach() for n, m in qwen.named_modules()
                if isinstance(m, nn.Linear) and m is not qwen.lm_head}
    names = inject_qwen_lora(qwen, rank=2, alpha=16, dropout=0)
    assert set(names) == set(expected)
    loss = 0
    for name in names:
        layer = qwen.get_submodule(name)
        assert isinstance(layer, LoRALinear)
        assert layer.scaling == 8
        assert not any(p.requires_grad for p in layer.base.parameters())
        torch.testing.assert_close(layer(x), expected[name])
        loss = loss + layer(x).square().sum()
    loss.backward()
    for name in names:
        layer = qwen.get_submodule(name)
        assert layer.lora_B.grad is not None
        assert layer.lora_B.grad.abs().sum() > 0
    assert isinstance(qwen.lm_head, nn.Linear)
    assert not qwen.lm_head.weight.requires_grad
    assert not qwen.embed_tokens.weight.requires_grad
    assert isinstance(qwen.model.visual['patch'], nn.Conv2d)


def test_legacy_scope_preserves_old_checkpoint_keys():
    qwen = MiniQwen()
    names = inject_qwen_lora(qwen, rank=2, scope='legacy_language')
    assert set(names) == {'q_proj', 'gate_proj'}
    assert 'model.language_model.q_proj.lora_A' in qwen.state_dict()
    assert isinstance(qwen.model.language_model['in_proj_qkv'], nn.Linear)
    assert isinstance(qwen.model.visual['linear_fc1'], nn.Linear)


def test_openvla_gaussian_initialization():
    torch.manual_seed(0)
    qwen = MiniQwen()
    names = inject_qwen_lora(qwen, rank=128)
    values = torch.cat([qwen.get_submodule(n).lora_A.flatten() for n in names])
    assert abs(values.std().item() - 1 / 128) < .001
    assert all(torch.count_nonzero(qwen.get_submodule(n).lora_B) == 0 for n in names)
