import torch
import pytest
from torch import nn
from torch.utils.data.distributed import DistributedSampler

from experiments.uavflow_qwen_pure.model import PureQwenActionPolicy
from experiments.uavflow_qwen_pure.utils import masked_l1
from experiments.uavflow_qwen_pure.utils import CursorSampler, atomic_save


@pytest.mark.parametrize('rank', [0, 1])
def test_sampler_resume_seeks_exact_next_batch(rank):
    dataset = list(range(101))
    original = DistributedSampler(dataset, num_replicas=2, rank=rank, seed=42, shuffle=True)
    original.set_epoch(3)
    sampler = CursorSampler(dataset, num_replicas=2, rank=rank, seed=42, shuffle=True,
                            batch_size=4, resume_epoch=3, resume_batch=5)
    sampler.set_epoch(3)
    assert list(sampler) == list(original)[20:]
    assert len(sampler) == len(original) - 20
    sampler.set_epoch(4); original.set_epoch(4)
    assert list(sampler) == list(original)


def test_terminal_zero_is_normalized_like_other_real_labels():
    from robot.data.dataset import ActionNormalizer
    normalizer = ActionNormalizer({'uavflow_sim': dict(q01=[-1.] * 4, q99=[3.] * 4, mask=[True] * 4)},
                                  default_key='uavflow_sim')
    zero = torch.zeros(2, 1, 5, 4)
    normalized = normalizer.normalize(zero)
    torch.testing.assert_close(normalized, torch.full_like(zero, -0.5))
    torch.testing.assert_close(normalizer.denormalize(normalized), zero)
    assert masked_l1(normalized, normalized, torch.ones_like(zero, dtype=torch.bool)) == 0


def test_checkpoint_saves_only_trainable_and_rejects_wrong_layout(tmp_path):
    policy = PureQwenActionPolicy.__new__(PureQwenActionPolicy)
    nn.Module.__init__(policy)
    policy.frozen = nn.Linear(3, 3).requires_grad_(False)
    policy.head = nn.Linear(3, 4)
    frozen_before = policy.frozen.weight.detach().clone()
    state = policy.trainable_state()
    assert set(state) == {'head.weight', 'head.bias'}
    atomic_save(state, tmp_path / 'last.pt')
    with torch.no_grad():
        policy.head.weight.zero_()
    policy.load_trainable_state(torch.load(tmp_path / 'last.pt', weights_only=True))
    torch.testing.assert_close(policy.head.weight, state['head.weight'])
    torch.testing.assert_close(policy.frozen.weight, frozen_before)
    assert not (tmp_path / 'last.pt.tmp').exists()
    with pytest.raises(ValueError):
        policy.load_trainable_state({})


def test_numeric_pose_only_at_head_has_gradient_and_pose_response():
    class FakeEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 8).requires_grad_(False)

        def forward(self, images, instructions, current_states=None, omit_prompt_pose=False):
            assert omit_prompt_pose and current_states is None
            tokens = torch.tensor([[1, 2, 3, 4, 5]]).repeat(len(instructions), 1)
            return {'joint_layers': self.embedding(tokens)[:, None]}

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(16, 4)

        def forward_slots(self, slots):
            return self.linear(slots)

    policy = PureQwenActionPolicy.__new__(PureQwenActionPolicy)
    nn.Module.__init__(policy)
    policy.pose_mode = 'numeric_head'
    encoder = FakeEncoder()
    policy.encoder = encoder
    policy.register_buffer('state_scale', torch.tensor([1000., 1000., 1000., 180.]))
    policy.state_encoder = nn.Linear(4, 8)
    policy.projection = nn.Identity()
    policy.head = Head()
    a = policy(None, ['Turn left'], [[0., 0., 0., 0.]])
    b = policy(None, ['Turn left'], [[100., 0., 0., 90.]])
    assert a.shape == (1, 1, 5, 4)
    assert not torch.allclose(a, b)
    b.sum().backward()
    assert policy.state_encoder.weight.grad.abs().sum() > 0
    with pytest.raises(ValueError, match='finite'):
        policy(None, ['Turn left'], [[float('nan'), 0., 0., 0.]])


def test_numeric_token_pose_enters_qwen_not_decoder():
    from types import SimpleNamespace
    from robot.modeling.action_head_v2 import ActionHeadV2

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(8, 8).requires_grad_(False)
            self.qwen = SimpleNamespace(get_input_embeddings=lambda: self.embedding)

        def forward(self, images, instructions, current_states=None,
                    omit_prompt_pose=False, state_placeholder=False):
            assert omit_prompt_pose and state_placeholder and current_states is None
            embeddings = self.embedding(torch.tensor([[7, 1, 2, 3, 4, 5]]))
            return {'joint_layers': (embeddings[:, 1:] + embeddings[:, :1])[:, None]}

    policy = PureQwenActionPolicy.__new__(PureQwenActionPolicy)
    nn.Module.__init__(policy)
    policy.pose_mode = 'numeric_token'
    policy.encoder = Encoder()
    policy.state_token_id = 7
    policy.register_buffer('state_scale', torch.tensor([1000., 1000., 1000., 180.]))
    policy.state_encoder = nn.Sequential(nn.Linear(4, 16), nn.GELU(), nn.Linear(16, 8))
    policy.projection = nn.Identity()
    policy.head = ActionHeadV2(input_dim=8, hidden_dim=8, chunk_size=5, n_dims=4,
                               n_views=1, chunk_position_encoding='learned')
    a = policy(None, ['Turn left'], [[0., 0., 0., 0.]])
    b = policy(None, ['Turn left'], [[100., 0., 0., 90.]])
    assert a.shape == b.shape == (1, 1, 5, 4)
    assert not torch.allclose(a, b)
    b.sum().backward()
    assert policy.state_encoder[0].weight.grad.abs().sum() > 0
    assert not policy.encoder.embedding._forward_hooks
