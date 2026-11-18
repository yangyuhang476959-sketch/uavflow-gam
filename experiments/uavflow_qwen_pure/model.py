"""One current image -> five native causal Qwen action slots -> continuous actions."""
import torch
from torch import nn

from experiments.uavflow_direct_visual_probe.qwen35_semantic import FrozenQwen35SemanticEncoder
from robot.modeling.action_head_v2 import ActionHeadV2


class PureQwenActionPolicy(nn.Module):
    def __init__(self, model_path, decoder_width=1536, pose_mode='prompt', *, numeric_pose_dim=4, numeric_pose_scale=None, pose_init_seed=None):
        super().__init__()
        if pose_mode not in {'none', 'prompt', 'numeric_head', 'numeric_token'}:
            raise ValueError('pose_mode must be none, prompt, numeric_head or numeric_token')
        self.pose_mode = pose_mode
        self.encoder = FrozenQwen35SemanticEncoder(
            model_path, layer_indices=(23,), prompt_mode='current_image_action_question',
            token_selection='action_placeholders', action_placeholder_count=5,
            lora_enabled=True, lora_rank=32, lora_alpha=16, lora_dropout=0,
            lora_scope='all_linear', action_attention_mode='causal')
        self.encoder.qwen.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
        self.encoder.qwen.config.use_cache = False
        if pose_mode in {'numeric_head', 'numeric_token'}:
            # Input is episode-relative centimetres/degrees, as in the prompt
            # baseline. Fixed scaling changes representation, not coordinates.
            scale = [1000., 1000., 1000., 180.] if numeric_pose_scale is None else list(numeric_pose_scale)
            if numeric_pose_dim < 4 or len(scale) != numeric_pose_dim or any(not __import__('math').isfinite(v) or v <= 0 for v in scale):
                raise ValueError('Numeric Pose needs one positive finite scale per dimension')
            self.register_buffer('state_scale', torch.tensor(scale))
            if pose_init_seed is None:
                self.state_encoder = nn.Sequential(nn.Linear(numeric_pose_dim, 256), nn.GELU(),
                    nn.Linear(256, 256 if pose_mode == 'numeric_head' else self.encoder.hidden_size))
            else:
                # Ordinary fan-in initialization for EVERY input column. Isolate
                # RNG per layer so common-shaped layers and the action head keep
                # identical initialization across representation dimensions.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(pose_init_seed)
                    first = nn.Linear(numeric_pose_dim, 256)
                    torch.manual_seed(pose_init_seed + 1)
                    second = nn.Linear(256, 256 if pose_mode == 'numeric_head' else self.encoder.hidden_size)
                    self.state_encoder = nn.Sequential(first, nn.GELU(), second)
        if pose_mode == 'numeric_token':
            tokenizer = self.encoder.processor.tokenizer
            tokenizer.add_special_tokens({'additional_special_tokens':
                                          self.encoder.action_placeholder_tokens + ['<|uav_state|>']})
            self.encoder.qwen.resize_token_embeddings(len(tokenizer))
            self.encoder.qwen.get_input_embeddings().requires_grad_(False)
            self.encoder.qwen.get_output_embeddings().requires_grad_(False)
            self.state_token_id = tokenizer.convert_tokens_to_ids('<|uav_state|>')
        self.projection = nn.Sequential(nn.LayerNorm(self.encoder.hidden_size),
                                       nn.Linear(self.encoder.hidden_size, decoder_width))
        self.head = ActionHeadV2(input_dim=decoder_width + (256 if pose_mode == 'numeric_head' else 0),
                                hidden_dim=decoder_width,
                                n_views=1, n_dims=4, chunk_size=5,
                                chunk_position_encoding='learned')

    def forward(self, images, instructions, states=None, *, return_hidden=False):
        if self.pose_mode == 'none':
            # Do not inspect states, create a state token, or substitute zeros.
            encoded = self.encoder(images, instructions, omit_prompt_pose=True)
        elif self.pose_mode in {'numeric_head', 'numeric_token'}:
            state = torch.as_tensor(states, device=self.state_scale.device, dtype=torch.float32)
            if state.shape != (len(instructions), self.state_scale.numel()) or not torch.isfinite(state).all():
                raise ValueError('Numeric state shape/values do not match configured Pose scales')
            if self.pose_mode == 'numeric_token':
                pose_embedding = self.state_encoder(state / self.state_scale)
                def inject(_module, inputs, output):
                    mask = inputs[0].eq(self.state_token_id)
                    if not mask.sum(1).eq(1).all():
                        raise ValueError('Expected exactly one numeric pose token per sample')
                    return torch.where(mask.unsqueeze(-1), pose_embedding.to(output.dtype)[:, None], output)
                hook = self.encoder.qwen.get_input_embeddings().register_forward_hook(inject)
                try:
                    encoded = self.encoder(images, instructions, omit_prompt_pose=True, state_placeholder=True)
                finally:
                    hook.remove()
            else:
                encoded = self.encoder(images, instructions, omit_prompt_pose=True)
        else:
            encoded = self.encoder(images, instructions, current_states=states)
        slots = self.projection(encoded['joint_layers'][:, 0])
        if self.pose_mode == 'numeric_head':
            pose_features = self.state_encoder(state / self.state_scale)
            slots = torch.cat((slots, pose_features[:, None].expand(-1, slots.shape[1], -1)), dim=-1)
        actions = self.head.forward_slots(slots[:, None])
        return (actions, slots[:, None]) if return_hidden else actions

    def optimizer_groups(self, lora_lr, head_lr, pose_lr=None):
        if pose_lr is not None:
            if self.pose_mode not in {'numeric_head', 'numeric_token'}:
                raise ValueError('Separate Pose LR requires a numeric Pose path')
            return [dict(name='qwen_lora', params=[p for p in self.encoder.parameters() if p.requires_grad], lr=lora_lr),
                    dict(name='action_head', params=list(self.projection.parameters()) + list(self.head.parameters()), lr=head_lr),
                    dict(name='pose_encoder', params=list(self.state_encoder.parameters()), lr=pose_lr)]
        return [dict(params=[p for p in self.encoder.parameters() if p.requires_grad], lr=lora_lr),
                dict(params=list(self.projection.parameters()) + list(self.head.parameters())
                     + (list(self.state_encoder.parameters()) if self.pose_mode in {'numeric_head', 'numeric_token'} else []), lr=head_lr)]

    def trainable_state(self):
        return {name: p.detach().cpu().clone() for name, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state(self, state):
        parameters = {name: p for name, p in self.named_parameters() if p.requires_grad}
        if parameters.keys() != state.keys():
            raise ValueError('Checkpoint trainable parameter names differ from pure Qwen model')
        with torch.no_grad():
            for name, p in parameters.items():
                p.copy_(state[name])
