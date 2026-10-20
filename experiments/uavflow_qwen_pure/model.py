"""One current image -> five native causal Qwen action slots -> continuous actions."""
import torch
from torch import nn

from experiments.uavflow_direct_visual_probe.qwen35_semantic import FrozenQwen35SemanticEncoder
from robot.modeling.action_head_v2 import ActionHeadV2


class PureQwenActionPolicy(nn.Module):
    def __init__(self, model_path, decoder_width=1536, pose_mode='prompt', pose_progress_enabled=False):
        super().__init__()
        if pose_mode not in {'prompt', 'numeric_head', 'numeric_token'}:
            raise ValueError('pose_mode must be prompt, numeric_head or numeric_token')
        self.pose_mode = pose_mode
        self.pose_progress_enabled = bool(pose_progress_enabled)
        if self.pose_progress_enabled and pose_mode != "numeric_token":
            raise ValueError("Pose progress requires numeric_token mode")
        self.encoder = FrozenQwen35SemanticEncoder(
            model_path, layer_indices=(23,), prompt_mode='current_image_action_question',
            token_selection='action_placeholders', action_placeholder_count=5,
            lora_enabled=True, lora_rank=32, lora_alpha=16, lora_dropout=0,
            lora_scope='all_linear', action_attention_mode='causal')
        self.encoder.qwen.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={'use_reentrant': False})
        self.encoder.qwen.config.use_cache = False
        if pose_mode in {'numeric_head', 'numeric_token'}:
            # Match correction-token progress scales; preserve the legacy4D
            # centimetre/degree representation when progress is disabled.
            scale = ([10., 10., 10., 1., 1., 2 * torch.pi, 10.]
                     if self.pose_progress_enabled else [1000., 1000., 1000., 180.])
            self.register_buffer('state_scale', torch.tensor(scale))
            self.state_encoder = nn.Sequential(nn.Linear(len(scale), 256), nn.GELU(),
                                               nn.Linear(256, 256 if pose_mode == 'numeric_head' else self.encoder.hidden_size))
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

    def forward(self, images, instructions, states):
        if self.pose_mode in {'numeric_head', 'numeric_token'}:
            state = torch.as_tensor(states, device=self.state_scale.device, dtype=torch.float32)
            dimension = self.state_scale.numel()
            if state.shape != (len(instructions), dimension) or not torch.isfinite(state).all():
                raise ValueError(f'Numeric state must be finite [B,{dimension}] in the configured pose convention')
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
        return self.head.forward_slots(slots[:, None])

    def optimizer_groups(self, lora_lr, head_lr, state_lr=None):
        state = list(self.state_encoder.parameters()) if self.pose_mode in {'numeric_head', 'numeric_token'} else []
        groups = [dict(params=[p for p in self.encoder.parameters() if p.requires_grad], lr=lora_lr),
                  dict(params=list(self.projection.parameters()) + list(self.head.parameters()), lr=head_lr)]
        if getattr(self, 'pose_progress_enabled', False):
            groups.append(dict(params=state, lr=lora_lr if state_lr is None else state_lr))
        else:
            # Preserve legacy 4D checkpoint optimizer layout (two groups).
            groups[1]['params'] += state
        return groups

    def trainable_state(self):
        return {name: p.detach().cpu().clone() for name, p in self.named_parameters() if p.requires_grad}

    def load_trainable_state(self, state):
        parameters = {name: p for name, p in self.named_parameters() if p.requires_grad}
        if parameters.keys() != state.keys():
            raise ValueError('Checkpoint trainable parameter names differ from pure Qwen model')
        with torch.no_grad():
            for name, p in parameters.items():
                p.copy_(state[name])
