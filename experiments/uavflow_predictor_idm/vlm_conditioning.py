"""Frozen text or joint vision-language conditioning for UAV-Flow GAM."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from robot.modeling.conditioning import TextConditioner
from experiments.uavflow_direct_visual_probe.qwen35_semantic import (
    FrozenQwen35SemanticEncoder,
)


class FrozenQwen35Conditioner(nn.Module):
    """Expose fixed-length joint F0/Ft/instruction tokens to GAM.

    DA3 remains the visual/predictive backbone.  These frozen Qwen tokens are
    only the K/V memory read by GAM's existing per-block language
    cross-attention, which makes this a true T5-conditioning replacement rather
    than the earlier VLM-centred direct-action probe.
    """

    is_multimodal = True
    # Qwen already supplies positionalized joint image/text tokens.  Keep the
    # complete sequence and let the processor pad only to the longest sample
    # in the current minibatch; GAM masks that transient batch padding.
    variable_length_tokens = True

    def __init__(
        self,
        model_name: str,
        *,
        layer_indices: Sequence[int] = (7, 15, 23),
        attention_implementation: str = "sdpa",
        use_reference_image: bool = True,
        prompt_mode: str = "temporal_pair",
        token_selection: str = "all",
        lora_enabled: bool = False,
        lora_rank: int = 32,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_scope: str = "all_linear",
        action_placeholder_count: int = 0,
        action_attention_mode: str = "causal",
        numeric_pose_token: bool = False,
        pose_progress_enabled: bool = False,
        pose5_enabled: bool = False,
        pose_init_seed: int | None = None,
    ) -> None:
        super().__init__()
        self.encoder = FrozenQwen35SemanticEncoder(
            model_name,
            layer_indices=layer_indices,
            attention_implementation=attention_implementation,
            prompt_mode=prompt_mode,
            token_selection=token_selection,
            lora_enabled=lora_enabled,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            lora_scope=lora_scope,
            action_placeholder_count=action_placeholder_count,
            action_attention_mode=action_attention_mode,
        )
        if pose5_enabled:
            self.encoder.qwen.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
            self.encoder.qwen.config.use_cache=False
        self.use_reference_image = bool(use_reference_image)
        self.hidden_size = int(self.encoder.hidden_size)
        self.pose5_enabled=bool(pose5_enabled)
        if self.pose5_enabled and (pose_progress_enabled or not numeric_pose_token):raise ValueError('5D Pose requires numeric token and disables 7D progress input')
        self.numeric_pose_token=bool(numeric_pose_token)
        self.pose_progress_enabled=bool(pose_progress_enabled)
        if self.pose_progress_enabled and not self.numeric_pose_token:
            raise ValueError('Pose progress requires numeric_pose_token')
        if self.numeric_pose_token:
            tokenizer=self.encoder.processor.tokenizer
            tokenizer.add_special_tokens({'additional_special_tokens':self.encoder.action_placeholder_tokens+['<|uav_state|>']})
            self.encoder.qwen.resize_token_embeddings(len(tokenizer))
            self.encoder.qwen.get_input_embeddings().requires_grad_(False)
            self.encoder.qwen.get_output_embeddings().requires_grad_(False)
            self.state_token_id=tokenizer.convert_tokens_to_ids('<|uav_state|>')
            scale=([10.,10.,10.,1.,1.,2*torch.pi,10.] if self.pose_progress_enabled
                   else ([1000.,1000.,1000.,180.,1.] if self.pose5_enabled else [1000.,1000.,1000.,180.]))
            self.register_buffer('state_scale',torch.tensor(scale))
            if pose_init_seed is None:
                self.state_encoder=nn.Sequential(nn.Linear(len(scale),256),nn.GELU(),nn.Linear(256,self.hidden_size))
            else:
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(pose_init_seed);first=nn.Linear(len(scale),256)
                    torch.manual_seed(pose_init_seed+1);second=nn.Linear(256,self.hidden_size)
                    self.state_encoder=nn.Sequential(first,nn.GELU(),second)

    def train(self, mode=True):
        super().train(mode)
        if getattr(self,'pose5_enabled',False):
            self.encoder.qwen.eval()
            if mode:
                visual=self.encoder.qwen.model.visual;visual.train()
                for i,block in enumerate(visual.blocks):block.gradient_checkpointing=i<8
        return self

    def encode_tokens(
        self,
        texts: Sequence[str],
        *,
        pad_to: int | None = None,
        images: torch.Tensor | None = None,
        current_pose: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if images is None:
            raise ValueError(
                "Qwen VLM conditioning requires images=[F0,Ft] shaped "
                "[B,2,V,3,H,W]."
            )
        current_states = None
        if not self.numeric_pose_token and self.encoder.prompt_mode in {
            "current_image_action_question",
            "current_image_openvla",
        }:
            if current_pose is None:
                raise ValueError(
                    f"{self.encoder.prompt_mode} requires current_pose for "
                    "the Current State prompt field."
                )
            pose = current_pose.detach().float()
            if pose.ndim != 2 or pose.shape[0] != len(texts):
                raise ValueError(
                    f"Expected current_pose [B,4/5], got {tuple(pose.shape)}."
                )
            if pose.shape[-1] == 5:
                yaw_deg = torch.rad2deg(torch.atan2(pose[:, 3], pose[:, 4]))
                pose = torch.cat([pose[:, :3], yaw_deg[:, None]], dim=-1)
            elif pose.shape[-1] != 4:
                raise ValueError(
                    f"Expected pose [x,y,z,yaw_deg] or [x,y,z,sin,cos], got "
                    f"{tuple(pose.shape)}."
                )
            current_states = pose.cpu().tolist()
        if self.numeric_pose_token:
            dimension=self.state_scale.numel()
            if current_pose is None or current_pose.shape!=(len(texts),dimension) or not torch.isfinite(current_pose).all():
                raise ValueError(f'Numeric pose token requires finite [B,{dimension}] canonical state')
            state=current_pose.to(self.state_scale.device,dtype=torch.float32)
            embedding=self.state_encoder(state/self.state_scale)
            def inject(_module,inputs,output):
                mask=inputs[0].eq(self.state_token_id)
                if not mask.sum(1).eq(1).all():raise ValueError('Expected one numeric pose token')
                return torch.where(mask[:,:,None],embedding.to(output.dtype)[:,None],output)
            hook=self.encoder.qwen.get_input_embeddings().register_forward_hook(inject)
            try:
                encoded=self.encoder(images,texts,omit_prompt_pose=True,state_placeholder=True)
            finally:
                hook.remove()
        else:
            encoded = self.encoder(images, texts, current_states=current_states)
        # A one-element layer list returns the true final-layer representation;
        # legacy multi-layer configs retain their historical equal average.
        hidden = encoded["joint_layers"].float().mean(dim=1).to(
            dtype=encoded["joint_layers"].dtype
        )
        keep = encoded["joint_mask"].bool()
        # Do not truncate or extend the VLM sequence to GAM's historical
        # fixed T5/CLIP language length. `hidden` is already batch-padded by
        # the Qwen processor and `keep` prevents those padding positions from
        # entering cross-attention.
        result = {"last_hidden_state": hidden, "attention_mask": keep}
        if self.numeric_pose_token and getattr(self,'pose5_enabled',False):
            # GAM reuses this SAME public MLP output at its native SA proprio
            # slot. Ordinary Warp/Pure consumers ignore the additional field.
            result['pose_embedding']=embedding
        # Preserve prompt token ids for diagnostics (for example mapping a
        # noun in a closed-loop instruction back to GAM's cross-attention K/V
        # positions).  Training and policy callers ignore these extra fields.
        for key in ("input_ids", "image_token_mask", "image_grid_thw"):
            if key in encoded and encoded[key] is not None:
                result[key] = encoded[key]
        return result


def build_stage2_conditioner(stage1_cfg, model_cfg) -> nn.Module:
    encoder_type = str(stage1_cfg.get("text_encoder_type", "clip")).lower()
    if stage1_cfg.get('paper_t5_pose_token',False):
        if encoder_type!='t5':raise ValueError('paper_t5_pose_token requires T5')
        from experiments.uavflow_paper_ablation.t5_pose import T5PoseConditioner
        return T5PoseConditioner(str(stage1_cfg.get('t5_model')),int(stage1_cfg.get('qwen_pose_init_seed',43)),
                                int(stage1_cfg.get('paper_t5_projection_seed',46)),native_text=bool(stage1_cfg.get('paper_t5_native_text',False)))
    if encoder_type in {"qwen", "qwen3_5", "qwen3.5", "vlm"}:
        return FrozenQwen35Conditioner(
            str(stage1_cfg.get("qwen_model")),
            layer_indices=tuple(
                int(value) for value in stage1_cfg.get("qwen_layers", [7, 15, 23])
            ),
            attention_implementation=str(
                stage1_cfg.get("qwen_attention_implementation", "sdpa")
            ),
            use_reference_image=bool(
                stage1_cfg.get("qwen_use_reference_image", True)
            ),
            prompt_mode=str(stage1_cfg.get("qwen_prompt_mode", "temporal_pair")),
            token_selection=str(stage1_cfg.get("qwen_token_selection", "all")),
            lora_enabled=bool(stage1_cfg.get("qwen_lora_enabled", False)),
            lora_rank=int(stage1_cfg.get("qwen_lora_rank", 32)),
            lora_alpha=float(stage1_cfg.get("qwen_lora_alpha", 16)),
            lora_dropout=float(stage1_cfg.get("qwen_lora_dropout", 0.0)),
            # Old checkpoints predate scope metadata; keep their exact module
            # names/shapes instead of silently introducing new adapters.
            lora_scope=str(stage1_cfg.get("qwen_lora_scope", "legacy_language")),
            action_placeholder_count=int(
                stage1_cfg.get("qwen_action_placeholder_count", 0)
            ),
            action_attention_mode=str(
                stage1_cfg.get("qwen_action_attention_mode", "causal")
            ),
            numeric_pose_token=bool(stage1_cfg.get('qwen_numeric_pose_token',False)),
            pose_progress_enabled=bool(stage1_cfg.get('qwen_pose_progress_enabled',False)),
            pose5_enabled=bool(stage1_cfg.get('qwen_pose5_enabled',False)),
            pose_init_seed=stage1_cfg.get('qwen_pose_init_seed'),
        )
    return TextConditioner(
        encoder_type=encoder_type,
        clip_model=str(stage1_cfg.get("clip_model", "openai/clip-vit-large-patch14")),
        t5_model=str(stage1_cfg.get("t5_model", "google-t5/t5-base")),
        proj_dim=768,
        cache_token_embeddings=True,
        cache_max_entries=4096,
        cache_device="cpu",
    )


def encode_stage2_condition(
    conditioner: nn.Module,
    texts: Sequence[str],
    *,
    reference_images: torch.Tensor,
    current_images: torch.Tensor,
    pad_to: int,
    current_pose: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    if bool(getattr(conditioner, "is_multimodal", False)):
        images = (
            torch.stack([reference_images, current_images], dim=1)
            if bool(getattr(conditioner, "use_reference_image", True))
            else current_images[:, None]
        )
        return conditioner.encode_tokens(
            texts, pad_to=pad_to, images=images, current_pose=current_pose
        )
    if getattr(conditioner,'pose_conditioned',False):
        return conditioner.encode_tokens(texts,pad_to=pad_to,current_pose=current_pose)
    return conditioner.encode_tokens(texts, pad_to=pad_to)
