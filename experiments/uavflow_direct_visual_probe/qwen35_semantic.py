"""Frozen Qwen3.5 multimodal features for the DA3 action probe.

The prompt follows OpenVLA-UAV's concise action question, with only one extra
hint that identifies the first of the two images as the episode's first frame.
Only selected language-model layers are retained. Qwen is frozen by default;
the optional dependency-free LoRA path keeps only its low-rank adapters
trainable and preserves their gradients through feature extraction.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext
import os
from types import MethodType

import torch
from PIL import Image
from torch import nn
from robot.modeling.lora import LoRALinear
from experiments.uavflow_direct_visual_probe.qwen_lora import inject_qwen_lora
from experiments.uavflow_predictor_idm.runtime import profile_phase


def _qwen_fla_npu_requested() -> bool:
    value = os.environ.get("UAVFLOW_QWEN_FLA_NPU", "0").strip().lower()
    if value not in {"", "0", "1", "false", "true", "no", "yes", "off", "on"}:
        raise ValueError(f"Invalid UAVFLOW_QWEN_FLA_NPU value: {value!r}")
    enabled = value in {"1", "true", "yes", "on"}
    accelerator = os.environ.get("UAVFLOW_ACCELERATOR", "auto").strip().lower()
    return enabled and accelerator in {"npu", "ascend"}


def _patch_qwen_fla_npu(qwen: nn.Module) -> int:
    """Patch only full-sequence GatedDeltaNet kernels with Ascend FLA.

    Imports are intentionally lazy: CUDA/CPU installations never need
    torch_npu, Triton-Ascend, or the Ascend FLA checkout. Decode/cache keeps
    HuggingFace's recurrent kernel unchanged.
    """
    if not _qwen_fla_npu_requested():
        return 0
    try:
        from fla.modules.convolution import causal_conv1d as fla_causal_conv1d
        from fla.ops.gated_delta_rule import (
            chunk_gated_delta_rule as fla_chunk_gated_delta_rule,
        )
    except ImportError as exc:
        raise RuntimeError(
            "UAVFLOW_QWEN_FLA_NPU=1 requires the pinned Ascend FLA checkout "
            "and isolated triton-ascend target on PYTHONPATH."
        ) from exc

    def causal_conv_wrapper(*, x, weight, bias=None, activation=None, seq_idx=None):
        if seq_idx is not None:
            raise ValueError("Ascend FLA training wrapper does not support seq_idx")
        result = fla_causal_conv1d(
            x.transpose(1, 2).contiguous(),
            weight,
            bias=bias,
            activation=activation,
        )
        if isinstance(result, tuple):
            result = result[0]
        return result.transpose(1, 2).contiguous()

    def chunk_gdr_wrapper(
        q, k, v, *, g, beta, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=True, **kwargs,
    ):
        # This flag is required for the Qwen3.5 parameterization; omitting it
        # produced NaNs on the validated 910B2 environment.
        if not use_qk_l2norm_in_kernel:
            raise ValueError("Qwen3.5 Ascend FLA requires Q/K L2 normalization")
        return fla_chunk_gated_delta_rule(
            q, k, v,
            g=g,
            beta=beta,
            initial_state=initial_state,
            output_final_state=output_final_state,
            use_qk_l2norm_in_kernel=True,
            **kwargs,
        )

    text_config = getattr(getattr(qwen, "config", None), "text_config", None)
    if text_config is None:
        text_config = getattr(qwen.model.language_model, "config", None)
    layer_types = getattr(text_config, "layer_types", None)
    if not layer_types:
        raise RuntimeError(
            "Qwen config does not expose layer_types; cannot verify a complete "
            "Ascend FLA patch"
        )
    expected = sum(str(kind).lower() == "linear_attention" for kind in layer_types)
    if expected <= 0:
        raise RuntimeError("Qwen config declares no linear-attention layers")

    count = 0
    for layer in qwen.model.language_model.layers:
        linear_attn = getattr(layer, "linear_attn", None)
        if linear_attn is None:
            continue
        linear_attn.causal_conv1d_fn = causal_conv_wrapper
        linear_attn.chunk_gated_delta_rule = chunk_gdr_wrapper
        count += 1
    if count != expected:
        raise RuntimeError(
            "Incomplete Qwen3.5 Ascend FLA patch: "
            f"patched {count} of {expected} declared linear-attention layers"
        )
    print(
        f"[QWEN-FLA-NPU] patched {count} linear-attention layers with FLA "
        "Triton-Ascend causal-conv + GDR",
        flush=True,
    )
    return count


def _replace_action_placeholder_embeddings(
    output: torch.Tensor,
    token_ids: torch.Tensor,
    placeholder_ids: Sequence[int],
    placeholder_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Replace only added action-token rows while preserving their gradient."""
    if placeholder_embeddings.shape != (len(placeholder_ids), output.shape[-1]):
        raise ValueError(
            "Action-placeholder embedding shape mismatch: "
            f"got {tuple(placeholder_embeddings.shape)}, expected "
            f"({len(placeholder_ids)},{output.shape[-1]})"
        )
    replaced = output
    for index, token_id in enumerate(placeholder_ids):
        value = placeholder_embeddings[index].to(
            device=output.device, dtype=output.dtype
        )
        replaced = torch.where(
            token_ids.eq(int(token_id)).unsqueeze(-1), value, replaced
        )
    return replaced


def _qwen35_text_forward_with_action_block(
    module,
    input_ids=None,
    attention_mask=None,
    position_ids=None,
    past_key_values=None,
    inputs_embeds=None,
    use_cache=None,
    **kwargs,
):
    """Make only Qwen full-attention action rows/columns bidirectional.

    Qwen3.5's recurrent GatedDeltaNet layers retain their pretrained causal
    recurrence. Feature extraction is full-sequence and cache-free.
    """
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ModelOutputWithPast

    if (input_ids is None) == (inputs_embeds is None):
        raise ValueError("Specify exactly one of input_ids or inputs_embeds")
    if use_cache or past_key_values is not None:
        raise ValueError("Action-block Qwen forward supports cache-free execution only")
    if inputs_embeds is None:
        inputs_embeds = module.embed_tokens(input_ids)
    batch, length = inputs_embeds.shape[:2]
    if position_ids is None:
        position_ids = torch.arange(length, device=inputs_embeds.device)
        position_ids = position_ids.view(1, 1, -1).expand(4, batch, -1)
    elif position_ids.ndim == 2:
        position_ids = position_ids[None].expand(4, position_ids.shape[0], -1)
    if position_ids.ndim == 3 and position_ids.shape[0] == 4:
        text_position_ids = position_ids[0]
        rope_position_ids = position_ids[1:]
    else:
        text_position_ids = None
        rope_position_ids = position_ids

    keep = (
        torch.ones(batch, length, device=inputs_embeds.device, dtype=torch.bool)
        if attention_mask is None else attention_mask.to(dtype=torch.bool)
    )
    if keep.ndim != 2 or keep.shape != (batch, length):
        raise ValueError("Action-block Qwen expects a 2-D padding mask")
    action = getattr(module, "_uav_action_token_mask", None)
    if action is None or action.shape != keep.shape:
        raise RuntimeError("Missing Qwen action-token mask for bidirectional full attention")
    action = action.to(device=inputs_embeds.device, dtype=torch.bool)
    indices = torch.arange(length, device=inputs_embeds.device)
    causal = indices.view(1, -1) <= indices.view(-1, 1)
    allowed = causal.view(1, 1, length, length).expand(batch, 1, -1, -1).clone()
    allowed |= action[:, None, :, None] & action[:, None, None, :]
    allowed &= keep[:, None, :, None] & keep[:, None, None, :]

    hidden_states = inputs_embeds
    position_embeddings = module.rotary_emb(hidden_states, rope_position_ids)
    linear_mask = module._update_linear_attn_mask(attention_mask, past_key_values)
    for index, decoder_layer in enumerate(module.layers[:module.config.num_hidden_layers]):
        layer_mask = (
            linear_mask
            if module.config.layer_types[index] == "linear_attention"
            else allowed
        )
        hidden_states = decoder_layer(
            hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=layer_mask,
            position_ids=text_position_ids,
            past_key_values=None,
            use_cache=False,
            **kwargs,
        )
    hidden_states = module.norm(hidden_states)
    return Qwen3_5ModelOutputWithPast(
        last_hidden_state=hidden_states, past_key_values=None,
    )


class FrozenQwen35SemanticEncoder(nn.Module):
    """Encode F0/Ft and instructions into dense, language-conditioned tokens."""

    def __init__(
        self,
        model_name: str,
        *,
        layer_indices: Sequence[int] = (7, 15, 23),
        attention_implementation: str = "sdpa",
        prompt_mode: str = "temporal_pair",
        token_selection: str = "all",
        lora_enabled: bool = False,
        lora_rank: int = 32,
        lora_alpha: float = 16.0,
        lora_dropout: float = 0.0,
        lora_scope: str = "all_linear",
        action_placeholder_count: int = 0,
        action_attention_mode: str = "causal",
    ) -> None:
        super().__init__()
        from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

        self.processor = AutoProcessor.from_pretrained(
            model_name, local_files_only=True
        )
        self.qwen = Qwen3_5ForConditionalGeneration.from_pretrained(
            model_name,
            local_files_only=True,
            dtype=torch.bfloat16,
            attn_implementation=attention_implementation,
        ).eval().requires_grad_(False)
        self.fla_npu_patched_layers = _patch_qwen_fla_npu(self.qwen)
        self.lora_enabled = bool(lora_enabled)
        self.lora_scope = str(lora_scope)
        self.lora_module_names = ()
        if self.lora_enabled:
            self.lora_module_names = inject_qwen_lora(
                self.qwen, rank=lora_rank, alpha=lora_alpha,
                dropout=lora_dropout, scope=self.lora_scope,
            )
            self.lora_module_count = len(self.lora_module_names)
            trainable = sum(p.numel() for p in self.qwen.parameters() if p.requires_grad)
            print(f"Qwen LoRA scope={self.lora_scope} modules={self.lora_module_count} "
                  f"rank={lora_rank} alpha={lora_alpha} dropout={lora_dropout} "
                  f"trainable_adapter_parameters={trainable}", flush=True)
            print(f"Qwen LoRA targets={list(self.lora_module_names)}", flush=True)
        else:
            self.lora_module_count = 0
        self.action_placeholder_count = int(action_placeholder_count)
        if self.action_placeholder_count < 0:
            raise ValueError("action_placeholder_count must be non-negative")
        self.action_placeholder_tokens = [
            f"<|uav_action_{index:03d}|>"
            for index in range(self.action_placeholder_count)
        ]
        if self.action_placeholder_tokens:
            added = self.processor.tokenizer.add_special_tokens({
                "additional_special_tokens": self.action_placeholder_tokens,
            })
            if added:
                self.qwen.resize_token_embeddings(len(self.processor.tokenizer))
            # Resizing may create fresh trainable embedding/head parameters.
            # Freeze those without disabling LoRA modules installed above.
            self.qwen.get_input_embeddings().requires_grad_(False)
            output_embeddings = self.qwen.get_output_embeddings()
            if output_embeddings is not None:
                output_embeddings.requires_grad_(False)
        self.action_placeholder_ids = tuple(
            int(self.processor.tokenizer.convert_tokens_to_ids(token))
            for token in self.action_placeholder_tokens
        )
        self.action_placeholder_embeddings = None
        self._action_embedding_hook = None
        if self.action_placeholder_ids:
            embedding = self.qwen.get_input_embeddings()
            initial = embedding.weight.detach()[
                list(self.action_placeholder_ids)
            ].clone()
            # The base embedding matrix remains frozen, but the new action
            # rows must be checkpointed and learnable. A separate parameter
            # avoids accidentally updating every Qwen vocabulary row.
            self.action_placeholder_embeddings = nn.Parameter(initial)

            def replace_action_embeddings(_module, hook_inputs, output):
                return _replace_action_placeholder_embeddings(
                    output,
                    hook_inputs[0],
                    self.action_placeholder_ids,
                    self.action_placeholder_embeddings,
                )

            self._action_embedding_hook = embedding.register_forward_hook(
                replace_action_embeddings
            )
        self.action_attention_mode = str(action_attention_mode).lower()
        if self.action_attention_mode not in {"causal", "full_attention_bidir"}:
            raise ValueError(
                "action_attention_mode must be causal or full_attention_bidir"
            )
        if self.action_attention_mode != "causal":
            if not self.action_placeholder_ids:
                raise ValueError("Action-block attention requires action placeholders")
            language_model = self.qwen.model.language_model
            language_model.forward = MethodType(
                _qwen35_text_forward_with_action_block, language_model
            )
        self.hidden_size = int(self.qwen.config.text_config.hidden_size)
        self.image_token_id = int(self.qwen.config.image_token_id)
        self.prompt_mode = str(prompt_mode)
        if self.prompt_mode not in {
            "temporal_pair",
            "current_image_instruction",
            "current_image_action_question",
            "current_image_openvla",
        }:
            raise ValueError(
                "prompt_mode must be 'temporal_pair', "
                "'current_image_instruction', 'current_image_action_question', "
                "or 'current_image_openvla'; "
                f"got {self.prompt_mode!r}."
            )
        self.token_selection = str(token_selection)
        if self.token_selection not in {"all", "text_after_image", "action_placeholders"}:
            raise ValueError(
                "token_selection must be all, text_after_image, or action_placeholders; got "
                f"{self.token_selection!r}."
            )
        self.layer_indices = tuple(int(index) for index in layer_indices)
        num_layers = len(self.qwen.model.language_model.layers)
        self.num_layers = int(num_layers)
        if not self.layer_indices:
            raise ValueError("At least one Qwen3.5 layer index is required.")
        if min(self.layer_indices) < 0 or max(self.layer_indices) >= num_layers:
            raise ValueError(
                f"Qwen layer indices {self.layer_indices} are outside [0,{num_layers})."
            )
        self._captured: dict[int, torch.Tensor] = {}
        self._hooks = [
            self.qwen.model.language_model.layers[index].register_forward_hook(
                self._capture(index)
            )
            for index in self.layer_indices
        ]

    @property
    def device(self) -> torch.device:
        return next(self.qwen.parameters()).device

    def _capture(self, index: int):
        def hook(_module, _inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self._captured[index] = hidden

        return hook

    @staticmethod
    def _to_pil(image: torch.Tensor) -> Image.Image:
        array = (
            image.detach()
            .float()
            .clamp(0.0, 1.0)
            .mul(255.0)
            .round()
            .to(torch.uint8)
            .permute(1, 2, 0)
            .cpu()
            .numpy()
        )
        return Image.fromarray(array, mode="RGB")

    def forward(
        self,
        images: torch.Tensor,
        instructions: Sequence[str],
        current_states: Sequence[Sequence[float]] | None = None,
    ) -> dict[str, torch.Tensor]:
        """Jointly encode instruction, F0, and Ft.

        ``images`` is ``[B,M,1,3,H,W]`` where ``M`` is selected by the prompt
        mode. The returned layer states retain all
        valid image and text positions from the joint multimodal prompt:

        - ``joint_layers``: ``[B,L,S,2048]``
        - ``joint_mask``: ``[B,S]`` (True means valid)
        - ``image_token_mask``: ``[B,S]`` for diagnostics
        """
        expected_images = (
            1
            if self.prompt_mode in {
                "current_image_instruction",
                "current_image_action_question",
                "current_image_openvla",
            }
            else 2
        )
        if images.ndim != 6 or images.shape[1] != expected_images:
            raise ValueError(
                f"Qwen prompt_mode={self.prompt_mode!r} requires "
                f"[B,{expected_images},V,3,H,W], got {tuple(images.shape)}."
            )
        if images.shape[2] != 1:
            raise ValueError(
                "The minimal Qwen adapter currently supports exactly one camera view."
            )
        batch_size = int(images.shape[0])
        if len(instructions) != batch_size:
            raise ValueError(
                f"Expected {batch_size} instructions, got {len(instructions)}."
            )
        if self.prompt_mode in {
            "current_image_action_question",
            "current_image_openvla",
        }:
            if current_states is None or len(current_states) != batch_size:
                raise ValueError(
                    f"{self.prompt_mode} requires one [x,y,z,yaw_deg] "
                    "OpenVLA-compatible state per image."
                )

        conversations = []
        # Only the internal Slot-VLA variants own assistant-side action
        # placeholders. External Query-VLA uses the same image + corrected
        # state + action-question user prompt, but must not receive an empty
        # assistant/generation-prefix token. Ordinary frozen Condition remains
        # image + raw instruction.
        uses_assistant_action_slots = bool(
            self.action_placeholder_tokens
            and self.token_selection == "action_placeholders"
        )
        for batch_index, instruction in enumerate(instructions):
            # Preserve the dataset instruction itself (apart from terminal
            # punctuation needed to embed it in the action question).
            instruction_text = str(instruction).strip()
            instruction_stem = instruction_text.rstrip(".?!")
            if self.prompt_mode in {
                "current_image_instruction",
                "current_image_action_question",
                "current_image_openvla",
            }:
                # Visual input always comes first. Ordinary Condition uses the
                # raw instruction; Query/Slot VLA use the shared state-aware
                # OpenVLA-style action question below.
                prompt_text = instruction_text
                if self.prompt_mode == "current_image_action_question":
                    state = [float(value) for value in current_states[batch_index]]
                    if len(state) != 4:
                        raise ValueError(
                            "Action-question prompt state must be "
                            f"[x,y,z,yaw_deg], got {state}."
                        )
                    proprio_str = ",".join(
                        str(round(value, 1)) for value in state
                    )
                    # Query-VLA and Slot-VLA share this user prompt.  Pose is
                    # the corrected OpenVLA-compatible episode-relative state:
                    # xyz in centimetres, yaw in degrees.  Unlike the legacy
                    # OpenVLA string below, intentionally omit literal In/Out.
                    prompt_text = (
                        f"Current State: {proprio_str}.\n"
                        "What action should the uav take to "
                        f"{instruction_stem}?"
                    )
                if self.prompt_mode == "current_image_openvla":
                    state = [float(value) for value in current_states[batch_index]]
                    if len(state) != 4:
                        raise ValueError(
                            "OpenVLA prompt state must be [x,y,z,yaw_deg], "
                            f"got {state}."
                        )
                    proprio_str = ",".join(str(round(value, 1)) for value in state)
                    # Match the official OpenVLA-UAV prompt exactly. This is
                    # textual numeric conditioning inside frozen Qwen; GAM
                    # still receives no dedicated pose token/history branch.
                    prompt_text = (
                        f"In: Current State: {proprio_str}, "
                        "What action should the uav take to "
                        f"{instruction_stem}?\nOut:"
                    )
                content = [
                    {
                        "type": "image",
                        "image": self._to_pil(images[batch_index, 0, 0]),
                    },
                    {"type": "text", "text": prompt_text},
                ]
            else:
                content = [
                    {
                        "type": "text",
                        "text": (
                            "You started from the first image and are now "
                            "at the second image. "
                            "What action should the uav take to "
                            f"{instruction_stem}?"
                        ),
                    },
                    {
                        "type": "image",
                        "image": self._to_pil(images[batch_index, 0, 0]),
                    },
                    {
                        "type": "image",
                        "image": self._to_pil(images[batch_index, 1, 0]),
                    },
                ]
            messages = [{"role": "user", "content": content}]
            if uses_assistant_action_slots:
                messages.append({
                    "role": "assistant",
                    "content": [{
                        "type": "text",
                        "text": "".join(self.action_placeholder_tokens),
                    }],
                })
            conversations.append(messages)

        with profile_phase("QWEN/processor"):
            inputs = self.processor.apply_chat_template(
                conversations,
                # Slot-VLA already contains an explicit assistant message holding
                # its K action slots.  Every other mode intentionally ends at the
                # user instruction and gets no assistant special token.
                add_generation_prompt=False,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                processor_kwargs={"padding": True},
            )
        with profile_phase("QWEN/h2d"):
            tensor_inputs = {
                key: value.to(self.device, non_blocking=True)
                for key, value in inputs.items()
                if isinstance(value, torch.Tensor)
            }
        input_ids = tensor_inputs["input_ids"]
        language_model = self.qwen.model.language_model
        if self.action_attention_mode == "full_attention_bidir":
            action_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            for token_id in self.action_placeholder_ids:
                action_mask |= input_ids.eq(token_id)
            language_model._uav_action_token_mask = action_mask
        self._captured.clear()
        # Slot placeholders remain trainable even in a frozen-Qwen control.
        # Do not key autograd solely on LoRA being enabled.
        has_trainable_conditioner = any(
            parameter.requires_grad for parameter in self.parameters()
        )
        grad_context = (
            nullcontext()
            if has_trainable_conditioner and torch.is_grad_enabled()
            else torch.no_grad()
        )
        try:
            with profile_phase("QWEN/model"):
                with grad_context:
                    output = self.qwen(
                        **tensor_inputs,
                        use_cache=False,
                        return_dict=True,
                    )
        finally:
            if hasattr(language_model, "_uav_action_token_mask"):
                delattr(language_model, "_uav_action_token_mask")
        del output

        attention_mask = tensor_inputs["attention_mask"].bool()
        image_masks = input_ids.eq(self.image_token_id)
        token_counts = image_masks.sum(dim=1)
        if int(token_counts.min().item()) <= 0:
            raise RuntimeError("Qwen prompt contains no image tokens.")

        joint_mask = attention_mask
        if self.token_selection == "text_after_image":
            # In the current OpenVLA-style prompt the image precedes the
            # instruction. Decoder text states after the final image token
            # have therefore already attended to the image, while the image
            # states themselves cannot see the later instruction under the
            # causal mask. Feed only those image-conditioned text/control
            # states to GAM and mask the instruction-agnostic image prefix.
            positions = torch.arange(
                input_ids.shape[1], device=input_ids.device
            ).view(1, -1)
            last_image_position = torch.where(
                image_masks, positions, positions.new_full((), -1)
            ).amax(dim=1, keepdim=True)
            joint_mask = attention_mask & positions.gt(last_image_position)
            if not bool(joint_mask.any(dim=1).all()):
                raise RuntimeError("No valid text tokens remain after the image prefix.")
        elif self.token_selection == "action_placeholders":
            if not self.action_placeholder_ids:
                raise ValueError(
                    "action_placeholders selection requires action_placeholder_count > 0"
                )
            joint_mask = torch.zeros_like(attention_mask)
            for token_id in self.action_placeholder_ids:
                joint_mask |= input_ids.eq(token_id)
            counts = joint_mask.sum(dim=1)
            if not bool(counts.eq(self.action_placeholder_count).all()):
                raise RuntimeError(
                    "Qwen prompt action-placeholder count mismatch: "
                    f"expected={self.action_placeholder_count}, got={counts.tolist()}"
                )

        layer_tokens = []
        for index in self.layer_indices:
            hidden = self._captured[index]
            # A decoder block hook observes its output immediately before the
            # model's final RMSNorm.  Apply that norm when selecting the actual
            # final layer so "layer 23" matches Qwen's final hidden state.
            if index == self.num_layers - 1:
                hidden = self.qwen.model.language_model.norm(hidden)
            layer_tokens.append(hidden)
        joint_layers = torch.stack(layer_tokens, dim=1)
        if self.token_selection == "action_placeholders":
            # Compact to a fixed [B,layers,K,D] sequence in prompt order.
            joint_layers = torch.stack([
                joint_layers[index, :, joint_mask[index], :]
                for index in range(batch_size)
            ], dim=0)
            joint_mask = torch.ones(
                batch_size, self.action_placeholder_count,
                device=joint_layers.device, dtype=torch.bool,
            )
        return {
            "joint_layers": joint_layers,
            "joint_mask": joint_mask,
            "image_token_mask": image_masks,
            # Retained for diagnostics such as image/text attention breakdown.
            # Training callers intentionally ignore these metadata tensors.
            "input_ids": input_ids,
            "image_grid_thw": tensor_inputs.get("image_grid_thw"),
        }
