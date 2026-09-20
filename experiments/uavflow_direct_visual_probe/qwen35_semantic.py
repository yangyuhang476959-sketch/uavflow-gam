"""Frozen Qwen3.5 multimodal features for the DA3 action probe.

The prompt follows OpenVLA-UAV's concise action question, with only one extra
hint that identifies the first of the two images as the episode's first frame.
Only selected language-model layers are retained; Qwen itself is always
frozen and runs under ``torch.no_grad``.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from PIL import Image
from torch import nn


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
        self.hidden_size = int(self.qwen.config.text_config.hidden_size)
        self.image_token_id = int(self.qwen.config.image_token_id)
        self.prompt_mode = str(prompt_mode)
        if self.prompt_mode not in {
            "temporal_pair",
            "current_image_instruction",
            "current_image_openvla",
        }:
            raise ValueError(
                "prompt_mode must be 'temporal_pair', "
                "'current_image_instruction', or 'current_image_openvla'; "
                f"got {self.prompt_mode!r}."
            )
        self.token_selection = str(token_selection)
        if self.token_selection not in {"all", "text_after_image"}:
            raise ValueError(
                "token_selection must be 'all' or 'text_after_image', got "
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

    @torch.no_grad()
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
        if self.prompt_mode == "current_image_openvla":
            if current_states is None or len(current_states) != batch_size:
                raise ValueError(
                    "current_image_openvla requires one [x,y,z,yaw_deg] "
                    "state per image."
                )

        conversations = []
        for batch_index, instruction in enumerate(instructions):
            instruction_text = str(instruction).strip().rstrip(".?!")
            if self.prompt_mode in {
                "current_image_instruction",
                "current_image_openvla",
            }:
                # Minimal OpenVLA-style conditioning: visual input first,
                # followed only by the dataset instruction. Chat-template
                # control tokens are the sole unavoidable textual overhead.
                prompt_text = instruction_text
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
                        f"{instruction_text}?\nOut:"
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
                            f"{instruction_text}?"
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
            conversations.append([{"role": "user", "content": content}])

        inputs = self.processor.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        )
        tensor_inputs = {
            key: value.to(self.device, non_blocking=True)
            for key, value in inputs.items()
            if isinstance(value, torch.Tensor)
        }
        self._captured.clear()
        output = self.qwen(
            **tensor_inputs,
            use_cache=False,
            return_dict=True,
        )
        del output

        input_ids = tensor_inputs["input_ids"]
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
        return {
            "joint_layers": joint_layers,
            "joint_mask": joint_mask,
            "image_token_mask": image_masks,
            # Retained for diagnostics such as image/text attention breakdown.
            # Training callers intentionally ignore these metadata tensors.
            "input_ids": input_ids,
            "image_grid_thw": tensor_inputs.get("image_grid_thw"),
        }
