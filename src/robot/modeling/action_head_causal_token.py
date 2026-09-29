"""Small OpenVLA-style causal decoder for discretized UAV action chunks.

OpenVLA represents each continuous action dimension with one of 256 bins and
predicts those bins autoregressively.  This head keeps that contract while
conditioning on a GAM/DA3 action token rather than on an LLM hidden sequence.
"""

from __future__ import annotations

import math

import torch
from torch import nn


class CausalTokenActionHead(nn.Module):
    """Decode ``chunk_size * action_dim`` normalized action-bin tokens.

    Normalized actions use the repository's q01/q99 convention.  Values beyond
    the robust training range are clipped to ``[-1, 1]`` before tokenization,
    matching OpenVLA's finite action vocabulary.
    """

    def __init__(
        self,
        input_dim: int,
        *,
        action_dim: int = 4,
        chunk_size: int = 5,
        n_bins: int = 256,
        model_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.n_bins = int(n_bins)
        self.sequence_length = self.action_dim * self.chunk_size
        if self.sequence_length <= 0 or self.n_bins < 2:
            raise ValueError("Action-token sequence and vocabulary must be non-empty.")

        self.condition = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(model_dim)),
        )
        self.token_embed = nn.Embedding(self.n_bins, int(model_dim))
        self.bos = nn.Parameter(torch.empty(1, 1, int(model_dim)))
        self.position = nn.Parameter(
            torch.empty(1, self.sequence_length, int(model_dim))
        )
        layer = nn.TransformerDecoderLayer(
            d_model=int(model_dim),
            nhead=int(num_heads),
            dim_feedforward=4 * int(model_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=int(num_layers))
        self.norm = nn.LayerNorm(int(model_dim))
        self.output = nn.Linear(int(model_dim), self.n_bins)
        nn.init.normal_(self.bos, std=0.02)
        nn.init.normal_(self.position, std=0.02)

    def encode(self, actions_norm: torch.Tensor) -> torch.Tensor:
        """Map normalized ``[..., chunk, dim]`` actions to integer bins."""
        clipped = actions_norm.float().clamp(-1.0, 1.0)
        return torch.round((clipped + 1.0) * 0.5 * (self.n_bins - 1)).long()

    def decode(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Map integer bins to their normalized continuous bin centres."""
        return token_ids.float() * (2.0 / float(self.n_bins - 1)) - 1.0

    def _decode_logits(
        self, condition: torch.Tensor, previous_ids: torch.Tensor
    ) -> torch.Tensor:
        batch = condition.shape[0]
        length = int(previous_ids.shape[1]) + 1
        if length > self.sequence_length:
            raise ValueError(
                f"Requested {length} action positions, maximum is {self.sequence_length}."
            )
        if previous_ids.shape[1]:
            previous = self.token_embed(previous_ids)
            target = torch.cat([self.bos.expand(batch, -1, -1), previous], dim=1)
        else:
            target = self.bos.expand(batch, -1, -1)
        target = target + self.position[:, :length].to(dtype=target.dtype)
        causal = torch.triu(
            torch.ones(length, length, device=target.device, dtype=torch.bool),
            diagonal=1,
        )
        memory = self.condition(condition).unsqueeze(1).to(dtype=target.dtype)
        hidden = self.decoder(target, memory, tgt_mask=causal)
        return self.output(self.norm(hidden))

    def forward(
        self,
        action_tokens: torch.Tensor,
        target_actions_norm: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Teacher-force during training and greedily decode at inference.

        ``action_tokens`` is ``[B,T,V,D]`` or ``[B,T,D]``.  Returned actions
        are ``[B,T,K,A]`` and logits are ``[B,T,K*A,256]``.
        """
        if action_tokens.ndim == 4:
            condition = action_tokens.mean(dim=2)
        elif action_tokens.ndim == 3:
            condition = action_tokens
        else:
            raise ValueError(f"Unexpected action token shape {tuple(action_tokens.shape)}")
        batch, steps, dim = condition.shape
        flat_condition = condition.reshape(batch * steps, dim)

        target_ids = None
        if target_actions_norm is not None:
            expected = (batch, steps, self.chunk_size, self.action_dim)
            if target_actions_norm.shape != expected:
                raise ValueError(
                    f"Action-token target shape {tuple(target_actions_norm.shape)} != {expected}."
                )
            target_ids = self.encode(target_actions_norm).reshape(
                batch * steps, self.sequence_length
            )
            previous = target_ids[:, :-1]
            logits = self._decode_logits(flat_condition, previous)
            predicted_ids = logits.argmax(dim=-1)
        else:
            generated = torch.empty(
                batch * steps, 0, device=flat_condition.device, dtype=torch.long
            )
            logits_per_position = []
            for _ in range(self.sequence_length):
                next_logits = self._decode_logits(flat_condition, generated)[:, -1]
                logits_per_position.append(next_logits)
                generated = torch.cat(
                    [generated, next_logits.argmax(dim=-1, keepdim=True)], dim=1
                )
            logits = torch.stack(logits_per_position, dim=1)
            predicted_ids = generated

        actions = self.decode(predicted_ids).reshape(
            batch, steps, self.chunk_size, self.action_dim
        )
        result = {
            "actions_norm": actions.to(dtype=action_tokens.dtype),
            "logits": logits.reshape(batch, steps, self.sequence_length, self.n_bins),
            "token_ids": predicted_ids.reshape(batch, steps, self.sequence_length),
        }
        if target_ids is not None:
            result["target_ids"] = target_ids.reshape(
                batch, steps, self.sequence_length
            )
        return result


def normalized_token_cross_entropy(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    """Masked CE divided by ``log(vocab)`` for GAM-scale loss weighting."""
    if logits.shape[:-1] != target_ids.shape:
        raise ValueError(
            f"Token logits/targets mismatch: {tuple(logits.shape)} vs {tuple(target_ids.shape)}"
        )
    mask = action_mask.bool().reshape_as(target_ids)
    losses = nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        target_ids.reshape(-1),
        reduction="none",
    ).reshape_as(target_ids)
    denom = mask.float().sum().clamp_min(1.0)
    return (losses * mask.float()).sum() / denom / math.log(float(logits.shape[-1]))
