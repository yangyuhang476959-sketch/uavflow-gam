"""OpenVLA-style all-linear LoRA, with a legacy checkpoint compatibility scope."""
from torch import nn
from robot.modeling.lora import LoRALinear

LEGACY_SUFFIXES = {
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
}


def inject_qwen_lora(qwen, *, rank=32, alpha=16., dropout=0., scope="all_linear"):
    if scope not in {"all_linear", "legacy_language"}:
        raise ValueError(f"Unknown Qwen LoRA scope: {scope}")
    root = qwen if scope == "all_linear" else qwen.model.language_model
    # PEFT all-linear excludes the final vocabulary output head. Embeddings,
    # convolutions and norms are not Linear targets either.
    output_head = qwen.get_output_embeddings()
    targets = [(name, module) for name, module in root.named_modules()
               if isinstance(module, nn.Linear) and module is not output_head
               and (scope == "all_linear" or name.rsplit('.', 1)[-1] in LEGACY_SUFFIXES)]
    if not targets:
        raise RuntimeError("Qwen LoRA enabled but no target Linear modules were found")
    for name, module in targets:
        parent_name, _, child_name = name.rpartition('.')
        parent = root.get_submodule(parent_name) if parent_name else root
        adapter = LoRALinear(module, rank=int(rank), alpha=float(alpha), dropout=float(dropout))
        if scope == "all_linear":
            # OpenVLA explicitly uses PEFT init_lora_weights='gaussian'.
            nn.init.normal_(adapter.lora_A, std=1. / int(rank))
        setattr(parent, child_name, adapter)
    return tuple(name for name, _ in targets)
