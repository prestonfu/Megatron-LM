# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import torch

from megatron.core import parallel_state


def clip_qk(model, log_max_only=False):
    """
    Clip the QK attention logits to the threshold, recommended for Muon optimizer.

    Args:
        model: The model to clip the QK attention logits, a list of model chunks.
        log_only: Whether to only log the max attention logit, without updating the weights.

    Returns:
        Tuple of (max_attention_logit, per_layer_max_attn_logits).
    """

    with torch.no_grad():
        log_max_attention_logit = 0
        per_layer_max_attn_logits = {}
        layer_idx = 0
        for model_chunk in model:
            for transformer_layer in model_chunk.module.module.decoder.layers:
                if not hasattr(transformer_layer, 'self_attention'):
                    layer_idx += 1
                    continue
                if hasattr(transformer_layer.self_attention, 'clip_qk'):
                    if (
                        transformer_layer.self_attention.core_attention.current_max_attn_logits
                        is None
                    ):
                        layer_idx += 1
                        continue
                    torch.distributed.all_reduce(
                        transformer_layer.self_attention.core_attention.current_max_attn_logits,
                        op=torch.distributed.ReduceOp.MAX,
                        group=parallel_state.get_data_parallel_group(with_context_parallel=True),
                    )
                    layer_max = torch.max(
                        transformer_layer.self_attention.core_attention.current_max_attn_logits
                    ).cpu().item()
                    transformer_layer.self_attention.core_attention.current_max_attn_logits = None
                    per_layer_max_attn_logits[layer_idx] = layer_max
                    log_max_attention_logit = max(log_max_attention_logit, layer_max)
                    if not log_max_only:
                        transformer_layer.self_attention.clip_qk()
                layer_idx += 1

    return log_max_attention_logit, per_layer_max_attn_logits
