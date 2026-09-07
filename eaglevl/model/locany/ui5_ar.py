"""Native slow AR with replayable sampled tokens and differentiable log-probs.

No coordinate averaging, format repair, forced EOS, or CE-as-probability.
The native vision/detail/relation/PBD and Qwen2 decoder are shared by both
sampling and scoring. Cached tensors are only accepted without autograd.
"""
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from torch.nn.attention import SDPBackend, sdpa_kernel


AR_NUMERICS_VERSION = "fp32-head-fixed-sdpa-v1"
AR_REPLAY_VERSION = "prefill-tokenwise-layer-replay-v1"


def align_ar_attention_mask(mask):
    # CUTLASS SDPA requires bias row strides aligned to 8 BF16 elements. Pad
    # storage only, then slice back: token counts and causal/window values stay
    # identical, including arbitrary prompt lengths and the 7268-token cap.
    if mask is None or all(stride % 8 == 0 for stride in mask.stride()[:-1]):
        return mask
    length = mask.shape[-1]
    return F.pad(mask, (0, (-length) % 8))[..., :length]


def ar_decoder_forward(decoder_layer, hidden_states, *args, **kwargs):
    # Auto dispatch can choose different kernels for cached q_len=1 (no mask)
    # and a full causal sequence, and again when autograd is enabled. Keep the
    # backend inside the checkpointed callable so backward replay uses it too.
    # CUDA's memory-efficient SDPA supports the native additive causal mask;
    # forcing math on long H20 sequences would materialize large attention maps.
    backend = (SDPBackend.EFFICIENT_ATTENTION if hidden_states.is_cuda else SDPBackend.MATH)
    with sdpa_kernel(backend):
        return decoder_layer(hidden_states, *args, **kwargs)


def _ar_spans(length, prompt_length):
    yield 0, prompt_length
    for offset in range(prompt_length, length):
        yield offset, offset + 1


def _replay_decoder_piece(decoder_layer, hidden, mask, positions, past_key, past_value):
    # Each checkpoint invocation owns a fresh cache. Never close over a mutable
    # DynamicCache: backward would append a token twice or see a future prefix.
    # The tensors remain attached to autograd, including all prompt K/V states.
    from transformers.cache_utils import DynamicCache
    cache = DynamicCache()
    layer_idx = decoder_layer.self_attn.layer_idx
    if past_key is not None:
        cache.update(past_key, past_value, layer_idx)
    output = ar_decoder_forward(decoder_layer, hidden, attention_mask=mask,
                                position_ids=positions, past_key_value=cache,
                                use_cache=True, output_attentions=False)
    key, value = cache[layer_idx]
    return output[0], key, value


def _replay_decoder_layer(decoder_layer, hidden, prompt_length, sliding_window):
    from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask
    parts, key, value = [], None, None
    for start, end in _ar_spans(hidden.shape[1], prompt_length):
        piece = hidden[:, start:end]
        positions = torch.arange(start, end, device=hidden.device).unsqueeze(0)
        mask = align_ar_attention_mask(_prepare_4d_causal_attention_mask(
            None, (1, end - start), piece, start, sliding_window=sliding_window))
        args = (decoder_layer, piece, mask, positions, key, value)
        if torch.is_grad_enabled():
            # Nested checkpointing avoids retaining attention's expanded GQA
            # K/V and MLP intermediates for every answer token during a layer's
            # backward replay. Only this layer's unexpanded prefix states live.
            result, key, value = checkpoint(_replay_decoder_piece, *args, use_reentrant=False)
        else:
            result, key, value = _replay_decoder_piece(*args)
        parts.append(result)
    return torch.cat(parts, dim=1)


def replay_ar_hidden(decoder, embeds, prompt_length):
    """Replay the sampled prefill + q_len=1 schedule, with full prefix gradients.

    Evaluate one whole decoder layer at a time. Causal dependencies make this
    equivalent to the sampling loop's token-first ordering, while per-layer
    checkpointing prevents retaining every layer's growing K/V history. All
    decoder calls (Q/K/V, SDPA, MLP) keep the sampling shapes and positions.
    """
    if not 1 <= prompt_length <= embeds.shape[1] or embeds.shape[0] != 1:
        raise ValueError("invalid unpadded AR replay prompt boundary")
    hidden = embeds
    for layer in decoder.layers:
        args = (layer, hidden, prompt_length, decoder.config.sliding_window)
        if torch.is_grad_enabled():
            hidden = checkpoint(_replay_decoder_layer, *args, use_reentrant=False)
        else:
            hidden = _replay_decoder_layer(*args)
    # Final normalization also uses the exact sampling shapes.
    return torch.cat([decoder.norm(hidden[:, start:end])
                      for start, end in _ar_spans(hidden.shape[1], prompt_length)], dim=1)


def _pbd(model, hidden, ids, relation):
    return model.relation_pbd(
        hidden_states=hidden, input_ids=ids,
        sub_sample_lengths=torch.tensor([ids.numel()], device=ids.device),
        relation_summary=relation.relation_summary, best_relation_token=relation.best_relation_token,
        box_start_token_id=model.config.box_start_token_id,
        text_mask_token_id=model.config.text_config.text_mask_token_id, block_size=1)


def _completion_piece(model, hidden, ids, summary, best, target, temperature):
    from types import SimpleNamespace
    fused = _pbd(model, hidden, ids, SimpleNamespace(relation_summary=summary, best_relation_token=best))
    # One predictor / vocabulary projection, exactly as when sampling.
    return _log_probs(fused.hidden_states[:, -1], model.language_model.lm_head.weight,
                      target.reshape(1), temperature)


def _vocab_logits(hidden, weight):
    # Converting AFTER a BF16 GEMM cannot recover rounded vocabulary logits.
    # Cast inside the checkpoint so no full FP32 head/gradient is kept per chunk.
    # Also defeat any caller's autocast; old/current/reference use this helper.
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        return F.linear(hidden.float(), weight.float())


@dataclass
class ARResult:
    logits: object = None
    completion_log_probs: object = None
    past_key_values: object = None
    visual_cache: object = None
    pbd_positions: object = None


def _log_probs(hidden, weight, targets, temperature):
    logits = _vocab_logits(hidden, weight) / temperature
    return logits.log_softmax(-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)


def forward_ar(model, *, input_ids, pixel_values=None, image_grid_hws=None,
               relation_family=None, defect_type=None, image_flags=None,
               position_ids=None, attention_mask=None, past_key_values=None,
               use_cache=False, visual_cache=None, completion_start=None,
               temperature=0.7, **unused):
    if model.training or any(module.training for module in model.modules()):
        raise ValueError("AR sampling/scoring requires eval mode (dropout disabled)")
    if input_ids.ndim != 2 or input_ids.shape[0] != 1:
        raise ValueError("native AR microbatch must be one unpadded crop/completion")
    if attention_mask is not None and not bool(attention_mask.all()):
        raise ValueError("pad tokens must be removed before native AR scoring")
    if torch.is_grad_enabled() and (visual_cache is not None or past_key_values is not None or use_cache):
        raise ValueError("gradient recomputation cannot use detached visual/prefix caches")
    if temperature <= 0:
        raise ValueError("positive sampling temperature required")
    if visual_cache is None:
        if pixel_values is None or relation_family is None or defect_type is None:
            raise ValueError("UI5 image and task/relation routing are required")
        vit, relation, _ = model.extract_ui_features(
            pixel_values=pixel_values, image_grid_hws=image_grid_hws,
            relation_family=relation_family, defect_type=defect_type,
            image_flags=image_flags, return_global_visual_cache=False)
        projected = model.mlp1(torch.cat(vit, dim=0))
        visual_cache = (projected, relation)
    projected, relation = visual_cache
    if relation is None:
        raise ValueError("UI5 GRPO requires the pretrained relation/PBD modules")
    embeds = model.language_model.get_input_embeddings()(input_ids).clone()
    selected = input_ids == model.image_token_index
    if completion_start is not None:
        # A literally sampled image special token is a text embedding in the
        # cached slow decoder. Only the PROMPT's image placeholders are visual.
        selected[:, completion_start:] = False
    if past_key_values is None:
        if int(selected.sum()) != projected.shape[0]:
            raise ValueError("image token/feature mismatch; never train a truncated visual input")
        embeds[selected] = projected.to(embeds.dtype)
    past_length = 0 if past_key_values is None else past_key_values[0][0].shape[-2]
    expected_positions = torch.arange(past_length, past_length + input_ids.shape[1],
                                      device=input_ids.device).unsqueeze(0)
    if position_ids is None:
        position_ids = expected_positions
    elif not torch.equal(position_ids, expected_positions):
        raise ValueError("AR positions must match the actual sampled sequence")
    if completion_start is not None:
        if not 1 <= completion_start < input_ids.shape[1] or past_key_values is not None or use_cache:
            raise ValueError("completion_start must mark actual completion in the full sequence")
        # The final sampled token (including actual EOS) is a TARGET, never a
        # predictor. Processing it would add a decoder step absent in sampling.
        hidden = model.language_model.model(
            inputs_embeds=embeds[:, :-1], use_cache=False, return_dict=True,
            output_attentions=False, output_hidden_states=False,
            ui5_ar_mode=True, ui5_ar_prompt_length=completion_start).last_hidden_state
        pieces, active = [], []
        for start, end in _ar_spans(hidden.shape[1], completion_start):
            ids = input_ids[:, start:end]
            args = (model, hidden[:, start:end], ids, relation.relation_summary,
                    relation.best_relation_token, input_ids[0, end], temperature)
            if torch.is_grad_enabled():
                pieces.append(checkpoint(_completion_piece, *args, use_reentrant=False))
            else:
                pieces.append(_completion_piece(*args))
            active.append((ids.reshape(-1) == model.config.box_start_token_id).nonzero().flatten() + start)
        return ARResult(completion_log_probs=torch.cat(pieces), pbd_positions=torch.cat(active))
    output = model.language_model.model(
        inputs_embeds=embeds, position_ids=position_ids, past_key_values=past_key_values,
        use_cache=use_cache, return_dict=True, ui5_ar_mode=True)
    # Slow decode processes one token per cached step. Only <box> anchors
    # receive PBD, including if the model literally samples <text_mask> tokens.
    fused = _pbd(model, output.last_hidden_state, input_ids, relation)
    weight = model.language_model.lm_head.weight
    return ARResult(logits=_vocab_logits(fused.hidden_states[:, -1], weight),
                    past_key_values=output.past_key_values, visual_cache=visual_cache,
                    pbd_positions=fused.active_positions)


@torch.no_grad()
def sample_ar(model, inputs, *, max_new_tokens, eos_token_id, seed,
              temperature=0.7, prefix=None, generation_mode="slow"):
    if generation_mode != "slow":
        raise ValueError("GRPO trajectories require generation_mode=slow")
    generator = torch.Generator(device=inputs["input_ids"].device).manual_seed(seed)
    if prefix is None:
        prefix = forward_ar(model, **inputs, use_cache=True, temperature=temperature)
    current = prefix
    tokens, log_probs = [], []
    for offset in range(max_new_tokens):
        logp = (current.logits[0].float() / temperature).log_softmax(-1)
        if not torch.isfinite(logp).all():
            raise FloatingPointError("non-finite AR sampling distribution")
        token = torch.multinomial(logp.exp(), 1, generator=generator)
        tokens.append(token.detach().cpu())
        log_probs.append(logp[token].detach().cpu())
        if int(token) == eos_token_id or offset + 1 == max_new_tokens:
            break
        current = forward_ar(model, input_ids=token.reshape(1, 1),
                             past_key_values=current.past_key_values, use_cache=True,
                             visual_cache=prefix.visual_cache, temperature=temperature)
    return {"tokens": torch.cat(tokens), "old_log_probs": torch.cat(log_probs),
            "seed": seed, "actual_eos": int(tokens[-1]) == eos_token_id,
            "termination": "eos" if int(tokens[-1]) == eos_token_id else "length_limit"}
