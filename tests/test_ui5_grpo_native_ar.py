"""Real tiny Qwen2 SDPA + real PBD: cached native AR vs differentiable replay."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from eaglevl.model.locany.configuration_qwen2 import Qwen2Config
from eaglevl.model.locany.modeling_qwen2 import Qwen2ForCausalLM
from eaglevl.model.locany.relation_modules import RelationToPBD
from eaglevl.model.locany.ui5_ar import forward_ar, sample_ar


class TinyNative(nn.Module):
    def __init__(self, kv_heads=2):
        super().__init__()
        config = Qwen2Config(vocab_size=40, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=kv_heads,
                            max_position_embeddings=128, attention_dropout=.4,
                            pad_token_id=0, eos_token_id=3, block_size=6, text_mask_token_id=9)
        config._attn_implementation = "sdpa"
        self.language_model = Qwen2ForCausalLM(config)
        self.mlp1 = nn.Linear(16, 16)
        self.relation_encoder = nn.Linear(16, 8)
        self.relation_pbd = RelationToPBD(8, 16)
        self.image_token_index = 4
        self.config = SimpleNamespace(box_start_token_id=7, text_config=config)
        self.eval()

    def extract_ui_features(self, *, pixel_values, **kwargs):
        relation = self.relation_encoder(pixel_values)
        return [pixel_values], SimpleNamespace(relation_summary=relation, best_relation_token=relation), None


def fixture_model(kv_heads=2):
    torch.manual_seed(12)
    model = TinyNative(kv_heads=kv_heads)
    inputs = dict(input_ids=torch.tensor([[1, 4, 2]]), pixel_values=torch.randn(1, 16),
                  image_grid_hws=torch.tensor([[1, 1]]), image_flags=torch.tensor([1]),
                  relation_family=torch.tensor([0]), defect_type=torch.tensor([0]))
    return model, inputs


def test_real_native_cached_teacher_forced_completion_and_gradients():
    model, inputs = fixture_model()
    tokens = torch.tensor([7, 9, 9, 9, 9, 9, 4, 10, 3])
    probabilities = []
    with torch.no_grad():
        result = forward_ar(model, **inputs, use_cache=True)
        for index, token in enumerate(tokens):
            probabilities.append((result.logits[0] / .7).log_softmax(-1)[token])
            if index + 1 < len(tokens):
                result = forward_ar(model, input_ids=token.reshape(1, 1), use_cache=True,
                                    past_key_values=result.past_key_values, visual_cache=result.visual_cache)
    model.language_model.model.gradient_checkpointing = True
    sequence = dict(inputs, input_ids=torch.cat([inputs["input_ids"], tokens.unsqueeze(0)], 1))
    full = forward_ar(model, **sequence, completion_start=3)
    torch.testing.assert_close(full.completion_log_probs, torch.stack(probabilities), atol=2e-6, rtol=2e-6)
    assert full.pbd_positions.tolist() == [3]  # literal text_mask runs are not MTP
    (-full.completion_log_probs.mean()).backward()
    for module in (model.language_model, model.mlp1, model.relation_encoder, model.relation_pbd):
        assert sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None) > 0
    assert not model.training


def test_sampling_returns_actual_eos_and_independent_repeatable_streams():
    model, inputs = fixture_model()
    with torch.no_grad():
        prefix = forward_ar(model, **inputs, use_cache=True)
        results = [sample_ar(model, inputs, prefix=prefix, max_new_tokens=12, eos_token_id=3, seed=s)
                   for s in (100, 101, 102, 103)]
        repeated = sample_ar(model, inputs, prefix=prefix, max_new_tokens=12, eos_token_id=3, seed=100)
    assert torch.equal(results[0]["tokens"], repeated["tokens"])
    assert len({tuple(r["tokens"].tolist()) for r in results}) == 4
    for result in results:
        assert not result["old_log_probs"].requires_grad
        assert result["tokens"].device.type == "cpu"
        full = forward_ar(model, **dict(inputs, input_ids=torch.cat([inputs["input_ids"], result["tokens"].unsqueeze(0)], 1)), completion_start=3)
        torch.testing.assert_close(full.completion_log_probs, result["old_log_probs"], atol=3e-6, rtol=3e-6)
    # Force a known sampled EOS and verify the real token remains in the loss.
    with torch.no_grad():
        prefix.logits.fill_(-1000)
        prefix.logits[0, 3] = 1000
        eos = sample_ar(model, inputs, prefix=prefix, max_new_tokens=12, eos_token_id=3, seed=1)
    assert eos["tokens"].tolist() == [3]
    assert len(eos["old_log_probs"]) == 1
    assert eos["actual_eos"]


def test_native_existing_slow_decoder_hidden_states_match_shared_ar():
    from eaglevl.utils.locany.modeling_qwen2 import Qwen2ForCausalLM as InferenceQwen2
    model, inputs = fixture_model()
    legacy = InferenceQwen2(deepcopy(model.config.text_config)).eval()
    legacy.load_state_dict(model.language_model.state_dict(), strict=True)
    with torch.no_grad():
        native = forward_ar(model, **inputs, use_cache=True)
        original = legacy(input_ids=inputs["input_ids"],
                          visual_features=model.mlp1(inputs["pixel_values"]), image_token_index=4,
                          position_ids=torch.arange(3).unsqueeze(0), use_cache=True, output_hidden_states=True)
        torch.testing.assert_close(native.logits, original.logits[:, -1], atol=2e-6, rtol=2e-6)
        # At the sampled <box> anchor, the preexisting slow loop applies PBD.
        token = torch.tensor([[7]])
        native = forward_ar(model, input_ids=token, use_cache=True,
                            past_key_values=native.past_key_values, visual_cache=native.visual_cache)
        original = legacy(input_ids=token, past_key_values=original.past_key_values,
                          position_ids=torch.tensor([[3]]), use_cache=True, output_hidden_states=True)
        relation = model.relation_encoder(inputs["pixel_values"])
        pbd = model.relation_pbd(hidden_states=original.hidden_states[-1], input_ids=token,
            sub_sample_lengths=torch.tensor([1]), relation_summary=relation, best_relation_token=relation,
            box_start_token_id=7, text_mask_token_id=9, block_size=6)
        expected = model.language_model.lm_head(pbd.hidden_states)[:, -1]
        torch.testing.assert_close(native.logits, expected, atol=2e-6, rtol=2e-6)


def test_no_dropout_or_detached_caches_in_recomputation():
    model, inputs = fixture_model()
    with torch.no_grad():
        cached = forward_ar(model, **inputs, use_cache=True)
    with pytest.raises(ValueError, match="detached"):
        forward_ar(model, **inputs, visual_cache=cached.visual_cache)
    model.train()
    with pytest.raises(ValueError, match="dropout"):
        forward_ar(model, **inputs)


@pytest.mark.parametrize("seed", [12, 42, 100])
def test_bf16_large_logits_cached_replay_reference_and_checkpointed_gradients(seed):
    # Small random FP32 logits hid the H20 failure. Exercise BF16 and logits
    # around 20-30, where a BF16 output quantization step is appreciable at T=.7.
    model, inputs = fixture_model()
    torch.manual_seed(seed)
    model.to(torch.bfloat16)
    inputs["pixel_values"] = inputs["pixel_values"].bfloat16()
    with torch.no_grad():
        model.language_model.lm_head.weight.mul_(100)
    tokens = torch.randint(10, 40, (64,))
    tokens[0] = tokens[30] = 7
    tokens[1:6] = 9
    tokens[-1] = 3
    old = []
    with torch.no_grad():
        cached = forward_ar(model, **inputs, use_cache=True)
        for index, token in enumerate(tokens):
            assert cached.logits.dtype == torch.float32
            old.append((cached.logits[0] / .7).log_softmax(-1)[token])
            if index < len(tokens) - 1:
                cached = forward_ar(model, input_ids=token.reshape(1, 1), use_cache=True,
                                    past_key_values=cached.past_key_values, visual_cache=cached.visual_cache)
    sequence = dict(inputs, input_ids=torch.cat([inputs["input_ids"], tokens[None]], 1))
    model.language_model.model.gradient_checkpointing = True
    with torch.no_grad():
        reference = forward_ar(model, **sequence, completion_start=3).completion_log_probs
    # The head must remain FP32 even if DeepSpeed/callers add autocast.
    with torch.autocast("cpu", dtype=torch.bfloat16):
        full = forward_ar(model, **sequence, completion_start=3)
    torch.testing.assert_close(full.completion_log_probs, torch.stack(old), atol=2e-5, rtol=2e-6)
    torch.testing.assert_close(full.completion_log_probs, reference, atol=2e-5, rtol=2e-6)
    assert full.pbd_positions.tolist() == [3, 33]
    (-full.completion_log_probs.mean()).backward()
    for module in (model.language_model, model.mlp1, model.relation_encoder, model.relation_pbd):
        gradients = [p.grad for p in module.parameters() if p.grad is not None]
        assert gradients and all(torch.isfinite(g).all() for g in gradients)
        assert sum(g.abs().sum().item() for g in gradients) > 0


def test_sdpa_backend_is_preserved_inside_backward_checkpoint_replay(monkeypatch):
    import torch.nn.functional as functional
    from torch.nn.attention import sdpa_kernel, SDPBackend
    observations = []
    original = functional.scaled_dot_product_attention
    def observed(*args, **kwargs):
        mask = kwargs.get("attn_mask")
        if mask is not None:
            assert mask.stride(-1) == 1
            assert all(stride % 8 == 0 for stride in mask.stride()[:-1])
        observations.append((torch.backends.cuda.math_sdp_enabled(),
                             torch.backends.cuda.mem_efficient_sdp_enabled(),
                             torch.backends.cuda.flash_sdp_enabled()))
        return original(*args, **kwargs)
    monkeypatch.setattr(functional, "scaled_dot_product_attention", observed)
    model, inputs = fixture_model()
    model.language_model.model.gradient_checkpointing = True
    tokens = torch.tensor([[7, 10, 3]])
    sequence = dict(inputs, input_ids=torch.cat([inputs["input_ids"], tokens], 1))
    # No global mutation, and a different surrounding default during backward
    # must not change the recomputed kernel selected inside each native layer.
    defaults = (torch.backends.cuda.math_sdp_enabled(), torch.backends.cuda.flash_sdp_enabled())
    with torch.no_grad():
        forward_ar(model, **inputs, use_cache=True)
    result = forward_ar(model, **sequence, completion_start=3)
    forward_count = len(observations)
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        result.completion_log_probs.sum().backward()
        assert not torch.backends.cuda.math_sdp_enabled()
    assert len(observations) > forward_count
    assert all(flags == (True, False, False) for flags in observations)
    assert defaults == (torch.backends.cuda.math_sdp_enabled(), torch.backends.cuda.flash_sdp_enabled())


def test_fp32_vocabulary_projection_keeps_gradients_and_avoids_bf16_output_rounding():
    from eaglevl.model.locany.ui5_ar import _vocab_logits, _log_probs
    torch.manual_seed(31)
    hidden = torch.randn(5, 16, dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(40, 16, dtype=torch.bfloat16, requires_grad=True)
    targets = torch.tensor([1, 7, 3, 9, 10])
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = _vocab_logits(hidden, weight)
        actual = _log_probs(hidden, weight, targets, .7)
    precise_logits = torch.nn.functional.linear(hidden.float(), weight.float())
    expected = (precise_logits / .7).log_softmax(-1).gather(-1, targets[:, None]).flatten()
    assert logits.dtype == actual.dtype == torch.float32
    assert not torch.equal(logits, logits.bfloat16().float())
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    actual_grads = torch.autograd.grad(actual.sum(), (hidden, weight))
    expected_grads = torch.autograd.grad(expected.sum(), (hidden, weight))
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad)


@pytest.mark.parametrize("length", [1, 8, 13, 7268])
def test_cutlass_mask_alignment_preserves_exact_attention_domain(length):
    from eaglevl.model.locany.ui5_ar import align_ar_attention_mask
    mask = torch.zeros(1, 1, 3, length, dtype=torch.bfloat16)
    mask[..., -1] = torch.finfo(mask.dtype).min
    aligned = align_ar_attention_mask(mask)
    assert aligned.shape == mask.shape
    assert torch.equal(aligned, mask)
    assert aligned.stride(-1) == 1
    assert all(stride % 8 == 0 for stride in aligned.stride()[:-1])
    assert align_ar_attention_mask(aligned) is aligned
    assert align_ar_attention_mask(None) is None


def sequential_gradient_oracle(model, inputs, tokens):
    """Conventional token-first AR graph; deliberately no checkpoint helpers.

    Keep every sampled-prefix K/V attached. This small-test oracle is too
    memory hungry for production, but gives an independent exact derivative.
    """
    model.language_model.model.gradient_checkpointing = False
    vit, relation, _ = model.extract_ui_features(**{k: v for k, v in inputs.items() if k != "input_ids"})
    projected = model.mlp1(torch.cat(vit))
    cache, scores = None, []
    for index, target in enumerate(tokens):
        ids = inputs["input_ids"] if index == 0 else tokens[index - 1].reshape(1, 1)
        embeds = model.language_model.get_input_embeddings()(ids).clone()
        if index == 0:
            embeds[ids == model.image_token_index] = projected
        start = 0 if index == 0 else inputs["input_ids"].shape[1] + index - 1
        output = model.language_model.model(inputs_embeds=embeds, past_key_values=cache, use_cache=True,
            position_ids=torch.arange(start, start + ids.shape[1]).unsqueeze(0), ui5_ar_mode=True, return_dict=True)
        cache = output.past_key_values
        pbd = model.relation_pbd(hidden_states=output.last_hidden_state, input_ids=ids,
            sub_sample_lengths=torch.tensor([ids.numel()]), relation_summary=relation.relation_summary,
            best_relation_token=relation.best_relation_token, box_start_token_id=7, text_mask_token_id=9, block_size=1)
        logits = torch.nn.functional.linear(pbd.hidden_states[:, -1].float(), model.language_model.lm_head.weight.float())
        scores.append((logits[0] / .7).log_softmax(-1)[target])
    return torch.stack(scores)


@pytest.mark.parametrize("sliding_window", [None, 4])
def test_replay_matches_token_first_gradient_oracle_and_keeps_prompt_gradients(sliding_window):
    model, inputs = fixture_model(kv_heads=1)
    model.config.text_config.sliding_window = sliding_window
    expected_model = deepcopy(model)
    tokens = torch.tensor([7, 9, 4, 7, 10, 3])
    model.language_model.model.gradient_checkpointing = True
    sequence = dict(inputs, input_ids=torch.cat([inputs["input_ids"], tokens[None]], 1))
    actual = forward_ar(model, **sequence, completion_start=3).completion_log_probs
    expected = sequential_gradient_oracle(expected_model, inputs, tokens)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    # The LAST token's probability must train the prompt/visual/relation path,
    # not only the last token. Detached inference KV would fail this comparison.
    actual[-1].backward()
    expected[-1].backward()
    for (name, p), (other_name, other) in zip(model.named_parameters(), expected_model.named_parameters()):
        assert name == other_name
        if other.grad is None:
            assert p.grad is None or not p.grad.count_nonzero()
        else:
            assert p.grad is not None, name
            torch.testing.assert_close(p.grad, other.grad, atol=2e-6, rtol=2e-5, msg=name)
    assert model.mlp1.weight.grad.abs().sum() > 0
    assert model.language_model.model.embed_tokens.weight.grad[1].abs().sum() > 0


@pytest.mark.parametrize("sliding_window", [None, 4])
def test_bf16_sampling_replay_and_backward_keep_each_layer_shape_mask_and_position(sliding_window):
    model, inputs = fixture_model(kv_heads=1)
    model.to(torch.bfloat16)
    model.config.text_config.sliding_window = sliding_window
    inputs["pixel_values"] = inputs["pixel_values"].bfloat16()
    tokens = torch.tensor([7, 9, 4, 7, 10, 3])
    observed = {}
    phase = "sample"
    def observe(name):
        def hook(module, args, kwargs):
            mask = kwargs["attention_mask"]
            position = kwargs["position_ids"]
            key = (name, tuple(position.flatten().tolist()))
            record = (kwargs["hidden_states"].detach().clone(), None if mask is None else mask.clone())
            if phase == "sample":
                assert key not in observed
                observed[key] = record
            else:
                assert key in observed, f"replay introduced a different decoder shape/position: {key}"
                expected_hidden, expected_mask = observed[key]
                torch.testing.assert_close(record[0], expected_hidden, atol=0, rtol=0)
                if expected_mask is None:
                    assert mask is None
                else:
                    assert torch.equal(mask, expected_mask)
        return hook
    handles = [layer.self_attn.register_forward_pre_hook(observe(str(i)), with_kwargs=True)
               for i, layer in enumerate(model.language_model.model.layers)]
    try:
        with torch.no_grad():
            result = forward_ar(model, **inputs, use_cache=True)
            for token in tokens[:-1]:
                result = forward_ar(model, input_ids=token.reshape(1, 1), use_cache=True,
                                    past_key_values=result.past_key_values, visual_cache=result.visual_cache)
        phase = "replay"
        model.language_model.model.gradient_checkpointing = True
        sequence = dict(inputs, input_ids=torch.cat([inputs["input_ids"], tokens[None]], 1))
        result = forward_ar(model, **sequence, completion_start=3)
        result.completion_log_probs.sum().backward()
    finally:
        for handle in handles:
            handle.remove()
    assert len(observed) == len(tokens) * len(model.language_model.model.layers)


def test_replay_checkpoints_do_not_save_all_layers_growing_kv_histories():
    from eaglevl.model.locany.ui5_ar import replay_ar_hidden
    model, _ = fixture_model(kv_heads=1)
    model.language_model.model.gradient_checkpointing = True
    hidden = torch.randn(1, 60, 16, requires_grad=True)
    saved_shapes = []
    def pack(tensor):
        saved_shapes.append(tuple(tensor.shape))
        return tensor
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda tensor: tensor):
        result = replay_ar_hidden(model.language_model.model, hidden, 17)
    # A cached token-first graph would retain 4-D attention tensors for every
    # token in every layer. Production retains layer inputs and the final norm.
    assert saved_shapes and all(len(shape) < 4 for shape in saved_shapes)
    result[:, -1].square().sum().backward()
    assert hidden.grad[:, :17].abs().sum() > 0


def test_bf16_on_policy_probabilities_remain_valid_after_five_weight_updates():
    from eaglevl.train.ui5_grpo_runtime import sampling_probability_report
    model, inputs = fixture_model()
    model.to(torch.bfloat16)
    model.language_model.model.gradient_checkpointing = True
    inputs["pixel_values"] = inputs["pixel_values"].bfloat16()
    with torch.no_grad():
        model.language_model.lm_head.weight.mul_(100)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    for step in range(5):
        result = sample_ar(model, inputs, max_new_tokens=16, eos_token_id=3, seed=step + 13)
        original_old = result["old_log_probs"].clone()
        sequence = dict(inputs, input_ids=torch.cat([inputs["input_ids"], result["tokens"][None]], 1))
        current = forward_ar(model, **sequence, completion_start=3).completion_log_probs
        assert sampling_probability_report(current, result["old_log_probs"], .2)["valid"]
        torch.testing.assert_close(current, original_old, atol=2e-5, rtol=2e-6)
        (-current.mean()).backward()
        optimizer.step()
        optimizer.zero_grad()
        assert torch.equal(result["old_log_probs"], original_old)
