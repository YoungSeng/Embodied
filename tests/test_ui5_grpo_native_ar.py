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
    def __init__(self):
        super().__init__()
        config = Qwen2Config(vocab_size=40, hidden_size=16, intermediate_size=32,
                            num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=2,
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


def fixture_model():
    torch.manual_seed(12)
    model = TinyNative()
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
