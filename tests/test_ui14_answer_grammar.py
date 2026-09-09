"""CPU tensor tests of the actual AR/MTP loop, without a VLM or detector."""
import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import time
from typing import Optional
import unittest
import numpy as np
import torch
from eaglevl.ui_answer_grammar import answer_state, raw_format_issue, VERSION

ROOT = Path(__file__).resolve().parents[1]
IDS = dict(box_start_token_id=1, box_end_token_id=2, ref_start_token_id=3,
           ref_end_token_id=4, none_token_id=5, im_end_token_id=6, null_token_id=7,
           coord_start_token_id=20, coord_end_token_id=1020, default_mask_token_id=10, switch_token_id=11)
LABEL = [8, 9]


def helpers():
    spec = importlib.util.spec_from_file_location("cpu_ui_generate_utils", ROOT / "eaglevl/utils/locany/generate_utils.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


class CPUCache:
    def __init__(self): self.tokens = []
    def get_seq_length(self): return len(self.tokens)
    def crop(self, size): self.tokens = self.tokens[:size]


class Tokenizer:
    model_max_length = 512
    def decode(self, values, **kwargs):
        tokens = {1: "<box>", 2: "</box>", 3: "<ref>", 4: "</ref>", 5: "none", 6: "<|im_end|>", 8: "misaligned", 9: " elements"}
        return "".join(tokens.get(int(x), f"<{int(x)-20}>") for x in values)
    def batch_decode(self, rows, **kwargs): return [self.decode(row) for row in rows]


class LanguageModel:
    dtype = torch.float32
    def __init__(self, positive=True, frame="valid", boxes=1):
        self.positive, self.frame, self.boxes = positive, frame, boxes
        self.mtp_calls = self.ar_calls = 0
        self.prefixes = []
        self.mtp_inputs = []

    def __call__(self, input_ids, past_key_values, **kwargs):
        cache = past_key_values or CPUCache()
        cache.tokens += input_ids[0].tolist()
        mtp = cache.tokens[-1] == 10
        prefix = cache.tokens[:-6] if mtp else cache.tokens
        state = answer_state(prefix[1:], IDS, LABEL, 8)
        self.prefixes.append(prefix[1:])
        length = 6 if mtp else 1
        logits = torch.full((1, length, 1100), -20.)
        if mtp:
            self.mtp_calls += 1
            self.mtp_inputs.append(input_ids.clone())
            frame = [1, 30, 40, 80, 100, 2]
            if self.frame == "empty": frame = [1, 5, 2, 7, 7, 7]
            if self.frame == "ambiguous": frame = [4, 8, 5, 30, 3, 2]
            for i, token in enumerate(frame): logits[0, i, token] = 20
        else:
            self.ar_calls += 1
            allowed = list(state["allowed"])
            token = allowed[0]
            if state["phase"] == "start": token = 3 if self.positive else 1
            elif state["phase"] == "between": token = 3 if state["boxes"] < self.boxes else 6
            elif state["phase"] == "coords":
                anchor = max(i for i, t in enumerate(prefix) if t == 1)
                token = [30, 40, 80, 100][len(prefix) - anchor - 1]
            logits[0, 0, token] = 20
            # Deliberately prefer illegal tag/none/coordinate continuations.
            # The real loop must exclude these before sampling, not repair text.
            if state["phase"] in ("label", "ref_end", "none_end", "end", "box_start"):
                logits[0, 0, 77] = 25
            if state["phase"] == "between": logits[0, 0, 1] = 25
        return SimpleNamespace(logits=logits, past_key_values=cache, hidden_states=None)


def actual_generator(lm):
    module = helpers()
    namespace = dict(torch=torch, np=np, time=time, Optional=Optional, DynamicCache=CPUCache, Cache=CPUCache)
    namespace.update({name: getattr(module, name) for name in
        ("sample_tokens", "handle_pattern", "constrain_ui5_bbox_logits", "constrain_ui_answer_ar_logits", "resolve_ui5_mtp_frame")})
    path = ROOT / "eaglevl/utils/locany/modeling_locateanything.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    generate = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "generate")
    generate.decorator_list = []
    cache = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_generation_cache_seq_length")
    exec(compile(ast.Module(body=[cache, generate], type_ignores=[]), str(path), "exec"), namespace)
    model = SimpleNamespace(language_model=lm, enable_ui_relation=False, token_ids=IDS,
        config=SimpleNamespace(text_config=SimpleNamespace(block_size=6, text_mask_token_id=10),
                               relation_num_slots=8, image_token_index=1001,
                               relation_constrained_bbox_decoding=True, relation_gate_mode="observe"))
    return lambda **kwargs: namespace["generate"](model, pixel_values=torch.zeros(1, 1),
        input_ids=torch.tensor([[1050]]), visual_features=torch.zeros(1, 1), tokenizer=Tokenizer(),
        use_cache=True, ui_answer_grammar=VERSION, ui_ref_label_ids=LABEL, **kwargs)


class AnswerGrammarTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(1)

    def test_none_is_terminal_and_cannot_mix_with_a_box(self):
        state = answer_state([1, 5, 2], IDS, LABEL)
        self.assertEqual(list(state["allowed"]), [6])
        with self.assertRaises(ValueError): answer_state([1, 5, 2, 1], IDS, LABEL)
        with self.assertRaises(ValueError): answer_state([1, 5, 30], IDS, LABEL)

    def test_training_label_syntax_and_invalid_forms(self):
        import sys
        sys.path.insert(0, str(ROOT / "scripts"))
        from ui14_annotations import answer
        self.assertIsNone(raw_format_issue(answer([], 375, 800, "misaligned elements")))
        self.assertIsNone(raw_format_issue(answer([[10, 20, 80, 100]], 375, 800, "misaligned elements")))
        for text, kind in ((None, "missing_raw_evidence"), ("", "empty_output"),
                           ("<box>none<30></box>", "none_coordinate_mix"),
                           ("<ref>misaligned</ref></ref><box><1><2><3><4></box>", "illegal_tag_structure"),
                           ("<box><1><2></box>", "bbox_parse_failure")):
            self.assertEqual(raw_format_issue(text), kind)

    def test_actual_loop_negative_in_ar_hybrid_and_fast(self):
        for mode in ("slow", "hybrid", "fast"):
            lm = LanguageModel(positive=False)
            self.assertEqual(actual_generator(lm)(generation_mode=mode, max_new_tokens=64), "<box>none</box><|im_end|>")
            self.assertEqual(lm.mtp_calls, 0)

    def test_actual_loop_positive_mtp_and_ar_share_complete_ref_state(self):
        expected = "<ref>misaligned elements</ref><box><10><20><60><80></box>" * 2 + "<|im_end|>"
        for mode in ("slow", "hybrid", "fast"):
            lm = LanguageModel(boxes=2)
            self.assertEqual(actual_generator(lm)(generation_mode=mode, max_new_tokens=128), expected)
            self.assertEqual(lm.mtp_calls, 0 if mode == "slow" else 2)

    def test_ambiguous_and_none_mtp_frames_fall_back_without_committing_illegal_tokens(self):
        for mode in ("hybrid", "fast"):
            for frame in ("empty", "ambiguous"):
                lm = LanguageModel(frame=frame)
                answer = actual_generator(lm)(generation_mode=mode, max_new_tokens=128)
                self.assertEqual(answer, "<ref>misaligned elements</ref><box><10><20><60><80></box><|im_end|>")
                self.assertEqual(lm.mtp_calls, 1)
                self.assertTrue(all(answer_state(p, IDS, LABEL) for p in lm.prefixes))

    def test_exhausted_budget_does_not_become_a_legal_negative(self):
        with self.assertRaisesRegex(RuntimeError, "unfinished"):
            actual_generator(LanguageModel())(generation_mode="hybrid", max_new_tokens=2)

    def test_positive_mtp_retains_exactly_six_real_pbd_prediction_positions(self):
        from typing import List, Tuple
        path = ROOT / "eaglevl/model/locany/relation_modules.py"
        fn = next(n for n in ast.parse(path.read_text(encoding="utf-8")).body
                  if isinstance(n, ast.FunctionDef) and n.name == "pbd_prediction_positions")
        namespace = dict(torch=torch, List=List, Tuple=Tuple)
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
        lm = LanguageModel(boxes=2)
        actual_generator(lm)(generation_mode="hybrid", max_new_tokens=128)
        self.assertEqual(len(lm.mtp_inputs), 2)
        for tokens in lm.mtp_inputs:
            positions, _ = namespace["pbd_prediction_positions"](tokens, torch.tensor([tokens.numel()]), 1, 10, 6)
            self.assertEqual(len(positions), 6)
            self.assertEqual(tokens.reshape(-1)[positions].tolist(), [1, 10, 10, 10, 10, 10])

    def test_hybrid_retains_mtp_block_size_validation_after_initial_ar(self):
        with self.assertRaisesRegex(ValueError, "must match"):
            actual_generator(LanguageModel())(generation_mode="hybrid", n_future_tokens=5, max_new_tokens=64)


if __name__ == "__main__": unittest.main()
