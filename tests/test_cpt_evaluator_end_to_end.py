from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = REPO_ROOT / "scripts" / "eval_locany_cpt_learning.py"


class FakeTensor:
    def __init__(self, value):
        self.value = np.asarray(value)

    def __getitem__(self, key):
        return FakeTensor(self.value[key])

    def __setitem__(self, key, value):
        self.value[key] = value.value if isinstance(value, FakeTensor) else value

    def __iter__(self):
        for value in self.value:
            yield FakeTensor(value)

    def __len__(self):
        return len(self.value)

    def eq(self, value):
        return FakeTensor(self.value == value)

    def ne(self, value):
        return FakeTensor(self.value != value)

    def sum(self):
        return FakeTensor(self.value.sum())

    def item(self):
        return self.value.item()

    def to(self, *args, **kwargs):
        return self

    def detach(self):
        return self

    def float(self):
        return self


def fake_torch_module() -> types.ModuleType:
    module = types.ModuleType("torch")
    module.long = "long"
    module.Tensor = FakeTensor
    module.full_like = lambda tensor, value: FakeTensor(
        np.full_like(tensor.value, value)
    )
    module.where = lambda tensor: tuple(FakeTensor(axis) for axis in np.where(tensor.value))
    module.tensor = lambda value, **_kwargs: FakeTensor(value)
    module.is_tensor = lambda value: isinstance(value, FakeTensor)

    class InferenceMode:
        def __call__(self, function):
            return function

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    def inference_mode():
        return InferenceMode()

    module.inference_mode = inference_mode
    module.manual_seed = lambda _seed: None
    module.cuda = SimpleNamespace(
        is_available=lambda: False,
        manual_seed_all=lambda _seed: None,
        empty_cache=lambda: None,
    )
    return module


def load_evaluator_module():
    torch_stub = fake_torch_module()
    inference_stub = types.ModuleType("scripts.inference_ui_defect_locany")
    inference_stub.LocateAnythingInferencer = object
    spec = importlib.util.spec_from_file_location(
        "cpt_evaluator_end_to_end_target", EVALUATOR
    )
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(
        sys.modules,
        {
            "torch": torch_stub,
            "scripts.inference_ui_defect_locany": inference_stub,
            spec.name: module,
        },
    ):
        assert spec.loader is not None
        spec.loader.exec_module(module)
    return module


class FakeTokenizer:
    IDS = {"<|im_start|>": 1, "assistant": 2, "<|im_end|>": 3}

    def convert_tokens_to_ids(self, token):
        return self.IDS[token]


class FakeProcessor:
    tokenizer = FakeTokenizer()

    def py_apply_chat_template(self, messages, **_kwargs):
        self.messages = messages
        return "rendered conversation"

    def process_vision_info(self, messages):
        return [messages[0]["content"][0]["image"]], None

    def __call__(self, **_kwargs):
        # <im_start>user\n...<im_end><im_start>assistant\nanswer<im_end>
        return {
            "input_ids": FakeTensor([[1, 4, 99, 3, 1, 2, 10, 42, 3]]),
            "attention_mask": FakeTensor([[1] * 9]),
            "pixel_values": FakeTensor([[0.25]]),
            "image_grid_hws": FakeTensor([[2, 2]]),
        }


class FakeModel:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(lm_loss=FakeTensor(2.5))


class FakeOpenedImage:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def convert(self, _mode):
        return object()


class FakeInferencer:
    instances = []

    def __init__(self, namespace):
        self.namespace = namespace
        self.processor = FakeProcessor()
        self.model = FakeModel()
        self.device = namespace.device
        self.dtype = namespace.dtype
        self.instances.append(self)

    def predict(self, *, image, question):
        del image, question
        return "正确"


class CPTEvaluatorEndToEndTest(unittest.TestCase):
    @staticmethod
    def example(evaluator):
        return evaluator.Example(
            key="vqa:record-1",
            task="vqa",
            record_id="record-1",
            group_id="group-1",
            split="heldout",
            image=Path("screen.png"),
            prompt="is this correct?",
            target="正确",
            source="fixture.jsonl",
            line=1,
        )

    def test_teacher_forced_example_reaches_model_and_returns_ce(self):
        evaluator = load_evaluator_module()
        model = FakeModel()
        inferencer = SimpleNamespace(
            processor=FakeProcessor(),
            model=model,
            device="cuda:0",
            dtype="bfloat16",
        )
        example = self.example(evaluator)

        metrics = evaluator.teacher_forced_main_ce(
            inferencer, object(), example
        )

        self.assertEqual(metrics["teacher_forced_main_tokens"], 2)
        self.assertAlmostEqual(metrics["teacher_forced_main_token_ce"], 2.5)
        self.assertAlmostEqual(metrics["teacher_forced_main_loss_sum"], 5.0)
        self.assertEqual(len(model.calls), 1)
        self.assertIn("labels", model.calls[0])
        self.assertIn("pixel_values", model.calls[0])
        self.assertIn("image_grid_hws", model.calls[0])
        self.assertIn("image_flags", model.calls[0])

    def test_run_model_keeps_teacher_forced_result_in_prediction_row(self):
        evaluator = load_evaluator_module()
        args = SimpleNamespace(
            processor_path=None,
            base_model="/models/base",
            device="cuda:0",
            dtype="bf16",
            attn_implementation="sdpa",
            vision_attn_implementation="sdpa",
            max_new_tokens=32,
            allow_download=False,
            teacher_forced=True,
            seed=20260826,
        )
        FakeInferencer.instances.clear()
        with mock.patch.object(
            evaluator, "LocateAnythingInferencer", FakeInferencer
        ), mock.patch.object(
            evaluator.Image, "open", return_value=FakeOpenedImage()
        ):
            results = evaluator.run_model(
                "checkpoint", "/models/checkpoint-20", [self.example(evaluator)], args
            )

        result = results["vqa:record-1"]
        self.assertIsNone(result["error"])
        self.assertEqual(result["teacher_forced_main_tokens"], 2)
        self.assertAlmostEqual(result["teacher_forced_main_token_ce"], 2.5)
        self.assertEqual(result["metrics"]["vqa_accuracy"], 1.0)
        self.assertEqual(len(FakeInferencer.instances[0].model.calls), 1)


if __name__ == "__main__":
    unittest.main()
