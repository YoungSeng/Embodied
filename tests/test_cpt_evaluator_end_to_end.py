from __future__ import annotations

import ast
import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EVALUATOR = REPO_ROOT / "scripts" / "eval_locany_cpt_learning.py"
INFERENCER = REPO_ROOT / "scripts" / "inference_ui_defect_locany.py"
REPOSITORY_MODEL = REPO_ROOT / "eaglevl" / "utils" / "locany" / "modeling_locateanything.py"


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
    module.as_tensor = lambda value, **_kwargs: FakeTensor(value)
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


class NumpyGridProcessor(FakeProcessor):
    def __call__(self, **kwargs):
        result = super().__call__(**kwargs)
        result["image_grid_hws"] = np.asarray([[2, 2]], dtype=np.int64)
        return result


class FakeModel:
    def __init__(self):
        self.calls = []
        self.language_model = SimpleNamespace(
            model=SimpleNamespace(training=False)
        )
        self.decoder_training_during_call = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        self.decoder_training_during_call.append(
            self.language_model.model.training
        )
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


class FailingInferencer(FakeInferencer):
    def predict(self, *, image, question):
        del image, question
        raise TypeError("generation sentinel")


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
        self.assertIs(model.calls[0]["use_cache"], False)
        self.assertEqual(model.decoder_training_during_call, [True])
        self.assertIs(model.language_model.model.training, False)

    def test_inference_namespace_disables_relation_and_pbd_for_cpt_eval(self):
        evaluator = load_evaluator_module()
        args = SimpleNamespace(
            processor_path=None,
            base_model="/models/base",
            device="cuda:0",
            dtype="bf16",
            attn_implementation="sdpa",
            vision_attn_implementation="flash_attention_2",
            max_new_tokens=1024,
            allow_download=False,
        )

        namespace = evaluator.inference_namespace(args, "/models/checkpoint-0")

        self.assertEqual(namespace.checkpoint, "/models/checkpoint-0")
        self.assertFalse(namespace.enable_ui_relation)
        self.assertFalse(namespace.enable_pbd)

        tree = ast.parse(INFERENCER.read_text(encoding="utf-8"))
        inferencer_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef)
            and node.name == "LocateAnythingInferencer"
        )
        required_args = set()
        for node in ast.walk(inferencer_class):
            if not isinstance(node, ast.Attribute):
                continue
            if isinstance(node.value, ast.Name) and node.value.id == "args":
                required_args.add(node.attr)
            if (
                isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id == "self"
                and node.value.attr == "args"
            ):
                required_args.add(node.attr)
        self.assertEqual(required_args.difference(vars(namespace)), set())

    def test_hash_subset_keeps_all_five_ui_defect_classes_when_available(self):
        evaluator = load_evaluator_module()
        labels = ["文字溢出", "文本省略", "元素遮挡", "元素裁切", "内容缺失"]
        candidates = [
            evaluator.Example(
                key=f"ui_defect:record-{index}",
                task="ui_defect",
                record_id=f"record-{index}",
                group_id=f"group-{index}",
                split="heldout",
                image="screen.png",
                prompt="find defects",
                target=(
                    f"<ref>{label}</ref>"
                    f"<box><{index}><{index}><{index + 10}><{index + 10}></box>"
                ),
                source="fixture.jsonl",
                line=index,
            )
            for index, label in enumerate(labels, start=1)
        ]

        selected = evaluator._select_examples(
            [*reversed(candidates)], 5, "hash", 20260826, "ui_defect"
        )
        selected_labels = {
            evaluator.canonical_defect_label(item["label"])
            for example in selected
            for item in evaluator.parse_labeled_boxes(example.target)
        }
        self.assertEqual(selected_labels, set(evaluator.UI_DEFECT_CLASSES))

    def test_teacher_forced_converts_numpy_image_grid_to_torch(self):
        evaluator = load_evaluator_module()
        model = FakeModel()
        inferencer = SimpleNamespace(
            processor=NumpyGridProcessor(),
            model=model,
            device="cuda:0",
            dtype="bfloat16",
        )
        metrics = evaluator.teacher_forced_main_ce(
            inferencer, object(), self.example(evaluator)
        )
        self.assertEqual(metrics["teacher_forced_main_tokens"], 2)
        self.assertIsInstance(model.calls[0]["image_grid_hws"], FakeTensor)

    def test_evaluator_uses_shared_nas_compatible_lock(self):
        source = EVALUATOR.read_text(encoding="utf-8")
        self.assertIn("from eaglevl.train.cpt_eval_queue import", source)
        self.assertIn("exclusive_file_lock", source)
        self.assertIn("fsync_if_supported", source)
        self.assertNotIn("fcntl.flock", source)

    def test_new_point_one_rows_remove_stale_point_five_heldout_rows(self):
        evaluator = load_evaluator_module()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output_dir = root / "eval/checkpoint-20"
            output_dir.mkdir(parents=True)
            append_path = root / "diagnostics/cpt_eval_metrics.jsonl"
            append_path.parent.mkdir(parents=True)
            append_path.write_text(
                "".join(
                    json.dumps(row) + "\n"
                    for row in (
                        {
                            "evaluation_id": "old-heldout",
                            "checkpoint": "/run/checkpoint-10",
                            "step": 10,
                            "split": "heldout",
                            "task": "ui_defect",
                            "iou_threshold": 0.5,
                        },
                        {
                            "evaluation_id": "external",
                            "checkpoint": "/run/checkpoint-10",
                            "step": 10,
                            "split": "external_ui5",
                            "task": "ui_defect_external",
                            "iou_threshold": 0.1,
                        },
                    )
                ),
                encoding="utf-8",
            )
            current = {
                "evaluation_id": "new-heldout",
                "checkpoint": "/run/checkpoint-20",
                "step": 20,
                "split": "heldout",
                "task": "ui_defect",
                "iou_threshold": 0.1,
            }

            evaluator.write_eval_metric_rows(output_dir, [current], append_path)

            rows = [
                json.loads(line)
                for line in append_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                {row["evaluation_id"] for row in rows},
                {"external", "new-heldout"},
            )

    def test_repository_generation_supports_legacy_base_without_relation_attribute(self):
        inferencer_source = INFERENCER.read_text(encoding="utf-8")
        model_source = REPOSITORY_MODEL.read_text(encoding="utf-8")
        self.assertIn(
            "self.model.enable_ui_relation = bool(",
            inferencer_source,
        )
        self.assertIn(
            'getattr(self, "enable_ui_relation", False)',
            model_source,
        )

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
            iou_threshold=0.1,
            allow_download=False,
            teacher_forced=True,
            seed=20260826,
        )
        FakeInferencer.instances.clear()
        stdout = io.StringIO()
        with mock.patch.object(
            evaluator, "LocateAnythingInferencer", FakeInferencer
        ), mock.patch.object(
            evaluator.Image, "open", return_value=FakeOpenedImage()
        ), mock.patch.object(evaluator.sys, "stdout", stdout):
            results = evaluator.run_model(
                "checkpoint", "/models/checkpoint-20", [self.example(evaluator)], args
            )

        result = results["vqa:record-1"]
        self.assertIsNone(result["error"])
        self.assertEqual(result["teacher_forced_main_tokens"], 2)
        self.assertAlmostEqual(result["teacher_forced_main_token_ce"], 2.5)
        self.assertEqual(result["metrics"]["vqa_accuracy"], 1.0)
        self.assertEqual(len(FakeInferencer.instances[0].model.calls), 1)
        progress = stdout.getvalue()
        self.assertIn(
            "model=checkpoint sample=1/1 task=vqa record=record-1 START",
            progress,
        )
        self.assertIn("record=record-1 DONE status=ok", progress)

    def test_run_model_fail_fast_preserves_phase_and_original_traceback(self):
        evaluator = load_evaluator_module()
        args = SimpleNamespace(
            processor_path=None,
            base_model="/models/base",
            device="cuda:0",
            dtype="bf16",
            attn_implementation="sdpa",
            vision_attn_implementation="flash_attention_2",
            max_new_tokens=32,
            iou_threshold=0.1,
            allow_download=False,
            teacher_forced=True,
            fail_fast_inference_errors=True,
            seed=20260826,
        )
        stderr = io.StringIO()
        with mock.patch.object(
            evaluator, "LocateAnythingInferencer", FailingInferencer
        ), mock.patch.object(
            evaluator.Image, "open", return_value=FakeOpenedImage()
        ), mock.patch.object(evaluator.sys, "stderr", stderr):
            with self.assertRaisesRegex(
                RuntimeError,
                r"(?s)phase=generation; TypeError: generation sentinel.*original traceback",
            ):
                evaluator.run_model(
                    "base", "/models/base", [self.example(evaluator)], args
                )

        diagnostic = stderr.getvalue()
        self.assertIn("phase=generation", diagnostic)
        self.assertIn("in predict", diagnostic)
        self.assertIn("TypeError: generation sentinel", diagnostic)


if __name__ == "__main__":
    unittest.main()
