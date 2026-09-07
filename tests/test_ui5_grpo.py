from copy import deepcopy
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from eaglevl.train.ui5_grpo_core import (
    GRPOConfig, TASKS, MixedSampler, centered_advantages, completion_loss_sums,
    lr_multiplier, remaining_budget,
)
from scripts.build_ui5_grpo_mixed_manifest import frozen_crop_routes, select_groups
from scripts.run_ui5_train_rollout_worker import load_module
from scripts.snapshot_ui5_train_rollouts import snapshot_rollout_payload
from scripts.ui5_grpo_reward import reward_from_boxes, score_trajectory


@pytest.fixture
def scorer():
    return load_module(ROOT / "qwen3vl_merge_and_score_fixed_5tasks.py", "grpo_test_scorer")


def test_reward_empty_invalid_partial_and_continuous_iou(scorer):
    assert reward_from_boxes(scorer, [], [])["reward"] == 1
    assert reward_from_boxes(scorer, [], [[0, 0, 10, 10]])["reward"] == 0
    assert reward_from_boxes(scorer, [[0, 0, 10, 10]], [])["reward"] == 0
    assert reward_from_boxes(scorer, [], [], invalid=True)["reward"] == -1
    partial = reward_from_boxes(scorer, [[0, 0, 10, 10], [20, 0, 30, 10]], [[0, 0, 10, 10]])
    assert partial["F_box"] == pytest.approx(2 / 3)
    assert partial["C_img"] == 1
    assert partial["S_iou"] == pytest.approx(2 / 3)
    low = reward_from_boxes(scorer, [[0, 0, 10, 10]], [[9, 0, 19, 10]])
    assert low["TP_box"] == 0
    assert 0 < low["S_iou"] < .1
    assert .2 < low["reward"] < .21


def test_production_parser_any_invalid_crop_penalizes_whole_trajectory(scorer):
    from scripts import inference_ui_defect_locany as parser
    group = dict(task="occlusion", width=200, height=100, gt_global=[[0, 0, 100, 100]],
                 views=[dict(crop_id="a", crop_index=0, crop_xyxy=[0, 0, 100, 100]),
                        dict(crop_id="b", crop_index=1, crop_xyxy=[100, 0, 200, 100])])
    outputs = [dict(raw_output="<ref>overlapping elements</ref><box><0><0><1000><1000></box><|im_end|>"),
               dict(raw_output="broken output")]
    result = score_trajectory(group, outputs, parser, scorer, GRPOConfig())
    assert result["reward"] == -1
    assert len(result["crop_outputs"]) == 2
    assert result["crop_outputs"][1]["raw_output"] == "broken output"
    assert result["pred_global"] == [[0, 0, 100, 100]]
    outputs[1]["raw_output"] = "<box>none</box><|im_end|>"
    assert score_trajectory(group, outputs, parser, scorer, GRPOConfig())["reward"] == pytest.approx(1)


def test_centering_without_standard_deviation_and_actual_token_loss():
    assert centered_advantages([.5] * 4) == [0] * 4
    assert centered_advantages([0, .2, .2, .4]) == pytest.approx([-.2, 0, 0, .2])
    current = torch.tensor([-.1, -.7, -2.0], requires_grad=True)
    old = current.detach().clone()
    policy, kl = completion_loss_sums(current, old, old, 0.5)
    assert policy.item() == pytest.approx(-1.5)
    assert kl.item() == pytest.approx(0)
    (policy / 3).backward()
    assert current.grad.tolist() == pytest.approx([-1 / 6] * 3)
    with pytest.raises(ValueError):
        completion_loss_sums(current, current, old, 1)


def test_task_first_available_strata_balance_and_resume():
    groups = [dict(group_id=f"{t}/{p}/{k}/{i}", task=t, polarity=p, crop_correct_count=k)
              for t in TASKS for p in ("positive", "negative") for k in (1, 2, 3)
              for i in range(1 if t == TASKS[0] else 4)]
    groups = [g for g in groups if not (g["task"] == TASKS[0] and g["polarity"] == "negative" and g["crop_correct_count"] == 3)]
    sampler = MixedSampler(groups)
    report = sampler.report(4800)
    for task in TASKS:
        assert sum(r["planned_draws"] for r in report if r["task"] == task) == 960
    missing = next(r for r in report if r["empty"])
    assert missing["planned_draws"] == 0
    assert [sampler.index_at(i) for i in range(401, 470)] == [MixedSampler(groups).index_at(i) for i in range(401, 470)]


def test_budget_and_optimizer_lr_horizon():
    config = GRPOConfig().validate()
    assert remaining_budget(7000, config) == 268
    assert remaining_budget(100, config) == 512
    with pytest.raises(ValueError):
        remaining_budget(7268, config)
    assert lr_multiplier(0, config) == 0
    assert lr_multiplier(40, config) == 1
    assert lr_multiplier(1200, config) == .5
    assert lr_multiplier(600, config) > .5


def raw_row(sid, route, correct=True):
    return dict(record_id=sid, sample_id=sid, source_image_id=sid, task="occlusion", prompt="task",
                width=20, height=20, gt_global=[], pred_global=[] if correct else [[1, 1, 5, 5]],
                model_id="crop", rollout_id=route, seed=100 + route, raw_output="raw",
                parse_status="ok", runtime_error=None, inference_success=True,
                exact_correct=correct, image_confusion="TN" if correct else "FP",
                grpo_eligible=True, image_relpath=sid + ".png")


def test_crop_only_selection_ignores_m31_and_rescores_old_exact(scorer):
    group = dict(raw_row("a", 0), m31_complete4=False, cross_model_complete8=False, difficulty=None)
    routes = {"a": {i: raw_row("a", i, i < 2) for i in range(4)}}
    # A previously wrong matching result is not accepted as the new count.
    routes["a"][0]["exact_correct"] = False
    selected, excluded, changed = select_groups({"a": group}, routes, scorer)
    assert selected[0]["crop_correct_count"] == 2
    assert len(changed) == 1
    assert not excluded
    routes["a"][2]["runtime_error"] = "old OOM"
    assert not select_groups({"a": group}, routes, scorer)[0]


def test_exact_frozen_boundary_uses_incomplete_records_and_rejects_mutation(tmp_path):
    snapshot, rollout = tmp_path / "snapshot", tmp_path / "rollout"
    snapshot.mkdir()
    rows = [raw_row("a", i, i == 1) for i in range(4)]
    frozen = dict(record_id="a", rollouts={"crop": [snapshot_rollout_payload("crop", i, row) for i, row in enumerate(rows)],
                                         "m31": [dict(status="missing")] * 4})
    (snapshot / "complete8.jsonl").write_text("")
    (snapshot / "incomplete_or_technical_error.jsonl").write_text(json.dumps(frozen) + "\n")
    for route, row in enumerate(rows):
        directory = rollout / "raw/crop" / f"rollout_{route}"
        directory.mkdir(parents=True)
        (directory / "part-00000.jsonl").write_text(json.dumps(row) + "\n" + json.dumps(raw_row("later", route)) + "\n")
    selected, evidence, scanned = frozen_crop_routes(snapshot, rollout)
    assert set(selected) == {"a"}
    assert len(evidence) == 4
    assert all(n == 2 for n in scanned.values())
    rows[0]["raw_output"] = "rewritten after freeze"
    (rollout / "raw/crop/rollout_0/part-00000.jsonl").write_text(json.dumps(rows[0]) + "\n")
    with pytest.raises(ValueError, match="raw differs"):
        frozen_crop_routes(snapshot, rollout)
