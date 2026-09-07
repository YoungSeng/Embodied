import inspect
import json
from pathlib import Path
import subprocess

import pytest
import torch

from eaglevl.train.ui5_grpo_core import deepspeed_slot_backward, digest
from eaglevl.train.ui5_grpo_runtime import (
    backward_api_info, bind_execution_code, verify_execution_code,
    sampling_probability_report,
)
from scripts.build_ui5_grpo_mixed_manifest import write_json


def git(root, *args):
    return subprocess.check_output(["git", *args], cwd=root, text=True).strip()


def commit(root, message):
    git(root, "add", ".")
    git(root, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", message)
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def run_repository(tmp_path):
    root, output = tmp_path / "code", tmp_path / "output"
    root.mkdir()
    output.mkdir()
    git(root, "init")
    source = root / "eaglevl/train/ui5_grpo_trainer.py"
    source.parent.mkdir(parents=True)
    source.write_text("original_runtime = True\n")
    original = commit(root, "original")
    run = dict(project_root=str(root), output_dir=str(output), code_sha=original,
               mixed_identity="frozen-pool", reference=dict(identity="fixed"))
    run["identity"] = digest(run)
    write_json(output / "run.json", run)
    return run, source


def test_instrumented_signature_forwards_keyword_without_double_scaling_or_retry():
    observed = []
    class Engine:
        def set_gradient_accumulation_boundary(self, value):
            observed.append(("boundary", value))
        def backward(self, loss, **kwargs):
            observed.append(("backward", kwargs))
            assert kwargs == {"scale_wrt_gas": False}
            loss.backward()
        def step(self):
            observed.append(("step",))
    engine = Engine()
    assert "scale_wrt_gas" not in inspect.signature(engine.backward).parameters
    assert backward_api_info(engine.backward)["keyword_forwarding"]
    parameter = torch.tensor(2., requires_grad=True)
    deepspeed_slot_backward(engine, parameter.square() / 8, final_slot=True)
    assert parameter.grad.item() == .5
    assert observed == [("boundary", True), ("backward", {"scale_wrt_gas": False}), ("step",)]


def test_no_unsupported_keyword_or_backward_retry_is_silently_allowed():
    with pytest.raises(RuntimeError, match="cannot accept"):
        backward_api_info(lambda loss, retain_graph=False: None)
    calls = []
    class FailingEngine:
        def set_gradient_accumulation_boundary(self, value):
            pass
        def backward(self, loss, **kwargs):
            calls.append(kwargs)
            raise TypeError("failure inside backward after side effect")
        def step(self):
            raise AssertionError("must not update after failed backward")
    with pytest.raises(TypeError, match="side effect"):
        deepspeed_slot_backward(FailingEngine(), torch.tensor(1.), final_slot=True)
    assert len(calls) == 1


def test_runtime_update_preserves_run_and_step_zero_and_binds_new_commit(run_repository):
    run, source = run_repository
    output, root = Path(run["output_dir"]), Path(run["project_root"])
    original_run = (output / "run.json").read_bytes()
    baseline = output / "evaluation/step-000000/ui5_metrics.json"
    baseline.parent.mkdir(parents=True)
    baseline.write_bytes(b'{"completed":true}')
    assert verify_execution_code(run) == run["code_sha"]
    source.write_text("nvtx_keyword_forwarding = True\n")
    newer = commit(root, "runtime repair")
    with pytest.raises(ValueError, match="resume-code-update"):
        bind_execution_code(run)
    with pytest.raises(ValueError, match="resume-code-update"):
        verify_execution_code(run)
    assert bind_execution_code(run, allow_update=True) == newer
    assert verify_execution_code(run) == newer
    record_path = output / "runtime_code_revision.json"
    saved = record_path.read_bytes()
    assert bind_execution_code(run) == newer
    assert record_path.read_bytes() == saved
    assert (output / "run.json").read_bytes() == original_run
    assert baseline.read_bytes() == b'{"completed":true}'
    assert len(list((output / "diagnostics/code_revisions").glob("*.json"))) == 1
    record = json.loads(saved)
    assert record["run_code_sha"] == run["code_sha"]
    assert record["execution_code_sha"] == newer
    record["changed_files"] = {}
    write_json(record_path, record)
    with pytest.raises(ValueError, match="record changed"):
        verify_execution_code(run)


def test_runtime_update_rejects_model_change_dirty_source_and_modified_run(run_repository):
    run, source = run_repository
    source.write_text("uncommitted = True\n")
    with pytest.raises(ValueError, match="uncommitted"):
        bind_execution_code(run, allow_update=True)
    root = Path(run["project_root"])
    model = root / "eaglevl/model/locany/relation_modules.py"
    model.parent.mkdir(parents=True)
    model.write_text("changed_model = True\n")
    commit(root, "model change requires a new experiment")
    with pytest.raises(ValueError, match="experiment contract"):
        bind_execution_code(run, allow_update=True)
    changed = dict(run, mixed_identity="different-pool")
    with pytest.raises(ValueError, match="run.json"):
        bind_execution_code(changed, allow_update=True)


def test_precision_repair_preserves_hybrid_baseline_and_can_then_resume_updates(run_repository):
    run, source = run_repository
    root, output = Path(run["project_root"]), Path(run["output_dir"])
    model = root / "eaglevl/model/locany/ui5_ar.py"
    model.parent.mkdir(parents=True)
    model.write_text("precision_version = 1\n")
    newer = commit(root, "AR numerical precision before first saved update")
    assert bind_execution_code(run, allow_update=True) == newer
    assert json.loads((output / "runtime_code_revision.json").read_text())["ar_precision_repair"]
    (output / "resume/latest").mkdir(parents=True)
    assert bind_execution_code(run) == newer
    assert verify_execution_code(run) == newer
    # Unrelated runtime fixes remain allowed after an already audited repair.
    source.write_text("runtime_telemetry = True\n")
    commit(root, "telemetry")
    bind_execution_code(run, allow_update=True)
    model.write_text("precision_version = 2\n")
    commit(root, "cannot switch AR numerics mid training")
    with pytest.raises(ValueError, match="after saved optimizer"):
        bind_execution_code(run, allow_update=True)


def test_precision_repair_cannot_accept_old_saved_optimizer_updates(run_repository):
    run, _ = run_repository
    root, output = Path(run["project_root"]), Path(run["output_dir"])
    model = root / "eaglevl/model/locany/ui5_ar.py"
    model.parent.mkdir(parents=True)
    model.write_text("precision_version = 1\n")
    commit(root, "precision repair")
    (output / "resume/.previous").mkdir(parents=True)
    with pytest.raises(ValueError, match="after saved optimizer"):
        bind_execution_code(run, allow_update=True)
    assert not (output / "runtime_code_revision.json").exists()


def test_on_policy_guard_handles_observed_bf16_scale_without_modifying_old():
    old = torch.full((100,), -5.)
    saved = old.clone()
    delta = torch.full_like(old, (100 * .01397884264588356 - .12348008155822754) / 99)
    delta[0] = .12348008155822754
    current = (old + delta).requires_grad_()
    report = sampling_probability_report(current, old, .2)
    assert report["valid"]
    assert report["max_logprob_error"] > .08 and report["mean_logprob_error"] > .01
    assert report["max_sampling_ratio"] < 1.2 and report["sampling_numerical_kl"] < .001
    assert report["worst_token_index"] == 0
    assert torch.equal(old, saved) and current.grad is None


@pytest.mark.parametrize("delta", [.3, -.3, .1, float("nan"), float("inf")])
def test_on_policy_guard_rejects_clipping_large_average_drift_and_nonfinite(delta):
    old = torch.full((10,), -5.)
    assert not sampling_probability_report(old + delta, old, .2)["valid"]
    with pytest.raises(ValueError, match="token counts"):
        sampling_probability_report(old[:-1], old, .2)
