"""Real DeepSpeed ZeRO-2 on CPU/Gloo, plus ordering/participation invariants.

DeepSpeed is optional in lightweight developer environments. These integration
tests use its actual leaf hooks, IPG buckets, partitioning and optimizer, not
DDP. Only the optional CPU shared-memory extension is disabled; communication
uses real torch/Gloo. They do not claim to exercise H20/CUDA kernels.
"""
from datetime import timedelta
import importlib.util
import inspect
from pathlib import Path
from types import SimpleNamespace
import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from eaglevl.train.ui5_grpo_core import aligned_slot_count, completion_loss_sums, zero_parameter_touch
from eaglevl.train.ui5_grpo_zero2 import OrderedZero2Reduction


def test_ready_order_is_fixed_and_missing_or_repeated_gradients_fail():
    params = [nn.Parameter(torch.ones(n)) for n in (2, 3, 5)]
    reduced = []
    zero = SimpleNamespace(partition_gradients=True, overlap_comm=False, cpu_offload=False,
                           use_grad_accum_attribute=False, contiguous_gradients=True,
                           bit16_groups=[params], reduce_bucket_size=4, reduce_scatter=True,
                           communication_data_type=torch.float32,
                           get_gradient_for_reduction=lambda p: p.grad,
                           reduce_ready_partitions_and_remove_grads=lambda p, i: reduced.append(id(p)),
                           overlapping_partition_gradients_reduce_epilogue=lambda: reduced.append("epilogue"))
    reducer = OrderedZero2Reduction(zero, [(str(i), p) for i, p in enumerate(params)])
    for order in ([0, 1, 2], [2, 0, 1]):
        reducer.begin_slot(1, 0)
        for i in order:
            params[i].grad = torch.ones_like(params[i])
            zero.reduce_ready_partitions_and_remove_grads(params[i], 0)
        zero.overlapping_partition_gradients_reduce_epilogue()
        assert reducer.end_slot()["reduced_parameters"] == 3
    assert reduced == [id(params[2]), id(params[1]), id(params[0]), "epilogue"] * 2
    reducer.begin_slot(2, 0)
    zero.reduce_ready_partitions_and_remove_grads(params[0], 0)
    with pytest.raises(RuntimeError, match="repeated"):
        zero.reduce_ready_partitions_and_remove_grads(params[0], 0)
    with pytest.raises(RuntimeError, match="incomplete"):
        zero.overlapping_partition_gradients_reduce_epilogue()


class BranchingPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.parameters_by_path = nn.ParameterList([
            nn.Parameter(torch.linspace(-.2, .3, n)) for n in (7, 13, 17, 29, 5, 31)])

    def forward(self, x=None, replay=False):
        zero = zero_parameter_touch(self)
        if x is None:
            return zero
        indices = (5, 2, 0, 4, 1, 3) if replay else (0, 2, 5)
        value = torch.tensor(.2)
        for i in indices:
            value = (value * (1 + self.parameters_by_path[i].sum() / 100)
                     + self.parameters_by_path[i].square().sum() * (i + 1) / 100).tanh()
        if replay:
            return (value - x).square() + zero
        logits = torch.stack([x * value, -x * value], -1)
        return logits.log_softmax(-1)[:, 0] + zero


def rank_objectives(model, rank):
    objectives = []
    # One rank reaches replay and padding while the other still runs AR.
    for group, crops in enumerate(([1, 3], [2, 1])[rank]):
        records = []
        for trajectory, advantage in enumerate((-.4375, -.1875, .0625, .5625)):
            for crop in range(crops):
                x = torch.arange(1 + (trajectory + crop) % 3, dtype=torch.float32) / 5 + .2
                records.append((x, advantage if group == 0 else 0.0))
        token_count = sum(len(x) for x, _ in records)
        for x, advantage in records:
            def objective(x=x, advantage=advantage, token_count=token_count):
                current = model(x)
                policy, kl = completion_loss_sums(current, current.detach() - .02,
                                                  current.detach() + .15, advantage)
                return (policy + .02 * kl) / (2 * token_count)
            objectives.append(objective)
    for _ in range(2):
        objectives.append(lambda: .1 * model(torch.tensor(.4 + rank * .1), replay=True) / 2)
    return objectives


def zero_worker(rank, rendezvous, output, ordered):
    os.environ["DS_ACCELERATOR"] = "cpu"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_SIZE"] = "2"
    torch.set_num_threads(1)
    import deepspeed.comm as ds_dist
    from importlib import import_module
    ds_torch = import_module("deepspeed.comm.torch")
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    from deepspeed.utils.timer import SynchronizedWallClockTimer
    # Optional CPU JIT extension is unrelated to ZeRO; use portable Gloo.
    ds_torch.build_shm_op = lambda: None
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2,
                            timeout=timedelta(seconds=45))
    ds_dist.init_distributed(dist_backend="gloo", dist_init_required=False)
    try:
        model = BranchingPolicy()
        named = list(model.named_parameters())
        base = torch.optim.AdamW([dict(params=list(model.parameters())[:3]),
                                 dict(params=list(model.parameters())[3:])], lr=.01)
        optimizer_options = ({"optimizer_params": {}} if "optimizer_params" in
                             inspect.signature(DeepSpeedZeroOptimizer).parameters else {})
        zero = DeepSpeedZeroOptimizer(base, {p: name for name, p in named}, SynchronizedWallClockTimer(),
                    **optimizer_options, dp_process_group=dist.group.WORLD,
                    partition_grads=True, contiguous_gradients=True, overlap_comm=False,
                    reduce_scatter=True, reduce_bucket_size=19, allgather_bucket_size=128,
                    gradient_accumulation_steps=2, gradient_accumulation_dtype=torch.float32,
                    communication_data_type=torch.float32, clip_grad=1.0)
        reducer = OrderedZero2Reduction(zero, model.named_parameters()) if ordered else None
        # CPU scheduling does not reproduce all CUDA hook interleavings. Delay
        # ready notifications until leaves are complete, then deliberately
        # deliver different legal permutations. Gradients, native leaf hooks
        # and all bucket/reduction/partition/optimizer math remain real ZeRO-2.
        ready_handler = zero.reduce_ready_partitions_and_remove_grads
        ready = []
        def permuted_ready(parameter, group):
            ready.append((parameter, group))
            if len(ready) == len(named):
                delivery = sorted(ready, key=lambda pair: zero.get_param_id(pair[0]), reverse=bool(rank))
                ready.clear()
                for p, i in delivery:
                    ready_handler(p, i)
        zero.reduce_ready_partitions_and_remove_grads = permuted_ready
        native_average = zero.average_tensor
        layouts = []
        def average(tensor, *args, **kwargs):
            # Verify parameter identity/order, not just tensor size. A same-size
            # misordered bucket could silently average unrelated parameters.
            if hasattr(zero, "ipg_buckets"):
                active = [p for bucket in zero.ipg_buckets.values() for p in bucket.params]
            else:
                active = zero.params_in_ipg_bucket
            layout = (tensor.numel(), tuple(tuple(p) for p in active))
            across = [None, None]
            dist.all_gather_object(across, layout)
            if across[0] != across[1]:
                raise RuntimeError("upstream hook order produced different ZeRO-2 buckets")
            layouts.append(layout)
            return native_average(tensor, *args, **kwargs)
        zero.average_tensor = average
        objectives = rank_objectives(model, rank)
        slots = aligned_slot_count(len(objectives), torch.device("cpu"))
        for step in range(2):
            for slot in range(slots):
                zero.is_gradient_accumulation_boundary = slot + 1 == slots
                loss = objectives[slot]() if slot < len(objectives) else model()
                if reducer:
                    reducer.begin_slot(step, slot)
                try:
                    zero.backward(loss)
                    zero.overlapping_partition_gradients_reduce_epilogue()
                except RuntimeError as error:
                    if not ordered and "different ZeRO-2 buckets" in str(error):
                        torch.save(dict(reproduced=True), Path(output) / f"rank{rank}.pt")
                        return
                    raise
                if reducer:
                    reducer.end_slot()
                if zero.is_gradient_accumulation_boundary:
                    zero.step()
            if step == 0:
                # Exercise native partitioned Adam state serialization/reload.
                states = [None, None]
                saved = Path(output) / f"optimizer-rank{rank}.pt"
                torch.save(zero.state_dict(), saved)
                dist.all_gather_object(states, torch.load(saved, weights_only=False))
                zero.load_state_dict(states, load_optimizer_states=True, load_from_fp32_weights=True)
        torch.save(dict(state=model.state_dict(), layouts=layouts, slots=slots), Path(output) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(importlib.util.find_spec("deepspeed") is None, reason="requires DeepSpeed's actual ZeRO-2 implementation")
@pytest.mark.parametrize("ordered", [False, True], ids=["permuted-hook-order-regression", "ordered-zero2-update"])
def test_real_zero2_different_graphs_buckets_and_dense_update(tmp_path, ordered):
    mp.spawn(zero_worker, args=(str(tmp_path / "gloo.store"), str(tmp_path), ordered), nprocs=2, join=True)
    results = [torch.load(tmp_path / f"rank{rank}.pt", weights_only=True) for rank in (0, 1)]
    if not ordered:
        assert all(r["reproduced"] for r in results)
        return
    assert results[0]["layouts"] == results[1]["layouts"]
    model = BranchingPolicy()
    optimizer = torch.optim.AdamW([dict(params=list(model.parameters())[:3]),
                                  dict(params=list(model.parameters())[3:])], lr=.01)
    for _ in range(2):
        sum(objective() / 2 for rank in (0, 1) for objective in rank_objectives(model, rank)).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    for result in results:
        for name, expected in model.state_dict().items():
            torch.testing.assert_close(result["state"][name], expected, atol=1e-6, rtol=1e-6)


def native_ar_zero_worker(rank, rendezvous, output):
    """Combine native nested AR checkpoints with the real ZeRO leaf hooks."""
    from copy import deepcopy
    from importlib import import_module
    from tests.test_ui5_grpo_native_ar import fixture_model
    from eaglevl.model.locany.ui5_ar import forward_ar, sample_ar
    from eaglevl.train.ui5_grpo_runtime import sampling_probability_report
    os.environ["DS_ACCELERATOR"] = "cpu"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["LOCAL_SIZE"] = "2"
    torch.set_num_threads(1)
    import deepspeed.comm as ds_dist
    from deepspeed.runtime.zero.stage_1_and_2 import DeepSpeedZeroOptimizer
    from deepspeed.utils.timer import SynchronizedWallClockTimer
    import_module("deepspeed.comm.torch").build_shm_op = lambda: None
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(), rank=rank, world_size=2,
                            timeout=timedelta(seconds=90))
    ds_dist.init_distributed(dist_backend="gloo", dist_init_required=False)
    try:
        model, inputs = fixture_model(kv_heads=1)
        model.to(torch.bfloat16)
        inputs["pixel_values"] = inputs["pixel_values"].bfloat16()
        model.language_model.model.gradient_checkpointing = True
        reference = deepcopy(model).requires_grad_(False)
        initial = {name: p.detach().clone() for name, p in model.named_parameters()}
        base = torch.optim.AdamW(model.parameters(), lr=.005)
        options = ({"optimizer_params": {}} if "optimizer_params" in
                   inspect.signature(DeepSpeedZeroOptimizer).parameters else {})
        zero = DeepSpeedZeroOptimizer(base, {p: name for name, p in model.named_parameters()},
                    SynchronizedWallClockTimer(), **options, dp_process_group=dist.group.WORLD,
                    partition_grads=True, contiguous_gradients=True, overlap_comm=False,
                    reduce_scatter=True, reduce_bucket_size=700, allgather_bucket_size=1024,
                    gradient_accumulation_steps=2, gradient_accumulation_dtype=torch.float32,
                    communication_data_type=torch.bfloat16, clip_grad=1.0)
        reducer = OrderedZero2Reduction(zero, model.named_parameters())
        reports = []
        for step in range(3):
            completions = []
            for trajectory, advantage in enumerate((-.5, -1 / 6, 1 / 6, .5)):
                for crop in range(rank + 1):
                    sampled = sample_ar(model, inputs, max_new_tokens=8, eos_token_id=3,
                                        seed=1000 * step + 100 * rank + 10 * trajectory + crop)
                    sequence = dict(inputs, input_ids=torch.cat([inputs["input_ids"], sampled["tokens"][None]], 1))
                    completions.append((sequence, sampled, advantage))
            with torch.no_grad():
                for sequence, sampled, _ in completions:
                    sampled["reference"] = forward_ar(reference, **sequence, completion_start=3).completion_log_probs
            tokens = sum(len(sampled["tokens"]) for _, sampled, _ in completions)
            slots = aligned_slot_count(len(completions) + 1, torch.device("cpu"))
            for slot in range(slots):
                zero.is_gradient_accumulation_boundary = slot + 1 == slots
                if slot < len(completions):
                    sequence, sampled, advantage = completions[slot]
                    current = forward_ar(model, **sequence, completion_start=3).completion_log_probs
                    report = sampling_probability_report(current, sampled["old_log_probs"], .2)
                    assert report["valid"], report
                    reports.append(report)
                    policy, kl = completion_loss_sums(current, sampled["old_log_probs"], sampled["reference"], advantage)
                    loss = (policy + .02 * kl) / tokens + zero_parameter_touch(model)
                elif slot == len(completions):
                    # A different supervised graph, like the existing branch
                    # regression. Full UI5 SFT/MTP data is not needed to test
                    # whether nested AR checkpoint hooks communicate correctly.
                    loss = .1 * model.mlp1(inputs["pixel_values"]).float().square().mean() + zero_parameter_touch(model)
                else:
                    loss = zero_parameter_touch(model)
                reducer.begin_slot(step, slot)
                zero.backward(loss)
                zero.overlapping_partition_gradients_reduce_epilogue()
                reducer.end_slot()
                if zero.is_gradient_accumulation_boundary:
                    zero.step()
        for prefix in ("language_model.", "mlp1.", "relation_encoder."):
            assert any(not torch.equal(p, initial[name]) for name, p in model.named_parameters() if name.startswith(prefix))
        torch.save(dict(state=model.state_dict(), reports=reports), Path(output) / f"native-rank{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(importlib.util.find_spec("deepspeed") is None, reason="requires DeepSpeed's actual ZeRO-2 implementation")
def test_native_bf16_ar_checkpointed_gradients_with_real_two_rank_zero2(tmp_path):
    mp.spawn(native_ar_zero_worker, args=(str(tmp_path / "native.store"), str(tmp_path)), nprocs=2, join=True)
    results = [torch.load(tmp_path / f"native-rank{rank}.pt", weights_only=True) for rank in (0, 1)]
    for name, tensor in results[0]["state"].items():
        torch.testing.assert_close(tensor, results[1]["state"][name], atol=0, rtol=0)
    assert len(results[0]["reports"]) == 3 * 4
    assert len(results[1]["reports"]) == 3 * 8
    assert all(report["valid"] for result in results for report in result["reports"])
