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
