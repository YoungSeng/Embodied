"""Two real CPU/Gloo ranks: uneven crops, empty slots, group weights and resume.

The public DeepSpeed boundary calls use a small DDP engine here. CUDA ZeRO-2
is checked in the formal trainer; this test does not pretend to run H20s.
"""
from datetime import timedelta
from pathlib import Path

import torch
import pytest
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from eaglevl.train.ui5_grpo_core import (
    aligned_slot_count, centered_advantages, completion_loss_sums,
    deepspeed_slot_backward, zero_parameter_touch,
)
from eaglevl.train.ui5_grpo_runtime import backward_api_info


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[.2, -.3], [.1, .4]], dtype=torch.float64))
        self.replay_head = nn.Parameter(torch.tensor(.7, dtype=torch.float64))

    def forward(self, x=None):
        zero = zero_parameter_touch(self)
        return zero if x is None else (x @ self.weight).log_softmax(-1)[:, 0] + zero


def groups_for(rank):
    groups = []
    for g, crops in enumerate(([1, 3], [2, 1])[rank]):
        rows = []
        advantages = centered_advantages([0, .25, .5, 1.0] if g == 0 else [.5] * 4)
        for trajectory, advantage in enumerate(advantages):
            for crop in range(crops):
                length = 1 + (rank + g + trajectory + crop) % 4
                x = torch.arange(length * 2, dtype=torch.float64).reshape(length, 2) / 9
                x += (rank + 1) * .1 + g * .2 + crop * .3
                rows.append((x, advantage))
        groups.append((rows, sum(len(x) for x, _ in rows)))
    return groups


def objective(model, x, advantage, denominator):
    current = model(x)
    old = current.detach() - .02
    reference = current.detach() + .15
    policy, kl = completion_loss_sums(current, old, reference, advantage)
    return (policy + .02 * kl) / denominator


class BoundaryEngine:
    def __init__(self, module):
        self.module = module
        self.optimizer = torch.optim.SGD(module.parameters(), lr=.1)
        self.global_steps, self.calls = 0, 0

    def set_gradient_accumulation_boundary(self, boundary):
        self.boundary = boundary

    def backward(self, loss, scale_wrt_gas=True):
        assert scale_wrt_gas is False  # group/token denominator already applied
        loss.backward()
        self.calls += 1

    def step(self):
        if self.boundary:
            torch.nn.utils.clip_grad_norm_(self.module.parameters(), 1.0)
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.global_steps += 1


def nvtx_style_wrapper(function):
    # DeepSpeed 0.16/0.17 instrumentation forwards kwargs without __wrapped__.
    def wrapped_fn(*args, **kwargs):
        return function(*args, **kwargs)
    return wrapped_fn


class InstrumentedBoundaryEngine(BoundaryEngine):
    backward = nvtx_style_wrapper(BoundaryEngine.backward)


def worker(rank, rendezvous, output, instrumented):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=Path(rendezvous).as_uri(),
                            rank=rank, world_size=2, timeout=timedelta(seconds=45))
    try:
        model = DistributedDataParallel(Policy())
        engine = (InstrumentedBoundaryEngine if instrumented else BoundaryEngine)(model)
        api = backward_api_info(engine.backward)
        assert api["keyword_forwarding"] is instrumented
        groups = groups_for(rank)
        slots = [(x, a, count) for rows, count in groups for x, a in rows]
        count = aligned_slot_count(len(slots) + 2, torch.device("cpu"))
        for _ in range(2):
            for slot in range(count):
                if slot < len(slots):
                    x, advantage, tokens = slots[slot]
                    loss = objective(model, x, advantage, tokens * 2)
                elif slot < len(slots) + 2:
                    loss = model() + .1 * (model.module.replay_head - rank * .1).square() / 2
                else:
                    loss = model()
                deepspeed_slot_backward(engine, loss, final_slot=slot + 1 == count)
        torch.save(dict(state=model.module.state_dict(), steps=engine.global_steps,
                        calls=engine.calls, slots=count, local=len(slots) + 2),
                   Path(output) / f"rank{rank}.pt")
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("instrumented", [False, True], ids=["explicit-api", "nvtx-wrapped-api"])
def test_two_ranks_match_dense_equal_group_update_with_variable_crop_slots(tmp_path, instrumented):
    mp.spawn(worker, args=(str(tmp_path / "gloo.store"), str(tmp_path), instrumented), nprocs=2, join=True)
    states = [torch.load(tmp_path / f"rank{rank}.pt", weights_only=True) for rank in (0, 1)]
    assert states[0]["local"] != states[1]["local"]
    assert states[0]["slots"] == states[1]["slots"] == max(s["local"] for s in states)
    model = Policy()
    optimizer = torch.optim.SGD(model.parameters(), lr=.1)
    for _ in range(2):
        loss = 0
        for rank in (0, 1):
            for rows, tokens in groups_for(rank):
                loss += sum(objective(model, x, a, tokens * 4) for x, a in rows)
                loss += .1 * (model.replay_head - rank * .1).square() / 4
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
    for state in states:
        assert state["steps"] == 2
        assert state["calls"] == state["slots"] * 2
        for name, parameter in model.state_dict().items():
            torch.testing.assert_close(state["state"][name], parameter, atol=1e-12, rtol=1e-12)
