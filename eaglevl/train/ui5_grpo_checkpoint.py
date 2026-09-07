"""Transactional ZeRO-2 resume with per-rank RNG/sampler and a fixed reference."""
from __future__ import annotations
import json
import os
from pathlib import Path
import shutil

from eaglevl.train.ui5_grpo_core import digest, file_sha


def validate_checkpoint(path, identity, expected_ranks=2):
    path = Path(path)
    marker = json.loads((path / "grpo_complete.json").read_text())
    if marker["run_identity"] != identity or marker["world_size"] != expected_ranks:
        raise ValueError("GRPO resume run/reference/manifest identity or world size changed")
    if marker["files_digest"] != digest(marker["files"]):
        raise ValueError("GRPO resume inventory identity changed")
    actual = {p.relative_to(path).as_posix(): p.stat().st_size for p in path.rglob("*")
              if p.is_file() and p.name != "grpo_complete.json"}
    if actual != {name: entry["bytes"] for name, entry in marker["files"].items()}:
        raise ValueError("GRPO resume file set or sizes changed")
    for name, metadata in marker["files"].items():
        if file_sha(path / name) != metadata["sha256"]:
            raise ValueError(f"GRPO resume checkpoint corrupted: {name}")
    required = {f"rng_sampler_rank{rank}.pt" for rank in range(expected_ranks)}
    if not required <= set(actual) or not (path / "trainer_state.json").is_file():
        raise ValueError("GRPO resume lacks rank state")
    optimizer_files = [name for name in actual if name.endswith("optim_states.pt")]
    if len(optimizer_files) != expected_ranks or not any(name.endswith("model_states.pt") for name in actual):
        raise ValueError("GRPO resume lacks complete ZeRO-2 optimizer/model state")
    trainer_state = json.loads((path / "trainer_state.json").read_text())
    if trainer_state["global_step"] != marker["step"]:
        raise ValueError("GRPO resume global step differs")
    return marker


def recover(root, identity):
    root = Path(root)
    latest, pending, previous = root / "latest", root / ".pending", root / ".previous"
    # An interrupted save without a commit marker is never accepted.
    if pending.exists() and not (pending / "grpo_complete.json").exists():
        shutil.rmtree(pending)
    if pending.exists():
        newer = validate_checkpoint(pending, identity)
        if latest.exists():
            older = validate_checkpoint(latest, identity)
            if newer["step"] <= older["step"]:
                raise ValueError("pending GRPO checkpoint does not advance latest")
            if previous.exists():
                shutil.rmtree(previous)
            os.replace(latest, previous)
        os.replace(pending, latest)
    if not latest.exists() and previous.exists():
        validate_checkpoint(previous, identity)
        os.replace(previous, latest)
    if latest.exists():
        marker = validate_checkpoint(latest, identity)
        if previous.exists():
            shutil.rmtree(previous)
        return marker["step"]
    return 0


def save_checkpoint(engine, run, step, rank_state):
    import numpy as np
    import random
    import torch
    import torch.distributed as dist
    from scripts.build_ui5_grpo_mixed_manifest import write_json
    from scripts.patch_locany_checkpoint import patch_checkpoint
    rank = dist.get_rank()
    root = Path(run["output_dir"]) / "resume"
    pending, latest, previous = root / ".pending", root / "latest", root / ".previous"
    if rank == 0:
        root.mkdir(parents=True, exist_ok=True)
        if pending.exists():
            raise ValueError("unrecovered pending GRPO checkpoint")
        pending.mkdir()
    dist.barrier()
    state = dict(rank_state, step=step, run_identity=run["identity"],
                 python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                 torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())
    torch.save(state, pending / f"rng_sampler_rank{rank}.pt")
    engine.save_checkpoint(str(pending / "deepspeed"), tag="state",
                           client_state=dict(step=step, run_identity=run["identity"]), save_latest=True)
    dist.barrier()
    if rank == 0:
        engine.module.save_pretrained(pending, safe_serialization=True)
        write_json(pending / "trainer_state.json", dict(global_step=step, training_mode="ui5_grpo",
                                                       run_identity=run["identity"]))
        patch_checkpoint(base_model=Path(run["processor_path"]), checkpoint=pending,
                         project_root=Path(run["project_root"]), force=True, validate_relation_weights=True)
        files = {}
        for path in sorted(pending.rglob("*")):
            if path.is_file():
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
                files[path.relative_to(pending).as_posix()] = dict(bytes=path.stat().st_size, sha256=file_sha(path))
        write_json(pending / "grpo_complete.json", dict(step=step, run_identity=run["identity"],
                   reference_identity=run["reference"]["identity"], world_size=dist.get_world_size(),
                   files=files, files_digest=digest(files)))
        if previous.exists():
            shutil.rmtree(previous)
        if latest.exists():
            os.replace(latest, previous)
        os.replace(pending, latest)
        if previous.exists():
            shutil.rmtree(previous)
    dist.barrier()


def restore_checkpoint(engine, run, rank):
    import numpy as np
    import random
    import torch
    path = Path(run["output_dir"]) / "resume/latest"
    marker = json.loads((path / "grpo_complete.json").read_text(encoding="utf-8"))
    if marker["reference_identity"] != run["reference"]["identity"]:
        raise ValueError("resume fixed reference identity differs")
    loaded, client = engine.load_checkpoint(str(path / "deepspeed"), tag="state",
                                           load_optimizer_states=True, load_lr_scheduler_states=True)
    state = torch.load(path / f"rng_sampler_rank{rank}.pt", map_location="cpu", weights_only=False)
    if not loaded or client["run_identity"] != run["identity"] or state["run_identity"] != run["identity"]:
        raise ValueError("GRPO model/optimizer/rank-state identities differ")
    if state["step"] != client["step"] or engine.global_steps != state["step"]:
        raise ValueError("GRPO model/optimizer/global-step continuity failed")
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    torch.cuda.set_rng_state_all(state["cuda_rng"])
    return state
