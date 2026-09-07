"""Two-H20 native AR GRPO. One process owns whole original-image groups."""
from __future__ import annotations
import argparse
from collections import Counter
import inspect
import json
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from eaglevl.train.ui5_grpo_core import (
    GRPOConfig, MixedSampler, aligned_slot_count, centered_advantages,
    completion_loss_sums, deepspeed_slot_backward, digest, file_sha,
    lr_multiplier, remaining_budget, stable_seed, zero_parameter_touch,
)
from eaglevl.train.ui5_grpo_checkpoint import restore_checkpoint, save_checkpoint
from scripts.build_ui5_grpo_mixed_manifest import read_json, read_rows, verify_manifest, write_json


def to_device(inputs, device):
    import numpy as np
    import torch
    result = {}
    for key, value in inputs.items():
        if isinstance(value, np.ndarray):
            value = torch.from_numpy(value)
        if isinstance(value, torch.Tensor):
            value = value.to(device, dtype=torch.bfloat16 if key == "pixel_values" else value.dtype)
        result[key] = value
    return result


def prompt_inputs(processor, group, view, config):
    import numpy as np
    import torch
    from PIL import Image
    from scripts.inference_ui_defect_locany import apply_chat_template
    from eaglevl.model.locany.relation_modules import UI_RELATION_PROMPT_SPECS
    spec = next(s for s in UI_RELATION_PROMPT_SPECS if s.task_name == group["task"])
    if file_sha(view["image"]) != view["image_sha256"]:
        raise ValueError(f"mixed image changed: {view['image']}")
    with Image.open(view["image"]) as source:
        image = source.convert("RGB")
        messages = [dict(role="user", content=[dict(type="image", image=image),
                    dict(type="text", text=group["prompt"])])]
        text = apply_chat_template(processor, messages)
        images, videos = processor.process_vision_info(messages)
        encoded = processor(text=[text], images=images, videos=videos,
                            return_tensors="pt", padding=False, truncation=False)
        image.close()
    inputs = {key: encoded[key] for key in ("input_ids", "pixel_values", "image_grid_hws")}
    inputs["attention_mask"] = encoded.get("attention_mask", torch.ones_like(inputs["input_ids"]))
    if not bool(inputs["attention_mask"].all()):
        raise ValueError("single-crop prompt unexpectedly contains padding")
    inputs["position_ids"] = torch.arange(inputs["input_ids"].shape[1]).unsqueeze(0)
    inputs["image_flags"] = torch.tensor([len(inputs["image_grid_hws"])])
    inputs["relation_family"] = torch.tensor([spec.relation_family])
    inputs["defect_type"] = torch.tensor([spec.defect_type])
    for key, value in inputs.items():
        if isinstance(value, np.ndarray):
            inputs[key] = torch.from_numpy(value)
    remaining_budget(inputs["input_ids"].shape[1], config)
    return inputs


def completion_inputs(inputs, tokens, device):
    import torch
    result = to_device(inputs, device)
    start = result["input_ids"].shape[1]
    result["input_ids"] = torch.cat((result["input_ids"], tokens.to(device).unsqueeze(0)), dim=1)
    result["position_ids"] = torch.arange(result["input_ids"].shape[1], device=device).unsqueeze(0)
    result["attention_mask"] = torch.ones_like(result["input_ids"])
    result.update(ui5_ar_mode=True, completion_start=start)
    return result


def collect_group(actor, processor, group, config, device, sampling_key, parser, scorer):
    import torch
    from eaglevl.model.locany.ui5_ar import forward_ar, sample_ar
    from scripts.ui5_grpo_reward import score_trajectory
    completion_lists = [[] for _ in range(config.group_size)]
    inputs_by_view = []
    with torch.no_grad():
        for view in group["views"]:
            cpu_inputs = prompt_inputs(processor, group, view, config)
            inputs_by_view.append(cpu_inputs)
            inputs = to_device(cpu_inputs, device)
            prefix = forward_ar(actor, **inputs, use_cache=True)
            for trajectory in range(config.group_size):
                seed = stable_seed(config.seed, sampling_key, group["group_id"], trajectory, view["crop_id"])
                result = sample_ar(actor, inputs, generation_mode="slow", prefix=prefix,
                                   max_new_tokens=remaining_budget(inputs["input_ids"].shape[1], config),
                                   eos_token_id=processor.tokenizer.convert_tokens_to_ids("<|im_end|>"),
                                   seed=seed, temperature=config.temperature)
                result["raw_output"] = processor.tokenizer.decode(result["tokens"], skip_special_tokens=False)
                completion_lists[trajectory].append(result)
            del prefix, inputs
    rewards = [score_trajectory(group, outputs, parser, scorer, config) for outputs in completion_lists]
    advantages = centered_advantages([r["reward"] for r in rewards])
    tokens = sum(c["tokens"].numel() for trajectory in completion_lists for c in trajectory)
    return dict(group=group, inputs=inputs_by_view, completions=completion_lists,
                rewards=rewards, advantages=advantages, token_count=tokens)


class Replay:
    """The existing task/source/label rotating sampler on ALL original pools."""
    def __init__(self, directory, processor, config, negative_ratio):
        from eaglevl.train.ui_defect_data import build_task_source_balanced_rotating_plan
        from eaglevl.train.locany_finetune_magi_stream import LazySupervisedDatasetMTP
        self.records = read_rows(directory / "replay.jsonl")
        self.plan = build_task_source_balanced_rotating_plan(self.records, negative_ratio)
        self.config = config
        self.epoch, self.indices = None, None
        self.encoder = LazySupervisedDatasetMTP.__new__(LazySupervisedDatasetMTP)
        self.encoder.processor = processor
        self.encoder.block_size = int(processor.tokenizer._ui5_block_size)
        self.encoder.ds_name = "ui5_grpo_full_source_replay"
        self.encoder.data_augment = False

    def at(self, ordinal):
        import torch
        from eaglevl.train.ui_defect_data import materialize_task_source_balanced_rotating_indices, extract_ui_defect_targets
        from eaglevl.train.tools import process_multimodal_sample
        from eaglevl.train.ui5_supervision import validate_answer
        epoch, position = divmod(ordinal, self.plan["epoch_length"])
        if epoch != self.epoch:
            self.indices = materialize_task_source_balanced_rotating_indices(self.plan,
                           seed=self.config.seed, epoch_index=epoch)
            self.epoch = epoch
        record = self.records[self.indices[position]]
        validate_answer(record)
        if file_sha(record["image"]) != record["_ui5_grpo_image_sha256"]:
            raise ValueError(f"replay image changed: {record['image']}")
        messages = process_multimodal_sample(record, "", 16, 2, 32000 * 28 * 28 * 0.9, visual_prompt=False)
        data = self.encoder.multi_modal_get_item(messages, ui_targets=extract_ui_defect_targets(record), truncation=False)
        length = data["input_ids"].numel()
        if length > self.config.max_num_tokens:
            raise ValueError(f"complete AR/MTP replay exceeds 7268: {record.get('_ui5_sample_id')} length={length}")
        for key in ("input_ids", "labels", "attention_mask", "position_ids"):
            data[key] = data[key].unsqueeze(0)
        return data, dict(sample_id=record.get("_ui5_sample_id"), task=record.get("_ui5_task"),
                          crop_id=record.get("_ui5_crop_id"), record_index=self.indices[position])


def optimizer_groups(model, run):
    from transformers.trainer_pt_utils import get_parameter_names
    from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
    decay = {name for name in get_parameter_names(model, ALL_LAYERNORM_LAYERS) if "bias" not in name}
    scales = run["optimizer"]["lr_scale"]
    buckets = {}
    report = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        component = "llm" if name.startswith("language_model.") else "mlp"
        scale = scales.get(component, 1.0)
        wd = run["optimizer"]["weight_decay"] if name in decay else 0.0
        buckets.setdefault((scale, wd), []).append(parameter)
        report[name] = dict(lr_multiplier=scale, weight_decay=wd, numel=parameter.numel())
    return [dict(params=params, lr=run["grpo"]["learning_rate"] * scale, weight_decay=wd)
            for (scale, wd), params in buckets.items()], report


def load_native(path, *, trainable):
    import torch
    from eaglevl.model.locany.modeling_locateanything import LocateAnythingForConditionalGeneration
    from eaglevl.model.locany.configuration_locateanything import LocateAnythingConfig
    config = LocateAnythingConfig.from_pretrained(path)
    config._attn_implementation = "sdpa"
    config.text_config._attn_implementation = "sdpa"
    model, information = LocateAnythingForConditionalGeneration.from_pretrained(
        path, config=config, torch_dtype=torch.bfloat16, attn_implementation="sdpa", output_loading_info=True)
    missing = information.get("missing_keys", [])
    if missing or information.get("mismatched_keys") or information.get("unexpected_keys"):
        raise ValueError(f"initial/resume checkpoint must load all original weights exactly: {information}")
    if not model.enable_ui_relation:
        raise ValueError("initial checkpoint lacks pretrained detail/relation/PBD")
    if model.language_model.model._attn_implementation != "sdpa":
        raise ValueError("formal LLM attention must be SDPA")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(trainable and not name.startswith("vision_model."))
    model.eval()
    if trainable:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.language_model.model.ui5_eval_gradient_checkpointing = True
    return model


def run_diagnostic(actor, processor, groups, run, config, device, step, rank, parser, scorer):
    import torch
    import torch.distributed as dist
    rows = []
    started = time.monotonic()
    # RNG, sampler and weights do not move when the fixed train diagnosis runs.
    with torch.random.fork_rng(devices=[device.index]):
        for group in groups[rank::config.world_size]:
            result = collect_group(actor, processor, group, config, device, "fixed_train_diagnostic", parser, scorer)
            count = sum(r["exact_correct"] for r in result["rewards"])
            rows.append(dict(step=step, scope="fixed mixed TRAIN diagnostic; generation_mode=slow",
                             group_id=group["group_id"], task=group["task"],
                             source_hybrid_correct_count=group["crop_correct_count"],
                             ar_correct_count=count, invalid=sum(r["invalid"] for r in result["rewards"])))
    gathered = [None] * config.world_size
    dist.all_gather_object(gathered, rows)
    if rank == 0:
        rows = sorted([row for part in gathered for row in part], key=lambda r: r["group_id"])
        path = Path(run["output_dir"]) / "diagnostics" / f"train_ar_step{step:06d}.json"
        if step:
            baseline = read_json(path.with_name("train_ar_step000000.json"))["rows"]
            lookup = {r["group_id"]: r["ar_correct_count"] for r in baseline}
            for row in rows:
                row["ar_step0_correct_count"] = lookup[row["group_id"]]
                row["transition"] = f"{lookup[row['group_id']]}/4 -> {row['ar_correct_count']}/4"
        write_json(path, dict(step=step, rows=rows, seconds=time.monotonic() - started,
                             run_identity=run["identity"]))
    dist.barrier()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-config", type=Path, required=True)
    parser.add_argument("--until-step", type=int, required=True)
    args = parser.parse_args()
    import numpy as np
    import torch
    import torch.distributed as dist
    import deepspeed
    from transformers import AutoProcessor
    from eaglevl.train.ui5_token_contract import tokenizer_contract
    from scripts.ui5_grpo_reward import formal_modules
    from scripts.ui5_grpo_artifacts import refresh_workbook
    run = read_json(args.run_config)
    payload = dict(run)
    if payload.pop("identity") != digest(payload):
        raise ValueError("run config identity mismatch")
    config = GRPOConfig(**run["grpo"]).validate()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    deepspeed.init_distributed(dist_backend="nccl")
    rank = dist.get_rank()
    if dist.get_world_size() != 2 or "H20" not in torch.cuda.get_device_name(device):
        raise ValueError("this formal launcher requires two H20 ranks")
    if "scale_wrt_gas" not in inspect.signature(deepspeed.DeepSpeedEngine.backward).parameters:
        raise RuntimeError("inherited DeepSpeed lacks the required scale_wrt_gas public API")
    torch.manual_seed(config.seed + rank)
    random.seed(config.seed + rank)
    np.random.seed(config.seed + rank)
    directory = Path(run["mixed_dir"])
    manifest = verify_manifest(directory)
    if manifest["identity"] != run["mixed_identity"]:
        raise ValueError("mixed identity differs from formal config")
    for name, expected in run["reference"]["files"].items():
        if file_sha(Path(run["reference"]["path"]) / name) != expected:
            raise ValueError("fixed reference checkpoint changed")
    groups = read_rows(directory / "groups.jsonl")
    sampler = MixedSampler(groups, config.seed)
    checkpoint = Path(run["output_dir"]) / "resume/latest"
    resume = checkpoint.exists()
    actor = load_native(checkpoint if resume else run["initial_model"], trainable=True)
    reference = load_native(run["reference"]["path"], trainable=False)
    processor = AutoProcessor.from_pretrained(run["processor_path"], trust_remote_code=True,
                                             use_fast=True, local_files_only=True)
    processor.tokenizer.model_max_length = config.max_seq_length
    processor.in_token_limit = run["processor_in_token_limit"]
    processor.tokenizer._ui5_block_size = actor.config.text_config.block_size
    audit = tokenizer_contract(processor.tokenizer, actor.config)
    if not audit["valid"]:
        raise ValueError(f"native processor/tokenizer contract failed: {audit}")
    replay = Replay(directory, processor, config, run["optimizer"]["replay_negative_to_positive_ratio"])
    param_groups, param_report = optimizer_groups(actor, run)
    optimizer = torch.optim.AdamW(param_groups, lr=config.learning_rate,
                                  betas=tuple(run["optimizer"]["betas"]), eps=run["optimizer"]["eps"])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: lr_multiplier(step, config))
    ds_config = read_json(Path(run["project_root"]) / "configs/ui5_grpo_zero2.json")
    engine, _, _, _ = deepspeed.initialize(model=actor, optimizer=optimizer,
                                          lr_scheduler=scheduler, config=ds_config)
    engine.eval()
    actor = engine.module
    start, sampling_counts = 0, Counter()
    if resume:
        state = restore_checkpoint(engine, run, rank)
        start = state["step"]
        sampling_counts.update(state["sampling_counts"])
        if sum(sampling_counts.values()) != start * config.groups_per_rank:
            raise ValueError("resume sampler counts do not match completed original-image groups")
        if state["sampler_identity"] != sampler.identity or state["next_draw"] != start * 4 + rank:
            raise ValueError("group sampler resume position differs")
    if not start <= args.until_step <= config.total_steps:
        raise ValueError("invalid segment optimizer-step range")
    parser_module, scorer = formal_modules()
    output = Path(run["output_dir"])
    if rank == 0:
        write_json(output / "diagnostics/optimizer_groups.json", param_report)
        write_json(output / "diagnostics/token_contract.json", audit)
    dist.barrier()
    diagnostic = read_rows(directory / "train_diagnostic.jsonl")
    if not (output / "diagnostics/train_ar_step000000.json").is_file():
        if start:
            raise ValueError("resume lacks fixed step-0 AR train diagnostic")
        run_diagnostic(actor, processor, diagnostic, run, config, device, 0, rank, parser_module, scorer)
    window = []
    for step in range(start + 1, args.until_step + 1):
        torch.cuda.reset_peak_memory_stats(device)
        started = time.monotonic()
        collected, replays, replay_ids = [], [], []
        # Two rounds, one whole group per rank/round. No weight update inside
        # either G=4 sampling phase or between the two accumulation rounds.
        for round_index in range(config.groups_per_rank):
            ordinal = (step - 1) * 4 + round_index * 2 + rank
            group = groups[sampler.index_at(ordinal)]
            sampling_counts[f"{group['task']}/{group['polarity']}/{group['crop_correct_count']}"] += 1
            collected.append(collect_group(actor, processor, group, config, device,
                                            ("train", ordinal), parser_module, scorer))
            data, replay_identity = replay.at(ordinal)
            replays.append(data)
            replay_ids.append(replay_identity)
        generation_seconds = time.monotonic() - started
        reference_started = time.monotonic()
        reference.to(device)
        with torch.no_grad():
            for group in collected:
                for trajectory in group["completions"]:
                    for inputs, completion in zip(group["inputs"], trajectory):
                        current_inputs = completion_inputs(inputs, completion["tokens"], device)
                        completion["reference_log_probs"] = reference(
                            **current_inputs, temperature=config.temperature).completion_log_probs.detach().cpu()
                        del current_inputs
        reference.cpu()
        torch.cuda.empty_cache()
        reference_seconds = time.monotonic() - reference_started
        local_slots = [(g, inputs, c, advantage) for g in collected
                       for trajectory, advantage in zip(g["completions"], g["advantages"])
                       for inputs, c in zip(g["inputs"], trajectory)]
        slot_count = aligned_slot_count(len(local_slots) + len(replays), device)
        metrics = dict(policy_loss=0.0, kl_loss=0.0, replay_loss=0.0, max_logprob_error=0.0)
        backward_started = time.monotonic()
        for slot in range(slot_count):
            if slot < len(local_slots):
                group, inputs, completion, advantage = local_slots[slot]
                current_inputs = completion_inputs(inputs, completion["tokens"], device)
                current = engine(**current_inputs, temperature=config.temperature).completion_log_probs
                old = completion["old_log_probs"].to(device)
                error = (current.detach() - old).abs()
                maximum_error = error.max().item()
                metrics["max_logprob_error"] = max(metrics["max_logprob_error"], maximum_error)
                # Verified every on-policy completion, before any optimizer update.
                if maximum_error > 0.08 or error.mean().item() > 0.01:
                    raise ValueError(f"AR sample/recompute log-prob mismatch: max={maximum_error}, mean={error.mean().item()}")
                policy, kl = completion_loss_sums(current, old, completion["reference_log_probs"].to(device),
                                                  advantage, config.clip)
                denominator = group["token_count"] * config.groups_per_rank
                loss = (policy + config.kl_beta * kl) / denominator
                metrics["policy_loss"] += policy.detach().item() / denominator
                metrics["kl_loss"] += kl.detach().item() / denominator
                loss = loss + zero_parameter_touch(actor)
            elif slot < len(local_slots) + len(replays):
                data = to_device(replays[slot - len(local_slots)], device)
                replay_loss = engine(**data).loss
                metrics["replay_loss"] += replay_loss.detach().item() / config.groups_per_rank
                loss = config.replay_weight * replay_loss / config.groups_per_rank + zero_parameter_touch(actor)
            else:
                loss = engine(pixel_values=None, ui5_zero_slot=True).loss
            deepspeed_slot_backward(engine, loss, final_slot=slot + 1 == slot_count)
            del loss
            if slot < len(local_slots):
                del current, current_inputs, policy, kl
        torch.cuda.synchronize(device)
        if engine.global_steps != step:
            raise ValueError(f"ZeRO optimizer boundary mismatch: {engine.global_steps} != {step}")
        all_rewards = [r for group in collected for r in group["rewards"]]
        metrics.update({name: sum(r[name] for r in all_rewards) / len(all_rewards)
                        for name in ("reward", "F_box", "C_img", "S_iou")})
        metrics.update(original_groups=2, original_trajectories=8,
                       valid_trajectories=sum(not r["invalid"] for r in all_rewards),
                       valid_groups=sum(not any(r["invalid"] for r in g["rewards"]) for g in collected),
                       invalid_trajectories=sum(r["invalid"] for r in all_rewards),
                       nonzero_advantage_groups=sum(any(a != 0 for a in g["advantages"]) for g in collected),
                       completion_tokens=sum(g["token_count"] for g in collected), slots=slot_count,
                       generation_seconds=generation_seconds, reference_seconds=reference_seconds,
                       backward_seconds=time.monotonic() - backward_started,
                       step_seconds=time.monotonic() - started,
                       peak_allocated_gb=torch.cuda.max_memory_allocated(device) / 2**30,
                       peak_reserved_gb=torch.cuda.max_memory_reserved(device) / 2**30)
        summaries = [None] * config.world_size
        dist.all_gather_object(summaries, metrics)
        # Persist true sampled token/old log-prob evidence; never retain pixels
        # or computation graphs in the trajectory archive.
        trace = []
        for group in collected:
            trace.append(dict(group_id=group["group"]["group_id"], rewards=group["rewards"],
                              advantages=group["advantages"], completions=group["completions"],
                              views=[dict(image=v["image"], crop_id=v["crop_id"],
                                         prompt_ids=x["input_ids"], prompt_positions=x["position_ids"],
                                         prompt_mask=x["attention_mask"])
                                     for v, x in zip(group["group"]["views"], group["inputs"])]))
        trace_dir = output / "trajectories" / f"step{step:06d}"
        trace_dir.mkdir(parents=True, exist_ok=True)
        torch.save(dict(step=step, run_identity=run["identity"], groups=trace, replay=replay_ids), trace_dir / f"rank{rank}.pt")
        if rank == 0:
            totals = {name for name in metrics if name.endswith("groups") or name.endswith("trajectories")}
            totals.add("completion_tokens")
            maxima = {name for name in metrics if name.endswith("seconds") or name.startswith("peak_") or name in {"max_logprob_error", "slots"}}
            merged = {name: (sum(s[name] for s in summaries) if name in totals else
                             max(s[name] for s in summaries) if name in maxima else
                             sum(s[name] for s in summaries) / len(summaries)) for name in metrics}
            merged.update(step=step, learning_rate=config.learning_rate * lr_multiplier(step, config),
                          loss_total=merged["policy_loss"] + config.kl_beta * merged["kl_loss"] + config.replay_weight * merged["replay_loss"],
                          nonzero_advantage_group_ratio=merged["nonzero_advantage_groups"] / merged["original_groups"],
                          invalid_rate=merged["invalid_trajectories"] / merged["original_trajectories"],
                          trajectories_per_second=merged["original_trajectories"] / merged["step_seconds"])
            window.append(merged)
            if step % config.log_interval == 0:
                report = {key: sum(row[key] for row in window) / len(window) for key in merged}
                report["step"] = step
                report["window_steps"] = len(window)
                for key in totals:
                    report[key] = sum(row[key] for row in window)
                for key in ("max_logprob_error", "peak_allocated_gb", "peak_reserved_gb", "slots"):
                    report[key] = max(row[key] for row in window)
                write_json(output / "metrics" / f"step{step:06d}.json", report)
                print("[GRPO] " + json.dumps(report, sort_keys=True), flush=True)
                window.clear()
        if step % config.excel_interval == 0:
            counts_by_rank = [None] * config.world_size
            dist.all_gather_object(counts_by_rank, dict(sampling_counts))
            if rank == 0:
                actual = Counter()
                for counts in counts_by_rank:
                    actual.update(counts)
                if sum(actual.values()) != step * config.world_size * config.groups_per_rank:
                    raise ValueError("actual group sampling count differs from optimizer steps")
                write_json(output / "diagnostics" / f"sampling_step{step:06d}.json",
                           dict(step=step, counts=dict(actual), total_draws=sum(actual.values()),
                                run_identity=run["identity"]))
                refresh_workbook(run)
        if step == args.until_step:
            run_diagnostic(actor, processor, diagnostic, run, config, device, step, rank, parser_module, scorer)
        if step % config.save_interval == 0 or step == args.until_step:
            save_checkpoint(engine, run, step, dict(next_draw=step * 4 + rank,
                            sampler_identity=sampler.identity, sampling_counts=dict(sampling_counts)))
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
