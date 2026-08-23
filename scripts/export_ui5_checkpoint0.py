#!/usr/bin/env python3
"""Export a full, deterministically initialized UI5 checkpoint-0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from eaglevl.model.locany.configuration_locateanything import LocateAnythingConfig
from eaglevl.model.locany.modeling_locateanything import (
    LocateAnythingForConditionalGeneration,
)
from eaglevl.model.locany.ui_relation_setup import (
    configure_ui5_model_config,
    initialize_or_validate_ui_relation,
)
from eaglevl.train.constants import (
    BOX_END_TOKEN,
    BOX_START_TOKEN,
    IMG_CONTEXT_TOKEN,
    NULL_TOKEN,
    REF_END_TOKEN,
    REF_START_TOKEN,
    TEXT_MASK_TOKEN,
    number_tokens_list,
    special_tokens_list,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--block-size", type=int, default=6)
    parser.add_argument("--attn-implementation", required=True)
    parser.add_argument("--relation-detail-hidden-size", type=int, default=256)
    parser.add_argument("--relation-num-slots", type=int, default=8)
    parser.add_argument("--relation-adapter-bottleneck", type=int, default=64)
    parser.add_argument("--relation-gate-loss-weight", type=float, default=1.0)
    parser.add_argument("--relation-slot-gate-loss-weight", type=float, default=0.1)
    parser.add_argument("--relation-attention-loss-weight", type=float, default=0.1)
    parser.add_argument("--relation-gate-threshold", type=float, default=0.5)
    parser.add_argument("--relation-focal-beta", type=float, default=0.999)
    parser.add_argument("--relation-focal-gamma", type=float, default=2.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base = args.base_model.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not base.is_dir():
        raise FileNotFoundError(base)
    torch.manual_seed(args.seed)
    complete_marker = output / "ui5_checkpoint0_manifest.json"
    if complete_marker.is_file() and (output / "config.json").is_file():
        print(f"[CHECKPOINT-0] reuse complete export: {output}")
        return 0

    tokenizer = AutoTokenizer.from_pretrained(
        base, add_eos_token=False, trust_remote_code=True, use_fast=False
    )
    num_new_tokens = tokenizer.add_tokens(
        special_tokens_list + number_tokens_list, special_tokens=True
    )
    if len(tokenizer.encode("assistant")) > 1:
        tokenizer.add_tokens(["assistant"], special_tokens=False)
        num_new_tokens += 1

    config = LocateAnythingConfig.from_pretrained(base)
    none_ids = tokenizer.encode("none", add_special_tokens=False)
    configure_ui5_model_config(
        config,
        attn_implementation=args.attn_implementation,
        image_token_index=tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN),
        block_size=args.block_size,
        causal_attn=False,
        text_mask_token_id=tokenizer.convert_tokens_to_ids(TEXT_MASK_TOKEN),
        null_token_id=tokenizer.convert_tokens_to_ids(NULL_TOKEN),
        box_start_token_id=tokenizer.convert_tokens_to_ids(BOX_START_TOKEN),
        box_end_token_id=tokenizer.convert_tokens_to_ids(BOX_END_TOKEN),
        coord_start_token_id=tokenizer.convert_tokens_to_ids(number_tokens_list[0]),
        coord_end_token_id=tokenizer.convert_tokens_to_ids(number_tokens_list[-1]),
        ref_start_token_id=tokenizer.convert_tokens_to_ids(REF_START_TOKEN),
        ref_end_token_id=tokenizer.convert_tokens_to_ids(REF_END_TOKEN),
        none_token_id=none_ids[0] if len(none_ids) == 1 else 4064,
        enable_ui_relation=True,
        relation_detail_hidden_size=args.relation_detail_hidden_size,
        relation_num_slots=args.relation_num_slots,
        relation_adapter_bottleneck=args.relation_adapter_bottleneck,
        relation_detail_layers=[5, 15, 26],
        relation_gate_loss_weight=args.relation_gate_loss_weight,
        relation_slot_gate_loss_weight=args.relation_slot_gate_loss_weight,
        relation_attention_loss_weight=args.relation_attention_loss_weight,
        relation_gate_threshold=args.relation_gate_threshold,
        relation_focal_beta=args.relation_focal_beta,
        relation_focal_gamma=args.relation_focal_gamma,
    )
    config.relation_gate_thresholds = {}
    config.ui_relation_initialization_seed = args.seed
    config.ui_relation_initialization_reason = "checkpoint-0-export"

    model, loading_info = LocateAnythingForConditionalGeneration.from_pretrained(
        base,
        config=config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        output_loading_info=True,
    )
    ui_load_report = initialize_or_validate_ui_relation(
        model,
        loading_info,
        seed=args.seed,
        all_missing_reason="all-ui-relation-keys-missing-checkpoint-0-export",
    )
    if ui_load_report["state"] != "all_missing":
        raise RuntimeError(
            "checkpoint-0 must be exported from a base checkpoint with every UI key missing; "
            f"state={ui_load_report['state']}"
        )
    init_report = ui_load_report["initialization"]
    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        embeddings = model.language_model.get_output_embeddings().weight.data
        embeddings[-num_new_tokens:] = embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        model.config.text_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output, safe_serialization=True, max_shard_size="5GB")
    tokenizer.save_pretrained(output)
    complete_marker.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "base_model": str(base),
                "checkpoint": str(output),
                "initialization": init_report,
                "num_new_tokens": num_new_tokens,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[CHECKPOINT-0] exported full UI model: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
