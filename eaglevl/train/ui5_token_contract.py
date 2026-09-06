"""Validate token identities without silently remapping IDs or model weights."""
import hashlib
import json

TOKEN_FIELDS = {"box_start_token_id": "<box>", "box_end_token_id": "</box>",
                "ref_start_token_id": "<ref>", "ref_end_token_id": "</ref>",
                "none_token_id": "none", "coord_start_token_id": "<0>", "coord_end_token_id": "<1000>"}


def tokenizer_contract(tokenizer, config, processor_tokenizer=None):
    identities, errors = {}, []
    text = getattr(config, "text_config", config)
    tokens = {**TOKEN_FIELDS, "null_token_id": "<null>", "im_end_token_id": "<|im_end|>"}
    for field, token in tokens.items():
        ids = tokenizer.encode(token, add_special_tokens=False)
        owner = text if field in {"null_token_id", "im_end_token_id"} else config
        expected = getattr(owner, "eos_token_id" if field == "im_end_token_id" else field, None)
        permitted = expected if isinstance(expected, list) else [expected]
        identities[field] = {"text": token, "ids": ids, "configured": expected}
        if len(ids) != 1 or ids[0] not in permitted:
            errors.append(f"{field}: tokenizer={ids}, model={expected}")
        if processor_tokenizer is not None and processor_tokenizer.encode(token, add_special_tokens=False) != ids:
            errors.append(f"processor/input tokenizer differs for {field}")
    coords = [tokenizer.encode(f"<{i}>", add_special_tokens=False) for i in range(1001)]
    start = identities["coord_start_token_id"]["ids"]
    if len(start) != 1 or coords != [[start[0] + i] for i in range(1001)]:
        errors.append("coordinate token IDs are not a contiguous <0>..<1000> range")
    negative = tokenizer.encode("<box>none</box>", add_special_tokens=False)
    expected_negative = [identities[key]["ids"][0] for key in
                         ("box_start_token_id", "none_token_id", "box_end_token_id")
                         if identities[key]["ids"]]
    if negative != expected_negative or len(negative) != 3:
        errors.append("negative <box>none</box> supervision is not the expected three tokens")
    if identities["null_token_id"]["ids"] == identities["im_end_token_id"]["ids"]:
        errors.append("null padding and EOS must have distinct token IDs")
    report = {"identities": identities, "negative_ids": negative,
              "tokenizer_eos": tokenizer.eos_token_id, "errors": errors, "valid": not errors}
    report["sha256"] = hashlib.sha256(json.dumps(report, sort_keys=True).encode()).hexdigest()
    return report


def negative_mtp_contract(input_ids, labels_dict, tokenizer, block_size):
    """Inspect the labels produced by the REAL processor + dataset MTP method."""
    original_length = len(input_ids)
    labels = labels_dict["labels"].detach().cpu().tolist()
    negative = tokenizer.encode("<box>none</box>", add_special_tokens=False)
    eos = tokenizer.convert_tokens_to_ids("<|im_end|>")
    null = tokenizer.convert_tokens_to_ids("<null>")
    ar = [token for token in labels[:original_length] if token != -100]
    mtp = labels[original_length + 1:]
    expected = negative + [null] * (block_size - len(negative)) + [eos] + [null] * (block_size - 1)
    errors = []
    if len(negative) != 3 or block_size < 3 or ar != negative + [eos]:
        errors.append("real negative AR labels must be exactly box/none/box_end/EOS (no ref or truncation)")
    if null == eos or mtp != expected or labels[original_length:original_length + 1] != [-100]:
        errors.append("real negative MTP must be box/none/box_end/null padding, then separate EOS/null padding")
    report = {"valid": not errors, "errors": errors, "negative_token_ids": negative,
              "original_sequence_length": original_length, "ar_labels": ar,
              "mtp_blocks": [mtp[i:i + block_size] for i in range(0, len(mtp), block_size)],
              "null_token_id": null, "eos_token_id": eos, "block_size": block_size,
              "input_source": "actual training processor + get_targets_flag_with_mtp"}
    if errors:
        raise ValueError(json.dumps(report, ensure_ascii=False))
    return report


def audit_training_negative_pools(train_dataset, output_dir, rank, recipe_path):
    """Exercise one deterministic real negative per loaded pool without sampler/RNG draws."""
    import random
    from pathlib import Path
    import numpy as np
    import torch
    from eaglevl.train.ui5_supervision import record_boxes, sample_id, validate_answer
    states = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
    reports = []
    try:
        for dataset in train_dataset.datasets:
            for logical_index, raw_index in enumerate(dataset.active_indices):
                record = dataset.lazy_loader[int(raw_index)]
                if record_boxes(record):
                    continue
                validate_answer(record)
                # Same image loading, chat template, tokenization, truncation and MTP
                # as an ordinary training item. Do not invoke the retrying sampler.
                materialized = dataset._materialize_logical_index(logical_index, audit_negative=True)
                reports.append({"pool": dataset.curriculum_pool, "sample_id": sample_id(record),
                                "crop_id": record.get("_ui5_crop_id"), "rank": rank,
                                **materialized["negative_supervision_audit"]})
                print(f"[NEGATIVE MTP PASS] pool={dataset.curriculum_pool} sample_id={sample_id(record)} "
                      f"labels={reports[-1]['ar_labels']} blocks={reports[-1]['mtp_blocks']}", flush=True)
                break
            else:
                raise ValueError(f"no real negative found in loaded dataset {dataset.ds_name}")
    finally:
        random.setstate(states[0])
        np.random.set_state(states[1])
        torch.set_rng_state(states[2])
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    if {r["pool"] for r in reports} != {"hard", "matched_anchor", "global_replay"}:
        raise ValueError("real MTP audit did not cover all three loaded pools")
    recipe = Path(recipe_path)
    target = Path(output_dir) / "diagnostics" / f"negative_mtp_audit_rank{rank}.json"
    target.parent.mkdir(exist_ok=True, parents=True)
    payload = {"recipe_path": str(recipe.resolve()), "recipe_sha256": hashlib.sha256(recipe.read_bytes()).hexdigest(),
               "pools": reports, "sampler_advanced": False, "rng_restored": True}
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)
    return payload
