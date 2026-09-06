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
