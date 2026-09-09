"""CPU-only UI answer grammar shared by generation and diagnostic audits.

The task prompt supplies the reference label; it never supplies GT counts or
boxes. Scorers keep their existing parser and penalties.
"""
from __future__ import annotations
import re

VERSION = "ui14_answer_v1"


def decode_contract(policy="legacy"):
    """Bind prediction reuse to the code actually used, not checkpoint code."""
    import hashlib
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    names = ("eaglevl/ui_answer_grammar.py", "eaglevl/utils/locany/generate_utils.py",
             "eaglevl/utils/locany/modeling_locateanything.py", "scripts/inference_ui_defect_locany.py")
    return {"policy": policy, "reference_label": "task_registry.prompt_label",
            "code_sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in names}}


def answer_state(tokens, token_ids, label_ids, max_boxes=8):
    """Validate a generated prefix and describe its next token/state.

    Positive: (<ref>task label</ref><box><x1><y1><x2><y2></box>)+
    Negative: <box>none</box>. EOS is permitted only after a whole answer.
    Only the positive six-token box frame may use parallel decoding, after AR
    has emitted its box anchor (the existing PBD input convention).
    """
    if not label_ids:
        raise ValueError("Structured UI decoding requires the task's reference label")
    ids = token_ids
    phase, label_pos, coords, boxes = "start", 0, 0, 0
    first, last = int(ids["coord_start_token_id"]), int(ids["coord_end_token_id"])

    def allowed():
        if phase == "start": return (int(ids["ref_start_token_id"]), int(ids["box_start_token_id"]))
        if phase == "label": return (int(label_ids[label_pos]),)
        if phase == "ref_end": return (int(ids["ref_end_token_id"]),)
        if phase == "box_start": return (int(ids["box_start_token_id"]),)
        if phase == "coords": return range(first, last + 1)
        if phase in ("box_end", "none_end"): return (int(ids["box_end_token_id"]),)
        if phase == "none": return (int(ids["none_token_id"]),)
        if phase == "between" and boxes < max_boxes:
            return (int(ids["ref_start_token_id"]), int(ids["im_end_token_id"]))
        if phase in ("between", "end"): return (int(ids["im_end_token_id"]),)
        return ()

    for index, raw in enumerate(tokens):
        token = int(raw)
        if token not in allowed():
            raise ValueError(f"Invalid UI answer prefix at token {index}: phase={phase}, token={token}")
        if phase == "start":
            phase = "label" if token == int(ids["ref_start_token_id"]) else "none"
        elif phase == "label":
            label_pos += 1
            if label_pos == len(label_ids): phase = "ref_end"
        elif phase == "ref_end": phase = "box_start"
        elif phase == "box_start": phase, coords = "coords", 0
        elif phase == "coords":
            coords += 1
            if coords == 4: phase = "box_end"
        elif phase == "box_end": boxes, phase = boxes + 1, "between"
        elif phase == "none": phase = "none_end"
        elif phase == "none_end": phase = "end"
        elif phase == "between" and token == int(ids["ref_start_token_id"]):
            phase, label_pos = "label", 0
        else: phase = "finished"
    return {"phase": phase, "allowed": allowed(), "boxes": boxes,
            "can_mtp": phase == "coords" and coords == 0, "complete": phase in ("between", "end", "finished")}


_REF = r"<ref>[^<>]+</ref>"
_COORD = r"<\s*-?\d+\s*>"
_BOX = r"<box>\s*" + (r"\s*".join([_COORD] * 4)) + r"\s*</box>"
_POSITIVE = re.compile(r"(?:\s*(?:" + _REF + r")?\s*" + _BOX + r")+\s*", re.I)
_NEGATIVE = re.compile(r"\s*(?:" + _REF + r")?\s*<box>\s*none\s*</box>\s*", re.I)


def raw_format_issue(answer):
    """Classify raw syntax only. Never turn an illegal prediction into none.

    Optional refs reflect the historical inference parser's accepted box-only
    output. Audit anomalies and the original scorer's invalid flag are recorded
    separately, because the scorer may accept a valid box inside bad text.
    """
    if answer is None: return "missing_raw_evidence"
    text = str(answer).strip()
    for suffix in ("<|im_end|>", "<|endoftext|>"):
        if text.endswith(suffix): text = text[:-len(suffix)].rstrip()
    if not text: return "empty_output"
    if re.search(r"\bnone\b", text, re.I) and re.search(_COORD, text):
        return "none_coordinate_mix"
    if _NEGATIVE.fullmatch(text) or _POSITIVE.fullmatch(text): return None
    # A balanced ref/box envelope with bad coordinate syntax is distinct from
    # malformed tags or a broken ref continuation.
    without_refs = re.sub(_REF, "", text, flags=re.I).strip()
    if (re.fullmatch(r"(?:\s*<box>[^<>]*(?:<[^<>]*>[^<>]*)*</box>\s*)+", without_refs, re.I)
            and without_refs.lower().count("<box>") == without_refs.lower().count("</box>")):
        return "bbox_parse_failure"
    if "<ref" in text.lower() or "</ref" in text.lower() or "<box" not in text.lower():
        return "illegal_tag_structure"
    return "bbox_parse_failure"
