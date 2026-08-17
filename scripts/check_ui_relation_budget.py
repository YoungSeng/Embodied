#!/usr/bin/env python3
"""Check the actual UI-relation parameter budget in a LocateAnything checkpoint."""

import argparse
import json

from transformers import AutoModel


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_path")
    args = parser.parse_args()
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        device_map="cpu",
    )
    if not hasattr(model, "ui_relation_parameter_report"):
        raise SystemExit("Checkpoint does not expose ui_relation_parameter_report()")
    report = model.ui_relation_parameter_report()
    print(json.dumps(report, indent=2))
    return 0 if report["within_five_percent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
