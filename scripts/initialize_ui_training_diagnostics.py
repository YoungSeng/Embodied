#!/usr/bin/env python3
"""Create the two diagnostic sheets before checkpoint export/training (CPU only)."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eaglevl.train.ui5_excel_logger import UI5ExcelLogger


def initialize(output_dir):
    workbook = UI5ExcelLogger(Path(output_dir) / "diagnostics/ui5_training_evaluation.xlsx")
    created = workbook.initialize()
    print(f"[UI Excel] {'created/migrated' if created else 'reused'}: {workbook.path}", flush=True)
    print("[UI Excel] train_100steps: 每 100 optimizer step 写真实窗口；eval_1000steps: 按配置写入完整评测。"
          "带表头的空 sheet 不代表已经完成训练或评测。", flush=True)
    return workbook.path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    initialize(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
