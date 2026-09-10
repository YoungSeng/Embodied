#!/usr/bin/env python3
"""Prepare and submit a separate neg11 alignment-crops run."""
from ui14_alignment_context import parse_args, run

if __name__ == "__main__":
    run(parse_args(alignment_crops_run=True))
