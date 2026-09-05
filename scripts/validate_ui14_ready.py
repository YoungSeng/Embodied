#!/usr/bin/env python3
"""Fast CPU guard for every resumed training segment."""
import os
from ui14_profile import validate_prepared_profile, validate_run_data_binding
from ui14_common import read_json
from pathlib import Path

if __name__ == "__main__":
    validate_prepared_profile(os.environ)
    validate_run_data_binding(os.environ, read_json(Path(os.environ["UI14_DATA_ROOT"]) / "source_snapshot.json"), create=True)
