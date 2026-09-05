#!/usr/bin/env python3
"""Fast CPU guard for every resumed training segment."""
import os
from ui14_profile import validate_prepared_profile

if __name__ == "__main__":
    validate_prepared_profile(os.environ)
