"""Wall-clock scheduling primitives with no Trainer dependency."""

from __future__ import annotations

import math


class PeriodicCheckpointSchedule:
    """Trigger a checkpoint after each complete wall-clock interval."""

    def __init__(self, interval_hours: float):
        interval_hours = float(interval_hours)
        if not math.isfinite(interval_hours) or interval_hours <= 0.0:
            raise ValueError("interval_hours must be finite and positive")
        self.interval_seconds = interval_hours * 3600.0
        self.last_save_time: float | None = None

    def start(self, now: float) -> None:
        self.last_save_time = float(now)

    def is_due(self, now: float) -> bool:
        if self.last_save_time is None:
            self.start(now)
            return False
        return float(now) - self.last_save_time >= self.interval_seconds

    def mark_saved(self, now: float) -> None:
        self.last_save_time = float(now)
