from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eaglevl.train.ui5_sampling_coverage import write_sampling_coverage_atomic


def payload(*, seen: int, event: str = "periodic_coverage") -> dict:
    return {
        "schema_version": 1,
        "global_step": 1000,
        "event": event,
        "datasets": [
            {
                "samples_drawn_with_repetition": seen,
                "seen_unique_records": seen,
                "seen_unique_crops": seen,
                "seen_unique_source_images": seen,
                "manual_repair_seen": min(seen, 1),
            }
        ],
    }


class SamplingCoveragePersistenceTest(unittest.TestCase):
    def test_resume_start_uses_separate_filename(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            periodic = root / "sampling_coverage_step_1000.json"
            resume = root / "sampling_coverage_resume_start_step_1000.json"
            self.assertTrue(write_sampling_coverage_atomic(periodic, payload(seen=50)))
            self.assertTrue(
                write_sampling_coverage_atomic(
                    resume, payload(seen=0, event="resume_start")
                )
            )
            self.assertEqual(
                json.loads(periodic.read_text(encoding="utf-8"))["datasets"][0][
                    "seen_unique_records"
                ],
                50,
            )
            self.assertEqual(
                json.loads(resume.read_text(encoding="utf-8"))["event"],
                "resume_start",
            )

    def test_same_step_seen_count_cannot_regress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "sampling_coverage_step_1000.json"
            self.assertTrue(write_sampling_coverage_atomic(path, payload(seen=50)))
            before = path.read_bytes()
            self.assertFalse(write_sampling_coverage_atomic(path, payload(seen=0)))
            self.assertEqual(path.read_bytes(), before)
            self.assertTrue(write_sampling_coverage_atomic(path, payload(seen=75)))
            self.assertEqual(
                json.loads(path.read_text(encoding="utf-8"))["datasets"][0][
                    "seen_unique_records"
                ],
                75,
            )


if __name__ == "__main__":
    unittest.main()
