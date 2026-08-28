from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from eaglevl.train.cpt_checkpoint_files import ensure_local_checkpoint_files


class CPTCheckpointFilesTest(unittest.TestCase):
    @staticmethod
    def write_base(base: Path) -> None:
        (base / "config.json").write_text('{"model_type":"locateanything"}\n')
        (base / "configuration_locateanything.py").write_text("BASE_CONFIG = 1\n")
        (base / "modeling_locateanything.py").write_text("BASE_MODEL = 1\n")
        (base / "modeling_qwen2.py").write_text("BASE_QWEN = 1\n")

    def test_missing_config_and_remote_code_are_copied_from_base(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            checkpoint = root / "checkpoint-1033"
            base.mkdir()
            checkpoint.mkdir()
            self.write_base(base)

            report = ensure_local_checkpoint_files(checkpoint, base)

            self.assertEqual(
                set(report["copied"]),
                {
                    "config.json",
                    "configuration_locateanything.py",
                    "modeling_locateanything.py",
                    "modeling_qwen2.py",
                },
            )
            self.assertEqual(
                (checkpoint / "config.json").read_text(),
                (base / "config.json").read_text(),
            )
            self.assertEqual(ensure_local_checkpoint_files(checkpoint, base)["copied"], [])

    def test_existing_checkpoint_files_are_not_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            checkpoint = root / "checkpoint-1033"
            base.mkdir()
            checkpoint.mkdir()
            self.write_base(base)
            (checkpoint / "config.json").write_text("CHECKPOINT_CONFIG\n")
            (checkpoint / "configuration_locateanything.py").write_text(
                "CHECKPOINT_CODE = 1\n"
            )

            report = ensure_local_checkpoint_files(checkpoint, base)

            self.assertNotIn("config.json", report["copied"])
            self.assertNotIn("configuration_locateanything.py", report["copied"])
            self.assertEqual(
                (checkpoint / "config.json").read_text(), "CHECKPOINT_CONFIG\n"
            )
            self.assertEqual(
                (checkpoint / "configuration_locateanything.py").read_text(),
                "CHECKPOINT_CODE = 1\n",
            )
            self.assertTrue((checkpoint / "modeling_locateanything.py").is_file())

    def test_incomplete_base_reports_the_unrepairable_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = root / "base"
            checkpoint = root / "checkpoint-1033"
            base.mkdir()
            checkpoint.mkdir()
            (base / "config.json").write_text("{}\n")

            with self.assertRaisesRegex(
                FileNotFoundError,
                "configuration_locateanything.py.*modeling_locateanything.py",
            ):
                ensure_local_checkpoint_files(checkpoint, base)


if __name__ == "__main__":
    unittest.main()
