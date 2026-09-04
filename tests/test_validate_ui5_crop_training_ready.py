import contextlib
import io
from pathlib import Path
import sys
import unittest
from unittest import mock


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import validate_ui5_crop_training_ready as validator


def crop_only_ready_payload() -> dict:
    return {
        "training_ready": True,
        "crop_train_mode": "crop_only",
        "crop_only_region_records": 100,
        "active_crop_retention_policy": "100%",
        "crop_only_negative_records": 20,
        "crop_only_local_task_full_image_records": 0,
        "crop_only_content_missing_global_records": 10,
        "crop_only_positive_negative_by_task": {},
    }


class ValidateExpectedTrainModeTest(unittest.TestCase):
    def test_crop_only_accepts_the_audited_crop_only_recipe(self):
        with mock.patch.object(
            validator,
            "validate_training_ready_marker",
            return_value=crop_only_ready_payload(),
        ) as validate_marker, contextlib.redirect_stdout(io.StringIO()):
            result = validator.main(
                [
                    "--audit-dir",
                    "/audit",
                    "--recipe",
                    "/audit/training_recipes/crop_only.json",
                    "--expected-train-mode",
                    "crop_only",
                ]
            )

        self.assertEqual(result, 0)
        validate_marker.assert_called_once_with(
            Path("/audit"),
            recipe_path=Path("/audit/training_recipes/crop_only.json"),
        )

    def test_crop_only_rejects_other_audited_recipe_modes(self):
        for audited_mode in ("full_only", "full_plus_crop"):
            with self.subTest(audited_mode=audited_mode), mock.patch.object(
                validator,
                "validate_training_ready_marker",
                return_value={"training_ready": True, "crop_train_mode": audited_mode},
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    rf"expected 'crop_only', got '{audited_mode}'",
                ):
                    validator.main(
                        [
                            "--audit-dir",
                            "/audit",
                            "--recipe",
                            f"/audit/training_recipes/{audited_mode}.json",
                            "--expected-train-mode",
                            "crop_only",
                        ]
                    )

    def test_expected_train_mode_is_required(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as raised:
                validator.parse_args(["--audit-dir", "/audit"])
        self.assertEqual(raised.exception.code, 2)

    def test_training_shell_binds_validator_to_runtime_train_mode(self):
        shell = (PROJECT_ROOT / "shell" / "train_locany_ui_defect.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn(
            '--expected-train-mode "${UI5_CROP_TRAIN_MODE}"',
            shell,
        )


if __name__ == "__main__":
    unittest.main()
