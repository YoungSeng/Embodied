from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "build_ui5_curriculum_recipe.py"
SPEC = importlib.util.spec_from_file_location("build_ui5_curriculum_recipe", SCRIPT)
assert SPEC and SPEC.loader
curriculum_recipe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(curriculum_recipe)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )


class CurriculumRecipeTests(unittest.TestCase):
    def _fixture(self, root: Path) -> tuple[Path, Path]:
        bundle = root / "bundle"
        (bundle / "images").mkdir(parents=True)
        groups = [
            ("hard-pos", "occlusion", True, 0),
            ("hard-neg", "text_overflow", False, 0),
            ("anchor-pos", "occlusion", True, 4),
            ("anchor-neg", "text_overflow", False, 4),
            ("replay", "cropping", True, 2),
        ]
        source_rows = []
        difficulty_rows = []
        unique_rows = []
        for sample_id, task, positive, correct in groups:
            image = bundle / "images" / f"{sample_id}.png"
            image.write_bytes(b"fake-image")
            unique_rows.append(
                {
                    "image_id": f"image-{sample_id}",
                    "image_relpath": f"images/{sample_id}.png",
                    "sha256": curriculum_recipe._sha256_file(image),
                }
            )
            answer = "<box><1><1><2><2></box>" if positive else "<box>none</box>"
            source_rows.append(
                {
                    "source_record_id": f"source-{sample_id}",
                    "sample_id": sample_id,
                    "image_id": f"image-{sample_id}",
                    "task": task,
                    "gt_boxes_global_xyxy": [[1, 1, 2, 2]] if positive else [],
                    "portable_training_record": {
                        "image": f"images/{sample_id}.png",
                        "conversations": [
                            {"from": "human", "value": "locate"},
                            {"from": "gpt", "value": answer},
                        ],
                    },
                }
            )
            difficulty_rows.append(
                {
                    "record_id": sample_id,
                    "sample_id": sample_id,
                    "source_image_id": f"image-{sample_id}",
                    "task": task,
                    "image_relpath": f"images/{sample_id}.png",
                    "prompt": "locate",
                    "gt_global": [[1, 1, 2, 2]] if positive else [],
                    "crop_correct_count": correct,
                    "crop_complete4": True,
                    "technical_error_free": True,
                    "runtime_error_count": 0,
                    "parse_error_count": 0,
                }
            )
        source_records = bundle / "manifest" / "source_records.jsonl"
        unique_images = bundle / "manifest" / "unique_images.jsonl"
        write_jsonl(source_records, source_rows)
        write_jsonl(unique_images, unique_rows)
        files = {}
        for path in (source_records, unique_images):
            relative = path.relative_to(bundle).as_posix()
            files[relative] = {
                "bytes": path.stat().st_size,
                "sha256": curriculum_recipe._sha256_file(path),
            }
        (bundle / "bundle_manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "complete": True,
                    "unique_images": len(unique_rows),
                    "files": files,
                }
            ),
            encoding="utf-8",
        )
        difficulty = root / "sample_difficulty.jsonl"
        write_jsonl(difficulty, difficulty_rows)
        return bundle, difficulty

    @staticmethod
    def _args(bundle: Path, difficulty: Path, output: Path) -> Namespace:
        return Namespace(
            base_recipe=None,
            rollout_difficulty=difficulty,
            rollout_bundle_root=bundle,
            output_dir=output,
            expected_hard_groups=2,
            seed=42,
        )

    @staticmethod
    def _refresh_bundle_file_inventory(bundle: Path) -> None:
        manifest_path = bundle / "bundle_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for relative, metadata in manifest["files"].items():
            path = bundle / relative
            metadata["bytes"] = path.stat().st_size
            metadata["sha256"] = curriculum_recipe._sha256_file(path)
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    @staticmethod
    def _project_difficulty(difficulty: Path) -> Path:
        authoritative_rows = curriculum_recipe._read_jsonl(difficulty)
        authoritative = difficulty.parent / "complete8.jsonl"
        write_jsonl(authoritative, authoritative_rows)
        projection_keys = (
            "record_id",
            "sample_id",
            "source_image_id",
            "task",
            "image_relpath",
            "crop_correct_count",
            "crop_complete4",
        )
        write_jsonl(
            difficulty,
            [
                {key: row.get(key) for key in projection_keys}
                for row in authoritative_rows
            ],
        )
        return authoritative

    def test_builds_disjoint_hard_anchor_and_replay_pools(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            output = root / "curriculum"
            summary = curriculum_recipe.build(self._args(bundle, difficulty, output))

            self.assertEqual(summary["hard_groups"], 2)
            self.assertEqual(summary["matched_anchor_groups"], 2)
            self.assertEqual(summary["pools"]["global_replay"]["sample_groups"], 1)
            recipe = json.loads(
                (output / "ui5_crop_rollout4_curriculum.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(
                {
                    entry["curriculum_pool"] for entry in recipe.values()
                },
                {"hard", "matched_anchor", "global_replay"},
            )
            hard_records = curriculum_recipe._read_jsonl(output / "hard.jsonl")
            self.assertEqual(
                {row["_ui5_sample_id"] for row in hard_records},
                {"hard-pos", "hard-neg"},
            )
            self.assertTrue(
                all(Path(row["image"]).is_absolute() for row in hard_records)
            )
            anchors = curriculum_recipe._read_jsonl(
                output / "matched_anchor_groups.jsonl"
            )
            self.assertEqual(
                {row["matched_hard_sample_id"] for row in anchors},
                {"hard-pos", "hard-neg"},
            )
            self.assertTrue((output / "_SUCCESS.json").is_file())

    def test_fails_closed_when_bundle_manifest_is_incomplete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            manifest_path = bundle / "bundle_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["complete"] = False
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "manifest is not complete"):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )
            self.assertFalse((root / "curriculum").exists())

    def test_fails_closed_when_declared_bundle_file_hash_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            source_records = bundle / "manifest" / "source_records.jsonl"
            payload = bytearray(source_records.read_bytes())
            payload[-1] = ord(" ")
            source_records.write_bytes(payload)

            with self.assertRaisesRegex(RuntimeError, "SHA-256 mismatch"):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_fails_closed_when_unique_image_hash_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            (bundle / "images" / "hard-pos.png").write_bytes(b"evil-image")

            with self.assertRaisesRegex(RuntimeError, "image SHA-256 mismatch"):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_fails_closed_when_source_record_uses_unverified_image(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            (bundle / "images" / "unverified.png").write_bytes(b"not-in-unique-images")
            source_records = bundle / "manifest" / "source_records.jsonl"
            rows = curriculum_recipe._read_jsonl(source_records)
            rows[-1]["portable_training_record"]["image"] = "images/unverified.png"
            write_jsonl(source_records, rows)
            self._refresh_bundle_file_inventory(bundle)

            with self.assertRaisesRegex(
                ValueError, "training record images are not verified"
            ):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_existing_curriculum_rejects_changed_bundle_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            output = root / "curriculum"
            args = self._args(bundle, difficulty, output)
            summary = curriculum_recipe.build(args)
            self.assertEqual(curriculum_recipe.build(args), summary)
            manifest_path = bundle / "bundle_manifest.json"
            manifest_path.write_text(
                manifest_path.read_text(encoding="utf-8") + "\n", encoding="utf-8"
            )

            with self.assertRaisesRegex(RuntimeError, r"changed \(rollout_bundle\)"):
                curriculum_recipe.build(args)

    def test_rejects_duplicate_difficulty_id_even_if_first_row_is_ineligible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            rows = curriculum_recipe._read_jsonl(difficulty)
            duplicate = dict(rows[0])
            duplicate["crop_complete4"] = False
            write_jsonl(difficulty, [duplicate, *rows])

            with self.assertRaisesRegex(
                ValueError, "duplicate .*rollout difficulty.*hard-pos"
            ):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_projected_difficulty_uses_verified_complete8_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            authoritative = self._project_difficulty(difficulty)

            summary = curriculum_recipe.build(
                self._args(bundle, difficulty, root / "curriculum")
            )

            state = summary["inputs"]["rollout_difficulty_authoritative"]
            self.assertTrue(state["projection_verified"])
            self.assertEqual(state["authoritative_path"], str(authoritative.resolve()))
            self.assertEqual(state["rows"], 5)

    def test_projected_difficulty_requires_complete8_sibling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            rows = curriculum_recipe._read_jsonl(difficulty)
            write_jsonl(
                difficulty,
                [
                    {
                        "sample_id": row["sample_id"],
                        "task": row["task"],
                        "crop_correct_count": row["crop_correct_count"],
                        "crop_complete4": row["crop_complete4"],
                    }
                    for row in rows
                ],
            )

            with self.assertRaisesRegex(FileNotFoundError, "complete8.jsonl"):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_projected_difficulty_must_match_complete8(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            self._project_difficulty(difficulty)
            projected = curriculum_recipe._read_jsonl(difficulty)
            projected[0]["crop_correct_count"] = 1
            write_jsonl(difficulty, projected)

            with self.assertRaisesRegex(
                ValueError, "projected/authoritative.*crop_correct_count"
            ):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_authoritative_parse_error_is_not_treated_as_clean(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            authoritative = self._project_difficulty(difficulty)
            rows = curriculum_recipe._read_jsonl(authoritative)
            rows[0]["parse_error_count"] = 1
            write_jsonl(authoritative, rows)

            with self.assertRaisesRegex(ValueError, "hard group count mismatch"):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_authoritative_polarity_conflict_with_training_record_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            rows = curriculum_recipe._read_jsonl(difficulty)
            rows[0]["gt_global"] = []
            write_jsonl(difficulty, rows)

            with self.assertRaisesRegex(ValueError, "polarity conflict.*hard-pos"):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_authoritative_task_conflict_with_training_record_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            rows = curriculum_recipe._read_jsonl(difficulty)
            rows[0]["task"] = "cropping"
            write_jsonl(difficulty, rows)

            with self.assertRaisesRegex(ValueError, "task conflict.*hard-pos"):
                curriculum_recipe.build(
                    self._args(bundle, difficulty, root / "curriculum")
                )

    def test_fails_closed_when_expected_hard_count_differs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle, difficulty = self._fixture(root)
            with self.assertRaisesRegex(ValueError, "hard group count mismatch"):
                curriculum_recipe.build(
                    Namespace(
                        base_recipe=None,
                        rollout_difficulty=difficulty,
                        rollout_bundle_root=bundle,
                        output_dir=root / "curriculum",
                        expected_hard_groups=72,
                        seed=42,
                    )
                )


if __name__ == "__main__":
    unittest.main()
