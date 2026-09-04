from __future__ import annotations

import hashlib
import json
import pickle
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from eaglevl.train.ui5_checkpoint_utils import (
    atomic_promote_checkpoint,
    checkpoint_step,
    recover_atomic_promotion,
    validate_checkpoint,
)
from eaglevl.train.ui5_curriculum import (
    CURRICULUM_POOLS,
    CurriculumGroupCycle,
    DeferredSampleLocations,
    UI5CurriculumSchedule,
    canonical_curriculum_pool,
    curriculum_artifact_identity,
    curriculum_pool_draw_counts,
    prepare_worker_states_for_resume,
    should_export_model_at_training_end,
    should_write_training_done_marker,
    training_continuity_config,
)


def formal_schedule() -> UI5CurriculumSchedule:
    return UI5CurriculumSchedule(
        total_steps=1200,
        hard_ratios=(0.60, 0.45, 0.30),
        matched_anchor_ratios=(0.25, 0.35, 0.30),
        global_replay_ratios=(0.15, 0.20, 0.40),
        llm_lrs=(1.0e-6, 7.0e-7, 5.0e-7),
        expected_hard_groups=114,
    )


class CurriculumScheduleTest(unittest.TestCase):
    def test_optimizer_step_boundaries_and_lr_are_exact(self) -> None:
        schedule = formal_schedule()
        expected = {
            1: (0, 1.0e-6),
            400: (0, 1.0e-6),
            401: (1, 7.0e-7),
            800: (1, 7.0e-7),
            801: (2, 5.0e-7),
            1200: (2, 5.0e-7),
        }
        for step, (stage, lr) in expected.items():
            with self.subTest(step=step):
                self.assertEqual(schedule.stage_for_optimizer_step(step).index, stage)
                self.assertEqual(
                    schedule.lr_for_completed_steps(step - 1), lr
                )

    def test_segment_may_end_at_but_not_cross_boundary(self) -> None:
        schedule = formal_schedule()
        schedule.validate_segment(200, 400)
        schedule.validate_segment(400, 600)
        schedule.validate_segment(800, 1000)
        with self.assertRaisesRegex(ValueError, "may not cross"):
            schedule.validate_segment(200, 401)

    def test_pool_aliases_persist_as_canonical_names(self) -> None:
        self.assertEqual(canonical_curriculum_pool("hard-matched"), "hard")
        self.assertEqual(canonical_curriculum_pool("anchor"), "matched_anchor")
        self.assertEqual(canonical_curriculum_pool("replay"), "global_replay")

    def test_cumulative_pool_draws_come_from_durable_iterator_cursors(self) -> None:
        counts = curriculum_pool_draw_counts(
            {
                "worker_0": {
                    "iterator_states": [
                        {"global_idx": 7},
                        {"global_idx": 2},
                        {"global_idx": 5},
                        {"global_idx": 3},
                    ]
                },
                "worker_1": {
                    "iterator_states": [
                        {"global_idx": 11},
                        {"global_idx": 13},
                        {"global_idx": 17},
                        {"global_idx": 19},
                    ]
                },
            },
            ("hard", "hard", "anchor", "replay"),
        )
        self.assertEqual(
            counts,
            {"hard": 33, "matched_anchor": 22, "global_replay": 22},
        )

    def test_cumulative_pool_draws_reject_incomplete_worker_snapshot(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not match dataset_pools"):
            curriculum_pool_draw_counts(
                {"worker_0": {"iterator_states": [{"global_idx": 1}]}},
                ("hard", "anchor"),
            )

    def test_pool_draw_counts_prefer_sampler_draws_over_iterator_cursors(self) -> None:
        counts = curriculum_pool_draw_counts(
            {
                "worker_0": {
                    "iterator_states": [
                        {"global_idx": 9},
                        {"global_idx": 8},
                        {"global_idx": 7},
                    ],
                    # A deferred hard sample was selected again without moving
                    # its already-advanced iterator cursor.
                    "dataset_sampler_draws": [10, 8, 7],
                }
            },
            CURRICULUM_POOLS,
        )
        self.assertEqual(
            counts,
            {"hard": 10, "matched_anchor": 8, "global_replay": 7},
        )

    def test_outer_weights_preserve_relative_weights_inside_each_pool(self) -> None:
        schedule = formal_schedule()
        weights = schedule.effective_dataset_weights(
            dataset_pools=("hard", "hard", "matched_anchor", "global_replay"),
            base_weights=(2.0, 1.0, 9.0, 4.0),
            completed_global_step=0,
        )
        for actual, expected in zip(weights, (0.4, 0.2, 0.25, 0.15)):
            self.assertAlmostEqual(actual, expected)
        self.assertAlmostEqual(sum(weights), 1.0)

    def test_sampler_state_binds_schedule_and_global_step(self) -> None:
        schedule = formal_schedule()
        state = schedule.sampler_state(
            completed_global_step=400, sampling_stage_index=0
        )
        schedule.validate_sampler_state(state, expected_global_step=400)
        with self.assertRaisesRegex(RuntimeError, "global step"):
            schedule.validate_sampler_state(state, expected_global_step=401)
        changed = formal_schedule().to_dict()
        changed["expected_hard_groups"] = 71
        state["schedule"] = changed
        with self.assertRaisesRegex(RuntimeError, "payload changed"):
            schedule.validate_sampler_state(state, expected_global_step=400)

    def test_stage_transition_defers_every_pending_sample_without_cursor_loss(self) -> None:
        original = {
            "worker_0": {
                "iterator_states": [{"seed": 1, "global_idx": 99}],
                "sample_rng_state": (3, (1, 2, 3), None),
                "dataset_sampler_draws": [99],
                "current_batch_locations": [(0, 90)],
                "buffer_locations": [(0, 91), (0, 92)],
            }
        }
        resumed, report = prepare_worker_states_for_resume(
            original, saved_sampling_stage=0, resume_sampling_stage=1
        )
        self.assertEqual(resumed["worker_0"]["current_batch_locations"], [])
        self.assertEqual(resumed["worker_0"]["buffer_locations"], [])
        self.assertEqual(
            resumed["worker_0"]["deferred_locations"],
            [(0, 90), (0, 91), (0, 92)],
        )
        self.assertEqual(
            resumed["worker_0"]["iterator_states"][0]["global_idx"], 99
        )
        self.assertEqual(resumed["worker_0"]["dataset_sampler_draws"], [99])
        self.assertEqual(report["current_batch_samples"], 1)
        self.assertEqual(report["buffer_samples"], 2)
        self.assertEqual(report["deferred_samples"], 3)
        self.assertEqual(original["worker_0"]["buffer_locations"], [(0, 91), (0, 92)])

        queue = DeferredSampleLocations(
            resumed["worker_0"]["deferred_locations"],
            dataset_count=1,
            iterator_states=resumed["worker_0"]["iterator_states"],
        )
        replayed = []
        while len(queue):
            replayed.append(queue.pop_for_dataset(0))
        self.assertEqual(replayed, [(0, 90), (0, 91), (0, 92)])

    def test_same_stage_resume_preserves_packing_state_byte_for_byte(self) -> None:
        original = {
            "worker_0": {
                "iterator_states": [{"seed": 1, "global_idx": 9}],
                "sample_rng_state": random.Random(5).getstate(),
                "dataset_sampler_draws": [11],
                "current_batch_locations": [(0, 7)],
                "buffer_locations": [(0, 8)],
                "deferred_locations": [(0, 6)],
            }
        }
        resumed, report = prepare_worker_states_for_resume(
            original, saved_sampling_stage=1, resume_sampling_stage=1
        )
        self.assertEqual(resumed, original)
        self.assertIsNot(resumed, original)
        self.assertEqual(report["deferred_samples"], 0)

    def test_deferred_locations_survive_repeated_checkpoint_resume(self) -> None:
        queue = DeferredSampleLocations(
            [(1, 8), (0, 5), (1, 7)],
            dataset_count=2,
            iterator_states=[{"global_idx": 6}, {"global_idx": 9}],
        )
        self.assertEqual(queue.pop_for_dataset(1), (1, 7))
        serialized = pickle.loads(pickle.dumps(queue.to_list()))
        resumed = DeferredSampleLocations(
            serialized,
            dataset_count=2,
            iterator_states=[{"global_idx": 6}, {"global_idx": 9}],
        )
        self.assertEqual(resumed.pop_for_dataset(0), (0, 5))
        self.assertEqual(resumed.pop_for_dataset(1), (1, 8))
        self.assertEqual(len(resumed), 0)

    def test_phase_transition_replays_exact_group_views_before_new_cursor(self) -> None:
        cycle = CurriculumGroupCycle(
            {"a": ("a0", "a1"), "b": ("b0", "b1", "b2")}
        )
        original = {
            "worker_0": {
                "iterator_states": [{"seed": 37, "global_idx": 6}],
                "sample_rng_state": random.Random(8).getstate(),
                "dataset_sampler_draws": [6],
                # Packing order need not equal iterator order.
                "current_batch_locations": [(0, 5)],
                "buffer_locations": [(0, 4)],
            }
        }
        prepared, _ = prepare_worker_states_for_resume(
            original, saved_sampling_stage=0, resume_sampling_stage=1
        )
        queue = DeferredSampleLocations(
            prepared["worker_0"]["deferred_locations"],
            dataset_count=1,
            iterator_states=prepared["worker_0"]["iterator_states"],
        )
        locations = [queue.pop_for_dataset(0), queue.pop_for_dataset(0)]
        locations.append((0, prepared["worker_0"]["iterator_states"][0]["global_idx"]))
        replayed = [cycle.draw_at(location[1], seed=37) for location in locations]
        expected = [cycle.draw_at(index, seed=37) for index in (4, 5, 6)]
        self.assertEqual(replayed, expected)

    def test_group_cycle_is_group_uniform_and_each_group_cycles_its_views(self) -> None:
        views = {
            "one-tile": ("o0",),
            "three-tiles": ("t0", "t1", "t2"),
            "seven-tiles": tuple(f"s{i}" for i in range(7)),
        }
        cycle = CurriculumGroupCycle(views)
        draws = [
            cycle.draw_at(index, seed=42)
            for index in range(cycle.group_count * 14)
        ]
        group_counts = {
            group_id: sum(draw["group_id"] == group_id for draw in draws)
            for group_id in views
        }
        self.assertEqual(set(group_counts.values()), {14})
        for group_id, group_views in views.items():
            selected = [draw for draw in draws if draw["group_id"] == group_id]
            for previous, current in zip(selected, selected[1:]):
                self.assertEqual(
                    current["view_index"],
                    (previous["view_index"] + 1) % len(group_views),
                )

    def test_tile_cardinality_cannot_change_pool_or_group_sequence(self) -> None:
        schedule = formal_schedule()
        small = {
            pool: CurriculumGroupCycle(
                {f"{pool}-a": ("a0",), f"{pool}-b": ("b0",)}
            )
            for pool in CURRICULUM_POOLS
        }
        uneven = {
            pool: CurriculumGroupCycle(
                {
                    f"{pool}-a": tuple(f"a{i}" for i in range(9)),
                    f"{pool}-b": tuple(f"b{i}" for i in range(2)),
                }
            )
            for pool in CURRICULUM_POOLS
        }
        rng = random.Random(42)
        cursors = {pool: 0 for pool in CURRICULUM_POOLS}
        small_sequence = []
        uneven_sequence = []
        # Four microbatches per optimizer step, with variable packed sample
        # counts, exercises both gradient accumulation and packing cardinality.
        packed_samples = (1, 3, 2, 4)
        for optimizer_step in range(1, 1201):
            weights = schedule.stage_for_optimizer_step(optimizer_step).pool_weights
            for sample_count in packed_samples:
                for _ in range(sample_count):
                    pool_index = rng.choices(range(3), weights=weights)[0]
                    pool = CURRICULUM_POOLS[pool_index]
                    cursor = cursors[pool]
                    small_draw = small[pool].draw_at(cursor, seed=123 + pool_index)
                    uneven_draw = uneven[pool].draw_at(
                        cursor, seed=123 + pool_index
                    )
                    small_sequence.append((pool, small_draw["group_id"]))
                    uneven_sequence.append((pool, uneven_draw["group_id"]))
                    cursors[pool] += 1
            for pool in CURRICULUM_POOLS:
                state = uneven[pool].iterator_state(
                    seed=123 + CURRICULUM_POOLS.index(pool),
                    global_idx=cursors[pool],
                )
                uneven[pool].validate_iterator_state(
                    pickle.loads(pickle.dumps(state)),
                    seed=123 + CURRICULUM_POOLS.index(pool),
                    global_idx=cursors[pool],
                )
        self.assertEqual(small_sequence, uneven_sequence)

    def test_group_cycle_arbitrary_resume_reproduces_future_sequence(self) -> None:
        cycle = CurriculumGroupCycle(
            {"a": ("a0", "a1"), "b": ("b0",), "c": ("c0", "c1", "c2")}
        )
        uninterrupted = [cycle.draw_at(index, seed=91) for index in range(300)]
        # Check every optimizer boundary for gradient_accumulation_steps=4;
        # packed microbatches contain a non-constant number of sample groups.
        cursor = 0
        packed_samples = (3, 1, 4, 2)
        for _optimizer_step in range(30):
            cursor += sum(packed_samples)
            state = pickle.loads(
                pickle.dumps(cycle.iterator_state(seed=91, global_idx=cursor))
            )
            cycle.validate_iterator_state(state, seed=91, global_idx=cursor)
            resumed = [
                cycle.draw_at(index, seed=state["seed"])
                for index in range(state["global_idx"], 300)
            ]
            self.assertEqual(resumed, uninterrupted[cursor:])

    def test_environment_defaults_reproduce_formal_schedule(self) -> None:
        schedule = UI5CurriculumSchedule.from_environment(
            {"CURRICULUM_MODE": "scheduled", "TOTAL_STEPS": "1200"},
            default_total_steps=1,
        )
        self.assertIsNotNone(schedule)
        self.assertEqual(schedule.stages[1].first_optimizer_step, 401)
        self.assertEqual(schedule.stages[2].pool_weights, (0.30, 0.30, 0.40))

    def test_training_continuity_binds_model_data_loss_and_implementation(self) -> None:
        training = SimpleNamespace(
            seed=42,
            data_seed=None,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=1.0e-6,
            weight_decay=0.01,
            adam_beta1=0.9,
            adam_beta2=0.999,
            adam_epsilon=1.0e-8,
            max_grad_norm=1.0,
            optim="adamw_torch",
            bf16=True,
            fp16=False,
            max_steps=1200,
            deepspeed=None,
        )
        model = SimpleNamespace(
            model_name_or_path="/models/crop",
            attn_implementation="sdpa",
            enable_ui_relation=True,
            relation_gate_loss_weight=1.0,
            relation_slot_gate_loss_weight=0.1,
            relation_attention_loss_weight=0.1,
            relation_gate_threshold=0.5,
            relation_focal_beta=0.999,
            relation_focal_gamma=2.0,
        )
        data = SimpleNamespace(
            max_seq_length=7268,
            meta_path="/run/curriculum.json",
            use_online_packing=True,
            max_num_tokens_per_sample=7268,
            max_num_tokens=7268,
            packing_buffer_size=32,
            ui_sampling_mode="fixed_ratio",
        )
        first = training_continuity_config(
            training, formal_schedule(), model_args=model, data_args=data
        )
        unchanged = training_continuity_config(
            training, formal_schedule(), model_args=model, data_args=data
        )
        self.assertEqual(first, unchanged)
        self.assertEqual(first["model_semantics"]["attn_implementation"], "sdpa")
        self.assertEqual(first["data_semantics"]["max_num_tokens"], 7268)
        expected_implementations = {
            "eaglevl/train/arguments.py",
            "eaglevl/train/locany_finetune_magi_stream.py",
            "eaglevl/train/ui5_checkpoint_utils.py",
            "eaglevl/train/ui5_curriculum.py",
            "eaglevl/model/locany/modeling_locateanything.py",
            "eaglevl/model/locany/relation_modules.py",
            "eaglevl/model/locany/ui_relation_setup.py",
        }
        self.assertEqual(set(first["implementation_sha256"]), expected_implementations)
        self.assertTrue(
            all(len(value) == 64 for value in first["implementation_sha256"].values())
        )

        changed_model = SimpleNamespace(**vars(model))
        changed_model.relation_gate_loss_weight = 1.25
        self.assertNotEqual(
            first,
            training_continuity_config(
                training,
                formal_schedule(),
                model_args=changed_model,
                data_args=data,
            ),
        )
        changed_data = SimpleNamespace(**vars(data))
        changed_data.max_num_tokens = 7000
        self.assertNotEqual(
            first,
            training_continuity_config(
                training,
                formal_schedule(),
                model_args=model,
                data_args=changed_data,
            ),
        )

    def test_segment_mode_never_exports_a_duplicate_root_model(self) -> None:
        self.assertFalse(should_export_model_at_training_end(segment_mode=True))
        self.assertTrue(should_export_model_at_training_end(segment_mode=False))

    def test_segment_mode_never_writes_training_done_marker(self) -> None:
        self.assertFalse(should_write_training_done_marker(segment_mode=True))
        self.assertTrue(should_write_training_done_marker(segment_mode=False))

    def test_artifact_identity_checks_hard_count_and_file_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            recipe = root / "ui5_crop_rollout4_curriculum.json"
            hard = root / "hard.jsonl"
            recipe.write_text("{}\n", encoding="utf-8")
            hard.write_text("{}\n", encoding="utf-8")
            manifest = {
                "schema_version": 1,
                "expected_hard_groups": 114,
                "hard_groups": 114,
                "matched_anchor_groups": 114,
            }
            identity = hashlib.sha256(
                json.dumps(
                    manifest,
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
            ).hexdigest()
            manifest["identity_digest"] = identity
            self._write_json_for_artifact(root / "curriculum_manifest.json", manifest)
            recipe_hash = hashlib.sha256(recipe.read_bytes()).hexdigest()
            hard_hash = hashlib.sha256(hard.read_bytes()).hexdigest()
            self._write_json_for_artifact(
                root / "_SUCCESS.json",
                {
                    "complete": True,
                    "identity_digest": identity,
                    "recipe_sha256": recipe_hash,
                    "files": {recipe.name: recipe_hash, hard.name: hard_hash},
                },
            )
            result = curriculum_artifact_identity(recipe, formal_schedule())
            self.assertEqual(result["hard_groups"], 114)
            hard.write_text('{"changed": true}\n', encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                curriculum_artifact_identity(recipe, formal_schedule())

    @staticmethod
    def _write_json_for_artifact(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")


class StrictCheckpointTest(unittest.TestCase):
    @staticmethod
    def _write_json(path: Path, payload: dict) -> None:
        path.write_text(json.dumps(payload), encoding="utf-8")

    def _make_checkpoint(self, root: Path, step: int, *, ranks: int = 1) -> Path:
        schedule = formal_schedule()
        group_cycle = CurriculumGroupCycle(
            {"sample-0": ("full_image:global",)}
        )
        checkpoint = root / f"checkpoint-{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "config.json").write_text("{}", encoding="utf-8")
        (checkpoint / "model.safetensors").write_bytes(b"model")
        (checkpoint / "training_args.bin").write_bytes(b"args")
        (checkpoint / "optimizer.pt").write_bytes(b"optimizer")
        (checkpoint / "scheduler.pt").write_bytes(b"scheduler")
        for rank in range(ranks):
            rng_name = "rng_state.pth" if ranks == 1 else f"rng_state_{rank}.pth"
            with (checkpoint / rng_name).open("wb") as handle:
                pickle.dump(
                    {"python": (3, (), None), "numpy": ("MT19937",), "cpu": b"cpu"},
                    handle,
                )
        self._write_json(checkpoint / "trainer_state.json", {"global_step": step})

        stream_config = {
            "base_seed": 42,
            "data_world_size": ranks,
            "datasets": [
                {
                    "name": "hard",
                    "rows": 114,
                    "sampling_unit": "sample_group",
                    "curriculum_group_identity": group_cycle.identity,
                    "base_probability": 1.0,
                    "curriculum_pool": "hard",
                }
            ],
            "curriculum_schedule": schedule.to_dict(),
        }
        rank_state = {
            "version": 8,
            "num_workers": 1,
            "stream_resume_config": stream_config,
            "worker_states": {
                "worker_0": {
                    "iterator_states": [
                        {
                            "seed": 42,
                            "global_idx": step,
                            "curriculum_group_cycle": group_cycle.iterator_state(
                                seed=42, global_idx=step
                            ),
                        }
                    ],
                    "sample_rng_state": (3, (1, 2, 3), None),
                    "dataset_sampler_draws": [step],
                    "current_batch_locations": [],
                    "buffer_locations": [],
                    "deferred_locations": [],
                }
            },
            "curriculum_sampler": schedule.sampler_state(
                completed_global_step=step,
                sampling_stage_index=schedule.sampling_stage_at_checkpoint(step).index,
            ),
        }
        for rank in range(ranks):
            with (checkpoint / f"dataloader_state_rank{rank}.pt").open("wb") as handle:
                pickle.dump(rank_state, handle)
        stream_digest = hashlib.sha256(
            json.dumps(
                stream_config,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        training_config = training_continuity_config(
            SimpleNamespace(
                seed=42,
                data_seed=None,
                per_device_train_batch_size=1,
                gradient_accumulation_steps=1,
                learning_rate=1.0e-6,
                weight_decay=0.01,
                adam_beta1=0.9,
                adam_beta2=0.999,
                adam_epsilon=1.0e-8,
                max_grad_norm=1.0,
                optim="adamw_torch",
                bf16=True,
                fp16=False,
                deepspeed=None,
            ),
            schedule,
        )
        training_digest = hashlib.sha256(
            json.dumps(
                training_config,
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        self._write_json(
            checkpoint / "continuity_state.json",
            {
                "schema_version": 1,
                "global_step": step,
                "source_global_step": max(0, step - 200),
                "segment_target_global_step": step,
                "target_total_steps": 1200,
                "world_size": ranks,
                "precision": "bf16",
                "gradient_scaler": {
                    "applicable": False,
                    "storage": "not_applicable",
                },
                "curriculum_mode": "scheduled",
                "dataloader_state_version": 8,
                "curriculum_schedule_fingerprint": schedule.fingerprint,
                "stream_resume_config_digest": stream_digest,
                "training_continuity_config": training_config,
                "training_continuity_config_digest": training_digest,
            },
        )
        self._write_json(
            checkpoint / "checkpoint_complete.json",
            {"schema_version": 1, "global_step": step},
        )
        return checkpoint

    @staticmethod
    def _fake_torch() -> SimpleNamespace:
        def load(path, **_kwargs):
            with Path(path).open("rb") as handle:
                return pickle.load(handle)

        return SimpleNamespace(load=load)

    def test_arbitrary_resume_latest_uses_trainer_global_step(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = self._make_checkpoint(root, 400)
            latest = root / "resume" / "latest"
            latest.parent.mkdir()
            source.rename(latest)
            with mock.patch.dict(sys.modules, {"torch": self._fake_torch()}):
                report = validate_checkpoint(
                    latest,
                    mode="resume",
                    expected_ranks=1,
                    strict=True,
                    expected_curriculum_fingerprint=formal_schedule().fingerprint,
                )
            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(checkpoint_step(latest), 400)

    def test_scheduled_resume_rejects_legacy_v7_dataloader_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._make_checkpoint(Path(temporary), 400)
            continuity_path = checkpoint / "continuity_state.json"
            continuity = json.loads(continuity_path.read_text(encoding="utf-8"))
            continuity["dataloader_state_version"] = 7
            self._write_json(continuity_path, continuity)
            rank_path = checkpoint / "dataloader_state_rank0.pt"
            with rank_path.open("rb") as handle:
                rank_state = pickle.load(handle)
            rank_state["version"] = 7
            with rank_path.open("wb") as handle:
                pickle.dump(rank_state, handle)
            with mock.patch.dict(sys.modules, {"torch": self._fake_torch()}):
                report = validate_checkpoint(
                    checkpoint, mode="resume", expected_ranks=1, strict=True
                )
            self.assertFalse(report["valid"])
            self.assertTrue(
                any("version >= 8" in error for error in report["errors"]),
                report["errors"],
            )
            self.assertTrue(
                any("older than 8" in error for error in report["errors"]),
                report["errors"],
            )

    def test_strict_validation_rejects_corrupt_group_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._make_checkpoint(Path(temporary), 400)
            rank_path = checkpoint / "dataloader_state_rank0.pt"
            with rank_path.open("rb") as handle:
                rank_state = pickle.load(handle)
            rank_state["worker_states"]["worker_0"]["iterator_states"][0][
                "curriculum_group_cycle"
            ]["next_draw"]["view_id"] = "wrong-view"
            with rank_path.open("wb") as handle:
                pickle.dump(rank_state, handle)
            with mock.patch.dict(sys.modules, {"torch": self._fake_torch()}):
                report = validate_checkpoint(
                    checkpoint, mode="resume", expected_ranks=1, strict=True
                )
            self.assertFalse(report["valid"])
            self.assertTrue(
                any("invalid group cursor" in error for error in report["errors"]),
                report["errors"],
            )

    def test_strict_validation_rejects_missing_rng(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._make_checkpoint(Path(temporary), 200)
            (checkpoint / "rng_state.pth").unlink()
            with mock.patch.dict(sys.modules, {"torch": self._fake_torch()}):
                report = validate_checkpoint(
                    checkpoint, mode="resume", expected_ranks=1, strict=True
                )
            self.assertFalse(report["valid"])
            self.assertTrue(any("RNG" in error for error in report["errors"]))

    def test_two_rank_rng_and_dataloader_filenames_are_required(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._make_checkpoint(Path(temporary), 200, ranks=2)
            with mock.patch.dict(sys.modules, {"torch": self._fake_torch()}):
                report = validate_checkpoint(
                    checkpoint, mode="resume", expected_ranks=2, strict=True
                )
            self.assertTrue(report["valid"], report["errors"])
            self.assertEqual(report["details"]["rng_state_files"], 2)
            self.assertEqual(report["details"]["dataloader_state_files"], 2)

            (checkpoint / "rng_state_1.pth").rename(checkpoint / "rng_state_9.pth")
            (checkpoint / "dataloader_state_rank1.pt").rename(
                checkpoint / "dataloader_state_rank9.pt"
            )
            with mock.patch.dict(sys.modules, {"torch": self._fake_torch()}):
                misnamed = validate_checkpoint(
                    checkpoint, mode="resume", expected_ranks=2, strict=True
                )
            self.assertFalse(misnamed["valid"])
            self.assertTrue(
                any("RNG state filenames" in error for error in misnamed["errors"])
            )
            self.assertTrue(
                any(
                    "dataloader state filenames" in error
                    for error in misnamed["errors"]
                )
            )
            (checkpoint / "rng_state_9.pth").rename(checkpoint / "rng_state_1.pth")
            (checkpoint / "dataloader_state_rank9.pt").rename(
                checkpoint / "dataloader_state_rank1.pt"
            )

            (checkpoint / "rng_state_1.pth").unlink()
            (checkpoint / "dataloader_state_rank1.pt").unlink()
            with mock.patch.dict(sys.modules, {"torch": self._fake_torch()}):
                incomplete = validate_checkpoint(
                    checkpoint, mode="resume", expected_ranks=2, strict=True
                )
            self.assertFalse(incomplete["valid"])
            self.assertTrue(
                any("RNG state count" in error for error in incomplete["errors"])
            )
            self.assertTrue(
                any("dataloader state count" in error for error in incomplete["errors"])
            )

    def test_strict_validation_requires_scaler_for_fp16(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._make_checkpoint(Path(temporary), 200)
            with mock.patch.dict(sys.modules, {"torch": self._fake_torch()}):
                report = validate_checkpoint(
                    checkpoint,
                    mode="resume",
                    expected_ranks=1,
                    strict=True,
                    scaler_required=True,
                )
            self.assertFalse(report["valid"])
            self.assertTrue(
                any("scaler" in error.lower() for error in report["errors"])
            )

    def test_strict_validation_binds_segment_source_and_target(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._make_checkpoint(Path(temporary), 400)
            manifest_path = checkpoint / "continuity_state.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_global_step"] = 400
            manifest["segment_target_global_step"] = 600
            self._write_json(manifest_path, manifest)
            with mock.patch.dict(sys.modules, {"torch": self._fake_torch()}):
                report = validate_checkpoint(
                    checkpoint, mode="resume", expected_ranks=1, strict=True
                )
            self.assertFalse(report["valid"])
            self.assertTrue(
                any("valid segment start" in error for error in report["errors"])
            )
            self.assertTrue(
                any("segment_target_global_step" in error for error in report["errors"])
            )

    def test_deepspeed_zero_optimizer_shards_cover_both_dp_ranks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = self._make_checkpoint(Path(temporary), 200, ranks=2)
            (checkpoint / "optimizer.pt").unlink()
            tag = checkpoint / "global_step200"
            tag.mkdir()
            (tag / "mp_rank_00_model_states.pt").write_bytes(b"model-state")
            for rank in range(2):
                (tag / f"bf16_zero_pp_rank_{rank}_mp_rank_00_optim_states.pt").write_bytes(
                    b"optimizer-shard"
                )

            with mock.patch.dict(sys.modules, {"torch": self._fake_torch()}):
                valid = validate_checkpoint(
                    checkpoint, mode="resume", expected_ranks=2, strict=True
                )
            self.assertTrue(valid["valid"], valid["errors"])
            self.assertEqual(
                valid["details"]["deepspeed_zero_optimizer_dp_ranks"], [0, 1]
            )

            rank_one = tag / "bf16_zero_pp_rank_1_mp_rank_00_optim_states.pt"
            rank_one.unlink()
            with mock.patch.dict(sys.modules, {"torch": self._fake_torch()}):
                missing = validate_checkpoint(
                    checkpoint, mode="resume", expected_ranks=2, strict=True
                )
            self.assertFalse(missing["valid"])
            self.assertTrue(
                any("shard count" in error for error in missing["errors"])
            )

            (tag / "bf16_zero_pp_rank_9_mp_rank_00_optim_states.pt").write_bytes(
                b"optimizer-shard"
            )
            with mock.patch.dict(sys.modules, {"torch": self._fake_torch()}):
                wrong_rank = validate_checkpoint(
                    checkpoint, mode="resume", expected_ranks=2, strict=True
                )
            self.assertFalse(wrong_rank["valid"])
            self.assertTrue(
                any("dp ranks are incomplete" in error for error in wrong_rank["errors"])
            )

    def test_move_promotion_replaces_latest_without_checkpoint_copy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resume = root / "resume"
            resume.mkdir()
            first = self._make_checkpoint(root, 200)
            with mock.patch.dict(sys.modules, {"torch": self._fake_torch()}):
                atomic_promote_checkpoint(
                    first,
                    resume / "latest",
                    expected_ranks=1,
                    strict=True,
                    move_source=True,
                )
            self.assertFalse(first.exists())
            self.assertEqual(checkpoint_step(resume / "latest"), 200)

            second = self._make_checkpoint(root, 400)
            with mock.patch.dict(sys.modules, {"torch": self._fake_torch()}):
                result = atomic_promote_checkpoint(
                    second,
                    resume / "latest",
                    expected_ranks=1,
                    strict=True,
                    move_source=True,
                )
            self.assertFalse(second.exists())
            self.assertEqual(result["global_step"], 400)
            self.assertEqual(checkpoint_step(resume / "latest"), 400)
            self.assertEqual(list(resume.glob(".latest.*")), [])

    def test_recovers_each_reachable_promotion_crash_phase(self) -> None:
        transaction_id = "a" * 32
        phases = (
            # source -> staging completed; old rolling checkpoint is still live
            ("destination_and_staging", True, True, False, 400),
            # old rolling checkpoint was moved to backup; staging is not committed
            ("staging_and_backup", False, True, True, 400),
            # staging was committed; obsolete backup has not been removed
            ("destination_and_backup", True, False, True, 400),
            # first promotion has a staging checkpoint and no previous destination
            ("initial_staging", False, True, False, 200),
        )
        for name, has_destination, has_staging, has_backup, new_step in phases:
            with self.subTest(phase=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                resume = root / "resume"
                resume.mkdir()
                destination = resume / "latest"
                staging = resume / f".latest.staging-{transaction_id}"
                backup = resume / f".latest.backup-{transaction_id}"

                if has_destination:
                    step = new_step if has_backup and not has_staging else 200
                    source = self._make_checkpoint(root / "destination", step)
                    source.rename(destination)
                if has_staging:
                    source = self._make_checkpoint(root / "staging", new_step)
                    source.rename(staging)
                if has_backup:
                    source = self._make_checkpoint(root / "backup", 200)
                    source.rename(backup)

                with mock.patch.dict(sys.modules, {"torch": self._fake_torch()}):
                    result = recover_atomic_promotion(
                        destination,
                        expected_ranks=1,
                        strict=True,
                        expected_step_delta=200,
                    )

                self.assertTrue(result["recovered"])
                self.assertEqual(result["global_step"], new_step)
                self.assertEqual(checkpoint_step(destination), new_step)
                self.assertFalse(staging.exists())
                self.assertFalse(backup.exists())

    def test_recovery_rejects_ambiguous_invalid_and_nonconsecutive_states(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resume = root / "resume"
            resume.mkdir()
            destination = resume / "latest"
            staging = resume / f".latest.staging-{'a' * 32}"
            backup = resume / f".latest.backup-{'b' * 32}"
            self._make_checkpoint(root / "staging", 400).rename(staging)
            self._make_checkpoint(root / "backup", 200).rename(backup)
            with self.assertRaisesRegex(RuntimeError, "multiple transaction ids"):
                recover_atomic_promotion(
                    destination,
                    expected_ranks=1,
                    strict=True,
                    expected_step_delta=200,
                )
            self.assertTrue(staging.is_dir())
            self.assertTrue(backup.is_dir())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resume = root / "resume"
            resume.mkdir()
            destination = resume / "latest"
            staging = resume / f".latest.staging-{'c' * 32}"
            self._make_checkpoint(root / "destination", 200).rename(destination)
            self._make_checkpoint(root / "staging", 600).rename(staging)
            with mock.patch.dict(sys.modules, {"torch": self._fake_torch()}):
                with self.assertRaisesRegex(RuntimeError, "not consecutive"):
                    recover_atomic_promotion(
                        destination,
                        expected_ranks=1,
                        strict=True,
                        expected_step_delta=200,
                    )
            self.assertEqual(checkpoint_step(destination), 200)
            self.assertEqual(checkpoint_step(staging), 600)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resume = root / "resume"
            resume.mkdir()
            destination = resume / "latest"
            backup = resume / f".latest.backup-{'d' * 32}"
            self._make_checkpoint(root / "backup", 200).rename(backup)
            with self.assertRaisesRegex(RuntimeError, "invalid rolling transaction state"):
                recover_atomic_promotion(
                    destination,
                    expected_ranks=1,
                    strict=True,
                    expected_step_delta=200,
                )
            self.assertTrue(backup.is_dir())

    def test_recovery_rejects_malformed_or_invalid_artifact_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resume = root / "resume"
            resume.mkdir()
            destination = resume / "latest"
            malformed = resume / ".latest.staging-not-a-uuid"
            malformed.mkdir()
            with self.assertRaisesRegex(RuntimeError, "malformed"):
                recover_atomic_promotion(destination, expected_step_delta=200)
            self.assertTrue(malformed.is_dir())

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            resume = root / "resume"
            resume.mkdir()
            destination = resume / "latest"
            staging = resume / f".latest.staging-{'e' * 32}"
            source = self._make_checkpoint(root / "staging", 200)
            (source / "scheduler.pt").unlink()
            source.rename(staging)
            with mock.patch.dict(sys.modules, {"torch": self._fake_torch()}):
                with self.assertRaisesRegex(RuntimeError, "invalid rolling staging"):
                    recover_atomic_promotion(
                        destination,
                        expected_ranks=1,
                        strict=True,
                        expected_step_delta=200,
                    )
            self.assertFalse(destination.exists())
            self.assertTrue(staging.is_dir())

    def test_recovery_cli_reports_clean_no_transaction_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "resume" / "latest"
            completed = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(
                        Path(__file__).resolve().parents[1]
                        / "scripts"
                        / "locany_ui5_checkpoint.py"
                    ),
                    "recover",
                    "--destination",
                    str(destination),
                    "--expected-ranks",
                    "2",
                    "--expected-step-delta",
                    "200",
                    "--strict",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads(completed.stdout)
            self.assertEqual(payload["action"], "no_transaction")
            self.assertFalse(payload["recovered"])


if __name__ == "__main__":
    unittest.main()
