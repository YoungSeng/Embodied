"""CPU-only equivalence and complexity checks against the pre-fix sampler."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import random
import unittest
from unittest import mock

from eaglevl.train import ui_defect_data as sampler


def legacy_permutation(values, seed, *parts):
    payload = "\x1f".join([str(seed), *(str(part) for part in parts)])
    seed = int.from_bytes(hashlib.sha256(payload.encode()).digest()[:8], "big")
    result = list(values)
    random.Random(seed).shuffle(result)
    return result


def legacy_draw(groups, manual, position, seed, task, polarity):
    source_ids = tuple(sorted(groups))
    cycle, offset = divmod(position, len(source_ids))
    source = legacy_permutation(source_ids, seed, "source", task, polarity, cycle)[offset]
    values = tuple(groups[source])
    record_cycle, record_offset = divmod(cycle, len(values))
    if record_cycle == 0:
        required = sorted(i for i in values if i in manual)
        ordinary = [i for i in values if i not in manual]
        order = [*required, *legacy_permutation(ordinary, seed, "record", task, polarity, source, 0)]
    else:
        order = legacy_permutation(values, seed, "record", task, polarity, source, record_cycle)
    return int(order[record_offset])


def legacy_materialize(plan, seed, epoch):
    streams = {}
    for task, polarities in sorted(plan["buckets"].items()):
        sides = {}
        for polarity in ("positive", "negative"):
            count = plan["slots_by_task"][task][polarity]
            sides[polarity] = [legacy_draw(polarities[polarity], plan["manual_indices"], epoch * count + i,
                                          seed, task, polarity) for i in range(count)]
        positive, negative = sides["positive"], sides["negative"]
        stream = []; p = n = 0
        for i in range(len(positive) + len(negative)):
            if (i + 1) * len(positive) // max(1, len(positive) + len(negative)) > p:
                stream.append(positive[p]); p += 1
            else:
                stream.append(negative[n]); n += 1
        streams[task] = stream
    tasks = legacy_permutation(sorted(streams), seed, "task-order", epoch)
    result = []
    for i in range(plan["per_task_records"]):
        offset = i % len(tasks)
        result.extend(streams[task][i] for task in tasks[offset:] + tasks[:offset])
    return result


def make_records(task, positive_sources, negative_sources, crops=3):
    records = []
    for positive, sources in ((True, positive_sources), (False, negative_sources)):
        for source in range(sources):
            for crop in range(crops if source % 2 == 0 else 1):
                records.append({"task_id": task, "source_image_id": f"source-{source:06d}",
                    "_ui5_crop_source": "manual_gt_repair" if positive and crop == 0 and source == 0 else "detector_scan",
                    "conversations": [{"from": "gpt", "value": "<box><1><2><30><40></box>" if positive else "<box>none</box>"}]})
    return records


class SourceCycleCacheTests(unittest.TestCase):
    def test_exact_legacy_order_all_14_tasks_multiple_epochs_seeds_and_manual_crops(self):
        records = []
        for task in range(14):
            records.extend(make_records(task, 0 if task == 13 else task % 5 + 1,
                                        0 if task == 11 else task % 7 + 1))
        plan = sampler.build_task_source_balanced_rotating_plan(records)
        for seed in (42, 202603, 30057):
            for epoch in (0, 1, 2, 7, 31, 1000):
                with self.subTest(seed=seed, epoch=epoch):
                    self.assertEqual(sampler.materialize_task_source_balanced_rotating_indices(plan, seed=seed, epoch_index=epoch),
                                     legacy_materialize(plan, seed, epoch))

    def test_single_task_single_sided_and_unaligned_cycle_boundaries(self):
        for positive, negative in ((7, 0), (0, 11), (7, 11), (1, 1), (1, 17)):
            plan = sampler.build_task_source_balanced_rotating_plan(make_records(8, positive, negative, crops=5))
            for epoch in (0, 1, 2, 19):
                actual = sampler.materialize_task_source_balanced_rotating_indices(plan, seed=52, epoch_index=epoch)
                self.assertEqual(actual, legacy_materialize(plan, 52, epoch))
                self.assertEqual(len(actual), plan["epoch_length"])

    def test_source_permutation_count_is_cycles_not_record_count(self):
        plan = sampler.build_task_source_balanced_rotating_plan(make_records(8, 2000, 1501, crops=1))
        original = sampler._deterministic_permutation
        for epoch in (0, 1, 123):
            calls = []
            def permutation(values, seed, *namespace):
                if namespace[0] == "source": calls.append(namespace)
                return original(values, seed, *namespace)
            with mock.patch.object(sampler, "_deterministic_permutation", side_effect=permutation):
                draws = sampler.materialize_task_source_balanced_rotating_indices(plan, seed=42, epoch_index=epoch)
            expected = 0
            for polarity, groups in plan["buckets"][8].items():
                count = plan["slots_by_task"][8][polarity]
                start = epoch * count
                expected += (start + count - 1) // len(groups) - start // len(groups) + 1
            self.assertEqual(len(calls), expected)
            self.assertLessEqual(len(calls), 5)
            self.assertEqual(len(draws), 6000)

    def test_workers_and_resumed_offsets_do_not_share_order_or_rng_state(self):
        plan = sampler.build_task_source_balanced_rotating_plan(make_records(8, 13, 9))
        jobs = [(42 + worker * 10000, epoch) for worker in range(16) for epoch in (0, 1, 3)]
        state = random.getstate()
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(lambda job: sampler.materialize_task_source_balanced_rotating_indices(
                plan, seed=job[0], epoch_index=job[1]), jobs))
        for (seed, epoch), values in zip(jobs, actual):
            self.assertEqual(values[17:], legacy_materialize(plan, seed, epoch)[17:])
        self.assertEqual(random.getstate(), state)

    def test_cycle_zero_visits_manual_then_rotates_crops_before_repeat(self):
        records = make_records(8, 1, 2, crops=4)
        plan = sampler.build_task_source_balanced_rotating_plan(records)
        positive_indices = [i for i, r in enumerate(records) if sampler.is_positive_ui_defect(r)]
        visits = []
        for epoch in range(4):
            values = sampler.materialize_task_source_balanced_rotating_indices(plan, seed=42, epoch_index=epoch)
            visits.extend(i for i in values if i in positive_indices)
        self.assertEqual(visits[0], next(iter(plan["manual_indices"])))
        self.assertEqual(Counter(visits), Counter(positive_indices))


if __name__ == "__main__":
    unittest.main()
