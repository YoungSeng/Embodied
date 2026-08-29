import unittest

from eaglevl.train.ui_defect_data import (
    TASK_SPECS,
    build_balanced_ui_indices,
    build_task_balanced_all_records_indices,
    extract_ui_defect_targets,
    identify_ui_defect_task,
)


def make_record(label: str, positive: bool, box_count: int = 1) -> dict:
    answer = "<box>none</box>"
    if positive:
        answer = f"<ref>{label}</ref>" + "".join(
            f"<box><{10 + i}><20><{30 + i}><40></box>" for i in range(box_count)
        )
    return {
        "conversations": [
            {
                "from": "human",
                "value": (
                    "Locate all the instances that match the following description: "
                    f"{label}."
                ),
            },
            {"from": "gpt", "value": answer},
        ],
        "image": "dummy.png",
    }


class UIDefectDataTest(unittest.TestCase):
    def test_five_task_mapping_and_boxes(self):
        expected_families = [0, 0, 1, 2, 3]
        for spec, expected_family in zip(TASK_SPECS, expected_families):
            task_name, defect_type, relation_family, aliases = spec
            record = make_record(aliases[0], positive=True, box_count=2)
            self.assertEqual(
                identify_ui_defect_task(record),
                (task_name, defect_type, relation_family),
            )
            self.assertEqual(relation_family, expected_family)
            targets = extract_ui_defect_targets(record, max_boxes=8)
            self.assertEqual(targets["target_boxes"].shape, (1, 8, 4))
            self.assertEqual(int(targets["target_box_mask"].sum()), 2)

    def test_balancing_keeps_fixed_class_size_and_one_to_two_ratio(self):
        records = []
        for _, _, _, aliases in TASK_SPECS:
            # Deliberately make positives rare; the sampler must use replacement.
            records.append(make_record(aliases[0], positive=True))
            records.extend(make_record(aliases[0], positive=False) for _ in range(9))

        indices = build_balanced_ui_indices(
            records,
            records_per_class=12,
            negative_to_positive_ratio=2.0,
            seed=7,
        )
        self.assertEqual(len(indices), 60)
        counts = {defect_type: [0, 0] for defect_type in range(5)}
        for index in indices:
            record = records[index]
            defect_type = identify_ui_defect_task(record)[1]
            positive = bool(extract_ui_defect_targets(record)["target_box_mask"].any())
            counts[defect_type][int(positive)] += 1
        self.assertTrue(all(value == [8, 4] for value in counts.values()))

    def test_all_106_manual_repair_gt_records_survive_balancing(self):
        # TASK_SPECS order is occlusion, cropping, text overflow, ellipsis,
        # content missing; these are the audited v4 repair counts.
        repair_counts = (46, 23, 2, 35, 0)
        records = []
        expected_repair_keys = set()
        sequence = 0
        for (_, _, _, aliases), repair_count in zip(TASK_SPECS, repair_counts):
            for _ in range(repair_count):
                sample_id = f"sample_repair_{sequence}"
                record = make_record(aliases[0], positive=True)
                record.update(
                    {
                        "_ui5_sample_id": sample_id,
                        "_ui5_crop_source": "manual_gt_repair",
                        "_ui5_manual_repair_gt_indices": [0],
                    }
                )
                records.append(record)
                expected_repair_keys.add((sample_id, 0))
                sequence += 1
            records.extend(make_record(aliases[0], positive=True) for _ in range(10))
            records.extend(make_record(aliases[0], positive=False) for _ in range(10))

        self.assertEqual(len(expected_repair_keys), 106)
        indices = build_balanced_ui_indices(
            records,
            records_per_class=180,
            negative_to_positive_ratio=2.0,
            seed=7,
        )
        self.assertEqual(len(indices), 5 * 180)
        selected_repair_keys = {
            (str(record["_ui5_sample_id"]), int(gt_index))
            for index in indices
            if (record := records[index]).get("_ui5_crop_source")
            == "manual_gt_repair"
            for gt_index in record["_ui5_manual_repair_gt_indices"]
        }
        self.assertEqual(selected_repair_keys, expected_repair_keys)

    def test_manual_repair_count_larger_than_bucket_quota_fails_closed(self):
        records = []
        for task_index, (_, _, _, aliases) in enumerate(TASK_SPECS):
            required = 5 if task_index == 0 else 0
            for sequence in range(required):
                record = make_record(aliases[0], positive=True)
                record.update(
                    {
                        "_ui5_sample_id": f"required_{sequence}",
                        "_ui5_crop_source": "manual_gt_repair",
                        "_ui5_manual_repair_gt_indices": [0],
                    }
                )
                records.append(record)
            records.extend(make_record(aliases[0], positive=True) for _ in range(2))
            records.extend(make_record(aliases[0], positive=False) for _ in range(2))

        with self.assertRaisesRegex(ValueError, "quota.*manual_gt_repair"):
            build_balanced_ui_indices(
                records,
                records_per_class=12,
                negative_to_positive_ratio=2.0,
                seed=7,
            )

    def test_task_balanced_all_records_drops_no_ordinary_or_repair_crop(self):
        records = []
        for task_index, (_, _, _, aliases) in enumerate(TASK_SPECS):
            for sequence in range(task_index + 2):
                record = make_record(aliases[0], positive=sequence % 3 == 0)
                record.update(
                    {
                        "_ui5_sample_id": f"task{task_index}_{sequence}",
                        "_ui5_crop_source": (
                            "manual_gt_repair" if sequence == 0 else "raw_detector_strip"
                        ),
                        "_ui5_manual_repair_gt_indices": [0] if sequence == 0 else [],
                    }
                )
                records.append(record)
        indices = build_task_balanced_all_records_indices(records, seed=17)
        self.assertEqual(set(indices), set(range(len(records))))
        self.assertEqual(len(indices), 5 * max(range(2, 7)))
        repair_indices = {
            index for index, record in enumerate(records)
            if record["_ui5_crop_source"] == "manual_gt_repair"
        }
        self.assertTrue(repair_indices.issubset(indices))

    def test_all_106_repair_gt_records_survive_all_record_sampling(self):
        repair_counts = (46, 23, 2, 35, 0)
        records = []
        expected = set()
        sequence = 0
        for (_, _, _, aliases), repair_count in zip(TASK_SPECS, repair_counts):
            for _ in range(repair_count):
                sample_id = f"all_record_repair_{sequence}"
                record = make_record(aliases[0], positive=True)
                record.update(
                    {
                        "_ui5_sample_id": sample_id,
                        "_ui5_crop_source": "manual_gt_repair",
                        "_ui5_manual_repair_gt_indices": [0],
                    }
                )
                records.append(record)
                expected.add((sample_id, 0))
                sequence += 1
            records.extend(make_record(aliases[0], positive=False) for _ in range(50))
        indices = build_task_balanced_all_records_indices(records, seed=37)
        actual = {
            (str(record["_ui5_sample_id"]), int(gt_index))
            for index in indices
            if (record := records[index]).get("_ui5_crop_source")
            == "manual_gt_repair"
            for gt_index in record["_ui5_manual_repair_gt_indices"]
        }
        self.assertEqual(len(expected), 106)
        self.assertEqual(actual, expected)
        self.assertEqual(set(indices), set(range(len(records))))

    def test_short_task_stream_is_complete_before_repetition(self):
        records = []
        task_ranges = {}
        for task_index, (_, _, _, aliases) in enumerate(TASK_SPECS):
            start = len(records)
            count = 2 if task_index == 0 else 5
            records.extend(make_record(aliases[0], positive=False) for _ in range(count))
            task_ranges[task_index] = set(range(start, start + count))
        indices = build_task_balanced_all_records_indices(records, seed=23)
        first_task_stream = [
            index for index in indices
            if identify_ui_defect_task(records[index])[1] == 0
        ]
        self.assertEqual(set(first_task_stream[:2]), task_ranges[0])

    def test_task_balanced_all_records_resume_sequence_is_deterministic(self):
        records = []
        for _, _, _, aliases in TASK_SPECS:
            records.extend(make_record(aliases[0], positive=False) for _ in range(7))
        first = build_task_balanced_all_records_indices(records, seed=101)
        resumed = build_task_balanced_all_records_indices(records, seed=101)
        self.assertEqual(first, resumed)
        self.assertEqual(first[13:], resumed[13:])


if __name__ == "__main__":
    unittest.main()

