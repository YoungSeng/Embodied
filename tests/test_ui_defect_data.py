import unittest

from eaglevl.train.ui_defect_data import (
    TASK_SPECS,
    build_balanced_ui_indices,
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

    def test_explicit_route_is_validated_against_fixed_table(self):
        self.assertEqual(
            identify_ui_defect_task({"defect_type": 2, "relation_family": 1}),
            ("overlap", 2, 1),
        )
        self.assertEqual(
            identify_ui_defect_task(
                {"defect_type": "element_cropping", "relation_family": 0}
            ),
            ("cropping", 1, 0),
        )
        with self.assertRaisesRegex(ValueError, "Unknown explicit"):
            identify_ui_defect_task({"defect_type": -1, "relation_family": -1})
        with self.assertRaisesRegex(ValueError, "disagrees"):
            identify_ui_defect_task({"defect_type": 2, "relation_family": 0})
        with self.assertRaisesRegex(ValueError, "provide both"):
            identify_ui_defect_task({"defect_type": 2})


if __name__ == "__main__":
    unittest.main()
