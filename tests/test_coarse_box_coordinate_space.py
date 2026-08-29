import sys
import unittest
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from collect_ui5_metrics import coarse_boxes_px_from_sidecar, norm1000_box_to_pixel


def iou(left, right):
    intersection = max(0.0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0.0, min(left[3], right[3]) - max(left[1], right[1])
    )
    areas = [(box[2] - box[0]) * (box[3] - box[1]) for box in (left, right)]
    return intersection / (sum(areas) - intersection)


class CoarseBoxCoordinateSpaceTest(unittest.TestCase):
    def test_square_conversion_is_identity(self):
        self.assertEqual(norm1000_box_to_pixel([10, 20, 300, 400], 1000, 1000), [10, 20, 300, 400])

    def test_non_square_conversion_and_iou_invariance(self):
        left = [100, 100, 500, 500]
        right = [200, 200, 600, 600]
        left_px = norm1000_box_to_pixel(left, 2000, 1000)
        right_px = norm1000_box_to_pixel(right, 2000, 1000)
        self.assertEqual(left_px, [200, 100, 1000, 500])
        self.assertAlmostEqual(iou(left, right), iou(left_px, right_px), places=7)

    def test_sidecar_requires_explicit_coordinate_metadata(self):
        with self.assertRaisesRegex(RuntimeError, "coordinate_space"):
            coarse_boxes_px_from_sidecar({"coarse_boxes_norm1000": [[0, 0, 10, 10]]})
        with self.assertRaisesRegex(RuntimeError, "migrate"):
            coarse_boxes_px_from_sidecar({"coarse_boxes": [[0, 0, 10, 10]]})
        with self.assertRaisesRegex(RuntimeError, "both coarse_boxes"):
            coarse_boxes_px_from_sidecar(
                {
                    "coordinate_space": "norm1000",
                    "image_width": 1000,
                    "image_height": 1000,
                    "coarse_boxes_px": [[0, 0, 10, 10]],
                }
            )

    def test_selected_slot_iou_cannot_exceed_oracle(self):
        target = [0, 0, 10, 10]
        all_slots = [[0, 0, 10, 10], [20, 20, 30, 30]]
        selected = [all_slots[1]]
        self.assertLessEqual(max(iou(target, value) for value in selected), max(iou(target, value) for value in all_slots))


if __name__ == "__main__":
    unittest.main()
