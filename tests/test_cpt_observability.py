from __future__ import annotations

import unittest

from eaglevl.train.cpt_observability import (
    add_sample_to_counter,
    empty_task_counter,
    merge_task_counters,
    reference_token_loss_aggregation,
    sample_length_metadata,
    serializable_counter,
    summarize_task_counter,
    supervision_kinds,
    validate_attempted_identity,
)


class CPTObservabilityTest(unittest.TestCase):
    def test_main_plus_mtp_equals_total_supervised_tokens(self):
        labels = [-100, 11, 12, -100, 21, 22]
        metadata = sample_length_metadata(
            labels, pre_mtp_length=4, vision_tokens=1
        )
        self.assertEqual(supervision_kinds(labels, 4), [0, 1, 1, 0, 2, 2])
        self.assertEqual(metadata["main_supervised_tokens"], 2)
        self.assertEqual(metadata["mtp_supervised_tokens"], 2)
        self.assertEqual(
            metadata["total_supervised_tokens"],
            metadata["main_supervised_tokens"] + metadata["mtp_supervised_tokens"],
        )

    def test_one_short_and_one_oversize_sample_reconcile(self):
        counter = empty_task_counter()
        short = {
            "post_mtp_seq_len": 10,
            "pre_mtp_seq_len": 6,
            "raw_text_tokens": 4,
            "vision_tokens": 2,
            "main_supervised_tokens": 3,
            "mtp_supervised_tokens": 2,
            "total_supervised_tokens": 5,
            "record_hash": 1,
            "group_hash": 2,
        }
        long = {**short, "post_mtp_seq_len": 100, "pre_mtp_oversize": False}
        add_sample_to_counter(counter, short, outcome="attempted")
        add_sample_to_counter(counter, short, outcome="accepted")
        add_sample_to_counter(counter, long, outcome="attempted")
        add_sample_to_counter(counter, long, outcome="oversize")
        add_sample_to_counter(counter, short, outcome="trained")
        validate_attempted_identity(counter)
        summary = summarize_task_counter("task", counter, dataset_rows=10)
        self.assertEqual(summary["attempted_samples"], 2)
        self.assertEqual(summary["accepted_samples"], 1)
        self.assertEqual(summary["oversize_skipped_samples"], 1)
        self.assertEqual(summary["oversize_mtp_expansion_samples"], 1)
        self.assertEqual(summary["oversize_pre_mtp_samples"], 0)
        self.assertEqual(summary["trained_samples"], 1)
        self.assertEqual(summary["unique_oversize_record_count"], 1)
        self.assertEqual(counter["oversize_record_hashes"], {1})

    def test_two_rank_merge_matches_single_process_union(self):
        counters = []
        for rank in range(2):
            counter = empty_task_counter()
            metadata = {
                "post_mtp_seq_len": 10 + rank,
                "pre_mtp_seq_len": 6,
                "raw_text_tokens": 4,
                "vision_tokens": 2,
                "main_supervised_tokens": 3,
                "mtp_supervised_tokens": 2,
                "total_supervised_tokens": 5,
                "record_hash": rank + 1,
                "group_hash": rank + 10,
            }
            add_sample_to_counter(counter, metadata, outcome="attempted")
            add_sample_to_counter(counter, metadata, outcome="accepted")
            add_sample_to_counter(counter, metadata, outcome="trained")
            counters.append(serializable_counter(counter))
        merged = merge_task_counters(counters)
        self.assertEqual(merged["trained_samples"], 2)
        self.assertEqual(merged["main_supervised_tokens"], 6)
        self.assertEqual(merged["unique_record_hashes"], {1, 2})
        self.assertEqual(merged["post_mtp_length_histogram"], {10: 1, 11: 1})

    def test_two_tasks_in_one_packed_batch_match_manual_shifted_ce(self):
        values = reference_token_loss_aggregation(
            losses=[9.0, 1.0, 3.0, 7.0, 2.0],
            labels=[-100, 11, 12, -100, 21, 22],
            task_ids=[0, 0, 0, 1, 1, 1],
            kinds=[0, 1, 2, 0, 1, 2],
        )
        self.assertEqual(values[0]["main_loss_sum"], 9.0)
        self.assertEqual(values[0]["mtp_loss_sum"], 1.0)
        self.assertEqual(values[1]["main_loss_sum"], 7.0)
        self.assertEqual(values[1]["mtp_loss_sum"], 2.0)

    def test_supervised_token_without_kind_is_not_silently_dropped(self):
        with self.assertRaisesRegex(RuntimeError, "alignment"):
            reference_token_loss_aggregation(
                losses=[1.0],
                labels=[-100, 11],
                task_ids=[0, 0],
                kinds=[0, 0],
            )

    def test_counter_round_trip_resume_is_monotonic_without_duplicate_unique(self):
        counter = empty_task_counter()
        metadata = {
            "post_mtp_seq_len": 10,
            "pre_mtp_seq_len": 6,
            "raw_text_tokens": 4,
            "vision_tokens": 2,
            "main_supervised_tokens": 3,
            "mtp_supervised_tokens": 2,
            "total_supervised_tokens": 5,
            "record_hash": 7,
            "group_hash": 9,
        }
        add_sample_to_counter(counter, metadata, outcome="attempted")
        add_sample_to_counter(counter, metadata, outcome="accepted")
        add_sample_to_counter(counter, metadata, outcome="trained")
        resumed = merge_task_counters([serializable_counter(counter)])
        add_sample_to_counter(resumed, metadata, outcome="attempted")
        add_sample_to_counter(resumed, metadata, outcome="accepted")
        add_sample_to_counter(resumed, metadata, outcome="trained")
        self.assertEqual(resumed["trained_samples"], 2)
        self.assertEqual(resumed["unique_record_hashes"], {7})
        self.assertEqual(resumed["total_supervised_tokens"], 10)


if __name__ == "__main__":
    unittest.main()
