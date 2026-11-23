"""CPU-only tests of different-question recipient sampling and targets."""

import unittest

from addition_recipient_schedule import (
    make_schedule,
    recipient_pool,
    target_for_recipient,
)


class RecipientScheduleTests(unittest.TestCase):
    def setUp(self):
        self.pool = [
            {"prompt": "rec_t4_c0", "C": 0, "T": 4, "ones": 5, "Y": 45, "hundreds": 0},
            {"prompt": "rec_t4_c1", "C": 1, "T": 4, "ones": 6, "Y": 46, "hundreds": 0},
            {"prompt": "rec_t3_c0", "C": 0, "T": 3, "ones": 7, "Y": 37, "hundreds": 0},
        ]
        self.donor = {
            "pair_type": "C_only", "base_input": "donor_base",
            "source_input": "donor_source", "base_ones_out": 2,
            "source_C": 1, "T_after_C_intervene": 4,
        }

    def test_t_replay_correct_target_for_new_context(self):
        for condition in ("same", "different"):
            schedule = make_schedule([self.donor], self.pool, "T", condition, 7)
            entry = schedule[0]
            recipient = entry["recipient"]
            self.assertNotEqual(recipient["ones"], self.donor["base_ones_out"])
            self.assertEqual(recipient["T"] == 4, condition == "same")
            self.assertEqual(entry["target_T"], 4)
            self.assertEqual(entry["target_Y"], 40 + recipient["ones"])

    def test_c_direct_transfer_targets_recipient_tens_sum(self):
        for condition in ("same", "different"):
            schedule = make_schedule([self.donor], self.pool, "C", condition, 12)
            entry = schedule[0]
            recipient = entry["recipient"]
            expected_tens = recipient["T"] - recipient["C"] + 1
            self.assertEqual(recipient["C"] == 1, condition == "same")
            self.assertEqual(entry["target_T"], expected_tens)
            self.assertEqual(entry["target_Y"], 10 * expected_tens + recipient["ones"])

    def test_seed_is_stable_and_original_question_excluded(self):
        a = make_schedule([self.donor], self.pool, "T", "same", 18)
        b = make_schedule([self.donor], self.pool, "T", "same", 18)
        self.assertEqual(a, b)
        donor = self.donor.copy()
        donor["base_input"] = "rec_t4_c0"
        donor["source_input"] = "rec_t4_c1"
        with self.assertRaises(RuntimeError):
            make_schedule([donor], self.pool, "T", "same", 18)

    def test_invalid_recipient_is_rejected(self):
        bad = self.pool[0].copy()
        bad["T"] = 9
        bad["C"] = 0
        with self.assertRaises(ValueError):
            target_for_recipient(bad, "C", 1)

    def test_factual_pool_requires_model_correct_consistent_rows(self):
        pair = {
            "base_input": "a", "base_C": 0, "base_T": 4,
            "base_ones_out": 5, "base_Y": 45, "base_hundreds_out": 0,
            "source_input": "b", "source_C": 1, "source_T": 4,
            "source_ones_out": 6, "source_Y": 46, "source_hundreds_out": 0,
        }
        test_rows = {"T_only": [pair], "C_only": [], "C_T": []}
        pool = recipient_pool(test_rows)
        self.assertEqual(len(pool), 2)
        self.assertEqual({p["prompt"] for p in pool}, {"a", "b"})
        pair["source_Y"] = 99
        with self.assertRaises(ValueError):
            recipient_pool(test_rows)


if __name__ == "__main__":
    unittest.main()
