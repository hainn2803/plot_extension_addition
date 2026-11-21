"""Profile isolation for independent Addition evaluation datasets and seeds."""

import tempfile
import unittest
from pathlib import Path

from addition_eval_profile import profile_directory, profile_name, profile_settings
from addition_eval_reports import config_dir, import_legacy, rebuild_summary
from addition_pipeline_utils import read_json, write_json


class EvalProfileTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.dataset_a = {"raw": {"test": "aaa"}, "filtered": {"test_C_only": "111"}}
        self.dataset_b = {"raw": {"test": "bbb"}, "filtered": {"test_C_only": "222"}}

    def settings(self, data, seed=0):
        return profile_settings(
            data, seed=seed, eval_batch_size=32, max_new_tokens=4,
            generation_cap=8, recipient_version=1,
        )

    def test_dataset_and_seed_produce_distinct_profiles(self):
        a = self.settings(self.dataset_a)
        b = self.settings(self.dataset_b)
        a_other_seed = self.settings(self.dataset_a, seed=1)
        self.assertEqual(profile_name(a), profile_name(self.settings(self.dataset_a)))
        names = {profile_name(a), profile_name(b), profile_name(a_other_seed)}
        self.assertEqual(len(names), 3)
        for name in names:
            self.assertRegex(name, r"^eval_configs_[0-9a-f]{10}$")

    def test_sdas_gdas_share_identical_profile_name_for_same_dataset(self):
        # Method is intentionally not part of the hash because the two runs
        # should be directly comparable with the same reader.
        settings = self.settings(self.dataset_a)
        sd = profile_directory(self.root / "sdas", settings)
        gd = profile_directory(self.root / "gdas", settings)
        self.assertEqual(sd.name, gd.name)
        self.assertEqual(read_json(sd / "eval_settings.json"), settings)
        self.assertEqual(config_dir(sd, 16, 18).name, "T_L18__C_L16")

    def test_import_does_not_cross_datasets_or_seeds(self):
        old = self.root / "evaluation_standard" / "C_16_T_18" / "metrics.json"
        write_json(old, {
            "method": "standard", "C_layer": 16, "T_layer": 18,
            "recipient_seed": 0,
            "signature": {"data": self.dataset_a},
        })
        pairs = [(16, 18)]
        a_dir = profile_directory(self.root, self.settings(self.dataset_a))
        b_dir = profile_directory(self.root, self.settings(self.dataset_b))
        seeded_dir = profile_directory(self.root, self.settings(self.dataset_a, seed=1))
        self.assertEqual(import_legacy(self.root, "standard", pairs, self.dataset_a, a_dir, 0), 1)
        self.assertTrue((config_dir(a_dir, 16, 18) / "metrics.json").exists())
        self.assertEqual(import_legacy(self.root, "standard", pairs, self.dataset_b, b_dir, 0), 0)
        self.assertEqual(import_legacy(self.root, "standard", pairs, self.dataset_a, seeded_dir, 1), 0)
        self.assertFalse((config_dir(b_dir, 16, 18) / "metrics.json").exists())
        self.assertFalse((config_dir(seeded_dir, 16, 18) / "metrics.json").exists())


if __name__ == "__main__":
    unittest.main()
