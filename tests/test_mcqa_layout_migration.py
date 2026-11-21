"""CPU-only integration tests: MCQA checkpoint/report naming and old-run migration."""

import tempfile
import unittest
from pathlib import Path

from addition_eval_reports import (
    config_dir, detail_rows, flatten,
)
from addition_eval_profile import profile_name, profile_settings
from addition_recipient_schedule import RECIPIENT_VERSION
from addition_gradual_snapshot import TRAINING_SELECTION_NAME, validate_gradual_pair_snapshot
from addition_migrate_mcqa_layout import migrate_standard, migrate_gradual
from addition_pipeline_utils import (
    file_hash, gradual_paths, handle_paths, read_json, write_json,
)


def fake_payload(method, c, t, data_sig):
    return {
        "method": method, "C_layer": c, "T_layer": t,
        "calibration": {"C_cal_iia": 1.0, "T_cal_iia": .97},
        "clean": {"C_base": 1.0, "C_source": 1.0,
                  "T_base": .93, "T_source": .94},
        "metrics": {
            "T_only": {"Y_after_T": .96, "T_after_T": .93},
            "C_only": {"Y_after_C": .99, "T_after_C": .42,
                       "Y_via_T_replay": .68, "Y_restored_via_T": .72},
            "C_T": {"Y_after_C": 1.0, "T_after_C": .57,
                    "Y_via_T_replay": .70},
        },
        "recipient": {
            "T_only": {"Y_via_T_replay_recipient_different": .75},
            "C_only": {"Y_via_T_replay_recipient_different": .78},
            "C_T": {"Y_via_T_replay_recipient_same": .92},
        },
        "signature": {"data": data_sig, "artifacts": {}},
    }


class McqaLayoutTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.standard = root / "standard"
        self.gradual = root / "gradual"
        self.c, self.t = 16, 18
        self.data = {"filtered": {"test": "sha"}}
        self.pair = {"C_layer": self.c, "T_layer": self.t}
        self.sdas_selection = self.standard / "selected_layers_standard.json"
        write_json(self.standard / "run_manifest.json", {
            "method": "standard", "data": self.data,
        })
        write_json(self.gradual / "run_manifest.json", {
            "method": "gradual", "data": self.data,
            "standard_root": str(self.standard.resolve()),
        })

        self.old_t_paths = self.make_old_handle(self.standard, "T", self.t)
        self.old_c_paths = self.make_old_handle(self.standard, "C", self.c)
        t_sig = self.old_cal_signature("T", self.t, self.old_t_paths)
        c_sig = self.old_cal_signature("C", self.c, self.old_c_paths)
        for sig in (t_sig, c_sig):
            write_json(
                self.standard / "calibration_standard" /
                f"{sig['variable']}_layer_{sig['layer']:02d}.json",
                {"cal_iia": .99, "signature": sig},
            )
        write_json(self.sdas_selection, {
            "method": "standard", "pairs": [self.pair],
            "calibration_signatures": {
                "T_L18": t_sig, "C_L16": c_sig,
            },
        })

        old_gradual_c = self.gradual / "pairs" / "C_16_T_18"
        old_gradual_c.mkdir(parents=True)
        (old_gradual_c / "das_c.pt").write_bytes(b"gradual-c-handle")
        (old_gradual_c / "c_readout_LogisticRegression.pt").write_bytes(b"gradual-c-readout")
        write_json(old_gradual_c / "train_signature.json", {
            "C_layer": self.c, "T_layer": self.t,
            "upstream_artifacts": {
                str(path): file_hash(path) for path in self.old_t_paths
            },
        })

        for method, root in (("standard", self.standard), ("gradual", self.gradual)):
            old = root / f"evaluation_{method}" / "C_16_T_18" / "metrics.json"
            write_json(old, fake_payload(method, self.c, self.t, self.data))

    def make_old_handle(self, root, variable, layer):
        folder = root / f"{variable}_handles" / f"{variable}_layer_{layer:02d}"
        folder.mkdir(parents=True)
        lower = variable.lower()
        paths = (
            folder / f"das_{lower}.pt",
            folder / f"{lower}_readout_LogisticRegression.pt",
        )
        for i, path in enumerate(paths):
            path.write_bytes(f"{variable}-L{layer}-{i}".encode())
        return paths

    def old_cal_signature(self, variable, layer, paths):
        return {
            "split": "validation", "variable": variable,
            "layer": layer, "generation_cap": 8,
            "artifacts": {str(path): file_hash(path) for path in paths},
        }

    def test_standard_and_gradual_move_to_same_layout(self):
        migrate_standard(self.standard)
        migrate_gradual(self.gradual, self.standard)
        self.assertFalse((self.standard / "C_handles").exists())
        self.assertFalse((self.standard / "T_handles").exists())
        self.assertFalse((self.gradual / "pairs").exists())
        self.assertFalse((self.gradual / "T_handles").exists())

        c_checkpoint, _ = handle_paths(self.standard, "C", self.c)
        t_checkpoint, _ = handle_paths(self.standard, "T", self.t)
        gdas_c_checkpoint, _ = gradual_paths(self.gradual, self.c, self.t)
        gdas_t_checkpoint, _ = handle_paths(self.gradual, "T", self.t)
        self.assertTrue(c_checkpoint.is_file())
        self.assertTrue(t_checkpoint.is_file())
        self.assertTrue(gdas_c_checkpoint.is_file())
        self.assertTrue(gdas_t_checkpoint.is_file())
        self.assertEqual(file_hash(t_checkpoint), file_hash(gdas_t_checkpoint))
        validate_gradual_pair_snapshot(self.gradual, self.c, self.t)

        settings = profile_settings(
            self.data, seed=0, eval_batch_size=32, max_new_tokens=4,
            generation_cap=8, recipient_version=RECIPIENT_VERSION,
        )
        profile = profile_name(settings)
        self.assertTrue(profile.startswith("eval_configs_"))
        self.assertEqual(
            config_dir(self.standard / profile, self.c, self.t).name,
            "T_L18__C_L16",
        )
        self.assertEqual(
            config_dir(self.gradual / profile, self.c, self.t).name,
            "T_L18__C_L16",
        )
        for root in (self.standard, self.gradual):
            eval_dir = root / profile
            for path in (
                eval_dir / "T_L18__C_L16" / "metrics.json",
                eval_dir / "T_L18__C_L16" / "eval_faithfulness.txt",
                eval_dir / "T_L18__C_L16" / "eval_faithfulness.csv",
                eval_dir / "config_summary.csv",
                eval_dir / "all_config_metrics.csv",
                eval_dir / "eval_settings.json",
            ):
                self.assertTrue(path.exists(), f"Missing {path}")
        self.assertEqual(
            read_json(self.gradual / TRAINING_SELECTION_NAME)["pairs"],
            [self.pair],
        )
        selection = read_json(self.sdas_selection)
        signature = selection["calibration_signatures"]["C_L16"]
        new_paths = handle_paths(self.standard, "C", self.c)
        self.assertEqual(
            signature["artifacts"],
            {str(path): file_hash(path) for path in new_paths},
        )
        # Safe to run again and preserve the full result.
        migrate_standard(self.standard)
        migrate_gradual(self.gradual, self.standard)

    def test_reports_categorize_and_flatten_recipient_metrics(self):
        payload = fake_payload("standard", self.c, self.t, self.data)
        rows = detail_rows(payload)
        self.assertTrue(any(row["family"] == "recipient" for row in rows))
        self.assertTrue(any(row["family"] == "restoration" for row in rows))
        self.assertTrue(any(row["family"] == "mediator" for row in rows))
        self.assertEqual(
            flatten(payload)["C_only_Y_via_T_replay_recipient_different"], .78
        )


if __name__ == "__main__":
    unittest.main()
