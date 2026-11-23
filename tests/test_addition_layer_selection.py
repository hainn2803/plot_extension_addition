"""CPU-only regression tests for calibration selection and bookkeeping."""

import tempfile
import unittest
from pathlib import Path

from addition_select_promising_layers import select_gradual, select_standard
from addition_pipeline_utils import parse_layers, read_json, write_json, write_csv


class LayerSelectionTests(unittest.TestCase):
    def test_standard_uses_absolute_percentage_points(self):
        rows = [
            {"variable": "T", "layer": "23", "cal_iia": "0.95"},
            {"variable": "T", "layer": "24", "cal_iia": "0.90"},
            {"variable": "T", "layer": "25", "cal_iia": "0.899"},
            {"variable": "C", "layer": "0", "cal_iia": "0.97"},
            {"variable": "C", "layer": "1", "cal_iia": "0.94"},
            {"variable": "C", "layer": "2", "cal_iia": "0.91"},
        ]
        selected = select_standard(rows, 5.0)
        self.assertEqual(selected["layers"]["T"], [23, 24])
        self.assertEqual(selected["layers"]["C"], [0, 1])
        self.assertEqual(len(selected["pairs"]), 4)

    def test_invalid_order_is_excluded(self):
        rows = [
            {"variable": "T", "layer": "3", "cal_iia": "0.9"},
            {"variable": "C", "layer": "4", "cal_iia": "0.9"},
        ]
        self.assertEqual(select_standard(rows, 5.0)["pairs"], [])

    def test_gradual_c_is_conditional_on_t(self):
        rows = [
            {"T_layer": "23", "C_layer": "2", "T_cal_iia": "0.95", "C_cal_iia": "0.90"},
            {"T_layer": "23", "C_layer": "4", "T_cal_iia": "0.95", "C_cal_iia": "0.84"},
            {"T_layer": "24", "C_layer": "1", "T_cal_iia": "0.92", "C_cal_iia": "0.96"},
            {"T_layer": "24", "C_layer": "3", "T_cal_iia": "0.92", "C_cal_iia": "0.91"},
            {"T_layer": "25", "C_layer": "4", "T_cal_iia": "0.89", "C_cal_iia": "0.99"},
        ]
        chosen = select_gradual(rows, 5.0)
        self.assertEqual(
            chosen["pairs"],
            [
                {"C_layer": 2, "T_layer": 23},
                {"C_layer": 1, "T_layer": 24},
                {"C_layer": 3, "T_layer": 24},
            ],
        )

    def test_parse_layers(self):
        self.assertEqual(parse_layers("all", 4), [0, 1, 2, 3])
        self.assertEqual(parse_layers("2,0:2,2", 5), [0, 1, 2])
        with self.assertRaises(ValueError):
            parse_layers("6", 5)

    def test_atomic_save(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "a.json"
            write_json(out, {"method": "standard", "pairs": []})
            self.assertEqual(read_json(out)["method"], "standard")
            csv_path = Path(tmp) / "b.csv"
            write_csv(csv_path, [{"C_layer": 1, "T_layer": 4}])
            self.assertIn("C_layer", csv_path.read_text())


if __name__ == "__main__":
    unittest.main()
