"""CPU-only checks for portable GDAS results and strict T provenance."""

import tempfile
import unittest
from pathlib import Path

from addition_gradual_snapshot import (
    TRAINING_SELECTION_NAME,
    artifact_hashes_by_name,
    freeze_t_snapshots,
    validate_gradual_pair_snapshot,
)
from addition_pipeline_utils import file_hash, gradual_paths, handle_paths, read_json, write_json


class GradualSnapshotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.standard = root / "sdas"
        self.gradual = root / "gdas"
        self.gradual.mkdir()
        self.c_layer = 16
        self.t_layer = 18
        write_json(self.gradual / "run_manifest.json", {
            "method": "gradual",
            "standard_root": str(self.standard.resolve()),
        })
        self.source = handle_paths(self.standard, "T", self.t_layer)
        for i, path in enumerate(self.source):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"frozen-original-T-" + bytes([i]))
        c_path, _ = gradual_paths(self.gradual, self.c_layer, self.t_layer)
        write_json(c_path.parent / "train_signature.json", {
            "C_layer": self.c_layer, "T_layer": self.t_layer,
            "upstream_artifacts": {str(p): file_hash(p) for p in self.source},
        })
        self.pairs = [(self.c_layer, self.t_layer)]

    def test_snapshots_are_local_and_verified(self):
        freeze_t_snapshots(self.gradual, self.standard, self.pairs)
        frozen = handle_paths(self.gradual, "T", self.t_layer)
        for original, snapshot in zip(self.source, frozen):
            self.assertEqual(original.read_bytes(), snapshot.read_bytes())
        self.assertEqual(len(read_json(self.gradual / TRAINING_SELECTION_NAME)["pairs"]), 1)
        validate_gradual_pair_snapshot(self.gradual, self.c_layer, self.t_layer)
        # Resuming does not mutate the frozen snapshots.
        freeze_t_snapshots(self.gradual, self.standard, self.pairs)
        self.assertEqual(file_hash(frozen[0]), file_hash(self.source[0]))

    def test_checkpoint_change_is_rejected(self):
        freeze_t_snapshots(self.gradual, self.standard, self.pairs)
        frozen = handle_paths(self.gradual, "T", self.t_layer)
        frozen[0].write_bytes(b"tampered-local-handle")
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            validate_gradual_pair_snapshot(self.gradual, self.c_layer, self.t_layer)
        with self.assertRaisesRegex(RuntimeError, "differs"):
            freeze_t_snapshots(self.gradual, self.standard, self.pairs)

    def test_source_change_cannot_backfill_old_gdas_c(self):
        self.source[1].write_bytes(b"changed-frozen-readout")
        with self.assertRaisesRegex(RuntimeError, "mismatch"):
            freeze_t_snapshots(self.gradual, self.standard, self.pairs)

    def test_training_pairs_are_immutable(self):
        freeze_t_snapshots(self.gradual, self.standard, self.pairs)
        # The pair selection cannot be silently replaced in a resumed run.
        with self.assertRaisesRegex(RuntimeError, "selection changed"):
            freeze_t_snapshots(self.gradual, self.standard, [(17, 18)])

    def test_compare_by_basename_ignores_old_absolute_paths(self):
        old = {
            "/some/other/sdas/T_handles/T_layer_18/das_t.pt": "aaa",
            "/some/other/sdas/T_handles/T_layer_18/t_readout_LogisticRegression.pt": "bbb",
        }
        new = {
            "/local/gdas/T_handles/T_layer_18/das_t.pt": "aaa",
            "/local/gdas/T_handles/T_layer_18/t_readout_LogisticRegression.pt": "bbb",
        }
        self.assertEqual(artifact_hashes_by_name(old), artifact_hashes_by_name(new))


if __name__ == "__main__":
    unittest.main()
