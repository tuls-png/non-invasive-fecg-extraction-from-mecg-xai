import tempfile
import unittest
from pathlib import Path

from run_experiment_new import build_tuning_candidates, write_tuned_config


class HyperparameterTuningTests(unittest.TestCase):
    def test_build_tuning_candidates_returns_multiple_configs(self):
        candidates = build_tuning_candidates("cinc2013")
        self.assertGreaterEqual(len(candidates), 3)
        for cfg in candidates:
            self.assertIn("WSVD_COMPONENT_CORR_THRESH", cfg)
            self.assertIn("PT_THRESHOLD_FACTOR", cfg)

    def test_write_tuned_config_creates_yaml(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "cinc2013_tuned.yaml"
            config = {
                "WSVD_COMPONENT_CORR_THRESH": 0.18,
                "PT_THRESHOLD_FACTOR": 0.40,
            }
            write_tuned_config(config, out_path)
            self.assertTrue(out_path.exists())
            text = out_path.read_text(encoding="utf-8")
            self.assertIn("WSVD_COMPONENT_CORR_THRESH", text)
            self.assertIn("PT_THRESHOLD_FACTOR", text)


if __name__ == "__main__":
    unittest.main()
