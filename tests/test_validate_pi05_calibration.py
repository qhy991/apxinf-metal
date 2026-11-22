import json
import pathlib
import tempfile
import unittest

import numpy as np

from scripts import validate_pi05_calibration


class ValidatePi05CalibrationTest(unittest.TestCase):
    def test_timing_percentiles_interpolate_between_samples(self):
        result = validate_pi05_calibration._stats([0.0, 10.0, 20.0, 30.0])

        self.assertEqual(result["p50"], 15.0)
        self.assertAlmostEqual(result["p95"], 28.5)

    def test_manifest_coverage_and_reproducibility_ignore_only_environment(self):
        manifest = {
            "schema": "apxinf.pi05.fp8-calibration.v1",
            "calibration_data": {"production": True, "sample_count": 2},
            "plan": {"sites": ["a", "b"]},
            "observed_sites": ["a", "b"],
            "scales": {"a": {"amax": 1.0}, "b": {"amax": 2.0}},
            "device": {"requested": "cuda:0", "host": "thor-run-1"},
        }
        repeated = json.loads(json.dumps(manifest))
        repeated["device"]["host"] = "thor-run-2"

        coverage = validate_pi05_calibration.validate_manifest(manifest)
        reproducibility = validate_pi05_calibration.compare_manifests(
            [manifest, repeated]
        )

        self.assertEqual(coverage["required"], 2)
        self.assertEqual(coverage["observed"], 2)
        self.assertEqual(coverage["unknown"], [])
        self.assertEqual(coverage["unused"], [])
        self.assertTrue(reproducibility["equivalent"])

    def test_manifest_rejects_unknown_and_unused_sites(self):
        manifest = {
            "schema": "apxinf.pi05.fp8-calibration.v1",
            "calibration_data": {"production": True, "sample_count": 1},
            "plan": {"sites": ["required", "missing"]},
            "observed_sites": ["required", "unknown"],
            "scales": {"required": {"amax": 1.0}, "unused": {"amax": 2.0}},
        }
        with self.assertRaisesRegex(ValueError, "missing=.*unknown=.*unused="):
            validate_pi05_calibration.validate_manifest(manifest)

    def test_business_output_metrics_use_independent_reference_values(self):
        reference = [np.asarray([[1.0, -2.0], [3.0, 4.0]], dtype=np.float32)]
        candidate = [np.asarray([[1.5, -1.0], [2.0, 4.0]], dtype=np.float32)]

        result = validate_pi05_calibration.summarize_errors(reference, candidate)

        self.assertAlmostEqual(result["max_abs"], 1.0)
        self.assertAlmostEqual(result["mean_abs"], 0.625)
        self.assertAlmostEqual(result["rmse"], 0.75)
        self.assertAlmostEqual(result["relative_l2"], 1.5 / np.sqrt(30.0))
        self.assertEqual(result["non_finite"], 0)

    def test_report_gate_uses_explicit_relative_l2_threshold(self):
        metrics = {"relative_l2": 0.09, "non_finite": 0}
        self.assertTrue(validate_pi05_calibration.accuracy_gate(metrics, 0.1))
        self.assertFalse(validate_pi05_calibration.accuracy_gate(metrics, 0.05))
        self.assertFalse(
            validate_pi05_calibration.accuracy_gate(
                {"relative_l2": 0.0, "non_finite": 1}, 0.1
            )
        )

    def test_validator_revision_can_be_supplied_outside_git(self):
        args = validate_pi05_calibration.parse_args(
            [
                "--model-dir",
                "/model",
                "--profile",
                "/profile-1",
                "--profile",
                "/profile-2",
                "--input",
                "/sample",
                "--out",
                "/report",
                "--max-relative-l2",
                "0.2",
                "--validator-revision",
                "release-1.2.3",
            ]
        )
        self.assertEqual(args.validator_revision, "release-1.2.3")

    def test_load_observation_keeps_only_public_business_fields(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "sample.npz"
            np.savez(
                path,
                **{
                    "observation/image": np.zeros((2, 3, 3), dtype=np.uint8),
                    "observation/wrist_image": np.ones((2, 3, 3), dtype=np.uint8),
                    "observation/state": np.arange(8, dtype=np.float32),
                    "prompt": np.asarray("move the mug"),
                    "noise": np.ones((2, 2), dtype=np.float32),
                    "token_ids": np.asarray([1, 2], dtype=np.uint32),
                },
            )

            observation = validate_pi05_calibration.load_observation(
                path,
                image_keys=("observation/image", "observation/wrist_image"),
                prompt_key="prompt",
                state_key="observation/state",
                require_state=True,
            )

            self.assertEqual(
                set(observation),
                {
                    "observation/image",
                    "observation/wrist_image",
                    "observation/state",
                    "prompt",
                },
            )
            self.assertEqual(observation["prompt"], "move the mug")


if __name__ == "__main__":
    unittest.main()
