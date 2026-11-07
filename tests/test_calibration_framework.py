import unittest

from apxinf.calibration import (
    CalibrationContext,
    CalibrationPlan,
    CalibrationRunner,
    CaptureSite,
    Fp8ExecutionPlan,
    QuantizationSpec,
    QuantizedOperator,
    adapt_records,
)


class TinyTransformer:
    """Public calibration target for a conventional Transformer fixture."""

    def __init__(self, records=None):
        self.observations = []
        self.records = records or {
            "embed.input": 2.0,
            "blocks.0.qkv.input": 4.0,
            "blocks.shared_mlp.input": 8.0,
            "blocks.0.fused_gate.input": 6.0,
        }

    @staticmethod
    def fp8_execution_plan(mode="static"):
        return fixture_plan(mode)

    def calibration_plan(self, mode="static"):
        return fixture_spec().plan_for(self.fp8_execution_plan(mode))

    def collect_calibration(self, observation, context: CalibrationContext):
        self.observations.append((observation, context))
        return self.records


def fixture_plan(mode="static"):
    return Fp8ExecutionPlan(
        activation_mode=mode,
        operators=(
            QuantizedOperator("embed", "gemm", output="hidden"),
            QuantizedOperator("blocks.0.qkv", "linear", output="qkv"),
            QuantizedOperator("blocks.0.mlp", "linear", output="mlp0"),
            QuantizedOperator("blocks.1.mlp", "linear", output="mlp1"),
            QuantizedOperator("blocks.0.fused_gate", "fused", output="gate"),
            QuantizedOperator("lm_head", "linear", output="logits"),
            # It is named Linear but absent from the quantized execution path.
            QuantizedOperator("diagnostic_linear", "linear", quantized=False),
        ),
    )


def fixture_spec(**overrides):
    values = dict(
        model_family="tiny_transformer",
        excluded_outputs=frozenset({"logits"}),
        shared_scales={
            "blocks.0.mlp": "blocks.shared_mlp.input",
            "blocks.1.mlp": "blocks.shared_mlp.input",
        },
        custom_captures=(
            CaptureSite(
                "blocks.0.fused_gate.input",
                consumer="blocks.0.fused_gate",
                statistic="percentile:99.9",
            ),
        ),
    )
    values.update(overrides)
    return QuantizationSpec(**values)


class CalibrationFrameworkTest(unittest.TestCase):
    def test_dataset_adapter_maps_records_only_to_public_observations(self):
        class Adapter:
            @staticmethod
            def to_observation(record):
                return {"tokens": record["external_token_ids"], "prompt": record["task"]}

        observations = list(
            adapt_records(
                [{"external_token_ids": [1, 2], "task": "move"}], Adapter()
            )
        )

        self.assertEqual(observations, [{"tokens": [1, 2], "prompt": "move"}])
        self.assertNotIn("hidden_states", observations[0])
        self.assertNotIn("activation", observations[0])

    def test_conventional_plan_uses_defaults_and_thin_spec_overrides(self):
        plan = TinyTransformer().calibration_plan()

        self.assertEqual(
            plan.capture_sites,
            (
                CaptureSite("embed.input", consumer="embed"),
                CaptureSite("blocks.0.qkv.input", consumer="blocks.0.qkv"),
                CaptureSite("blocks.shared_mlp.input", consumer="blocks.0.mlp"),
                CaptureSite(
                    "blocks.0.fused_gate.input",
                    consumer="blocks.0.fused_gate",
                    statistic="percentile:99.9",
                ),
            ),
        )
        self.assertEqual(plan.consumers["blocks.1.mlp"], "blocks.shared_mlp.input")
        self.assertNotIn("lm_head", plan.consumers)
        self.assertNotIn("diagnostic_linear", plan.consumers)

    def test_runner_accepts_public_observations_and_generates_manifest(self):
        target = TinyTransformer()
        runner = CalibrationRunner(
            target,
            target.calibration_plan(),
            checkpoint="sha256:fixture",
            data_identity="dataset:fixture-v1",
            source_revision="test-revision",
            device={"requested": "cpu", "host": "test"},
            margin=2.0,
            seed=11,
        )
        observations = [
            {"tokens": [1, 2, 3]},
            {"tokens": [4, 5]},
        ]

        document = runner.run(observations)

        self.assertIs(target.observations[0][0], observations[0])
        self.assertEqual(target.observations[1][1].sample_index, 1)
        self.assertEqual(document["model"]["family"], "tiny_transformer")
        self.assertEqual(document["calibration_data"]["sample_count"], 2)
        self.assertEqual(
            document["plan"]["consumers"]["blocks.1.mlp"],
            "blocks.shared_mlp.input",
        )
        self.assertEqual(
            document["plan"]["statistics"]["blocks.0.fused_gate.input"],
            "percentile:99.9",
        )
        self.assertAlmostEqual(document["scales"]["embed.input"]["scale"], 4 / 448)

    def test_dynamic_activation_fp8_is_calibration_free(self):
        target = TinyTransformer()
        plan = target.calibration_plan("dynamic")

        self.assertFalse(plan.requires_calibration)
        self.assertIsNone(
            CalibrationRunner(
                target,
                plan,
                checkpoint="sha256:fixture",
                data_identity="dataset:fixture-v1",
                source_revision="test-revision",
                device={"requested": "cpu", "host": "test"},
            ).run([{"tokens": [1]}])
        )
        self.assertEqual(target.observations, [])

    def test_coverage_rejects_unobserved_custom_site(self):
        target = TinyTransformer(records={"embed.input": 1.0})
        plan = QuantizationSpec(
            model_family="tiny_transformer",
            custom_captures=(CaptureSite("custom.required", consumer="custom"),),
        ).plan_for(
            Fp8ExecutionPlan(
                operators=(
                    QuantizedOperator("embed", "linear"),
                    QuantizedOperator("custom", "fused"),
                ),
            )
        )
        runner = CalibrationRunner(
            target,
            plan,
            checkpoint="sha256:fixture",
            data_identity="dataset:fixture-v1",
            source_revision="test-revision",
            device={"requested": "cpu", "host": "test"},
        )

        with self.assertRaisesRegex(ValueError, "missing=.*custom.required"):
            runner.run([{"tokens": [1]}])

    def test_plan_rejects_generated_scale_without_fp8_consumer(self):
        with self.assertRaisesRegex(ValueError, "no FP8 consumer.*orphan"):
            QuantizationSpec(
                model_family="tiny_transformer",
                custom_captures=(CaptureSite("orphan"),),
            ).plan_for(Fp8ExecutionPlan(operators=()))

    def test_plan_rejects_untyped_consumer_contract(self):
        with self.assertRaisesRegex(ValueError, "ConsumerContract"):
            CalibrationPlan(
                model_family="tiny_transformer",
                capture_sites=(CaptureSite("orphan"),),
                consumers={},
                consumer_contract="manifest",
            )

    def test_plan_rejects_quantized_custom_operator_without_capture_override(self):
        with self.assertRaisesRegex(ValueError, "fused.*requires a custom capture"):
            QuantizationSpec(model_family="tiny_transformer").plan_for(
                Fp8ExecutionPlan(
                    operators=(QuantizedOperator("fused", "fused"),),
                )
            )

    def test_plan_rejects_conflicting_statistics_for_shared_scale(self):
        with self.assertRaisesRegex(ValueError, "conflicting statistics.*shared"):
            QuantizationSpec(
                model_family="tiny_transformer",
                shared_scales={"linear": "shared"},
                custom_captures=(
                    CaptureSite(
                        "shared", consumer="fused", statistic="percentile:99.9"
                    ),
                ),
            ).plan_for(
                Fp8ExecutionPlan(
                    operators=(
                        QuantizedOperator("linear", "linear"),
                        QuantizedOperator("fused", "fused"),
                    )
                )
            )

    def test_plan_rejects_excluded_output_absent_from_execution_plan(self):
        with self.assertRaisesRegex(ValueError, "excluded outputs.*logtits"):
            QuantizationSpec(
                model_family="tiny_transformer",
                excluded_outputs=frozenset({"logtits"}),
            ).plan_for(
                Fp8ExecutionPlan(
                    operators=(
                        QuantizedOperator("lm_head", "linear", output="logits"),
                    )
                )
            )


if __name__ == "__main__":
    unittest.main()
