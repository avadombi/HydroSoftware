import unittest

import numpy as np

from hydrosoftware.calibration import (
    CalibrationEngine,
    CalibrationProblem,
    DifferentialEvolutionSearchConfig,
    NashSutcliffeEfficiency,
    RootMeanSquareError,
    UniformRandomSearchConfig,
    build_exp_hydro_calibration_problem,
)
from hydrosoftware.core.specifications import ParameterSpec
from hydrosoftware.core.results import SimulationResult
from hydrosoftware.execution import CancellationToken, ProgressSeverity, TaskState
from hydrosoftware.models.conceptual.exp_hydro import (
    ExpHydroModel,
    ExpHydroParameters,
)
from hydrosoftware.processes.snow import NoSnowModule


class CalibrationTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(4)
        self.precipitation = rng.gamma(1.2, 3.0, size=60)
        self.temperature = np.full(60, 8.0)
        self.pet = rng.uniform(0.1, 2.0, size=60)
        self.snow = NoSnowModule()
        model = ExpHydroModel(
            ExpHydroParameters(0.05, 500.0, 30.0), self.snow
        )
        self.observations = model.simulate(
            self.precipitation, self.temperature, self.pet
        ).outputs[:, 0]

    def make_problem(self, observations=None, mask=None):
        return build_exp_hydro_calibration_problem(
            self.precipitation,
            self.temperature,
            self.pet,
            self.observations if observations is None else observations,
            self.snow,
            evaluation_mask=mask,
        )

    def test_default_candidate_reproduces_synthetic_observations(self):
        events = []
        result = CalibrationEngine().run(
            self.make_problem(),
            UniformRandomSearchConfig(n_evaluations=5, seed=7),
            task_id="test-calibration",
            on_progress=events.append,
        )

        self.assertEqual(result.state, TaskState.COMPLETED)
        self.assertAlmostEqual(result.best_score, 1.0, places=12)
        self.assertEqual(result.best_parameters["maximum_storage"], 500.0)
        self.assertEqual(len(result.records), 5)
        self.assertEqual(events[0].state, TaskState.PREPARING)
        self.assertEqual(events[-1].state, TaskState.COMPLETED)
        self.assertEqual(events[-1].fraction, 1.0)
        self.assertEqual(
            [event.sequence for event in events], list(range(len(events)))
        )

    def test_warmup_mask_is_shared_with_objective(self):
        corrupted = self.observations.copy()
        corrupted[:10] += 1000.0
        mask = np.ones(60, dtype=bool)
        mask[:10] = False

        result = CalibrationEngine().run(
            self.make_problem(corrupted, mask),
            UniformRandomSearchConfig(n_evaluations=1),
        )

        self.assertEqual(result.state, TaskState.COMPLETED)
        self.assertAlmostEqual(result.best_score, 1.0, places=12)

    def test_cancellation_preserves_completed_evaluations(self):
        token = CancellationToken()
        events = []

        def observe(event):
            events.append(event)
            if event.state is TaskState.RUNNING and event.completed_units == 2:
                token.request()

        result = CalibrationEngine().run(
            self.make_problem(),
            UniformRandomSearchConfig(n_evaluations=10),
            cancellation_token=token,
            on_progress=observe,
        )

        self.assertEqual(result.state, TaskState.CANCELLED)
        self.assertEqual(len(result.records), 2)
        self.assertIsNotNone(result.best_score)
        self.assertIn(TaskState.CANCELLATION_REQUESTED, [e.state for e in events])
        self.assertEqual(events[-1].state, TaskState.CANCELLED)

    def test_constant_target_returns_a_structured_preparation_error(self):
        events = []
        result = CalibrationEngine().run(
            self.make_problem(np.ones(60)),
            UniformRandomSearchConfig(n_evaluations=3),
            on_progress=events.append,
        )

        self.assertEqual(result.state, TaskState.FAILED)
        self.assertEqual(result.error.code, "invalid_calibration_problem")
        self.assertIn("constant", result.error.explanation)
        self.assertEqual(events[-1].severity, ProgressSeverity.ERROR)
        self.assertEqual(len(result.records), 0)

    def test_candidate_errors_are_contained_until_all_candidates_fail(self):
        def fail(_parameters):
            raise RuntimeError("synthetic model failure")

        problem = CalibrationProblem(
            parameter_specs=(ParameterSpec("p", 0.0, 1.0, 0.5),),
            simulate=fail,
            observations=np.arange(5, dtype=float),
            objective=NashSutcliffeEfficiency(),
        )
        events = []
        result = CalibrationEngine().run(
            problem,
            UniformRandomSearchConfig(n_evaluations=3),
            on_progress=events.append,
        )

        self.assertEqual(result.state, TaskState.FAILED)
        self.assertEqual(result.error.code, "all_evaluations_failed")
        self.assertEqual(len(result.records), 3)
        self.assertTrue(all(not record.succeeded for record in result.records))
        self.assertTrue(
            any(event.severity is ProgressSeverity.WARNING for event in events)
        )

    def test_differential_evolution_is_reproducible_and_converges(self):
        basis = np.arange(1.0, 7.0)
        target_slope = 1.2345

        def simulate(parameters):
            output = basis * parameters["slope"]
            return SimulationResult(
                outputs=output[:, None],
                states=np.zeros((len(basis) + 1, 0)),
                fluxes=np.zeros((len(basis), 0)),
                output_names=("streamflow",),
                state_names=(),
                flux_names=(),
            )

        problem = CalibrationProblem(
            parameter_specs=(
                ParameterSpec("slope", 0.0, 5.0, 4.0),
            ),
            simulate=simulate,
            observations=basis * target_slope,
            objective=RootMeanSquareError(),
        )
        config = DifferentialEvolutionSearchConfig(
            population_size=12,
            generations=25,
            seed=19,
            include_defaults=True,
        )

        first = CalibrationEngine().run(problem, config)
        second = CalibrationEngine().run(problem, config)

        self.assertEqual(first.state, TaskState.COMPLETED)
        self.assertEqual(first.algorithm_id, "differential_evolution.rand1bin.v1")
        self.assertEqual(first.termination_reason, "generation_budget_exhausted")
        self.assertEqual(first.generations_completed, 25)
        self.assertEqual(len(first.records), 12 * 26)
        self.assertLess(first.best_score, 1e-5)
        self.assertAlmostEqual(first.best_parameters["slope"], target_slope, places=5)
        self.assertEqual(
            [record.score for record in first.records],
            [record.score for record in second.records],
        )
        self.assertEqual(first.records[0].generation, 0)
        self.assertEqual(first.records[-1].generation, 25)

    def test_parallel_differential_evolution_matches_exact_serial_scores(self):
        config = DifferentialEvolutionSearchConfig(
            population_size=8,
            generations=3,
            seed=29,
            include_defaults=True,
        )

        serial = CalibrationEngine().run(
            self.make_problem(),
            config,
            task_id="serial-exp-hydro",
        )
        parallel = CalibrationEngine().run(
            self.make_problem(),
            config,
            task_id="parallel-exp-hydro",
            max_workers=2,
        )

        self.assertEqual(serial.state, TaskState.COMPLETED)
        self.assertEqual(parallel.state, TaskState.COMPLETED)
        self.assertEqual(serial.best_score, parallel.best_score)
        self.assertEqual(serial.best_parameters, parallel.best_parameters)
        self.assertEqual(
            [record.score for record in serial.records],
            [record.score for record in parallel.records],
        )
        self.assertEqual(
            [record.parameters for record in serial.records],
            [record.parameters for record in parallel.records],
        )

    def test_differential_evolution_rounds_integer_parameters_only_at_evaluation(self):
        observations = np.full(8, 4.0)
        evaluated = []

        def simulate(parameters):
            evaluated.append(parameters["integer_parameter"])
            return SimulationResult(
                outputs=np.full((8, 1), parameters["integer_parameter"]),
                states=np.zeros((9, 0)),
                fluxes=np.zeros((8, 0)),
                output_names=("streamflow",),
                state_names=(),
                flux_names=(),
            )

        problem = CalibrationProblem(
            parameter_specs=(
                ParameterSpec(
                    "integer_parameter",
                    1.0,
                    7.0,
                    2.0,
                    value_type="integer",
                ),
            ),
            simulate=simulate,
            observations=observations,
            objective=RootMeanSquareError(),
        )
        result = CalibrationEngine().run(
            problem,
            DifferentialEvolutionSearchConfig(
                population_size=8,
                generations=8,
                seed=8,
            ),
        )

        self.assertEqual(result.best_parameters["integer_parameter"], 4.0)
        self.assertEqual(result.best_score, 0.0)
        self.assertTrue(all(value == round(value) for value in evaluated))
        self.assertTrue(all(1.0 <= value <= 7.0 for value in evaluated))

    def test_differential_evolution_cancels_at_a_candidate_boundary(self):
        token = CancellationToken()

        def observe(event):
            if event.completed_units == 10:
                token.request()

        result = CalibrationEngine().run(
            self.make_problem(),
            DifferentialEvolutionSearchConfig(
                population_size=8,
                generations=5,
                seed=3,
            ),
            cancellation_token=token,
            on_progress=observe,
        )

        self.assertEqual(result.state, TaskState.CANCELLED)
        self.assertEqual(result.termination_reason, "cancelled")
        self.assertEqual(len(result.records), 10)
        self.assertIsNotNone(result.best_parameters)


if __name__ == "__main__":
    unittest.main()
