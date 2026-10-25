"""Run the bundled experiment directly through the service-independent core."""

from pathlib import Path

from hydrosoftware.configuration import load_experiment_config
from hydrosoftware.execution.experiments import ExperimentRunner


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config = load_experiment_config(project_root / "examples/exp_hydro_calibration.yml")
    runner = ExperimentRunner(project_root)
    result = runner.run(
        config,
        task_id="bundled-example",
        on_progress=lambda event: print(
            f"[{event.state.value:>22}] {event.stage}: {event.message}"
        ),
    )
    print(f"Best {result.calibration.objective_name}: {result.calibration.best_score:.12f}")
    print(f"Best parameters: {dict(result.calibration.best_parameters)}")
    for metric in result.evaluation_metrics:
        value = "unavailable" if metric.value is None else f"{metric.value:.12f}"
        print(f"{metric.symbol}: {value}")


if __name__ == "__main__":
    main()
