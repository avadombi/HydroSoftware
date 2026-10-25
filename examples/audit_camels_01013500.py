"""Print the frozen data audit for the CAMELS-US 01013500 example."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from hydrosoftware.configuration import load_experiment_config
from hydrosoftware.data_sources import analyze_calibration_csv


def main() -> None:
    project_root = Path(__file__).resolve().parents[1]
    configuration = load_experiment_config(
        project_root / "examples" / "camels_us_01013500_exp_hydro.yml"
    )
    analysis = analyze_calibration_csv(configuration.data, workspace_root=project_root)
    print(json.dumps(asdict(analysis.report), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
