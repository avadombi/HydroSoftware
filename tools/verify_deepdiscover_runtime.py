#!/usr/bin/env python3
"""Fast runtime check for the validated NumPy DeepDiscover implementation."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np

from hydrosoftware.configuration import (
    DeepDiscoverModelConfig,
    DiscoveryAdamTrainingConfig,
    DiscoveryLossConfig,
    NoSnowConfig,
)
from hydrosoftware.models.deepdiscover_numpy import (
    fit_deepdiscover,
    simulate_deepdiscover_from_artifact,
)
from hydrosoftware.processes.snow import NoSnowModule


def main() -> None:
    length = 120
    rng = np.random.default_rng(481)
    precipitation = rng.gamma(1.7, 2.0, size=length)
    temperature = rng.normal(7.0, 5.0, size=length)
    pet = rng.uniform(0.1, 2.0, size=length)
    streamflow = np.zeros(length, dtype=float)
    storage = 18.0
    for index in range(length):
        available = storage + precipitation[index]
        actual_et = 0.22 * min(pet[index], available)
        remaining = available - actual_et
        streamflow[index] = 0.29 * remaining
        storage = remaining - streamflow[index]

    model = DeepDiscoverModelConfig(
        snow=NoSnowConfig(),
        initial_state=(0.0, 18.0),
        process_count=3,
        hidden_units=8,
        train_input_switches=False,
        train_process_switches=False,
        loss=DiscoveryLossConfig(
            decorrelation_weight=0.02,
            mass_balance_weight=0.0,
            smoothness_weight=0.001,
            binary_weight=0.0,
            sparsity_weight=0.0,
        ),
    )
    fit = fit_deepdiscover(
        model,
        DiscoveryAdamTrainingConfig(
            epochs=20,
            learning_rate=0.01,
            patience=20,
            gradient_clip_norm=5.0,
            seed=42,
        ),
        NoSnowModule(),
        precipitation,
        temperature,
        pet,
        streamflow,
        start_index=0,
        train_end_index=75,
        validation_end_index=95,
        test_end_index=120,
        warmup_steps=5,
        objective_id="nash_sutcliffe_efficiency",
    )
    replay = simulate_deepdiscover_from_artifact(
        fit.artifact,
        NoSnowModule(),
        precipitation,
        temperature,
        pet,
        initial_state=(0.0, 18.0),
    )
    np.testing.assert_allclose(replay.outputs, fit.simulation.outputs, rtol=1e-10, atol=1e-10)
    residual = float(fit.simulation.diagnostics["maximum_absolute_mass_balance_error"])
    if residual >= 1e-10:
        raise AssertionError(f"Mass-balance residual too large: {residual:.3e}")
    process_positions = [
        fit.simulation.flux_names.index(f"discovered_process_{i}")
        for i in range(1, 4)
    ]
    processes = fit.simulation.fluxes[:, process_positions]
    pair_differences = [
        float(np.max(np.abs(processes[:, i] - processes[:, j])))
        for i in range(3)
        for j in range(i + 1, 3)
    ]
    if min(pair_differences) <= 1e-10:
        raise AssertionError("Distinct candidate processes collapsed to identical trajectories.")
    print(f"PASS DeepDiscover NumPy runtime; max balance residual={residual:.3e}")
    print("maximum pairwise process differences:", pair_differences)


if __name__ == "__main__":
    main()
