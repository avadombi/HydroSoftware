from __future__ import annotations

import numpy as np

from hydrosoftware.configuration import (
    DeepDiscoverModelConfig,
    DiscoveryAdamTrainingConfig,
    DiscoveryLossConfig,
    NoSnowConfig,
)
from hydrosoftware.models.deepdiscover_numpy import fit_deepdiscover
from hydrosoftware.processes.snow import NoSnowModule


def make_series(n: int = 180):
    rng = np.random.default_rng(1234)
    p = rng.gamma(1.5, 2.5, size=n)
    t = rng.normal(8.0, 5.0, size=n)
    pet = rng.uniform(0.2, 2.2, size=n)
    q = np.zeros(n, dtype=float)
    s = 20.0
    for i in range(n):
        a = s + p[i]
        eta = 0.28
        e = eta * min(pet[i], a)
        b = a - e
        rho = 0.18 + 0.012 * s
        q[i] = b * rho / (1.0 + rho)
        s = b / (1.0 + rho)
    return p, t, pet, q


def nse(obs, sim):
    obs = np.asarray(obs, float)
    sim = np.asarray(sim, float)
    den = np.sum((obs - obs.mean()) ** 2)
    return 1.0 - np.sum((sim - obs) ** 2) / den


def main():
    p, t, pet, q = make_series()
    model = DeepDiscoverModelConfig(
        snow=NoSnowConfig(),
        initial_state=(0.0, 20.0),
        process_count=1,
        hidden_units=8,
        train_input_switches=False,
        train_process_switches=False,
        loss=DiscoveryLossConfig(
            decorrelation_weight=0.0,
            mass_balance_weight=0.0,
            smoothness_weight=0.001,
            binary_weight=0.0,
            sparsity_weight=0.0,
        ),
    )
    fit = fit_deepdiscover(
        model,
        DiscoveryAdamTrainingConfig(
            epochs=40,
            learning_rate=0.01,
            patience=20,
            minimum_delta=1e-5,
            gradient_clip_norm=5.0,
            seed=200,
        ),
        NoSnowModule(),
        p, t, pet, q,
        start_index=0,
        train_end_index=110,
        validation_end_index=145,
        test_end_index=180,
        warmup_steps=10,
        objective_id="nash_sutcliffe_efficiency",
    )
    sim = fit.simulation.outputs[:, 0]
    val_nse = nse(q[110:145], sim[110:145])
    residual = float(fit.simulation.diagnostics["maximum_absolute_mass_balance_error"])
    print(f"best_epoch={fit.artifact.best_epoch}")
    print(f"validation_nse={val_nse:.6f}")
    print(f"maximum_absolute_mass_balance_error={residual:.3e}")
    print("process_count=", fit.artifact.process_count)
    print("hidden_units=", fit.artifact.hidden_units)
    if not np.isfinite(val_nse):
        raise SystemExit("FAIL: validation NSE is non-finite")
    if residual > 1e-10:
        raise SystemExit("FAIL: interval mass balance is not at floating-point precision")
    print("PASS DeepDiscover controlled single-DU runtime")


if __name__ == "__main__":
    main()
