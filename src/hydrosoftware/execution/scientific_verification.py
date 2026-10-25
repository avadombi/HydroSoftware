"""On-demand numerical verification of immutable HydroSoftware run archives.

The checks in this module are deliberately different from goodness-of-fit
metrics.  They ask whether a saved numerical experiment can be replayed, split
and restarted, and perturbed without violating causal time order.  They also
make the relationship between the executable implementation and the historical
research sources explicit; they do not claim numerical equivalence where the
new engine intentionally changed ambiguous or unsafe behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hydrosoftware.configuration import ExperimentConfigV1, PamlModelConfig
from hydrosoftware.core.features import LaggedFeature, build_feature_mapping, maximum_lag
from hydrosoftware.execution.ensembles import (
    aggregate_member_predictions,
    member_artifact_payloads,
)
from hydrosoftware.execution.forecasting import run_forecast
from hydrosoftware.models.neural import (
    S2S_ARTIFACT_TYPE,
    S2SArtifact,
    predict_s2s_partitioned_from_artifact,
)


VERIFICATION_METHOD_ID = "scientific_run_verification.v1"


class ScientificVerificationError(RuntimeError):
    """Stable failure raised when an archive cannot be scientifically checked."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class _ArchivedData:
    """Minimum processed-data definition required by ``run_forecast``."""

    precipitation: np.ndarray
    temperature: np.ndarray
    potential_evapotranspiration: np.ndarray
    streamflow: np.ndarray
    additional_predictors: Mapping[str, np.ndarray]
    timestamps: tuple[str, ...] | None
    source_path: Path
    report: Any


@dataclass(frozen=True)
class _ArchivedReport:
    """Serializable audit stub for data already frozen inside a run archive."""

    transformed_value_count: int = 0
    warning_count: int = 0
    verification_source: str = "immutable_run_archive"


def _check(
    check_id: str,
    name: str,
    status: str,
    description: str,
    *,
    maximum_absolute_difference: float | None = None,
    tolerance: float | None = None,
    details: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if status not in {"passed", "failed", "not_applicable"}:
        raise ValueError(f"Unsupported verification status: {status}")
    return {
        "check_id": check_id,
        "name": name,
        "status": status,
        "description": description,
        "maximum_absolute_difference": maximum_absolute_difference,
        "tolerance": tolerance,
        "details": dict(details or {}),
    }


def _array(value: Any, *, name: str, dimensions: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ScientificVerificationError(
            "run_archive_array_invalid",
            f"The archived {name} array cannot be interpreted numerically.",
        ) from exc
    if array.ndim != dimensions:
        raise ScientificVerificationError(
            "run_archive_array_invalid",
            f"The archived {name} array must have {dimensions} dimensions.",
        )
    return array


def _maximum_difference(left: Any, right: Any) -> float:
    """Return a strict NaN-aware maximum difference, or infinity on mismatch."""

    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    if left_array.shape != right_array.shape:
        return float("inf")
    if not np.array_equal(np.isnan(left_array), np.isnan(right_array)):
        return float("inf")
    if np.any(np.isinf(left_array)) or np.any(np.isinf(right_array)):
        if not np.array_equal(left_array, right_array):
            return float("inf")
    finite = np.isfinite(left_array) & np.isfinite(right_array)
    if not np.any(finite):
        return 0.0
    return float(np.max(np.abs(left_array[finite] - right_array[finite])))


def _simulation_difference(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float]:
    return {
        name: _maximum_difference(left.get(name), right.get(name))
        for name in ("outputs", "states", "fluxes")
    }


def _archive_data(
    result: Mapping[str, Any],
    interval: slice | None = None,
    *,
    perturb_after: int | None = None,
) -> _ArchivedData:
    forcings = result.get("forcings")
    if not isinstance(forcings, Mapping):
        raise ScientificVerificationError(
            "run_archive_forcings_missing",
            "The run archive does not contain the forcings required for replay.",
        )
    required = (
        "precipitation",
        "temperature",
        "potential_evapotranspiration",
    )
    missing = [name for name in required if name not in forcings]
    if missing:
        raise ScientificVerificationError(
            "run_archive_forcings_missing",
            "The run archive is missing required forcings: " + ", ".join(missing),
        )
    selected = interval or slice(None)
    arrays = {
        name: np.asarray(values, dtype=float)[selected].copy()
        for name, values in forcings.items()
    }
    length = len(arrays["precipitation"])
    if any(len(values) != length for values in arrays.values()):
        raise ScientificVerificationError(
            "run_archive_forcings_misaligned",
            "All archived forcing and predictor channels must share one time dimension.",
        )
    if perturb_after is not None and perturb_after < length:
        arrays["precipitation"][perturb_after:] = (
            arrays["precipitation"][perturb_after:] * 1.37 + 0.123
        )
        arrays["temperature"][perturb_after:] += 2.5
        arrays["potential_evapotranspiration"][perturb_after:] = (
            arrays["potential_evapotranspiration"][perturb_after:] * 0.73 + 0.05
        )
        for name, values in arrays.items():
            if name in required:
                continue
            finite = values[np.isfinite(values)]
            scale = max(float(np.std(finite)) if len(finite) else 0.0, 1.0)
            values[perturb_after:] += 0.5 * scale

    observations = result.get("observations")
    streamflow = (
        np.asarray(observations, dtype=float)[selected].copy()
        if observations is not None
        else np.full(length, np.nan, dtype=float)
    )
    timestamps = result.get("timestamps")
    selected_timestamps = (
        tuple(np.asarray(timestamps, dtype=object)[selected].tolist())
        if timestamps is not None
        else None
    )
    report = _ArchivedReport()
    return _ArchivedData(
        precipitation=arrays["precipitation"],
        temperature=arrays["temperature"],
        potential_evapotranspiration=arrays["potential_evapotranspiration"],
        streamflow=streamflow,
        additional_predictors={
            name: values for name, values in arrays.items() if name not in required
        },
        timestamps=selected_timestamps,
        source_path=Path("run-archive"),
        report=report,
    )


def _implementation_source(config: ExperimentConfigV1) -> dict[str, Any]:
    model_id = config.model.model_id
    native_prefixes = ("mlp.", "lstm.", "gru.", "cnn1d.")
    if model_id.startswith(native_prefixes):
        return {
            "relationship": "hydrosoftware_native",
            "historical_sources": [],
            "equivalence_claim": "not_applicable",
            "statement": (
                "This TensorFlow neural engine is a HydroSoftware implementation; "
                "it is not presented as a conversion of a supplied historical model file."
            ),
            "declared_adaptations": [],
        }
    if model_id.startswith("paml."):
        return {
            "relationship": "jiang_prnn_cnn_implementation",
            "historical_sources": ["PaML_hydrolayer.py"],
            "equivalence_claim": "formulation_aligned",
            "statement": (
                "The PaML engine follows the jointly trained P-RNN plus causal-CNN "
                "architecture of Jiang, Zheng and Solomatine (2020)."
            ),
            "declared_adaptations": [
                "HydroSoftware accepts supplied potential evapotranspiration instead of deriving it from day length.",
                "The physical recurrent backbone can be EXP-HYDRO or the HydroSoftware daily HBV formulation.",
                "Scaling statistics are estimated from the fitting period and retained for later periods.",
            ],
        }
    if model_id.startswith("deepdiscover."):
        return {
            "relationship": "corrected_instantaneous_variant",
            "historical_sources": [
                "DD_hydrolayer.py",
                "DD_hydroutils.py",
                "deep_discover_demo.ipynb",
            ],
            "equivalence_claim": "not_claimed",
            "statement": (
                "The executable model is the explicitly versioned instantaneous "
                "DeepDiscover variant; it does not reproduce the historical layer weights "
                "or its ambiguous post-update flux calculation."
            ),
            "declared_adaptations": [
                "One allocation produces both reported process fluxes and the state transition.",
                "Scaling statistics are fitted on training rows only.",
                "Switches are bounded and every scientific penalty is archived separately.",
                "No recurrent recession or learned routing is claimed for this variant.",
            ],
        }

    source = "MC_conceptual_model_classes.py"
    adaptations = [
        "Forcings are constant within each declared daily interval.",
        "Outgoing water is limited explicitly to preserve non-negative storage.",
        "Boundary states and interval fluxes use one unambiguous timing convention.",
    ]
    statement = (
        "The hydrological relationships are retained where applicable, but numerical "
        "equivalence with the historical RK4 code is not claimed."
    )
    if model_id == "exp_hydro.mass_conservative.v1":
        sources = [source, "MC_OdeSolver.py", "PaML_hydrolayer.py"]
        relationship = "mass_conservative_exp_hydro_adaptation"
        adaptations.append(
            "Explicit conservative substeps replace the historical RK4 forcing index based on round(t)."
        )
    elif model_id == "hbv.mass_conservative_daily.v1":
        sources = [source, "MC_conceptual_model_params.py"]
        relationship = "mass_conservative_daily_adaptation"
        adaptations.append("Optional MAXBAS routing is discrete, conservative, and restartable.")
    else:
        sources = []
        relationship = "unregistered"
        statement = "No historical-source relationship is registered for this model identifier."
        adaptations = []
    return {
        "relationship": relationship,
        "historical_sources": sources,
        "equivalence_claim": "not_claimed" if sources else "unregistered",
        "statement": statement,
        "declared_adaptations": adaptations,
    }


def _physical_nonnegativity(
    config: ExperimentConfigV1,
    simulation: Mapping[str, Any],
    tolerance: float,
) -> dict[str, Any]:
    states = np.asarray(simulation.get("states"), dtype=float)
    fluxes = np.asarray(simulation.get("fluxes"), dtype=float)
    outputs = np.asarray(simulation.get("outputs"), dtype=float)
    state_names = tuple(simulation.get("state_names") or ())
    flux_names = tuple(simulation.get("flux_names") or ())
    output_names = tuple(simulation.get("output_names") or ())
    physical = bool(state_names or flux_names) or config.model.model_id.startswith(
        "deepdiscover."
    )
    if not physical:
        return _check(
            "physical_nonnegativity",
            "Physical non-negativity",
            "not_applicable",
            "The purely neural engine exposes no internal water storages or physical fluxes.",
            tolerance=tolerance,
        )

    minima: dict[str, float] = {}
    if states.size:
        minima["states"] = float(np.min(states))
    if fluxes.size:
        minima["fluxes"] = float(np.min(fluxes))
    physical_output_positions = [
        index
        for index, name in enumerate(output_names)
        if name == "streamflow" and not isinstance(config.model, PamlModelConfig)
        or name == "physical_streamflow"
    ]
    if isinstance(config.model, PamlModelConfig) and config.model.enforce_nonnegative_output:
        physical_output_positions.extend(
            index for index, name in enumerate(output_names) if name == "streamflow"
        )
    if physical_output_positions:
        minima["physical_outputs"] = float(
            np.nanmin(outputs[:, sorted(set(physical_output_positions))])
        )
    minimum = min(minima.values(), default=0.0)
    passed = minimum >= -tolerance
    return _check(
        "physical_nonnegativity",
        "Physical non-negativity",
        "passed" if passed else "failed",
        (
            "Every exposed physical storage, water flux, and constrained physical "
            "output remains non-negative within numerical tolerance."
            if passed
            else "At least one exposed physical quantity is negative beyond tolerance."
        ),
        maximum_absolute_difference=max(0.0, -minimum),
        tolerance=tolerance,
        details={"minimum_by_group": minima},
    )


def _mass_balance(simulation: Mapping[str, Any], tolerance: float) -> dict[str, Any]:
    diagnostics = simulation.get("diagnostics") or {}
    maximum = diagnostics.get("maximum_absolute_mass_balance_error")
    if maximum is None:
        return _check(
            "mass_balance",
            "Water balance closure",
            "not_applicable",
            "This model does not expose a physical water-balance diagnostic.",
            tolerance=tolerance,
        )
    try:
        maximum_value = abs(float(maximum))
    except (TypeError, ValueError) as exc:
        raise ScientificVerificationError(
            "run_archive_balance_invalid",
            "The archived maximum mass-balance error is not numeric.",
        ) from exc
    passed = np.isfinite(maximum_value) and maximum_value <= tolerance
    return _check(
        "mass_balance",
        "Water balance closure",
        "passed" if passed else "failed",
        (
            "The maximum interval water-balance residual is within the declared tolerance."
            if passed
            else "The archived interval water-balance residual exceeds tolerance."
        ),
        maximum_absolute_difference=maximum_value,
        tolerance=tolerance,
    )


def _replay(
    archive: Mapping[str, Any],
    data: _ArchivedData,
    *,
    initialization: str,
) -> dict[str, Any]:
    return run_forecast(
        archive,
        data,
        initialization=initialization,
        observations_available=False,
    )


def _replay_archived_trajectory(
    archive: Mapping[str, Any],
    config: ExperimentConfigV1,
    result: Mapping[str, Any],
    data: _ArchivedData,
) -> dict[str, Any]:
    """Replay saved fitting geometry when it differs from frozen forecasting.

    Causal S2S training restarts its masked recurrent blocks at the validation
    and test boundaries. Frozen forecasting instead has one continuing block
    phase. Both are valid, but the latter cannot verify the former bit for bit.
    """

    training = result.get("training") or {}
    artifact_payload = training.get("artifact") or {}
    members = member_artifact_payloads(artifact_payload)
    if not members or str(members[0].get("artifact_type", "")) != S2S_ARTIFACT_TYPE:
        return _replay(archive, data, initialization="configured")
    if not hasattr(config.task, "period"):
        raise ScientificVerificationError(
            "s2s_replay_period_missing",
            "The S2S archive does not expose its training partitions.",
        )
    corrector = (
        config.model.corrector
        if isinstance(config.model, PamlModelConfig)
        else config.model
    )
    recipes = tuple(
        LaggedFeature(source=item.source, lag_steps=item.lag_steps)
        for item in corrector.lagged_features
    )
    if recipes:
        raise ScientificVerificationError(
            "s2s_replay_lag_context_missing",
            "Exact S2S archive replay with lagged predictors requires pre-period "
            "raw context that is not stored in this archive format.",
        )
    raw = {
        "precipitation": data.precipitation,
        "temperature": data.temperature,
        "potential_evapotranspiration": data.potential_evapotranspiration,
        **{
            name: np.asarray(values, dtype=float)
            for name, values in data.additional_predictors.items()
        },
    }
    archived_simulation = result.get("best_simulation") or {}
    if isinstance(config.model, PamlModelConfig):
        archived_outputs = _array(
            archived_simulation.get("outputs"),
            name="outputs",
            dimensions=2,
        )
        archived_states = _array(
            archived_simulation.get("states"),
            name="states",
            dimensions=2,
        )
        archived_fluxes = _array(
            archived_simulation.get("fluxes"),
            name="fluxes",
            dimensions=2,
        )
        raw.update(
            {
                f"physical_output__{name}": archived_outputs[:, position]
                for position, name in enumerate(
                    archived_simulation.get("output_names") or ()
                )
                if name == "physical_streamflow"
            }
        )
        if "physical_output__physical_streamflow" in raw:
            raw["physical_output__streamflow"] = raw.pop(
                "physical_output__physical_streamflow"
            )
        raw.update(
            {
                f"physical_state__{name}": archived_states[1:, position]
                for position, name in enumerate(
                    archived_simulation.get("state_names") or ()
                )
            }
        )
        raw.update(
            {
                f"physical_flux__{name}": archived_fluxes[:, position]
                for position, name in enumerate(
                    archived_simulation.get("flux_names") or ()
                )
            }
        )
    features = build_feature_mapping(raw, corrector.feature_roles, recipes)
    period = config.task.period
    selected_start = int(period.start_index)
    effective_start = max(selected_start, maximum_lag(recipes)) - selected_start
    boundaries = {
        "start_index": effective_start,
        "train_end_index": int(period.train_end_index) - selected_start,
        "validation_end_index": int(period.validation_end_index) - selected_start,
        "test_end_index": int(period.test_end_index) - selected_start,
    }
    predictions = [
        predict_s2s_partitioned_from_artifact(
            S2SArtifact.from_mapping(member),
            features,
            **boundaries,
        )
        for member in members
    ]
    if len(predictions) == 1:
        prediction = predictions[0]
    else:
        prediction = aggregate_member_predictions(
            predictions,
            lower_quantile=float(artifact_payload.get("lower_quantile", 0.1)),
            upper_quantile=float(artifact_payload.get("upper_quantile", 0.9)),
        )[0]
    if isinstance(config.model, PamlModelConfig):
        physical_streamflow = raw["physical_output__streamflow"]
        if config.model.correction_strategy == "additive_residual":
            prediction = physical_streamflow + prediction
        if config.model.enforce_nonnegative_output:
            finite = np.isfinite(prediction)
            prediction[finite] = np.maximum(prediction[finite], 0.0)
        replay_outputs = np.column_stack((prediction, physical_streamflow))
    else:
        replay_outputs = prediction[:, None]
    return {
        "best_simulation": {
            **dict(archived_simulation),
            "outputs": replay_outputs,
        }
    }


def verify_run_archive(
    archive: Mapping[str, Any],
    *,
    comparison_tolerance: float = 1e-10,
    mass_balance_tolerance: float = 1e-8,
    nonnegative_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Verify one completed immutable archive without refitting its model."""

    if min(comparison_tolerance, mass_balance_tolerance, nonnegative_tolerance) < 0:
        raise ValueError("Verification tolerances must be non-negative.")
    try:
        config = ExperimentConfigV1.model_validate(archive.get("config"))
    except (TypeError, ValueError) as exc:
        raise ScientificVerificationError(
            "run_archive_configuration_invalid",
            "The saved run does not contain a valid experiment configuration.",
        ) from exc
    result = archive.get("result")
    if not isinstance(result, Mapping) or result.get("state") != "completed":
        raise ScientificVerificationError(
            "run_archive_not_completed",
            "Scientific verification requires one completed run archive.",
        )
    simulation = result.get("best_simulation")
    if not isinstance(simulation, Mapping):
        raise ScientificVerificationError(
            "run_archive_simulation_missing",
            "The completed run archive does not contain a best simulation.",
        )

    outputs = _array(simulation.get("outputs"), name="outputs", dimensions=2)
    states = _array(simulation.get("states"), name="states", dimensions=2)
    fluxes = _array(simulation.get("fluxes"), name="fluxes", dimensions=2)
    row_count = len(outputs)
    restart_tolerance = (
        max(comparison_tolerance, 1e-6)
        if ".tensorflow_" in config.model.model_id
        else comparison_tolerance
    )
    checks: list[dict[str, Any]] = []
    names_valid = (
        outputs.shape[1] == len(simulation.get("output_names") or ())
        and states.shape[1] == len(simulation.get("state_names") or ())
        and fluxes.shape[1] == len(simulation.get("flux_names") or ())
    )
    dimensions_valid = len(states) == row_count + 1 and len(fluxes) == row_count
    numerical_valid = (
        not np.any(np.isinf(outputs))
        and np.all(np.isfinite(states))
        and np.all(np.isfinite(fluxes))
        and outputs.size > 0
        and np.any(np.isfinite(outputs))
    )
    archive_valid = names_valid and dimensions_valid and numerical_valid
    checks.append(
        _check(
            "archive_integrity",
            "Archive consistency",
            "passed" if archive_valid else "failed",
            (
                "Outputs and interval fluxes have one row per forcing interval, states "
                "have one additional boundary, and the stored arrays are numerically usable."
                if archive_valid
                else "The archived arrays violate their declared dimensions, names, or finite-value rules."
            ),
            details={
                "forcing_intervals": row_count,
                "output_columns": outputs.shape[1],
                "state_boundaries": len(states),
                "state_columns": states.shape[1],
                "flux_columns": fluxes.shape[1],
                "output_missing_values": int(np.count_nonzero(np.isnan(outputs))),
            },
        )
    )
    checks.append(_physical_nonnegativity(config, simulation, nonnegative_tolerance))
    checks.append(_mass_balance(simulation, mass_balance_tolerance))

    if not archive_valid or row_count < 2:
        reason = (
            "Replay-based checks require a valid archive containing at least two intervals."
        )
        for check_id, name in (
            ("deterministic_replay", "Deterministic frozen replay"),
            ("restart_equivalence", "Continuous-versus-restarted equivalence"),
            ("future_forcing_causality", "Future-forcing causality"),
        ):
            checks.append(
                _check(check_id, name, "not_applicable", reason, tolerance=comparison_tolerance)
            )
    else:
        full_data = _archive_data(result)
        try:
            replay = _replay_archived_trajectory(
                archive,
                config,
                result,
                full_data,
            )
            replay_simulation = replay["best_simulation"]
            replay_differences = _simulation_difference(simulation, replay_simulation)
            replay_maximum = max(replay_differences.values())
            checks.append(
                _check(
                    "deterministic_replay",
                    "Deterministic frozen replay",
                    "passed" if replay_maximum <= comparison_tolerance else "failed",
                    (
                        "The archived parameters or weights reproduce the saved trajectory "
                        "without optimization."
                        if replay_maximum <= comparison_tolerance
                        else "A frozen replay differs from the trajectory stored in the archive."
                    ),
                    maximum_absolute_difference=replay_maximum,
                    tolerance=comparison_tolerance,
                    details={"differences": replay_differences},
                )
            )
        except Exception as exc:  # converted to a reported verification result
            replay = None
            replay_simulation = None
            checks.append(
                _check(
                    "deterministic_replay",
                    "Deterministic frozen replay",
                    "failed",
                    "The frozen model could not replay its archived forcings.",
                    tolerance=comparison_tolerance,
                    details={"error": f"{type(exc).__name__}: {exc}"},
                )
            )

        split = max(1, min(row_count - 1, row_count // 2))
        if replay_simulation is None:
            checks.append(
                _check(
                    "restart_equivalence",
                    "Continuous-versus-restarted equivalence",
                    "not_applicable",
                    "Restart verification depends on a successful frozen replay.",
                    tolerance=comparison_tolerance,
                )
            )
        else:
            try:
                uninterrupted = _replay(
                    archive,
                    full_data,
                    initialization="configured",
                )
                first = _replay(
                    archive,
                    _archive_data(result, slice(0, split)),
                    initialization="configured",
                )
                continuation_archive = {
                    **dict(archive),
                    "task_id": first["task_id"],
                    "result": first,
                }
                second = _replay(
                    continuation_archive,
                    _archive_data(result, slice(split, row_count)),
                    initialization="source_final",
                )
                first_simulation = first["best_simulation"]
                second_simulation = second["best_simulation"]
                restarted = {
                    "outputs": np.concatenate(
                        (
                            np.asarray(first_simulation["outputs"], dtype=float),
                            np.asarray(second_simulation["outputs"], dtype=float),
                        ),
                        axis=0,
                    ),
                    "states": np.concatenate(
                        (
                            np.asarray(first_simulation["states"], dtype=float),
                            np.asarray(second_simulation["states"], dtype=float)[1:],
                        ),
                        axis=0,
                    ),
                    "fluxes": np.concatenate(
                        (
                            np.asarray(first_simulation["fluxes"], dtype=float),
                            np.asarray(second_simulation["fluxes"], dtype=float),
                        ),
                        axis=0,
                    ),
                }
                restart_differences = _simulation_difference(
                    uninterrupted["best_simulation"], restarted
                )
                restart_maximum = max(restart_differences.values())
                checks.append(
                    _check(
                        "restart_equivalence",
                        "Continuous-versus-restarted equivalence",
                        "passed" if restart_maximum <= restart_tolerance else "failed",
                        (
                            "Splitting the forcing record and restoring the complete model "
                            "state or antecedent context reproduces one uninterrupted run."
                            if restart_maximum <= restart_tolerance
                            else "The restarted trajectory differs from the uninterrupted frozen replay."
                        ),
                        maximum_absolute_difference=restart_maximum,
                        tolerance=restart_tolerance,
                        details={"split_after_intervals": split, "differences": restart_differences},
                    )
                )
            except Exception as exc:
                checks.append(
                    _check(
                        "restart_equivalence",
                        "Continuous-versus-restarted equivalence",
                        "failed",
                        "The archived model could not complete a split-and-restart verification.",
                        tolerance=comparison_tolerance,
                        details={
                            "split_after_intervals": split,
                            "error": f"{type(exc).__name__}: {exc}",
                        },
                    )
                )

        try:
            baseline = _replay(
                archive,
                full_data,
                initialization="configured",
            )
            perturbed = _replay(
                archive,
                _archive_data(result, perturb_after=split),
                initialization="configured",
            )
            baseline_outputs = np.asarray(
                baseline["best_simulation"]["outputs"], dtype=float
            )[:split]
            perturbed_outputs = np.asarray(
                perturbed["best_simulation"]["outputs"], dtype=float
            )[:split]
            causal_difference = _maximum_difference(
                baseline_outputs, perturbed_outputs
            )
            checks.append(
                _check(
                    "future_forcing_causality",
                    "Future-forcing causality",
                    "passed" if causal_difference <= comparison_tolerance else "failed",
                    (
                        "Changing every forcing channel after the split does not alter any "
                        "earlier model output."
                        if causal_difference <= comparison_tolerance
                        else "At least one output before the split changes when only future forcings change."
                    ),
                    maximum_absolute_difference=causal_difference,
                    tolerance=comparison_tolerance,
                    details={
                        "perturbation_starts_at_interval": split,
                        "earlier_intervals_compared": split,
                    },
                )
            )
        except Exception as exc:
            checks.append(
                _check(
                    "future_forcing_causality",
                    "Future-forcing causality",
                    "failed",
                    "The archive could not complete the future-forcing perturbation test.",
                    tolerance=comparison_tolerance,
                    details={"error": f"{type(exc).__name__}: {exc}"},
                )
            )

    counts = {
        status: sum(item["status"] == status for item in checks)
        for status in ("passed", "failed", "not_applicable")
    }
    overall_status = "passed" if counts["failed"] == 0 else "failed"
    return {
        "method_id": VERIFICATION_METHOD_ID,
        "task_id": str(archive.get("task_id", result.get("task_id", ""))),
        "experiment_id": config.experiment_id,
        "model_id": config.model.model_id,
        "status": overall_status,
        "checked_intervals": row_count,
        "tolerances": {
            "replay_restart_and_causality": comparison_tolerance,
            "tensorflow_restart_float32": restart_tolerance,
            "mass_balance_mm_per_interval": mass_balance_tolerance,
            "nonnegative_physical_quantities": nonnegative_tolerance,
        },
        "summary": counts,
        "checks": checks,
        "implementation_source": _implementation_source(config),
        "limitations": [
            "Passing numerical verification does not establish hydrological accuracy.",
            "Replay and restart checks reuse the archived parameters, weights, and transformations; they do not refit the model.",
            "Causality is tested by one deterministic future-forcing perturbation at the midpoint of this record.",
            "The recorded implementation source states the relationship to historical code and does not claim bitwise equivalence.",
        ],
    }
