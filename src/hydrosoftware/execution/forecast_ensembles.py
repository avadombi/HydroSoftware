"""Frozen-model forecasts driven by multiple meteorological forcing scenarios.

The ensemble defined here is deliberately distinct from the neural
initialization ensemble.  Every forcing member uses the same frozen model and
the same initial hydrological information.  If that model is itself an
initialization ensemble, its archived arithmetic mean is evaluated separately
for each meteorological member.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from typing import Any
from uuid import uuid4

import numpy as np

from hydrosoftware.core.evaluation import evaluate_metrics
from hydrosoftware.core.results import SimulationResult
from hydrosoftware.data_sources.quality import DataQualityAnalysis
from hydrosoftware.execution.forecasting import run_forecast
from hydrosoftware.execution.serialization import to_jsonable


FORCING_ENSEMBLE_METHOD_ID = "meteorological_forcing_ensemble.v1"
MEMBERWISE_CYCLING_METHOD_ID = "memberwise_forecast_cycle.v1"


def empirical_crps(member_values: Sequence[float], observation: float) -> float:
    """Return the finite-ensemble CRPS for one forecast and observation.

    The implementation uses the standard energy representation

    ``mean(|x_m - y|) - 0.5 * mean(|x_m - x_n|)``.
    """

    members = np.asarray(member_values, dtype=float)
    observed = float(observation)
    if members.ndim != 1 or len(members) < 2:
        raise ValueError("CRPS requires at least two ensemble members.")
    if not np.all(np.isfinite(members)) or not np.isfinite(observed):
        raise ValueError("CRPS inputs must be finite.")
    observation_term = float(np.mean(np.abs(members - observed)))
    member_term = 0.5 * float(
        np.mean(np.abs(members[:, None] - members[None, :]))
    )
    return observation_term - member_term


def _as_array(value: Any, *, name: str, dimensions: int) -> np.ndarray:
    array = np.asarray(value, dtype=float)
    if array.ndim != dimensions:
        raise ValueError(
            f"Forecast ensemble member {name!r} must have {dimensions} dimensions."
        )
    return array


def _same_optional_series(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is None and right is None
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    return left_array.shape == right_array.shape and bool(
        np.allclose(left_array, right_array, rtol=0.0, atol=1e-12, equal_nan=True)
    )


def _aligned_member_results(
    member_results: Sequence[tuple[str, Mapping[str, Any]]],
) -> tuple[
    list[np.ndarray],
    list[np.ndarray],
    list[np.ndarray],
    tuple[str, ...],
    tuple[str, ...],
    tuple[str, ...],
]:
    reference_id, reference = member_results[0]
    reference_simulation = reference["best_simulation"]
    output_names = tuple(str(name) for name in reference_simulation["output_names"])
    state_names = tuple(str(name) for name in reference_simulation["state_names"])
    flux_names = tuple(str(name) for name in reference_simulation["flux_names"])
    outputs: list[np.ndarray] = []
    states: list[np.ndarray] = []
    fluxes: list[np.ndarray] = []
    reference_timestamps = reference.get("timestamps")
    reference_length = len(reference_simulation["outputs"])
    reference_finite: np.ndarray | None = None

    for member_id, result in member_results:
        simulation = result["best_simulation"]
        member_outputs = _as_array(
            simulation["outputs"], name=f"{member_id} outputs", dimensions=2
        )
        member_states = _as_array(
            simulation["states"], name=f"{member_id} states", dimensions=2
        )
        member_fluxes = _as_array(
            simulation["fluxes"], name=f"{member_id} fluxes", dimensions=2
        )
        if len(member_outputs) != reference_length:
            raise ValueError(
                "Every meteorological member must have the same forecast horizon."
            )
        if result.get("timestamps") != reference_timestamps:
            raise ValueError(
                "Every meteorological member must use identical forecast timestamps."
            )
        if (
            tuple(simulation["output_names"]) != output_names
            or tuple(simulation["state_names"]) != state_names
            or tuple(simulation["flux_names"]) != flux_names
        ):
            raise ValueError(
                "Every meteorological member must expose the same model variables."
            )
        if (
            member_outputs.shape != np.asarray(
                reference_simulation["outputs"], dtype=float
            ).shape
            or member_states.shape
            != np.asarray(reference_simulation["states"], dtype=float).shape
            or member_fluxes.shape
            != np.asarray(reference_simulation["fluxes"], dtype=float).shape
        ):
            raise ValueError(
                "Every meteorological member must share the same simulation layout."
            )
        finite = np.isfinite(member_outputs[:, 0])
        if reference_finite is None:
            reference_finite = finite
        elif not np.array_equal(finite, reference_finite):
            raise ValueError(
                "Every meteorological member must have predictions at the same leads."
            )
        outputs.append(member_outputs)
        states.append(member_states)
        fluxes.append(member_fluxes)

    if reference_id == "":
        raise ValueError("Meteorological member identifiers cannot be empty.")
    return outputs, states, fluxes, output_names, state_names, flux_names


def _common_observations(
    member_results: Sequence[tuple[str, Mapping[str, Any]]],
) -> np.ndarray | None:
    candidates = [
        (member_id, result.get("observations"))
        for member_id, result in member_results
        if result.get("observations") is not None
    ]
    if not candidates:
        return None
    reference_id, reference = candidates[0]
    for member_id, observations in candidates[1:]:
        if not _same_optional_series(reference, observations):
            raise ValueError(
                "Observed streamflow differs between meteorological members "
                f"{reference_id!r} and {member_id!r}."
            )
    return np.asarray(reference, dtype=float)


def _mean_forcings(
    member_results: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, np.ndarray]:
    reference_keys = tuple(member_results[0][1]["forcings"].keys())
    forcings: dict[str, np.ndarray] = {}
    for member_id, result in member_results:
        if tuple(result["forcings"].keys()) != reference_keys:
            raise ValueError(
                f"Meteorological member {member_id!r} does not expose the same "
                "ordered forcing channels as the other members."
            )
    for name in reference_keys:
        stack = np.stack(
            [
                np.asarray(result["forcings"][name], dtype=float)
                for _, result in member_results
            ]
        )
        if stack.ndim != 2 or not np.all(np.isfinite(stack)):
            raise ValueError(
                f"Meteorological forcing channel {name!r} must be finite and aligned."
            )
        forcings[name] = np.mean(stack, axis=0)
    return forcings


def _prepare_forecast_period(
    member_results: Sequence[tuple[str, Mapping[str, Any]]],
    *,
    spinup_steps: int,
) -> list[tuple[str, dict[str, Any]]]:
    """Validate shared antecedent forcings and exclude them from forecast leads."""

    if spinup_steps < 0:
        raise ValueError("Forecast-ensemble spin-up steps cannot be negative.")
    horizon = len(member_results[0][1]["best_simulation"]["outputs"])
    if spinup_steps >= horizon:
        raise ValueError(
            "Forecast-ensemble spin-up must leave at least one forecast lead."
        )
    if spinup_steps == 0:
        return [(member_id, dict(result)) for member_id, result in member_results]

    reference_forcings = member_results[0][1].get("forcings") or {}
    for member_id, result in member_results[1:]:
        candidate = result.get("forcings") or {}
        if tuple(candidate.keys()) != tuple(reference_forcings.keys()):
            raise ValueError(
                f"Meteorological member {member_id!r} does not expose the same "
                "spin-up forcing channels."
            )
        for name, reference_values in reference_forcings.items():
            left = np.asarray(reference_values, dtype=float)[:spinup_steps]
            right = np.asarray(candidate[name], dtype=float)[:spinup_steps]
            if (
                left.shape != right.shape
                or not np.allclose(
                    left,
                    right,
                    rtol=0.0,
                    atol=1e-12,
                    equal_nan=True,
                )
            ):
                raise ValueError(
                    "Every meteorological member must use identical observed "
                    f"spin-up values for channel {name!r}."
                )

    prepared: list[tuple[str, dict[str, Any]]] = []
    for member_id, result_mapping in member_results:
        result = dict(result_mapping)
        simulation = dict(result["best_simulation"])
        simulation["outputs"] = simulation["outputs"][spinup_steps:]
        simulation["states"] = simulation["states"][spinup_steps:]
        simulation["fluxes"] = simulation["fluxes"][spinup_steps:]
        diagnostics = dict(simulation.get("diagnostics") or {})
        for name in (
            "initialization_ensemble_lower",
            "initialization_ensemble_upper",
        ):
            values = diagnostics.get(name)
            if isinstance(values, Sequence) and not isinstance(
                values,
                (str, bytes),
            ) and len(values) == horizon:
                diagnostics[name] = values[spinup_steps:]
        simulation["diagnostics"] = diagnostics
        result["best_simulation"] = simulation
        if result.get("timestamps") is not None:
            result["timestamps"] = result["timestamps"][spinup_steps:]
        if result.get("observations") is not None:
            result["observations"] = result["observations"][spinup_steps:]
        result["forcings"] = {
            name: values[spinup_steps:]
            for name, values in (result.get("forcings") or {}).items()
        }
        observations = result.get("observations")
        outputs = np.asarray(simulation["outputs"], dtype=float)
        if observations is not None:
            observed = np.asarray(observations, dtype=float)
            mask = np.isfinite(observed) & np.isfinite(outputs[:, 0])
            result["evaluation_metrics"] = (
                evaluate_metrics(observed, outputs[:, 0], mask)
                if np.any(mask)
                else ()
            )
        prepared.append((member_id, result))
    return prepared


def _daily_date(value: Any, *, label: str) -> date:
    """Parse one archived daily timestamp without weakening chronology checks."""

    text = str(value).strip()
    if not text:
        raise ValueError(f"{label} is missing.")
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        try:
            return date.fromisoformat(text[:10])
        except ValueError as exc:
            raise ValueError(
                f"{label} must be an ISO daily timestamp; received {text!r}."
            ) from exc


def _continuation_members(
    continuation_result: Mapping[str, Any],
    member_ids: Sequence[str],
    members: Sequence[tuple[str, DataQualityAnalysis, bool]],
) -> tuple[dict[str, Mapping[str, Any]], int]:
    """Validate one fixed member-wise source cycle and its next daily period."""

    ensemble = continuation_result.get("forecast_ensemble") or {}
    if ensemble.get("method_id") != FORCING_ENSEMBLE_METHOD_ID:
        raise ValueError(
            "Member-wise continuation requires a completed meteorological "
            "forcing-ensemble source."
        )
    archived_members = ensemble.get("members")
    if not isinstance(archived_members, Sequence) or isinstance(
        archived_members, (str, bytes)
    ):
        raise ValueError("The source ensemble does not contain member records.")
    by_id: dict[str, Mapping[str, Any]] = {}
    for item in archived_members:
        if not isinstance(item, Mapping):
            raise ValueError("A source forcing-member record is invalid.")
        member_id = str(item.get("member_id", ""))
        if not member_id or member_id in by_id:
            raise ValueError(
                "Source forcing-member identifiers must be non-empty and unique."
            )
        by_id[member_id] = item
    if set(by_id) != set(member_ids) or len(by_id) != len(member_ids):
        missing = sorted(set(by_id) - set(member_ids))
        extra = sorted(set(member_ids) - set(by_id))
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if extra:
            details.append("unexpected " + ", ".join(extra))
        raise ValueError(
            "A continued ensemble must preserve the exact meteorological member "
            "identities"
            + (f" ({'; '.join(details)})." if details else ".")
        )

    previous_timestamps = continuation_result.get("timestamps")
    if not isinstance(previous_timestamps, Sequence) or isinstance(
        previous_timestamps, (str, bytes)
    ) or not previous_timestamps:
        raise ValueError(
            "Member-wise continuation requires timestamps in the source ensemble."
        )
    next_timestamps = members[0][1].timestamps
    if not next_timestamps:
        raise ValueError(
            "Member-wise continuation requires timestamps in every new member."
        )
    previous_last = _daily_date(
        previous_timestamps[-1],
        label="The source ensemble's last valid time",
    )
    next_first = _daily_date(
        next_timestamps[0],
        label="The continued ensemble's first input time",
    )
    expected = previous_last + timedelta(days=1)
    if next_first != expected:
        raise ValueError(
            "A member-wise continuation must start on the next daily interval "
            f"after the source ensemble ({expected.isoformat()}); received "
            f"{next_first.isoformat()}."
        )

    prior_cycle = int(ensemble.get("cycle_index", 1))
    if prior_cycle < 1:
        raise ValueError("The source ensemble cycle index must be positive.")
    return by_id, prior_cycle


def _lead_diagnostics(
    predictions: np.ndarray,
    observations: np.ndarray | None,
    timestamps: Sequence[str] | None,
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[list[dict[str, Any]], dict[str, float | int | None]]:
    diagnostics: list[dict[str, Any]] = []
    crps_values: list[float] = []
    signed_errors: list[float] = []
    absolute_errors: list[float] = []
    squared_errors: list[float] = []
    spreads: list[float] = []

    for index in range(predictions.shape[1]):
        values = predictions[:, index]
        finite = np.isfinite(values)
        valid_values = values[finite]
        observation = (
            float(observations[index])
            if observations is not None and np.isfinite(observations[index])
            else None
        )
        if len(valid_values) == 0:
            mean = median = lower = upper = spread = None
        else:
            mean = float(np.mean(valid_values))
            median = float(np.median(valid_values))
            lower = float(np.quantile(valid_values, lower_quantile))
            upper = float(np.quantile(valid_values, upper_quantile))
            spread = float(np.std(valid_values, ddof=0))
            spreads.append(spread)
        signed_error = absolute_error = crps = None
        if observation is not None and len(valid_values) == predictions.shape[0]:
            signed_error = float(mean - observation)
            absolute_error = abs(signed_error)
            crps = empirical_crps(valid_values, observation)
            signed_errors.append(signed_error)
            absolute_errors.append(absolute_error)
            squared_errors.append(signed_error**2)
            crps_values.append(crps)
        diagnostics.append(
            {
                "lead_step": index + 1,
                "timestamp": timestamps[index] if timestamps is not None else None,
                "valid_members": int(len(valid_values)),
                "observation": observation,
                "ensemble_mean": mean,
                "ensemble_median": median,
                "lower": lower,
                "upper": upper,
                "spread": spread,
                "signed_error": signed_error,
                "absolute_error": absolute_error,
                "crps": crps,
            }
        )

    scored = len(crps_values)
    aggregate: dict[str, float | int | None] = {
        "scored_leads": scored,
        "mean_crps": float(np.mean(crps_values)) if scored else None,
        "mean_error_bias": float(np.mean(signed_errors)) if scored else None,
        "mean_absolute_error": float(np.mean(absolute_errors)) if scored else None,
        "root_mean_square_error": (
            float(np.sqrt(np.mean(squared_errors))) if scored else None
        ),
        "mean_ensemble_spread": float(np.mean(spreads)) if spreads else None,
    }
    return diagnostics, aggregate


def run_forcing_ensemble_forecast(
    source_archive: Mapping[str, Any],
    members: Sequence[tuple[str, DataQualityAnalysis, bool]],
    *,
    initialization: str,
    continuation_result: Mapping[str, Any] | None = None,
    issue_id: str | None = None,
    issue_time: str | None = None,
    spinup_steps: int = 0,
    custom_initial_state: Sequence[float] | None = None,
    antecedent_forcings: Mapping[str, Sequence[float]] | None = None,
    lower_quantile: float = 0.1,
    upper_quantile: float = 0.9,
) -> dict[str, Any]:
    """Apply one frozen model to aligned meteorological forcing scenarios.

    A continuation never initializes from the displayed aggregate state.  Each
    meteorological member is matched by its stable identifier to its own final
    storage vector and, when present, its own neural antecedent context.
    """

    if not 2 <= len(members) <= 50:
        raise ValueError(
            "A meteorological forcing ensemble requires between 2 and 50 members."
        )
    member_ids = [member_id for member_id, _, _ in members]
    if len(set(member_ids)) != len(member_ids):
        raise ValueError("Meteorological forcing member identifiers must be unique.")
    if any(not member_id for member_id in member_ids):
        raise ValueError("Meteorological forcing member identifiers cannot be empty.")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Forecast ensemble quantiles must satisfy 0 <= lower < upper <= 1.")

    continuation_members: dict[str, Mapping[str, Any]] | None = None
    prior_cycle = 0
    if continuation_result is not None:
        if initialization != "source_final":
            raise ValueError(
                "A member-wise ensemble continuation must use source_final "
                "initialization."
            )
        if custom_initial_state is not None or antecedent_forcings is not None:
            raise ValueError(
                "Aggregate custom states or antecedent contexts cannot be supplied "
                "for member-wise continuation."
            )
        if (
            str(continuation_result.get("source_task_id", ""))
            != str(source_archive.get("task_id", ""))
        ):
            raise ValueError(
                "The continued ensemble and frozen base model do not share the "
                "same source task."
            )
        if (
            str(continuation_result.get("configuration_hash", ""))
            != str((source_archive.get("result") or {}).get("configuration_hash", ""))
            or str(continuation_result.get("model_id", ""))
            != str((source_archive.get("config") or {}).get("model", {}).get("model_id", ""))
        ):
            raise ValueError(
                "The continued ensemble does not match the frozen base model "
                "configuration."
            )
        continuation_members, prior_cycle = _continuation_members(
            continuation_result,
            member_ids,
            members,
        )

    full_member_results: list[tuple[str, Mapping[str, Any]]] = []
    for member_id, data, observations_available in members:
        member_initialization = initialization
        member_state = custom_initial_state
        member_context = antecedent_forcings
        if continuation_members is not None:
            previous = continuation_members[member_id]
            raw_state = previous.get("final_state")
            raw_context = previous.get("neural_context")
            has_state = (
                isinstance(raw_state, Sequence)
                and not isinstance(raw_state, (str, bytes))
                and len(raw_state) > 0
            )
            has_context = isinstance(raw_context, Mapping) and bool(raw_context)
            if has_state:
                member_initialization = "custom"
                member_state = tuple(float(value) for value in raw_state)
                member_context = raw_context if has_context else None
            elif has_context:
                # Standalone neural models do not use storage states.  Their
                # source_final path accepts the explicitly archived member context.
                member_initialization = "source_final"
                member_state = None
                member_context = raw_context
            else:
                raise ValueError(
                    f"Source member {member_id!r} lacks the state or neural context "
                    "required for continuation. Re-run its preceding cycle with "
                    "a HydroSoftware archive that stores member-wise continuation state."
                )
        full_member_results.append(
            (
                member_id,
                run_forecast(
                    source_archive,
                    data,
                    initialization=member_initialization,
                    observations_available=observations_available,
                    custom_initial_state=member_state,
                    antecedent_forcings=member_context,
                ),
            )
        )
    member_results = _prepare_forecast_period(
        full_member_results,
        spinup_steps=spinup_steps,
    )
    (
        output_arrays,
        state_arrays,
        flux_arrays,
        output_names,
        state_names,
        flux_names,
    ) = _aligned_member_results(member_results)
    output_stack = np.stack(output_arrays)
    state_stack = np.stack(state_arrays)
    flux_stack = np.stack(flux_arrays)
    streamflow_members = output_stack[:, :, 0]
    common_finite = np.all(np.isfinite(streamflow_members), axis=0)
    mean_outputs = np.full(output_stack.shape[1:], np.nan, dtype=float)
    lower_outputs = np.full(output_stack.shape[1:], np.nan, dtype=float)
    upper_outputs = np.full(output_stack.shape[1:], np.nan, dtype=float)
    for position in range(output_stack.shape[2]):
        finite = np.all(np.isfinite(output_stack[:, :, position]), axis=0)
        mean_outputs[finite, position] = np.mean(
            output_stack[:, finite, position], axis=0
        )
        lower_outputs[finite, position] = np.quantile(
            output_stack[:, finite, position], lower_quantile, axis=0
        )
        upper_outputs[finite, position] = np.quantile(
            output_stack[:, finite, position], upper_quantile, axis=0
        )
    mean_states = np.mean(state_stack, axis=0)
    mean_fluxes = np.mean(flux_stack, axis=0)
    first_result = member_results[0][1]
    timestamps = first_result.get("timestamps")
    observations = _common_observations(member_results)
    lead_diagnostics, aggregate_scores = _lead_diagnostics(
        streamflow_members,
        observations,
        timestamps,
        lower_quantile=lower_quantile,
        upper_quantile=upper_quantile,
    )
    metrics = ()
    if observations is not None:
        score_mask = common_finite & np.isfinite(observations)
        if np.any(score_mask):
            metrics = evaluate_metrics(
                observations,
                mean_outputs[:, 0],
                score_mask,
            )
    forcing_means = _mean_forcings(member_results)
    balance_errors = [
        float(result["best_simulation"]["diagnostics"][
            "maximum_absolute_mass_balance_error"
        ])
        for _, result in member_results
        if isinstance(
            result["best_simulation"].get("diagnostics", {}).get(
                "maximum_absolute_mass_balance_error"
            ),
            (int, float),
        )
    ]
    simulation_diagnostics = {
        **dict(first_result["best_simulation"].get("diagnostics") or {}),
        "engine_id": FORCING_ENSEMBLE_METHOD_ID,
        "forcing_ensemble_method": FORCING_ENSEMBLE_METHOD_ID,
        "forcing_ensemble_members": len(member_results),
        "forcing_ensemble_lower_quantile": lower_quantile,
        "forcing_ensemble_upper_quantile": upper_quantile,
        "forcing_ensemble_lower": lower_outputs[:, 0],
        "forcing_ensemble_upper": upper_outputs[:, 0],
        "process_trajectory": "arithmetic_member_mean",
    }
    if balance_errors:
        simulation_diagnostics["maximum_absolute_mass_balance_error"] = max(
            balance_errors
        )
    simulation = SimulationResult(
        outputs=mean_outputs,
        states=mean_states,
        fluxes=mean_fluxes,
        output_names=output_names,
        state_names=state_names,
        flux_names=flux_names,
        diagnostics=simulation_diagnostics,
    )
    member_payloads = []
    for member_id, result in member_results:
        member_simulation = result["best_simulation"]
        diagnostics = member_simulation.get("diagnostics") or {}
        member_payloads.append(
            {
                "member_id": member_id,
                "predictions": member_simulation["outputs"],
                "initial_state": result.get("initial_state"),
                "final_state": member_simulation["states"][-1],
                "neural_context": result.get("neural_context"),
                "evaluation_metrics": result.get("evaluation_metrics") or [],
                "initialization_ensemble_lower": diagnostics.get(
                    "initialization_ensemble_lower"
                ),
                "initialization_ensemble_upper": diagnostics.get(
                    "initialization_ensemble_upper"
                ),
            }
        )

    transformed_count = sum(
        int(result["validation"].get("transformed_value_count", 0))
        for _, result in member_results
    )
    warning_count = sum(
        int(result["validation"].get("data_quality_warning_count", 0))
        for _, result in member_results
    )
    scored = int(
        np.count_nonzero(common_finite & np.isfinite(observations))
        if observations is not None
        else 0
    )
    task_id = f"forecast-ensemble-{uuid4()}"
    resolved_issue_id = str(
        issue_id
        or (timestamps[0] if timestamps else task_id)
    )
    result = {
        "task_id": task_id,
        "run_kind": "forecast",
        "source_task_id": str(source_archive.get("task_id")),
        "continuation_task_id": (
            str(continuation_result.get("task_id"))
            if continuation_result is not None
            else None
        ),
        "state": "completed",
        "experiment_id": first_result["experiment_id"],
        "configuration_hash": first_result["configuration_hash"],
        "model_id": first_result["model_id"],
        "initialization": (
            "member_final" if continuation_result is not None else initialization
        ),
        "initial_state": first_result.get("initial_state"),
        "validation": {
            **dict(first_result["validation"]),
            "source_rows": len(mean_outputs),
            "selected_rows": len(mean_outputs),
            "scored_observations": scored,
            "transformed_value_count": transformed_count,
            "data_quality_warning_count": warning_count,
            "warnings": [
                {
                    "code": "forcing_ensemble_scope",
                    "message": (
                        "The ensemble spread reflects the supplied meteorological "
                        "forcing scenarios only. It is not a calibrated predictive "
                        "interval."
                    ),
                },
                *(
                    [
                        {
                            "code": "memberwise_forecast_cycle",
                            "message": (
                                "Every forcing member was initialized from its "
                                "matching archived state and antecedent context."
                            ),
                        }
                    ]
                    if continuation_result is not None
                    else []
                ),
            ],
        },
        "calibration": first_result.get("calibration"),
        "training": first_result.get("training"),
        "best_simulation": simulation,
        "timestamps": timestamps,
        "observations": observations,
        "forcings": forcing_means,
        "evaluation_metrics": metrics,
        "data_quality": {
            "member_count": len(member_results),
            "member_reports": [
                {
                    "member_id": member_id,
                    "report": result.get("data_quality"),
                }
                for member_id, result in member_results
            ],
        },
        "forecast_ensemble": {
            "method_id": FORCING_ENSEMBLE_METHOD_ID,
            "cycling_method_id": (
                MEMBERWISE_CYCLING_METHOD_ID
                if continuation_result is not None
                else None
            ),
            "cycle_index": prior_cycle + 1,
            "parent_task_id": (
                str(continuation_result.get("task_id"))
                if continuation_result is not None
                else None
            ),
            "parent_issue_id": (
                str(
                    (
                        continuation_result.get("forecast_ensemble") or {}
                    ).get("issue_id")
                )
                if continuation_result is not None
                else None
            ),
            "issue_id": resolved_issue_id,
            "issue_time": issue_time,
            "first_valid_time": timestamps[0] if timestamps else None,
            "spinup_steps": spinup_steps,
            "scope": "meteorological_forcing_scenarios",
            "member_count": len(member_results),
            "member_ids": member_ids,
            "aggregation": "arithmetic_mean",
            "lower_quantile": lower_quantile,
            "upper_quantile": upper_quantile,
            "members": member_payloads,
            "lead_time_diagnostics": lead_diagnostics,
            "aggregate_scores": aggregate_scores,
            "warnings": [
                {
                    "code": "forcing_ensemble_not_total_uncertainty",
                    "message": (
                        "The member envelope and CRPS use only the supplied forcing "
                        "scenarios. Parameter, structural and observation uncertainty "
                        "are not represented."
                    ),
                },
                {
                    "code": "ensemble_process_mean",
                    "message": (
                        "Displayed internal states and fluxes are arithmetic member "
                        "means, not one simulated member trajectory."
                    ),
                },
                {
                    "code": "memberwise_state_preserved",
                    "message": (
                        "Member states and neural contexts are archived separately. "
                        "A later cycle can continue them only by matching member "
                        "identifier."
                    ),
                },
            ],
        },
    }
    return to_jsonable(result)
