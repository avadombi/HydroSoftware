"""Deterministic neural-initialization ensembles.

The ensemble setup is deliberately narrow.  Every member shares the same
data partitions, preprocessing recipe, architecture, and optimization
settings; only the recorded random seed changes.  The resulting spread
therefore describes sensitivity to neural initialization, not predictive,
forcing, parameter, or structural uncertainty.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray


ENSEMBLE_ARTIFACT_TYPE = "hydrosoftware.initialization_ensemble.v1"


def derive_member_seeds(base_seed: int, member_count: int) -> tuple[int, ...]:
    """Derive distinct reproducible 32-bit seeds from one recorded seed."""

    if member_count < 1:
        raise ValueError("member_count must be positive.")
    normalized_seed = int(base_seed) % (2**32)
    children = np.random.SeedSequence(normalized_seed).spawn(member_count)
    seeds = tuple(
        int(child.generate_state(1, dtype=np.uint32)[0])
        for child in children
    )
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("Seed derivation produced duplicate ensemble members.")
    return seeds


def aggregate_member_predictions(
    predictions: Sequence[NDArray],
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> tuple[NDArray[np.floating], NDArray[np.floating], NDArray[np.floating]]:
    """Return the member mean and an empirical central member envelope."""

    if not predictions:
        raise ValueError("At least one member prediction is required.")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Ensemble quantiles must satisfy 0 <= lower < upper <= 1.")
    stack = np.stack([np.asarray(values, dtype=float) for values in predictions])
    if stack.ndim != 2:
        raise ValueError("Every member prediction must be one-dimensional.")
    common_finite = np.all(np.isfinite(stack), axis=0)
    mean = np.full(stack.shape[1], np.nan, dtype=float)
    lower = np.full(stack.shape[1], np.nan, dtype=float)
    upper = np.full(stack.shape[1], np.nan, dtype=float)
    if np.any(common_finite):
        finite_stack = stack[:, common_finite]
        mean[common_finite] = np.mean(finite_stack, axis=0)
        lower[common_finite] = np.quantile(
            finite_stack, lower_quantile, axis=0
        )
        upper[common_finite] = np.quantile(
            finite_stack, upper_quantile, axis=0
        )
    return mean, lower, upper


def build_ensemble_artifact(
    member_artifacts: Sequence[Mapping[str, Any]],
    member_seeds: Sequence[int],
    *,
    lower_quantile: float,
    upper_quantile: float,
) -> dict[str, Any]:
    """Wrap frozen member artifacts without modifying their native payloads."""

    if len(member_artifacts) != len(member_seeds) or not member_artifacts:
        raise ValueError("Every ensemble artifact requires one seed per member.")
    return {
        "artifact_type": ENSEMBLE_ARTIFACT_TYPE,
        "method_id": "neural_initialization_ensemble.v1",
        "aggregation": "arithmetic_mean",
        "member_count": len(member_artifacts),
        "member_seeds": [int(seed) for seed in member_seeds],
        "lower_quantile": float(lower_quantile),
        "upper_quantile": float(upper_quantile),
        "members": [dict(artifact) for artifact in member_artifacts],
    }


def member_artifact_payloads(
    payload: Mapping[str, Any],
) -> tuple[Mapping[str, Any], ...]:
    """Return native member payloads for a single model or ensemble wrapper."""

    if str(payload.get("artifact_type", "")) != ENSEMBLE_ARTIFACT_TYPE:
        return (payload,)
    members = payload.get("members")
    if (
        not isinstance(members, Sequence)
        or isinstance(members, (str, bytes))
        or not members
        or not all(isinstance(member, Mapping) for member in members)
    ):
        raise ValueError("The initialization-ensemble artifact has no valid members.")
    declared = int(payload.get("member_count", len(members)))
    if declared != len(members):
        raise ValueError(
            "The initialization-ensemble member count does not match its artifacts."
        )
    return tuple(members)
