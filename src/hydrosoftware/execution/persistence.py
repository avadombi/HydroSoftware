"""Atomic local persistence for experiment definitions and terminal run archives."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from threading import RLock
from typing import Any

from hydrosoftware.configuration import ExperimentConfigV1, configuration_hash
from hydrosoftware.execution.experiments import ExperimentRunResult


STORE_VERSION = "1.0"
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class ProjectStoreError(RuntimeError):
    """Stable persistence failure that can be translated at the API boundary."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class SavedExperimentSummary:
    """Small experiment descriptor used by the project picker."""

    experiment_id: str
    name: str
    description: str
    configuration_hash: str
    created_at: str
    updated_at: str
    latest_run_id: str | None
    run_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "configuration_hash": self.configuration_hash,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "latest_run_id": self.latest_run_id,
            "run_count": self.run_count,
        }


class ProjectStore:
    """Persist portable JSON records beneath one workspace-local state directory."""

    def __init__(self, workspace_root: str | Path) -> None:
        self.workspace_root = Path(workspace_root).resolve()
        self.root = self.workspace_root / ".hydrosoftware"
        self.experiments_dir = self.root / "experiments"
        self.runs_dir = self.root / "runs"
        self.temporal_validations_dir = self.root / "temporal-validations"
        self.forecast_ensembles_dir = self.root / "forecast-ensembles"
        self.forecast_verifications_dir = self.root / "forecast-verifications"
        self.forecast_postprocessings_dir = self.root / "forecast-postprocessings"
        self.forecast_event_verifications_dir = (
            self.root / "forecast-event-verifications"
        )
        self._lock = RLock()

    @staticmethod
    def _identifier(value: str, *, kind: str) -> str:
        if not _IDENTIFIER.fullmatch(value):
            raise ProjectStoreError(
                f"invalid_{kind}_identifier",
                f"The {kind} identifier contains unsupported characters.",
            )
        return value

    @staticmethod
    def _read_json(path: Path, *, missing_code: str) -> dict[str, Any]:
        if not path.is_file():
            raise ProjectStoreError(missing_code, f"No saved record exists at {path.name!r}.")
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ProjectStoreError(
                "saved_record_unreadable",
                f"The saved record {path.name!r} could not be read: {exc}",
            ) from exc
        if not isinstance(payload, dict):
            raise ProjectStoreError(
                "saved_record_invalid",
                f"The saved record {path.name!r} does not contain a JSON object.",
            )
        return payload

    @staticmethod
    def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        except (OSError, TypeError, ValueError) as exc:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
            raise ProjectStoreError(
                "saved_record_write_failed",
                f"The record {path.name!r} could not be saved: {exc}",
            ) from exc

    def _experiment_path(self, experiment_id: str) -> Path:
        identifier = self._identifier(experiment_id, kind="experiment")
        return self.experiments_dir / f"{identifier}.json"

    def _run_path(self, task_id: str) -> Path:
        identifier = self._identifier(task_id, kind="run")
        return self.runs_dir / f"{identifier}.json"

    def _temporal_validation_path(self, task_id: str) -> Path:
        identifier = self._identifier(task_id, kind="temporal_validation")
        return self.temporal_validations_dir / f"{identifier}.json"

    def _forecast_ensemble_path(self, task_id: str) -> Path:
        identifier = self._identifier(task_id, kind="forecast_ensemble")
        return self.forecast_ensembles_dir / f"{identifier}.json"

    def _forecast_verification_path(self, task_id: str) -> Path:
        identifier = self._identifier(task_id, kind="forecast_verification")
        return self.forecast_verifications_dir / f"{identifier}.json"

    def _forecast_postprocessing_path(self, task_id: str) -> Path:
        identifier = self._identifier(task_id, kind="forecast_postprocessing")
        return self.forecast_postprocessings_dir / f"{identifier}.json"

    def _forecast_event_verification_path(self, task_id: str) -> Path:
        identifier = self._identifier(task_id, kind="forecast_event_verification")
        return self.forecast_event_verifications_dir / f"{identifier}.json"

    def save_experiment(self, config: ExperimentConfigV1) -> SavedExperimentSummary:
        """Create or replace one definition while preserving its creation time."""

        with self._lock:
            path = self._experiment_path(config.experiment_id)
            existing: dict[str, Any] | None = None
            if path.is_file():
                existing = self._read_json(path, missing_code="experiment_not_found")
            now = _utc_now()
            created_at = str(existing.get("created_at", now)) if existing else now
            latest_run_id = existing.get("latest_run_id") if existing else None
            run_count = int(existing.get("run_count", 0)) if existing else 0
            payload = {
                "record_type": "hydrosoftware.saved-experiment",
                "record_version": STORE_VERSION,
                "experiment_id": config.experiment_id,
                "configuration_hash": configuration_hash(config),
                "created_at": created_at,
                "updated_at": now,
                "latest_run_id": latest_run_id,
                "run_count": run_count,
                "config": config.model_dump(mode="json"),
            }
            self._write_json_atomic(path, payload)
            return self._summary(payload)

    @staticmethod
    def _summary(payload: dict[str, Any]) -> SavedExperimentSummary:
        try:
            config = ExperimentConfigV1.model_validate(payload["config"])
            return SavedExperimentSummary(
                experiment_id=config.experiment_id,
                name=config.name,
                description=config.description,
                configuration_hash=str(payload["configuration_hash"]),
                created_at=str(payload["created_at"]),
                updated_at=str(payload["updated_at"]),
                latest_run_id=(
                    str(payload["latest_run_id"])
                    if payload.get("latest_run_id") is not None
                    else None
                ),
                run_count=int(payload.get("run_count", 0)),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectStoreError(
                "saved_experiment_invalid",
                f"A saved experiment record is invalid: {exc}",
            ) from exc

    def load_experiment(self, experiment_id: str) -> dict[str, Any]:
        """Return a validated saved experiment record and configuration."""

        payload = self._read_json(
            self._experiment_path(experiment_id),
            missing_code="experiment_not_found",
        )
        summary = self._summary(payload)
        config = ExperimentConfigV1.model_validate(payload["config"])
        return {"summary": summary.as_dict(), "config": config.model_dump(mode="json")}

    def list_experiments(self) -> tuple[SavedExperimentSummary, ...]:
        """List valid experiment records in reverse modification order."""

        if not self.experiments_dir.is_dir():
            return ()
        summaries = [
            self._summary(self._read_json(path, missing_code="experiment_not_found"))
            for path in self.experiments_dir.glob("*.json")
        ]
        return tuple(sorted(summaries, key=lambda item: item.updated_at, reverse=True))

    def save_run(
        self,
        config: ExperimentConfigV1,
        result: ExperimentRunResult,
    ) -> dict[str, Any]:
        """Archive a terminal result and link it to its experiment atomically per file."""

        with self._lock:
            saved = self.save_experiment(config)
            now = _utc_now()
            run_path = self._run_path(result.task_id)
            is_new_archive = not run_path.is_file()
            run_payload = {
                "record_type": "hydrosoftware.run-archive",
                "record_version": STORE_VERSION,
                "task_id": result.task_id,
                "experiment_id": result.experiment_id,
                "configuration_hash": result.configuration_hash,
                "state": result.state.value,
                "saved_at": now,
                "config": config.model_dump(mode="json"),
                "result": result.as_dict(),
            }
            self._write_json_atomic(run_path, run_payload)

            experiment_path = self._experiment_path(config.experiment_id)
            experiment_payload = self._read_json(
                experiment_path,
                missing_code="experiment_not_found",
            )
            experiment_payload["latest_run_id"] = result.task_id
            experiment_payload["run_count"] = saved.run_count + int(is_new_archive)
            experiment_payload["updated_at"] = now
            self._write_json_atomic(experiment_path, experiment_payload)
            return self._run_summary(run_payload)

    @staticmethod
    def _run_summary(payload: dict[str, Any]) -> dict[str, Any]:
        result = payload.get("result") or {}
        calibration = result.get("calibration") or {}
        training = result.get("training") or {}
        config = payload.get("config") or {}
        model = config.get("model") or {}
        return {
            "task_id": str(payload["task_id"]),
            "experiment_id": str(payload["experiment_id"]),
            "experiment_name": str(config.get("name", payload["experiment_id"])),
            "model_id": model.get("model_id"),
            "run_kind": result.get("run_kind", "calibration"),
            "configuration_hash": str(payload["configuration_hash"]),
            "state": str(payload["state"]),
            "saved_at": str(payload["saved_at"]),
            "objective_name": calibration.get("objective_name") or training.get("selection_metric"),
            "best_score": calibration.get("best_score") if calibration else training.get("best_validation_score"),
        }

    def load_run(self, task_id: str) -> dict[str, Any]:
        """Return one complete immutable run archive."""

        payload = self._read_json(
            self._run_path(task_id),
            missing_code="run_archive_not_found",
        )
        required = {"task_id", "experiment_id", "config", "result"}
        if not required.issubset(payload):
            raise ProjectStoreError(
                "saved_run_invalid",
                f"Run archive {task_id!r} is missing required fields.",
            )
        try:
            ExperimentConfigV1.model_validate(payload["config"])
        except (TypeError, ValueError) as exc:
            message = str(exc)
            raise ProjectStoreError(
                "saved_run_invalid",
                f"Run archive {task_id!r} cannot be opened: {message}",
            ) from exc
        return payload

    def list_runs(self, experiment_id: str) -> tuple[dict[str, Any], ...]:
        """List terminal archives associated with one saved experiment."""

        self._identifier(experiment_id, kind="experiment")
        if not self.runs_dir.is_dir():
            return ()
        summaries: list[dict[str, Any]] = []
        for path in self.runs_dir.glob("*.json"):
            payload = self._read_json(path, missing_code="run_archive_not_found")
            if payload.get("experiment_id") == experiment_id:
                summaries.append(self._run_summary(payload))
        return tuple(sorted(summaries, key=lambda item: item["saved_at"], reverse=True))

    def save_temporal_validation(
        self,
        task_id: str,
        result: dict[str, Any] | Any,
    ) -> dict[str, Any]:
        """Persist one completed rolling-origin analysis as an immutable archive."""

        with self._lock:
            if str(result.get("state")) != "completed":
                raise ProjectStoreError(
                    "temporal_validation_not_completed",
                    "Only completed temporal validations can be archived.",
                )
            now = _utc_now()
            payload = {
                "record_type": "hydrosoftware.temporal-validation",
                "record_version": STORE_VERSION,
                "task_id": task_id,
                "saved_at": now,
                "result": result,
            }
            self._write_json_atomic(
                self._temporal_validation_path(task_id),
                payload,
            )
            return self._temporal_validation_summary(payload)

    @staticmethod
    def _temporal_validation_summary(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            result = payload["result"]
            settings = result["settings"]
            models = result["models"]
            return {
                "task_id": str(payload["task_id"]),
                "saved_at": str(payload["saved_at"]),
                "method_id": str(result["method_id"]),
                "state": str(result["state"]),
                "window_mode": str(settings["window_mode"]),
                "folds": int(settings["folds"]),
                "test_steps": int(settings["test_steps"]),
                "pooled_scored_observations": int(
                    result["pooled_scored_observations"]
                ),
                "experiment_ids": [
                    str(model["experiment_id"]) for model in models
                ],
                "experiment_names": [
                    str(model["experiment_name"]) for model in models
                ],
                "model_ids": [str(model["model_id"]) for model in models],
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectStoreError(
                "saved_temporal_validation_invalid",
                f"A temporal-validation archive is invalid: {exc}",
            ) from exc

    def list_temporal_validations(self) -> tuple[dict[str, Any], ...]:
        """List rolling-origin archives in reverse creation order."""

        if not self.temporal_validations_dir.is_dir():
            return ()
        summaries = [
            self._temporal_validation_summary(
                self._read_json(
                    path,
                    missing_code="temporal_validation_archive_not_found",
                )
            )
            for path in self.temporal_validations_dir.glob("*.json")
        ]
        return tuple(
            sorted(
                summaries,
                key=lambda item: str(item["saved_at"]),
                reverse=True,
            )
        )

    def load_temporal_validation(self, task_id: str) -> dict[str, Any]:
        """Return one complete immutable rolling-origin archive."""

        payload = self._read_json(
            self._temporal_validation_path(task_id),
            missing_code="temporal_validation_archive_not_found",
        )
        self._temporal_validation_summary(payload)
        return payload

    def save_forecast_ensemble(self, result: dict[str, Any]) -> dict[str, Any]:
        """Persist one completed forcing-ensemble forecast."""

        with self._lock:
            if (
                str(result.get("state")) != "completed"
                or str((result.get("forecast_ensemble") or {}).get("method_id"))
                != "meteorological_forcing_ensemble.v1"
            ):
                raise ProjectStoreError(
                    "forecast_ensemble_not_completed",
                    "Only completed meteorological forcing ensembles can be archived.",
                )
            task_id = str(result.get("task_id", ""))
            now = _utc_now()
            payload = {
                "record_type": "hydrosoftware.forecast-ensemble",
                "record_version": STORE_VERSION,
                "task_id": task_id,
                "saved_at": now,
                "result": result,
            }
            self._write_json_atomic(
                self._forecast_ensemble_path(task_id),
                payload,
            )
            return self._forecast_ensemble_summary(payload)

    @staticmethod
    def _forecast_ensemble_summary(payload: dict[str, Any]) -> dict[str, Any]:
        try:
            result = payload["result"]
            ensemble = result["forecast_ensemble"]
            return {
                "task_id": str(payload["task_id"]),
                "saved_at": str(payload["saved_at"]),
                "source_task_id": str(result["source_task_id"]),
                "continuation_task_id": (
                    str(result["continuation_task_id"])
                    if result.get("continuation_task_id") is not None
                    else None
                ),
                "experiment_id": str(result["experiment_id"]),
                "model_id": str(result["model_id"]),
                "issue_id": str(ensemble["issue_id"]),
                "issue_time": (
                    str(ensemble["issue_time"])
                    if ensemble.get("issue_time") is not None
                    else None
                ),
                "first_valid_time": (
                    str(ensemble["first_valid_time"])
                    if ensemble.get("first_valid_time") is not None
                    else None
                ),
                "member_count": int(ensemble["member_count"]),
                "horizon": len(result["best_simulation"]["outputs"]),
                "spinup_steps": int(ensemble.get("spinup_steps", 0)),
                "cycle_index": int(ensemble.get("cycle_index", 1)),
                "parent_task_id": (
                    str(ensemble["parent_task_id"])
                    if ensemble.get("parent_task_id") is not None
                    else None
                ),
                "scored_leads": int(
                    ensemble["aggregate_scores"]["scored_leads"]
                ),
                "lower_quantile": float(ensemble["lower_quantile"]),
                "upper_quantile": float(ensemble["upper_quantile"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectStoreError(
                "saved_forecast_ensemble_invalid",
                f"A forcing-ensemble archive is invalid: {exc}",
            ) from exc

    def list_forecast_ensembles(self) -> tuple[dict[str, Any], ...]:
        """List forcing-ensemble archives in reverse creation order."""

        if not self.forecast_ensembles_dir.is_dir():
            return ()
        summaries = [
            self._forecast_ensemble_summary(
                self._read_json(
                    path,
                    missing_code="forecast_ensemble_archive_not_found",
                )
            )
            for path in self.forecast_ensembles_dir.glob("*.json")
        ]
        return tuple(
            sorted(
                summaries,
                key=lambda item: str(item["saved_at"]),
                reverse=True,
            )
        )

    def load_forecast_ensemble(self, task_id: str) -> dict[str, Any]:
        """Return one complete immutable forcing-ensemble archive."""

        payload = self._read_json(
            self._forecast_ensemble_path(task_id),
            missing_code="forecast_ensemble_archive_not_found",
        )
        self._forecast_ensemble_summary(payload)
        return payload

    def save_forecast_verification(
        self,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist one completed multi-origin ensemble verification."""

        with self._lock:
            if (
                str(result.get("state")) != "completed"
                or str(result.get("method_id"))
                != "meteorological_ensemble_hindcast_verification.v1"
            ):
                raise ProjectStoreError(
                    "forecast_verification_not_completed",
                    "Only completed hindcast verifications can be archived.",
                )
            task_id = str(result.get("task_id", ""))
            now = _utc_now()
            payload = {
                "record_type": "hydrosoftware.forecast-verification",
                "record_version": STORE_VERSION,
                "task_id": task_id,
                "saved_at": now,
                "result": result,
            }
            self._write_json_atomic(
                self._forecast_verification_path(task_id),
                payload,
            )
            return self._forecast_verification_summary(payload)

    @staticmethod
    def _forecast_verification_summary(
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            result = payload["result"]
            aggregate = result["aggregate"]
            return {
                "task_id": str(payload["task_id"]),
                "saved_at": str(payload["saved_at"]),
                "method_id": str(result["method_id"]),
                "source_task_id": str(result["source_task_id"]),
                "experiment_id": str(result["experiment_id"]),
                "model_id": str(result["model_id"]),
                "case_count": int(result["case_count"]),
                "member_count": int(result["member_count"]),
                "horizon": int(result["horizon"]),
                "mean_crps": float(aggregate["mean_crps"]),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectStoreError(
                "saved_forecast_verification_invalid",
                f"A hindcast-verification archive is invalid: {exc}",
            ) from exc

    def list_forecast_verifications(self) -> tuple[dict[str, Any], ...]:
        """List hindcast-verification archives in reverse creation order."""

        if not self.forecast_verifications_dir.is_dir():
            return ()
        summaries = [
            self._forecast_verification_summary(
                self._read_json(
                    path,
                    missing_code="forecast_verification_archive_not_found",
                )
            )
            for path in self.forecast_verifications_dir.glob("*.json")
        ]
        return tuple(
            sorted(
                summaries,
                key=lambda item: str(item["saved_at"]),
                reverse=True,
            )
        )

    def load_forecast_verification(self, task_id: str) -> dict[str, Any]:
        """Return one complete immutable hindcast-verification archive."""

        payload = self._read_json(
            self._forecast_verification_path(task_id),
            missing_code="forecast_verification_archive_not_found",
        )
        self._forecast_verification_summary(payload)
        return payload

    def save_forecast_postprocessing(
        self,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist one completed chronological ensemble recalibration."""

        with self._lock:
            if (
                str(result.get("state")) != "completed"
                or str(result.get("method_id"))
                != "affine_member_recalibration.v1"
            ):
                raise ProjectStoreError(
                    "forecast_postprocessing_not_completed",
                    "Only completed forecast post-processing analyses can be archived.",
                )
            task_id = str(result.get("task_id", ""))
            payload = {
                "record_type": "hydrosoftware.forecast-postprocessing",
                "record_version": STORE_VERSION,
                "task_id": task_id,
                "saved_at": _utc_now(),
                "result": result,
            }
            self._write_json_atomic(
                self._forecast_postprocessing_path(task_id),
                payload,
            )
            return self._forecast_postprocessing_summary(payload)

    @staticmethod
    def _forecast_postprocessing_summary(
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            result = payload["result"]
            split = result["split"]
            aggregate = result["aggregate"]
            return {
                "task_id": str(payload["task_id"]),
                "saved_at": str(payload["saved_at"]),
                "method_id": str(result["method_id"]),
                "source_task_id": str(result["source_task_id"]),
                "experiment_id": str(result["experiment_id"]),
                "model_id": str(result["model_id"]),
                "member_count": int(result["member_count"]),
                "horizon": int(result["horizon"]),
                "training_case_count": int(split["training_case_count"]),
                "evaluation_case_count": int(split["evaluation_case_count"]),
                "raw_mean_crps": float(aggregate["raw"]["mean_crps"]),
                "adjusted_mean_crps": float(
                    aggregate["adjusted"]["mean_crps"]
                ),
                "crps_skill_score": (
                    float(aggregate["crps_skill_score"])
                    if aggregate.get("crps_skill_score") is not None
                    else None
                ),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectStoreError(
                "saved_forecast_postprocessing_invalid",
                f"A forecast post-processing archive is invalid: {exc}",
            ) from exc

    def list_forecast_postprocessings(self) -> tuple[dict[str, Any], ...]:
        """List post-processing archives in reverse creation order."""

        if not self.forecast_postprocessings_dir.is_dir():
            return ()
        summaries = [
            self._forecast_postprocessing_summary(
                self._read_json(
                    path,
                    missing_code="forecast_postprocessing_archive_not_found",
                )
            )
            for path in self.forecast_postprocessings_dir.glob("*.json")
        ]
        return tuple(
            sorted(
                summaries,
                key=lambda item: str(item["saved_at"]),
                reverse=True,
            )
        )

    def load_forecast_postprocessing(self, task_id: str) -> dict[str, Any]:
        """Return one immutable post-processing archive."""

        payload = self._read_json(
            self._forecast_postprocessing_path(task_id),
            missing_code="forecast_postprocessing_archive_not_found",
        )
        self._forecast_postprocessing_summary(payload)
        return payload

    def save_forecast_event_verification(
        self,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist one completed threshold-event decision analysis."""

        with self._lock:
            if (
                str(result.get("state")) != "completed"
                or str(result.get("method_id"))
                != "threshold_event_decision_verification.v1"
            ):
                raise ProjectStoreError(
                    "forecast_event_verification_not_completed",
                    "Only completed threshold-event analyses can be archived.",
                )
            task_id = str(result.get("task_id", ""))
            payload = {
                "record_type": "hydrosoftware.forecast-event-verification",
                "record_version": STORE_VERSION,
                "task_id": task_id,
                "saved_at": _utc_now(),
                "result": result,
            }
            self._write_json_atomic(
                self._forecast_event_verification_path(task_id),
                payload,
            )
            return self._forecast_event_verification_summary(payload)

    @staticmethod
    def _forecast_event_verification_summary(
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        try:
            result = payload["result"]
            split = result["split"]
            aggregate = result["aggregate"]
            threshold = result["threshold"]
            return {
                "task_id": str(payload["task_id"]),
                "saved_at": str(payload["saved_at"]),
                "method_id": str(result["method_id"]),
                "source_task_id": str(result["source_task_id"]),
                "experiment_id": str(result["experiment_id"]),
                "model_id": str(result["model_id"]),
                "member_count": int(result["member_count"]),
                "horizon": int(result["horizon"]),
                "training_case_count": int(split["training_case_count"]),
                "evaluation_case_count": int(split["evaluation_case_count"]),
                "threshold_value": float(threshold["value"]),
                "threshold_mode": str(threshold["mode"]),
                "raw_brier_score": float(aggregate["raw"]["brier_score"]),
                "adjusted_brier_score": float(
                    aggregate["adjusted"]["brier_score"]
                ),
                "adjusted_brier_skill_vs_raw": (
                    float(aggregate["adjusted_brier_skill_vs_raw"])
                    if aggregate.get("adjusted_brier_skill_vs_raw") is not None
                    else None
                ),
            }
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectStoreError(
                "saved_forecast_event_verification_invalid",
                f"A threshold-event analysis archive is invalid: {exc}",
            ) from exc

    def list_forecast_event_verifications(self) -> tuple[dict[str, Any], ...]:
        """List threshold-event archives in reverse creation order."""

        if not self.forecast_event_verifications_dir.is_dir():
            return ()
        summaries = [
            self._forecast_event_verification_summary(
                self._read_json(
                    path,
                    missing_code="forecast_event_verification_archive_not_found",
                )
            )
            for path in self.forecast_event_verifications_dir.glob("*.json")
        ]
        return tuple(
            sorted(
                summaries,
                key=lambda item: str(item["saved_at"]),
                reverse=True,
            )
        )

    def load_forecast_event_verification(self, task_id: str) -> dict[str, Any]:
        """Return one immutable threshold-event analysis archive."""

        payload = self._read_json(
            self._forecast_event_verification_path(task_id),
            missing_code="forecast_event_verification_archive_not_found",
        )
        self._forecast_event_verification_summary(payload)
        return payload
