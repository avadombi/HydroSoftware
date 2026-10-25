"""Versioned localhost API used by the HydroSoftware interface."""

from __future__ import annotations

import base64
import binascii
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from hydrosoftware import __version__
from hydrosoftware.configuration import (
    ExperimentConfigV1,
    experiment_json_schema,
)
from hydrosoftware.execution.experiments import ExperimentExecutionError, ExperimentRunner
from hydrosoftware.execution.forecasting import run_forecast
from hydrosoftware.execution.forecast_ensembles import (
    run_forcing_ensemble_forecast,
)
from hydrosoftware.execution.forecast_verification import (
    ForecastVerificationError,
    verify_forcing_ensemble_hindcasts,
)
from hydrosoftware.execution.forecast_postprocessing import (
    postprocess_forcing_ensemble_hindcasts,
)
from hydrosoftware.execution.forecast_events import (
    verify_threshold_event_decisions,
)
from hydrosoftware.execution.comparison import (
    RunComparisonError,
    compare_run_archives,
)
from hydrosoftware.execution.explainability import (
    ExplainabilityError,
    explain_run_archive,
)
from hydrosoftware.execution.scientific_verification import (
    ScientificVerificationError,
    verify_run_archive,
)
from hydrosoftware.execution.uncertainty import (
    UncertaintyAnalysisError,
    analyze_run_uncertainty,
)
from hydrosoftware.execution.temporal_validation import (
    RollingOriginSettings,
    RollingOriginValidator,
    TemporalValidationTaskManager,
)
from hydrosoftware.execution.manager import (
    ExperimentTaskManager,
    ResultNotReadyError,
    TaskNotFoundError,
)
from hydrosoftware.execution.serialization import to_jsonable
from hydrosoftware.execution.persistence import ProjectStore, ProjectStoreError
from hydrosoftware.data_sources import (
    DataSourceError,
    analyze_calibration_csv,
    import_csv_bytes,
    preview_csv,
)
from hydrosoftware.models.conceptual.exp_hydro import EXP_HYDRO_MANIFEST
from hydrosoftware.models.conceptual.hbv import HBV_MANIFEST
from hydrosoftware.models.neural import (
    CNN1D_TENSORFLOW_MANIFEST,
    GRU_TENSORFLOW_MANIFEST,
    LSTM_TENSORFLOW_MANIFEST,
    MLP_TENSORFLOW_MANIFEST,
    STACKED_GRU_TENSORFLOW_MANIFEST,
    STACKED_LSTM_TENSORFLOW_MANIFEST,
    CNN1D_S2S_TENSORFLOW_MANIFEST,
    GRU_S2S_TENSORFLOW_MANIFEST,
    LSTM_S2S_TENSORFLOW_MANIFEST,
    MLP_S2S_TENSORFLOW_MANIFEST,
)
from hydrosoftware.models.paml import (
    PAML_PRNN_CNN_MANIFEST,
)
from hydrosoftware.models.deepdiscover_numpy import DEEPDISCOVER_NUMPY_MANIFEST
from hydrosoftware.service.schemas import (
    CsvImportRequest,
    CsvPreviewRequest,
    ForecastEnsembleRunRequest,
    ForecastHindcastVerificationRequest,
    ForecastEventVerificationRequest,
    ForecastPostprocessingRequest,
    ForecastRunRequest,
    RunComparisonRequest,
    RunExplanationRequest,
    RunScientificVerificationRequest,
    TemporalValidationRequest,
    RunUncertaintyRequest,
)


API_PREFIX = "/api/v1"


def create_app(
    *,
    workspace_root: str | Path,
    max_workers: int = 2,
) -> FastAPI:
    """Create an isolated service instance rooted in one project workspace."""

    runner = ExperimentRunner(workspace_root)
    project_store = ProjectStore(runner.workspace_root)
    manager = ExperimentTaskManager(
        runner,
        max_workers=max_workers,
        on_result=project_store.save_run,
    )
    temporal_validation_manager = TemporalValidationTaskManager(
        RollingOriginValidator(runner),
        max_workers=1,
        on_result=project_store.save_temporal_validation,
    )

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        yield
        manager.shutdown(wait=True)
        temporal_validation_manager.shutdown(wait=True)

    app = FastAPI(
        title="HydroSoftware Local API",
        version=__version__,
        description=(
            "Local execution boundary for versioned hydrological experiments. "
            "This service is intended to bind to the loopback interface only."
        ),
        lifespan=lifespan,
    )
    app.state.task_manager = manager
    app.state.experiment_runner = runner
    app.state.workspace_root = runner.workspace_root
    app.state.project_store = project_store
    app.state.temporal_validation_manager = temporal_validation_manager

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://tauri.localhost",
            "tauri://localhost",
        ],
        allow_credentials=False,
        allow_methods=["GET", "POST", "PUT"],
        allow_headers=["Content-Type"],
    )

    def task_or_404(task_id: str):
        try:
            return manager.snapshot(task_id)
        except TaskNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "task_not_found",
                    "message": f"No task is known with identifier {task_id!r}.",
                },
            ) from exc

    def data_error(exc: DataSourceError) -> HTTPException:
        return HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": exc.code,
                "message": str(exc),
            },
        )

    def store_error(exc: ProjectStoreError) -> HTTPException:
        missing = exc.code in {
            "experiment_not_found",
            "run_archive_not_found",
            "temporal_validation_archive_not_found",
            "forecast_ensemble_archive_not_found",
            "forecast_verification_archive_not_found",
            "forecast_postprocessing_archive_not_found",
        }
        return HTTPException(
            status_code=(
                status.HTTP_404_NOT_FOUND
                if missing
                else status.HTTP_422_UNPROCESSABLE_ENTITY
            ),
            detail={
                "code": exc.code,
                "message": str(exc),
            },
        )

    @app.get(f"{API_PREFIX}/health")
    def health() -> dict:
        return {
            "status": "ok",
            "service": "hydrosoftware-local-api",
            "version": __version__,
            "api_version": "1",
        }

    @app.get(f"{API_PREFIX}/models")
    def list_models() -> dict:
        return {
            "models": [
                to_jsonable(EXP_HYDRO_MANIFEST),
                to_jsonable(HBV_MANIFEST),
                to_jsonable(MLP_TENSORFLOW_MANIFEST),
                to_jsonable(LSTM_TENSORFLOW_MANIFEST),
                to_jsonable(STACKED_LSTM_TENSORFLOW_MANIFEST),
                to_jsonable(GRU_TENSORFLOW_MANIFEST),
                to_jsonable(STACKED_GRU_TENSORFLOW_MANIFEST),
                to_jsonable(CNN1D_TENSORFLOW_MANIFEST),
                to_jsonable(MLP_S2S_TENSORFLOW_MANIFEST),
                to_jsonable(LSTM_S2S_TENSORFLOW_MANIFEST),
                to_jsonable(GRU_S2S_TENSORFLOW_MANIFEST),
                to_jsonable(CNN1D_S2S_TENSORFLOW_MANIFEST),
                to_jsonable(PAML_PRNN_CNN_MANIFEST),
                to_jsonable(DEEPDISCOVER_NUMPY_MANIFEST),
            ]
        }

    @app.get(f"{API_PREFIX}/schemas/experiment")
    def experiment_schema() -> dict:
        return experiment_json_schema()

    @app.post(f"{API_PREFIX}/data/preview")
    def inspect_data(request: CsvPreviewRequest) -> dict:
        try:
            preview = preview_csv(
                request.path,
                workspace_root=runner.workspace_root,
                delimiter=request.delimiter,
                encoding=request.encoding,
            )
        except DataSourceError as exc:
            raise data_error(exc) from exc
        return to_jsonable(preview)

    @app.post(f"{API_PREFIX}/data/imports", status_code=status.HTTP_201_CREATED)
    def import_data(request: CsvImportRequest) -> dict:
        try:
            content = base64.b64decode(request.content_base64, validate=True)
        except (binascii.Error, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "invalid_base64_upload",
                    "message": "The uploaded CSV content is not valid base64 data.",
                },
            ) from exc
        try:
            preview = import_csv_bytes(
                request.filename,
                content,
                workspace_root=runner.workspace_root,
                delimiter=request.delimiter,
                encoding=request.encoding,
            )
        except DataSourceError as exc:
            raise data_error(exc) from exc
        return to_jsonable(preview)

    @app.post(f"{API_PREFIX}/data/quality")
    def inspect_data_quality(config: ExperimentConfigV1) -> dict:
        """Apply the recorded data rules and return their complete audit report."""

        try:
            analysis = analyze_calibration_csv(
                config.data,
                workspace_root=runner.workspace_root,
            )
        except DataSourceError as exc:
            raise data_error(exc) from exc
        return to_jsonable(analysis.report)

    @app.post(f"{API_PREFIX}/experiments/validate")
    def validate_experiment(config: ExperimentConfigV1) -> dict:
        try:
            return runner.validate(config).as_dict()
        except ExperimentExecutionError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=exc.error.as_dict(),
            ) from exc

    @app.get(f"{API_PREFIX}/experiments")
    def list_saved_experiments() -> dict:
        try:
            experiments = project_store.list_experiments()
        except ProjectStoreError as exc:
            raise store_error(exc) from exc
        return {"experiments": [item.as_dict() for item in experiments]}

    @app.put(f"{API_PREFIX}/experiments/{{experiment_id}}")
    def save_experiment(experiment_id: str, config: ExperimentConfigV1) -> dict:
        if experiment_id != config.experiment_id:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "experiment_identifier_mismatch",
                    "message": "The URL and configuration experiment identifiers must match.",
                },
            )
        try:
            return project_store.save_experiment(config).as_dict()
        except ProjectStoreError as exc:
            raise store_error(exc) from exc

    @app.get(f"{API_PREFIX}/experiments/{{experiment_id}}")
    def load_saved_experiment(experiment_id: str) -> dict:
        try:
            return project_store.load_experiment(experiment_id)
        except ProjectStoreError as exc:
            raise store_error(exc) from exc

    @app.get(f"{API_PREFIX}/experiments/{{experiment_id}}/runs")
    def list_saved_runs(experiment_id: str) -> dict:
        try:
            return {"runs": list(project_store.list_runs(experiment_id))}
        except ProjectStoreError as exc:
            raise store_error(exc) from exc

    @app.get(f"{API_PREFIX}/run-archives/{{task_id}}")
    def load_run_archive(task_id: str) -> dict:
        try:
            return project_store.load_run(task_id)
        except ProjectStoreError as exc:
            raise store_error(exc) from exc

    @app.post(f"{API_PREFIX}/comparisons")
    def compare_saved_runs(request: RunComparisonRequest) -> dict:
        """Recompute model metrics on one identical overlapping support."""

        try:
            archives = tuple(
                project_store.load_run(task_id)
                for task_id in request.task_ids
            )
            return compare_run_archives(archives)
        except ProjectStoreError as exc:
            raise store_error(exc) from exc
        except RunComparisonError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": exc.code,
                    "message": str(exc),
                },
            ) from exc

    @app.post(f"{API_PREFIX}/explanations")
    def explain_saved_run(request: RunExplanationRequest) -> dict:
        """Explain held-out predictions with training-referenced SHAP."""

        try:
            archive = project_store.load_run(request.task_id)
            return explain_run_archive(
                archive,
                explained_rows=request.explained_rows,
                background_rows=request.background_rows,
                permutations=request.permutations,
                seed=request.seed,
            )
        except ProjectStoreError as exc:
            raise store_error(exc) from exc
        except ExplainabilityError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": exc.code,
                    "message": str(exc),
                },
            ) from exc

    @app.post(f"{API_PREFIX}/scientific-verifications")
    def verify_saved_run_scientifically(
        request: RunScientificVerificationRequest,
    ) -> dict:
        """Replay one immutable run and test its numerical definitions."""

        try:
            archive = project_store.load_run(request.task_id)
            return verify_run_archive(
                archive,
                comparison_tolerance=request.comparison_tolerance,
                mass_balance_tolerance=request.mass_balance_tolerance,
                nonnegative_tolerance=request.nonnegative_tolerance,
            )
        except ProjectStoreError as exc:
            raise store_error(exc) from exc
        except ScientificVerificationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": exc.code, "message": str(exc)},
            ) from exc

    @app.post(f"{API_PREFIX}/uncertainty")
    def quantify_saved_run_uncertainty(request: RunUncertaintyRequest) -> dict:
        """Bootstrap metric uncertainty without refitting selected models."""

        task_ids = list(request.task_ids)
        if request.reference_task_id is not None:
            task_ids.remove(request.reference_task_id)
            task_ids.insert(0, request.reference_task_id)
        try:
            archives = tuple(
                project_store.load_run(task_id)
                for task_id in task_ids
            )
            return analyze_run_uncertainty(
                archives,
                bootstrap_replicates=request.bootstrap_replicates,
                confidence_level=request.confidence_level,
                block_length=request.block_length,
                seed=request.seed,
            )
        except ProjectStoreError as exc:
            raise store_error(exc) from exc
        except UncertaintyAnalysisError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": exc.code,
                    "message": str(exc),
                },
            ) from exc

    def temporal_task_or_404(task_id: str):
        try:
            return temporal_validation_manager.snapshot(task_id)
        except TaskNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "code": "temporal_validation_task_not_found",
                    "message": (
                        "No temporal-validation task is known with identifier "
                        f"{task_id!r}."
                    ),
                },
            ) from exc

    @app.post(
        f"{API_PREFIX}/temporal-validations",
        status_code=status.HTTP_202_ACCEPTED,
    )
    def start_temporal_validation(request: TemporalValidationRequest) -> dict:
        """Refit saved experiments on common chronological future blocks."""

        try:
            configs = tuple(
                ExperimentConfigV1.model_validate(
                    project_store.load_experiment(experiment_id)["config"]
                )
                for experiment_id in request.experiment_ids
            )
        except ProjectStoreError as exc:
            raise store_error(exc) from exc
        settings = RollingOriginSettings(
            start_index=request.start_index,
            initial_training_steps=request.initial_training_steps,
            validation_steps=request.validation_steps,
            test_steps=request.test_steps,
            step_steps=request.step_steps,
            folds=request.folds,
            window_mode=request.window_mode,
            minimum_scored_observations=request.minimum_scored_observations,
        )
        return temporal_validation_manager.submit(configs, settings).as_dict()

    @app.get(f"{API_PREFIX}/temporal-validations")
    def list_temporal_validation_archives() -> dict:
        try:
            return {
                "temporal_validations": list(
                    project_store.list_temporal_validations()
                )
            }
        except ProjectStoreError as exc:
            raise store_error(exc) from exc

    @app.get(f"{API_PREFIX}/temporal-validations/archives/{{task_id}}")
    def load_temporal_validation_archive(task_id: str) -> dict:
        try:
            return project_store.load_temporal_validation(task_id)
        except ProjectStoreError as exc:
            raise store_error(exc) from exc

    @app.get(f"{API_PREFIX}/temporal-validations/tasks/{{task_id}}")
    def temporal_validation_status(task_id: str) -> dict:
        return temporal_task_or_404(task_id).as_dict()

    @app.get(f"{API_PREFIX}/temporal-validations/tasks/{{task_id}}/events")
    def temporal_validation_events(
        task_id: str,
        after_sequence: int = Query(default=-1, ge=-1),
    ) -> dict:
        temporal_task_or_404(task_id)
        return {
            "task_id": task_id,
            "events": [
                event.as_dict()
                for event in temporal_validation_manager.events(
                    task_id,
                    after_sequence=after_sequence,
                )
            ],
        }

    @app.post(f"{API_PREFIX}/temporal-validations/tasks/{{task_id}}/cancel")
    def cancel_temporal_validation(task_id: str) -> dict:
        temporal_task_or_404(task_id)
        return temporal_validation_manager.cancel(task_id).as_dict()

    @app.get(f"{API_PREFIX}/temporal-validations/tasks/{{task_id}}/result")
    def temporal_validation_result(task_id: str) -> dict:
        snapshot = temporal_task_or_404(task_id)
        try:
            return dict(temporal_validation_manager.result(task_id))
        except ResultNotReadyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "temporal_validation_result_not_ready",
                    "message": str(exc),
                    "task": snapshot.as_dict(),
                },
            ) from exc

    @app.post(f"{API_PREFIX}/runs", status_code=status.HTTP_202_ACCEPTED)
    def start_run(config: ExperimentConfigV1) -> dict:
        try:
            project_store.save_experiment(config)
        except ProjectStoreError as exc:
            raise store_error(exc) from exc
        return manager.submit(config).as_dict()

    @app.post(f"{API_PREFIX}/forecasts")
    def start_forecast(request: ForecastRunRequest) -> dict:
        """Apply a completed saved calibration to a new forcing partition."""

        try:
            source_archive = project_store.load_run(request.source_task_id)
            analysis = analyze_calibration_csv(
                request.data,
                workspace_root=runner.workspace_root,
            )
            if not analysis.report.ready:
                first_error = next(
                    issue
                    for issue in analysis.report.issues
                    if issue.severity == "error"
                )
                raise DataSourceError(first_error.code, first_error.explanation)
            return run_forecast(
                source_archive,
                analysis,
                initialization=request.initialization,
                observations_available=request.data.columns.streamflow is not None,
                custom_initial_state=request.initial_state,
                antecedent_forcings=(
                    request.antecedent_forcings.model_dump(mode="json")
                    if request.antecedent_forcings is not None
                    else None
                ),
            )
        except ProjectStoreError as exc:
            raise store_error(exc) from exc
        except DataSourceError as exc:
            raise data_error(exc) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "forecast_configuration_invalid",
                    "message": str(exc),
                },
            ) from exc

    @app.post(f"{API_PREFIX}/forecast-ensembles")
    def start_forecast_ensemble(request: ForecastEnsembleRunRequest) -> dict:
        """Apply one frozen model to aligned meteorological forcing members."""

        try:
            source_archive = project_store.load_run(request.source_task_id)
            continuation_result = None
            if request.continuation_task_id is not None:
                continuation_archive = project_store.load_forecast_ensemble(
                    request.continuation_task_id
                )
                continuation_result = continuation_archive["result"]
            members = []
            for member in request.members:
                try:
                    analysis = analyze_calibration_csv(
                        member.data,
                        workspace_root=runner.workspace_root,
                    )
                except DataSourceError as exc:
                    raise DataSourceError(
                        exc.code,
                        f"Meteorological member {member.member_id!r}: {exc}",
                    ) from exc
                if not analysis.report.ready:
                    first_error = next(
                        issue
                        for issue in analysis.report.issues
                        if issue.severity == "error"
                    )
                    raise DataSourceError(
                        first_error.code,
                        (
                            f"Meteorological member {member.member_id!r}: "
                            f"{first_error.explanation}"
                        ),
                    )
                members.append(
                    (
                        member.member_id,
                        analysis,
                        member.data.columns.streamflow is not None,
                    )
                )
            result = run_forcing_ensemble_forecast(
                source_archive,
                members,
                initialization=request.initialization,
                continuation_result=continuation_result,
                issue_id=request.issue_id,
                issue_time=(
                    request.issue_time.isoformat()
                    if request.issue_time is not None
                    else None
                ),
                spinup_steps=request.spinup_steps,
                custom_initial_state=request.initial_state,
                antecedent_forcings=(
                    request.antecedent_forcings.model_dump(mode="json")
                    if request.antecedent_forcings is not None
                    else None
                ),
                lower_quantile=request.lower_quantile,
                upper_quantile=request.upper_quantile,
            )
            project_store.save_forecast_ensemble(result)
            return result
        except ProjectStoreError as exc:
            raise store_error(exc) from exc
        except DataSourceError as exc:
            raise data_error(exc) from exc
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": "forecast_ensemble_configuration_invalid",
                    "message": str(exc),
                },
            ) from exc

    @app.get(f"{API_PREFIX}/forecast-ensembles")
    def list_forecast_ensemble_archives() -> dict:
        try:
            return {
                "forecast_ensembles": list(
                    project_store.list_forecast_ensembles()
                )
            }
        except ProjectStoreError as exc:
            raise store_error(exc) from exc

    @app.get(f"{API_PREFIX}/forecast-ensembles/{{task_id}}")
    def load_forecast_ensemble_archive(task_id: str) -> dict:
        try:
            return project_store.load_forecast_ensemble(task_id)
        except ProjectStoreError as exc:
            raise store_error(exc) from exc

    @app.post(f"{API_PREFIX}/forecast-verifications")
    def verify_forecast_ensembles(
        request: ForecastHindcastVerificationRequest,
    ) -> dict:
        """Verify archived forcing ensembles across repeated forecast origins."""

        try:
            archives = [
                project_store.load_forecast_ensemble(task_id)
                for task_id in request.task_ids
            ]
            result = verify_forcing_ensemble_hindcasts(
                archives,
                seed=request.seed,
                minimum_cases_per_lead=request.minimum_cases_per_lead,
            )
            project_store.save_forecast_verification(result)
            return result
        except ProjectStoreError as exc:
            raise store_error(exc) from exc
        except ForecastVerificationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": exc.code,
                    "message": str(exc),
                },
            ) from exc

    @app.get(f"{API_PREFIX}/forecast-verifications")
    def list_forecast_verification_archives() -> dict:
        try:
            return {
                "forecast_verifications": list(
                    project_store.list_forecast_verifications()
                )
            }
        except ProjectStoreError as exc:
            raise store_error(exc) from exc

    @app.get(f"{API_PREFIX}/forecast-verifications/{{task_id}}")
    def load_forecast_verification_archive(task_id: str) -> dict:
        try:
            return project_store.load_forecast_verification(task_id)
        except ProjectStoreError as exc:
            raise store_error(exc) from exc

    @app.post(f"{API_PREFIX}/forecast-postprocessings")
    def postprocess_forecast_ensembles(
        request: ForecastPostprocessingRequest,
    ) -> dict:
        """Fit an affine correction on early origins and test it on later ones."""

        try:
            archives = [
                project_store.load_forecast_ensemble(task_id)
                for task_id in request.task_ids
            ]
            result = postprocess_forcing_ensemble_hindcasts(
                archives,
                training_case_count=request.training_case_count,
                seed=request.seed,
                minimum_training_cases_per_lead=(
                    request.minimum_training_cases_per_lead
                ),
            )
            project_store.save_forecast_postprocessing(result)
            return result
        except ProjectStoreError as exc:
            raise store_error(exc) from exc
        except ForecastVerificationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": exc.code,
                    "message": str(exc),
                },
            ) from exc

    @app.get(f"{API_PREFIX}/forecast-postprocessings")
    def list_forecast_postprocessing_archives() -> dict:
        try:
            return {
                "forecast_postprocessings": list(
                    project_store.list_forecast_postprocessings()
                )
            }
        except ProjectStoreError as exc:
            raise store_error(exc) from exc

    @app.get(f"{API_PREFIX}/forecast-postprocessings/{{task_id}}")
    def load_forecast_postprocessing_archive(task_id: str) -> dict:
        try:
            return project_store.load_forecast_postprocessing(task_id)
        except ProjectStoreError as exc:
            raise store_error(exc) from exc

    @app.post(f"{API_PREFIX}/forecast-event-verifications")
    def verify_forecast_events(
        request: ForecastEventVerificationRequest,
    ) -> dict:
        """Evaluate raw and adjusted threshold decisions on later origins."""

        try:
            archives = [
                project_store.load_forecast_ensemble(task_id)
                for task_id in request.task_ids
            ]
            result = verify_threshold_event_decisions(
                archives,
                training_case_count=request.training_case_count,
                threshold_mode=request.threshold_mode,
                threshold_value=request.threshold_value,
                training_quantile=request.training_quantile,
                probability_bin_count=request.probability_bin_count,
                minimum_evaluation_cases_per_lead=(
                    request.minimum_evaluation_cases_per_lead
                ),
            )
            project_store.save_forecast_event_verification(result)
            return result
        except ProjectStoreError as exc:
            raise store_error(exc) from exc
        except ForecastVerificationError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={
                    "code": exc.code,
                    "message": str(exc),
                },
            ) from exc

    @app.get(f"{API_PREFIX}/forecast-event-verifications")
    def list_forecast_event_verification_archives() -> dict:
        try:
            return {
                "forecast_event_verifications": list(
                    project_store.list_forecast_event_verifications()
                )
            }
        except ProjectStoreError as exc:
            raise store_error(exc) from exc

    @app.get(f"{API_PREFIX}/forecast-event-verifications/{{task_id}}")
    def load_forecast_event_verification_archive(task_id: str) -> dict:
        try:
            return project_store.load_forecast_event_verification(task_id)
        except ProjectStoreError as exc:
            raise store_error(exc) from exc

    @app.get(f"{API_PREFIX}/tasks")
    def list_tasks() -> dict:
        return {"tasks": [snapshot.as_dict() for snapshot in manager.list_snapshots()]}

    @app.get(f"{API_PREFIX}/tasks/{{task_id}}")
    def task_status(task_id: str) -> dict:
        return task_or_404(task_id).as_dict()

    @app.get(f"{API_PREFIX}/tasks/{{task_id}}/events")
    def task_events(
        task_id: str,
        after_sequence: int = Query(default=-1, ge=-1),
    ) -> dict:
        task_or_404(task_id)
        return {
            "task_id": task_id,
            "events": [
                event.as_dict()
                for event in manager.events(task_id, after_sequence=after_sequence)
            ],
        }

    @app.post(f"{API_PREFIX}/tasks/{{task_id}}/cancel")
    def cancel_task(task_id: str) -> dict:
        task_or_404(task_id)
        return manager.cancel(task_id).as_dict()

    @app.get(f"{API_PREFIX}/tasks/{{task_id}}/result")
    def task_result(task_id: str) -> dict:
        snapshot = task_or_404(task_id)
        try:
            result = manager.result(task_id)
        except ResultNotReadyError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail={
                    "code": "result_not_ready",
                    "message": str(exc),
                    "task": snapshot.as_dict(),
                },
            ) from exc
        return result.as_dict()

    return app
