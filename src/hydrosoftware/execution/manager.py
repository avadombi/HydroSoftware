"""Thread-safe in-memory task manager for the local desktop service."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from threading import RLock
from typing import Callable, Mapping
from uuid import uuid4

from hydrosoftware.configuration import ExperimentConfigV1
from hydrosoftware.execution.experiments import (
    ExperimentExecutionError,
    ExperimentRunResult,
    ExperimentRunner,
)
from hydrosoftware.execution.serialization import to_jsonable
from hydrosoftware.execution.tasks import (
    CancellationToken,
    ProgressEvent,
    ProgressSeverity,
    StructuredError,
    TaskState,
)


TERMINAL_STATES = {TaskState.COMPLETED, TaskState.CANCELLED, TaskState.FAILED}


class TaskNotFoundError(KeyError):
    """Raised when a task identifier is unknown to the current service process."""


class ResultNotReadyError(RuntimeError):
    """Raised when a terminal result has not yet been produced."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class TaskSnapshot:
    """Serializable status returned to the UI without exposing mutable state."""

    task_id: str
    experiment_id: str
    state: TaskState
    created_at: str
    updated_at: str
    event_count: int
    latest_event: ProgressEvent
    has_result: bool
    error: StructuredError | None = None

    @property
    def terminal(self) -> bool:
        if self.state is TaskState.FAILED:
            return True
        if self.state in {TaskState.COMPLETED, TaskState.CANCELLED}:
            return self.has_result
        return False

    def as_dict(self) -> dict:
        payload = to_jsonable(self)
        payload["terminal"] = self.terminal
        return payload


@dataclass
class _ManagedTask:
    task_id: str
    experiment_id: str
    created_at: str
    updated_at: str
    state: TaskState
    cancellation_token: CancellationToken
    events: list[ProgressEvent] = field(default_factory=list)
    result: ExperimentRunResult | None = None
    error: StructuredError | None = None
    future: Future | None = None


class ExperimentTaskManager:
    """Run experiments in worker threads and retain their progress in memory."""

    def __init__(
        self,
        runner: ExperimentRunner,
        *,
        max_workers: int = 2,
        on_result: Callable[[ExperimentConfigV1, ExperimentRunResult], object] | None = None,
    ) -> None:
        if max_workers < 1:
            raise ValueError("max_workers must be at least one.")
        self._runner = runner
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="hydrosoftware",
        )
        self._tasks: dict[str, _ManagedTask] = {}
        self._lock = RLock()
        self._on_result = on_result

    def _get_record(self, task_id: str) -> _ManagedTask:
        try:
            return self._tasks[task_id]
        except KeyError as exc:
            raise TaskNotFoundError(task_id) from exc

    @staticmethod
    def _snapshot(record: _ManagedTask) -> TaskSnapshot:
        return TaskSnapshot(
            task_id=record.task_id,
            experiment_id=record.experiment_id,
            state=record.state,
            created_at=record.created_at,
            updated_at=record.updated_at,
            event_count=len(record.events),
            latest_event=record.events[-1],
            has_result=record.result is not None,
            error=record.error,
        )

    def _record_event(self, task_id: str, event: ProgressEvent) -> ProgressEvent:
        with self._lock:
            record = self._get_record(task_id)
            normalized = replace(
                event,
                task_id=task_id,
                sequence=len(record.events),
            )
            if (
                record.cancellation_token.is_requested
                and normalized.state in {TaskState.PREPARING, TaskState.RUNNING}
            ):
                normalized = replace(
                    normalized,
                    state=TaskState.CANCELLATION_REQUESTED,
                )
            record.events.append(normalized)
            record.state = normalized.state
            record.updated_at = _utc_now()
            if normalized.error is not None and normalized.severity is ProgressSeverity.ERROR:
                record.error = normalized.error
            return normalized

    def submit(self, config: ExperimentConfigV1) -> TaskSnapshot:
        """Queue one validated configuration for asynchronous execution."""

        task_id = f"run-{uuid4()}"
        created = _utc_now()
        queued = ProgressEvent(
            task_id=task_id,
            sequence=0,
            state=TaskState.QUEUED,
            stage="queue",
            message="The experiment is waiting for an execution worker.",
        )
        record = _ManagedTask(
            task_id=task_id,
            experiment_id=config.experiment_id,
            created_at=created,
            updated_at=created,
            state=TaskState.QUEUED,
            cancellation_token=CancellationToken(),
            events=[queued],
        )
        with self._lock:
            self._tasks[task_id] = record
            record.future = self._executor.submit(self._execute, task_id, config)
            return self._snapshot(record)

    def _execute(self, task_id: str, config: ExperimentConfigV1) -> None:
        with self._lock:
            token = self._get_record(task_id).cancellation_token
        try:
            result = self._runner.run(
                config,
                task_id=task_id,
                cancellation_token=token,
                on_progress=lambda event: self._record_event(task_id, event),
            )
            archive_error: Exception | None = None
            if self._on_result is not None:
                try:
                    self._on_result(config, result)
                except Exception as exc:
                    archive_error = exc
            with self._lock:
                record = self._get_record(task_id)
                record.result = result
                record.state = result.state
                record.updated_at = _utc_now()
            if archive_error is not None:
                self._record_event(
                    task_id,
                    ProgressEvent(
                        task_id=task_id,
                        sequence=0,
                        state=result.state,
                        stage="persistence",
                        message=(
                            "The scientific result is available, but its persistent "
                            "archive could not be written."
                        ),
                        severity=ProgressSeverity.WARNING,
                        error=StructuredError(
                            code="run_archive_write_failed",
                            summary="The run archive could not be saved.",
                            explanation=str(archive_error),
                            suggested_action=(
                                "Keep the service open, export the configuration, and "
                                "check that the project folder is writable."
                            ),
                            technical_details=(
                                f"{type(archive_error).__name__}: {archive_error}"
                            ),
                        ),
                    ),
                )
        except ExperimentExecutionError as exc:
            self._fail_task(task_id, exc.error)
        except Exception as exc:
            self._fail_task(
                task_id,
                StructuredError(
                    code="unexpected_execution_failure",
                    summary="The experiment stopped because of an unexpected error.",
                    explanation=(
                        "The local service contained the failure and preserved earlier tasks."
                    ),
                    suggested_action=(
                        "Open the expert details, save the experiment configuration, and report "
                        "the error if it is reproducible."
                    ),
                    technical_details=f"{type(exc).__name__}: {exc}",
                ),
            )

    def _fail_task(self, task_id: str, error: StructuredError) -> None:
        with self._lock:
            record = self._get_record(task_id)
            latest = record.events[-1]
        self._record_event(
            task_id,
            ProgressEvent(
                task_id=task_id,
                sequence=0,
                state=TaskState.FAILED,
                stage="execution",
                message=error.summary,
                completed_units=latest.completed_units,
                total_units=latest.total_units,
                fraction=latest.fraction,
                elapsed_seconds=latest.elapsed_seconds,
                best_score=latest.best_score,
                severity=ProgressSeverity.ERROR,
                error=error,
            ),
        )

    def snapshot(self, task_id: str) -> TaskSnapshot:
        with self._lock:
            return self._snapshot(self._get_record(task_id))

    def list_snapshots(self) -> tuple[TaskSnapshot, ...]:
        with self._lock:
            return tuple(self._snapshot(record) for record in self._tasks.values())

    def events(self, task_id: str, *, after_sequence: int = -1) -> tuple[ProgressEvent, ...]:
        if after_sequence < -1:
            raise ValueError("after_sequence cannot be less than -1.")
        with self._lock:
            record = self._get_record(task_id)
            return tuple(event for event in record.events if event.sequence > after_sequence)

    def cancel(self, task_id: str) -> TaskSnapshot:
        """Request cooperative cancellation and return the updated task state."""

        with self._lock:
            record = self._get_record(task_id)
            if record.state in TERMINAL_STATES:
                return self._snapshot(record)
            record.cancellation_token.request()
            latest = record.events[-1]
        self._record_event(
            task_id,
            ProgressEvent(
                task_id=task_id,
                sequence=0,
                state=TaskState.CANCELLATION_REQUESTED,
                stage="cancellation",
                message="Cancellation was requested and will occur at a safe boundary.",
                completed_units=latest.completed_units,
                total_units=latest.total_units,
                fraction=latest.fraction,
                elapsed_seconds=latest.elapsed_seconds,
                best_score=latest.best_score,
            ),
        )
        return self.snapshot(task_id)

    def result(self, task_id: str) -> ExperimentRunResult:
        with self._lock:
            record = self._get_record(task_id)
            if record.result is not None:
                return record.result
            raise ResultNotReadyError(
                "The task has not produced a scientific result. Inspect its status and error."
            )

    def wait(self, task_id: str, *, timeout: float | None = None) -> TaskSnapshot:
        """Block test or command-line callers until the worker function returns."""

        with self._lock:
            future = self._get_record(task_id).future
        if future is not None:
            future.result(timeout=timeout)
        return self.snapshot(task_id)

    def shutdown(self, *, wait: bool = True) -> None:
        """Release worker threads without cancelling running scientific tasks."""

        self._executor.shutdown(wait=wait, cancel_futures=False)
