"""Structured task progress, cancellation, and error reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from threading import Event
from typing import Any, Mapping


class TaskState(str, Enum):
    """Lifecycle states understood by the future task centre and UI."""

    QUEUED = "queued"
    PREPARING = "preparing"
    RUNNING = "running"
    CANCELLATION_REQUESTED = "cancellation_requested"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ProgressSeverity(str, Enum):
    """User-facing importance of a progress event."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass(frozen=True)
class StructuredError:
    """Stable error payload suitable for both simple and expert views."""

    code: str
    summary: str
    explanation: str
    suggested_action: str
    technical_details: str | None = None
    context: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.code.strip() or " " in self.code:
            raise ValueError("Error code must be a non-empty, space-free identifier.")
        if not self.summary.strip():
            raise ValueError("Error summary cannot be empty.")
        if not self.explanation.strip():
            raise ValueError("Error explanation cannot be empty.")
        if not self.suggested_action.strip():
            raise ValueError("Error suggested_action cannot be empty.")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)


@dataclass(frozen=True)
class ProgressEvent:
    """One immutable task update emitted to observers."""

    task_id: str
    sequence: int
    state: TaskState
    stage: str
    message: str
    completed_units: int = 0
    total_units: int | None = None
    fraction: float | None = None
    elapsed_seconds: float = 0.0
    estimated_remaining_seconds: float | None = None
    best_score: float | None = None
    severity: ProgressSeverity = ProgressSeverity.INFO
    error: StructuredError | None = None

    def __post_init__(self) -> None:
        if not self.task_id.strip():
            raise ValueError("task_id cannot be empty.")
        if self.sequence < 0:
            raise ValueError("sequence cannot be negative.")
        if not self.stage.strip() or not self.message.strip():
            raise ValueError("Progress stage and message cannot be empty.")
        if self.completed_units < 0:
            raise ValueError("completed_units cannot be negative.")
        if self.total_units is not None:
            if self.total_units < 0:
                raise ValueError("total_units cannot be negative.")
            if self.completed_units > self.total_units:
                raise ValueError("completed_units cannot exceed total_units.")
        if self.fraction is not None and not 0.0 <= self.fraction <= 1.0:
            raise ValueError("fraction must lie in [0, 1].")
        if self.elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds cannot be negative.")
        if (
            self.estimated_remaining_seconds is not None
            and self.estimated_remaining_seconds < 0.0
        ):
            raise ValueError("estimated_remaining_seconds cannot be negative.")
        if self.severity is ProgressSeverity.ERROR and self.error is None:
            raise ValueError("Error progress events must include a StructuredError.")

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation for the local API."""

        payload = asdict(self)
        payload["state"] = self.state.value
        payload["severity"] = self.severity.value
        return payload


class CancellationToken:
    """Thread-safe cooperative cancellation signal.

    Scientific code is never terminated in the middle of a model evaluation.
    Engines inspect this token only at documented safe boundaries.
    """

    def __init__(self) -> None:
        self._event = Event()

    def request(self) -> None:
        """Request cancellation without blocking the calling thread."""

        self._event.set()

    @property
    def is_requested(self) -> bool:
        """Return whether cancellation has been requested."""

        return self._event.is_set()
