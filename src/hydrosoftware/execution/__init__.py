"""Execution-state definitions shared by calibration and training engines."""

from hydrosoftware.execution.tasks import (
    CancellationToken,
    ProgressEvent,
    ProgressSeverity,
    StructuredError,
    TaskState,
)

# Higher-level execution modules are intentionally not imported here. Keeping
# this package boundary light prevents the calibration engine from depending on
# Pydantic configuration or service-layer modules.

__all__ = [
    "CancellationToken",
    "ProgressEvent",
    "ProgressSeverity",
    "StructuredError",
    "TaskState",
]
