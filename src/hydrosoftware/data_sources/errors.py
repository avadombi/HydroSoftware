"""Stable user-correctable failures raised by data-source adapters."""


class DataSourceError(ValueError):
    """Data-source failure with a stable machine-readable error code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
