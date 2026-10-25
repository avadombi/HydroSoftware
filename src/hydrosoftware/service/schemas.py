"""Strict request schemas used only by the localhost service boundary."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    RootModel,
    field_validator,
    model_validator,
)

from hydrosoftware.configuration import CsvDataSourceConfig


class StrictServiceModel(BaseModel):
    """Reject fields the current service does not understand."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class CsvPreviewRequest(StrictServiceModel):
    """Request inspection of an existing workspace-relative CSV."""

    path: str = Field(min_length=1, max_length=1024)
    delimiter: str = ","
    encoding: str = "utf-8-sig"

    @field_validator("delimiter")
    @classmethod
    def validate_delimiter(cls, value: str) -> str:
        if len(value) != 1:
            raise ValueError("CSV delimiter must contain exactly one character.")
        return value


class CsvImportRequest(StrictServiceModel):
    """Base64-preserving CSV import from the local interface."""

    filename: str = Field(min_length=1, max_length=255)
    content_base64: str = Field(min_length=1)
    delimiter: str = ","
    encoding: Literal["utf-8", "utf-8-sig"] = "utf-8-sig"

    @field_validator("delimiter")
    @classmethod
    def validate_delimiter(cls, value: str) -> str:
        if len(value) != 1:
            raise ValueError("CSV delimiter must contain exactly one character.")
        return value


class NeuralAntecedentForcings(
    RootModel[dict[str, tuple[float, ...]]]
):
    """Explicit named channel context used to chain neural and PaML forecasts."""

    model_config = ConfigDict(allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_named_context(self) -> "NeuralAntecedentForcings":
        if not self.root:
            raise ValueError("Neural antecedent context cannot be empty.")
        lengths: set[int] = set()
        for name, values in self.root.items():
            if (
                not name
                or not name.replace("_", "").replace("-", "").isalnum()
            ):
                raise ValueError(
                    "Antecedent context channel names must be stable identifiers."
                )
            if not values:
                raise ValueError(
                    f"Antecedent context channel {name!r} cannot be empty."
                )
            lengths.add(len(values))
        if len(lengths) != 1:
            raise ValueError(
                "All antecedent context channels must share the time dimension."
            )
        return self


class ForecastRunRequest(StrictServiceModel):
    """Run one saved calibrated model on a new forcing partition."""

    source_task_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    data: CsvDataSourceConfig
    initialization: Literal["source_final", "configured", "custom"] = (
        "source_final"
    )
    initial_state: tuple[float, ...] | None = None
    antecedent_forcings: NeuralAntecedentForcings | None = None

    @field_validator("initial_state")
    @classmethod
    def validate_initial_state(
        cls, value: tuple[float, ...] | None
    ) -> tuple[float, ...] | None:
        if value is not None and (not value or any(item < 0.0 for item in value)):
            raise ValueError(
                "A custom forecast state must contain non-negative storages."
            )
        return value


class ForecastEnsembleMemberRequest(StrictServiceModel):
    """One aligned meteorological member and its portable CSV mapping."""

    member_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
    data: CsvDataSourceConfig


class ForecastEnsembleRunRequest(StrictServiceModel):
    """Run one frozen model over aligned meteorological forcing scenarios."""

    source_task_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    continuation_task_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    )
    issue_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    )
    issue_time: datetime | None = None
    spinup_steps: int = Field(default=0, ge=0)
    members: tuple[ForecastEnsembleMemberRequest, ...] = Field(
        min_length=2,
        max_length=50,
    )
    initialization: Literal["source_final", "configured", "custom"] = (
        "source_final"
    )
    initial_state: tuple[float, ...] | None = None
    antecedent_forcings: NeuralAntecedentForcings | None = None
    lower_quantile: float = Field(default=0.1, ge=0.0, le=1.0)
    upper_quantile: float = Field(default=0.9, ge=0.0, le=1.0)

    @field_validator("initial_state")
    @classmethod
    def validate_initial_state(
        cls, value: tuple[float, ...] | None
    ) -> tuple[float, ...] | None:
        if value is not None and (not value or any(item < 0.0 for item in value)):
            raise ValueError(
                "A custom forecast state must contain non-negative storages."
            )
        return value

    @model_validator(mode="after")
    def validate_members_and_quantiles(self) -> "ForecastEnsembleRunRequest":
        member_ids = [member.member_id for member in self.members]
        if len(set(member_ids)) != len(member_ids):
            raise ValueError(
                "Meteorological forcing member identifiers must be unique."
            )
        if self.lower_quantile >= self.upper_quantile:
            raise ValueError(
                "Forecast ensemble quantiles must satisfy lower < upper."
            )
        if self.continuation_task_id is not None:
            if self.initialization != "source_final":
                raise ValueError(
                    "Member-wise ensemble continuation requires source_final "
                    "initialization."
                )
            if self.initial_state is not None or self.antecedent_forcings is not None:
                raise ValueError(
                    "Member-wise continuation restores states and contexts per "
                    "member; aggregate initial values are not accepted."
                )
        return self


class ForecastHindcastVerificationRequest(StrictServiceModel):
    """Aggregate archived forcing ensembles over repeated forecast origins."""

    task_ids: tuple[str, ...] = Field(min_length=2, max_length=500)
    seed: int = Field(default=0, ge=0)
    minimum_cases_per_lead: int = Field(default=2, ge=2, le=500)

    @field_validator("task_ids")
    @classmethod
    def validate_task_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError(
                "Hindcast-verification forecast identifiers must be unique."
            )
        for task_id in value:
            if (
                not task_id
                or len(task_id) > 128
                or not task_id[0].isalnum()
                or any(
                    not (character.isalnum() or character in "._-")
                    for character in task_id
                )
            ):
                raise ValueError(
                    "Hindcast-verification identifiers contain unsupported "
                    "characters."
                )
        return value


class ForecastPostprocessingRequest(StrictServiceModel):
    """Fit on early ensemble origins and evaluate on later origins."""

    task_ids: tuple[str, ...] = Field(min_length=7, max_length=500)
    training_case_count: int = Field(ge=5, le=498)
    seed: int = Field(default=0, ge=0)
    minimum_training_cases_per_lead: int = Field(default=5, ge=5, le=498)

    @field_validator("task_ids")
    @classmethod
    def validate_task_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError(
                "Post-processing forecast identifiers must be unique."
            )
        for task_id in value:
            if (
                not task_id
                or len(task_id) > 128
                or not task_id[0].isalnum()
                or any(
                    not (character.isalnum() or character in "._-")
                    for character in task_id
                )
            ):
                raise ValueError(
                    "Post-processing identifiers contain unsupported characters."
                )
        return value

    @model_validator(mode="after")
    def validate_chronological_split(self) -> "ForecastPostprocessingRequest":
        if self.training_case_count < self.minimum_training_cases_per_lead:
            raise ValueError(
                "The fitting split cannot be smaller than the per-lead minimum."
            )
        if self.training_case_count > len(self.task_ids) - 2:
            raise ValueError(
                "At least two selected origins must remain for independent evaluation."
            )
        return self


class ForecastEventVerificationRequest(StrictServiceModel):
    """Evaluate threshold-event decisions on later ensemble origins."""

    task_ids: tuple[str, ...] = Field(min_length=7, max_length=500)
    training_case_count: int = Field(ge=5, le=498)
    threshold_mode: Literal["absolute", "training_quantile"] = (
        "training_quantile"
    )
    threshold_value: float | None = Field(default=None, ge=0.0)
    training_quantile: float = Field(default=0.9, ge=0.5, le=0.995)
    probability_bin_count: int = Field(default=10, ge=3, le=20)
    minimum_evaluation_cases_per_lead: int = Field(default=5, ge=2, le=498)

    @field_validator("task_ids")
    @classmethod
    def validate_task_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError(
                "Event-verification forecast identifiers must be unique."
            )
        for task_id in value:
            if (
                not task_id
                or len(task_id) > 128
                or not task_id[0].isalnum()
                or any(
                    not (character.isalnum() or character in "._-")
                    for character in task_id
                )
            ):
                raise ValueError(
                    "Event-verification identifiers contain unsupported characters."
                )
        return value

    @model_validator(mode="after")
    def validate_event_protocol(self) -> "ForecastEventVerificationRequest":
        if self.training_case_count > len(self.task_ids) - 2:
            raise ValueError(
                "At least two later origins must remain for event evaluation."
            )
        evaluation_count = len(self.task_ids) - self.training_case_count
        if self.minimum_evaluation_cases_per_lead > evaluation_count:
            raise ValueError(
                "The per-lead reporting minimum cannot exceed the number of "
                "later evaluation origins."
            )
        if self.threshold_mode == "absolute" and self.threshold_value is None:
            raise ValueError(
                "Absolute event verification requires a discharge threshold."
            )
        if (
            self.threshold_mode == "training_quantile"
            and self.threshold_value is not None
        ):
            raise ValueError(
                "A fitting-archive quantile threshold cannot also receive an "
                "absolute discharge threshold."
            )
        return self


class RunComparisonRequest(StrictServiceModel):
    """Select completed run archives for one guarded common-support comparison."""

    task_ids: tuple[str, ...] = Field(min_length=2, max_length=6)

    @field_validator("task_ids")
    @classmethod
    def validate_task_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("Comparison task identifiers must be unique.")
        for task_id in value:
            if (
                not task_id
                or len(task_id) > 128
                or not task_id[0].isalnum()
                or any(
                    not (character.isalnum() or character in "._-")
                    for character in task_id
                )
            ):
                raise ValueError(
                    "Comparison task identifiers contain unsupported characters."
                )
        return value


class RunExplanationRequest(StrictServiceModel):
    """Request deterministic post-training SHAP for one saved run."""

    task_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    explained_rows: int = Field(default=8, ge=1, le=32)
    background_rows: int = Field(default=4, ge=1, le=32)
    permutations: int = Field(default=8, ge=1, le=128)
    seed: int = 0


class RunScientificVerificationRequest(StrictServiceModel):
    """Request frozen replay, restart, causality, and physical checks."""

    task_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    comparison_tolerance: float = Field(default=1e-10, ge=0.0, le=1e-4)
    mass_balance_tolerance: float = Field(default=1e-8, ge=0.0, le=1e-2)
    nonnegative_tolerance: float = Field(default=1e-12, ge=0.0, le=1e-4)


class RunUncertaintyRequest(StrictServiceModel):
    """Request block-bootstrap metric uncertainty for aligned saved runs."""

    task_ids: tuple[str, ...] = Field(min_length=1, max_length=6)
    bootstrap_replicates: int = Field(default=1000, ge=200, le=5000)
    confidence_level: float = Field(default=0.95, ge=0.8, le=0.99)
    block_length: int | None = Field(default=None, ge=1, le=366)
    seed: int = 0
    reference_task_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$",
    )

    @field_validator("task_ids")
    @classmethod
    def validate_task_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("Uncertainty task identifiers must be unique.")
        for task_id in value:
            if (
                not task_id
                or len(task_id) > 128
                or not task_id[0].isalnum()
                or any(
                    not (character.isalnum() or character in "._-")
                    for character in task_id
                )
            ):
                raise ValueError(
                    "Uncertainty task identifiers contain unsupported characters."
                )
        return value

    @model_validator(mode="after")
    def validate_reference(self) -> "RunUncertaintyRequest":
        if (
            self.reference_task_id is not None
            and self.reference_task_id not in self.task_ids
        ):
            raise ValueError(
                "The uncertainty reference must be one of the selected runs."
            )
        return self


class TemporalValidationRequest(StrictServiceModel):
    """Refit saved experiments on common rolling-origin test blocks."""

    experiment_ids: tuple[str, ...] = Field(min_length=1, max_length=4)
    start_index: int = Field(default=0, ge=0)
    initial_training_steps: int = Field(default=1_095, ge=1)
    validation_steps: int = Field(default=365, ge=1)
    test_steps: int = Field(default=365, ge=1)
    step_steps: int = Field(default=365, ge=1)
    folds: int = Field(default=3, ge=1, le=10)
    window_mode: Literal["expanding", "sliding"] = "expanding"
    minimum_scored_observations: int = Field(default=30, ge=2)

    @field_validator("experiment_ids")
    @classmethod
    def validate_experiment_ids(
        cls,
        value: tuple[str, ...],
    ) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError(
                "Temporal-validation experiment identifiers must be unique."
            )
        for experiment_id in value:
            if (
                not experiment_id
                or len(experiment_id) > 128
                or not experiment_id[0].isalnum()
                or any(
                    not (character.isalnum() or character in "._-")
                    for character in experiment_id
                )
            ):
                raise ValueError(
                    "Temporal-validation experiment identifiers contain "
                    "unsupported characters."
                )
        return value

    @model_validator(mode="after")
    def validate_non_overlapping_test_blocks(self) -> "TemporalValidationRequest":
        if self.step_steps < self.test_steps:
            raise ValueError(
                "step_steps must be at least test_steps so held-out test blocks "
                "do not overlap."
            )
        return self
