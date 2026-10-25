"""Pydantic schemas for the first versioned experiment configuration."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Annotated, Any, Literal, Mapping

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


CURRENT_SCHEMA_VERSION = "1.21"


class StrictConfigModel(BaseModel):
    """Configuration model that rejects undeclared fields."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class AdditionalPredictorColumnConfig(StrictConfigModel):
    """One optional numeric CSV column exposed to models that accept extra inputs."""

    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_]{0,63}$")
    column: str
    unit: str = Field(default="unspecified", min_length=1, max_length=64)
    missing: Literal["error", "linear_interpolation"] = "error"

    @field_validator("column", "unit")
    @classmethod
    def validate_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Additional predictor fields cannot be blank.")
        return normalized

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        normalized = value.strip()
        if normalized.startswith("physical_") or "__lag_" in normalized:
            raise ValueError(
                "Additional predictor names cannot use reserved physical or lag prefixes."
            )
        return normalized


class CsvColumnMapping(StrictConfigModel):
    """CSV columns mapped to standard roles and optional additional predictors."""

    precipitation: str
    temperature: str
    potential_evapotranspiration: str
    streamflow: str | None = None
    timestamp: str | None = None
    additional_predictors: tuple[AdditionalPredictorColumnConfig, ...] = ()

    @field_validator(
        "precipitation",
        "temperature",
        "potential_evapotranspiration",
        "streamflow",
        "timestamp",
    )
    @classmethod
    def validate_column_name(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("CSV column names cannot be empty.")
        return value

    @model_validator(mode="after")
    def validate_unique_columns(self) -> "CsvColumnMapping":
        values = [
            self.precipitation,
            self.temperature,
            self.potential_evapotranspiration,
        ]
        if self.streamflow is not None:
            values.append(self.streamflow)
        if self.timestamp is not None:
            values.append(self.timestamp)
        values.extend(item.column for item in self.additional_predictors)
        if len(set(values)) != len(values):
            raise ValueError("Every variable must map to a distinct CSV column.")
        predictor_names = [item.name for item in self.additional_predictors]
        reserved = {
            "precipitation",
            "temperature",
            "potential_evapotranspiration",
            "streamflow",
            "timestamp",
        }
        if reserved.intersection(predictor_names):
            raise ValueError(
                "Additional predictor names cannot duplicate standard variable roles."
            )
        if len(set(predictor_names)) != len(predictor_names):
            raise ValueError("Additional predictor names must be unique.")
        return self


class UnitConversionConfig(StrictConfigModel):
    """Declared source units converted to the daily model units."""

    precipitation: Literal["mm/day", "inch/day"] = "mm/day"
    temperature: Literal["degC", "degF", "K"] = "degC"
    potential_evapotranspiration: Literal["mm/day", "inch/day"] = "mm/day"
    streamflow: Literal["mm/day", "m3/s"] = "mm/day"
    basin_area_km2: float | None = Field(default=None, gt=0.0)

    @model_validator(mode="after")
    def validate_streamflow_conversion(self) -> "UnitConversionConfig":
        if self.streamflow == "m3/s" and self.basin_area_km2 is None:
            raise ValueError(
                "basin_area_km2 is required to convert streamflow from m3/s to mm/day."
            )
        return self


class MissingValueConfig(StrictConfigModel):
    """Explicit policies for missing numeric values after parsing."""

    precipitation: Literal["error", "zero", "linear_interpolation"] = "error"
    temperature: Literal["error", "linear_interpolation"] = "error"
    potential_evapotranspiration: Literal[
        "error", "zero", "linear_interpolation"
    ] = "error"
    streamflow: Literal["exclude", "linear_interpolation"] = "exclude"
    maximum_interpolation_gap: int = Field(default=3, ge=1, le=366)
    text_tokens: tuple[str, ...] = ("", "NA", "N/A", "NaN", "null")
    numeric_sentinels: tuple[float, ...] = ()

    @field_validator("text_tokens")
    @classmethod
    def validate_text_tokens(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(dict.fromkeys(token.strip() for token in value))
        if len(normalized) > 50:
            raise ValueError("At most 50 text missing-value tokens can be declared.")
        return normalized


class NegativeValueConfig(StrictConfigModel):
    """Explicit policies for physically questionable negative values."""

    precipitation: Literal["error", "clip_to_zero", "set_missing"] = "error"
    potential_evapotranspiration: Literal[
        "error", "clip_to_zero", "set_missing"
    ] = "error"
    streamflow: Literal["keep", "error", "exclude", "clip_to_zero"] = "keep"


class TimestampQualityConfig(StrictConfigModel):
    """Timestamp checks required by the current daily model units."""

    expected_step: Literal["daily"] = "daily"
    require_timestamp: bool = False


class DataProcessingConfig(StrictConfigModel):
    """Reproducible data-quality and transformation choices."""

    units: UnitConversionConfig = Field(default_factory=UnitConversionConfig)
    missing: MissingValueConfig = Field(default_factory=MissingValueConfig)
    negative: NegativeValueConfig = Field(default_factory=NegativeValueConfig)
    timestamps: TimestampQualityConfig = Field(default_factory=TimestampQualityConfig)


class CsvDataSourceConfig(StrictConfigModel):
    """Portable CSV source resolved inside the configured project workspace."""

    kind: Literal["csv"] = "csv"
    path: str
    columns: CsvColumnMapping
    delimiter: str = ","
    encoding: str = "utf-8-sig"
    expected_sha256: str | None = None
    processing: DataProcessingConfig = Field(default_factory=DataProcessingConfig)

    @field_validator("path")
    @classmethod
    def validate_portable_relative_path(cls, value: str) -> str:
        candidate = value.strip()
        if not candidate:
            raise ValueError("CSV path cannot be empty.")
        if PurePosixPath(candidate).is_absolute() or PureWindowsPath(candidate).is_absolute():
            raise ValueError("CSV path must be relative to the project workspace.")
        normalized_parts = PurePosixPath(candidate.replace("\\", "/")).parts
        if ".." in normalized_parts:
            raise ValueError("CSV path cannot leave the project workspace.")
        return candidate.replace("\\", "/")

    @field_validator("delimiter")
    @classmethod
    def validate_delimiter(cls, value: str) -> str:
        if len(value) != 1:
            raise ValueError("CSV delimiter must contain exactly one character.")
        return value

    @field_validator("encoding")
    @classmethod
    def validate_encoding(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("CSV encoding cannot be empty.")
        return value

    @field_validator("expected_sha256")
    @classmethod
    def validate_expected_sha256(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip().lower()
        if len(normalized) != 64 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("expected_sha256 must contain 64 hexadecimal characters.")
        return normalized


class NoSnowConfig(StrictConfigModel):
    """Configuration for simulations without a snow process."""

    module_id: Literal["snow.none.v1"] = "snow.none.v1"


class ParameterTreatmentConfig(StrictConfigModel):
    """Whether one process parameter is fixed or included in calibration."""

    mode: Literal["fixed", "calibrate"] = "fixed"
    lower_bound: float
    upper_bound: float

    @model_validator(mode="after")
    def validate_bounds(self) -> "ParameterTreatmentConfig":
        if self.lower_bound >= self.upper_bound:
            raise ValueError("Calibration lower_bound must be smaller than upper_bound.")
        return self


class DegreeDaySnowCalibrationConfig(StrictConfigModel):
    """Parameter treatments for the hard degree-day snow process."""

    snowfall_temperature: ParameterTreatmentConfig = Field(
        default_factory=lambda: ParameterTreatmentConfig(
            lower_bound=-3.0, upper_bound=0.0
        )
    )
    melt_temperature: ParameterTreatmentConfig = Field(
        default_factory=lambda: ParameterTreatmentConfig(
            lower_bound=0.0, upper_bound=3.0
        )
    )
    degree_day_factor: ParameterTreatmentConfig = Field(
        default_factory=lambda: ParameterTreatmentConfig(
            lower_bound=0.0, upper_bound=5.0
        )
    )


class SmoothDegreeDaySnowCalibrationConfig(DegreeDaySnowCalibrationConfig):
    """Parameter treatments for the smooth degree-day snow process."""

    sharpness: ParameterTreatmentConfig = Field(
        default_factory=lambda: ParameterTreatmentConfig(
            lower_bound=0.5, upper_bound=20.0
        )
    )


class DegreeDaySnowConfig(StrictConfigModel):
    """Configuration for the hard-threshold degree-day snow module."""

    module_id: Literal["snow.degree_day.hard.v1"] = "snow.degree_day.hard.v1"
    snowfall_temperature: float = 0.0
    melt_temperature: float = 0.0
    degree_day_factor: float = Field(default=2.0, ge=0.0)
    calibration: DegreeDaySnowCalibrationConfig = Field(
        default_factory=DegreeDaySnowCalibrationConfig
    )

    @model_validator(mode="after")
    def validate_calibrated_defaults(self) -> "DegreeDaySnowConfig":
        values = {
            "snowfall_temperature": self.snowfall_temperature,
            "melt_temperature": self.melt_temperature,
            "degree_day_factor": self.degree_day_factor,
        }
        for name, value in values.items():
            treatment = getattr(self.calibration, name)
            if treatment.mode == "calibrate" and not (
                treatment.lower_bound <= value <= treatment.upper_bound
            ):
                raise ValueError(
                    f"Initial {name} must lie inside its calibration bounds."
                )
        return self


class SmoothDegreeDaySnowConfig(StrictConfigModel):
    """Configuration for the differentiable smooth degree-day snow module."""

    module_id: Literal["snow.degree_day.smooth.v1"] = "snow.degree_day.smooth.v1"
    snowfall_temperature: float = 0.0
    melt_temperature: float = 0.0
    degree_day_factor: float = Field(default=2.0, ge=0.0)
    sharpness: float = Field(default=5.0, gt=0.0)
    calibration: SmoothDegreeDaySnowCalibrationConfig = Field(
        default_factory=SmoothDegreeDaySnowCalibrationConfig
    )

    @model_validator(mode="after")
    def validate_calibrated_defaults(self) -> "SmoothDegreeDaySnowConfig":
        values = {
            "snowfall_temperature": self.snowfall_temperature,
            "melt_temperature": self.melt_temperature,
            "degree_day_factor": self.degree_day_factor,
            "sharpness": self.sharpness,
        }
        for name, value in values.items():
            treatment = getattr(self.calibration, name)
            if treatment.mode == "calibrate" and not (
                treatment.lower_bound <= value <= treatment.upper_bound
            ):
                raise ValueError(
                    f"Initial {name} must lie inside its calibration bounds."
                )
        return self


SnowModuleConfig = Annotated[
    NoSnowConfig | DegreeDaySnowConfig | SmoothDegreeDaySnowConfig,
    Field(discriminator="module_id"),
]


class NoRoutingConfig(StrictConfigModel):
    """Configuration for direct use of model-response discharge."""

    process_id: Literal["routing.none.v1"] = "routing.none.v1"


class MaxbasRoutingCalibrationConfig(StrictConfigModel):
    """Treatment of the discrete HBV MAXBAS routing parameter."""

    maxbas: ParameterTreatmentConfig = Field(
        default_factory=lambda: ParameterTreatmentConfig(
            lower_bound=1.0, upper_bound=7.0
        )
    )

    @model_validator(mode="after")
    def validate_integer_bounds(self) -> "MaxbasRoutingCalibrationConfig":
        if any(
            not float(value).is_integer()
            for value in (self.maxbas.lower_bound, self.maxbas.upper_bound)
        ):
            raise ValueError("MAXBAS calibration bounds must be whole days.")
        if not 1 <= self.maxbas.lower_bound < self.maxbas.upper_bound <= 7:
            raise ValueError("MAXBAS calibration bounds must lie in [1, 7] days.")
        return self


class TriangularMaxbasRoutingConfig(StrictConfigModel):
    """Explicit HBV-light triangular transfer function."""

    process_id: Literal["routing.maxbas.triangular.v1"] = (
        "routing.maxbas.triangular.v1"
    )
    maxbas: int = Field(default=3, ge=1, le=7)
    calibration: MaxbasRoutingCalibrationConfig = Field(
        default_factory=MaxbasRoutingCalibrationConfig
    )
    initial_state: tuple[float, float, float, float, float, float] = (
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    )

    @field_validator("initial_state")
    @classmethod
    def validate_initial_state(
        cls, value: tuple[float, float, float, float, float, float]
    ) -> tuple[float, float, float, float, float, float]:
        if any(item < 0.0 for item in value):
            raise ValueError("Pending routing volumes cannot be negative.")
        return value

    @model_validator(mode="after")
    def validate_calibrated_default(self) -> "TriangularMaxbasRoutingConfig":
        treatment = self.calibration.maxbas
        if treatment.mode == "calibrate" and not (
            treatment.lower_bound <= self.maxbas <= treatment.upper_bound
        ):
            raise ValueError(
                "Initial MAXBAS must lie inside its calibration bounds."
            )
        return self


HbvRoutingConfig = Annotated[
    NoRoutingConfig | TriangularMaxbasRoutingConfig,
    Field(discriminator="process_id"),
]


class ExpHydroModelConfig(StrictConfigModel):
    """Scientific choices for the mass-conservative EXP-HYDRO variant."""

    model_id: Literal["exp_hydro.mass_conservative.v1"] = (
        "exp_hydro.mass_conservative.v1"
    )
    snow: SnowModuleConfig = Field(default_factory=NoSnowConfig)
    initial_state: tuple[float, float] = (0.0, 0.0)
    substeps: int = Field(default=4, ge=1, le=96)

    @field_validator("initial_state")
    @classmethod
    def validate_initial_state(
        cls, value: tuple[float, float]
    ) -> tuple[float, float]:
        if any(item < 0.0 for item in value):
            raise ValueError("Initial snow and soil storages cannot be negative.")
        return value

    @model_validator(mode="after")
    def validate_snow_state_compatibility(self) -> "ExpHydroModelConfig":
        if isinstance(self.snow, NoSnowConfig) and self.initial_state[0] != 0.0:
            raise ValueError("snow.none.v1 requires zero initial snow storage.")
        return self


class HbvModelConfig(StrictConfigModel):
    """Scientific choices for the conservative daily HBV variant."""

    model_id: Literal["hbv.mass_conservative_daily.v1"] = (
        "hbv.mass_conservative_daily.v1"
    )
    snow: SnowModuleConfig = Field(default_factory=NoSnowConfig)
    routing: HbvRoutingConfig = Field(default_factory=NoRoutingConfig)
    initial_state: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    @field_validator("initial_state")
    @classmethod
    def validate_initial_state(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        if any(item < 0.0 for item in value):
            raise ValueError(
                "Initial snow, soil, upper-zone, and lower-zone storages "
                "cannot be negative."
            )
        return value

    @model_validator(mode="after")
    def validate_snow_state_compatibility(self) -> "HbvModelConfig":
        if isinstance(self.snow, NoSnowConfig) and self.initial_state[0] != 0.0:
            raise ValueError("snow.none.v1 requires zero initial snow storage.")
        return self


ConceptualModelConfig = Annotated[
    ExpHydroModelConfig | HbvModelConfig,
    Field(discriminator="model_id"),
]


class LaggedFeatureConfig(StrictConfigModel):
    """One causal shifted copy of a raw or physically exposed predictor."""

    source: str = Field(min_length=1, max_length=128)
    lag_steps: int = Field(ge=1, le=1095)

    @field_validator("source")
    @classmethod
    def validate_source(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("Lagged feature sources cannot be blank.")
        return normalized


def _validate_neural_features(
    feature_roles: tuple[str, ...],
    lagged_features: tuple[LaggedFeatureConfig, ...],
    *,
    model_label: str,
) -> None:
    if not feature_roles and not lagged_features:
        raise ValueError(
            f"At least one {model_label} input feature or lagged feature must be selected."
        )
    if len(set(feature_roles)) != len(feature_roles):
        raise ValueError(f"{model_label} feature roles cannot contain duplicates.")
    lag_keys = [(item.source, item.lag_steps) for item in lagged_features]
    if len(set(lag_keys)) != len(lag_keys):
        raise ValueError(f"{model_label} lagged features cannot contain duplicates.")


class MlpModelConfig(StrictConfigModel):
    """TensorFlow MLP for rainfall-runoff prediction."""

    model_id: Literal[
        "mlp.tensorflow_s2v.v1",
        "mlp.tensorflow_s2s.v2",
    ] = "mlp.tensorflow_s2v.v1"
    feature_roles: tuple[str, ...] = (
        "precipitation",
        "temperature",
        "potential_evapotranspiration",
    )
    lagged_features: tuple[LaggedFeatureConfig, ...] = ()
    lookback: int = Field(default=30, ge=1, le=1095)
    s2s_warmup: int = Field(default=0, ge=0, le=1094)
    hidden_layers: tuple[int, ...] = (64, 32)
    activation: Literal["relu", "tanh"] = "relu"
    output_activation: Literal["linear", "softplus"] = "softplus"
    input_transform: Literal["standardize_training_only"] = (
        "standardize_training_only"
    )
    target_transform: Literal["standardize_training_only"] = (
        "standardize_training_only"
    )

    @field_validator("hidden_layers")
    @classmethod
    def validate_hidden_layers(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not 1 <= len(value) <= 5:
            raise ValueError("MLP requires between one and five hidden layers.")
        if any(width < 1 or width > 2048 for width in value):
            raise ValueError("Every MLP hidden-layer width must lie in [1, 2048].")
        return value

    @model_validator(mode="after")
    def validate_features(self) -> "MlpModelConfig":
        _validate_neural_features(
            self.feature_roles, self.lagged_features, model_label="MLP"
        )
        if self.model_id == "mlp.tensorflow_s2v.v1" and self.s2s_warmup:
            raise ValueError("The S2V MLP cannot define an S2S warm-up.")
        if self.model_id == "mlp.tensorflow_s2s.v2" and self.s2s_warmup >= self.lookback:
            raise ValueError("MLP S2S warm-up must be shorter than its sequence.")
        return self


class LstmModelConfig(StrictConfigModel):
    """TensorFlow LSTM for rainfall-runoff prediction."""

    model_id: Literal[
        "lstm.tensorflow_s2v.v1",
        "lstm.tensorflow_stacked_s2v.v2",
        "lstm.tensorflow_stacked_s2s.v3",
    ] = "lstm.tensorflow_s2v.v1"
    feature_roles: tuple[str, ...] = (
        "precipitation",
        "temperature",
        "potential_evapotranspiration",
    )
    lagged_features: tuple[LaggedFeatureConfig, ...] = ()
    lookback: int = Field(default=30, ge=1, le=1095)
    s2s_warmup: int = Field(default=0, ge=0, le=1094)
    recurrent_units: int = Field(default=32, ge=1, le=2048)
    recurrent_layers: tuple[int, ...] = (32,)
    gradient_clip_norm: float = Field(default=5.0, gt=0.0, le=1_000_000.0)
    output_activation: Literal["linear", "softplus"] = "softplus"
    input_transform: Literal["standardize_training_only"] = (
        "standardize_training_only"
    )
    target_transform: Literal["standardize_training_only"] = (
        "standardize_training_only"
    )

    @model_validator(mode="after")
    def validate_features(self) -> "LstmModelConfig":
        _validate_neural_features(
            self.feature_roles, self.lagged_features, model_label="LSTM"
        )
        if not 1 <= len(self.recurrent_layers) <= 5:
            raise ValueError("Stacked LSTM requires between one and five layers.")
        if any(width < 1 or width > 2048 for width in self.recurrent_layers):
            raise ValueError("Every LSTM recurrent-layer width must lie in [1, 2048].")
        if self.model_id == "lstm.tensorflow_s2v.v1":
            if (
                "recurrent_layers" in self.model_fields_set
                and self.recurrent_layers != (self.recurrent_units,)
            ):
                raise ValueError(
                    "The LSTM v1 artifact has exactly one recurrent layer; "
                    "recurrent_layers must match recurrent_units."
                )
            object.__setattr__(
                self, "recurrent_layers", (self.recurrent_units,)
            )
        elif (
            "recurrent_units" in self.model_fields_set
            and self.recurrent_units != self.recurrent_layers[0]
        ):
            raise ValueError(
                "recurrent_units must equal the first stacked LSTM layer width."
            )
        else:
            object.__setattr__(
                self, "recurrent_units", self.recurrent_layers[0]
            )
        if self.model_id != "lstm.tensorflow_stacked_s2s.v3" and self.s2s_warmup:
            raise ValueError("S2V LSTM variants cannot define an S2S warm-up.")
        if (
            self.model_id == "lstm.tensorflow_stacked_s2s.v3"
            and self.s2s_warmup >= self.lookback
        ):
            raise ValueError("LSTM S2S warm-up must be shorter than its sequence.")
        return self


class GruModelConfig(StrictConfigModel):
    """TensorFlow GRU for rainfall-runoff prediction."""

    model_id: Literal[
        "gru.tensorflow_s2v.v1",
        "gru.tensorflow_stacked_s2v.v2",
        "gru.tensorflow_stacked_s2s.v3",
    ] = "gru.tensorflow_s2v.v1"
    feature_roles: tuple[str, ...] = (
        "precipitation",
        "temperature",
        "potential_evapotranspiration",
    )
    lagged_features: tuple[LaggedFeatureConfig, ...] = ()
    lookback: int = Field(default=30, ge=1, le=1095)
    s2s_warmup: int = Field(default=0, ge=0, le=1094)
    recurrent_units: int = Field(default=32, ge=1, le=2048)
    recurrent_layers: tuple[int, ...] = (32,)
    gradient_clip_norm: float = Field(default=5.0, gt=0.0, le=1_000_000.0)
    output_activation: Literal["linear", "softplus"] = "softplus"
    input_transform: Literal["standardize_training_only"] = (
        "standardize_training_only"
    )
    target_transform: Literal["standardize_training_only"] = (
        "standardize_training_only"
    )

    @model_validator(mode="after")
    def validate_features(self) -> "GruModelConfig":
        _validate_neural_features(
            self.feature_roles, self.lagged_features, model_label="GRU"
        )
        if not 1 <= len(self.recurrent_layers) <= 5:
            raise ValueError("Stacked GRU requires between one and five layers.")
        if any(width < 1 or width > 2048 for width in self.recurrent_layers):
            raise ValueError("Every GRU recurrent-layer width must lie in [1, 2048].")
        if self.model_id == "gru.tensorflow_s2v.v1":
            if (
                "recurrent_layers" in self.model_fields_set
                and self.recurrent_layers != (self.recurrent_units,)
            ):
                raise ValueError(
                    "The GRU v1 artifact has exactly one recurrent layer; "
                    "recurrent_layers must match recurrent_units."
                )
            object.__setattr__(
                self, "recurrent_layers", (self.recurrent_units,)
            )
        elif (
            "recurrent_units" in self.model_fields_set
            and self.recurrent_units != self.recurrent_layers[0]
        ):
            raise ValueError(
                "recurrent_units must equal the first stacked GRU layer width."
            )
        else:
            object.__setattr__(
                self, "recurrent_units", self.recurrent_layers[0]
            )
        if self.model_id != "gru.tensorflow_stacked_s2s.v3" and self.s2s_warmup:
            raise ValueError("S2V GRU variants cannot define an S2S warm-up.")
        if (
            self.model_id == "gru.tensorflow_stacked_s2s.v3"
            and self.s2s_warmup >= self.lookback
        ):
            raise ValueError("GRU S2S warm-up must be shorter than its sequence.")
        return self


class Conv1dLayerConfig(StrictConfigModel):
    """One causal convolutional layer in the TensorFlow CNN-1D stack."""

    filters: int = Field(default=16, ge=1, le=2048)
    kernel_size: int = Field(default=5, ge=1, le=365)
    activation: Literal["relu", "tanh"] = "relu"


class Cnn1dModelConfig(StrictConfigModel):
    """TensorFlow causal CNN-1D for rainfall-runoff prediction."""

    model_id: Literal[
        "cnn1d.tensorflow_s2v.v1",
        "cnn1d.tensorflow_s2s.v2",
    ] = "cnn1d.tensorflow_s2v.v1"
    feature_roles: tuple[str, ...] = (
        "precipitation",
        "temperature",
        "potential_evapotranspiration",
    )
    lagged_features: tuple[LaggedFeatureConfig, ...] = ()
    lookback: int = Field(default=30, ge=1, le=1095)
    s2s_warmup: int = Field(default=0, ge=0, le=1094)
    convolution_layers: tuple[Conv1dLayerConfig, ...] = (
        Conv1dLayerConfig(filters=16, kernel_size=5, activation="relu"),
        Conv1dLayerConfig(filters=8, kernel_size=3, activation="relu"),
    )
    gradient_clip_norm: float = Field(default=5.0, gt=0.0, le=1_000_000.0)
    output_activation: Literal["linear", "softplus"] = "softplus"
    input_transform: Literal["standardize_training_only"] = (
        "standardize_training_only"
    )
    target_transform: Literal["standardize_training_only"] = (
        "standardize_training_only"
    )

    @field_validator("convolution_layers")
    @classmethod
    def validate_convolution_layers(
        cls, value: tuple[Conv1dLayerConfig, ...]
    ) -> tuple[Conv1dLayerConfig, ...]:
        if not 1 <= len(value) <= 5:
            raise ValueError("CNN-1D requires between one and five convolutional layers.")
        return value

    @model_validator(mode="after")
    def validate_receptive_field(self) -> "Cnn1dModelConfig":
        _validate_neural_features(
            self.feature_roles, self.lagged_features, model_label="CNN-1D"
        )
        receptive_field = 1 + sum(
            layer.kernel_size - 1 for layer in self.convolution_layers
        )
        if receptive_field > self.lookback:
            raise ValueError(
                "CNN-1D lookback must be at least as long as the convolutional "
                f"receptive field ({receptive_field})."
            )
        if self.model_id == "cnn1d.tensorflow_s2v.v1" and self.s2s_warmup:
            raise ValueError("The S2V CNN-1D cannot define an S2S warm-up.")
        if self.model_id == "cnn1d.tensorflow_s2s.v2":
            if self.s2s_warmup >= self.lookback:
                raise ValueError(
                    "CNN-1D S2S warm-up must be shorter than its sequence."
                )
            if self.s2s_warmup < receptive_field - 1:
                raise ValueError(
                    "CNN-1D S2S warm-up must cover the complete causal "
                    f"receptive field ({receptive_field - 1} rows)."
                )
        return self


NeuralCorrectorConfig = Annotated[
    MlpModelConfig | LstmModelConfig | GruModelConfig | Cnn1dModelConfig,
    Field(discriminator="model_id"),
]


class PamlPhysicalBackboneConfig(StrictConfigModel):
    """Physical backbone selected for the differentiable P-RNN."""

    model_id: Literal[
        "exp_hydro.mass_conservative.v1",
        "hbv.mass_conservative_daily.v1",
    ] = "exp_hydro.mass_conservative.v1"


class PamlModelConfig(StrictConfigModel):
    """Joint P-RNN and causal-CNN hybrid following Jiang et al. (2020)."""

    model_id: Literal["paml.tensorflow_prnn_cnn_s2s.v1"] = (
        "paml.tensorflow_prnn_cnn_s2s.v1"
    )
    physical_model: PamlPhysicalBackboneConfig = Field(
        default_factory=PamlPhysicalBackboneConfig
    )
    cnn_filters: int = Field(default=8, ge=1, le=256)
    cnn_kernel_size: int = Field(default=10, ge=1, le=365)
    training_strategy: Literal["joint_end_to_end"] = "joint_end_to_end"
    enforce_nonnegative_output: bool = True


class DiscoveryLossConfig(StrictConfigModel):
    """Independently reported scientific penalties for process discovery."""

    decorrelation_weight: float = Field(default=0.0, ge=0.0, le=1_000_000.0)
    mass_balance_weight: float = Field(default=0.0, ge=0.0, le=1_000_000.0)
    smoothness_weight: float = Field(default=0.01, ge=0.0, le=1_000_000.0)
    smoothness_threshold: float = Field(default=0.1, ge=0.0)
    binary_weight: float = Field(default=0.0, ge=0.0, le=1_000_000.0)
    sparsity_weight: float = Field(default=0.0, ge=0.0, le=1_000_000.0)
    target_process_correlation: float = Field(default=0.0, ge=-1.0, le=1.0)


class DeepDiscoverModelConfig(StrictConfigModel):
    """Controlled single-DU DeepDiscover using the validated NumPy reference engine."""

    model_id: Literal["deepdiscover.numpy_instantaneous_s2s.v1"] = (
        "deepdiscover.numpy_instantaneous_s2s.v1"
    )
    snow: SnowModuleConfig = Field(
        default_factory=lambda: SmoothDegreeDaySnowConfig()
    )
    initial_state: tuple[float, float] = (0.0, 0.0)
    process_count: int = Field(default=1, ge=1, le=8)
    hidden_units: int = Field(default=8, ge=1, le=256)
    train_input_switches: bool = False
    train_process_switches: bool = False
    input_transform: Literal["standardize_training_only"] = "standardize_training_only"
    loss: DiscoveryLossConfig = Field(default_factory=DiscoveryLossConfig)

    @field_validator("initial_state")
    @classmethod
    def validate_initial_state(
        cls, value: tuple[float, float]
    ) -> tuple[float, float]:
        if any(item < 0.0 for item in value):
            raise ValueError(
                "DeepDiscover initial snow and discovered storages cannot be negative."
            )
        return value

    @model_validator(mode="after")
    def validate_snow_process(self) -> "DeepDiscoverModelConfig":
        if isinstance(self.snow, NoSnowConfig) and self.initial_state[0] != 0.0:
            raise ValueError("snow.none.v1 requires zero initial snow storage.")
        return self


ModelConfig = Annotated[
    ExpHydroModelConfig
    | HbvModelConfig
    | MlpModelConfig
    | LstmModelConfig
    | GruModelConfig
    | Cnn1dModelConfig
    | PamlModelConfig
    | DeepDiscoverModelConfig,
    Field(discriminator="model_id"),
]


class CalibrationPeriodConfig(StrictConfigModel):
    """Chronological conceptual calibration and held-out evaluation periods."""

    protocol: Literal[
        "calibration_validation_test",
        "calibration_test",
        "legacy_calibration_only",
    ] = "calibration_validation_test"
    start_index: int = Field(default=0, ge=0)
    calibration_end_index: int | None = Field(default=1000, gt=0)
    validation_end_index: int | None = Field(default=1300, gt=0)
    test_end_index: int | None = Field(default=None, gt=0)
    warmup_steps: int = Field(default=0, ge=0)

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_end_index(cls, value: Any) -> Any:
        if isinstance(value, dict) and "protocol" not in value:
            period = dict(value)
            if "end_index" in period:
                end = period.pop("end_index")
                period.update(
                    {
                        "protocol": "legacy_calibration_only",
                        "calibration_end_index": end,
                        "validation_end_index": None,
                        "test_end_index": None,
                    }
                )
                return period
            if period.get("validation_end_index") is not None:
                period["protocol"] = "calibration_validation_test"
                return period
            if period.get("test_end_index") is not None:
                period["protocol"] = "calibration_test"
                period["validation_end_index"] = None
                return period
        return value

    @model_validator(mode="after")
    def validate_period(self) -> "CalibrationPeriodConfig":
        calibration_end = self.calibration_end_index
        if calibration_end is None:
            if self.protocol != "legacy_calibration_only":
                raise ValueError(
                    "calibration_end_index is required for held-out evaluation."
                )
            if self.validation_end_index is not None or self.test_end_index is not None:
                raise ValueError(
                    "Legacy calibration-only periods cannot define holdout boundaries."
                )
            return self
        if calibration_end <= self.start_index:
            raise ValueError(
                "calibration_end_index must be greater than start_index."
            )
        if self.warmup_steps >= calibration_end - self.start_index:
            raise ValueError(
                "warmup_steps must be shorter than the calibration partition."
            )
        if self.protocol == "legacy_calibration_only":
            if self.validation_end_index is not None or self.test_end_index is not None:
                raise ValueError(
                    "Legacy calibration-only periods cannot define holdout boundaries."
                )
            return self
        if self.protocol == "calibration_test":
            if self.validation_end_index is not None:
                raise ValueError(
                    "calibration_test periods cannot define validation_end_index."
                )
            if (
                self.test_end_index is not None
                and self.test_end_index <= calibration_end
            ):
                raise ValueError(
                    "test_end_index must be greater than calibration_end_index."
                )
            return self
        if (
            self.validation_end_index is None
            or self.validation_end_index <= calibration_end
        ):
            raise ValueError(
                "calibration_validation_test requires validation_end_index "
                "greater than calibration_end_index."
            )
        if (
            self.test_end_index is not None
            and self.test_end_index <= self.validation_end_index
        ):
            raise ValueError(
                "test_end_index must be greater than validation_end_index."
            )
        return self


class ObjectiveConfig(StrictConfigModel):
    """Deterministic fitting and checkpoint objective."""

    objective_id: Literal[
        "nash_sutcliffe_efficiency",
        "kling_gupta_efficiency_2009",
        "mean_square_error",
        "root_mean_square_error",
        "range_normalized_root_mean_square_error",
        "mean_absolute_error",
    ] = "nash_sutcliffe_efficiency"


class UniformRandomSearchExperimentConfig(StrictConfigModel):
    """Portable configuration for the transparent reference search."""

    algorithm_id: Literal["uniform_random_search.v1"] = "uniform_random_search.v1"
    n_evaluations: int = Field(default=100, ge=1, le=1_000_000)
    seed: int = 0
    include_defaults: bool = True
    continue_on_candidate_error: bool = True


class DifferentialEvolutionSearchExperimentConfig(StrictConfigModel):
    """Bounded DE/rand/1/bin configuration for conceptual parameters."""

    algorithm_id: Literal["differential_evolution.rand1bin.v1"] = (
        "differential_evolution.rand1bin.v1"
    )
    generations: int = Field(default=25, ge=1, le=100_000)
    population_size: int | None = Field(default=None, ge=4, le=100_000)
    mutation_factor: float = Field(default=0.8, gt=0.0, le=2.0)
    crossover_probability: float = Field(default=0.9, ge=0.0, le=1.0)
    seed: int = 0
    include_defaults: bool = True
    continue_on_candidate_error: bool = True


CalibrationSearchExperimentConfig = (
    UniformRandomSearchExperimentConfig
    | DifferentialEvolutionSearchExperimentConfig
)


class CalibrationTaskConfig(StrictConfigModel):
    """First supported task: deterministic conceptual-model calibration."""

    task_type: Literal["calibration"] = "calibration"
    period: CalibrationPeriodConfig = Field(default_factory=CalibrationPeriodConfig)
    objective: ObjectiveConfig = Field(default_factory=ObjectiveConfig)
    search: CalibrationSearchExperimentConfig = Field(
        default_factory=DifferentialEvolutionSearchExperimentConfig
    )


class TrainingPeriodConfig(StrictConfigModel):
    """Disjoint row partitions used by a supervised neural experiment."""

    start_index: int = Field(default=0, ge=0)
    train_end_index: int = Field(gt=0)
    validation_end_index: int = Field(gt=0)
    test_end_index: int | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def validate_order(self) -> "TrainingPeriodConfig":
        end = self.test_end_index
        if not self.start_index < self.train_end_index < self.validation_end_index:
            raise ValueError(
                "Training boundaries must satisfy start < train end < validation end."
            )
        if end is not None and self.validation_end_index >= end:
            raise ValueError("test_end_index must be greater than validation_end_index.")
        return self


class AdamTrainingConfig(StrictConfigModel):
    """Mini-batch Adam settings for TensorFlow neural models."""

    optimizer_id: Literal["adam.tensorflow.v1"] = "adam.tensorflow.v1"
    epochs: int = Field(default=200, ge=1, le=100_000)
    batch_size: int = Field(default=64, ge=1, le=65_536)
    learning_rate: float = Field(default=0.001, gt=0.0, le=1.0)
    patience: int = Field(default=20, ge=1, le=10_000)
    minimum_delta: float = Field(default=1e-6, ge=0.0)
    gradient_clip_norm: float = Field(default=5.0, gt=0.0, le=1_000_000.0)
    seed: int = 0
    shuffle: bool = True


class InitializationEnsembleConfig(StrictConfigModel):
    """Repeated neural fitting with identical settings and distinct seeds."""

    enabled: bool = False
    members: int = Field(default=5, ge=3, le=20)
    aggregation: Literal["arithmetic_mean"] = "arithmetic_mean"
    lower_quantile: float = Field(default=0.1, ge=0.0, le=1.0)
    upper_quantile: float = Field(default=0.9, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_quantiles(self) -> "InitializationEnsembleConfig":
        if self.lower_quantile >= self.upper_quantile:
            raise ValueError(
                "Initialization-ensemble lower_quantile must be smaller than "
                "upper_quantile."
            )
        return self


class NeuralTrainingTaskConfig(StrictConfigModel):
    """Supervised streamflow task with explicit temporal alignment."""

    task_type: Literal["training"] = "training"
    period: TrainingPeriodConfig
    target_role: Literal["streamflow"] = "streamflow"
    lead_steps: Literal[0, 1] = 1
    stride: int = Field(default=1, ge=1, le=366)
    objective: ObjectiveConfig = Field(
        default_factory=lambda: ObjectiveConfig(
            objective_id="root_mean_square_error"
        )
    )
    training: AdamTrainingConfig = Field(default_factory=AdamTrainingConfig)
    ensemble: InitializationEnsembleConfig = Field(
        default_factory=InitializationEnsembleConfig
    )


class PamlTrainingTaskConfig(StrictConfigModel):
    """Joint end-to-end PaML training task."""

    task_type: Literal["paml_training"] = "paml_training"
    period: TrainingPeriodConfig
    target_role: Literal["streamflow"] = "streamflow"
    lead_steps: Literal[0] = 0
    stride: Literal[1] = 1
    objective: ObjectiveConfig = Field(
        default_factory=lambda: ObjectiveConfig(
            objective_id="nash_sutcliffe_efficiency"
        )
    )
    warmup_steps: int = Field(default=365, ge=0)
    training: AdamTrainingConfig = Field(default_factory=AdamTrainingConfig)

    @model_validator(mode="after")
    def validate_warmup(self) -> "PamlTrainingTaskConfig":
        training_length = self.period.train_end_index - self.period.start_index
        if self.warmup_steps >= training_length:
            raise ValueError("PaML warm-up must be shorter than the training partition.")
        return self


class DiscoveryAdamTrainingConfig(StrictConfigModel):
    """Full-sequence Adam settings for a continuous recurrent DD trajectory."""

    optimizer_id: Literal["adam.numpy.full_sequence.v1"] = (
        "adam.numpy.full_sequence.v1"
    )
    epochs: int = Field(default=600, ge=1, le=100_000)
    learning_rate: float = Field(default=0.01, gt=0.0, le=1.0)
    patience: int = Field(default=75, ge=1, le=10_000)
    minimum_delta: float = Field(default=0.000001, ge=0.0)
    gradient_clip_norm: float = Field(default=5.0, gt=0.0, le=1_000_000.0)
    seed: int = 200


class DiscoveryTrainingTaskConfig(StrictConfigModel):
    """Continuous same-time sequence training for instantaneous DD."""

    task_type: Literal["discovery_training"] = "discovery_training"
    period: TrainingPeriodConfig
    target_role: Literal["streamflow"] = "streamflow"
    warmup_steps: int = Field(default=365, ge=0)
    objective: ObjectiveConfig = Field(
        default_factory=lambda: ObjectiveConfig(
            objective_id="nash_sutcliffe_efficiency"
        )
    )
    training: DiscoveryAdamTrainingConfig = Field(
        default_factory=DiscoveryAdamTrainingConfig
    )

    @model_validator(mode="after")
    def validate_warmup(self) -> "DiscoveryTrainingTaskConfig":
        training_length = self.period.train_end_index - self.period.start_index
        if self.warmup_steps >= training_length:
            raise ValueError(
                "DeepDiscover warm-up must be shorter than the training partition."
            )
        return self


ExperimentTaskConfig = Annotated[
    CalibrationTaskConfig
    | NeuralTrainingTaskConfig
    | PamlTrainingTaskConfig
    | DiscoveryTrainingTaskConfig,
    Field(discriminator="task_type"),
]


class ExperimentConfigV1(StrictConfigModel):
    """Complete, versioned definition of one reproducible experiment."""

    schema_version: Literal[
        "1.0",
        "1.1",
        "1.2",
        "1.3",
        "1.4",
        "1.5",
        "1.6",
        "1.7",
        "1.8",
        "1.9",
        "1.10",
        "1.11",
        "1.12",
        "1.13",
        "1.14",
        "1.15",
        "1.16",
        "1.17",
        "1.18",
        "1.19",
        "1.20",
        "1.21",
    ] = (
        CURRENT_SCHEMA_VERSION
    )
    experiment_id: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    name: str = Field(min_length=1, max_length=200)
    description: str = ""
    data: CsvDataSourceConfig
    model: ModelConfig = Field(default_factory=ExpHydroModelConfig)
    task: ExperimentTaskConfig = Field(default_factory=CalibrationTaskConfig)
    metadata: Mapping[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def migrate_single_model_configuration(cls, value: Any) -> Any:
        """Read configurations created before model selection was introduced."""

        if isinstance(value, dict):
            value = dict(value)
            schema_version = str(value.get("schema_version", "1.0"))
            if isinstance(value.get("model"), dict):
                model = value["model"]
                if "model_id" not in model:
                    value["model"] = {
                        "model_id": "exp_hydro.mass_conservative.v1",
                        **model,
                    }
            if isinstance(value.get("task"), dict):
                task = dict(value["task"])
                task.setdefault("task_type", "calibration")
                task_type = task["task_type"]
                if task_type == "calibration" and isinstance(
                    task.get("period"), dict
                ):
                    period = dict(task["period"])
                    if "protocol" not in period:
                        if (
                            schema_version in {"1.18", "1.19", "1.20"}
                            and period.get("validation_end_index") is not None
                        ):
                            period["protocol"] = (
                                "calibration_validation_test"
                            )
                        elif (
                            schema_version in {"1.18", "1.19", "1.20"}
                            and period.get("test_end_index") is not None
                        ):
                            period["protocol"] = "calibration_test"
                            period["validation_end_index"] = None
                        else:
                            period = {
                                "protocol": "legacy_calibration_only",
                                "start_index": period.get("start_index", 0),
                                "calibration_end_index": period.get(
                                    "end_index",
                                    period.get("calibration_end_index"),
                                ),
                                "validation_end_index": None,
                                "test_end_index": None,
                                "warmup_steps": period.get("warmup_steps", 0),
                            }
                    task["period"] = period
                if task_type in {
                    "training",
                    "paml_training",
                    "discovery_training",
                }:
                    task.setdefault(
                        "objective",
                        {"objective_id": "root_mean_square_error"},
                    )
                if schema_version not in {
                    "1.16",
                    "1.17",
                    "1.18",
                    "1.19",
                    "1.20",
                    "1.21",
                }:
                    if task["task_type"] == "calibration" and "search" not in task:
                        task["search"] = {
                            "algorithm_id": "uniform_random_search.v1"
                        }
                value["task"] = task
            elif schema_version not in {
                "1.16",
                "1.17",
                "1.18",
                "1.19",
                "1.20",
                "1.21",
            }:
                value["task"] = {
                    "task_type": "calibration",
                    "search": {
                        "algorithm_id": "uniform_random_search.v1"
                    },
                }
        return value

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Experiment name cannot be blank.")
        return value.strip()

    @model_validator(mode="after")
    def require_calibration_target(self) -> "ExperimentConfigV1":
        if self.data.columns.streamflow is None:
            raise ValueError(
                "Calibration and training experiments require a mapped streamflow column."
            )
        neural_model = isinstance(
            self.model,
            (MlpModelConfig, LstmModelConfig, GruModelConfig, Cnn1dModelConfig),
        )
        if neural_model and not isinstance(self.task, NeuralTrainingTaskConfig):
            raise ValueError(
                "Standalone neural models require task_type='training'."
            )
        if isinstance(self.model, PamlModelConfig) and not isinstance(
            self.task, PamlTrainingTaskConfig
        ):
            raise ValueError(
                "PaML models require task_type='paml_training'."
            )
        s2s_model = (
            isinstance(
                self.model,
                (MlpModelConfig, LstmModelConfig, GruModelConfig, Cnn1dModelConfig),
            )
            and self.model.model_id
            in {
                "mlp.tensorflow_s2s.v2",
                "lstm.tensorflow_stacked_s2s.v3",
                "gru.tensorflow_stacked_s2s.v3",
                "cnn1d.tensorflow_s2s.v2",
            }
        ) or (
            isinstance(self.model, PamlModelConfig)
            and self.model.model_id == "paml.tensorflow_prnn_cnn_s2s.v1"
        )
        if s2s_model and getattr(self.task, "stride", None) != 1:
            raise ValueError(
                "Causal S2S uses every date exactly once and requires stride=1."
            )
        task_lead_steps = getattr(self.task, "lead_steps", None)
        if s2s_model and task_lead_steps != 0:
            raise ValueError(
                "Same-time causal S2S requires lead_steps=0."
            )
        if (
            isinstance(
                self.task, (NeuralTrainingTaskConfig, PamlTrainingTaskConfig)
            )
            and not s2s_model
            and task_lead_steps != 1
        ):
            raise ValueError(
                "Sequence-to-value training requires lead_steps=1."
            )
        if isinstance(self.model, DeepDiscoverModelConfig) and not isinstance(
            self.task, DiscoveryTrainingTaskConfig
        ):
            raise ValueError(
                "DeepDiscover models require task_type='discovery_training'."
            )
        if isinstance(self.model, (ExpHydroModelConfig, HbvModelConfig)) and not isinstance(
            self.task, CalibrationTaskConfig
        ):
            raise ValueError(
                "Conceptual models require task_type='calibration'."
            )
        return self


def configuration_hash(config: ExperimentConfigV1) -> str:
    """Return a stable SHA-256 digest of the normalized configuration payload."""

    payload = config.model_dump(mode="json", exclude_none=False)
    normalized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(normalized).hexdigest()


def experiment_json_schema() -> dict[str, Any]:
    """Return the JSON Schema published by the local service."""

    return ExperimentConfigV1.model_json_schema()


def load_experiment_config(path: str | Path) -> ExperimentConfigV1:
    """Load and validate a YAML or JSON experiment configuration."""

    source = Path(path)
    suffix = source.suffix.lower()
    if suffix not in {".yaml", ".yml", ".json"}:
        raise ValueError("Experiment configuration must use YAML or JSON.")
    with source.open("r", encoding="utf-8") as handle:
        if suffix == ".json":
            payload = json.load(handle)
        else:
            payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("Experiment configuration must contain a mapping at its root.")
    return ExperimentConfigV1.model_validate(payload)
