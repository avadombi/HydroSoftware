"""Configuration metadata for physics-aware recurrent-neural rainfall-runoff models."""

from __future__ import annotations

from hydrosoftware.core.specifications import (
    ModelCapabilities,
    ModelFamily,
    ModelManifest,
    PredictionLayout,
    VariableSpec,
)


PAML_PRNN_CNN_MANIFEST = ModelManifest(
    model_id="paml.tensorflow_prnn_cnn_s2s.v1",
    version="1.0.0",
    display_name="PaML P-RNN + causal CNN",
    family=ModelFamily.PAML,
    capabilities=ModelCapabilities(
        layouts=(PredictionLayout.SEQUENCE_TO_SEQUENCE,),
        supports_fit=True,
        supports_calibration=False,
        differentiable=True,
        supports_initial_state=True,
        exposes_states=True,
        exposes_fluxes=True,
    ),
    inputs=(
        VariableSpec("precipitation", "forcing", "mm/day", lower_bound=0.0),
        VariableSpec("temperature", "forcing", "degC"),
        VariableSpec(
            "potential_evapotranspiration",
            "forcing",
            "mm/day",
            lower_bound=0.0,
        ),
    ),
    outputs=(
        VariableSpec("streamflow", "hybrid_output", "mm/day"),
        VariableSpec("physical_streamflow", "physical_output", "mm/day"),
    ),
    states=(
        VariableSpec(
            "physical_state__*",
            "dynamic_physical_state",
            "model-defined",
            description="Hydrological states exposed by the selected P-RNN backbone.",
        ),
    ),
    fluxes=(
        VariableSpec(
            "physical_flux__*",
            "dynamic_physical_flux",
            "model-defined",
            description="Hydrological fluxes exposed by the selected P-RNN backbone.",
        ),
    ),
    references=(
        "Jiang, Zheng and Solomatine (2020), Geophysical Research Letters 47, e2020GL088229.",
    ),
    metadata={
        "training_strategy": "joint_end_to_end",
        "physical_backbones": "EXP-HYDRO,HBV",
        "neural_corrector": "causal two-layer CNN",
        "architecture": "P-RNN output concatenated with forcing, scaled, then passed to causal CNN",
    },
)
