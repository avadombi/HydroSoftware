"""Public model families exposed by HydroSoftware."""

from .deepdiscover import DeepDiscoverConfig, DiscoveryLossConfig, ProcessDynamics
from .deepdiscover_numpy import (
    DEEPDISCOVER_NUMPY_MANIFEST,
    DeepDiscoverArtifact,
    DeepDiscoverFitResult,
    deepdiscover_loss_and_gradients,
    fit_deepdiscover,
    simulate_deepdiscover_from_artifact,
)
from .paml import PAML_PRNN_CNN_MANIFEST


__all__ = [name for name in globals() if not name.startswith("_")]
