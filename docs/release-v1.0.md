# HydroSoftware 1.0 release notes

HydroSoftware 1.0 is the first public release of the software.

The release provides a local visual environment for rainfall-runoff modelling
with conceptual, neural, hybrid, and process-discovery approaches. The Python
scientific backend and the React/TypeScript interface share the same model and
experiment definitions, allowing simulations configured through the interface
to use the same scientific implementations as direct Python workflows.

## Model families

- Conceptual rainfall-runoff models: EXP-HYDRO and HBV.
- Neural models: MLP, LSTM, GRU, and causal CNN-1D.
- Hybrid modelling: PaML with an EXP-HYDRO or HBV differentiable P-RNN
  backbone coupled to a causal CNN.
- Process discovery: DeepDiscover using the conservative NumPy full-sequence
  implementation provided with HydroSoftware.

## Main capabilities

HydroSoftware 1.0 includes data import and quality control, chronological
experiment definition, calibration or training, simulation, saved-run reuse,
model comparison, numerical checks, visualization of model-specific states and
fluxes when available, neural-model explainability tools, deterministic and
ensemble forecasting, and forecast verification.

A CAMELS-US example for basin `01013500` is included under `examples/`.
Installation and launch instructions are provided in the repository README.
