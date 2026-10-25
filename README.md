# HydroSoftware

HydroSoftware is a local visual environment for rainfall-runoff modelling. It
combines data preparation, model configuration, calibration or training,
simulation, inspection, comparison, numerical verification, and forecasting in
one application. The scientific engine is written in Python and the graphical
interface is written in React/TypeScript.

The application runs locally on the user's computer. The browser is used only
to display the graphical interface; model calculations and project files remain
on the local machine.

## Model families

HydroSoftware 1.0 includes:

- conceptual models: EXP-HYDRO and HBV;
- causal neural models: MLP, LSTM, GRU, and CNN-1D;
- physics-aware machine learning (PaML) with a selectable EXP-HYDRO or HBV
  P-RNN backbone and a causal convolutional component trained jointly;
- DeepDiscover for process discovery using the conservative NumPy
  full-sequence implementation supplied with the software.

Conceptual and process-discovery models expose hydrological states and fluxes
when these quantities are explicitly represented by their equations. Neural
models expose their architecture and trained weights without treating hidden
neural variables as hydrological states. PaML retains the physical-model
trajectory separately from the final hybrid discharge.

## Main capabilities

- CSV import with explicit variable and unit mapping;
- data-quality checks and chronological experiment definition;
- parameter calibration and neural-model training;
- calibration/training, validation, and independent test periods;
- streamflow evaluation with NSE, KGE, MSE, RMSE, NRMSE, MAE, and PBIAS;
- saved experiments and completed runs;
- model comparison and time-series visualization;
- visualization of hydrological states and fluxes when defined by the model;
- SHAP-based explainability for supported neural components;
- deterministic and meteorological-ensemble forecasting;
- forecast verification, including CRPS and deterministic diagnostics;
- replay, continuation, causality, non-negativity, and water-balance checks
  when applicable to the selected formulation.

## System requirements

HydroSoftware has two local components:

1. a **Python scientific backend**, which performs data processing, model
   fitting, simulation, and diagnostics;
2. a **React/TypeScript interface**, which is displayed in a web browser and
   communicates only with the local Python service.

The following software is required:

- **Python 3.10 to 3.13**;
- **Node.js 20.19 or newer**. Node.js 22 LTS is recommended. `npm` is installed
  automatically with Node.js;
- a modern web browser such as Chrome, Edge, or Firefox.

Using **Conda** (Anaconda or Miniconda) is recommended because it keeps the
Python dependencies isolated from other projects.

## Installation

### Option A: Conda installation (recommended)

Download or clone the HydroSoftware repository, then open a terminal in the
repository root (the directory containing `environment.yml`).

Create the environment:

```bash
conda env create -f environment.yml
```

Activate it:

```bash
conda activate hydrosoftware
```

This installs HydroSoftware and the Python dependencies required by the local
service and model implementations, including TensorFlow 2.21.0.

Install the interface dependencies once:

```bash
cd ui
npm ci
cd ..
```

### Option B: Python virtual environment

If Conda is not used, create and activate a Python virtual environment from the
repository root.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[service]"
```

macOS/Linux:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[service]"
```

Then install the interface dependencies:

```bash
cd ui
npm ci
cd ..
```

## Running HydroSoftware

HydroSoftware uses two local processes. Keep both terminals open while using
the application.

### 1. Start the Python backend

From the repository root, with the `hydrosoftware` Conda environment or the
virtual environment activated, run:

```bash
hydrosoftware-api --workspace .
```

The local API starts on:

```text
http://127.0.0.1:8787
```

### 2. Start the graphical interface

Open a second terminal, go to the `ui` directory, and run:

```bash
npm run dev
```

The interface starts on:

```text
http://127.0.0.1:5173
```

Open that address in a web browser. The lower-left status indicator in the
interface should report **Engine connected** when the Python backend is running.

To stop HydroSoftware, press `Ctrl+C` in each terminal.

## Production interface build

A production build of the interface can be created with:

```bash
cd ui
npm run build
```

The generated files are written to `ui/dist/`. Building the interface is not
required for normal local use with `npm run dev`.

## Demonstration data and examples

The bundled CAMELS-US example uses basin `01013500` (Fish River near Fort Kent,
Maine) and demonstrates EXP-HYDRO, LSTM, PaML, and DeepDiscover within a common
rainfall-runoff experiment.

Example configuration files are stored under `examples/`. The prepared CAMELS-
US daily series is stored under `examples/data/camels_us/`.

A small synthetic forcing ensemble is also included under
`examples/data/forcing_ensemble/` for exercising the ensemble-forecasting
workflow.

## Python use

The scientific backend can also be used directly from Python. After installation,
see `examples/run_example.py` and the YAML configurations under `examples/` for
examples of programmatic workflows.

## Tests

From the repository root, the Python test suite can be run with:

```bash
PYTHONPATH=src python -m unittest discover -s tests -p "test_*.py"
```

On Windows PowerShell, the equivalent command is:

```powershell
$env:PYTHONPATH="src"
python -m unittest discover -s tests -p "test_*.py"
```

The TensorFlow and DeepDiscover runtime checks can be run with:

```bash
PYTHONPATH=src python tools/verify_tensorflow_backend.py
PYTHONPATH=src python tools/verify_deepdiscover_runtime.py
```

The interface test suite and production build can be checked from `ui/` with:

```bash
npm run test
npm run build
```

## Project structure

```text
HydroSoftware/
├── src/hydrosoftware/        Python scientific backend
├── ui/                       React/TypeScript graphical interface
├── examples/                 Example configurations and datasets
├── tests/                    Python test suite
├── tools/                    Runtime verification utilities
├── docs/                     Technical documentation and release notes
├── environment.yml           Recommended Conda environment
├── pyproject.toml            Python package metadata and dependencies
├── CITATION.cff              Software citation metadata
└── LICENSE                   Apache License 2.0
```

## Reproducibility and model-specific diagnostics

Completed runs retain the experiment configuration, fitted parameters or model
weights, transformations, predictions, performance metrics, and model-specific
states and fluxes when available. Numerical checks are applied according to the
properties of each model. A check that is not scientifically meaningful for a
particular formulation is reported as not applicable rather than interpreted as
successful.

## Citation

Citation metadata are provided in `CITATION.cff`. HydroSoftware v1.0.0 is permanently archived on Zenodo (doi:10.5281/zenodo.23646680). Users should cite both the software release and the accompanying publication when applicable.
Users should cite both the software release and the accompanying publication
when applicable.

## Disclaimer

HydroSoftware is intended for research, education, and scientific modelling. Users are responsible for assessing the suitability of the software, model configurations, input data, and resulting simulations for their specific applications. The software is provided without warranty; see the Apache License 2.0 for the applicable terms and limitations of liability.

## License

HydroSoftware is released under the Apache License 2.0. External data products
and published model formulations retain their original attribution and licensing
requirements.
