import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import App, {
  CalibrationPanel,
  ForecastEventVerificationResults,
  ForecastPostprocessingResults,
  HindcastVerificationDialog,
  liveTaskTiming,
  ModelPanel,
  QualityPanel,
  ResultsPanel,
  TemporalValidationDialog,
} from "./App";
import { defaultForm } from "./config";
import type { DataQualityReport, ForecastEventVerification, ForecastPostprocessing, RunResult, TaskSnapshot } from "./types";

const readyReport: DataQualityReport = {
  source_path: "data/basin.csv",
  sha256: "a".repeat(64),
  row_count: 3,
  ready: true,
  error_count: 0,
  warning_count: 1,
  transformed_value_count: 1,
  variables: [
    {
      variable: "temperature",
      source_column: "t",
      source_unit: "degC",
      model_unit: "degC",
      row_count: 3,
      raw_valid_count: 2,
      raw_missing_count: 1,
      raw_invalid_count: 0,
      raw_negative_count: 0,
      processed_valid_count: 3,
      processed_missing_count: 0,
      transformed_count: 1,
      raw_minimum: 0,
      raw_maximum: 2,
      processed_minimum: 0,
      processed_maximum: 2,
      processed_mean: 1,
      processed_standard_deviation: 0.816,
      processed_q25: 0.5,
      processed_median: 1,
      processed_q75: 1.5,
    },
  ],
  timestamps: {
    assigned: true,
    source_column: "date",
    first: "2020-01-01",
    last: "2020-01-03",
    valid_count: 3,
    missing_count: 0,
    invalid_count: 0,
    duplicate_or_reversed_count: 0,
    irregular_step_count: 0,
    expected_step: "daily",
  },
  issues: [
    {
      code: "missing_streamflow_excluded",
      severity: "warning",
      summary: "One observation will not be scored.",
      explanation: "A missing target remains excluded.",
      suggested_action: "Confirm the gap.",
      variable: "streamflow",
      count: 1,
      csv_rows: [3],
    },
  ],
  timestamp_labels: ["2020-01-01", "2020-01-02", "2020-01-03"],
  observation_valid: [true, false, true],
  correlation_variables: ["temperature"],
  correlation_matrix: [[1]],
};

describe("live task timing", () => {
  it("advances elapsed time and reduces a known estimate between server events", () => {
    const updatedAt = "2026-09-12T12:00:00Z";
    const task: TaskSnapshot = {
      task_id: "run-progress",
      experiment_id: "experiment",
      state: "running",
      created_at: updatedAt,
      updated_at: updatedAt,
      event_count: 2,
      latest_event: {
        task_id: "run-progress",
        sequence: 1,
        state: "running",
        stage: "deepdiscover_training",
        message: "Training epoch 1 of 10.",
        completed_units: 1,
        total_units: 10,
        fraction: 0.1,
        elapsed_seconds: 10,
        estimated_remaining_seconds: 90,
        best_score: 0.2,
        severity: "info",
        error: null,
      },
      has_result: false,
      terminal: false,
      error: null,
    };
    expect(liveTaskTiming(task, Date.parse(updatedAt) + 5000)).toEqual({
      elapsed: 15,
      remaining: 85,
    });
  });
});

const completedResult: RunResult = {
  task_id: "completed-run",
  state: "completed",
  experiment_id: "experiment",
  configuration_hash: "hash",
  validation: {
    experiment_id: "experiment",
    configuration_hash: "hash",
    source_rows: 2,
    selected_rows: 2,
    warmup_steps: 0,
    scored_observations: 2,
    transformed_value_count: 0,
    data_quality_warning_count: 0,
    primary_evaluation_partition: "calibration",
    partition_scored_observations: { calibration: 2 },
    warnings: [],
  },
  calibration: {
    objective_name: "nash_sutcliffe_efficiency",
    objective_direction: "maximize",
    records: [
      { index: 1, parameters: { recession_factor: 0.02 }, score: 0.7, succeeded: true, generation: 0, population_member: 1 },
      { index: 2, parameters: { recession_factor: 0.05 }, score: 0.9, succeeded: true, generation: 1, population_member: 1 },
    ],
    best_parameters: { recession_factor: 0.05 },
    best_score: 0.9,
    warnings: [],
    algorithm_id: "differential_evolution.rand1bin.v1",
    termination_reason: "generation_budget_exhausted",
    generations_completed: 1,
  },
  best_simulation: {
    outputs: [[1], [2]],
    states: [[0, 10], [1, 11], [0, 12]],
    fluxes: [[1, 1, 0, 0, 0, 1, 0], [2, 2, 0, 0, 0, 2, 0]],
    output_names: ["streamflow"],
    state_names: ["snow_storage", "soil_storage"],
    flux_names: ["precipitation", "rain", "snowfall", "snowmelt", "actual_evapotranspiration", "subsurface_flow", "surface_flow"],
    diagnostics: { maximum_absolute_mass_balance_error: 0 },
  },
  timestamps: ["2020-01-01", "2020-01-02"],
  observations: [1.1, 1.9],
  forcings: { precipitation: [1, 2], temperature: [1, 2], potential_evapotranspiration: [0, 0] },
  evaluation_metrics: [
    { metric_id: "nash_sutcliffe_efficiency", symbol: "NSE", name: "Nash-Sutcliffe efficiency", value: 0.9, direction: "maximize", ideal_value: 1, unit: "dimensionless", definition: "NSE formula", unavailable_reason: null },
    { metric_id: "root_mean_square_error", symbol: "RMSE", name: "Root mean square error", value: 0.1, direction: "minimize", ideal_value: 0, unit: "mm/day", definition: "RMSE formula", unavailable_reason: null },
    { metric_id: "kling_gupta_efficiency_2009", symbol: "KGE", name: "Kling-Gupta efficiency (2009)", value: 0.88, direction: "maximize", ideal_value: 1, unit: "dimensionless", definition: "KGE formula", unavailable_reason: null },
    { metric_id: "percent_bias_moriasi", symbol: "PBIAS", name: "Percent bias", value: 2.5, direction: "target", ideal_value: 0, unit: "%", definition: "PBIAS formula", unavailable_reason: null },
    { metric_id: "range_normalized_root_mean_square_error", symbol: "NRMSE", name: "Range-normalized RMSE", value: 0.1, direction: "minimize", ideal_value: 0, unit: "dimensionless", definition: "NRMSE formula", unavailable_reason: null },
  ],
};

describe("application shell", () => {
  it("renders the first scientific workflow step without a browser runtime", () => {
    const markup = renderToStaticMarkup(<App />);

    expect(markup).toContain("HydroSoftware");
    expect(markup).toContain("Import and map basin data");
    expect(markup).toContain("Choose a CSV file");
    expect(markup).toContain("EXP-HYDRO");
  });

  it("renders a ready data-quality report", () => {
    const markup = renderToStaticMarkup(
      <QualityPanel
        form={defaultForm}
        report={readyReport}
        busy={false}
        onForm={() => undefined}
        onAnalyze={() => undefined}
        onNext={() => undefined}
      />,
    );

    expect(markup).toContain("Ready with documented warnings");
    expect(markup).toContain("temperature");
    expect(markup).toContain("2020-01-01");
    expect(markup).toContain("One observation will not be scored.");
    expect(markup).toContain("Exploratory data analysis");
    expect(markup).toContain("Processed-variable summary");
    expect(markup).toContain("Pearson correlation matrix");
    expect(markup).toContain("Correlation is a descriptive linear association");
    expect(markup).toContain("exploratory-analysis-card");
    expect((markup.match(/class="eda-panel"/g) ?? [])).toHaveLength(2);
  });

  it("offers fixed or calibrated treatment for active snow parameters", () => {
    const markup = renderToStaticMarkup(
      <ModelPanel form={defaultForm} onForm={() => undefined} onNext={() => undefined} />,
    );
    expect(markup).toContain("Parameter treatment");
    expect(markup).toContain("Fixed");
    expect(markup).toContain("Calibrate");
    expect(markup).toContain("Lower bound");
    expect(markup).toContain("parameter-treatment-fields");
  });

  it("renders the two HBV response zones without hiding the soil or snow states", () => {
    const markup = renderToStaticMarkup(
      <ModelPanel
        form={{ ...defaultForm, modelId: "hbv.mass_conservative_daily.v1" }}
        onForm={() => undefined}
        onNext={() => undefined}
      />,
    );
    expect(markup).toContain("Upper-zone storage");
    expect(markup).toContain("Lower-zone storage");
    expect(markup).toContain("bounded percolation");
    expect(markup).toContain("percolation-link");
    expect(markup).toContain('data-state="upper_zone_storage"');
    expect(markup).toContain('data-state="lower_zone_storage"');
    expect((markup.match(/data-state=/g) ?? [])).toHaveLength(4);
  });

  it("renders MAXBAS as a separate routed process with persistent memory", () => {
    const markup = renderToStaticMarkup(
      <ModelPanel
        form={{
          ...defaultForm,
          modelId: "hbv.mass_conservative_daily.v1",
          routingProcess: "routing.maxbas.triangular.v1",
          maxbas: 3,
        }}
        onForm={() => undefined}
        onNext={() => undefined}
      />,
    );
    expect(markup).toContain("Runoff routing");
    expect(markup).toContain("Triangular MAXBAS");
    expect(markup).toContain('data-process="routing.maxbas.triangular.v1"');
    expect(markup).toContain("Initial routing memory");
    expect((markup.match(/Pending slot/g) ?? [])).toHaveLength(6);
    expect(markup).toContain("MAXBAS-routed response");
  });

  it("renders the MLP as a distinct leakage-safe neural structure", () => {
    const markup = renderToStaticMarkup(
      <ModelPanel
        form={{ ...defaultForm, modelId: "mlp.tensorflow_s2v.v1" }}
        onForm={() => undefined}
        onNext={() => undefined}
      />,
    );
    expect(markup).toContain("Multilayer perceptron");
    expect(markup).toContain("mlp.tensorflow_s2v.v1");
    expect(markup).toContain("uses TensorFlow 2.21");
    expect(markup).toContain("No validation/test leakage");
    expect(markup).toContain("one-step-ahead sequence-to-value");
  });

  it("renders causal S2S as a distinct aligned neural layout", () => {
    const markup = renderToStaticMarkup(
      <ModelPanel
        form={{
          ...defaultForm,
          modelId: "lstm.tensorflow_stacked_s2s.v3",
          mlpLookback: 20,
          s2sWarmup: 7,
          recurrentLayers: "24, 12",
        }}
        onForm={() => undefined}
        onNext={() => undefined}
      />,
    );
    expect(markup).toContain("lstm.tensorflow_stacked_s2s.v3");
    expect(markup).toContain("causal S2S");
    expect(markup).toContain("Aligned causal sequence");
    expect(markup).toContain("Causal warm-up");
    expect(markup).toContain("Q at every causal t");
    expect(markup).toContain("20-day sequence");
  });

  it("renders the stacked LSTM gates and leakage-safe pathway", () => {
    const markup = renderToStaticMarkup(
      <ModelPanel
        form={{
          ...defaultForm,
          modelId: "lstm.tensorflow_stacked_s2v.v2",
          recurrentLayers: "24, 12",
        }}
        onForm={() => undefined}
        onNext={() => undefined}
      />,
    );
    expect(markup).toContain("Long short-term memory");
    expect(markup).toContain("lstm.tensorflow_stacked_s2v.v2");
    expect(markup).toContain("input, forget, candidate, and output gates");
    expect(markup).toContain("2 recurrent layers · 24 → 12 units");
    expect(markup).toContain("LSTM recurrent layer 2");
    expect(markup).toContain("12 units · four learned gates");
    expect(markup).toContain("Recurrent layers");
    expect(markup).toContain("No validation/test leakage");
  });

  it("renders the stacked reset-before GRU and its recurrent pathway", () => {
    const markup = renderToStaticMarkup(
      <ModelPanel
        form={{
          ...defaultForm,
          modelId: "gru.tensorflow_stacked_s2v.v2",
          recurrentLayers: "20, 10",
        }}
        onForm={() => undefined}
        onNext={() => undefined}
      />,
    );
    expect(markup).toContain("Gated recurrent unit");
    expect(markup).toContain("gru.tensorflow_stacked_s2v.v2");
    expect(markup).toContain("update and reset gates");
    expect(markup).toContain("2 recurrent layers · 20 → 10 units");
    expect(markup).toContain("GRU recurrent layer 2");
    expect(markup).toContain("10 units · update and reset gates");
    expect(markup).toContain("Fixed reset-before GRU equations");
  });

  it("renders a configurable stacked causal CNN-1D pathway", () => {
    const markup = renderToStaticMarkup(
      <ModelPanel
        form={{
          ...defaultForm,
          modelId: "cnn1d.tensorflow_s2v.v1",
          mlpLookback: 20,
          cnnLayers: [
            { filters: 24, kernelSize: 5, activation: "relu" },
            { filters: 12, kernelSize: 3, activation: "tanh" },
          ],
        }}
        onForm={() => undefined}
        onNext={() => undefined}
      />,
    );
    expect(markup).toContain("One-dimensional convolutional network");
    expect(markup).toContain("cnn1d.tensorflow_s2v.v1");
    expect(markup).toContain("Add convolutional layer");
    expect(markup).toContain("Receptive field: 7 days");
    expect(markup).toContain("Causal Conv1D layer 1");
    expect(markup).toContain("24 filters · kernel 5 · relu");
    expect(markup).toContain("Final time position");
  });

  it("lets DD estimate selected snow parameters without exposing engine state", () => {
    const markup = renderToStaticMarkup(
      <ModelPanel
        form={{
          ...defaultForm,
          modelId: "deepdiscover.numpy_instantaneous_s2s.v1",
          snowCalibration: {
            ...defaultForm.snowCalibration,
            degreeDayFactor: {
              mode: "calibrate",
              lowerBound: 0.5,
              upperBound: 6,
            },
          },
        }}
        onForm={() => undefined}
        onNext={() => undefined}
      />,
    );
    expect(markup).toContain("Snow module and parameter treatment");
    expect(markup).toContain("1 estimated");
    expect(markup).toContain("bounded alternating coordinate calibration");
    expect(markup).toContain("joint checkpoint is chosen on validation only");
    expect(markup).not.toContain("Initial routing storage");
    expect(markup).not.toContain("Initial memory");
    expect(markup).not.toContain("Gain bounds");
  });

  it("configures chronological neural partitions and validation checkpointing", () => {
    const markup = renderToStaticMarkup(
      <CalibrationPanel
        form={{
          ...defaultForm,
          modelId: "mlp.tensorflow_s2v.v1",
          mlpLookback: 10,
          trainEndIndex: 60,
          validationEndIndex: 80,
          testEndIndex: 100,
          objective: "kling_gupta_efficiency_2009",
        }}
        preview={{ relative_path: "data/neural.csv", columns: ["p", "t", "pet", "q"], rows: [], row_count: 100, file_size_bytes: 1, sha256: "a".repeat(64) }}
        quality={readyReport}
        onForm={() => undefined}
        onNext={() => undefined}
      />,
    );
    expect(markup).toContain("Separate training, validation, and test");
    expect(markup).toContain("Validation KGE");
    expect(markup).toContain("full training partition as one objective batch");
    expect(markup).toContain("Initialization robustness");
    expect(markup).toContain("Initialization ensemble");
    expect(markup).toContain("not a predictive interval");
    expect(markup).toContain("test set is evaluated once");
    expect(markup).toContain("Training setup is internally consistent");
  });

  it("offers bounded differential evolution with a visible simulation budget", () => {
    const markup = renderToStaticMarkup(
      <CalibrationPanel
        form={{
          ...defaultForm,
          endIndex: 100,
          warmupSteps: 5,
          dePopulationSize: 30,
          deGenerations: 10,
        }}
        preview={{ relative_path: "data/conceptual.csv", columns: ["p", "t", "pet", "q"], rows: [], row_count: 100, file_size_bytes: 1, sha256: "a".repeat(64) }}
        quality={readyReport}
        onForm={() => undefined}
        onNext={() => undefined}
      />,
    );
    expect(markup).toContain("Differential evolution");
    expect(markup).toContain("DE/rand/1/bin");
    expect(markup).toContain("Advanced DE controls");
    expect(markup).toContain("330");
    expect(markup).toContain("does not claim proof of a global optimum");
  });

  it("separates conceptual calibration, validation, and test with six objectives", () => {
    const markup = renderToStaticMarkup(
      <CalibrationPanel
        form={{
          ...defaultForm,
          modelId: "exp_hydro.mass_conservative.v1",
          conceptualProtocol: "calibration_validation_test",
          endIndex: 60,
          validationEndIndex: 80,
          testEndIndex: 100,
          warmupSteps: 5,
        }}
        preview={{ relative_path: "data/conceptual.csv", columns: ["p", "t", "pet", "q"], rows: [], row_count: 100, file_size_bytes: 1, sha256: "a".repeat(64) }}
        quality={readyReport}
        onForm={() => undefined}
        onNext={() => undefined}
      />,
    );
    expect(markup).toContain("Calibration · validation · test");
    expect(markup).toContain("60 calibration");
    expect(markup).toContain("20 validation");
    expect(markup).toContain("20 test");
    expect(markup).toContain("states remain continuous across every boundary");
    expect(markup).toContain("KGE");
    expect(markup).toContain("MSE");
    expect(markup).toContain("NRMSE");
    expect(markup).toContain("MAE");
    expect(markup).toContain("Primary reported partition");
    expect(markup).toContain("Held-out test");
  });

  it("configures full-sequence DeepDiscover training without fake mini-batches", () => {
    const markup = renderToStaticMarkup(
      <CalibrationPanel
        form={{
          ...defaultForm,
          modelId: "deepdiscover.numpy_instantaneous_s2s.v1",
          trainEndIndex: 60,
          validationEndIndex: 80,
          testEndIndex: 100,
          warmupSteps: 5,
        }}
        preview={{ relative_path: "data/discovery.csv", columns: ["p", "t", "pet", "q"], rows: [], row_count: 100, file_size_bytes: 1, sha256: "a".repeat(64) }}
        quality={readyReport}
        onForm={() => undefined}
        onNext={() => undefined}
      />,
    );
    expect(markup).toContain("Constrained process discovery");
    expect(markup).toContain("Complete chronological sequence");
    expect(markup).toContain("Training warm-up");
    expect(markup).toContain("active inputs, raw and corrected processes");
    expect(markup).not.toContain("Batch size");
    expect(markup).not.toContain("Window stride");
  });

  it("defers the heavy process explorer on the initial results render", () => {
    const markup = renderToStaticMarkup(
      <ResultsPanel result={completedResult} busy={false} onNewRun={() => undefined} onRerun={() => undefined} />,
    );
    expect(markup).toContain("Open process explorer");
    expect(markup).not.toContain("Synchronized process view");
    expect(markup).not.toContain("Internal storages");
  });

  it("provides one precise temporal viewport shared by the result plots", () => {
    const markup = renderToStaticMarkup(
      <ResultsPanel result={completedResult} busy={false} onNewRun={() => undefined} onRerun={() => undefined} />,
    );
    expect(markup).toContain("Temporal viewport");
    expect(markup).toContain("First visible result row");
    expect(markup).toContain("Last visible result row");
    expect(markup).toContain("Reset zoom");
    expect(markup).toContain("same window is used by every result plot");
  });

  it("separates the calibration objective from selectable result metrics", () => {
    const markup = renderToStaticMarkup(
      <ResultsPanel result={completedResult} busy={false} onNewRun={() => undefined} onRerun={() => undefined} />,
    );
    expect(markup).toContain("Partition diagnostics");
    expect(markup).toContain("Displayed result metrics");
    expect(markup).toContain("KGE");
    expect(markup).toContain("PBIAS");
    expect(markup).toContain("NRMSE");
    expect(markup).toContain("Calibration objective");
    expect(markup).toContain("5 selected");
    expect(markup).toContain('aria-haspopup="true"');
    expect(markup).toContain("Compare runs");
    expect(markup).toContain("Temporal validation");
    expect(markup).toContain("Sampling uncertainty");
    expect(markup).toContain("Scientific verification");
    expect(markup).toContain("Verify this run");
    expect(markup).toContain("Integrity is not predictive skill");
    expect(markup).toContain("Calibration convergence");
    expect(markup).toContain("DE/rand/1/bin");
    expect(markup).toContain("Compute intervals");
    expect(markup).toContain("Temporal blocks, not independent days");
    expect((markup.match(/type="checkbox" checked=""/g) ?? [])).toHaveLength(5);
  });

  it("configures non-overlapping rolling-origin validation from saved setups", () => {
    const markup = renderToStaticMarkup(
      <TemporalValidationDialog
        projects={[
          {
            experiment_id: "experiment",
            name: "Reference experiment",
            description: "Reference model",
            configuration_hash: "hash",
            created_at: "2026-07-29T00:00:00Z",
            updated_at: "2026-07-29T00:00:00Z",
            latest_run_id: "completed-run",
            run_count: 1,
          },
        ]}
        currentExperimentId="experiment"
        onClose={() => undefined}
        onError={() => undefined}
      />,
    );

    expect(markup).toContain("Rolling-origin validation");
    expect(markup).toContain("Expanding history");
    expect(markup).toContain("Fixed sliding history");
    expect(markup).toContain("Test blocks cannot overlap");
    expect(markup).toContain("Fixed across folds");
    expect(markup).toContain("This measures temporal robustness, not predictive uncertainty");
    expect(markup).toContain("3 model-fold executions");
    expect(markup).toContain("Recent temporal validations");
  });

  it("offers outputs, intermediate processes, states, and explicit decimal precision", () => {
    const markup = renderToStaticMarkup(
      <ResultsPanel result={completedResult} busy={false} onNewRun={() => undefined} onRerun={() => undefined} />,
    );
    expect(markup).toContain("Displayed temporal variable");
    expect(markup).toContain("Intermediate fluxes");
    expect(markup).toContain("Actual evapotranspiration");
    expect(markup).toContain("Internal states");
    expect(markup).toContain("Displayed decimal places");
    expect(markup).toContain("Use new forcings");
  });

  it("does not invent observations or metrics for a forcing-only forecast", () => {
    const forecast: RunResult = {
      ...completedResult,
      task_id: "forecast-1",
      run_kind: "forecast",
      source_task_id: completedResult.task_id,
      observations: null,
      evaluation_metrics: [],
      validation: { ...completedResult.validation, scored_observations: 0, warmup_steps: 0 },
    };
    const markup = renderToStaticMarkup(
      <ResultsPanel result={forecast} busy={false} onNewRun={() => undefined} onRerun={() => undefined} />,
    );
    expect(markup).toContain("No observed streamflow was supplied");
    expect(markup).toContain("New forecast");
    expect(markup).not.toContain("Observed vs simulated streamflow");
    expect(markup).not.toContain("Run unchanged");
    expect(markup).not.toContain("Sampling uncertainty");
  });

  it("renders forcing-ensemble scope and lead-time probabilistic scores", () => {
    const forecast: RunResult = {
      ...completedResult,
      task_id: "forecast-ensemble-1",
      run_kind: "forecast",
      source_task_id: completedResult.task_id,
      best_simulation: {
        ...completedResult.best_simulation,
        diagnostics: {
          engine_id: "meteorological_forcing_ensemble.v1",
          forcing_ensemble_lower: [0.8, 1.7],
          forcing_ensemble_upper: [1.2, 2.3],
        },
      },
      forecast_ensemble: {
        method_id: "meteorological_forcing_ensemble.v1",
        issue_id: "hindcast-2030-01-01",
        issue_time: "2030-01-01",
        first_valid_time: "2030-01-02",
        spinup_steps: 0,
        scope: "meteorological_forcing_scenarios",
        member_count: 3,
        member_ids: ["member-01", "member-02", "member-03"],
        aggregation: "arithmetic_mean",
        lower_quantile: 0.1,
        upper_quantile: 0.9,
        members: [],
        lead_time_diagnostics: [
          {
            lead_step: 1,
            timestamp: "2020-01-01",
            valid_members: 3,
            observation: 1.1,
            ensemble_mean: 1,
            ensemble_median: 1,
            lower: 0.8,
            upper: 1.2,
            spread: 0.16,
            signed_error: -0.1,
            absolute_error: 0.1,
            crps: 0.08,
          },
        ],
        aggregate_scores: {
          scored_leads: 1,
          mean_crps: 0.08,
          mean_error_bias: -0.1,
          mean_absolute_error: 0.1,
          root_mean_square_error: 0.1,
          mean_ensemble_spread: 0.16,
        },
        warnings: [],
      },
    };
    const markup = renderToStaticMarkup(
      <ResultsPanel result={forecast} busy={false} onNewRun={() => undefined} onRerun={() => undefined} />,
    );
    expect(markup).toContain("Forcing-ensemble diagnostics");
    expect(markup).toContain("Mean CRPS");
    expect(markup).toContain("Mean signed error");
    expect(markup).toContain("forecast mean − observed");
    expect(markup).toContain("Forcing spread");
    expect(markup).toContain("Continuation uses the separately archived state and context");
    expect(markup).toContain("Verify hindcasts");
    expect(markup).toContain("Continue ensemble");
    expect(markup).not.toContain("New forecast");
  });

  it("opens the repeated-origin ensemble verification workspace", () => {
    const forecast: RunResult = {
      ...completedResult,
      task_id: "forecast-ensemble-current",
      source_task_id: "source-run",
      model_id: "exp_hydro.mass_conservative.v1",
      run_kind: "forecast",
      forecast_ensemble: {
        method_id: "meteorological_forcing_ensemble.v1",
        issue_id: "hindcast-2030-01-01",
        issue_time: "2030-01-01",
        first_valid_time: "2030-01-02",
        spinup_steps: 0,
        scope: "meteorological_forcing_scenarios",
        member_count: 2,
        member_ids: ["member-01", "member-02"],
        aggregation: "arithmetic_mean",
        lower_quantile: 0.1,
        upper_quantile: 0.9,
        members: [],
        lead_time_diagnostics: [],
        aggregate_scores: {
          scored_leads: 3,
          mean_crps: 0.1,
          mean_error_bias: 0,
          mean_absolute_error: 0.1,
          root_mean_square_error: 0.1,
          mean_ensemble_spread: 0.2,
        },
        warnings: [],
      },
    };
    const markup = renderToStaticMarkup(
      <HindcastVerificationDialog currentResult={forecast} onClose={() => undefined} onError={() => undefined} />,
    );
    expect(markup).toContain("Ensemble hindcast verification");
    expect(markup).toContain("Compatible forecast cases");
    expect(markup).toContain("Minimum cases per lead");
    expect(markup).toContain("Chronological post-processing");
    expect(markup).toContain("Oldest origins used for fitting");
    expect(markup).toContain("Threshold-event decisions");
    expect(markup).toContain("Evaluate event decisions");
    expect(markup).toContain("Post-process chronologically");
    expect(markup).toContain("Loading archived ensembles");
  });

  it("renders raw and independent adjusted ensemble scores together", () => {
    const diagnostics = {
      scored_cases: 4,
      mean_crps: 0.4,
      mean_error_bias: 0.2,
      mean_absolute_error: 0.3,
      root_mean_square_error: 0.35,
      root_mean_ensemble_variance: 0.2,
      spread_skill_ratio: 0.57,
      interval_coverage: 0.75,
      nominal_interval_coverage: 0.8,
      coverage_error: -0.05,
      rank_counts: [1, 2, 1],
      rank_relative_frequencies: [0.25, 0.5, 0.25],
    };
    const result: ForecastPostprocessing = {
      task_id: "forecast-postprocessing-test",
      state: "completed",
      method_id: "affine_member_recalibration.v1",
      source_task_id: "source-run",
      experiment_id: "experiment",
      configuration_hash: "hash",
      model_id: "exp_hydro.mass_conservative.v1",
      member_count: 2,
      horizon: 1,
      lower_quantile: 0.1,
      upper_quantile: 0.9,
      seed: 0,
      split: {
        strategy: "chronological_oldest_fit_latest_evaluate",
        training_case_count: 6,
        evaluation_case_count: 4,
        training_task_ids: [],
        evaluation_task_ids: [],
        last_training_origin: "2030-01-06T00:00:00+00:00",
        first_evaluation_origin: "2030-01-07T00:00:00+00:00",
      },
      parameter_bounds: {
        mean_slope: [0, 3],
        spread_multiplier: [0, 5],
      },
      minimum_training_cases_per_lead: 5,
      parameters: [{
        lead_step: 1,
        intercept: -0.2,
        mean_slope: 1,
        spread_multiplier: 1.4,
        training_case_count: 6,
        training_raw_mean_crps: 0.5,
        training_adjusted_mean_crps: 0.3,
      }],
      evaluation_cases: [],
      lead_time_diagnostics: [{
        lead_step: 1,
        available_cases: 4,
        reported: true,
        raw: diagnostics,
        adjusted: { ...diagnostics, mean_crps: 0.3, mean_error_bias: 0.05 },
        crps_skill_score: 0.25,
      }],
      aggregate: {
        raw: diagnostics,
        adjusted: { ...diagnostics, mean_crps: 0.3, mean_error_bias: 0.05 },
        crps_skill_score: 0.25,
      },
      warnings: [{
        code: "small_archive",
        message: "Small archive warning.",
      }],
    };
    const markup = renderToStaticMarkup(
      <ForecastPostprocessingResults result={result} onBack={() => undefined} />,
    );
    expect(markup).toContain("Independent post-processing evaluation");
    expect(markup).toContain("Raw mean CRPS");
    expect(markup).toContain("Adjusted mean CRPS");
    expect(markup).toContain("CRPSS");
    expect(markup).toContain("Fitted lead-wise coefficients");
    expect(markup).toContain("Small archive warning");
  });

  it("renders independent threshold-event accuracy, reliability, and value", () => {
    const diagnostics = {
      scored_cases: 20,
      event_count: 8,
      nonevent_count: 12,
      event_frequency: 0.4,
      mean_forecast_probability: 0.45,
      brier_score: 0.18,
      climatology_brier_score: 0.24,
      brier_skill_score: 0.25,
      roc_auc: 0.8,
      reliability_bins: Array.from({ length: 5 }, (_, index) => ({
        bin: index + 1,
        lower_probability: index / 5,
        upper_probability: (index + 1) / 5,
        count: index === 0 ? 8 : 3,
        mean_forecast_probability: (index + 0.5) / 5,
        observed_frequency: index / 5,
        wilson_lower_95: 0,
        wilson_upper_95: 0.5,
      })),
      roc_curve: [
        { probability_threshold: 1.01, false_positive_rate: 0, true_positive_rate: 0 },
        { probability_threshold: 0.5, false_positive_rate: 0.2, true_positive_rate: 0.8 },
        { probability_threshold: -0.01, false_positive_rate: 1, true_positive_rate: 1 },
      ],
      cost_loss_curve: Array.from({ length: 19 }, (_, index) => ({
        cost_loss_ratio: (index + 1) * 0.05,
        forecast_expense: 0.1,
        climatology_expense: 0.2,
        perfect_expense: 0.05,
        relative_economic_value: 0.5,
      })),
      maximum_relative_economic_value: 0.5,
      cost_loss_ratio_at_maximum_value: 0.3,
    };
    const result: ForecastEventVerification = {
      task_id: "event-verification-test",
      state: "completed",
      method_id: "threshold_event_decision_verification.v1",
      source_task_id: "source-run",
      experiment_id: "experiment",
      configuration_hash: "hash",
      model_id: "exp_hydro.mass_conservative.v1",
      member_count: 3,
      horizon: 1,
      threshold: {
        mode: "training_quantile",
        value: 2.5,
        unit: "mm/day",
        event_definition: "observed_or_member_streamflow > threshold",
        training_quantile: 0.9,
        training_observation_count: 100,
      },
      split: {
        strategy: "chronological_oldest_fit_latest_evaluate",
        training_case_count: 20,
        evaluation_case_count: 10,
        training_task_ids: [],
        evaluation_task_ids: [],
        last_training_origin: "2030-01-20T00:00:00+00:00",
        first_evaluation_origin: "2030-01-21T00:00:00+00:00",
      },
      climatology: {
        training_event_probability: 0.1,
        training_event_count: 10,
        training_nonevent_count: 90,
      },
      probability_bin_count: 5,
      minimum_evaluation_cases_per_lead: 5,
      lead_time_diagnostics: [{
        lead_step: 1,
        available_cases: 20,
        reported: true,
        raw: diagnostics,
        adjusted: { ...diagnostics, brier_score: 0.12, brier_skill_score: 0.5, roc_auc: 0.9 },
        adjusted_brier_skill_vs_raw: 1 - 0.12 / 0.18,
      }],
      aggregate: {
        raw: diagnostics,
        adjusted: { ...diagnostics, brier_score: 0.12, brier_skill_score: 0.5, roc_auc: 0.9 },
        adjusted_brier_skill_vs_raw: 1 - 0.12 / 0.18,
      },
      postprocessing_method_id: "affine_member_recalibration.v1",
      warnings: [{ code: "scope", message: "Interpret decision value conditionally." }],
    };
    const markup = renderToStaticMarkup(
      <ForecastEventVerificationResults result={result} onBack={() => undefined} />,
    );
    expect(markup).toContain("Independent threshold-event evaluation");
    expect(markup).toContain("Raw Brier score");
    expect(markup).toContain("Adjusted Brier score");
    expect(markup).toContain("Reliability diagram");
    expect(markup).toContain("Relative economic value");
    expect(markup).toContain("Interpret decision value conditionally.");
  });

  it("renders trained MLP results without inventing conceptual states or fluxes", () => {
    const neural: RunResult = {
      ...completedResult,
      task_id: "mlp-run",
      run_kind: "training",
      calibration: null,
      training: {
        engine_id: "tensorflow.keras.2.21",
        artifact: {},
        history: [{ epoch: 1, training_rmse: 0.4, validation_rmse: 0.5 }],
        best_epoch: 1,
        epochs_completed: 1,
        stopped_early: false,
        selection_metric: "validation_root_mean_square_error",
        best_validation_score: 0.5,
        feature_roles: ["precipitation", "temperature"],
        lookback: 10,
        partition_metrics: {},
      },
      best_simulation: {
        outputs: [[null], [1.2]],
        states: [[], [], []],
        fluxes: [[], []],
        output_names: ["streamflow"],
        state_names: [],
        flux_names: [],
        diagnostics: { engine_id: "tensorflow.keras.2.21", lookback: 10 },
      },
    };
    const markup = renderToStaticMarkup(
      <ResultsPanel result={neural} busy={false} onNewRun={() => undefined} onRerun={() => undefined} />,
    );
    expect(markup).toContain("Completed training");
    expect(markup).toContain("Saved neural model");
    expect(markup).toContain("Epoch 1");
    expect(markup).toContain("SHAP explainability");
    expect(markup).toContain("Compute SHAP");
    expect(markup).toContain("Model explanation, not causal attribution");
    expect(markup).not.toContain("Process explorer");
    expect(markup).not.toContain("Calibrated parameters");
  });

  it("renders initialization-member diagnostics and the empirical trajectory band", () => {
    const neural: RunResult = {
      ...completedResult,
      task_id: "mlp-ensemble-run",
      run_kind: "training",
      calibration: null,
      training: {
        engine_id: "neural_initialization_ensemble.v1",
        artifact: {},
        history: [{ epoch: 1, training_rmse: 0.4, validation_rmse: 0.5 }],
        best_epoch: 1,
        epochs_completed: 3,
        stopped_early: false,
        selection_metric: "member_validation_rmse_then_arithmetic_mean",
        best_validation_score: 0.45,
        feature_roles: ["precipitation"],
        lookback: 2,
        partition_metrics: {},
        initialization_ensemble: {
          method_id: "neural_initialization_ensemble.v1",
          member_count_requested: 3,
          member_count_completed: 3,
          member_seeds: [11, 22, 33],
          aggregation: "arithmetic_mean",
          lower_quantile: 0.1,
          upper_quantile: 0.9,
          representative_member: 2,
          scope: "neural_weight_initialization",
          members: [11, 22, 33].map((seed, index) => ({
            member: index + 1,
            seed,
            best_epoch: index + 1,
            epochs_completed: 1,
            stopped_early: false,
            cancelled: false,
            best_validation_rmse: 0.4 + index * 0.05,
            partition_metrics: {},
          })),
        },
      },
      best_simulation: {
        outputs: [[1.1], [2.1]],
        states: [[], [], []],
        fluxes: [[], []],
        output_names: ["streamflow"],
        state_names: [],
        flux_names: [],
        diagnostics: {
          engine_id: "neural_initialization_ensemble.v1",
          initialization_ensemble_lower: [1, 2],
          initialization_ensemble_upper: [1.2, 2.2],
        },
      },
    };
    const markup = renderToStaticMarkup(
      <ResultsPanel result={neural} busy={false} onNewRun={() => undefined} onRerun={() => undefined} />,
    );
    expect(markup).toContain("Neural ensemble diagnostics");
    expect(markup).toContain("Member 01");
    expect(markup).toContain("Initialization spread");
    expect(markup).toContain("Ensemble mean");
    expect(markup).toContain("not a calibrated probability interval");
  });

  it("renders DeepDiscover processes, switches, and analysis trajectories", () => {
    const discovery: RunResult = {
      ...completedResult,
      task_id: "discovery-run",
      run_kind: "training",
      model_id: "deepdiscover.numpy_instantaneous_s2s.v1",
      calibration: null,
      training: {
        engine_id: "tensorflow.keras.2.21",
        model_id: "deepdiscover.numpy_instantaneous_s2s.v1",
        artifact: {},
        history: [{
          epoch: 1,
          training_rmse: 0.3,
          validation_rmse: 0.4,
          data_loss: 0.2,
          decorrelation_loss: 0.03,
          mass_balance_loss: 0,
          smoothness_loss: 0.01,
          binary_loss: 0.04,
          sparsity_loss: 0.5,
        }],
        best_epoch: 1,
        epochs_completed: 1,
        stopped_early: false,
        selection_metric: "validation_root_mean_square_error",
        best_validation_score: 0.4,
        prediction_layout: "sequence_to_sequence",
        process_dynamics: "instantaneous",
        process_count: 2,
        hidden_units: 8,
        input_switch_probabilities: [0.9, 0.7, 0.8],
        process_switch_probabilities: [0.8, 0.6],
        partition_metrics: {},
      },
      best_simulation: {
        outputs: [[1], [2]],
        states: [[0, 18], [0, 17], [0, 16]],
        fluxes: [[1, 1, 0, 0, 0.1, 0.6, 0.4], [2, 2, 0, 0, 0.2, 1.2, 0.8]],
        output_names: ["streamflow"],
        state_names: ["snow_storage", "discovered_storage"],
        flux_names: ["precipitation", "rain", "snowfall", "snowmelt", "actual_evapotranspiration", "discovered_process_1", "discovered_process_2"],
        diagnostics: {
          engine_id: "tensorflow.keras.2.21",
          maximum_absolute_mass_balance_error: 1e-15,
          analysis__active_rain: [0.2, 0.4],
          analysis__raw_process_1: [0.6, 0.7],
          analysis__balance_residual: [0, 0],
        },
      },
    };
    const markup = renderToStaticMarkup(
      <ResultsPanel result={discovery} busy={false} onNewRun={() => undefined} onRerun={() => undefined} />,
    );
    expect(markup).toContain("Saved process-discovery model");
    expect(markup).toContain("Discovery analysis");
    expect(markup).toContain("Active rain");
    expect(markup).toContain("Raw process 1");
    expect(markup).toContain("Input switches");
    expect(markup).toContain("Process switches");
    expect(markup).toContain("Selected epoch · Decorrelation loss");
    expect(markup).toContain("Selected epoch · Mass-balance loss");
    expect(markup).toContain("Mapping weights, input and process switches");
    expect(markup).not.toContain("SHAP explainability");
  });


  it("renders joint PaML with a selectable hydrological backbone", () => {
    const markup = renderToStaticMarkup(
      <ModelPanel
        form={{ ...defaultForm, modelId: "paml.tensorflow_prnn_cnn_s2s.v1", pamlPhysicalModelId: "hbv.mass_conservative_daily.v1", cnnLayers: [{ filters: 12, kernelSize: 7, activation: "relu" }] }}
        onForm={() => undefined}
        onNext={() => undefined}
      />,
    );
    expect(markup).toContain("P-RNN + causal CNN");
    expect(markup).toContain("HBV");
    expect(markup).toContain("optimized jointly");
    expect(markup).toContain("12 filters");
  });
});
