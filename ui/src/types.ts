export type PanelId = "data" | "quality" | "model" | "calibration" | "run" | "results";

export interface StructuredError {
  code: string;
  summary?: string;
  message?: string;
  explanation?: string;
  suggested_action?: string;
  technical_details?: string | null;
}

export interface CsvPreview {
  relative_path: string;
  columns: string[];
  rows: Record<string, string>[];
  row_count: number;
  file_size_bytes: number;
  sha256: string;
}

export interface ColumnMapping {
  timestamp: string;
  precipitation: string;
  temperature: string;
  potential_evapotranspiration: string;
  streamflow: string;
}

export interface AdditionalPredictorForm {
  name: string;
  column: string;
  unit: string;
  missing: "error" | "linear_interpolation";
}

export interface LaggedFeatureForm {
  source: string;
  lagSteps: number;
}

export type SnowModuleId =
  | "snow.none.v1"
  | "snow.degree_day.hard.v1"
  | "snow.degree_day.smooth.v1";

export type ConceptualModelId =
  | "exp_hydro.mass_conservative.v1"
  | "hbv.mass_conservative_daily.v1";

export type NeuralModelId =
  | "mlp.tensorflow_s2v.v1"
  | "mlp.tensorflow_s2s.v2"
  | "lstm.tensorflow_s2v.v1"
  | "lstm.tensorflow_stacked_s2v.v2"
  | "lstm.tensorflow_stacked_s2s.v3"
  | "gru.tensorflow_s2v.v1"
  | "gru.tensorflow_stacked_s2v.v2"
  | "gru.tensorflow_stacked_s2s.v3"
  | "cnn1d.tensorflow_s2v.v1"
  | "cnn1d.tensorflow_s2s.v2";
export type PamlModelId = "paml.tensorflow_prnn_cnn_s2s.v1";
export type DiscoveryModelId = "deepdiscover.numpy_instantaneous_s2s.v1";
export type ModelId =
  | ConceptualModelId
  | NeuralModelId
  | PamlModelId
  | DiscoveryModelId;
export type MlpFeatureRole = string;
export type OptimizationObjectiveId =
  | "nash_sutcliffe_efficiency"
  | "kling_gupta_efficiency_2009"
  | "mean_square_error"
  | "root_mean_square_error"
  | "range_normalized_root_mean_square_error"
  | "mean_absolute_error";

export interface Cnn1dLayerForm {
  filters: number;
  kernelSize: number;
  activation: "relu" | "tanh";
}

export type RoutingProcessId =
  | "routing.none.v1"
  | "routing.maxbas.triangular.v1";

export interface ParameterTreatmentForm {
  mode: "fixed" | "calibrate";
  lowerBound: number;
  upperBound: number;
}

export interface SnowCalibrationForm {
  snowfallTemperature: ParameterTreatmentForm;
  meltTemperature: ParameterTreatmentForm;
  degreeDayFactor: ParameterTreatmentForm;
  sharpness: ParameterTreatmentForm;
}

export interface WorkspaceForm {
  modelId: ModelId;
  experimentId: string;
  name: string;
  description: string;
  delimiter: string;
  encoding: string;
  mapping: ColumnMapping;
  additionalPredictors: AdditionalPredictorForm[];
  precipitationUnit: "mm/day" | "inch/day";
  temperatureUnit: "degC" | "degF" | "K";
  potentialEvapotranspirationUnit: "mm/day" | "inch/day";
  streamflowUnit: "mm/day" | "m3/s";
  basinAreaKm2: number | null;
  precipitationMissing: "error" | "zero" | "linear_interpolation";
  temperatureMissing: "error" | "linear_interpolation";
  potentialEvapotranspirationMissing: "error" | "zero" | "linear_interpolation";
  streamflowMissing: "exclude" | "linear_interpolation";
  maximumInterpolationGap: number;
  missingTextTokens: string;
  numericSentinels: string;
  precipitationNegative: "error" | "clip_to_zero" | "set_missing";
  potentialEvapotranspirationNegative: "error" | "clip_to_zero" | "set_missing";
  streamflowNegative: "keep" | "error" | "exclude" | "clip_to_zero";
  requireTimestamp: boolean;
  periodMode: "rows" | "dates";
  snowModule: SnowModuleId;
  snowfallTemperature: number;
  meltTemperature: number;
  degreeDayFactor: number;
  sharpness: number;
  snowCalibration: SnowCalibrationForm;
  routingProcess: RoutingProcessId;
  maxbas: number;
  maxbasTreatment: ParameterTreatmentForm;
  initialRoutingState: [number, number, number, number, number, number];
  initialSnowStorage: number;
  initialSoilStorage: number;
  initialGroundwaterStorage: number;
  initialUpperZoneStorage: number;
  initialLowerZoneStorage: number;
  substeps: number;
  startIndex: number;
  endIndex: number | null;
  conceptualProtocol:
    | "calibration_validation_test"
    | "calibration_test"
    | "legacy_calibration_only";
  warmupSteps: number;
  objective: OptimizationObjectiveId;
  pamlPhysicalObjective: OptimizationObjectiveId;
  calibrationAlgorithm:
    | "differential_evolution.rand1bin.v1"
    | "uniform_random_search.v1";
  evaluations: number;
  dePopulationSize: number | null;
  deGenerations: number;
  deMutationFactor: number;
  deCrossoverProbability: number;
  seed: number;
  mlpFeatureRoles: MlpFeatureRole[];
  laggedFeatures: LaggedFeatureForm[];
  mlpLookback: number;
  s2sWarmup: number;
  mlpHiddenLayers: string;
  mlpActivation: "relu" | "tanh";
  mlpOutputActivation: "linear" | "softplus";
  recurrentUnits: number;
  recurrentLayers: string;
  cnnLayers: Cnn1dLayerForm[];
  trainStartIndex: number;
  trainEndIndex: number;
  validationEndIndex: number;
  testEndIndex: number | null;
  trainingEpochs: number;
  trainingBatchSize: number;
  trainingLearningRate: number;
  trainingPatience: number;
  trainingMinimumDelta: number;
  trainingStride: number;
  trainingGradientClipNorm: number;
  initializationEnsembleEnabled: boolean;
  initializationEnsembleMembers: number;
  pamlPhysicalModelId: ConceptualModelId;
  pamlCorrectorId: NeuralModelId;
  pamlCorrectionStrategy: "direct" | "additive_residual";
  pamlInputRoles: string[];
  discoveryProcessCount: number;
  discoveryHiddenUnits: number;
  discoveryTrainInputSwitches: boolean;
  discoveryTrainProcessSwitches: boolean;
  discoveryDecorrelationWeight: number;
  discoveryMassBalanceWeight: number;
  discoverySmoothnessWeight: number;
  discoverySmoothnessThreshold: number;
  discoveryBinaryWeight: number;
  discoverySparsityWeight: number;
  discoveryTargetCorrelation: number;
}

export interface ParameterTreatmentConfig {
  mode: "fixed" | "calibrate";
  lower_bound: number;
  upper_bound: number;
}

export interface SnowConfiguration {
  module_id: SnowModuleId;
  snowfall_temperature?: number;
  melt_temperature?: number;
  degree_day_factor?: number;
  sharpness?: number;
  calibration?: {
    snowfall_temperature: ParameterTreatmentConfig;
    melt_temperature: ParameterTreatmentConfig;
    degree_day_factor: ParameterTreatmentConfig;
    sharpness?: ParameterTreatmentConfig;
  };
}

export interface ExperimentConfig {
  schema_version: "1.0" | "1.1" | "1.2" | "1.3" | "1.4" | "1.5" | "1.6" | "1.7" | "1.8" | "1.9" | "1.10" | "1.11" | "1.12" | "1.13" | "1.14" | "1.15" | "1.16" | "1.17" | "1.18" | "1.19" | "1.20";
  experiment_id: string;
  name: string;
  description: string;
  data: {
    kind: "csv";
    path: string;
    columns: {
      timestamp?: string;
      precipitation: string;
      temperature: string;
      potential_evapotranspiration: string;
      streamflow: string;
      additional_predictors?: Array<{
        name: string;
        column: string;
        unit: string;
        missing: "error" | "linear_interpolation";
      }>;
    };
    delimiter: string;
    encoding: string;
    expected_sha256?: string;
    processing: {
      units: {
        precipitation: WorkspaceForm["precipitationUnit"];
        temperature: WorkspaceForm["temperatureUnit"];
        potential_evapotranspiration: WorkspaceForm["potentialEvapotranspirationUnit"];
        streamflow: WorkspaceForm["streamflowUnit"];
        basin_area_km2: number | null;
      };
      missing: {
        precipitation: WorkspaceForm["precipitationMissing"];
        temperature: WorkspaceForm["temperatureMissing"];
        potential_evapotranspiration: WorkspaceForm["potentialEvapotranspirationMissing"];
        streamflow: WorkspaceForm["streamflowMissing"];
        maximum_interpolation_gap: number;
        text_tokens: string[];
        numeric_sentinels: number[];
      };
      negative: {
        precipitation: WorkspaceForm["precipitationNegative"];
        potential_evapotranspiration: WorkspaceForm["potentialEvapotranspirationNegative"];
        streamflow: WorkspaceForm["streamflowNegative"];
      };
      timestamps: {
        expected_step: "daily";
        require_timestamp: boolean;
      };
    };
  };
  model: {
    model_id: WorkspaceForm["modelId"];
    snow?: SnowConfiguration;
    routing?: {
      process_id: RoutingProcessId;
      maxbas?: number;
      calibration?: { maxbas: ParameterTreatmentConfig };
      initial_state?: [number, number, number, number, number, number];
    };
    initial_state?: [number, number] | [number, number, number] | [number, number, number, number];
    substeps?: number;
    feature_roles?: MlpFeatureRole[];
    lookback?: number;
    s2s_warmup?: number;
    hidden_layers?: number[];
    recurrent_units?: number;
    recurrent_layers?: number[];
    convolution_layers?: Array<{
      filters: number;
      kernel_size: number;
      activation: "relu" | "tanh";
    }>;
    gradient_clip_norm?: number;
    activation?: WorkspaceForm["mlpActivation"];
    output_activation?: WorkspaceForm["mlpOutputActivation"];
    input_transform?: "standardize_training_only" | "standardize_training_only";
    target_transform?: "standardize_training_only";
    lagged_features?: Array<{ source: string; lag_steps: number }>;
    physical_model?: { model_id: ConceptualModelId };
    cnn_filters?: number;
    cnn_kernel_size?: number;
    corrector?: ExperimentConfig["model"];
    correction_strategy?: WorkspaceForm["pamlCorrectionStrategy"];
    training_strategy?: "joint_end_to_end";
    enforce_nonnegative_output?: boolean;
    process_count?: number;
    hidden_units?: number;
    train_input_switches?: boolean;
    train_process_switches?: boolean;
    loss?: {
      decorrelation_weight: number;
      mass_balance_weight: number;
      smoothness_weight: number;
      smoothness_threshold: number;
      binary_weight: number;
      sparsity_weight: number;
      target_process_correlation: number;
    };
  };
  task: {
    task_type: "calibration" | "training" | "paml_training" | "discovery_training";
    period: {
      start_index: number;
      end_index?: number | null;
      protocol?: WorkspaceForm["conceptualProtocol"];
      calibration_end_index?: number | null;
      warmup_steps?: number;
      train_end_index?: number;
      validation_end_index?: number | null;
      test_end_index?: number | null;
    };
    objective?: { objective_id: WorkspaceForm["objective"] };
    search?: {
      algorithm_id: "uniform_random_search.v1";
      n_evaluations: number;
      seed: number;
      include_defaults: true;
      continue_on_candidate_error: true;
    } | {
      algorithm_id: "differential_evolution.rand1bin.v1";
      generations: number;
      population_size: number | null;
      mutation_factor: number;
      crossover_probability: number;
      seed: number;
      include_defaults: true;
      continue_on_candidate_error: true;
    };
    target_role?: "streamflow";
    lead_steps?: 0 | 1;
    stride?: number;
    training?: {
      optimizer_id: "adam.tensorflow.v1" | "adam.numpy.full_sequence.v1";
      epochs: number;
      batch_size?: number;
      learning_rate: number;
      patience: number;
      minimum_delta: number;
      seed: number;
      shuffle?: boolean;
      gradient_clip_norm?: number;
    };
    ensemble?: {
      enabled: boolean;
      members: number;
      aggregation: "arithmetic_mean";
      lower_quantile: number;
      upper_quantile: number;
    };
    warmup_steps?: number;
    physical_warmup_steps?: number;
    physical_objective?: { objective_id: WorkspaceForm["objective"] };
    physical_search?: {
      algorithm_id: "uniform_random_search.v1";
      n_evaluations: number;
      seed: number;
      include_defaults: true;
      continue_on_candidate_error: true;
    } | {
      algorithm_id: "differential_evolution.rand1bin.v1";
      generations: number;
      population_size: number | null;
      mutation_factor: number;
      crossover_probability: number;
      seed: number;
      include_defaults: true;
      continue_on_candidate_error: true;
    };
  };
  metadata: Record<string, string>;
}

export interface ValidationSummary {
  experiment_id: string;
  configuration_hash: string;
  source_rows: number;
  selected_rows: number;
  warmup_steps: number;
  scored_observations: number;
  transformed_value_count: number;
  data_quality_warning_count: number;
  primary_evaluation_partition: string;
  partition_scored_observations: Record<string, number>;
  warnings: StructuredError[];
}

export interface DataQualityIssue {
  code: string;
  severity: "warning" | "error";
  summary: string;
  explanation: string;
  suggested_action: string;
  variable: string | null;
  count: number;
  csv_rows: number[];
}

export interface VariableQualitySummary {
  variable: string;
  source_column: string;
  source_unit: string;
  model_unit: string;
  row_count: number;
  raw_valid_count: number;
  raw_missing_count: number;
  raw_invalid_count: number;
  raw_negative_count: number;
  processed_valid_count: number;
  processed_missing_count: number;
  transformed_count: number;
  raw_minimum: number | null;
  raw_maximum: number | null;
  processed_minimum: number | null;
  processed_maximum: number | null;
  processed_mean: number | null;
  processed_standard_deviation: number | null;
  processed_q25: number | null;
  processed_median: number | null;
  processed_q75: number | null;
}

export interface DataQualityReport {
  source_path: string;
  sha256: string;
  row_count: number;
  ready: boolean;
  error_count: number;
  warning_count: number;
  transformed_value_count: number;
  variables: VariableQualitySummary[];
  timestamps: {
    assigned: boolean;
    source_column: string | null;
    first: string | null;
    last: string | null;
    valid_count: number;
    missing_count: number;
    invalid_count: number;
    duplicate_or_reversed_count: number;
    irregular_step_count: number;
    expected_step: "daily";
  };
  issues: DataQualityIssue[];
  timestamp_labels: string[] | null;
  observation_valid: boolean[] | null;
  correlation_variables: string[];
  correlation_matrix: Array<Array<number | null>>;
}

export interface ProgressEvent {
  task_id: string;
  sequence: number;
  state: string;
  stage: string;
  message: string;
  completed_units: number;
  total_units: number | null;
  fraction: number | null;
  elapsed_seconds: number;
  estimated_remaining_seconds: number | null;
  best_score: number | null;
  severity: "info" | "warning" | "error";
  error: StructuredError | null;
}

export interface TaskSnapshot {
  task_id: string;
  experiment_id: string;
  state: string;
  created_at: string;
  updated_at: string;
  event_count: number;
  latest_event: ProgressEvent;
  has_result: boolean;
  terminal: boolean;
  error: StructuredError | null;
}

export interface SimulationResult {
  outputs: Array<Array<number | null>>;
  states: number[][];
  fluxes: number[][];
  output_names: string[];
  state_names: string[];
  flux_names: string[];
  diagnostics: Record<string, number | string | number[]>;
}

export interface EvaluationMetricResult {
  metric_id: string;
  symbol: string;
  name: string;
  value: number | null;
  direction: "maximize" | "minimize" | "target";
  ideal_value: number;
  unit: string;
  definition: string;
  unavailable_reason: string | null;
}

export interface RunResult {
  task_id: string;
  state: string;
  experiment_id: string;
  configuration_hash: string;
  validation: ValidationSummary;
  calibration: {
    objective_name: string;
    objective_direction: "maximize" | "minimize";
    records: Array<{
      index: number;
      parameters: Record<string, number>;
      score: number | null;
      succeeded: boolean;
      generation?: number | null;
      population_member?: number | null;
    }>;
    best_parameters: Record<string, number>;
    best_score: number;
    warnings: StructuredError[];
    algorithm_id?: string;
    termination_reason?: string;
    generations_completed?: number;
  } | null;
  training?: {
    engine_id: string;
    model_id?: ModelId;
    corrector_model_id?: NeuralModelId;
    physical_model_id?: ConceptualModelId;
    physical_parameters?: Record<string, number>;
    correction_strategy?: WorkspaceForm["pamlCorrectionStrategy"];
    training_strategy?: "joint_end_to_end";
    artifact: Record<string, unknown>;
    history: Array<{
      epoch: number;
      training_rmse: number;
      validation_rmse: number;
      training_objective?: number;
      validation_objective?: number;
      gradient_norm_before_clipping?: number;
      total_loss?: number;
      data_loss?: number;
      decorrelation_loss?: number;
      mass_balance_loss?: number;
      smoothness_loss?: number;
      binary_loss?: number;
      sparsity_loss?: number;
    }>;
    best_epoch: number;
    epochs_completed: number;
    stopped_early: boolean;
    selection_metric: string;
    best_validation_score: number | null;
    objective?: {
      objective_id: OptimizationObjectiveId;
      symbol: string;
      name: string;
      direction: "maximize" | "minimize";
      ideal_value: number;
      unit: string;
      requires_global_batch: boolean;
    };
    feature_roles?: string[];
    base_feature_roles?: string[];
    feature_pipeline?: {
      base_features: string[];
      lagged_features: Array<{ source: string; lag_steps: number; name: string }>;
      expanded_features: string[];
      maximum_lag: number;
      causal: boolean;
    };
    lookback?: number;
    recurrent_units?: number;
    recurrent_layers?: number[];
    prediction_layout?: "sequence_to_value" | "sequence_to_sequence";
    process_dynamics?: "instantaneous";
    process_count?: number;
    hidden_units?: number;
    snow_module_id?: SnowModuleId | null;
    snow_parameters?: Record<string, number>;
    calibrated_snow_parameters?: string[];
    snow_calibration_method?: "fixed" | "bounded_alternating_coordinate.v1";
    input_roles?: string[];
    input_switch_probabilities?: number[];
    process_switch_probabilities?: number[];
    loss_configuration?: Record<string, number>;
    convolution_layers?: Array<{
      filters: number;
      kernel_size: number;
      activation: "relu" | "tanh";
    }> | null;
    partition_metrics: Record<string, EvaluationMetricResult[]>;
    initialization_ensemble?: {
      method_id: "neural_initialization_ensemble.v1";
      member_count_requested: number;
      member_count_completed: number;
      member_seeds: number[];
      aggregation: "arithmetic_mean";
      lower_quantile: number;
      upper_quantile: number;
      representative_member: number;
      scope: "neural_weight_initialization" | "neural_corrector_weight_initialization";
      physical_calibration_reused?: boolean;
      members: Array<{
        member: number;
        seed: number;
        best_epoch: number;
        epochs_completed: number;
        stopped_early: boolean;
        cancelled: boolean;
        best_validation_rmse: number | null;
        best_validation_score?: number | null;
        partition_metrics: Record<string, EvaluationMetricResult[]>;
      }>;
    } | null;
  } | null;
  best_simulation: SimulationResult;
  timestamps: string[] | null;
  observations: Array<number | null> | null;
  forcings: Record<string, number[]>;
  neural_context?: Record<string, number[]>;
  evaluation_metrics?: EvaluationMetricResult[];
  partition_metrics?: Record<string, EvaluationMetricResult[]>;
  run_kind?: "calibration" | "training" | "forecast";
  source_task_id?: string;
  continuation_task_id?: string | null;
  model_id?: ModelId;
  initialization?: "source_final" | "configured" | "custom" | "member_final";
  initial_state?: number[];
  forecast_ensemble?: {
    method_id: "meteorological_forcing_ensemble.v1";
    cycling_method_id?: "memberwise_forecast_cycle.v1" | null;
    cycle_index?: number;
    parent_task_id?: string | null;
    parent_issue_id?: string | null;
    issue_id: string;
    issue_time: string | null;
    first_valid_time: string | null;
    spinup_steps: number;
    scope: "meteorological_forcing_scenarios";
    member_count: number;
    member_ids: string[];
    aggregation: "arithmetic_mean";
    lower_quantile: number;
    upper_quantile: number;
    members: Array<{
      member_id: string;
      predictions: number[][];
      initial_state?: number[] | null;
      final_state: number[];
      neural_context?: Record<string, number[]> | null;
      evaluation_metrics: EvaluationMetricResult[];
      initialization_ensemble_lower: number[] | null;
      initialization_ensemble_upper: number[] | null;
    }>;
    lead_time_diagnostics: Array<{
      lead_step: number;
      timestamp: string | null;
      valid_members: number;
      observation: number | null;
      ensemble_mean: number | null;
      ensemble_median: number | null;
      lower: number | null;
      upper: number | null;
      spread: number | null;
      signed_error: number | null;
      absolute_error: number | null;
      crps: number | null;
    }>;
    aggregate_scores: {
      scored_leads: number;
      mean_crps: number | null;
      mean_error_bias: number | null;
      mean_absolute_error: number | null;
      root_mean_square_error: number | null;
      mean_ensemble_spread: number | null;
    };
    warnings: Array<{
      code: string;
      message: string;
    }>;
  };
}

export type ScientificVerificationStatus =
  | "passed"
  | "failed"
  | "not_applicable";

export interface ScientificVerificationCheck {
  check_id:
    | "archive_integrity"
    | "physical_nonnegativity"
    | "mass_balance"
    | "deterministic_replay"
    | "restart_equivalence"
    | "future_forcing_causality";
  name: string;
  status: ScientificVerificationStatus;
  description: string;
  maximum_absolute_difference: number | null;
  tolerance: number | null;
  details: Record<string, unknown>;
}

export interface RunScientificVerification {
  method_id: "scientific_run_verification.v1";
  task_id: string;
  experiment_id: string;
  model_id: ModelId;
  status: "passed" | "failed";
  checked_intervals: number;
  tolerances: {
    replay_restart_and_causality: number;
    mass_balance_mm_per_interval: number;
    nonnegative_physical_quantities: number;
  };
  summary: {
    passed: number;
    failed: number;
    not_applicable: number;
  };
  checks: ScientificVerificationCheck[];
  implementation_source: {
    relationship: string;
    historical_sources: string[];
    equivalence_claim: "not_applicable" | "not_claimed" | "unregistered";
    statement: string;
    declared_adaptations: string[];
  };
  limitations: string[];
}

export interface ForecastRequest {
  source_task_id: string;
  initialization: "source_final" | "configured" | "custom";
  initial_state?: number[];
  antecedent_forcings?: Record<string, number[]>;
  data: Omit<ExperimentConfig["data"], "columns"> & {
    columns: {
      timestamp?: string;
      precipitation: string;
      temperature: string;
      potential_evapotranspiration: string;
      streamflow?: string;
      additional_predictors?: Array<{
        name: string;
        column: string;
        unit: string;
        missing: "error" | "linear_interpolation";
      }>;
    };
  };
}

export interface ForecastEnsembleRequest {
  source_task_id: string;
  continuation_task_id?: string;
  issue_id?: string;
  issue_time?: string;
  spinup_steps?: number;
  initialization: "source_final" | "configured" | "custom";
  initial_state?: number[];
  antecedent_forcings?: Record<string, number[]>;
  lower_quantile: number;
  upper_quantile: number;
  members: Array<{
    member_id: string;
    data: ForecastRequest["data"];
  }>;
}

export interface ForecastEnsembleArchiveSummary {
  task_id: string;
  saved_at: string;
  source_task_id: string;
  continuation_task_id?: string | null;
  experiment_id: string;
  model_id: ModelId;
  issue_id: string;
  issue_time: string | null;
  first_valid_time: string | null;
  spinup_steps: number;
  cycle_index?: number;
  parent_task_id?: string | null;
  member_count: number;
  horizon: number;
  scored_leads: number;
  lower_quantile: number;
  upper_quantile: number;
}

export interface ForecastEnsembleArchive {
  record_type: "hydrosoftware.forecast-ensemble";
  record_version: string;
  task_id: string;
  saved_at: string;
  result: RunResult;
}

export interface ForecastVerificationLead {
  lead_step: number;
  available_cases: number;
  reported: boolean;
  scored_cases?: number;
  mean_crps?: number;
  mean_error_bias?: number;
  mean_absolute_error?: number;
  root_mean_square_error?: number;
  root_mean_ensemble_variance?: number;
  spread_skill_ratio?: number | null;
  interval_coverage?: number;
  nominal_interval_coverage?: number;
  coverage_error?: number;
  rank_counts?: number[];
  rank_relative_frequencies?: number[];
}

export interface ForecastVerificationDiagnostics {
  scored_cases: number;
  mean_crps: number;
  mean_error_bias: number;
  mean_absolute_error: number;
  root_mean_square_error: number;
  root_mean_ensemble_variance: number;
  spread_skill_ratio: number | null;
  interval_coverage: number;
  nominal_interval_coverage: number;
  coverage_error: number;
  rank_counts: number[];
  rank_relative_frequencies: number[];
  deterministic_metrics?: EvaluationMetricResult[];
}

export interface ForecastVerification {
  task_id: string;
  state: "completed";
  method_id: "meteorological_ensemble_hindcast_verification.v1";
  source_task_id: string;
  experiment_id: string;
  configuration_hash: string;
  model_id: ModelId;
  seed: number;
  case_count: number;
  member_count: number;
  horizon: number;
  minimum_cases_per_lead: number;
  lower_quantile: number;
  upper_quantile: number;
  cases: Array<{
    task_id: string;
    issue_id: string;
    issue_time: string | null;
    first_valid_time: string | null;
    last_valid_time: string | null;
    spinup_steps: number;
    initialization: string;
    scored_leads: number;
  }>;
  lead_time_diagnostics: ForecastVerificationLead[];
  aggregate: ForecastVerificationDiagnostics;
  duplicate_valid_time_events: number;
  warnings: Array<{ code: string; message: string }>;
}

export interface ForecastVerificationArchiveSummary {
  task_id: string;
  saved_at: string;
  method_id: "meteorological_ensemble_hindcast_verification.v1";
  source_task_id: string;
  experiment_id: string;
  model_id: ModelId;
  case_count: number;
  member_count: number;
  horizon: number;
  mean_crps: number;
}

export interface ForecastVerificationArchive {
  record_type: "hydrosoftware.forecast-verification";
  record_version: string;
  task_id: string;
  saved_at: string;
  result: ForecastVerification;
}

export interface ForecastPostprocessingLead {
  lead_step: number;
  available_cases: number;
  reported: boolean;
  raw?: ForecastVerificationDiagnostics;
  adjusted?: ForecastVerificationDiagnostics;
  crps_skill_score?: number | null;
}

export interface ForecastPostprocessing {
  task_id: string;
  state: "completed";
  method_id: "affine_member_recalibration.v1";
  source_task_id: string;
  experiment_id: string;
  configuration_hash: string;
  model_id: ModelId;
  member_count: number;
  horizon: number;
  lower_quantile: number;
  upper_quantile: number;
  seed: number;
  split: {
    strategy: "chronological_oldest_fit_latest_evaluate";
    training_case_count: number;
    evaluation_case_count: number;
    training_task_ids: string[];
    evaluation_task_ids: string[];
    last_training_origin: string;
    first_evaluation_origin: string;
  };
  parameter_bounds: {
    mean_slope: [number, number];
    spread_multiplier: [number, number];
  };
  minimum_training_cases_per_lead: number;
  parameters: Array<{
    lead_step: number;
    intercept: number;
    mean_slope: number;
    spread_multiplier: number;
    training_case_count: number;
    training_raw_mean_crps: number;
    training_adjusted_mean_crps: number;
  }>;
  evaluation_cases: Array<{
    task_id: string;
    issue_id: string;
    issue_time: string | null;
    timestamps: string[] | null;
    observations: number[];
    raw_member_predictions: number[][];
    adjusted_member_predictions: number[][];
  }>;
  lead_time_diagnostics: ForecastPostprocessingLead[];
  aggregate: {
    raw: ForecastVerificationDiagnostics;
    adjusted: ForecastVerificationDiagnostics;
    crps_skill_score: number | null;
  };
  warnings: Array<{ code: string; message: string }>;
}

export interface ForecastPostprocessingArchiveSummary {
  task_id: string;
  saved_at: string;
  method_id: "affine_member_recalibration.v1";
  source_task_id: string;
  experiment_id: string;
  model_id: ModelId;
  member_count: number;
  horizon: number;
  training_case_count: number;
  evaluation_case_count: number;
  raw_mean_crps: number;
  adjusted_mean_crps: number;
  crps_skill_score: number | null;
}

export interface ForecastPostprocessingArchive {
  record_type: "hydrosoftware.forecast-postprocessing";
  record_version: string;
  task_id: string;
  saved_at: string;
  result: ForecastPostprocessing;
}

export interface ForecastEventDiagnostics {
  scored_cases: number;
  event_count: number;
  nonevent_count: number;
  event_frequency: number;
  mean_forecast_probability: number;
  brier_score: number;
  climatology_brier_score: number;
  brier_skill_score: number | null;
  roc_auc: number | null;
  reliability_bins: Array<{
    bin: number;
    lower_probability: number;
    upper_probability: number;
    count: number;
    mean_forecast_probability: number | null;
    observed_frequency: number | null;
    wilson_lower_95: number | null;
    wilson_upper_95: number | null;
  }>;
  roc_curve: Array<{
    probability_threshold: number;
    false_positive_rate: number;
    true_positive_rate: number;
  }>;
  cost_loss_curve: Array<{
    cost_loss_ratio: number;
    forecast_expense: number;
    climatology_expense: number;
    perfect_expense: number;
    relative_economic_value: number | null;
  }>;
  maximum_relative_economic_value: number | null;
  cost_loss_ratio_at_maximum_value: number | null;
}

export interface ForecastEventVerification {
  task_id: string;
  state: "completed";
  method_id: "threshold_event_decision_verification.v1";
  source_task_id: string;
  experiment_id: string;
  configuration_hash: string;
  model_id: ModelId;
  member_count: number;
  horizon: number;
  threshold: {
    mode: "absolute" | "training_quantile";
    value: number;
    unit: "mm/day";
    event_definition: "observed_or_member_streamflow > threshold";
    training_quantile: number | null;
    training_observation_count: number;
  };
  split: ForecastPostprocessing["split"];
  climatology: {
    training_event_probability: number;
    training_event_count: number;
    training_nonevent_count: number;
  };
  probability_bin_count: number;
  minimum_evaluation_cases_per_lead: number;
  lead_time_diagnostics: Array<{
    lead_step: number;
    available_cases: number;
    reported: boolean;
    raw?: ForecastEventDiagnostics;
    adjusted?: ForecastEventDiagnostics;
    adjusted_brier_skill_vs_raw?: number | null;
  }>;
  aggregate: {
    raw: ForecastEventDiagnostics;
    adjusted: ForecastEventDiagnostics;
    adjusted_brier_skill_vs_raw: number | null;
  };
  postprocessing_method_id: "affine_member_recalibration.v1";
  warnings: Array<{ code: string; message: string }>;
}

export interface ForecastEventVerificationArchiveSummary {
  task_id: string;
  saved_at: string;
  method_id: "threshold_event_decision_verification.v1";
  source_task_id: string;
  experiment_id: string;
  model_id: ModelId;
  member_count: number;
  horizon: number;
  training_case_count: number;
  evaluation_case_count: number;
  threshold_value: number;
  threshold_mode: "absolute" | "training_quantile";
  raw_brier_score: number;
  adjusted_brier_score: number;
  adjusted_brier_skill_vs_raw: number | null;
}

export interface ForecastEventVerificationArchive {
  record_type: "hydrosoftware.forecast-event-verification";
  record_version: string;
  task_id: string;
  saved_at: string;
  result: ForecastEventVerification;
}

export interface SavedExperimentSummary {
  experiment_id: string;
  name: string;
  description: string;
  configuration_hash: string;
  created_at: string;
  updated_at: string;
  latest_run_id: string | null;
  run_count: number;
}

export interface SavedExperiment {
  summary: SavedExperimentSummary;
  config: ExperimentConfig;
}

export interface RunArchiveSummary {
  task_id: string;
  experiment_id: string;
  experiment_name: string;
  model_id: ModelId;
  run_kind: "calibration" | "training" | "forecast";
  configuration_hash: string;
  state: string;
  saved_at: string;
  objective_name: string | null;
  best_score: number | null;
}

export interface RunArchive {
  task_id: string;
  experiment_id: string;
  configuration_hash: string;
  state: string;
  saved_at: string;
  config: ExperimentConfig;
  result: RunResult;
}

export interface RunComparison {
  alignment: "timestamp" | "source_row";
  source_sha256: string | null;
  aligned_rows: number;
  scored_observations: number;
  first_scored_label: string;
  last_scored_label: string;
  timestamps: string[] | null;
  source_rows: number[] | null;
  observations: number[];
  score_mask: boolean[];
  runs: Array<{
    task_id: string;
    experiment_id: string;
    experiment_name: string;
    model_id: ModelId;
    run_kind: "calibration" | "training" | "forecast";
    saved_at: string;
    metrics: EvaluationMetricResult[];
    predictions: number[];
  }>;
  warnings: Array<{
    code: string;
    message: string;
  }>;
}

export interface UncertaintyMetric extends EvaluationMetricResult {
  confidence_level: number;
  lower: number | null;
  upper: number | null;
  bootstrap_standard_error: number | null;
  valid_replicates: number;
  interval_unavailable_reason: string | null;
}

export interface RunUncertainty {
  schema_version: "1.0";
  method_id: "moving_block_bootstrap_metrics.v1";
  alignment: "timestamp" | "source_row";
  source_sha256: string | null;
  scored_observations: number;
  longest_contiguous_scored_run: number;
  eligible_overlapping_blocks: number;
  block_length: number;
  block_length_source: "autocorrelation_cutoff_heuristic" | "user";
  bootstrap_replicates: number;
  confidence_level: number;
  interval_method: "percentile";
  paired_resampling: boolean;
  seed: number;
  reference_task_id: string;
  runs: Array<{
    task_id: string;
    experiment_id: string;
    experiment_name: string;
    model_id: ModelId;
    run_kind: "calibration" | "training" | "forecast";
    metrics: UncertaintyMetric[];
  }>;
  comparisons: Array<{
    reference_task_id: string;
    reference_experiment_name: string;
    candidate_task_id: string;
    candidate_experiment_name: string;
    metrics: Array<{
      metric_id: string;
      symbol: string;
      name: string;
      direction: "maximize" | "minimize" | "target";
      ideal_value: number;
      unit: string;
      reference_value: number | null;
      candidate_value: number | null;
      raw_difference: number | null;
      raw_difference_lower: number | null;
      raw_difference_upper: number | null;
      improvement: number | null;
      improvement_lower: number | null;
      improvement_upper: number | null;
      bootstrap_fraction_better: number | null;
      valid_replicates: number;
      interpretation:
        | "interval_favors_candidate"
        | "interval_favors_reference"
        | "interval_overlaps_zero";
      interval_unavailable_reason: string | null;
    }>;
  }>;
  warnings: Array<{
    code: string;
    message: string;
  }>;
}

export interface TemporalValidationRequest {
  experiment_ids: string[];
  start_index: number;
  initial_training_steps: number;
  validation_steps: number;
  test_steps: number;
  step_steps: number;
  folds: number;
  window_mode: "expanding" | "sliding";
  minimum_scored_observations: number;
}

export interface TemporalMetricSummary {
  metric_id: string;
  symbol: string;
  available_folds: number;
  mean: number | null;
  median: number | null;
  standard_deviation: number | null;
  minimum: number | null;
  maximum: number | null;
  mean_rank: number;
  best_fold_count: number;
}

export interface TemporalValidation {
  task_id: string;
  state: "completed";
  method_id: "rolling_origin_temporal_validation.v1";
  settings: Omit<TemporalValidationRequest, "experiment_ids">;
  source: {
    sha256: string;
    rows: number;
    timestamps_available: boolean;
  };
  folds: Array<{
    fold: number;
    train_start_index: number;
    train_end_index: number;
    validation_end_index: number;
    test_end_index: number;
    scored_observations: number;
    first_test_label: string;
    last_test_label: string;
    timestamps: string[] | null;
    source_rows: number[];
    observations: number[];
    score_mask: boolean[];
    models: Array<{
      experiment_id: string;
      experiment_name: string;
      model_id: ModelId;
      configuration_hash: string;
      fold_configuration_hash: string;
      fitting_end_index: number;
      validation_used_for:
        | "conceptual_parameter_calibration"
        | "checkpoint_selection";
      metrics: EvaluationMetricResult[];
      predictions: number[];
    }>;
  }>;
  pooled_scored_observations: number;
  models: Array<{
    experiment_id: string;
    experiment_name: string;
    model_id: ModelId;
    configuration_hash: string;
    pooled_metrics: EvaluationMetricResult[];
    fold_metric_summaries: TemporalMetricSummary[];
  }>;
  pairwise: Array<{
    reference_experiment_id: string;
    reference_experiment_name: string;
    candidate_experiment_id: string;
    candidate_experiment_name: string;
    metrics: Array<{
      metric_id: string;
      symbol: string;
      positive_means: "candidate_closer_to_ideal";
      available_folds: number;
      mean: number | null;
      median: number | null;
      standard_deviation: number | null;
      minimum: number | null;
      maximum: number | null;
      candidate_better_fraction: number | null;
    }>;
  }>;
  warnings: Array<{
    code: string;
    message: string;
  }>;
}

export interface TemporalValidationArchiveSummary {
  task_id: string;
  saved_at: string;
  method_id: "rolling_origin_temporal_validation.v1";
  state: "completed";
  window_mode: "expanding" | "sliding";
  folds: number;
  test_steps: number;
  pooled_scored_observations: number;
  experiment_ids: string[];
  experiment_names: string[];
  model_ids: ModelId[];
}

export interface TemporalValidationArchive {
  record_type: "hydrosoftware.temporal-validation";
  record_version: string;
  task_id: string;
  saved_at: string;
  result: TemporalValidation;
}

export interface ShapAttribution {
  feature: string;
  lag_steps: number;
  shap_value: number;
  standard_error: number;
}

export interface RunExplanation {
  schema_version: "1.0";
  method_id: "empirical_interventional_permutation_shap.v1";
  task_id: string;
  model_id: ModelId;
  corrector_model_id: NeuralModelId;
  output_name: "streamflow";
  output_unit: "mm/day";
  background_partition: "training";
  background_rows: number;
  background_source_rows: number[];
  explained_partition: "test";
  explained_rows: number;
  permutations: number;
  seed: number;
  lookback: number;
  feature_roles: string[];
  player_count: number;
  correction_strategy: WorkspaceForm["pamlCorrectionStrategy"] | null;
  local: Array<{
    result_index: number;
    source_row: number;
    label: string;
    prediction: number;
    archived_prediction: number;
    observation: number | null;
    baseline: number;
    physical_streamflow: number | null;
    additivity_error: number;
    attributions: ShapAttribution[];
  }>;
  global_by_feature: Array<{
    feature: string;
    mean_absolute_shap: number;
    mean_shap: number;
  }>;
  global_by_lag: Array<{
    lag_steps: number;
    mean_absolute_shap: number;
    mean_shap: number;
  }>;
  global_cells: Array<{
    feature: string;
    lag_steps: number;
    mean_absolute_shap: number;
    mean_shap: number;
  }>;
  maximum_additivity_error: number;
  maximum_archive_prediction_difference: number;
  warnings: Array<{
    code: string;
    message: string;
  }>;
}
