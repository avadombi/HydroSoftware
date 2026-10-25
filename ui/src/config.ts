import type {
  ColumnMapping,
  ExperimentConfig,
  SnowConfiguration,
  SnowModuleId,
  WorkspaceForm,
} from "./types";

export const interfaceVersion = "1.0.0";

export const emptyMapping: ColumnMapping = {
  timestamp: "",
  precipitation: "",
  temperature: "",
  potential_evapotranspiration: "",
  streamflow: "",
};

export const defaultForm: WorkspaceForm = {
  modelId: "exp_hydro.mass_conservative.v1",
  experimentId: "exp-hydro-calibration",
  name: "EXP-HYDRO calibration",
  description: "Hydrological modelling experiment configured in HydroSoftware.",
  delimiter: ",",
  encoding: "utf-8-sig",
  mapping: emptyMapping,
  additionalPredictors: [],
  precipitationUnit: "mm/day",
  temperatureUnit: "degC",
  potentialEvapotranspirationUnit: "mm/day",
  streamflowUnit: "mm/day",
  basinAreaKm2: null,
  precipitationMissing: "error",
  temperatureMissing: "error",
  potentialEvapotranspirationMissing: "error",
  streamflowMissing: "exclude",
  maximumInterpolationGap: 3,
  missingTextTokens: ", NA, N/A, NaN, null",
  numericSentinels: "",
  precipitationNegative: "error",
  potentialEvapotranspirationNegative: "error",
  streamflowNegative: "keep",
  requireTimestamp: false,
  periodMode: "rows",
  snowModule: "snow.degree_day.hard.v1",
  snowfallTemperature: 0,
  meltTemperature: 0,
  degreeDayFactor: 2,
  sharpness: 5,
  snowCalibration: {
    snowfallTemperature: { mode: "fixed", lowerBound: -3, upperBound: 0 },
    meltTemperature: { mode: "fixed", lowerBound: 0, upperBound: 3 },
    degreeDayFactor: { mode: "fixed", lowerBound: 0, upperBound: 5 },
    sharpness: { mode: "fixed", lowerBound: 0.5, upperBound: 20 },
  },
  routingProcess: "routing.none.v1",
  maxbas: 3,
  maxbasTreatment: { mode: "fixed", lowerBound: 1, upperBound: 7 },
  initialRoutingState: [0, 0, 0, 0, 0, 0],
  initialSnowStorage: 0,
  initialSoilStorage: 300,
  initialGroundwaterStorage: 50,
  initialUpperZoneStorage: 20,
  initialLowerZoneStorage: 80,
  substeps: 4,
  startIndex: 0,
  endIndex: 1000,
  conceptualProtocol: "calibration_validation_test",
  warmupSteps: 5,
  objective: "nash_sutcliffe_efficiency",
  pamlPhysicalObjective: "nash_sutcliffe_efficiency",
  calibrationAlgorithm: "differential_evolution.rand1bin.v1",
  evaluations: 100,
  dePopulationSize: null,
  deGenerations: 25,
  deMutationFactor: 0.8,
  deCrossoverProbability: 0.9,
  seed: 2026,
  mlpFeatureRoles: ["precipitation", "temperature", "potential_evapotranspiration"],
  laggedFeatures: [],
  mlpLookback: 30,
  s2sWarmup: 10,
  mlpHiddenLayers: "64, 32",
  mlpActivation: "relu",
  mlpOutputActivation: "linear",
  recurrentUnits: 32,
  recurrentLayers: "32",
  cnnLayers: [
    { filters: 16, kernelSize: 5, activation: "relu" },
    { filters: 8, kernelSize: 3, activation: "relu" },
  ],
  trainStartIndex: 0,
  trainEndIndex: 1000,
  validationEndIndex: 1300,
  testEndIndex: null,
  trainingEpochs: 200,
  trainingBatchSize: 64,
  trainingLearningRate: 0.001,
  trainingPatience: 20,
  trainingMinimumDelta: 0.000001,
  trainingStride: 1,
  trainingGradientClipNorm: 5,
  initializationEnsembleEnabled: false,
  initializationEnsembleMembers: 5,
  pamlPhysicalModelId: "exp_hydro.mass_conservative.v1",
  pamlCorrectorId: "cnn1d.tensorflow_s2v.v1",
  pamlCorrectionStrategy: "additive_residual",
  pamlInputRoles: [
    "precipitation",
    "temperature",
    "potential_evapotranspiration",
    "physical_output__streamflow",
  ],
  discoveryProcessCount: 1,
  discoveryHiddenUnits: 8,
  discoveryTrainInputSwitches: false,
  discoveryTrainProcessSwitches: false,
  discoveryDecorrelationWeight: 0,
  discoveryMassBalanceWeight: 0,
  discoverySmoothnessWeight: 0.01,
  discoverySmoothnessThreshold: 0.1,
  discoveryBinaryWeight: 0,
  discoverySparsityWeight: 0,
  discoveryTargetCorrelation: 0,
};

const aliases: Record<keyof ColumnMapping, string[]> = {
  timestamp: ["date", "datetime", "timestamp", "time", "day"],
  precipitation: ["precipitation", "precip", "rainfall", "rain", "prcp", "p"],
  temperature: ["temperature", "temp", "tmean", "tavg", "t"],
  potential_evapotranspiration: [
    "potential_evapotranspiration",
    "potential evapotranspiration",
    "pet",
    "eto",
    "et0",
  ],
  streamflow: ["streamflow", "discharge", "runoff", "flow", "qobs", "q"],
};

function normalizeColumn(value: string): string {
  return value.toLowerCase().trim().replace(/[_-]+/g, " ");
}

export function autoMapColumns(columns: string[]): ColumnMapping {
  const normalized = columns.map((column) => ({
    original: column,
    normalized: normalizeColumn(column),
  }));
  const used = new Set<string>();
  const mapping = { ...emptyMapping };

  (Object.keys(mapping) as Array<keyof ColumnMapping>).forEach((role) => {
    const match = normalized.find(
      ({ original, normalized: candidate }) =>
        !used.has(original) &&
        aliases[role].some((alias) => candidate === normalizeColumn(alias)),
    );
    if (match) {
      mapping[role] = match.original;
      used.add(match.original);
    }
  });
  return mapping;
}

export function mappingIsComplete(
  mapping: ColumnMapping,
  additionalPredictors: WorkspaceForm["additionalPredictors"] = [],
): boolean {
  const required = [
    mapping.precipitation,
    mapping.temperature,
    mapping.potential_evapotranspiration,
    mapping.streamflow,
  ];
  const assigned = [
    ...required,
    ...(mapping.timestamp ? [mapping.timestamp] : []),
    ...additionalPredictors.map((item) => item.column),
  ];
  const predictorNames = additionalPredictors.map((item) => item.name);
  return required.every(Boolean)
    && additionalPredictors.every((item) => item.name && item.column && item.unit)
    && new Set(assigned.filter(Boolean)).size === assigned.length
    && new Set(predictorNames).size === predictorNames.length;
}

function parameterTreatment(
  treatment: WorkspaceForm["snowCalibration"]["snowfallTemperature"],
) {
  return {
    mode: treatment.mode,
    lower_bound: treatment.lowerBound,
    upper_bound: treatment.upperBound,
  } as const;
}

function snowConfiguration(form: WorkspaceForm): SnowConfiguration {
  if (form.snowModule === "snow.none.v1") {
    return { module_id: form.snowModule };
  }
  const shared = {
    module_id: form.snowModule,
    snowfall_temperature: form.snowfallTemperature,
    melt_temperature: form.meltTemperature,
    degree_day_factor: form.degreeDayFactor,
    calibration: {
      snowfall_temperature: parameterTreatment(form.snowCalibration.snowfallTemperature),
      melt_temperature: parameterTreatment(form.snowCalibration.meltTemperature),
      degree_day_factor: parameterTreatment(form.snowCalibration.degreeDayFactor),
    },
  };
  return form.snowModule === "snow.degree_day.smooth.v1"
    ? {
      ...shared,
      sharpness: form.sharpness,
      calibration: {
        ...shared.calibration,
        sharpness: parameterTreatment(form.snowCalibration.sharpness),
      },
    }
    : shared;
}

function calibrationSearchConfiguration(
  form: WorkspaceForm,
): NonNullable<ExperimentConfig["task"]["search"]> {
  return form.calibrationAlgorithm === "differential_evolution.rand1bin.v1"
    ? {
      algorithm_id: "differential_evolution.rand1bin.v1",
      generations: form.deGenerations,
      population_size: form.dePopulationSize,
      mutation_factor: form.deMutationFactor,
      crossover_probability: form.deCrossoverProbability,
      seed: form.seed,
      include_defaults: true,
      continue_on_candidate_error: true,
    }
    : {
      algorithm_id: "uniform_random_search.v1",
      n_evaluations: form.evaluations,
      seed: form.seed,
      include_defaults: true,
      continue_on_candidate_error: true,
    };
}

function conceptualModelConfiguration(
  form: WorkspaceForm,
  modelId: WorkspaceForm["pamlPhysicalModelId"],
): ExperimentConfig["model"] {
  if (modelId === "exp_hydro.mass_conservative.v1") {
    return {
      model_id: modelId,
      snow: snowConfiguration(form),
      initial_state: [form.initialSnowStorage, form.initialSoilStorage],
      substeps: form.substeps,
    };
  }
  return {
    model_id: modelId,
    snow: snowConfiguration(form),
    routing: form.routingProcess === "routing.maxbas.triangular.v1"
      ? {
        process_id: form.routingProcess,
        maxbas: form.maxbas,
        calibration: {
          maxbas: parameterTreatment(form.maxbasTreatment),
        },
        initial_state: form.initialRoutingState,
      }
      : { process_id: "routing.none.v1" },
    initial_state: [
      form.initialSnowStorage,
      form.initialSoilStorage,
      form.initialUpperZoneStorage,
      form.initialLowerZoneStorage,
    ],
  };
}

function neuralModelConfiguration(
  form: WorkspaceForm,
  modelId: WorkspaceForm["pamlCorrectorId"],
  featureRoles: string[],
): ExperimentConfig["model"] {
  const common = {
    model_id: modelId,
    feature_roles: featureRoles,
    lagged_features: form.laggedFeatures.map((item) => ({
      source: item.source,
      lag_steps: item.lagSteps,
    })),
    lookback: form.mlpLookback,
    s2s_warmup: modelId.includes("_s2s.") ? form.s2sWarmup : 0,
    output_activation: form.mlpOutputActivation,
    input_transform: "standardize_training_only" as const,
    target_transform: "standardize_training_only" as const,
  };
  if (modelId === "mlp.tensorflow_s2v.v1" || modelId === "mlp.tensorflow_s2s.v2") {
    return {
      ...common,
      hidden_layers: form.mlpHiddenLayers
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean)
        .map(Number),
      activation: form.mlpActivation,
    };
  }
  if (modelId === "cnn1d.tensorflow_s2v.v1" || modelId === "cnn1d.tensorflow_s2s.v2") {
    return {
      ...common,
      convolution_layers: form.cnnLayers.map((layer) => ({
        filters: layer.filters,
        kernel_size: layer.kernelSize,
        activation: layer.activation,
      })),
      gradient_clip_norm: form.trainingGradientClipNorm,
    };
  }
  const recurrentLayers = (
    modelId === "lstm.tensorflow_s2v.v1" || modelId === "gru.tensorflow_s2v.v1"
      ? [form.recurrentUnits]
      : form.recurrentLayers
        .split(",")
        .map((value) => value.trim())
        .filter(Boolean)
        .map(Number)
  );
  return {
    ...common,
    recurrent_units: recurrentLayers[0] ?? form.recurrentUnits,
    recurrent_layers: recurrentLayers,
    gradient_clip_norm: form.trainingGradientClipNorm,
  };
}

export function buildExperimentConfig(
  form: WorkspaceForm,
  relativePath: string,
  expectedSha256?: string,
): ExperimentConfig {
  const columns: ExperimentConfig["data"]["columns"] = {
    precipitation: form.mapping.precipitation,
    temperature: form.mapping.temperature,
    potential_evapotranspiration: form.mapping.potential_evapotranspiration,
    streamflow: form.mapping.streamflow,
  };
  if (form.mapping.timestamp) columns.timestamp = form.mapping.timestamp;
  if (form.additionalPredictors.length) {
    columns.additional_predictors = form.additionalPredictors.map((item) => ({
      name: item.name,
      column: item.column,
      unit: item.unit,
      missing: item.missing,
    }));
  }

  return {
    schema_version: "1.20",
    experiment_id: form.experimentId,
    name: form.name,
    description: form.description,
    data: {
      kind: "csv",
      path: relativePath,
      columns,
      delimiter: form.delimiter,
      encoding: form.encoding,
      ...(expectedSha256 ? { expected_sha256: expectedSha256 } : {}),
      processing: {
        units: {
          precipitation: form.precipitationUnit,
          temperature: form.temperatureUnit,
          potential_evapotranspiration: form.potentialEvapotranspirationUnit,
          streamflow: form.streamflowUnit,
          basin_area_km2: form.basinAreaKm2,
        },
        missing: {
          precipitation: form.precipitationMissing,
          temperature: form.temperatureMissing,
          potential_evapotranspiration: form.potentialEvapotranspirationMissing,
          streamflow: form.streamflowMissing,
          maximum_interpolation_gap: form.maximumInterpolationGap,
          text_tokens: form.missingTextTokens.split(",").map((token) => token.trim()),
          numeric_sentinels: form.numericSentinels
            .split(",")
            .map((token) => token.trim())
            .filter(Boolean)
            .map((token) => Number(token))
            .filter((value, index, items) => Number.isFinite(value) && items.indexOf(value) === index),
        },
        negative: {
          precipitation: form.precipitationNegative,
          potential_evapotranspiration: form.potentialEvapotranspirationNegative,
          streamflow: form.streamflowNegative,
        },
        timestamps: {
          expected_step: "daily",
          require_timestamp: form.requireTimestamp,
        },
      },
    },
    model: form.modelId === "deepdiscover.numpy_instantaneous_s2s.v1"
      ? {
        model_id: form.modelId,
        snow: snowConfiguration(form),
        initial_state: [form.initialSnowStorage, form.initialSoilStorage],
        process_count: form.discoveryProcessCount,
        hidden_units: form.discoveryHiddenUnits,
        train_input_switches: form.discoveryTrainInputSwitches,
        train_process_switches: form.discoveryTrainProcessSwitches,
        input_transform: "standardize_training_only",
        loss: {
          decorrelation_weight: form.discoveryDecorrelationWeight,
          mass_balance_weight: form.discoveryMassBalanceWeight,
          smoothness_weight: form.discoverySmoothnessWeight,
          smoothness_threshold: form.discoverySmoothnessThreshold,
          binary_weight: form.discoveryBinaryWeight,
          sparsity_weight: form.discoverySparsityWeight,
          target_process_correlation: form.discoveryTargetCorrelation,
        },
      }
      : form.modelId === "paml.tensorflow_prnn_cnn_s2s.v1"
      ? {
        model_id: form.modelId,
        physical_model: { model_id: form.pamlPhysicalModelId },
        cnn_filters: form.cnnLayers[0]?.filters ?? 8,
        cnn_kernel_size: form.cnnLayers[0]?.kernelSize ?? 10,
        training_strategy: "joint_end_to_end",
        enforce_nonnegative_output: true,
      }
      : form.modelId === "mlp.tensorflow_s2v.v1"
        || form.modelId === "mlp.tensorflow_s2s.v2"
        || form.modelId === "lstm.tensorflow_s2v.v1"
        || form.modelId === "lstm.tensorflow_stacked_s2v.v2"
        || form.modelId === "lstm.tensorflow_stacked_s2s.v3"
        || form.modelId === "gru.tensorflow_s2v.v1"
        || form.modelId === "gru.tensorflow_stacked_s2v.v2"
        || form.modelId === "gru.tensorflow_stacked_s2s.v3"
        || form.modelId === "cnn1d.tensorflow_s2v.v1"
        || form.modelId === "cnn1d.tensorflow_s2s.v2"
        ? neuralModelConfiguration(form, form.modelId, form.mlpFeatureRoles)
        : conceptualModelConfiguration(form, form.modelId),
    task: form.modelId === "deepdiscover.numpy_instantaneous_s2s.v1" ? {
      task_type: "discovery_training",
      period: {
        start_index: form.trainStartIndex,
        train_end_index: form.trainEndIndex,
        validation_end_index: form.validationEndIndex,
        test_end_index: form.testEndIndex,
      },
      target_role: "streamflow",
      warmup_steps: form.warmupSteps,
      objective: { objective_id: form.objective },
      training: {
        optimizer_id: "adam.numpy.full_sequence.v1",
        epochs: form.trainingEpochs,
        learning_rate: form.trainingLearningRate,
        patience: form.trainingPatience,
        minimum_delta: form.trainingMinimumDelta,
        gradient_clip_norm: form.trainingGradientClipNorm,
        seed: form.seed,
      },
    } : form.modelId === "paml.tensorflow_prnn_cnn_s2s.v1" ? {
      task_type: "paml_training",
      period: {
        start_index: form.trainStartIndex,
        train_end_index: form.trainEndIndex,
        validation_end_index: form.validationEndIndex,
        test_end_index: form.testEndIndex,
      },
      target_role: "streamflow",
      lead_steps: 0,
      stride: 1,
      objective: { objective_id: form.objective },
      warmup_steps: form.warmupSteps,
      training: {
        optimizer_id: "adam.tensorflow.v1",
        epochs: form.trainingEpochs,
        batch_size: form.trainingBatchSize,
        learning_rate: form.trainingLearningRate,
        patience: form.trainingPatience,
        minimum_delta: form.trainingMinimumDelta,
        gradient_clip_norm: form.trainingGradientClipNorm,
        seed: form.seed,
        shuffle: false,
      },
    } : form.modelId === "mlp.tensorflow_s2v.v1"
      || form.modelId === "mlp.tensorflow_s2s.v2"
      || form.modelId === "lstm.tensorflow_s2v.v1"
      || form.modelId === "lstm.tensorflow_stacked_s2v.v2"
      || form.modelId === "lstm.tensorflow_stacked_s2s.v3"
      || form.modelId === "gru.tensorflow_s2v.v1"
      || form.modelId === "gru.tensorflow_stacked_s2v.v2"
      || form.modelId === "gru.tensorflow_stacked_s2s.v3"
      || form.modelId === "cnn1d.tensorflow_s2v.v1"
      || form.modelId === "cnn1d.tensorflow_s2s.v2" ? {
      task_type: "training",
      period: {
        start_index: form.trainStartIndex,
        train_end_index: form.trainEndIndex,
        validation_end_index: form.validationEndIndex,
        test_end_index: form.testEndIndex,
      },
      target_role: "streamflow",
      lead_steps: form.modelId.includes("_s2s.") ? 0 : 1,
      stride: form.modelId.includes("_s2s.") ? 1 : form.trainingStride,
      objective: { objective_id: form.objective },
      training: {
        optimizer_id: "adam.tensorflow.v1",
        epochs: form.trainingEpochs,
        batch_size: form.trainingBatchSize,
        learning_rate: form.trainingLearningRate,
        patience: form.trainingPatience,
        minimum_delta: form.trainingMinimumDelta,
        seed: form.seed,
        shuffle: true,
      },
      ensemble: {
        enabled: form.initializationEnsembleEnabled,
        members: form.initializationEnsembleMembers,
        aggregation: "arithmetic_mean",
        lower_quantile: 0.1,
        upper_quantile: 0.9,
      },
    } : {
      task_type: "calibration",
      period: {
        protocol: form.conceptualProtocol,
        start_index: form.startIndex,
        calibration_end_index: form.endIndex,
        validation_end_index: form.conceptualProtocol === "calibration_validation_test"
          ? form.validationEndIndex
          : null,
        test_end_index: form.conceptualProtocol === "legacy_calibration_only"
          ? null
          : form.testEndIndex,
        warmup_steps: form.warmupSteps,
      },
      objective: { objective_id: form.objective },
      search: calibrationSearchConfiguration(form),
    },
    metadata: {
      interface_version: interfaceVersion,
      created_with: "HydroSoftware",
    },
  };
}

export function workspaceFormFromConfig(config: ExperimentConfig): WorkspaceForm {
  const isPaml = config.model.model_id === "paml.tensorflow_prnn_cnn_s2s.v1";
  const physicalModel = isPaml ? undefined : config.model;
  const pamlBackbone = isPaml
    ? config.model.physical_model?.model_id
    : undefined;
  const neuralModel = config.model;
  const snow = physicalModel?.snow ?? { module_id: "snow.none.v1" as const };
  const moduleId = snow.module_id as SnowModuleId;
  const routing = physicalModel?.routing;
  const processing = config.data.processing;
  const calibrationSearch = config.task.search;
  const treatment = (
    name: "snowfall_temperature" | "melt_temperature" | "degree_day_factor" | "sharpness",
    fallback: WorkspaceForm["snowCalibration"]["snowfallTemperature"],
  ) => {
    const configured = snow.calibration?.[name];
    return configured ? {
      mode: configured.mode,
      lowerBound: configured.lower_bound,
      upperBound: configured.upper_bound,
    } : fallback;
  };
  return {
    modelId: config.model.model_id,
    experimentId: config.experiment_id,
    name: config.name,
    description: config.description,
    delimiter: config.data.delimiter,
    encoding: config.data.encoding,
    mapping: {
      timestamp: config.data.columns.timestamp ?? "",
      precipitation: config.data.columns.precipitation,
      temperature: config.data.columns.temperature,
      potential_evapotranspiration: config.data.columns.potential_evapotranspiration,
      streamflow: config.data.columns.streamflow,
    },
    additionalPredictors: (config.data.columns.additional_predictors ?? []).map((item) => ({
      name: item.name,
      column: item.column,
      unit: item.unit,
      missing: item.missing,
    })),
    precipitationUnit: processing.units.precipitation,
    temperatureUnit: processing.units.temperature,
    potentialEvapotranspirationUnit: processing.units.potential_evapotranspiration,
    streamflowUnit: processing.units.streamflow,
    basinAreaKm2: processing.units.basin_area_km2,
    precipitationMissing: processing.missing.precipitation,
    temperatureMissing: processing.missing.temperature,
    potentialEvapotranspirationMissing: processing.missing.potential_evapotranspiration,
    streamflowMissing: processing.missing.streamflow,
    maximumInterpolationGap: processing.missing.maximum_interpolation_gap,
    missingTextTokens: processing.missing.text_tokens.join(", "),
    numericSentinels: processing.missing.numeric_sentinels.join(", "),
    precipitationNegative: processing.negative.precipitation,
    potentialEvapotranspirationNegative: processing.negative.potential_evapotranspiration,
    streamflowNegative: processing.negative.streamflow,
    requireTimestamp: processing.timestamps.require_timestamp,
    periodMode: "rows",
    snowModule: moduleId,
    snowfallTemperature: Number(snow.snowfall_temperature ?? 0),
    meltTemperature: Number(snow.melt_temperature ?? 0),
    degreeDayFactor: Number(snow.degree_day_factor ?? 0),
    sharpness: Number(snow.sharpness ?? 5),
    snowCalibration: {
      snowfallTemperature: treatment("snowfall_temperature", defaultForm.snowCalibration.snowfallTemperature),
      meltTemperature: treatment("melt_temperature", defaultForm.snowCalibration.meltTemperature),
      degreeDayFactor: treatment("degree_day_factor", defaultForm.snowCalibration.degreeDayFactor),
      sharpness: treatment("sharpness", defaultForm.snowCalibration.sharpness),
    },
    routingProcess: routing?.process_id ?? "routing.none.v1",
    maxbas: Number(routing?.maxbas ?? 3),
    maxbasTreatment: routing?.calibration?.maxbas
      ? {
        mode: routing.calibration.maxbas.mode,
        lowerBound: routing.calibration.maxbas.lower_bound,
        upperBound: routing.calibration.maxbas.upper_bound,
      }
      : defaultForm.maxbasTreatment,
    initialRoutingState: routing?.initial_state
      ? [...routing.initial_state]
      : [...defaultForm.initialRoutingState],
    initialSnowStorage: physicalModel?.initial_state?.[0] ?? 0,
    initialSoilStorage: physicalModel?.initial_state?.[1] ?? 300,
    initialGroundwaterStorage: physicalModel?.initial_state?.length === 3 ? physicalModel.initial_state[2] : 50,
    initialUpperZoneStorage: physicalModel?.initial_state?.length === 4 ? physicalModel.initial_state[2] : 20,
    initialLowerZoneStorage: physicalModel?.initial_state?.length === 4 ? physicalModel.initial_state[3] : 80,
    substeps: physicalModel?.substeps ?? 4,
    startIndex: config.task.period.start_index,
    endIndex: config.task.period.calibration_end_index
      ?? config.task.period.end_index
      ?? null,
    conceptualProtocol: config.task.period.protocol
      ?? "legacy_calibration_only",
    warmupSteps: config.task.physical_warmup_steps
      ?? config.task.warmup_steps
      ?? config.task.period.warmup_steps
      ?? 0,
    objective: config.task.objective?.objective_id
      ?? "root_mean_square_error",
    pamlPhysicalObjective: config.task.objective?.objective_id
      ?? "nash_sutcliffe_efficiency",
    calibrationAlgorithm: calibrationSearch?.algorithm_id
      ?? defaultForm.calibrationAlgorithm,
    evaluations: calibrationSearch?.algorithm_id === "uniform_random_search.v1"
      ? calibrationSearch.n_evaluations
      : defaultForm.evaluations,
    dePopulationSize:
      calibrationSearch?.algorithm_id === "differential_evolution.rand1bin.v1"
        ? calibrationSearch.population_size
        : defaultForm.dePopulationSize,
    deGenerations:
      calibrationSearch?.algorithm_id === "differential_evolution.rand1bin.v1"
        ? calibrationSearch.generations
        : defaultForm.deGenerations,
    deMutationFactor:
      calibrationSearch?.algorithm_id === "differential_evolution.rand1bin.v1"
        ? calibrationSearch.mutation_factor
        : defaultForm.deMutationFactor,
    deCrossoverProbability:
      calibrationSearch?.algorithm_id === "differential_evolution.rand1bin.v1"
        ? calibrationSearch.crossover_probability
        : defaultForm.deCrossoverProbability,
    seed: calibrationSearch?.seed
      ?? config.task.training?.seed
      ?? 2026,
    mlpFeatureRoles: neuralModel.feature_roles ?? [...defaultForm.mlpFeatureRoles],
    laggedFeatures: (neuralModel.lagged_features ?? []).map((item) => ({
      source: item.source,
      lagSteps: item.lag_steps,
    })),
    mlpLookback: neuralModel.lookback ?? defaultForm.mlpLookback,
    s2sWarmup: neuralModel.s2s_warmup ?? 0,
    mlpHiddenLayers: (neuralModel.hidden_layers ?? [64, 32]).join(", "),
    mlpActivation: neuralModel.activation ?? "relu",
    mlpOutputActivation: neuralModel.output_activation ?? "linear",
    recurrentUnits: neuralModel.recurrent_units ?? defaultForm.recurrentUnits,
    recurrentLayers: (
      neuralModel.recurrent_layers
      ?? [neuralModel.recurrent_units ?? defaultForm.recurrentUnits]
    ).join(", "),
    cnnLayers: neuralModel.convolution_layers
      ? neuralModel.convolution_layers.map((layer) => ({
        filters: layer.filters,
        kernelSize: layer.kernel_size,
        activation: layer.activation,
      }))
      : defaultForm.cnnLayers.map((layer) => ({ ...layer })),
    trainStartIndex: config.task.period.start_index,
    trainEndIndex: config.task.period.train_end_index ?? defaultForm.trainEndIndex,
    validationEndIndex: config.task.period.validation_end_index ?? defaultForm.validationEndIndex,
    testEndIndex: config.task.period.test_end_index ?? null,
    trainingEpochs: config.task.training?.epochs ?? defaultForm.trainingEpochs,
    trainingBatchSize: config.task.training?.batch_size ?? defaultForm.trainingBatchSize,
    trainingLearningRate: config.task.training?.learning_rate ?? defaultForm.trainingLearningRate,
    trainingPatience: config.task.training?.patience ?? defaultForm.trainingPatience,
    trainingMinimumDelta: config.task.training?.minimum_delta ?? defaultForm.trainingMinimumDelta,
    trainingStride: config.task.stride ?? defaultForm.trainingStride,
    trainingGradientClipNorm: config.task.training?.gradient_clip_norm
      ?? neuralModel.gradient_clip_norm
      ?? defaultForm.trainingGradientClipNorm,
    initializationEnsembleEnabled: config.task.ensemble?.enabled
      ?? defaultForm.initializationEnsembleEnabled,
    initializationEnsembleMembers: config.task.ensemble?.members
      ?? defaultForm.initializationEnsembleMembers,
    pamlPhysicalModelId: (
      isPaml
        ? pamlBackbone
        : (
          physicalModel?.model_id === "exp_hydro.mass_conservative.v1"
          || physicalModel?.model_id === "hbv.mass_conservative_daily.v1"
            ? physicalModel.model_id
            : defaultForm.pamlPhysicalModelId
        )
    ) ?? defaultForm.pamlPhysicalModelId,
    pamlCorrectorId: defaultForm.pamlCorrectorId,
    pamlCorrectionStrategy: defaultForm.pamlCorrectionStrategy,
    pamlInputRoles: [...defaultForm.pamlInputRoles],
    discoveryProcessCount: config.model.process_count
      ?? defaultForm.discoveryProcessCount,
    discoveryHiddenUnits: config.model.hidden_units
      ?? defaultForm.discoveryHiddenUnits,
    discoveryTrainInputSwitches: config.model.train_input_switches
      ?? defaultForm.discoveryTrainInputSwitches,
    discoveryTrainProcessSwitches: config.model.train_process_switches
      ?? defaultForm.discoveryTrainProcessSwitches,
    discoveryDecorrelationWeight: config.model.loss?.decorrelation_weight
      ?? defaultForm.discoveryDecorrelationWeight,
    discoveryMassBalanceWeight: config.model.loss?.mass_balance_weight
      ?? defaultForm.discoveryMassBalanceWeight,
    discoverySmoothnessWeight: config.model.loss?.smoothness_weight
      ?? defaultForm.discoverySmoothnessWeight,
    discoverySmoothnessThreshold: config.model.loss?.smoothness_threshold
      ?? defaultForm.discoverySmoothnessThreshold,
    discoveryBinaryWeight: config.model.loss?.binary_weight
      ?? defaultForm.discoveryBinaryWeight,
    discoverySparsityWeight: config.model.loss?.sparsity_weight
      ?? defaultForm.discoverySparsityWeight,
    discoveryTargetCorrelation: config.model.loss?.target_process_correlation
      ?? defaultForm.discoveryTargetCorrelation,
  };
}

export function snowLabel(moduleId: SnowModuleId): string {
  if (moduleId === "snow.none.v1") return "No snow process";
  if (moduleId === "snow.degree_day.smooth.v1") return "Smooth degree-day";
  return "Degree-day";
}
