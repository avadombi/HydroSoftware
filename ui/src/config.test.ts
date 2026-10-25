import { describe, expect, it } from "vitest";
import {
  autoMapColumns,
  buildExperimentConfig,
  defaultForm,
  mappingIsComplete,
  workspaceFormFromConfig,
} from "./config";

describe("experiment configuration", () => {
  it("maps common hydrological CSV column names without duplicating roles", () => {
    const mapping = autoMapColumns(["date", "precipitation", "temperature", "pet", "q"]);
    expect(mapping).toEqual({
      timestamp: "date",
      precipitation: "precipitation",
      temperature: "temperature",
      potential_evapotranspiration: "pet",
      streamflow: "q",
    });
    expect(mappingIsComplete(mapping)).toBe(true);
  });

  it("builds the exact versioned service configuration", () => {
    const mapping = autoMapColumns(["date", "p", "t", "pet", "q"]);
    const config = buildExperimentConfig(
      {
        ...defaultForm,
        mapping,
        calibrationAlgorithm: "uniform_random_search.v1",
        evaluations: 25,
      },
      "data/imports/basin.csv",
    );

    expect(config.schema_version).toBe("1.20");
    expect(config.metadata.interface_version).toBe("1.0.0");
    expect(config.data.columns.timestamp).toBe("date");
    expect(config.model.snow?.module_id).toBe("snow.degree_day.hard.v1");
    expect(config.model.snow?.calibration?.degree_day_factor.mode).toBe("fixed");
    expect(config.task.search?.algorithm_id).toBe("uniform_random_search.v1");
    expect(
      config.task.search?.algorithm_id === "uniform_random_search.v1"
        ? config.task.search.n_evaluations
        : null,
    ).toBe(25);
    expect(config.data.processing.missing.streamflow).toBe("exclude");
    expect(config.data.processing.missing.numeric_sentinels).toEqual([]);
    expect(config.data.processing.units.streamflow).toBe("mm/day");
    expect(config.task.period).toMatchObject({
      protocol: "calibration_validation_test",
      start_index: 0,
      calibration_end_index: defaultForm.endIndex,
      validation_end_index: defaultForm.validationEndIndex,
      test_end_index: defaultForm.testEndIndex,
    });
    expect(config.task.objective?.objective_id).toBe(
      defaultForm.objective,
    );

    const reopened = workspaceFormFromConfig(config);
    expect(reopened.mapping).toEqual(mapping);
    expect(reopened.snowModule).toBe(defaultForm.snowModule);
    expect(reopened.evaluations).toBe(25);
    expect(reopened.conceptualProtocol).toBe(
      "calibration_validation_test",
    );
    expect(reopened.objective).toBe(defaultForm.objective);
  });

  it("round-trips differential evolution as the recommended calibration search", () => {
    const mapping = autoMapColumns(["date", "p", "t", "pet", "q"]);
    const config = buildExperimentConfig({
      ...defaultForm,
      mapping,
      dePopulationSize: 36,
      deGenerations: 18,
      deMutationFactor: 0.7,
      deCrossoverProbability: 0.85,
    }, "data/imports/basin.csv");

    expect(config.task.search).toEqual({
      algorithm_id: "differential_evolution.rand1bin.v1",
      generations: 18,
      population_size: 36,
      mutation_factor: 0.7,
      crossover_probability: 0.85,
      seed: defaultForm.seed,
      include_defaults: true,
      continue_on_candidate_error: true,
    });
    const reopened = workspaceFormFromConfig(config);
    expect(reopened.calibrationAlgorithm).toBe(
      "differential_evolution.rand1bin.v1",
    );
    expect(reopened.dePopulationSize).toBe(36);
    expect(reopened.deGenerations).toBe(18);
  });

  it("round-trips calibrated snow-parameter bounds", () => {
    const mapping = autoMapColumns(["date", "p", "t", "pet", "q"]);
    const config = buildExperimentConfig({
      ...defaultForm,
      mapping,
      snowCalibration: {
        ...defaultForm.snowCalibration,
        degreeDayFactor: { mode: "calibrate", lowerBound: 0.5, upperBound: 4.5 },
      },
    }, "data/imports/basin.csv");

    expect(config.model.snow?.calibration?.degree_day_factor).toEqual({
      mode: "calibrate",
      lower_bound: 0.5,
      upper_bound: 4.5,
    });
    expect(workspaceFormFromConfig(config).snowCalibration.degreeDayFactor).toEqual({
      mode: "calibrate",
      lowerBound: 0.5,
      upperBound: 4.5,
    });
  });

  it("round-trips the daily HBV variant with both response-zone states", () => {
    const mapping = autoMapColumns(["date", "p", "t", "pet", "q"]);
    const config = buildExperimentConfig({
      ...defaultForm,
      modelId: "hbv.mass_conservative_daily.v1",
      initialUpperZoneStorage: 25,
      initialLowerZoneStorage: 90,
      mapping,
    }, "data/imports/basin.csv");

    expect(config.model.model_id).toBe("hbv.mass_conservative_daily.v1");
    expect(config.model.initial_state).toEqual([0, 300, 25, 90]);
    expect(config.model.substeps).toBeUndefined();
    const reopened = workspaceFormFromConfig(config);
    expect(reopened.modelId).toBe("hbv.mass_conservative_daily.v1");
    expect(reopened.initialUpperZoneStorage).toBe(25);
    expect(reopened.initialLowerZoneStorage).toBe(90);
  });

  it("round-trips discrete MAXBAS treatment and routing memory", () => {
    const mapping = autoMapColumns(["date", "p", "t", "pet", "q"]);
    const config = buildExperimentConfig({
      ...defaultForm,
      modelId: "hbv.mass_conservative_daily.v1",
      routingProcess: "routing.maxbas.triangular.v1",
      maxbas: 4,
      maxbasTreatment: { mode: "calibrate", lowerBound: 1, upperBound: 7 },
      initialRoutingState: [0.2, 0.1, 0, 0, 0, 0],
      mapping,
    }, "data/imports/basin.csv");

    expect(config.model.routing).toEqual({
      process_id: "routing.maxbas.triangular.v1",
      maxbas: 4,
      calibration: {
        maxbas: { mode: "calibrate", lower_bound: 1, upper_bound: 7 },
      },
      initial_state: [0.2, 0.1, 0, 0, 0, 0],
    });
    const reopened = workspaceFormFromConfig(config);
    expect(reopened.routingProcess).toBe("routing.maxbas.triangular.v1");
    expect(reopened.maxbas).toBe(4);
    expect(reopened.maxbasTreatment.mode).toBe("calibrate");
    expect(reopened.initialRoutingState).toEqual([0.2, 0.1, 0, 0, 0, 0]);
  });

  it("round-trips the leakage-safe MLP training setup", () => {
    const mapping = autoMapColumns(["date", "p", "t", "pet", "q"]);
    const config = buildExperimentConfig({
      ...defaultForm,
      modelId: "mlp.tensorflow_s2v.v1",
      mapping,
      mlpFeatureRoles: ["precipitation", "temperature"],
      mlpLookback: 14,
      mlpHiddenLayers: "32, 16",
      trainEndIndex: 500,
      validationEndIndex: 700,
      testEndIndex: 900,
      trainingEpochs: 80,
      objective: "kling_gupta_efficiency_2009",
    }, "data/imports/basin.csv");

    expect(config.model.model_id).toBe("mlp.tensorflow_s2v.v1");
    expect(config.model.hidden_layers).toEqual([32, 16]);
    expect(config.task.task_type).toBe("training");
    expect(config.task.period.validation_end_index).toBe(700);
    expect(config.task.training?.epochs).toBe(80);
    expect(config.task.objective?.objective_id).toBe(
      "kling_gupta_efficiency_2009",
    );
    const reopened = workspaceFormFromConfig(config);
    expect(reopened.mlpFeatureRoles).toEqual(["precipitation", "temperature"]);
    expect(reopened.mlpLookback).toBe(14);
    expect(reopened.mlpHiddenLayers).toBe("32, 16");
    expect(reopened.objective).toBe("kling_gupta_efficiency_2009");
  });

  it("round-trips causal S2S alignment and warm-up without a future lead", () => {
    const mapping = autoMapColumns(["date", "p", "t", "pet", "q"]);
    const config = buildExperimentConfig({
      ...defaultForm,
      modelId: "gru.tensorflow_stacked_s2s.v3",
      mapping,
      mlpLookback: 18,
      s2sWarmup: 6,
      recurrentLayers: "24, 12",
      trainingStride: 7,
    }, "data/imports/basin.csv");

    expect(config.model.model_id).toBe("gru.tensorflow_stacked_s2s.v3");
    expect(config.model.lookback).toBe(18);
    expect(config.model.s2s_warmup).toBe(6);
    expect(config.model.recurrent_layers).toEqual([24, 12]);
    expect(config.task.lead_steps).toBe(0);
    expect(config.task.stride).toBe(1);
    const reopened = workspaceFormFromConfig(config);
    expect(reopened.modelId).toBe("gru.tensorflow_stacked_s2s.v3");
    expect(reopened.s2sWarmup).toBe(6);
    expect(reopened.recurrentLayers).toBe("24, 12");
  });

  it("round-trips neural initialization ensembles without affecting DD", () => {
    const mapping = autoMapColumns(["date", "p", "t", "pet", "q"]);
    const config = buildExperimentConfig({
      ...defaultForm,
      modelId: "mlp.tensorflow_s2v.v1",
      mapping,
      trainEndIndex: 500,
      validationEndIndex: 700,
      testEndIndex: 900,
      initializationEnsembleEnabled: true,
      initializationEnsembleMembers: 7,
    }, "data/imports/basin.csv");

    expect(config.task.ensemble).toEqual({
      enabled: true,
      members: 7,
      aggregation: "arithmetic_mean",
      lower_quantile: 0.1,
      upper_quantile: 0.9,
    });
    const reopened = workspaceFormFromConfig(config);
    expect(reopened.initializationEnsembleEnabled).toBe(true);
    expect(reopened.initializationEnsembleMembers).toBe(7);

    const discovery = buildExperimentConfig({
      ...defaultForm,
      modelId: "deepdiscover.numpy_instantaneous_s2s.v1",
      mapping,
      trainEndIndex: 500,
      validationEndIndex: 700,
      testEndIndex: 900,
      initializationEnsembleEnabled: true,
    }, "data/imports/basin.csv");
    expect(discovery.task.ensemble).toBeUndefined();
  });

  it("round-trips the leakage-safe stacked LSTM training setup", () => {
    const mapping = autoMapColumns(["date", "p", "t", "pet", "q"]);
    const config = buildExperimentConfig({
      ...defaultForm,
      modelId: "lstm.tensorflow_stacked_s2v.v2",
      mapping,
      mlpFeatureRoles: ["precipitation", "potential_evapotranspiration"],
      mlpLookback: 21,
      recurrentLayers: "48, 24",
      trainingGradientClipNorm: 3,
      trainEndIndex: 500,
      validationEndIndex: 700,
      testEndIndex: 900,
    }, "data/imports/basin.csv");

    expect(config.model.model_id).toBe("lstm.tensorflow_stacked_s2v.v2");
    expect(config.model.recurrent_units).toBe(48);
    expect(config.model.recurrent_layers).toEqual([48, 24]);
    expect(config.model.gradient_clip_norm).toBe(3);
    const reopened = workspaceFormFromConfig(config);
    expect(reopened.modelId).toBe("lstm.tensorflow_stacked_s2v.v2");
    expect(reopened.mlpLookback).toBe(21);
    expect(reopened.recurrentUnits).toBe(48);
    expect(reopened.recurrentLayers).toBe("48, 24");
    expect(reopened.trainingGradientClipNorm).toBe(3);
  });

  it("round-trips the leakage-safe stacked GRU training setup", () => {
    const mapping = autoMapColumns(["date", "p", "t", "pet", "q"]);
    const config = buildExperimentConfig({
      ...defaultForm,
      modelId: "gru.tensorflow_stacked_s2v.v2",
      mapping,
      mlpFeatureRoles: ["precipitation", "temperature"],
      mlpLookback: 18,
      recurrentLayers: "40, 20, 10",
      trainingGradientClipNorm: 2.5,
      trainEndIndex: 500,
      validationEndIndex: 700,
      testEndIndex: 900,
    }, "data/imports/basin.csv");

    expect(config.model.model_id).toBe("gru.tensorflow_stacked_s2v.v2");
    expect(config.model.recurrent_units).toBe(40);
    expect(config.model.recurrent_layers).toEqual([40, 20, 10]);
    expect(config.model.gradient_clip_norm).toBe(2.5);
    const reopened = workspaceFormFromConfig(config);
    expect(reopened.modelId).toBe("gru.tensorflow_stacked_s2v.v2");
    expect(reopened.mlpLookback).toBe(18);
    expect(reopened.recurrentUnits).toBe(40);
    expect(reopened.recurrentLayers).toBe("40, 20, 10");
  });

  it("round-trips a configurable causal CNN-1D layer stack", () => {
    const mapping = autoMapColumns(["date", "p", "t", "pet", "q"]);
    const config = buildExperimentConfig({
      ...defaultForm,
      modelId: "cnn1d.tensorflow_s2v.v1",
      mapping,
      mlpFeatureRoles: ["precipitation", "temperature"],
      mlpLookback: 20,
      cnnLayers: [
        { filters: 24, kernelSize: 5, activation: "relu" },
        { filters: 12, kernelSize: 3, activation: "tanh" },
      ],
      trainingGradientClipNorm: 4,
      trainEndIndex: 500,
      validationEndIndex: 700,
      testEndIndex: 900,
    }, "data/imports/basin.csv");

    expect(config.model.model_id).toBe("cnn1d.tensorflow_s2v.v1");
    expect(config.model.convolution_layers).toEqual([
      { filters: 24, kernel_size: 5, activation: "relu" },
      { filters: 12, kernel_size: 3, activation: "tanh" },
    ]);
    expect(config.model.gradient_clip_norm).toBe(4);
    const reopened = workspaceFormFromConfig(config);
    expect(reopened.modelId).toBe("cnn1d.tensorflow_s2v.v1");
    expect(reopened.cnnLayers).toEqual([
      { filters: 24, kernelSize: 5, activation: "relu" },
      { filters: 12, kernelSize: 3, activation: "tanh" },
    ]);
  });

  it("round-trips additional predictors and causal lag recipes", () => {
    const mapping = autoMapColumns(["date", "p", "t", "pet", "q", "rh"]);
    const config = buildExperimentConfig({
      ...defaultForm,
      modelId: "mlp.tensorflow_s2v.v1",
      mapping,
      additionalPredictors: [{
        name: "relative_humidity",
        column: "rh",
        unit: "%",
        missing: "linear_interpolation",
      }],
      mlpFeatureRoles: ["precipitation", "relative_humidity"],
      laggedFeatures: [
        { source: "relative_humidity", lagSteps: 2 },
        { source: "precipitation", lagSteps: 7 },
      ],
    }, "data/imports/basin.csv");

    expect(config.data.columns.additional_predictors).toEqual([{
      name: "relative_humidity",
      column: "rh",
      unit: "%",
      missing: "linear_interpolation",
    }]);
    expect(config.model.feature_roles).toEqual(["precipitation", "relative_humidity"]);
    expect(config.model.lagged_features).toEqual([
      { source: "relative_humidity", lag_steps: 2 },
      { source: "precipitation", lag_steps: 7 },
    ]);
    const reopened = workspaceFormFromConfig(config);
    expect(reopened.additionalPredictors).toEqual([{
      name: "relative_humidity",
      column: "rh",
      unit: "%",
      missing: "linear_interpolation",
    }]);
    expect(reopened.laggedFeatures).toEqual([
      { source: "relative_humidity", lagSteps: 2 },
      { source: "precipitation", lagSteps: 7 },
    ]);
  });

  it("round-trips single-DU DeepDiscover dimensions, switches, and losses", () => {
    const mapping = autoMapColumns(["date", "p", "t", "pet", "q"]);
    const config = buildExperimentConfig({
      ...defaultForm,
      modelId: "deepdiscover.numpy_instantaneous_s2s.v1",
      mapping,
      snowModule: "snow.degree_day.hard.v1",
      initialSnowStorage: 0,
      initialSoilStorage: 18,
      discoveryProcessCount: 4,
      discoveryHiddenUnits: 20,
      discoveryTrainInputSwitches: false,
      discoveryTrainProcessSwitches: true,
      discoverySmoothnessThreshold: 0.25,
      snowCalibration: {
        ...defaultForm.snowCalibration,
        degreeDayFactor: { mode: "calibrate", lowerBound: 0.75, upperBound: 6 },
      },
      trainEndIndex: 500,
      validationEndIndex: 700,
      testEndIndex: 900,
      warmupSteps: 30,
      trainingGradientClipNorm: 3,
      objective: "mean_square_error",
    }, "data/imports/basin.csv");

    expect(config.model.model_id).toBe("deepdiscover.numpy_instantaneous_s2s.v1");
    expect(config.model.initial_state).toEqual([0, 18]);
    expect(config.model.process_count).toBe(4);
    expect(config.model.hidden_units).toBe(20);
    expect(config.model.train_input_switches).toBe(false);
    expect(config.model.loss?.smoothness_threshold).toBe(0.25);
    expect(config.model.snow?.calibration?.degree_day_factor).toEqual({
      mode: "calibrate",
      lower_bound: 0.75,
      upper_bound: 6,
    });
    expect(config.task.task_type).toBe("discovery_training");
    expect(config.task.warmup_steps).toBe(30);
    expect(config.task.training?.optimizer_id).toBe("adam.numpy.full_sequence.v1");
    expect(config.task.training?.gradient_clip_norm).toBe(3);
    expect(config.task.objective?.objective_id).toBe("mean_square_error");

    const reopened = workspaceFormFromConfig(config);
    expect(reopened.modelId).toBe("deepdiscover.numpy_instantaneous_s2s.v1");
    expect(reopened.discoveryProcessCount).toBe(4);
    expect(reopened.discoveryHiddenUnits).toBe(20);
    expect(reopened.discoveryTrainInputSwitches).toBe(false);
    expect(reopened.discoverySmoothnessThreshold).toBe(0.25);
    expect(reopened.snowCalibration.degreeDayFactor).toEqual({
      mode: "calibrate",
      lowerBound: 0.75,
      upperBound: 6,
    });
    expect(reopened.warmupSteps).toBe(30);
  });


  it("round-trips joint PaML with a selectable P-RNN backbone", () => {
    const config = buildExperimentConfig({
      ...defaultForm,
      modelId: "paml.tensorflow_prnn_cnn_s2s.v1",
      pamlPhysicalModelId: "hbv.mass_conservative_daily.v1",
      cnnLayers: [{ filters: 12, kernelSize: 7, activation: "relu" }],
      trainEndIndex: 500,
      validationEndIndex: 700,
      testEndIndex: 900,
      warmupSteps: 30,
    }, "data/imports/basin.csv");

    expect(config.model.model_id).toBe("paml.tensorflow_prnn_cnn_s2s.v1");
    expect(config.model.physical_model?.model_id).toBe("hbv.mass_conservative_daily.v1");
    expect(config.model.cnn_filters).toBe(12);
    expect(config.model.cnn_kernel_size).toBe(7);
    expect(config.model.training_strategy).toBe("joint_end_to_end");
    expect(config.task.task_type).toBe("paml_training");
    expect(config.task.warmup_steps).toBe(30);

    const reopened = workspaceFormFromConfig(config);
    expect(reopened.modelId).toBe("paml.tensorflow_prnn_cnn_s2s.v1");
    expect(reopened.pamlPhysicalModelId).toBe("hbv.mass_conservative_daily.v1");
  });
});
