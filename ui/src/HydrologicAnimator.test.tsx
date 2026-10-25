import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";
import { HydrologicAnimator } from "./HydrologicAnimator";
import type { RunResult } from "./types";


const result: RunResult = {
  task_id: "run-test",
  state: "completed",
  experiment_id: "test",
  configuration_hash: "abc",
  validation: {
    experiment_id: "test",
    configuration_hash: "abc",
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
    records: [],
    best_parameters: {
      recession_factor: 0.05,
      maximum_storage: 500,
      maximum_subsurface_flow: 30,
    },
    best_score: 0.9,
    warnings: [],
  },
  best_simulation: {
    outputs: [[1.2], [1.4]],
    states: [[0, 300], [2, 301], [1, 302]],
    fluxes: [
      [5, 3, 2, 0, 1, 1.1, 0.1],
      [4, 4, 0, 1, 1, 1.2, 0.2],
    ],
    output_names: ["streamflow"],
    state_names: ["snow_storage", "soil_storage"],
    flux_names: ["precipitation", "rain", "snowfall", "snowmelt", "actual_evapotranspiration", "subsurface_flow", "surface_flow"],
    diagnostics: { maximum_absolute_mass_balance_error: 0 },
  },
  timestamps: ["2020-01-01", "2020-01-02"],
  observations: [1.1, 1.5],
  forcings: {
    precipitation: [5, 4],
    temperature: [-2, 2],
    potential_evapotranspiration: [1, 1],
  },
};


describe("hydrological process animation", () => {
  it("states the interval and boundary timing and exposes flux values", () => {
    const markup = renderToStaticMarkup(<HydrologicAnimator result={result} />);
    expect(markup).toContain("Fluxes belong to interval");
    expect(markup).toContain("Snowfall 2.000");
    expect(markup).toContain("Soil storage");
    expect(markup).toContain("2020-01-01");
  });

  it("uses a dedicated two-column groundwater summary", () => {
    const groundwaterResult: RunResult = {
      ...result,
      best_simulation: {
        ...result.best_simulation,
        states: [[0, 300, 40], [2, 301, 42], [1, 302, 43]],
        state_names: ["snow_storage", "soil_storage", "groundwater_storage"],
        fluxes: [
          [5, 3, 2, 0, 1, 0.1, 0.6, 0.5],
          [4, 4, 0, 1, 1, 0.2, 0.7, 0.5],
        ],
        flux_names: ["precipitation", "rain", "snowfall", "snowmelt", "actual_evapotranspiration", "direct_runoff", "groundwater_recharge", "groundwater_discharge"],
      },
    };
    const markup = renderToStaticMarkup(<HydrologicAnimator result={groundwaterResult} />);
    expect(markup).toContain("groundwater-summary");
    expect(markup).toContain("Groundwater storage");
    expect(markup).toContain("Recharge");
  });

  it("shows both HBV response zones, percolation, and three runoff components", () => {
    const hbvResult: RunResult = {
      ...result,
      calibration: {
        ...result.calibration!,
        best_parameters: {
          field_capacity: 500,
          recharge_exponent: 2,
          evap_reduction_threshold: 0.7,
          maximum_percolation: 1.5,
          upper_zone_threshold: 30,
          fast_recession: 0.3,
          intermediate_recession: 0.1,
          slow_recession: 0.02,
        },
      },
      best_simulation: {
        ...result.best_simulation,
        states: [[0, 300, 20, 80], [2, 301, 22, 81], [1, 302, 23, 82]],
        state_names: ["snow_storage", "soil_storage", "upper_zone_storage", "lower_zone_storage"],
        fluxes: [
          [5, 3, 2, 0, 3, 0, 0.6, 1, 0.5, 0.2, 0.3, 0.2, 0.7],
          [4, 4, 0, 1, 5, 0, 0.7, 1, 0.5, 0.3, 0.4, 0.2, 0.9],
        ],
        flux_names: [
          "precipitation", "rain", "snowfall", "snowmelt", "soil_water_input",
          "soil_excess", "groundwater_recharge", "actual_evapotranspiration",
          "percolation", "fast_flow", "intermediate_flow", "baseflow",
          "unrouted_streamflow",
        ],
      },
    };
    const markup = renderToStaticMarkup(<HydrologicAnimator result={hbvResult} />);
    expect(markup).toContain("hbv-response-summary");
    expect(markup).toContain("Upper-zone storage");
    expect(markup).toContain("Lower-zone storage");
    expect(markup).toContain("Percolation");
    expect(markup).toContain("Fast flow Q0");
    expect(markup).toContain("Intermediate Q1");
    expect(markup).toContain("Baseflow Q2");
  });

  it("shows response inflow, pending routing storage, and routed discharge", () => {
    const routedResult: RunResult = {
      ...result,
      best_simulation: {
        ...result.best_simulation,
        states: [
          [0, 300, 20, 80, 0, 0, 0, 0, 0, 0],
          [0, 301, 21, 81, 0.4, 0.2, 0, 0, 0, 0],
          [0, 302, 22, 82, 0.2, 0, 0, 0, 0, 0],
        ],
        state_names: [
          "snow_storage", "soil_storage", "upper_zone_storage", "lower_zone_storage",
          "routing_pending_day_1", "routing_pending_day_2", "routing_pending_day_3",
          "routing_pending_day_4", "routing_pending_day_5", "routing_pending_day_6",
        ],
        fluxes: [
          [5, 3, 2, 0, 3, 0, 0.6, 1, 0.5, 0.2, 0.3, 0.2, 0.7, 0.1],
          [4, 4, 0, 1, 5, 0, 0.7, 1, 0.5, 0.3, 0.4, 0.2, 0.9, 0.6],
        ],
        flux_names: [
          "precipitation", "rain", "snowfall", "snowmelt", "soil_water_input",
          "soil_excess", "groundwater_recharge", "actual_evapotranspiration",
          "percolation", "fast_flow", "intermediate_flow", "baseflow",
          "unrouted_streamflow", "routed_streamflow",
        ],
      },
    };
    const markup = renderToStaticMarkup(<HydrologicAnimator result={routedResult} />);
    expect(markup).toContain("routing-summary");
    expect(markup).toContain("HBV response discharge");
    expect(markup).toContain("Routing storage");
    expect(markup).toContain("Routed discharge");
  });

  it("labels discovered storage and every discovered process without conceptual placeholders", () => {
    const discoveryResult: RunResult = {
      ...result,
      calibration: null,
      model_id: "deepdiscover.numpy_instantaneous_s2s.v1",
      best_simulation: {
        ...result.best_simulation,
        states: [[0, 18], [0, 17], [0, 16]],
        state_names: ["snow_storage", "discovered_storage"],
        fluxes: [
          [5, 5, 0, 0, 0.4, 0.7, 0.5],
          [4, 4, 0, 0, 0.3, 0.8, 0.6],
        ],
        flux_names: [
          "precipitation", "rain", "snowfall", "snowmelt",
          "actual_evapotranspiration", "discovered_process_1",
          "discovered_process_2",
        ],
      },
    };
    const markup = renderToStaticMarkup(
      <HydrologicAnimator result={discoveryResult} />,
    );
    expect(markup).toContain("Discovered storage");
    expect(markup).toContain("run maximum");
    expect(markup).toContain("Discovered Q1");
    expect(markup).toContain("Discovered Q2");
    expect(markup).toContain("discovery-flows");
    expect(markup).not.toContain("Subsurface");
    expect(markup).not.toContain("Surface");
  });

  it("keeps physical and corrected PaML discharge distinct and uses trained physical capacity", () => {
    const pamlResult: RunResult = {
      ...result,
      calibration: null,
      training: {
        engine_id: "paml.tensorflow_prnn_cnn.v1",
        model_id: "paml.tensorflow_prnn_cnn_s2s.v1",
        physical_model_id: "exp_hydro.mass_conservative.v1",
        training_strategy: "joint_end_to_end",
        physical_parameters: { maximum_storage: 600 },
        artifact: {},
        history: [],
        best_epoch: 1,
        epochs_completed: 1,
        stopped_early: false,
        selection_metric: "validation_nash_sutcliffe_efficiency",
        best_validation_score: 0.5,
        partition_metrics: {},
      },
      best_simulation: {
        ...result.best_simulation,
        outputs: [[1.3, 1.0], [1.5, 1.2]],
        output_names: ["streamflow", "physical_streamflow"],
      },
    };
    const markup = renderToStaticMarkup(<HydrologicAnimator result={pamlResult} />);
    expect(markup).toContain("paml-discharge-summary");
    expect(markup).toContain("Physical streamflow");
    expect(markup).toContain("Corrected streamflow");
    expect(markup).toContain('class="physical"');
    expect(markup).toContain("PaML output");
    expect(markup).toContain("50.1667% of trained physical capacity");
    expect(markup).not.toContain("100.0000% of calibrated capacity");
  });
});
