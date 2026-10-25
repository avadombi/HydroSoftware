import { useEffect, useMemo, useState } from "react";
import {
  ChevronLeft,
  ChevronRight,
  CloudRain,
  Droplets,
  Pause,
  Play,
  Snowflake,
  SunMedium,
  Waves,
} from "lucide-react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { RunResult } from "./types";
import { downsampleForChart } from "./chartData";


function finite(value: number | null | undefined): number {
  return typeof value === "number" && Number.isFinite(value) ? value : 0;
}

function smoothHeaviside(value: number): number {
  return (Math.tanh(5 * value) + 1) / 2;
}

function formatHydrologicValue(value: number, decimalPlaces: number): string {
  const rounded = value.toFixed(decimalPlaces);
  return Number(rounded) === 0 ? (0).toFixed(decimalPlaces) : rounded;
}

export function HydrologicAnimator({ result, decimalPlaces = 4 }: { result: RunResult; decimalPlaces?: number }) {
  const hydrologicValue = (value: number) => formatHydrologicValue(value, decimalPlaces);
  const hasObservations = result.observations !== null
    && result.observations.some((value) => value !== null && Number.isFinite(value));
  const nIntervals = result.best_simulation.outputs.length;
  const [index, setIndex] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);

  const fluxIndex = useMemo(
    () => new Map(result.best_simulation.flux_names.map((name, position) => [name, position])),
    [result],
  );
  const stateIndex = useMemo(
    () => new Map(result.best_simulation.state_names.map((name, position) => [name, position])),
    [result],
  );
  const correctedOutputPosition = Math.max(
    0,
    result.best_simulation.output_names.indexOf("streamflow"),
  );
  const physicalOutputPosition = result.best_simulation.output_names.indexOf(
    "physical_streamflow",
  );
  const hasPhysicalOutput = physicalOutputPosition >= 0;
  const trainedPhysicalParameters = result.training?.physical_parameters ?? {};
  const isPaml = result.training?.training_strategy === "joint_end_to_end";
  const legacyPamlSnowFluxes = useMemo(() => {
    if (!isPaml || fluxIndex.has("precipitation")) return null;
    const precipitation = result.forcings.precipitation ?? [];
    const temperature = result.forcings.temperature ?? [];
    if (precipitation.length !== nIntervals || temperature.length !== nIntervals) return null;
    const snowfallThreshold = Number(
      trainedPhysicalParameters.snow_temperature_threshold
        ?? trainedPhysicalParameters.tmin,
    );
    const meltThreshold = Number(
      trainedPhysicalParameters.melt_temperature_threshold
        ?? trainedPhysicalParameters.tmax,
    );
    const degreeDayFactor = Number(
      trainedPhysicalParameters.degree_day_factor
        ?? trainedPhysicalParameters.ddf,
    );
    if (![snowfallThreshold, meltThreshold, degreeDayFactor].every(Number.isFinite)) return null;
    const snowfall: number[] = [];
    const rain: number[] = [];
    const snowmelt: number[] = [];
    for (let position = 0; position < nIntervals; position += 1) {
      const p = finite(precipitation[position]);
      const t = finite(temperature[position]);
      const snowBefore = finite(result.best_simulation.states[position]?.[0]);
      snowfall.push(smoothHeaviside(snowfallThreshold - t) * p);
      rain.push(smoothHeaviside(t - snowfallThreshold) * p);
      snowmelt.push(
        smoothHeaviside(t - meltThreshold)
          * smoothHeaviside(snowBefore)
          * Math.min(snowBefore, degreeDayFactor * Math.max(t - meltThreshold, 0)),
      );
    }
    return { precipitation: [...precipitation], snowfall, rain, snowmelt };
  }, [fluxIndex, isPaml, nIntervals, result, trainedPhysicalParameters]);
  const trajectory = useMemo(() => {
    const rows = result.best_simulation.outputs.map((row, position) => ({
      index: position,
      time: result.timestamps?.[position] ?? String(position + 1),
      observed: result.observations !== null && result.observations[position] !== null && Number.isFinite(result.observations[position]) ? result.observations[position] : null,
      simulated: row[correctedOutputPosition],
      physical: hasPhysicalOutput ? row[physicalOutputPosition] : null,
    }));
    return downsampleForChart(
      rows,
      hasPhysicalOutput
        ? ["observed", "physical", "simulated"]
        : ["observed", "simulated"],
      700,
    );
  }, [correctedOutputPosition, hasPhysicalOutput, physicalOutputPosition, result]);
  const maxima = useMemo(() => {
    const values: Record<string, number> = {};
    result.best_simulation.flux_names.forEach((name) => { values[name] = 0; });
    result.best_simulation.fluxes.forEach((fluxRow) => fluxRow.forEach((value, position) => {
      const name = result.best_simulation.flux_names[position];
      if (name !== undefined) values[name] = Math.max(values[name], finite(value));
    }));
    if (legacyPamlSnowFluxes !== null) {
      (["precipitation", "rain", "snowfall", "snowmelt"] as const).forEach((name) => {
        const series = legacyPamlSnowFluxes[name];
        values[name] = series.reduce((maximum, value) => Math.max(maximum, finite(value)), 0);
      });
    }
    return values;
  }, [legacyPamlSnowFluxes, result]);
  const maximumSnowStorage = useMemo(
    () => result.best_simulation.states.reduce(
      (maximum, row) => Math.max(maximum, finite(row[0])),
      0,
    ),
    [result],
  );

  useEffect(() => {
    setIndex(0);
    setPlaying(false);
  }, [result.task_id]);

  useEffect(() => {
    if (!playing || nIntervals < 2) return;
    const timer = window.setInterval(() => {
      setIndex((current) => {
        if (current >= nIntervals - 1) {
          setPlaying(false);
          return current;
        }
        return current + 1;
      });
    }, 700 / speed);
    return () => window.clearInterval(timer);
  }, [playing, speed, nIntervals]);

  const row = result.best_simulation.fluxes[index] ?? [];
  const stateBefore = result.best_simulation.states[index] ?? [0, 0];
  const stateAfter = result.best_simulation.states[index + 1] ?? stateBefore;
  const flux = (name: string) => {
    const position = fluxIndex.get(name);
    if (position !== undefined) return finite(row[position]);
    if (legacyPamlSnowFluxes === null) return 0;
    if (name === "precipitation") return finite(legacyPamlSnowFluxes.precipitation[index]);
    if (name === "snowfall") return finite(legacyPamlSnowFluxes.snowfall[index]);
    if (name === "snowmelt") return finite(legacyPamlSnowFluxes.snowmelt[index]);
    if (name === "rain") return finite(legacyPamlSnowFluxes.rain[index]);
    return 0;
  };
  const relativeFlux = (name: string) => {
    const maximum = maxima[name] ?? 0;
    return maximum > 0 ? Math.max(5, 100 * flux(name) / maximum) : 0;
  };
  const calibratedParameters = result.calibration?.best_parameters ?? {};
  const isDiscovery = stateIndex.has("discovered_storage")
    && result.best_simulation.flux_names.some(
      (name) => name.startsWith("discovered_process_"),
    );
  const discoveredStoragePosition = stateIndex.get("discovered_storage");
  const maximumDiscoveredStorage = discoveredStoragePosition === undefined
    ? 0
    : result.best_simulation.states.reduce(
      (maximum, state) => Math.max(
        maximum,
        finite(state[discoveredStoragePosition]),
      ),
      0,
    );
  const maximumStorage = (isDiscovery ? maximumDiscoveredStorage : 0)
    || finite(trainedPhysicalParameters.maximum_storage)
    || finite(trainedPhysicalParameters.fc)
    || finite(calibratedParameters.maximum_storage)
    || finite(calibratedParameters.retention_limit)
    || finite(calibratedParameters.field_capacity)
    || finite(calibratedParameters.maximum_soil_storage)
    || 1;
  const snowLevel = maximumSnowStorage > 0 ? 100 * finite(stateAfter[0]) / maximumSnowStorage : 0;
  const soilLevel = Math.min(100, 100 * finite(stateAfter[1]) / maximumStorage);
  const groundwaterPosition = stateIndex.get("groundwater_storage");
  const groundwaterStorage = groundwaterPosition === undefined ? null : finite(stateAfter[groundwaterPosition]);
  const upperZonePosition = stateIndex.get("upper_zone_storage");
  const lowerZonePosition = stateIndex.get("lower_zone_storage");
  const upperZoneStorage = upperZonePosition === undefined ? null : finite(stateAfter[upperZonePosition]);
  const lowerZoneStorage = lowerZonePosition === undefined ? null : finite(stateAfter[lowerZonePosition]);
  const isHbv = upperZoneStorage !== null && lowerZoneStorage !== null;
  const routingPositions = result.best_simulation.state_names
    .map((name, position) => name.startsWith("routing_pending_day_") ? position : -1)
    .filter((position) => position >= 0);
  const routingStorage = routingPositions.length
    ? routingPositions.reduce((total, position) => total + finite(stateAfter[position]), 0)
    : null;
  const hasRouting = routingStorage !== null && fluxIndex.has("routed_streamflow");
  const subsurfaceName = fluxIndex.has("subsurface_flow") ? "subsurface_flow" : "groundwater_discharge";
  const surfaceName = fluxIndex.has("surface_flow") ? "surface_flow" : "direct_runoff";
  const discoveredProcessNames = result.best_simulation.flux_names.filter(
    (name) => name.startsWith("discovered_process_"),
  );
  const legacyHbvPamlFluxes = isPaml
    && isHbv
    && fluxIndex.has("upper_zone_flow")
    && !fluxIndex.has("fast_flow");
  const outflowItems = isDiscovery
    ? [
      ["Actual ET", "actual_evapotranspiration"],
      ...discoveredProcessNames.map((name, position) => [
        `Discovered Q${position + 1}`,
        name,
      ]),
    ]
    : isHbv
    ? legacyHbvPamlFluxes
      ? [
        ["Actual ET", "actual_evapotranspiration"],
        ["Upper-zone flow Q0 + Q1", "upper_zone_flow"],
        ["Baseflow Q2", "baseflow"],
      ]
      : [
        ["Actual ET", "actual_evapotranspiration"],
        ["Fast flow Q0", "fast_flow"],
        ["Intermediate Q1", "intermediate_flow"],
        ["Baseflow Q2", "baseflow"],
      ]
    : [
        ["Actual ET", "actual_evapotranspiration"],
        [groundwaterStorage === null ? "Subsurface" : "Groundwater discharge", subsurfaceName],
        [groundwaterStorage === null ? "Surface" : "Direct runoff", surfaceName],
      ];
  const simulated = finite(
    result.best_simulation.outputs[index]?.[correctedOutputPosition],
  );
  const physical = hasPhysicalOutput
    ? finite(result.best_simulation.outputs[index]?.[physicalOutputPosition])
    : null;
  const observed = result.observations?.[index];
  const time = result.timestamps?.[index] ?? `Row ${index + 1}`;

  const move = (next: number) => {
    setPlaying(false);
    setIndex(Math.max(0, Math.min(nIntervals - 1, next)));
  };

  const toggle = () => {
    if (index >= nIntervals - 1) setIndex(0);
    setPlaying((current) => !current);
  };

  return (
    <section className="card hydrologic-animator" aria-label="Synchronized hydrological animation">
      <div className="animator-heading">
        <div>
          <p className="section-kicker">Synchronized process view</p>
          <h2>Reservoirs, fluxes, and hydrograph</h2>
          <p>Fluxes belong to interval <em>t</em>; displayed storages are the boundary states after that interval, at <em>t + 1</em>.</p>
        </div>
        <div className="animation-time">
          <span>Interval {index + 1} of {nIntervals}</span>
          <strong>{time}</strong>
        </div>
      </div>

      <div className="animation-layout">
        <div className="process-animation">
          <div className="forcing-strip">
            <div><CloudRain size={17} /><span>Precipitation</span><strong>{hydrologicValue(flux("precipitation"))} mm</strong></div>
            <div><Snowflake size={17} /><span>Temperature</span><strong>{hydrologicValue(finite(result.forcings.temperature[index]))} °C</strong></div>
            <div><SunMedium size={17} /><span>Potential ET</span><strong>{hydrologicValue(finite(result.forcings.potential_evapotranspiration[index]))} mm</strong></div>
          </div>

          <div className="animated-system">
            <div className="animated-reservoir snow-reservoir">
              <div className="animated-water snow-water" style={{ height: `${Math.min(100, snowLevel)}%` }} />
              <div className="reservoir-reading"><Snowflake size={18} /><span>Snow storage</span><strong>{hydrologicValue(finite(stateAfter[0]))} mm</strong><small>before {hydrologicValue(finite(stateBefore[0]))} mm</small></div>
            </div>

            <div className="process-transfer">
              <div className="flux-track"><span style={{ width: `${relativeFlux("snowfall")}%` }} /><small>Snowfall {hydrologicValue(flux("snowfall"))}</small></div>
              <div className="flux-track melt"><span style={{ width: `${relativeFlux("snowmelt")}%` }} /><small>Snowmelt {hydrologicValue(flux("snowmelt"))}</small></div>
              <div className="flux-track rain"><span style={{ width: `${relativeFlux("rain")}%` }} /><small>Rain {hydrologicValue(flux("rain"))}</small></div>
            </div>

            <div className="animated-reservoir soil-reservoir">
              <div className="animated-water soil-water" style={{ height: `${soilLevel}%` }} />
              <div className="reservoir-reading"><Droplets size={18} /><span>{isDiscovery ? "Discovered storage" : "Soil storage"}</span><strong>{hydrologicValue(finite(stateAfter[1]))} mm</strong><small>{soilLevel.toFixed(decimalPlaces)}% of {isDiscovery ? "run maximum" : isPaml ? "trained physical capacity" : "calibrated capacity"}</small></div>
            </div>

            <div className={`outflow-grid ${isDiscovery ? "discovery-flows" : isHbv ? "four-flows" : ""}`}>
              {outflowItems.map(([label, name], position) => (
                <div key={name}>
                  {position === 0 ? <SunMedium size={15} /> : <Waves size={15} />}
                  <span>{label}</span>
                  <strong>{hydrologicValue(flux(name))}</strong>
                </div>
              ))}
            </div>
            {groundwaterStorage !== null && <div className="discharge-comparison groundwater-summary"><div><span>Groundwater storage</span><strong>{hydrologicValue(groundwaterStorage)}</strong><small>mm at boundary t + 1</small></div><div><span>Recharge</span><strong>{hydrologicValue(flux("groundwater_recharge"))}</strong><small>mm / interval</small></div></div>}
            {isHbv && <div className="discharge-comparison hbv-response-summary">
              <div><span>Upper-zone storage</span><strong>{hydrologicValue(upperZoneStorage ?? 0)}</strong><small>Recharge {hydrologicValue(flux("groundwater_recharge"))} mm</small></div>
              <div><span>Percolation</span><strong>{hydrologicValue(flux("percolation"))}</strong><small>upper → lower zone</small></div>
              <div><span>Lower-zone storage</span><strong>{hydrologicValue(lowerZoneStorage ?? 0)}</strong><small>mm at boundary t + 1</small></div>
            </div>}
            {hasRouting && <div className="discharge-comparison routing-summary">
              <div><span>HBV response discharge</span><strong>{hydrologicValue(flux("unrouted_streamflow"))}</strong><small>mm entering routing</small></div>
              <div><span>Routing storage</span><strong>{hydrologicValue(routingStorage ?? 0)}</strong><small>mm pending after interval</small></div>
              <div><span>Routed discharge</span><strong>{hydrologicValue(flux("routed_streamflow"))}</strong><small>mm leaving routing</small></div>
            </div>}
          </div>

          <div className={`discharge-comparison${hasPhysicalOutput ? " paml-discharge-summary" : ""}`}>
            <div><span>Observed streamflow</span><strong>{observed !== undefined && observed !== null && Number.isFinite(observed) ? hydrologicValue(observed) : "Not supplied"}</strong><small>mm / interval</small></div>
            {hasPhysicalOutput
              ? <>
                <div><span>Physical streamflow</span><strong>{hydrologicValue(physical ?? 0)}</strong><small>embedded model</small></div>
                <div><span>Corrected streamflow</span><strong>{hydrologicValue(simulated)}</strong><small>PaML output</small></div>
              </>
              : <>
                <div className="comparison-arrow">→</div>
                <div><span>Simulated streamflow</span><strong>{hydrologicValue(simulated)}</strong><small>mm / interval</small></div>
              </>}
          </div>
          <p className="animation-scale-note">Flow bars show each flux relative to its own maximum over this run; numeric values remain the scientific reference.</p>
        </div>

        <div className="animation-chart">
          <div className="chart-heading compact-animation-heading"><div><p className="section-kicker">Current position</p><h2>Output trajectory</h2></div></div>
          <div className="chart-wrap animation-chart-wrap">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={trajectory} margin={{ top: 10, right: 14, left: -8, bottom: 4 }}>
                <CartesianGrid strokeDasharray="4 6" vertical={false} stroke="var(--chart-grid)" />
                <XAxis dataKey="index" type="number" domain={[0, Math.max(1, nIntervals - 1)]} tickFormatter={(value) => result.timestamps?.[Math.round(Number(value))] ?? String(Number(value) + 1)} minTickGap={36} tick={{ fill: "var(--text-muted)", fontSize: 12 }} tickLine={false} axisLine={false} />
                <YAxis tickFormatter={(value) => hydrologicValue(Number(value))} tick={{ fill: "var(--text-muted)", fontSize: 12 }} tickLine={false} axisLine={false} width={60} />
                <Tooltip formatter={(value) => hydrologicValue(Number(value))} labelFormatter={(value) => result.timestamps?.[Math.round(Number(value))] ?? value} contentStyle={{ background: "var(--surface-elevated)", border: "1px solid var(--border)", borderRadius: 10, fontSize: 12 }} />
                <ReferenceLine x={index} stroke="var(--warning)" strokeWidth={2} />
                {hasObservations && <Line type="monotone" dataKey="observed" stroke="var(--chart-observed)" strokeWidth={2} dot={false} connectNulls={false} />}
                {hasPhysicalOutput && <Line type="monotone" dataKey="physical" stroke="var(--chart-physical)" strokeWidth={1.8} strokeDasharray="6 4" dot={false} />}
                <Line type="monotone" dataKey="simulated" stroke="var(--chart-simulated)" strokeWidth={2} dot={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className="animation-legend">{hasObservations && <span><i className="observed" /> Observed</span>}{hasPhysicalOutput && <span><i className="physical" /> Physical</span>}<span><i className="simulated" /> {hasPhysicalOutput ? "Corrected" : "Simulated"}</span><span><i className="cursor" /> Selected interval</span></div>
        </div>
      </div>

      <div className="animation-controls">
        <button className="icon-button" onClick={() => move(index - 1)} disabled={index === 0} aria-label="Previous interval"><ChevronLeft size={18} /></button>
        <button className="animation-play" onClick={toggle}>{playing ? <Pause size={17} fill="currentColor" /> : <Play size={17} fill="currentColor" />} {playing ? "Pause" : index >= nIntervals - 1 ? "Replay" : "Play"}</button>
        <button className="icon-button" onClick={() => move(index + 1)} disabled={index >= nIntervals - 1} aria-label="Next interval"><ChevronRight size={18} /></button>
        <input aria-label="Selected hydrological interval" type="range" min={0} max={Math.max(0, nIntervals - 1)} value={index} onChange={(event) => move(Number(event.target.value))} />
        <label className="speed-control"><span>Speed</span><select value={speed} onChange={(event) => setSpeed(Number(event.target.value))}><option value={0.5}>0.5×</option><option value={1}>1×</option><option value={2}>2×</option><option value={4}>4×</option></select></label>
      </div>
    </section>
  );
}
