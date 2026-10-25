import { useEffect, useMemo, useRef, useState } from "react";
import {
  Activity,
  ArrowRight,
  BarChart3,
  Boxes,
  BrainCircuit,
  Check,
  ChevronDown,
  CircleAlert,
  CircleCheck,
  CloudRain,
  CloudSun,
  Database,
  Download,
  Droplets,
  FileText,
  FlaskConical,
  FolderOpen,
  Gauge,
  History,
  Info,
  LineChart as LineChartIcon,
  Moon,
  Play,
  Plus,
  RotateCcw,
  Save,
  Settings2,
  ShieldCheck,
  Snowflake,
  Square,
  Sun,
  Trash2,
  Upload,
  Waves,
} from "lucide-react";
import {
  Area,
  Bar,
  BarChart,
  Brush,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Scatter,
  ScatterChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ApiError,
  analyzeDataQuality,
  cancelTemporalValidation,
  cancelTask,
  compareSavedRuns,
  explainSavedRun,
  health,
  importCsv,
  listForecastEnsembles,
  listForecastEventVerifications,
  listForecastPostprocessings,
  listForecastVerifications,
  listSavedExperiments,
  listSavedRuns,
  listTemporalValidations,
  loadRunArchive,
  loadForecastEventVerification,
  loadForecastPostprocessing,
  loadForecastVerification,
  loadSavedExperiment,
  loadTemporalValidation,
  previewWorkspaceCsv,
  postprocessForecastEnsembles,
  quantifyRunUncertainty,
  runResult,
  startRun,
  startForecast,
  startForecastEnsemble,
  startTemporalValidation,
  saveExperiment,
  taskStatus,
  temporalValidationResult,
  temporalValidationStatus,
  validateExperiment,
  verifyForecastEvents,
  verifyForecastEnsembles,
  verifySavedRunScientifically,
} from "./api";
import {
  autoMapColumns,
  buildExperimentConfig,
  defaultForm,
  interfaceVersion,
  mappingIsComplete,
  snowLabel,
  workspaceFormFromConfig,
} from "./config";
import type {
  ColumnMapping,
  CsvPreview,
  DataQualityReport,
  PanelId,
  RunResult,
  RunArchiveSummary,
  RunComparison,
  RunExplanation,
  RunScientificVerification,
  RunUncertainty,
  ExperimentConfig,
  ForecastEnsembleArchiveSummary,
  ForecastEnsembleRequest,
  ForecastEventVerification,
  ForecastEventVerificationArchiveSummary,
  ForecastPostprocessing,
  ForecastPostprocessingArchiveSummary,
  ForecastRequest,
  ForecastVerification,
  ForecastVerificationArchiveSummary,
  SavedExperimentSummary,
  StructuredError,
  TaskSnapshot,
  TemporalValidation,
  TemporalValidationArchiveSummary,
  TemporalValidationRequest,
  ValidationSummary,
  WorkspaceForm,
} from "./types";
import { HydrologicAnimator } from "./HydrologicAnimator";
import { downsampleForChart } from "./chartData";

const isDiscoveryModel = (modelId: WorkspaceForm["modelId"]): boolean =>
  modelId === "deepdiscover.numpy_instantaneous_s2s.v1";

const isLstmModel = (modelId: WorkspaceForm["modelId"]): boolean =>
  modelId === "lstm.tensorflow_s2v.v1"
  || modelId === "lstm.tensorflow_stacked_s2v.v2"
  || modelId === "lstm.tensorflow_stacked_s2s.v3";

const isGruModel = (modelId: WorkspaceForm["modelId"]): boolean =>
  modelId === "gru.tensorflow_s2v.v1"
  || modelId === "gru.tensorflow_stacked_s2v.v2"
  || modelId === "gru.tensorflow_stacked_s2s.v3";

const isMlpModel = (modelId: WorkspaceForm["modelId"]): boolean =>
  modelId === "mlp.tensorflow_s2v.v1" || modelId === "mlp.tensorflow_s2s.v2";

const isCnnModel = (modelId: WorkspaceForm["modelId"]): boolean =>
  modelId === "cnn1d.tensorflow_s2v.v1" || modelId === "cnn1d.tensorflow_s2s.v2";

const isPamlModel = (modelId: WorkspaceForm["modelId"]): boolean =>
  modelId === "paml.tensorflow_prnn_cnn_s2s.v1";

const isS2SNeuralModel = (modelId: WorkspaceForm["modelId"]): boolean =>
  modelId === "mlp.tensorflow_s2s.v2"
  || modelId === "lstm.tensorflow_stacked_s2s.v3"
  || modelId === "gru.tensorflow_stacked_s2s.v3"
  || modelId === "cnn1d.tensorflow_s2s.v2";

const s2sCounterpart = (modelId: WorkspaceForm["pamlCorrectorId"]) =>
  modelId.startsWith("mlp.") ? "mlp.tensorflow_s2s.v2" as const
  : modelId.startsWith("lstm.") ? "lstm.tensorflow_stacked_s2s.v3" as const
  : modelId.startsWith("gru.") ? "gru.tensorflow_stacked_s2s.v3" as const
  : "cnn1d.tensorflow_s2s.v2" as const;

const s2vCounterpart = (modelId: WorkspaceForm["pamlCorrectorId"]) =>
  modelId.startsWith("mlp.") ? "mlp.tensorflow_s2v.v1" as const
  : modelId.startsWith("lstm.") ? "lstm.tensorflow_stacked_s2v.v2" as const
  : modelId.startsWith("gru.") ? "gru.tensorflow_stacked_s2v.v2" as const
  : "cnn1d.tensorflow_s2v.v1" as const;

const isRecurrentModel = (modelId: WorkspaceForm["modelId"]): boolean =>
  isLstmModel(modelId) || isGruModel(modelId);

const isTrainableModel = (modelId: WorkspaceForm["modelId"]): boolean =>
  isMlpModel(modelId)
  || isRecurrentModel(modelId)
  || isCnnModel(modelId)
  || isPamlModel(modelId)
  || isDiscoveryModel(modelId);

const objectiveOptions: Array<{
  id: WorkspaceForm["objective"];
  symbol: string;
  name: string;
  guidance: string;
}> = [
  { id: "nash_sutcliffe_efficiency", symbol: "NSE", name: "Nash–Sutcliffe efficiency", guidance: "Maximize · squared-error efficiency" },
  { id: "kling_gupta_efficiency_2009", symbol: "KGE", name: "Kling–Gupta efficiency (2009)", guidance: "Maximize · correlation, variability, and bias" },
  { id: "mean_square_error", symbol: "MSE", name: "Mean square error", guidance: "Minimize · strongly weights large errors" },
  { id: "root_mean_square_error", symbol: "RMSE", name: "Root mean square error", guidance: "Minimize · retains target units" },
  { id: "range_normalized_root_mean_square_error", symbol: "NRMSE", name: "Range-normalized RMSE", guidance: "Minimize · dimensionless range scaling" },
  { id: "mean_absolute_error", symbol: "MAE", name: "Mean absolute error", guidance: "Minimize · less dominated by large residuals" },
];

const objectiveOption = (id: WorkspaceForm["objective"]) =>
  objectiveOptions.find((option) => option.id === id) ?? objectiveOptions[0];

const navigation: Array<{
  id: PanelId;
  label: string;
  detail: string;
  icon: typeof Database;
}> = [
  { id: "data", label: "Data", detail: "Import and map", icon: Database },
  { id: "quality", label: "Quality", detail: "Checks and units", icon: CircleCheck },
  { id: "model", label: "Model", detail: "Structure and parameters", icon: Boxes },
  {
    id: "calibration",
    label: "Calibration",
    detail: "Objective and search",
    icon: Settings2,
  },
  { id: "run", label: "Run", detail: "Preflight and execution", icon: Play },
  {
    id: "results",
    label: "Results",
    detail: "Metrics and diagnostics",
    icon: LineChartIcon,
  },
];

const dataQualityFields = new Set<keyof WorkspaceForm>([
  "delimiter",
  "encoding",
  "mapping",
  "additionalPredictors",
  "precipitationUnit",
  "temperatureUnit",
  "potentialEvapotranspirationUnit",
  "streamflowUnit",
  "basinAreaKm2",
  "precipitationMissing",
  "temperatureMissing",
  "potentialEvapotranspirationMissing",
  "streamflowMissing",
  "maximumInterpolationGap",
  "missingTextTokens",
  "numericSentinels",
  "precipitationNegative",
  "potentialEvapotranspirationNegative",
  "streamflowNegative",
  "requireTimestamp",
]);

function formatBytes(value: number): string {
  if (value < 1024) return `${value} B`;
  if (value < 1024 * 1024) return `${(value / 1024).toFixed(1)} KB`;
  return `${(value / (1024 * 1024)).toFixed(1)} MB`;
}

function formatScore(value: number | null | undefined): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return value.toFixed(4);
}

function formatMetricValue(value: number | null, unit: string, decimalPlaces: number): string {
  if (value === null || !Number.isFinite(value)) return "Unavailable";
  const rounded = value.toFixed(decimalPlaces);
  const formatted = Number(rounded) === 0 ? (0).toFixed(decimalPlaces) : rounded;
  return unit === "%" ? `${formatted}%` : formatted;
}

function formatDecimal(value: number, decimalPlaces: number): string {
  if (!Number.isFinite(value)) return "—";
  const rounded = value.toFixed(decimalPlaces);
  return Number(rounded) === 0 ? (0).toFixed(decimalPlaces) : rounded;
}

function formatMetricInterval(
  lower: number | null,
  upper: number | null,
  unit: string,
  decimalPlaces: number,
): string {
  if (lower === null || upper === null) return "Interval unavailable";
  const suffix = unit === "%"
    ? "%"
    : unit === "dimensionless"
      ? ""
      : ` ${unit}`;
  return `[${formatDecimal(lower, decimalPlaces)}, ${formatDecimal(upper, decimalPlaces)}]${suffix}`;
}

function variableLabel(name: string): string {
  const labels: Record<string, string> = {
    streamflow: "Streamflow",
    actual_evapotranspiration: "Actual evapotranspiration",
    potential_evapotranspiration: "Potential evapotranspiration",
    groundwater_recharge: "Groundwater recharge",
    groundwater_discharge: "Groundwater discharge",
    direct_runoff: "Direct runoff",
    subsurface_flow: "Subsurface flow",
    surface_flow: "Surface flow",
    rainfall_runoff: "Rainfall runoff",
    saturation_excess: "Saturation excess",
    upper_zone_storage: "Upper-zone storage",
    lower_zone_storage: "Lower-zone storage",
    groundwater_storage: "Groundwater storage",
    soil_storage: "Soil storage",
    snow_storage: "Snow storage",
    discovered_storage: "Discovered storage",
    routed_streamflow: "Routed streamflow",
    unrouted_streamflow: "Unrouted streamflow",
    physical_output__streamflow: "Physical streamflow",
  };
  return labels[name] ?? name.replaceAll("_", " ").replace(/^./, (letter) => letter.toUpperCase());
}

function metricGuidance(
  direction: "maximize" | "minimize" | "target",
  idealValue: number,
): string {
  if (direction === "maximize") return `${idealValue} is ideal · higher is better`;
  if (direction === "minimize") return `${idealValue} is ideal · lower is better`;
  return `${idealValue} is ideal · magnitude measures bias`;
}

function formatSeconds(value: number | null | undefined): string {
  if (value === null || value === undefined) return "Estimating…";
  if (value < 60) return `${Math.max(0, Math.round(value))} sec`;
  return `${Math.floor(value / 60)} min ${Math.round(value % 60)} sec`;
}

export function liveTaskTiming(
  task: TaskSnapshot,
  nowMilliseconds: number = Date.now(),
): { elapsed: number; remaining: number | null } {
  const event = task.latest_event;
  if (task.terminal) {
    return {
      elapsed: event.elapsed_seconds,
      remaining: event.estimated_remaining_seconds,
    };
  }
  const updatedMilliseconds = Date.parse(task.updated_at);
  const eventAge = Number.isFinite(updatedMilliseconds)
    ? Math.max(0, (nowMilliseconds - updatedMilliseconds) / 1000)
    : 0;
  return {
    elapsed: Math.max(0, event.elapsed_seconds + eventAge),
    remaining: event.estimated_remaining_seconds === null
      ? null
      : Math.max(0, event.estimated_remaining_seconds - eventAge),
  };
}

function formatRange(minimum: number | null, maximum: number | null): string {
  if (minimum === null || maximum === null) return "—";
  return `${minimum.toLocaleString(undefined, { maximumFractionDigits: 3 })} – ${maximum.toLocaleString(undefined, { maximumFractionDigits: 3 })}`;
}

function errorFrom(value: unknown): StructuredError {
  if (value instanceof ApiError) return value.detail;
  return {
    code: "unexpected_interface_error",
    summary: "The operation could not be completed.",
    explanation: value instanceof Error ? value.message : String(value),
    suggested_action: "Review the entries and retry. The current project was not changed.",
  };
}

function PageHeader({
  eyebrow,
  title,
  description,
  actions,
}: {
  eyebrow: string;
  title: string;
  description: string;
  actions?: React.ReactNode;
}) {
  return (
    <div className="page-header">
      <div>
        <p className="eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        <p className="page-description">{description}</p>
      </div>
      {actions && <div className="page-actions">{actions}</div>}
    </div>
  );
}

function Field({
  label,
  hint,
  children,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
}) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

function ErrorNotice({
  error,
  onDismiss,
}: {
  error: StructuredError;
  onDismiss: () => void;
}) {
  return (
    <div className="notice notice-error" role="alert">
      <CircleAlert size={20} />
      <div>
        <strong>{error.summary ?? "Something needs attention"}</strong>
        <p>{error.message ?? error.explanation}</p>
        {error.suggested_action && <small>{error.suggested_action}</small>}
      </div>
      <button className="notice-dismiss" onClick={onDismiss} aria-label="Dismiss error">
        ×
      </button>
    </div>
  );
}

type SnowTreatmentKey = keyof WorkspaceForm["snowCalibration"];

function activeSnowTreatmentKeys(form: WorkspaceForm): SnowTreatmentKey[] {
  if (form.snowModule === "snow.none.v1") return [];
  const keys: SnowTreatmentKey[] = [
    "snowfallTemperature",
    "meltTemperature",
    "degreeDayFactor",
  ];
  if (form.snowModule === "snow.degree_day.smooth.v1") keys.push("sharpness");
  return keys;
}

function calibratedParameterCount(form: WorkspaceForm): number {
  const conceptualModelId = isPamlModel(form.modelId)
    ? form.pamlPhysicalModelId
    : form.modelId;
  const modelCount = conceptualModelId === "hbv.mass_conservative_daily.v1" ? 8 : 3;
  const routingCount = conceptualModelId === "hbv.mass_conservative_daily.v1"
    && form.routingProcess === "routing.maxbas.triangular.v1"
    && form.maxbasTreatment.mode === "calibrate"
    ? 1
    : 0;
  return modelCount + routingCount + activeSnowTreatmentKeys(form).filter(
    (key) => form.snowCalibration[key].mode === "calibrate",
  ).length;
}

function calibrationSearchIsValid(form: WorkspaceForm): boolean {
  if (form.calibrationAlgorithm === "uniform_random_search.v1") {
    return Number.isInteger(form.evaluations)
      && form.evaluations >= 1
      && form.evaluations <= 1_000_000;
  }
  return Number.isInteger(form.deGenerations)
    && form.deGenerations >= 1
    && form.deGenerations <= 100_000
    && (
      form.dePopulationSize === null
      || (
        Number.isInteger(form.dePopulationSize)
        && form.dePopulationSize >= 4
        && form.dePopulationSize <= 100_000
      )
    )
    && form.deMutationFactor > 0
    && form.deMutationFactor <= 2
    && form.deCrossoverProbability >= 0
    && form.deCrossoverProbability <= 1;
}

function calibrationEvaluationBudget(form: WorkspaceForm): number {
  if (form.calibrationAlgorithm === "uniform_random_search.v1") {
    return form.evaluations;
  }
  const population = form.dePopulationSize
    ?? Math.max(8, 10 * calibratedParameterCount(form));
  return population * (form.deGenerations + 1);
}

function calibrationAlgorithmLabel(form: WorkspaceForm): string {
  return form.calibrationAlgorithm === "differential_evolution.rand1bin.v1"
    ? "Differential evolution · DE/rand/1/bin"
    : "Uniform random search · reference";
}

function ParameterTreatmentControl({
  label,
  unit,
  value,
  treatment,
  onValue,
  onTreatment,
  step = 0.1,
  minimum,
  maximum,
}: {
  label: string;
  unit: string;
  value: number;
  treatment: WorkspaceForm["maxbasTreatment"];
  onValue: (value: number) => void;
  onTreatment: (update: Partial<WorkspaceForm["maxbasTreatment"]>) => void;
  step?: number;
  minimum?: number;
  maximum?: number;
}) {
  const calibrating = treatment.mode === "calibrate";
  const valid = !calibrating || (
    treatment.lowerBound < treatment.upperBound
    && value >= treatment.lowerBound
    && value <= treatment.upperBound
  );
  return (
    <div className={`parameter-treatment-row ${calibrating ? "calibrating" : "fixed"}`}>
      <div className="parameter-treatment-name">
        <strong>{label}</strong>
        <span>{unit}</span>
      </div>
      <div className="parameter-treatment-fields">
        <Field label={calibrating ? "Starting value" : "Fixed value"}>
          <input type="number" step={step} min={minimum} max={maximum} value={value} onChange={(event) => onValue(Number(event.target.value))} />
        </Field>
        <Field label="Treatment">
          <select value={treatment.mode} onChange={(event) => onTreatment({ mode: event.target.value as "fixed" | "calibrate" })}>
            <option value="fixed">Fixed</option>
            <option value="calibrate">Calibrate</option>
          </select>
        </Field>
        <Field label="Lower bound">
          <input type="number" step={step} min={minimum} max={maximum} value={treatment.lowerBound} disabled={!calibrating} onChange={(event) => onTreatment({ lowerBound: Number(event.target.value) })} />
        </Field>
        <Field label="Upper bound">
          <input type="number" step={step} min={minimum} max={maximum} value={treatment.upperBound} disabled={!calibrating} onChange={(event) => onTreatment({ upperBound: Number(event.target.value) })} />
        </Field>
      </div>
      {!valid && <span className="parameter-treatment-error">The starting value must lie inside two ordered bounds.</span>}
    </div>
  );
}

function ProjectLibrary({
  projects,
  busy,
  onClose,
  onOpenSetup,
  onOpenResult,
}: {
  projects: SavedExperimentSummary[];
  busy: boolean;
  onClose: () => void;
  onOpenSetup: (experimentId: string) => void;
  onOpenResult: (taskId: string) => void;
}) {
  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <section
        className="project-library"
        role="dialog"
        aria-modal="true"
        aria-labelledby="project-library-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="project-library-heading">
          <div>
            <p className="section-kicker">Persistent workspace</p>
            <h2 id="project-library-title">Saved experiments</h2>
            <p>Reopen an exact configuration or return to its latest archived result.</p>
          </div>
          <button className="notice-dismiss modal-close" onClick={onClose} aria-label="Close saved experiments">×</button>
        </div>
        <div className="project-library-list">
          {projects.length === 0 ? (
            <div className="library-empty">
              <FolderOpen size={28} />
              <strong>No experiment has been saved yet</strong>
              <span>Use Save experiment after importing and mapping a data set.</span>
            </div>
          ) : projects.map((project) => (
            <article className="saved-project" key={project.experiment_id}>
              <div className="saved-project-main">
                <div className="saved-project-icon"><FileText size={18} /></div>
                <div>
                  <strong>{project.name}</strong>
                  <span>{project.description || "No description"}</span>
                  <small>Updated {new Date(project.updated_at).toLocaleString()} · {project.run_count} archived run{project.run_count === 1 ? "" : "s"}</small>
                </div>
              </div>
              <div className="saved-project-actions">
                <button className="button button-secondary" disabled={busy} onClick={() => onOpenSetup(project.experiment_id)}>
                  <FolderOpen size={15} /> Open setup
                </button>
                <button className="button button-primary" disabled={busy || !project.latest_run_id} onClick={() => project.latest_run_id && onOpenResult(project.latest_run_id)}>
                  <History size={15} /> Latest result
                </button>
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

function ComparisonDialog({
  projects,
  currentTaskId,
  onClose,
  onError,
}: {
  projects: SavedExperimentSummary[];
  currentTaskId: string;
  onClose: () => void;
  onError: (error: StructuredError) => void;
}) {
  const [runs, setRuns] = useState<RunArchiveSummary[]>([]);
  const [selected, setSelected] = useState<Set<string>>(
    () => new Set(currentTaskId ? [currentTaskId] : []),
  );
  const [comparison, setComparison] = useState<RunComparison | null>(null);
  const [uncertainty, setUncertainty] = useState<RunUncertainty | null>(null);
  const [loading, setLoading] = useState(true);
  const [comparing, setComparing] = useState(false);
  const [uncertaintyBusy, setUncertaintyBusy] = useState(false);

  useEffect(() => {
    let active = true;
    Promise.all(projects.map((project) => listSavedRuns(project.experiment_id)))
      .then((groups) => {
        if (!active) return;
        const available = groups
          .flat()
          .filter((run) => run.state === "completed")
          .sort((left, right) => right.saved_at.localeCompare(left.saved_at));
        setRuns(available);
        if (
          currentTaskId
          && !available.some((run) => run.task_id === currentTaskId)
        ) {
          setSelected(new Set());
        }
      })
      .catch((value) => {
        if (active) onError(errorFrom(value));
      })
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [currentTaskId, onError, projects]);

  const toggle = (taskId: string) => {
    setComparison(null);
    setUncertainty(null);
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(taskId)) {
        next.delete(taskId);
      } else if (next.size < 6) {
        next.add(taskId);
      }
      return next;
    });
  };
  const runComparison = async () => {
    if (selected.size < 2) return;
    setComparing(true);
    try {
      setComparison(await compareSavedRuns([...selected]));
      setUncertainty(null);
    } catch (value) {
      onError(errorFrom(value));
    } finally {
      setComparing(false);
    }
  };
  const runUncertainty = async () => {
    if (!comparison || selected.size < 2) return;
    setUncertaintyBusy(true);
    try {
      setUncertainty(await quantifyRunUncertainty([...selected], {
        referenceTaskId: comparison.runs[0].task_id,
      }));
    } catch (value) {
      onError(errorFrom(value));
    } finally {
      setUncertaintyBusy(false);
    }
  };
  const metricSymbols = ["NSE", "KGE", "MSE", "RMSE", "MAE", "PBIAS", "NRMSE"];
  const colors = [
    "var(--chart-simulated)",
    "var(--purple)",
    "var(--warning)",
    "var(--accent-cyan)",
    "var(--danger)",
    "var(--text-muted)",
  ];
  const comparisonRows = useMemo(() => {
    if (!comparison) return [];
    const rows = comparison.observations.map((observed, index) => {
      const row: {
        index: number;
        time: string;
        observed: number | null;
        [key: string]: string | number | null;
      } = {
        index,
        time: comparison.timestamps?.[index]
          ?? String((comparison.source_rows?.[index] ?? index) + 1),
        observed: Number.isFinite(observed) ? observed : null,
      };
      comparison.runs.forEach((run, position) => {
        const value = run.predictions[index];
        row[`run_${position}`] = Number.isFinite(value) ? value : null;
      });
      return row;
    });
    return downsampleForChart(
      rows,
      ["observed", ...comparison.runs.map((_, position) => `run_${position}`)],
      1000,
    );
  }, [comparison]);

  return (
    <div className="modal-backdrop comparison-backdrop" role="presentation" onMouseDown={onClose}>
      <section
        className={`comparison-dialog${comparison ? " comparison-dialog-results" : ""}`}
        role="dialog"
        aria-modal="true"
        aria-labelledby="comparison-dialog-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="project-library-heading comparison-heading">
          <div>
            <p className="section-kicker">Common-support evaluation</p>
            <h2 id="comparison-dialog-title">Compare model runs</h2>
            <p>Select two to six archived runs. Metrics are recomputed on one identical finite evaluation support.</p>
          </div>
          <button className="notice-dismiss modal-close" onClick={onClose} aria-label="Close model comparison">×</button>
        </div>

        {!comparison ? (
          <>
            <div className="comparison-run-list">
              {loading ? (
                <div className="library-empty"><Activity size={27} /><strong>Loading archived runs…</strong></div>
              ) : runs.length < 2 ? (
                <div className="library-empty"><BarChart3 size={28} /><strong>At least two completed runs are required</strong><span>Save and execute another experiment in this workspace first.</span></div>
              ) : runs.map((run) => {
                const checked = selected.has(run.task_id);
                return (
                  <label className={`comparison-run ${checked ? "selected" : ""}`} key={run.task_id}>
                    <input type="checkbox" checked={checked} disabled={!checked && selected.size >= 6} onChange={() => toggle(run.task_id)} />
                    <span className="comparison-run-main">
                      <strong>{run.experiment_name}</strong>
                      <span>{run.model_id.replaceAll("_", " ")}</span>
                      <small>{new Date(run.saved_at).toLocaleString()} · {run.run_kind.replaceAll("_", " ")}</small>
                    </span>
                    <span className="status-pill neutral">{run.objective_name ? `${run.objective_name.replaceAll("_", " ")} ${formatScore(run.best_score)}` : "Completed"}</span>
                  </label>
                );
              })}
            </div>
            <div className="comparison-footer">
              <span>{selected.size} of 6 runs selected</span>
              <button className="button button-primary" disabled={selected.size < 2 || comparing} onClick={runComparison}><BarChart3 size={16} /> {comparing ? "Aligning runs…" : "Compare selected runs"}</button>
            </div>
          </>
        ) : (
          <div className="comparison-results">
            <div className="comparison-summary">
              <div><span>Alignment</span><strong>{comparison.alignment === "timestamp" ? "Timestamp" : "Source row"}</strong></div>
              <div><span>Common scored observations</span><strong>{comparison.scored_observations.toLocaleString()}</strong></div>
              <div><span>Scored coverage</span><strong>{comparison.first_scored_label} → {comparison.last_scored_label}</strong></div>
              <div className="comparison-summary-actions">
                <button className="button button-secondary" disabled={uncertaintyBusy} onClick={runUncertainty}><Gauge size={15} /> {uncertaintyBusy ? "Resampling…" : uncertainty ? "Recompute uncertainty" : "Quantify uncertainty"}</button>
                <button className="button button-secondary" onClick={() => { setComparison(null); setUncertainty(null); }}><RotateCcw size={15} /> Change selection</button>
              </div>
            </div>
            {comparison.warnings.map((warning) => <div className="notice comparison-warning" key={warning.code}><CircleAlert size={18} /><div><strong>Interpretation limit</strong><p>{warning.message}</p></div></div>)}
            <div className="table-scroll comparison-table-scroll">
              <table className="data-table comparison-metric-table">
                <thead><tr><th>Experiment</th><th>Model</th>{metricSymbols.map((symbol) => <th key={symbol}>{symbol}</th>)}</tr></thead>
                <tbody>{comparison.runs.map((run) => (
                  <tr key={run.task_id}>
                    <td><strong>{run.experiment_name}</strong><small>{run.task_id}</small></td>
                    <td>{run.model_id.replaceAll("_", " ")}</td>
                    {metricSymbols.map((symbol) => {
                      const metric = run.metrics.find((item) => item.symbol === symbol);
                      return <td key={symbol}>{metric ? formatMetricValue(metric.value, metric.unit, 4) : "—"}</td>;
                    })}
                  </tr>
                ))}</tbody>
              </table>
            </div>
            {uncertainty && <section className="comparison-uncertainty" aria-labelledby="comparison-uncertainty-title">
              <div className="uncertainty-heading">
                <div><p className="section-kicker">Paired temporal resampling</p><h3 id="comparison-uncertainty-title">Sampling uncertainty</h3><p>Every model uses the same resampled daily blocks. Positive improvement means the candidate is closer to the metric ideal.</p></div>
                <div className="uncertainty-method"><span>{Math.round(uncertainty.confidence_level * 100)}% percentile intervals</span><strong>{uncertainty.bootstrap_replicates.toLocaleString()} replicates · {uncertainty.block_length}-day blocks</strong><small>{uncertainty.block_length_source === "user" ? "User block length" : "Automatic autocorrelation cutoff"}</small></div>
              </div>
              <div className="table-scroll comparison-table-scroll">
                <table className="data-table uncertainty-metric-table">
                  <thead><tr><th>Experiment</th>{metricSymbols.map((symbol) => <th key={symbol}>{symbol} · interval</th>)}</tr></thead>
                  <tbody>{uncertainty.runs.map((run) => (
                    <tr key={`uncertainty-${run.task_id}`}>
                      <td><strong>{run.experiment_name}</strong><small>{run.task_id === uncertainty.reference_task_id ? "Reference" : "Candidate"}</small></td>
                      {metricSymbols.map((symbol) => {
                        const metric = run.metrics.find((item) => item.symbol === symbol);
                        return <td key={symbol}>{metric ? <><strong>{formatMetricValue(metric.value, metric.unit, 4)}</strong><small>{formatMetricInterval(metric.lower, metric.upper, metric.unit, 4)}</small></> : "—"}</td>;
                      })}
                    </tr>
                  ))}</tbody>
                </table>
              </div>
              {uncertainty.comparisons.map((paired) => <div className="paired-comparison" key={paired.candidate_task_id}>
                <div className="paired-comparison-heading"><span>Reference: {paired.reference_experiment_name}</span><strong>{paired.candidate_experiment_name}</strong></div>
                <div className="paired-metric-grid">
                  {paired.metrics.map((metric) => <div className={`paired-metric ${metric.interpretation}`} key={metric.metric_id}>
                    <span>{metric.symbol} improvement</span>
                    <strong>{metric.improvement === null ? "—" : formatDecimal(metric.improvement, 4)}</strong>
                    <small>{formatMetricInterval(metric.improvement_lower, metric.improvement_upper, metric.unit, 4)}</small>
                    <em>{metric.interpretation === "interval_favors_candidate" ? "Interval favors candidate" : metric.interpretation === "interval_favors_reference" ? "Interval favors reference" : "Interval overlaps zero"}</em>
                  </div>)}
                </div>
              </div>)}
              <div className="uncertainty-warnings">{uncertainty.warnings.map((warning) => <div key={warning.code}><Info size={15} /><span>{warning.message}</span></div>)}</div>
            </section>}
            <section className="chart-card comparison-chart-card">
              <div className="chart-heading"><div><p className="section-kicker">Aligned trajectories</p><h2>Observed and simulated streamflow</h2></div><span className="render-note">Common source · full values retained</span></div>
              <div className="chart-wrap comparison-chart-wrap">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={comparisonRows} margin={{ top: 10, right: 18, left: 0, bottom: 4 }}>
                    <CartesianGrid strokeDasharray="4 6" vertical={false} stroke="var(--chart-grid)" />
                    <XAxis dataKey="time" minTickGap={36} tick={{ fill: "var(--text-muted)", fontSize: 11 }} tickLine={false} axisLine={false} />
                    <YAxis tick={{ fill: "var(--text-muted)", fontSize: 11 }} tickLine={false} axisLine={false} width={58} />
                    <Tooltip contentStyle={{ background: "var(--surface-elevated)", border: "1px solid var(--border)", borderRadius: 10 }} />
                    <Legend />
                    <Line type="linear" dataKey="observed" name="Observed" stroke="var(--chart-observed)" strokeWidth={2.2} dot={false} />
                    {comparison.runs.map((run, position) => <Line key={run.task_id} type="linear" dataKey={`run_${position}`} name={run.experiment_name} stroke={colors[position]} strokeWidth={1.8} dot={false} />)}
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </section>
          </div>
        )}
      </section>
    </div>
  );
}

const defaultTemporalValidationRequest = (
  experimentId: string,
): TemporalValidationRequest => ({
  experiment_ids: experimentId ? [experimentId] : [],
  start_index: 0,
  initial_training_steps: 1095,
  validation_steps: 365,
  test_steps: 365,
  step_steps: 365,
  folds: 3,
  window_mode: "expanding",
  minimum_scored_observations: 30,
});

export function TemporalValidationDialog({
  projects,
  currentExperimentId,
  onClose,
  onError,
}: {
  projects: SavedExperimentSummary[];
  currentExperimentId: string;
  onClose: () => void;
  onError: (error: StructuredError) => void;
}) {
  const [requestForm, setRequestForm] = useState<TemporalValidationRequest>(
    () => defaultTemporalValidationRequest(currentExperimentId),
  );
  const [task, setTask] = useState<TaskSnapshot | null>(null);
  const [result, setResult] = useState<TemporalValidation | null>(null);
  const [recent, setRecent] = useState<TemporalValidationArchiveSummary[]>([]);
  const [recentLoading, setRecentLoading] = useState(true);
  const [starting, setStarting] = useState(false);
  const requiredRows = requestForm.start_index
    + requestForm.initial_training_steps
    + (requestForm.folds - 1) * requestForm.step_steps
    + requestForm.validation_steps
    + requestForm.test_steps;
  const fitCount = requestForm.experiment_ids.length * requestForm.folds;
  const configurationValid = requestForm.experiment_ids.length > 0
    && requestForm.experiment_ids.length <= 4
    && requestForm.initial_training_steps > 0
    && requestForm.validation_steps > 0
    && requestForm.test_steps > 0
    && requestForm.step_steps >= requestForm.test_steps
    && requestForm.folds >= 1
    && requestForm.folds <= 10
    && requestForm.minimum_scored_observations >= 2
    && requestForm.minimum_scored_observations <= requestForm.test_steps;

  const refreshRecent = async () => {
    try {
      setRecent(await listTemporalValidations());
    } catch (value) {
      onError(errorFrom(value));
    } finally {
      setRecentLoading(false);
    }
  };

  useEffect(() => {
    void refreshRecent();
  }, []);

  const updateNumber = (
    key: keyof TemporalValidationRequest,
    value: string,
  ) => {
    setRequestForm((current) => ({
      ...current,
      [key]: Math.max(0, Math.trunc(Number(value) || 0)),
    }));
  };
  const toggleExperiment = (experimentId: string) => {
    setRequestForm((current) => {
      const selected = new Set(current.experiment_ids);
      if (selected.has(experimentId)) {
        selected.delete(experimentId);
      } else if (selected.size < 4) {
        selected.add(experimentId);
      }
      return { ...current, experiment_ids: [...selected] };
    });
  };
  const runTemporalValidation = async () => {
    if (!configurationValid) return;
    setStarting(true);
    setResult(null);
    try {
      let snapshot = await startTemporalValidation(requestForm);
      setTask(snapshot);
      while (!snapshot.terminal) {
        await new Promise((resolve) => window.setTimeout(resolve, 250));
        snapshot = await temporalValidationStatus(snapshot.task_id);
        setTask(snapshot);
      }
      if (snapshot.state === "completed") {
        setResult(await temporalValidationResult(snapshot.task_id));
        void refreshRecent();
      } else if (snapshot.state === "failed" && snapshot.error) {
        onError(snapshot.error);
      }
    } catch (value) {
      onError(errorFrom(value));
    } finally {
      setStarting(false);
    }
  };
  const cancelCurrent = async () => {
    if (!task || task.terminal) return;
    try {
      setTask(await cancelTemporalValidation(task.task_id));
    } catch (value) {
      onError(errorFrom(value));
    }
  };
  const reopen = async (taskId: string) => {
    setStarting(true);
    try {
      setResult((await loadTemporalValidation(taskId)).result);
      setTask(null);
    } catch (value) {
      onError(errorFrom(value));
    } finally {
      setStarting(false);
    }
  };
  const metricSymbols = ["NSE", "KGE", "MSE", "RMSE", "MAE", "PBIAS", "NRMSE"];
  const colors = [
    "var(--chart-simulated)",
    "var(--purple)",
    "var(--warning)",
    "var(--accent-cyan)",
  ];
  const foldNseRows = result?.folds.map((fold) => {
    const row: Record<string, string | number | null> = {
      fold: `Fold ${fold.fold}`,
    };
    fold.models.forEach((model, position) => {
      row[`model_${position}`] = model.metrics.find(
        (metric) => metric.symbol === "NSE",
      )?.value ?? null;
    });
    return row;
  }) ?? [];

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <section
        className="comparison-dialog temporal-validation-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="temporal-validation-title"
        onMouseDown={(event) => event.stopPropagation()}
      >
        <div className="project-library-heading comparison-heading">
          <div>
            <p className="section-kicker">Repeated future-block testing</p>
            <h2 id="temporal-validation-title">Rolling-origin validation</h2>
            <p>Refit saved experiment definitions on several chronological folds. Every selected model is scored on exactly the same unseen test rows.</p>
          </div>
          <button className="notice-dismiss modal-close" onClick={onClose} aria-label="Close temporal validation">×</button>
        </div>

        {!result ? (
          <div className="temporal-validation-setup">
            <section className="temporal-validation-section">
              <div className="forecast-section-heading">
                <div><strong>1. Saved experiment definitions</strong><span>Select one to four setups. Parameters and weights will be refitted; archived predictions are not reused.</span></div>
                <span>{requestForm.experiment_ids.length} selected</span>
              </div>
              <div className="comparison-run-list temporal-experiment-list">
                {projects.length === 0 ? (
                  <div className="library-empty"><FolderOpen size={28} /><strong>No saved experiment is available</strong><span>Save the current experiment before launching temporal validation.</span></div>
                ) : projects.map((project) => {
                  const checked = requestForm.experiment_ids.includes(project.experiment_id);
                  return (
                    <label className={`comparison-run ${checked ? "selected" : ""}`} key={project.experiment_id}>
                      <input type="checkbox" checked={checked} disabled={!checked && requestForm.experiment_ids.length >= 4} onChange={() => toggleExperiment(project.experiment_id)} />
                      <span className="comparison-run-main">
                        <strong>{project.name}</strong>
                        <span>{project.description || "No description"}</span>
                        <small>{project.run_count} archived run{project.run_count === 1 ? "" : "s"} · updated {new Date(project.updated_at).toLocaleDateString()}</small>
                      </span>
                      <span className="status-pill neutral">{checked ? "Included" : "Available"}</span>
                    </label>
                  );
                })}
              </div>
            </section>

            <section className="temporal-validation-section">
              <div className="forecast-section-heading">
                <div><strong>2. Chronological protocol</strong><span>All boundaries are zero-based source-row positions. Test blocks cannot overlap.</span></div>
              </div>
              <div className="temporal-protocol-grid">
                <label className="field"><span>Window mode</span><select value={requestForm.window_mode} onChange={(event) => setRequestForm((current) => ({ ...current, window_mode: event.target.value as "expanding" | "sliding" }))}><option value="expanding">Expanding history</option><option value="sliding">Fixed sliding history</option></select><small>{requestForm.window_mode === "expanding" ? "Every fold retains all earlier training rows." : "Every fold keeps the same training-window length."}</small></label>
                <label className="field"><span>First source row</span><input type="number" min={0} value={requestForm.start_index} onChange={(event) => updateNumber("start_index", event.target.value)} /><small>Beginning of the first fitting window.</small></label>
                <label className="field"><span>Initial training length</span><input type="number" min={1} value={requestForm.initial_training_steps} onChange={(event) => updateNumber("initial_training_steps", event.target.value)} /><small>Rows used to fit weights before validation.</small></label>
                <label className="field"><span>Validation length</span><input type="number" min={1} value={requestForm.validation_steps} onChange={(event) => updateNumber("validation_steps", event.target.value)} /><small>Checkpoint selection for neural, PaML, and DD.</small></label>
                <label className="field"><span>Test length</span><input type="number" min={1} value={requestForm.test_steps} onChange={(event) => updateNumber("test_steps", event.target.value)} /><small>Strictly unseen rows scored in each fold.</small></label>
                <label className="field"><span>Origin step</span><input type="number" min={requestForm.test_steps} value={requestForm.step_steps} onChange={(event) => updateNumber("step_steps", event.target.value)} /><small>At least the test length; overlap is forbidden.</small></label>
                <label className="field"><span>Number of folds</span><input type="number" min={1} max={10} value={requestForm.folds} onChange={(event) => updateNumber("folds", event.target.value)} /><small>One complete refit per model and fold.</small></label>
                <label className="field"><span>Minimum finite test rows</span><input type="number" min={2} max={requestForm.test_steps} value={requestForm.minimum_scored_observations} onChange={(event) => updateNumber("minimum_scored_observations", event.target.value)} /><small>Required on the common model support.</small></label>
              </div>
              {requestForm.step_steps < requestForm.test_steps && <div className="notice notice-error"><CircleAlert size={18} /><div><strong>Test blocks overlap</strong><p>Increase the origin step to at least {requestForm.test_steps} rows.</p></div></div>}
              <div className="temporal-budget-strip">
                <div><span>Last required source row</span><strong>{Math.max(0, requiredRows - 1)}</strong></div>
                <div><span>Model × fold executions</span><strong>{fitCount.toLocaleString()}</strong></div>
                <div><span>Held-out test rows</span><strong>{(requestForm.folds * requestForm.test_steps).toLocaleString()}</strong></div>
                <div><span>Seed policy</span><strong>Fixed across folds</strong></div>
              </div>
              <div className="notice"><Info size={18} /><div><strong>This measures temporal robustness, not predictive uncertainty</strong><p>Architecture, feature recipe, optimizer budget, and seeds remain fixed. Only parameters or weights are refitted as the chronological origin moves.</p></div></div>
            </section>

            {task && !task.terminal && (
              <section className="temporal-task-card">
                <div>
                  <Activity size={20} />
                  <div><strong>{task.latest_event.message}</strong><span>{task.latest_event.stage.replaceAll("_", " ")}</span></div>
                </div>
                <div className="temporal-task-progress"><span style={{ width: `${Math.round((task.latest_event.fraction ?? 0) * 100)}%` }} /></div>
                <button className="button button-secondary" onClick={cancelCurrent}><Square size={14} /> Cancel safely</button>
              </section>
            )}

            <div className="comparison-footer temporal-validation-footer">
              <span>{configurationValid ? `${fitCount} model-fold executions; active ensembles train every member` : "Complete a valid non-overlapping protocol"}</span>
              <button className="button button-primary" disabled={!configurationValid || starting} onClick={runTemporalValidation}><History size={16} /> {starting ? "Running folds…" : "Start rolling-origin validation"}</button>
            </div>

            <section className="temporal-validation-section recent-temporal-validations">
              <div className="forecast-section-heading">
                <div><strong>Recent temporal validations</strong><span>Completed analyses are archived separately from ordinary model runs.</span></div>
              </div>
              {recentLoading ? <div className="library-empty"><Activity size={24} /><strong>Loading archives…</strong></div> : recent.length === 0 ? <div className="library-empty"><History size={24} /><strong>No archived temporal validation yet</strong></div> : (
                <div className="temporal-archive-list">
                  {recent.slice(0, 8).map((archive) => <button type="button" key={archive.task_id} onClick={() => reopen(archive.task_id)}><span><strong>{archive.experiment_names.join(" · ")}</strong><small>{archive.window_mode} · {archive.folds} folds · {archive.pooled_scored_observations.toLocaleString()} scored rows</small></span><em>{new Date(archive.saved_at).toLocaleString()}</em></button>)}
                </div>
              )}
            </section>
          </div>
        ) : (
          <div className="comparison-results temporal-validation-results">
            <div className="comparison-summary">
              <div><span>Protocol</span><strong>{result.settings.window_mode === "expanding" ? "Expanding origin" : "Sliding origin"}</strong></div>
              <div><span>Completed folds</span><strong>{result.folds.length}</strong></div>
              <div><span>Pooled scored observations</span><strong>{result.pooled_scored_observations.toLocaleString()}</strong></div>
              <div className="comparison-summary-actions"><button className="button button-secondary" onClick={() => { setResult(null); setTask(null); }}><RotateCcw size={15} /> New protocol</button></div>
            </div>
            {result.warnings.map((warning) => <div className="notice comparison-warning" key={warning.code}><Info size={18} /><div><strong>Interpretation rule</strong><p>{warning.message}</p></div></div>)}

            <section className="temporal-validation-section">
              <div className="uncertainty-heading"><div><p className="section-kicker">All out-of-sample rows pooled</p><h3>Overall performance</h3><p>Each metric is recomputed from the concatenated test predictions. Fold NSE and KGE are not averaged to obtain these values.</p></div></div>
              <div className="table-scroll comparison-table-scroll">
                <table className="data-table comparison-metric-table">
                  <thead><tr><th>Experiment</th><th>Model</th>{metricSymbols.map((symbol) => <th key={symbol}>{symbol}</th>)}</tr></thead>
                  <tbody>{result.models.map((model) => <tr key={model.experiment_id}><td><strong>{model.experiment_name}</strong><small>{model.experiment_id}</small></td><td>{model.model_id.replaceAll("_", " ")}</td>{metricSymbols.map((symbol) => { const metric = model.pooled_metrics.find((item) => item.symbol === symbol); return <td key={symbol}>{metric ? formatMetricValue(metric.value, metric.unit, 4) : "—"}</td>; })}</tr>)}</tbody>
                </table>
              </div>
            </section>

            <section className="temporal-validation-section">
              <div className="uncertainty-heading"><div><p className="section-kicker">Fold-level stability</p><h3>Performance through time</h3><p>Standard deviations describe variation among the chronological tests; the folds are not assumed independent.</p></div></div>
              <div className="table-scroll comparison-table-scroll">
                <table className="data-table temporal-fold-table">
                  <thead><tr><th>Fold</th><th>Unseen test block</th><th>Experiment</th>{metricSymbols.map((symbol) => <th key={symbol}>{symbol}</th>)}</tr></thead>
                  <tbody>{result.folds.flatMap((fold) => fold.models.map((model, position) => <tr key={`${fold.fold}-${model.experiment_id}`}><td>{position === 0 ? fold.fold : ""}</td><td>{position === 0 ? <><strong>{fold.first_test_label}</strong><small>to {fold.last_test_label} · {fold.scored_observations} rows</small></> : ""}</td><td><strong>{model.experiment_name}</strong></td>{metricSymbols.map((symbol) => { const metric = model.metrics.find((item) => item.symbol === symbol); return <td key={symbol}>{metric ? formatMetricValue(metric.value, metric.unit, 3) : "—"}</td>; })}</tr>))}</tbody>
                </table>
              </div>
              <div className="chart-card temporal-fold-chart">
                <div className="chart-heading"><div><p className="section-kicker">Temporal trajectory</p><h2>NSE by held-out fold</h2></div><span className="render-note">Same test rows for every model</span></div>
                <div className="chart-wrap comparison-chart-wrap">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart data={foldNseRows} margin={{ top: 10, right: 18, left: 0, bottom: 4 }}>
                      <CartesianGrid strokeDasharray="4 6" vertical={false} stroke="var(--chart-grid)" />
                      <XAxis dataKey="fold" tick={{ fill: "var(--text-muted)", fontSize: 11 }} tickLine={false} axisLine={false} />
                      <YAxis tick={{ fill: "var(--text-muted)", fontSize: 11 }} tickLine={false} axisLine={false} width={58} />
                      <Tooltip contentStyle={{ background: "var(--surface-elevated)", border: "1px solid var(--border)", borderRadius: 10 }} />
                      <Legend />
                      {result.models.map((model, position) => <Line key={model.experiment_id} type="linear" dataKey={`model_${position}`} name={model.experiment_name} stroke={colors[position]} strokeWidth={2} dot={{ r: 3 }} />)}
                    </LineChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </section>

            {result.pairwise.length > 0 && <section className="temporal-validation-section">
              <div className="uncertainty-heading"><div><p className="section-kicker">Ranking stability</p><h3>Paired fold improvements</h3><p>Positive improvement always means the candidate moved closer to the ideal value. Fractions summarize how often that happened, not statistical significance.</p></div></div>
              {result.pairwise.map((paired) => <div className="paired-comparison" key={paired.candidate_experiment_id}><div className="paired-comparison-heading"><span>Reference: {paired.reference_experiment_name}</span><strong>{paired.candidate_experiment_name}</strong></div><div className="paired-metric-grid">{paired.metrics.map((metric) => <div className="paired-metric" key={metric.metric_id}><span>{metric.symbol} mean improvement</span><strong>{metric.mean === null ? "—" : formatDecimal(metric.mean, 4)}</strong><small>Candidate better in {metric.candidate_better_fraction === null ? "—" : `${Math.round(metric.candidate_better_fraction * 100)}%`} of folds</small><em>{metric.available_folds} paired folds</em></div>)}</div></div>)}
            </section>}
          </div>
        )}
      </section>
    </div>
  );
}

function DataPanel({
  preview,
  form,
  busy,
  onForm,
  onMapping,
  onImport,
  onLoadExample,
  onNext,
}: {
  preview: CsvPreview | null;
  form: WorkspaceForm;
  busy: boolean;
  onForm: (update: Partial<WorkspaceForm>) => void;
  onMapping: (role: keyof ColumnMapping, value: string) => void;
  onImport: () => void;
  onLoadExample: () => void;
  onNext: () => void;
}) {
  const complete = preview !== null
    && mappingIsComplete(form.mapping, form.additionalPredictors);
  const assignedColumns = new Set([
    ...Object.values(form.mapping).filter(Boolean),
    ...form.additionalPredictors.map((item) => item.column),
  ]);
  const optionalColumns = preview?.columns.filter(
    (column) => !assignedColumns.has(column),
  ) ?? [];
  const addPredictor = (column: string) => {
    if (!column) return;
    let name = column
      .trim()
      .toLowerCase()
      .replace(/[^a-z0-9_]+/g, "_")
      .replace(/^([^a-z])/, "x_$1")
      .replace(/^_+|_+$/g, "") || "predictor";
    const used = new Set(form.additionalPredictors.map((item) => item.name));
    const base = name;
    let suffix = 2;
    while (used.has(name)) {
      name = `${base}_${suffix}`;
      suffix += 1;
    }
    onForm({
      additionalPredictors: [
        ...form.additionalPredictors,
        { name, column, unit: "unspecified", missing: "error" },
      ],
    });
  };
  const updatePredictor = (
    index: number,
    update: Partial<WorkspaceForm["additionalPredictors"][number]>,
  ) => {
    const previousName = form.additionalPredictors[index]?.name;
    const nextName = update.name ?? previousName;
    const rename = (role: string) => (
      previousName && nextName && role === previousName ? nextName : role
    );
    onForm({
      additionalPredictors: form.additionalPredictors.map((item, position) => (
        position === index ? { ...item, ...update } : item
      )),
      ...(previousName !== nextName ? {
        mlpFeatureRoles: form.mlpFeatureRoles.map(rename),
        pamlInputRoles: form.pamlInputRoles.map(rename),
        laggedFeatures: form.laggedFeatures.map((item) => ({
          ...item,
          source: rename(item.source),
        })),
      } : {}),
    });
  };
  const removePredictor = (index: number) => {
    const removedName = form.additionalPredictors[index]?.name;
    onForm({
      additionalPredictors: form.additionalPredictors.filter(
        (_, position) => position !== index,
      ),
      mlpFeatureRoles: form.mlpFeatureRoles.filter((role) => role !== removedName),
      pamlInputRoles: form.pamlInputRoles.filter((role) => role !== removedName),
      laggedFeatures: form.laggedFeatures.filter((item) => item.source !== removedName),
    });
  };

  return (
    <>
      <PageHeader
        eyebrow="Experiment configuration"
        title="Import and map basin data"
        description="Import a daily CSV, review its structure, and assign each column a hydrological role before analysis."
        actions={
          <button className="button button-secondary" onClick={onLoadExample} disabled={busy}>
            <FlaskConical size={16} />
            Load example dataset
          </button>
        }
      />

      <div className="content-grid data-grid">
        <div className="stack">
          <section className="card">
            <div className="card-heading">
              <div className="icon-tile cyan"><Upload size={19} /></div>
              <div>
                <h2>Data source</h2>
                <p>CSV files remain inside this project workspace.</p>
              </div>
            </div>

            {!preview ? (
              <button className="dropzone" onClick={onImport} disabled={busy}>
                <span className="dropzone-icon"><Upload size={24} /></span>
                <strong>{busy ? "Reading the selected file…" : "Choose a CSV file"}</strong>
                <span>UTF-8 · up to 25 MB · no data leaves this computer</span>
              </button>
            ) : (
              <div className="file-summary">
                <div className="file-icon"><FileText size={22} /></div>
                <div className="file-meta">
                  <strong>{preview.relative_path.split("/").at(-1)}</strong>
                  <span>{preview.row_count.toLocaleString()} rows · {preview.columns.length} columns · {formatBytes(preview.file_size_bytes)}</span>
                  <small>SHA-256 {preview.sha256.slice(0, 12)}…</small>
                </div>
                <button className="icon-button" onClick={onImport} title="Replace data file">
                  <RotateCcw size={17} />
                </button>
              </div>
            )}
          </section>

          <section className="card">
            <div className="card-heading compact">
              <div>
                <p className="section-kicker">Identity</p>
                <h2>Experiment details</h2>
              </div>
            </div>
            <div className="field-grid two-columns">
              <Field label="Experiment name">
                <input
                  value={form.name}
                  onChange={(event) => onForm({ name: event.target.value })}
                />
              </Field>
              <Field label="Reproducible ID">
                <input
                  value={form.experimentId}
                  onChange={(event) => onForm({ experimentId: event.target.value })}
                />
              </Field>
            </div>
            <Field label="Description">
              <textarea
                rows={3}
                value={form.description}
                onChange={(event) => onForm({ description: event.target.value })}
              />
            </Field>
          </section>

          <section className={`card ${preview ? "" : "card-disabled"}`}>
            <div className="card-heading compact">
              <div>
                <p className="section-kicker">Standard variables</p>
                <h2>Column mapping</h2>
              </div>
              {complete && <span className="status-pill success"><Check size={13} /> Complete</span>}
            </div>
            <div className="mapping-list">
              {(
                [
                  ["timestamp", "Time", "Optional ISO 8601 date"],
                  ["precipitation", "Precipitation", "Required · ≥ 0"],
                  ["temperature", "Temperature", "Required"],
                  ["potential_evapotranspiration", "Potential ET", "Required · ≥ 0"],
                  ["streamflow", "Observed streamflow", "Required target"],
                ] as Array<[keyof ColumnMapping, string, string]>
              ).map(([role, label, detail]) => (
                <div className="mapping-row" key={role}>
                  <div><strong>{label}</strong><span>{detail}</span></div>
                  <ArrowRight size={15} />
                  <select
                    value={form.mapping[role]}
                    disabled={!preview}
                    onChange={(event) => onMapping(role, event.target.value)}
                  >
                    <option value="">{role === "timestamp" ? "Not assigned" : "Select column"}</option>
                    {preview?.columns.map((column) => <option key={column}>{column}</option>)}
                  </select>
                </div>
              ))}
            </div>
          </section>

          <section className={`card ${preview ? "" : "card-disabled"}`}>
            <div className="card-heading compact">
              <div>
                <p className="section-kicker">Optional additional predictors</p>
                <h2>Additional numeric variables</h2>
                <p>Expose any other measured or derived CSV column to neural and PaML models.</p>
              </div>
              <span className="status-pill neutral">
                {form.additionalPredictors.length} mapped
              </span>
            </div>
            <div className="additional-predictor-picker">
              <select
                aria-label="Add additional predictor column"
                value=""
                disabled={!preview || optionalColumns.length === 0}
                onChange={(event) => addPredictor(event.target.value)}
              >
                <option value="">
                  {optionalColumns.length ? "Choose another CSV column" : "No unused columns"}
                </option>
                {optionalColumns.map((column) => <option key={column}>{column}</option>)}
              </select>
            </div>
            {form.additionalPredictors.length > 0 ? (
              <div className="additional-predictor-list">
                {form.additionalPredictors.map((item, index) => (
                  <div className="additional-predictor-row" key={`${item.column}-${index}`}>
                    <div className="predictor-source">
                      <span>CSV column</span><strong>{item.column}</strong>
                    </div>
                    <Field label="Feature name" hint="Stable model identifier">
                      <input
                        value={item.name}
                        onChange={(event) => updatePredictor(index, { name: event.target.value })}
                      />
                    </Field>
                    <Field label="Unit">
                      <input
                        value={item.unit}
                        onChange={(event) => updatePredictor(index, { unit: event.target.value })}
                      />
                    </Field>
                    <Field label="Missing values">
                      <select
                        value={item.missing}
                        onChange={(event) => updatePredictor(index, {
                          missing: event.target.value as typeof item.missing,
                        })}
                      >
                        <option value="error">Stop with error</option>
                        <option value="linear_interpolation">Interpolate short gaps</option>
                      </select>
                    </Field>
                    <button
                      type="button"
                      className="icon-button"
                      aria-label={`Remove ${item.name || item.column}`}
                      onClick={() => removePredictor(index)}
                    >
                      <Trash2 size={16} />
                    </button>
                  </div>
                ))}
              </div>
            ) : (
              <div className="inline-note">
                <Info size={16} />
                <span>This is optional. Conceptual models still use the standard precipitation, temperature, and PET forcings.</span>
              </div>
            )}
          </section>
        </div>

        <section className="card preview-card">
          <div className="card-heading compact">
            <div>
              <p className="section-kicker">Read-only inspection</p>
              <h2>Data preview</h2>
            </div>
            {preview && <span className="status-pill neutral">First {preview.rows.length} rows</span>}
          </div>
          {preview ? (
            <div className="table-scroll">
              <table className="data-table">
                <thead><tr>{preview.columns.map((column) => <th key={column}>{column}</th>)}</tr></thead>
                <tbody>
                  {preview.rows.map((row, index) => (
                    <tr key={index}>{preview.columns.map((column) => <td key={column}>{row[column] || <span className="missing">missing</span>}</td>)}</tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <div className="empty-state">
              <div className="empty-illustration"><Database size={30} /></div>
              <h3>No data loaded yet</h3>
              <p>Select a CSV or open the deterministic example. Nothing is processed until you validate the experiment.</p>
            </div>
          )}
          <div className="inline-note">
            <Info size={16} />
            <span>The imported file is left unchanged. Any cleaning or unit conversion is recorded and applied only when the experiment is prepared.</span>
          </div>
        </section>
      </div>

      <div className="step-footer">
        <span>{complete ? "Data roles and optional predictors are ready." : "Import data and complete the four required roles."}</span>
        <button className="button button-primary" disabled={!complete} onClick={onNext}>
          Check data quality <ArrowRight size={16} />
        </button>
      </div>
    </>
  );
}

export function QualityPanel({
  form,
  report,
  busy,
  onForm,
  onAnalyze,
  onNext,
}: {
  form: WorkspaceForm;
  report: DataQualityReport | null;
  busy: boolean;
  onForm: (update: Partial<WorkspaceForm>) => void;
  onAnalyze: () => void;
  onNext: () => void;
}) {
  const unitSetupValid = form.streamflowUnit !== "m3/s" || (form.basinAreaKm2 ?? 0) > 0;
  const statusLabel = report
    ? report.ready
      ? report.warning_count
        ? "Ready with documented warnings"
        : "Ready for modelling"
      : "Corrections are required"
    : "Quality checks have not run";

  return (
    <>
      <PageHeader
        eyebrow="Data validation"
        title="Review units, completeness, and time alignment"
        description="Profile the complete series, declare missing-value rules, and verify the daily time axis before model execution."
        actions={
          <button className="button button-primary" disabled={busy || !unitSetupValid} onClick={onAnalyze}>
            <Activity size={16} /> {busy ? "Analyzing…" : "Run quality checks"}
          </button>
        }
      />

      <div className="content-grid quality-grid">
        <div className="stack">
          <section className="card quality-settings-card">
            <div className="card-heading">
              <div className="icon-tile cyan"><Gauge size={19} /></div>
              <div><h2>Source units</h2><p>All model forcings and streamflow are converted to the daily millimetre units.</p></div>
            </div>
            <div className="field-grid four-columns">
              <Field label="Precipitation">
                <select value={form.precipitationUnit} onChange={(event) => onForm({ precipitationUnit: event.target.value as WorkspaceForm["precipitationUnit"] })}>
                  <option value="mm/day">mm/day</option><option value="inch/day">inch/day</option>
                </select>
              </Field>
              <Field label="Temperature">
                <select value={form.temperatureUnit} onChange={(event) => onForm({ temperatureUnit: event.target.value as WorkspaceForm["temperatureUnit"] })}>
                  <option value="degC">°C</option><option value="degF">°F</option><option value="K">K</option>
                </select>
              </Field>
              <Field label="Potential ET">
                <select value={form.potentialEvapotranspirationUnit} onChange={(event) => onForm({ potentialEvapotranspirationUnit: event.target.value as WorkspaceForm["potentialEvapotranspirationUnit"] })}>
                  <option value="mm/day">mm/day</option><option value="inch/day">inch/day</option>
                </select>
              </Field>
              <Field label="Streamflow">
                <select value={form.streamflowUnit} onChange={(event) => onForm({ streamflowUnit: event.target.value as WorkspaceForm["streamflowUnit"] })}>
                  <option value="mm/day">mm/day</option><option value="m3/s">m³/s</option>
                </select>
              </Field>
            </div>
            {form.streamflowUnit === "m3/s" && (
              <Field label="Basin area" hint="Required for Q: m³/s → mm/day">
                <input type="number" min="0.000001" step="0.1" value={form.basinAreaKm2 ?? ""} onChange={(event) => onForm({ basinAreaKm2: event.target.value ? Number(event.target.value) : null })} placeholder="Area in km²" />
              </Field>
            )}
          </section>

          <section className="card quality-settings-card">
            <div className="card-heading">
              <div className="icon-tile purple"><Settings2 size={19} /></div>
              <div><h2>Missing values</h2><p>Interpolation is restricted to short interior gaps and never extends the ends of a series.</p></div>
            </div>
            <div className="field-grid four-columns">
              <Field label="Precipitation">
                <select value={form.precipitationMissing} onChange={(event) => onForm({ precipitationMissing: event.target.value as WorkspaceForm["precipitationMissing"] })}>
                  <option value="error">Stop with error</option><option value="zero">Fill with zero</option><option value="linear_interpolation">Interpolate</option>
                </select>
              </Field>
              <Field label="Temperature">
                <select value={form.temperatureMissing} onChange={(event) => onForm({ temperatureMissing: event.target.value as WorkspaceForm["temperatureMissing"] })}>
                  <option value="error">Stop with error</option><option value="linear_interpolation">Interpolate</option>
                </select>
              </Field>
              <Field label="Potential ET">
                <select value={form.potentialEvapotranspirationMissing} onChange={(event) => onForm({ potentialEvapotranspirationMissing: event.target.value as WorkspaceForm["potentialEvapotranspirationMissing"] })}>
                  <option value="error">Stop with error</option><option value="zero">Fill with zero</option><option value="linear_interpolation">Interpolate</option>
                </select>
              </Field>
              <Field label="Observed streamflow">
                <select value={form.streamflowMissing} onChange={(event) => onForm({ streamflowMissing: event.target.value as WorkspaceForm["streamflowMissing"] })}>
                  <option value="exclude">Exclude from score</option><option value="linear_interpolation">Interpolate explicitly</option>
                </select>
              </Field>
            </div>
            <div className="field-grid three-columns quality-secondary-fields">
              <Field label="Maximum interpolation gap" hint="Consecutive daily steps">
                <input type="number" min="1" max="366" value={form.maximumInterpolationGap} onChange={(event) => onForm({ maximumInterpolationGap: Number(event.target.value) })} />
              </Field>
              <Field label="Text missing tokens" hint="Comma-separated, blank first token allowed">
                <input value={form.missingTextTokens} onChange={(event) => onForm({ missingTextTokens: event.target.value })} />
              </Field>
              <Field label="Numeric sentinels" hint="Example: -9999, -999">
                <input value={form.numericSentinels} onChange={(event) => onForm({ numericSentinels: event.target.value })} placeholder="None" />
              </Field>
            </div>
          </section>

          <section className="card quality-settings-card">
            <div className="card-heading">
              <div className="icon-tile amber"><CircleAlert size={19} /></div>
              <div><h2>Physical checks</h2><p>Negative values are flagged first, then handled only by the selected rule.</p></div>
            </div>
            <div className="field-grid three-columns">
              <Field label="Negative precipitation">
                <select value={form.precipitationNegative} onChange={(event) => onForm({ precipitationNegative: event.target.value as WorkspaceForm["precipitationNegative"] })}>
                  <option value="error">Stop with error</option><option value="clip_to_zero">Clip to zero</option><option value="set_missing">Treat as missing</option>
                </select>
              </Field>
              <Field label="Negative potential ET">
                <select value={form.potentialEvapotranspirationNegative} onChange={(event) => onForm({ potentialEvapotranspirationNegative: event.target.value as WorkspaceForm["potentialEvapotranspirationNegative"] })}>
                  <option value="error">Stop with error</option><option value="clip_to_zero">Clip to zero</option><option value="set_missing">Treat as missing</option>
                </select>
              </Field>
              <Field label="Negative streamflow">
                <select value={form.streamflowNegative} onChange={(event) => onForm({ streamflowNegative: event.target.value as WorkspaceForm["streamflowNegative"] })}>
                  <option value="keep">Keep and warn</option><option value="error">Stop with error</option><option value="exclude">Exclude from score</option><option value="clip_to_zero">Clip to zero</option>
                </select>
              </Field>
            </div>
            <label className="toggle-row">
              <input type="checkbox" checked={form.requireTimestamp} onChange={(event) => onForm({ requireTimestamp: event.target.checked })} />
              <span><strong>Require an explicit timestamp column</strong><small>Daily regularity is otherwise assumed from row order and reported as a warning.</small></span>
            </label>
          </section>
        </div>

        <aside className="card quality-report-card">
          <div className="card-heading compact">
            <div><p className="section-kicker">Audit report</p><h2>{statusLabel}</h2></div>
            {report && <span className={`status-pill ${report.ready ? "success" : "danger"}`}>{report.ready ? <CircleCheck size={13} /> : <CircleAlert size={13} />}{report.ready ? "Ready" : `${report.error_count} error${report.error_count === 1 ? "" : "s"}`}</span>}
          </div>
          {!report ? (
            <div className="quality-empty">
              <Activity size={30} /><strong>No audit report yet</strong><p>Run quality checks after declaring how the source values should be interpreted.</p>
            </div>
          ) : (
            <>
              <div className="quality-kpis">
                <div><span>Rows</span><strong>{report.row_count.toLocaleString()}</strong></div>
                <div><span>Warnings</span><strong>{report.warning_count}</strong></div>
                <div><span>Changed values</span><strong>{report.transformed_value_count.toLocaleString()}</strong></div>
              </div>
              <div className="quality-time">
                <span>Time coverage</span>
                <strong>{report.timestamps.assigned ? `${report.timestamps.first ?? "?"} → ${report.timestamps.last ?? "?"}` : "Daily order assumed"}</strong>
                <small>{report.timestamps.assigned ? `${report.timestamps.irregular_step_count} irregular intervals` : "No timestamp column assigned"}</small>
              </div>
              <div className="table-scroll quality-table-scroll">
                <table className="data-table quality-table">
                  <thead><tr><th>Variable</th><th>Usable</th><th>Missing</th><th>Invalid</th><th>Range after rules</th><th>Changed</th></tr></thead>
                  <tbody>{report.variables.map((variable) => (
                    <tr key={variable.variable}>
                      <td><strong>{variable.variable.replaceAll("_", " ")}</strong><small>{variable.source_unit} → {variable.model_unit}</small></td>
                      <td>{variable.processed_valid_count.toLocaleString()}</td>
                      <td>{variable.processed_missing_count.toLocaleString()}</td>
                      <td>{variable.raw_invalid_count.toLocaleString()}</td>
                      <td>{formatRange(variable.processed_minimum, variable.processed_maximum)}</td>
                      <td>{variable.transformed_count.toLocaleString()}</td>
                    </tr>
                  ))}</tbody>
                </table>
              </div>
              <div className="quality-issues">
                {report.issues.length === 0 ? <div className="quality-clean"><CircleCheck size={16} /> No issue detected.</div> : report.issues.map((issue) => (
                  <article className={`quality-issue ${issue.severity}`} key={`${issue.code}-${issue.variable ?? "global"}`}>
                    {issue.severity === "error" ? <CircleAlert size={16} /> : <Info size={16} />}
                    <div><strong>{issue.summary}</strong><p>{issue.explanation}</p><small>{issue.suggested_action}{issue.csv_rows.length ? ` Example CSV rows: ${issue.csv_rows.join(", ")}.` : ""}</small></div>
                  </article>
                ))}
              </div>
            </>
          )}
        </aside>
      </div>

      {report && <section className="card exploratory-analysis-card">
        <div className="card-heading">
          <div className="icon-tile purple"><BarChart3 size={19} /></div>
          <div>
            <p className="section-kicker">Exploratory data analysis</p>
            <h2>Distributions and pairwise relationships</h2>
            <p>Descriptive only: these summaries never select, transform, or discard features automatically.</p>
          </div>
        </div>
        <div className="eda-grid">
          <div className="eda-panel">
            <div className="eda-section-heading"><strong>Processed-variable summary</strong><span>Complete series after declared quality rules</span></div>
            <div className="table-scroll">
              <table className="data-table eda-table">
                <thead><tr><th>Variable</th><th>Mean</th><th>Std.</th><th>Q25</th><th>Median</th><th>Q75</th></tr></thead>
                <tbody>{report.variables.map((variable) => (
                  <tr key={variable.variable}>
                    <td><strong>{variable.variable.replaceAll("_", " ")}</strong><small>{variable.model_unit}</small></td>
                    <td>{formatDecimal(variable.processed_mean ?? Number.NaN, 3)}</td>
                    <td>{formatDecimal(variable.processed_standard_deviation ?? Number.NaN, 3)}</td>
                    <td>{formatDecimal(variable.processed_q25 ?? Number.NaN, 3)}</td>
                    <td>{formatDecimal(variable.processed_median ?? Number.NaN, 3)}</td>
                    <td>{formatDecimal(variable.processed_q75 ?? Number.NaN, 3)}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          </div>
          <div className="eda-panel">
            <div className="eda-section-heading"><strong>Pearson correlation matrix</strong><span>Pairwise finite rows; “—” denotes a constant or insufficient series</span></div>
            <div className="table-scroll correlation-scroll">
              <table className="data-table correlation-table">
                <thead><tr><th>Variable</th>{report.correlation_variables.map((name) => <th key={name} title={name}>{name.replaceAll("_", " ")}</th>)}</tr></thead>
                <tbody>{report.correlation_variables.map((name, row) => (
                  <tr key={name}>
                    <td><strong>{name.replaceAll("_", " ")}</strong></td>
                    {report.correlation_matrix[row]?.map((value, column) => (
                      <td key={`${name}-${report.correlation_variables[column]}`} className={value !== null && Math.abs(value) >= 0.7 ? "strong-correlation" : ""}>
                        {value === null ? "—" : formatDecimal(value, 2)}
                      </td>
                    ))}
                  </tr>
                ))}</tbody>
              </table>
            </div>
          </div>
        </div>
        <div className="inline-note"><Info size={16} /><span>Correlation is a descriptive linear association and does not establish causality or select features. Training-only scaling remains independent of this report.</span></div>
      </section>}

      <div className="step-footer">
        <span>{report?.ready ? "The processed data record is ready and reproducible." : "A successful audit is required before model configuration."}</span>
        <button className="button button-primary" disabled={!report?.ready} onClick={onNext}>Configure model <ArrowRight size={16} /></button>
      </div>
    </>
  );
}

export function ModelPanel({
  form,
  onForm,
  onNext,
}: {
  form: WorkspaceForm;
  onForm: (update: Partial<WorkspaceForm>) => void;
  onNext: () => void;
}) {
  const [lagDraftSource, setLagDraftSource] = useState("precipitation");
  const [lagDraftSteps, setLagDraftSteps] = useState(1);
  const isMlp = isMlpModel(form.modelId);
  const isLstm = isLstmModel(form.modelId);
  const isGru = isGruModel(form.modelId);
  const isCnn = isCnnModel(form.modelId);
  const isPaml = isPamlModel(form.modelId);
  const isDiscovery = isDiscoveryModel(form.modelId);
  const isRecurrent = isLstm || isGru;
  const isStackedRecurrent = form.modelId === "lstm.tensorflow_stacked_s2v.v2"
    || form.modelId === "gru.tensorflow_stacked_s2v.v2"
    || form.modelId === "lstm.tensorflow_stacked_s2s.v3"
    || form.modelId === "gru.tensorflow_stacked_s2s.v3";
  const isS2S = isS2SNeuralModel(form.modelId);
  const configuredRecurrentLayers = form.recurrentLayers
    .split(",")
    .map((value) => Number(value.trim()))
    .filter((value) => Number.isFinite(value));
  const recurrentLayers = (
    form.modelId === "lstm.tensorflow_s2v.v1"
    || form.modelId === "gru.tensorflow_s2v.v1"
      ? [form.recurrentUnits]
      : configuredRecurrentLayers
  );
  const isNeural = isMlp || isRecurrent || isCnn;
  const availableExternalFeatures = [
    { role: "precipitation", label: "Precipitation", unit: "mm/day" },
    { role: "temperature", label: "Temperature", unit: "°C" },
    { role: "potential_evapotranspiration", label: "Potential ET", unit: "mm/day" },
    ...form.additionalPredictors.map((item) => ({
      role: item.name,
      label: item.name.replaceAll("_", " "),
      unit: item.unit,
    })),
  ];
  const snowEnabled = form.snowModule !== "snow.none.v1";
  const isHbv = form.modelId === "hbv.mass_conservative_daily.v1";
  const isExpHydro = form.modelId === "exp_hydro.mass_conservative.v1";
  const routingEnabled = isHbv && form.routingProcess === "routing.maxbas.triangular.v1";
  const modelName = isDiscovery ? "DeepDiscover" : isMlp ? "MLP" : isLstm ? "LSTM" : isGru ? "GRU" : isCnn ? "CNN-1D" : isHbv ? "HBV" : "EXP-HYDRO";
  const stateSummary = routingEnabled ? "4 + 6 routing-memory" : String(isHbv ? 4 : 2);
  const fluxCount = isHbv ? (routingEnabled ? 14 : 13) : 7;
  const updateSnowTreatment = (
    key: SnowTreatmentKey,
    update: Partial<WorkspaceForm["snowCalibration"][SnowTreatmentKey]>,
  ) => onForm({
    snowCalibration: {
      ...form.snowCalibration,
      [key]: { ...form.snowCalibration[key], ...update },
    },
  });
  const activeTreatments = activeSnowTreatmentKeys(form);
  const snowParametersValid = activeTreatments.every((key) => {
    const treatment = form.snowCalibration[key];
    const value = key === "snowfallTemperature" ? form.snowfallTemperature
      : key === "meltTemperature" ? form.meltTemperature
      : key === "degreeDayFactor" ? form.degreeDayFactor
      : form.sharpness;
    return Number.isFinite(value) && (
      treatment.mode === "fixed" || (
        treatment.lowerBound < treatment.upperBound
        && value >= treatment.lowerBound
        && value <= treatment.upperBound
      )
    );
  }) && form.degreeDayFactor >= 0 && form.sharpness > 0;
  const routingParametersValid = !routingEnabled || (
    Number.isInteger(form.maxbas)
    && form.maxbas >= 1
    && form.maxbas <= 7
    && form.initialRoutingState.every((value) => Number.isFinite(value) && value >= 0)
    && (
      form.maxbasTreatment.mode === "fixed"
      || (
        Number.isInteger(form.maxbasTreatment.lowerBound)
        && Number.isInteger(form.maxbasTreatment.upperBound)
        && form.maxbasTreatment.lowerBound >= 1
        && form.maxbasTreatment.upperBound <= 7
        && form.maxbasTreatment.lowerBound < form.maxbasTreatment.upperBound
        && form.maxbas >= form.maxbasTreatment.lowerBound
        && form.maxbas <= form.maxbasTreatment.upperBound
      )
    )
  );
  const modelParametersValid = snowParametersValid && routingParametersValid;
  if (isDiscovery) {
    const lossValues = [
      form.discoveryDecorrelationWeight,
      form.discoveryMassBalanceWeight,
      form.discoverySmoothnessWeight,
      form.discoverySmoothnessThreshold,
      form.discoveryBinaryWeight,
      form.discoverySparsityWeight,
    ];
    const discoveryIssues: string[] = [];
    if (
      !Number.isInteger(form.discoveryProcessCount)
      || form.discoveryProcessCount < 1
      || form.discoveryProcessCount > 8
    ) {
      discoveryIssues.push("Discovered processes must be an integer from 1 to 8.");
    }
    if (
      !Number.isInteger(form.discoveryHiddenUnits)
      || form.discoveryHiddenUnits < 1
      || form.discoveryHiddenUnits > 256
    ) {
      discoveryIssues.push("Hidden units must be an integer from 1 to 256.");
    }
    if (!Number.isFinite(form.initialSnowStorage) || form.initialSnowStorage < 0) {
      discoveryIssues.push("Initial snow storage must be a finite non-negative value.");
    }
    if (!Number.isFinite(form.initialSoilStorage) || form.initialSoilStorage < 0) {
      discoveryIssues.push("Initial discovered storage must be a finite non-negative value.");
    }
    if (form.snowModule === "snow.none.v1" && form.initialSnowStorage !== 0) {
      discoveryIssues.push("Initial snow storage must be zero when the snow module is disabled.");
    }
    if (!Number.isFinite(form.degreeDayFactor) || form.degreeDayFactor < 0) {
      discoveryIssues.push("The degree-day factor must be finite and non-negative.");
    }
    if (!Number.isFinite(form.sharpness) || form.sharpness <= 0) {
      discoveryIssues.push("Transition sharpness must be finite and greater than zero.");
    }
    if (!lossValues.every((value) => Number.isFinite(value) && value >= 0)) {
      discoveryIssues.push("Every regularization weight and threshold must be finite and non-negative.");
    }
    if (
      !Number.isFinite(form.discoveryTargetCorrelation)
      || form.discoveryTargetCorrelation < -1
      || form.discoveryTargetCorrelation > 1
    ) {
      discoveryIssues.push("Target process correlation must lie between −1 and 1.");
    }
    const discoveryValid = discoveryIssues.length === 0;
    const library: Array<[WorkspaceForm["modelId"], string, string]> = [
      ["exp_hydro.mass_conservative.v1", "EXP-HYDRO", "Conceptual"],
      ["hbv.mass_conservative_daily.v1", "HBV", "Conceptual"],
      ["mlp.tensorflow_s2v.v1", "MLP", "Neural"],
      ["lstm.tensorflow_stacked_s2v.v2", "LSTM", "Neural"],
      ["gru.tensorflow_stacked_s2v.v2", "GRU", "Neural"],
      ["cnn1d.tensorflow_s2v.v1", "CNN-1D", "Neural"],
      ["paml.tensorflow_prnn_cnn_s2s.v1", "PaML", "Hybrid"],
      ["deepdiscover.numpy_instantaneous_s2s.v1", "DeepDiscover", "Discovery"],
    ];
    const regularizationFields = (
      <div className="field-grid adaptive-fields discovery-loss-grid">
        <Field label="Decorrelation weight"><input type="number" min="0" step="0.001" value={form.discoveryDecorrelationWeight} onChange={(event) => onForm({ discoveryDecorrelationWeight: Number(event.target.value) })} /></Field>
        <Field label="Target process correlation"><input type="number" min="-1" max="1" step="0.05" value={form.discoveryTargetCorrelation} onChange={(event) => onForm({ discoveryTargetCorrelation: Number(event.target.value) })} /></Field>
        <Field label="Smoothness weight"><input type="number" min="0" step="0.001" value={form.discoverySmoothnessWeight} onChange={(event) => onForm({ discoverySmoothnessWeight: Number(event.target.value) })} /></Field>
        <Field label="Smoothness threshold" hint="mm per interval"><input type="number" min="0" step="0.01" value={form.discoverySmoothnessThreshold} onChange={(event) => onForm({ discoverySmoothnessThreshold: Number(event.target.value) })} /></Field>
        <Field label="Binary-relaxation weight"><input type="number" min="0" step="0.001" value={form.discoveryBinaryWeight} onChange={(event) => onForm({ discoveryBinaryWeight: Number(event.target.value) })} /></Field>
        <Field label="Sparsity weight"><input type="number" min="0" step="0.001" value={form.discoverySparsityWeight} onChange={(event) => onForm({ discoverySparsityWeight: Number(event.target.value) })} /></Field>
        <Field label="Mass-balance weight" hint="Logged; balance is structural"><input type="number" min="0" step="0.01" value={form.discoveryMassBalanceWeight} onChange={(event) => onForm({ discoveryMassBalanceWeight: Number(event.target.value) })} /></Field>
      </div>
    );
    return <>
      <PageHeader
        eyebrow="Process discovery"
        title="Discover an instantaneous bucket-type model"
        description="Learn active inputs, a storage-dependent mapping, and multiple non-negative discharge components while preserving the interval water balance by construction."
      />
      <div className="content-grid model-grid">
        <div className="stack">
          <section className="card model-identity">
            <div className="model-title-row">
              <div className="model-mark"><FlaskConical size={25} /></div>
              <div><p className="section-kicker">Discovery model</p><h2>DeepDiscover</h2><span>Instantaneous storage-only S2S variant · v1.0.0</span></div>
              <span className="status-pill success"><CircleCheck size={14} /> Available</span>
            </div>
            <div className="capability-row">
              <span>{form.discoveryProcessCount} discovered processes</span>
              <span>1 recurrent storage</span>
              <span>same-time S2S</span>
              <span>exact analytical balance</span>
            </div>
            <div className="model-note"><Info size={17} /><p>This single-DU baseline carries hydrological memory through the discovered storage. It contains no separate process-memory or routing state.</p></div>
          </section>

          <section className="card">
            <div className="card-heading compact"><div><p className="section-kicker">Model library</p><h2>Available structures</h2></div></div>
            <div className="choice-grid model-library-grid">
              {library.map(([id, label, family]) => <button key={id} type="button" className={`choice-card ${form.modelId === id ? "selected" : ""}`} onClick={() => onForm({ modelId: id })}>
                {family === "Discovery" ? <FlaskConical size={20} /> : family === "Conceptual" ? <Waves size={20} /> : <BrainCircuit size={20} />}
                <strong>{label}</strong><span>{family}</span>{form.modelId === id && <i><Check size={12} /></i>}
              </button>)}
            </div>
          </section>

          <section className="card">
            <div className="card-heading compact"><div><p className="section-kicker">Composable process</p><h2>Snow module and parameter treatment</h2><p>Fix known values or estimate selected snow parameters together with the DD mapping.</p></div></div>
            <div className="choice-grid adaptive-choice-grid">
              {([
                ["snow.none.v1", "None", "All precipitation is rain."],
                ["snow.degree_day.hard.v1", "Degree-day", "Hard temperature thresholds."],
                ["snow.degree_day.smooth.v1", "Smooth degree-day", "Continuous threshold transitions."],
              ] as const).map(([id, label, detail]) => <button key={id} type="button" className={`choice-card ${form.snowModule === id ? "selected" : ""}`} onClick={() => onForm({ snowModule: id, ...(id === "snow.none.v1" ? { initialSnowStorage: 0 } : {}) })}><Snowflake size={20} /><strong>{label}</strong><span>{detail}</span>{form.snowModule === id && <i><Check size={12} /></i>}</button>)}
            </div>
            {form.snowModule !== "snow.none.v1" && <div className="parameters-block snow-parameters">
              <div className="snow-parameters-heading"><div><strong>Parameter treatment</strong><span>Every estimated value remains inside explicit bounds and is selected with the validation checkpoint.</span></div><span className="status-pill neutral">{activeTreatments.filter((key) => form.snowCalibration[key].mode === "calibrate").length} estimated</span></div>
              <ParameterTreatmentControl label="Snowfall threshold" unit="°C" value={form.snowfallTemperature} treatment={form.snowCalibration.snowfallTemperature} onValue={(value) => onForm({ snowfallTemperature: value })} onTreatment={(update) => updateSnowTreatment("snowfallTemperature", update)} />
              <ParameterTreatmentControl label="Melt threshold" unit="°C" value={form.meltTemperature} treatment={form.snowCalibration.meltTemperature} onValue={(value) => onForm({ meltTemperature: value })} onTreatment={(update) => updateSnowTreatment("meltTemperature", update)} />
              <ParameterTreatmentControl label="Degree-day factor" unit="mm °C⁻¹ day⁻¹" value={form.degreeDayFactor} treatment={form.snowCalibration.degreeDayFactor} onValue={(value) => onForm({ degreeDayFactor: value })} onTreatment={(update) => updateSnowTreatment("degreeDayFactor", update)} />
              {form.snowModule === "snow.degree_day.smooth.v1" && <ParameterTreatmentControl label="Transition sharpness" unit="°C⁻¹" value={form.sharpness} treatment={form.snowCalibration.sharpness} onValue={(value) => onForm({ sharpness: value })} onTreatment={(update) => updateSnowTreatment("sharpness", update)} />}
              {activeTreatments.some((key) => form.snowCalibration[key].mode === "calibrate") && <div className="inline-note"><Info size={16} /><span>Snow parameters use bounded alternating coordinate calibration. The learned mapping still uses full-sequence Adam; the joint checkpoint is chosen on validation only.</span></div>}
            </div>}
          </section>

          <section className="card">
            <div className="card-heading compact"><div><p className="section-kicker">Learned mapping</p><h2>Discovery dimensions and states</h2></div></div>
            <div className="field-grid adaptive-fields discovery-mapping-grid">
              <Field label="Discovered processes" hint="Independent non-negative components"><input type="number" min="1" max="8" step="1" value={form.discoveryProcessCount} onChange={(event) => onForm({ discoveryProcessCount: Number(event.target.value) })} /></Field>
              <Field label="Hidden units" hint="Single-DU storage-dependent mapping"><input type="number" min="1" max="256" step="1" value={form.discoveryHiddenUnits} onChange={(event) => onForm({ discoveryHiddenUnits: Number(event.target.value) })} /></Field>
              <Field label="Initial snow storage" hint="First boundary · mm"><input type="number" min="0" disabled={form.snowModule === "snow.none.v1"} value={form.initialSnowStorage} onChange={(event) => onForm({ initialSnowStorage: Number(event.target.value) })} /></Field>
              <Field label="Initial discovered storage" hint="First boundary · mm"><input type="number" min="0" value={form.initialSoilStorage} onChange={(event) => onForm({ initialSoilStorage: Number(event.target.value) })} /></Field>
            </div>
            <div className="discovery-switch-grid">
              <label className="toggle-row"><input type="checkbox" checked={form.discoveryTrainInputSwitches} onChange={(event) => onForm({ discoveryTrainInputSwitches: event.target.checked })} /><span><strong>Learn input switches</strong><small>Bounded rain, snowmelt, and PET participation probabilities.</small></span></label>
              <label className="toggle-row"><input type="checkbox" checked={form.discoveryTrainProcessSwitches} onChange={(event) => onForm({ discoveryTrainProcessSwitches: event.target.checked })} /><span><strong>Learn process switches</strong><small>Bounded activation probabilities for each discovered component. Leave disabled when all requested candidate processes should remain active.</small></span></label>
            </div>
          </section>

          <section className="card">
            <div className="card-heading compact"><div><p className="section-kicker">Scientific constraints</p><h2>Discovery regularization</h2><p>Defaults are ready to run. Every term remains archived independently for reproducibility.</p></div></div>
            {regularizationFields}
            <div className="inline-note"><Info size={16} /><span>Mass balance is enforced by differentiable water allocation, so its residual is expected to remain at numerical precision. The configured weight is retained for transparent loss accounting.</span></div>
          </section>
        </div>

        <section className="card process-card discovery-pathway-card">
          <div className="card-heading compact"><div><p className="section-kicker">Live structure</p><h2>DD pathway</h2></div></div>
          <div className="paml-pathway discovery-pathway">
            <div className="pathway-node"><CloudRain size={21} /><strong>Rain · snowmelt · PET</strong><span>Training-only normalization</span></div>
            <div className="pathway-arrow">↓</div>
            <div className="pathway-node exposed"><Settings2 size={21} /><strong>Active weighted inputs</strong><span>{form.discoveryTrainInputSwitches ? "Learned bounded switches" : "All inputs active"}</span></div>
            <div className="pathway-arrow">↓</div>
            <div className="pathway-node corrector"><BrainCircuit size={21} /><strong>Storage-dependent mapping</strong><span>{form.discoveryHiddenUnits} hidden units · storage recurrence</span></div>
            <div className="pathway-arrow">↓</div>
            <div className="pathway-node fusion"><FlaskConical size={21} /><strong>{form.discoveryProcessCount} discovered components</strong><span>Analytical allocation · Q = Σ processes · retained water updates storage</span></div>
          </div>
          <div className="model-note"><Info size={17} /><p>The process components, raw propensities, active inputs, state trajectory, switches, and interval balance residual are all saved for analysis.</p></div>
        </section>
      </div>
      {!discoveryValid && <div className="validation-notice" role="alert">
        <CircleAlert size={18} />
        <div>
          <strong>Complete the model definition</strong>
          <ul>{discoveryIssues.map((issue) => <li key={issue}>{issue}</li>)}</ul>
        </div>
      </div>}
      <div className="step-footer"><span>{discoveryValid ? "The DD structure is ready; numerical safeguards are handled consistently by the engine." : "Correct the specific issue listed above to continue."}</span><button className="button button-primary" disabled={!discoveryValid} onClick={onNext}>Set discovery training <ArrowRight size={16} /></button></div>
    </>;
  }
  if (isPaml) {
    const physicalLabel = form.pamlPhysicalModelId === "hbv.mass_conservative_daily.v1"
      ? "HBV"
      : "EXP-HYDRO";
    const firstCnn = form.cnnLayers[0] ?? { filters: 8, kernelSize: 10, activation: "relu" as const };
    const pamlValid = modelParametersValid
      && Number.isInteger(firstCnn.filters)
      && firstCnn.filters >= 1
      && firstCnn.filters <= 256
      && Number.isInteger(firstCnn.kernelSize)
      && firstCnn.kernelSize >= 1
      && firstCnn.kernelSize <= 365;
    const library: Array<[WorkspaceForm["modelId"], string, string]> = [
      ["exp_hydro.mass_conservative.v1", "EXP-HYDRO", "Conceptual"],
      ["hbv.mass_conservative_daily.v1", "HBV", "Conceptual"],
      ["lstm.tensorflow_stacked_s2v.v2", "LSTM", "Deep learning"],
      ["paml.tensorflow_prnn_cnn_s2s.v1", "PaML", "Hybrid"],
      ["deepdiscover.numpy_instantaneous_s2s.v1", "DeepDiscover", "Discovery"],
    ];
    return <>
      <PageHeader eyebrow="Hybrid structure" title="Configure PaML explicitly" description="Configure the P-RNN + causal CNN architecture by selecting its physical recurrent backbone and downstream correction." />
      <div className="content-grid model-grid">
        <div className="stack">
          <section className="card">
            <div className="card-heading compact"><div><p className="section-kicker">Model family</p><h2>Select a rainfall-runoff model</h2></div></div>
            <div className="choice-grid model-library-grid">
              {library.map(([id, label, family]) => <button type="button" key={id} className={`choice-card ${form.modelId === id ? "selected" : ""}`} onClick={() => onForm({ modelId: id })}><strong>{label}</strong><span>{family}</span>{form.modelId === id && <i><Check size={12} /></i>}</button>)}
            </div>
          </section>

          <section className="card">
            <div className="card-heading compact"><div><p className="section-kicker">Physical recurrent component</p><h2>Select the P-RNN backbone</h2><p>The hydrological parameters and the downstream convolutional layers are trained jointly.</p></div></div>
            <div className="choice-grid two-column">
              <button type="button" className={`choice-card ${form.pamlPhysicalModelId === "exp_hydro.mass_conservative.v1" ? "selected" : ""}`} onClick={() => onForm({ pamlPhysicalModelId: "exp_hydro.mass_conservative.v1" })}><Waves size={20} /><strong>EXP-HYDRO P-RNN</strong><span>Snow and soil storage with subsurface and surface runoff</span>{form.pamlPhysicalModelId === "exp_hydro.mass_conservative.v1" && <i><Check size={12} /></i>}</button>
              <button type="button" className={`choice-card ${form.pamlPhysicalModelId === "hbv.mass_conservative_daily.v1" ? "selected" : ""}`} onClick={() => onForm({ pamlPhysicalModelId: "hbv.mass_conservative_daily.v1" })}><Droplets size={20} /><strong>HBV P-RNN</strong><span>Soil moisture and two-zone runoff response</span>{form.pamlPhysicalModelId === "hbv.mass_conservative_daily.v1" && <i><Check size={12} /></i>}</button>
            </div>
          </section>

          <section className="card">
            <div className="card-heading compact"><div><p className="section-kicker">Learned component</p><h2>Causal CNN</h2><p>The physical discharge is concatenated with meteorological forcing before the causal convolutional correction.</p></div></div>
            <div className="field-grid adaptive-fields">
              <Field label="First-layer filters"><input type="number" min="1" max="256" value={firstCnn.filters} onChange={(event) => onForm({ cnnLayers: [{ ...firstCnn, filters: Number(event.target.value) }, ...form.cnnLayers.slice(1)] })} /></Field>
              <Field label="Causal kernel size"><input type="number" min="1" max="365" value={firstCnn.kernelSize} onChange={(event) => onForm({ cnnLayers: [{ ...firstCnn, kernelSize: Number(event.target.value) }, ...form.cnnLayers.slice(1)] })} /></Field>
              <Field label="Output layer"><input value="1 filter · kernel 1 · causal" disabled /></Field>
            </div>
          </section>
        </div>

        <section className="card process-card paml-pathway-card">
          <div className="card-heading compact"><div><p className="section-kicker">Live structure</p><h2>PaML pathway</h2></div></div>
          <div className="paml-pathway">
            <div className="pathway-node physical"><Waves size={21} /><strong>{physicalLabel} P-RNN</strong><span>Trainable hydrological parameters</span></div>
            <div className="pathway-arrow">↓</div>
            <div className="pathway-node exposed"><Database size={21} /><strong>Forcing + physical discharge</strong><span>Training-period scaling</span></div>
            <div className="pathway-arrow">↓</div>
            <div className="pathway-node corrector"><BrainCircuit size={21} /><strong>Causal CNN</strong><span>{firstCnn.filters} filters · kernel {firstCnn.kernelSize}</span></div>
            <div className="pathway-arrow">↓</div>
            <div className="pathway-node fusion"><Activity size={21} /><strong>Hybrid discharge</strong><span>Physical and final discharge are retained separately</span></div>
          </div>
          <div className="inline-note"><Info size={16} /><span>The P-RNN and convolutional layers are optimized jointly. Hydrological states and fluxes shown in the interface refer to the selected physical recurrent component.</span></div>
        </section>
      </div>
      <div className="step-footer"><span>{pamlValid ? "P-RNN backbone and causal CNN are configured." : "Review the physical component and CNN settings."}</span><button className="button button-primary" disabled={!pamlValid} onClick={onNext}>Set training <ArrowRight size={16} /></button></div>
    </>;
  }
  if (isNeural) {
    const hiddenLayerTokens = form.mlpHiddenLayers.split(",").map((value) => value.trim()).filter(Boolean);
    const hiddenLayers = hiddenLayerTokens.map((value) => Number(value));
    const receptiveField = 1 + form.cnnLayers.reduce((total, layer) => total + layer.kernelSize - 1, 0);
    const cnnValid = form.cnnLayers.length >= 1
      && form.cnnLayers.length <= 5
      && form.cnnLayers.every((layer) => Number.isInteger(layer.filters)
        && layer.filters >= 1
        && layer.filters <= 2048
        && Number.isInteger(layer.kernelSize)
        && layer.kernelSize >= 1
        && layer.kernelSize <= 365)
      && receptiveField <= form.mlpLookback;
    const neuralValid = (form.mlpFeatureRoles.length > 0 || form.laggedFeatures.length > 0)
      && Number.isInteger(form.mlpLookback)
      && form.mlpLookback >= 1
      && form.mlpLookback <= 1095
      && (
        !isS2S
        || (
          Number.isInteger(form.s2sWarmup)
          && form.s2sWarmup >= 0
          && form.s2sWarmup < form.mlpLookback
          && (!isCnn || form.s2sWarmup >= receptiveField - 1)
        )
      )
      && (isMlp
        ? hiddenLayers.length >= 1
          && hiddenLayers.length <= 5
          && hiddenLayers.every((value) => Number.isInteger(value) && value > 0 && value <= 2048)
        : isCnn
          ? cnnValid
        : recurrentLayers.length >= 1
          && recurrentLayers.length <= 5
          && recurrentLayers.every(
            (width) => Number.isInteger(width) && width >= 1 && width <= 2048,
          ));
    const toggleFeature = (role: string) => {
      const selected = form.mlpFeatureRoles.includes(role);
      onForm({
        mlpFeatureRoles: selected
          ? form.mlpFeatureRoles.filter((item) => item !== role)
          : [...form.mlpFeatureRoles, role],
      });
    };
    const addLag = () => {
      const available = new Set(availableExternalFeatures.map((item) => item.role));
      if (
        available.has(lagDraftSource)
        && lagDraftSteps >= 1
        && !form.laggedFeatures.some(
          (item) => item.source === lagDraftSource && item.lagSteps === lagDraftSteps,
        )
      ) {
        onForm({
          laggedFeatures: [
            ...form.laggedFeatures,
            { source: lagDraftSource, lagSteps: lagDraftSteps },
          ],
        });
      }
    };
    const updateCnnLayer = (
      index: number,
      update: Partial<WorkspaceForm["cnnLayers"][number]>,
    ) => onForm({
      cnnLayers: form.cnnLayers.map((layer, position) => (
        position === index ? { ...layer, ...update } : layer
      )),
    });
    const addCnnLayer = () => {
      if (form.cnnLayers.length < 5) {
        onForm({
          cnnLayers: [
            ...form.cnnLayers,
            { filters: 8, kernelSize: 3, activation: "relu" },
          ],
        });
      }
    };
    const removeCnnLayer = (index: number) => {
      if (form.cnnLayers.length > 1) {
        onForm({
          cnnLayers: form.cnnLayers.filter((_, position) => position !== index),
        });
      }
    };
    return <>
      <PageHeader eyebrow="Neural structure" title={`Configure the ${modelName} explicitly`} description={isS2S ? "This reference engine emits one causally aligned discharge for every sequence position. Warm-up overlap conditions the graph without duplicating scored dates." : "This reference engine uses one-step-ahead sequence-to-value windows. Transformations are fitted on training rows only and are stored with the learned weights."} />
      <div className="content-grid model-grid">
        <div className="stack">
          <section className="card model-identity">
            <div className="model-title-row"><div className="model-mark"><BrainCircuit size={25} /></div><div><p className="section-kicker">Neural model</p><h2>{isLstm ? "Long short-term memory" : isGru ? "Gated recurrent unit" : isCnn ? "One-dimensional convolutional network" : "Multilayer perceptron"}</h2><span>TensorFlow {isStackedRecurrent ? "stacked recurrent " : isRecurrent ? "recurrent " : isCnn ? "causal convolution " : ""}reference engine · {isS2S ? "causal S2S" : "S2V"} {isS2S ? (isRecurrent ? "v3.0.0" : "v2.0.0") : isStackedRecurrent ? "v2.0.0" : "v1.0.0"}</span></div><span className="status-pill success"><CircleCheck size={14} /> Available</span></div>
            <div className="capability-row"><span>{form.mlpFeatureRoles.length + form.laggedFeatures.length} features</span><span>{form.mlpLookback}-day {isS2S ? "sequence" : "lookback"}</span><span>{isRecurrent ? `${recurrentLayers.length} recurrent layer${recurrentLayers.length === 1 ? "" : "s"} · ${recurrentLayers.join(" → ")} units` : isCnn ? `${form.cnnLayers.length} convolutional layers` : `${hiddenLayers.length} hidden layers`}</span><span>{isS2S ? "same-time causal outputs" : "one-step forecast"}</span></div>
            <div className="model-note"><Info size={17} /><p>This implementation is identified as <code>{form.modelId}</code> and uses TensorFlow 2.21. {isLstm ? (isStackedRecurrent ? "Each LSTM layer passes its complete hidden sequence to the next layer; the input, forget, candidate, and output gates are trained jointly through time." : "The model uses one LSTM layer with input, forget, candidate, and output gates followed by the configured streamflow readout.") : isGru ? (isStackedRecurrent ? "Each GRU layer passes its complete hidden sequence to the next layer; the recurrent gates are trained jointly through time." : "The model uses one GRU layer followed by the configured streamflow readout.") : isCnn ? (isS2S ? "The CNN uses stride-one causal convolutions and a shared output head at every time position." : "The CNN uses stride-one causal convolutions and reads the final time position.") : isS2S ? "The MLP is applied at each time position using the configured input features." : "The MLP uses the configured antecedent window to predict streamflow."}</p></div>
          </section>
          <section className="card">
            <div className="card-heading compact"><div><p className="section-kicker">Model library</p><h2>Available structures</h2></div></div>
            <div className="choice-grid model-library-grid">
              <button className="choice-card" onClick={() => onForm({ modelId: "exp_hydro.mass_conservative.v1" })}><Waves size={20} /><strong>EXP-HYDRO</strong><span>Conceptual</span></button>
              <button className="choice-card" onClick={() => onForm({ modelId: "hbv.mass_conservative_daily.v1" })}><Boxes size={20} /><strong>HBV</strong><span>Conceptual</span></button>
              <button className={`choice-card ${isMlp ? "selected" : ""}`} onClick={() => onForm({ modelId: "mlp.tensorflow_s2v.v1" })}><BrainCircuit size={20} /><strong>MLP</strong><span>Neural · S2V</span>{isMlp && <i><Check size={12} /></i>}</button>
              <button className={`choice-card ${isLstm ? "selected" : ""}`} onClick={() => onForm({ modelId: "lstm.tensorflow_stacked_s2v.v2" })}><BrainCircuit size={20} /><strong>LSTM</strong><span>Stacked recurrent · S2V</span>{isLstm && <i><Check size={12} /></i>}</button>
              <button className={`choice-card ${isGru ? "selected" : ""}`} onClick={() => onForm({ modelId: "gru.tensorflow_stacked_s2v.v2" })}><BrainCircuit size={20} /><strong>GRU</strong><span>Stacked recurrent · S2V</span>{isGru && <i><Check size={12} /></i>}</button>
              <button className={`choice-card ${isCnn ? "selected" : ""}`} onClick={() => onForm({ modelId: "cnn1d.tensorflow_s2v.v1" })}><Activity size={20} /><strong>CNN-1D</strong><span>Causal convolution · S2V</span>{isCnn && <i><Check size={12} /></i>}</button>
              <button className="choice-card" onClick={() => onForm({ modelId: "paml.tensorflow_prnn_cnn_s2s.v1" })}><BrainCircuit size={20} /><strong>PaML</strong><span>P-RNN + causal CNN</span></button>
              <button className="choice-card" onClick={() => onForm({
                modelId: "deepdiscover.numpy_instantaneous_s2s.v1",
                initialSoilStorage: 0,
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
                trainingEpochs: 600,
                trainingLearningRate: 0.01,
                trainingPatience: 75,
                trainingMinimumDelta: 0.000001,
                trainingGradientClipNorm: 5,
                seed: 200,
              })}><FlaskConical size={20} /><strong>DeepDiscover</strong><span>Process discovery · S2S</span></button>
            </div>
          </section>
          <section className="card">
            <div className="card-heading compact"><div><p className="section-kicker">Predictors</p><h2>Input features and temporal context</h2></div></div>
            <div className="choice-grid adaptive-choice-grid">
              {availableExternalFeatures.map(({ role, label, unit }) => {
                const selected = form.mlpFeatureRoles.includes(role);
                return <button key={role} type="button" className={`choice-card ${selected ? "selected" : ""}`} aria-pressed={selected} onClick={() => toggleFeature(role)}><Activity size={20} /><strong>{label}</strong><span>{unit} · standardized on training only</span>{selected && <i><Check size={12} /></i>}</button>;
              })}
            </div>
            <div className="field-grid two-columns neural-fields">
              <Field label={isS2S ? "Sequence length" : "Lookback"} hint={isS2S ? "Causal input/output positions" : "Antecedent daily intervals"}><input type="number" min="1" max="1095" value={form.mlpLookback} onChange={(event) => onForm({ mlpLookback: Number(event.target.value) })} /></Field>
              {isS2S && <Field label="Causal warm-up" hint="Overlapped rows excluded from the loss"><input type="number" min={isCnn ? receptiveField - 1 : 0} max={Math.max(0, form.mlpLookback - 1)} value={form.s2sWarmup} onChange={(event) => onForm({ s2sWarmup: Number(event.target.value) })} /></Field>}
            </div>
            <div className="choice-grid two-choices neural-fields">
              <button className={`choice-card ${!isS2S ? "selected" : ""}`} onClick={() => onForm({ modelId: s2vCounterpart(form.modelId as WorkspaceForm["pamlCorrectorId"]) })}><Activity size={20} /><strong>S2V</strong><span>Antecedent window → next discharge</span>{!isS2S && <i><Check size={12} /></i>}</button>
              <button className={`choice-card ${isS2S ? "selected" : ""}`} onClick={() => onForm({ modelId: s2sCounterpart(form.modelId as WorkspaceForm["pamlCorrectorId"]), trainingStride: 1 })}><Activity size={20} /><strong>S2S</strong><span>Aligned causal sequence → discharge sequence</span>{isS2S && <i><Check size={12} /></i>}</button>
            </div>
            <div className="feature-engineering-block">
              <div className="feature-group-heading"><strong>Causal shifted variables</strong><span>Optional feature engineering before windowing</span></div>
              <div className="feature-lag-builder">
                <Field label="Source variable"><select value={lagDraftSource} onChange={(event) => setLagDraftSource(event.target.value)}>{availableExternalFeatures.map((item) => <option key={item.role} value={item.role}>{item.label}</option>)}</select></Field>
                <Field label="Shift" hint="Past daily steps"><input type="number" min="1" max="1095" value={lagDraftSteps} onChange={(event) => setLagDraftSteps(Number(event.target.value))} /></Field>
                <button type="button" className="button button-secondary" disabled={lagDraftSteps < 1} onClick={addLag}><Plus size={16} /> Add shifted feature</button>
              </div>
              <div className="lag-chip-list">
                {form.laggedFeatures.length ? form.laggedFeatures.map((item, index) => <span className="lag-chip" key={`${item.source}-${item.lagSteps}`}><strong>{item.source.replaceAll("_", " ")}</strong><small>t − {item.lagSteps}</small><button aria-label={`Remove lag ${item.source} ${item.lagSteps}`} onClick={() => onForm({ laggedFeatures: form.laggedFeatures.filter((_, position) => position !== index) })}><Trash2 size={13} /></button></span>) : <span className="empty-lag-note">No shifted feature added.</span>}
              </div>
            </div>
          </section>
          <section className="card">
            <div className="card-heading compact"><div><p className="section-kicker">Learned mapping</p><h2>{isRecurrent ? "Recurrent architecture" : isCnn ? "Convolutional architecture" : "Network architecture"}</h2></div></div>
            {isCnn ? <>
              <div className="cnn-layer-stack">
                {form.cnnLayers.map((layer, index) => <div className="cnn-layer-row" key={index}>
                  <div className="cnn-layer-index"><span>Layer</span><strong>{index + 1}</strong></div>
                  <Field label="Filters"><input type="number" min="1" max="2048" value={layer.filters} onChange={(event) => updateCnnLayer(index, { filters: Number(event.target.value) })} /></Field>
                  <Field label="Kernel size" hint="Daily positions"><input type="number" min="1" max="365" value={layer.kernelSize} onChange={(event) => updateCnnLayer(index, { kernelSize: Number(event.target.value) })} /></Field>
                  <Field label="Activation"><select value={layer.activation} onChange={(event) => updateCnnLayer(index, { activation: event.target.value as WorkspaceForm["cnnLayers"][number]["activation"] })}><option value="relu">ReLU</option><option value="tanh">tanh</option></select></Field>
                  <button type="button" className="icon-button cnn-remove-layer" aria-label={`Remove convolutional layer ${index + 1}`} disabled={form.cnnLayers.length === 1} onClick={() => removeCnnLayer(index)}><Trash2 size={16} /></button>
                </div>)}
              </div>
              <div className="cnn-layer-actions"><button type="button" className="button button-secondary" disabled={form.cnnLayers.length === 5} onClick={addCnnLayer}><Plus size={16} /> Add convolutional layer</button><span>Receptive field: {receptiveField} days · causal padding</span></div>
              <div className="field-grid two-columns neural-fields">
                <Field label="Readout" hint={isS2S ? "Shared causal head" : "Final-position S2V readout"}><input value={isS2S ? "Every time position" : "Final time position"} disabled /></Field>
                <Field label="Output activation"><select value={form.mlpOutputActivation} onChange={(event) => onForm({ mlpOutputActivation: event.target.value as WorkspaceForm["mlpOutputActivation"] })}><option value="linear">Linear</option><option value="softplus">Softplus · non-negative</option></select></Field>
              </div>
            </> : <div className="field-grid adaptive-fields">
              {isRecurrent ? <>
                <Field label="Recurrent layers" hint="Comma-separated widths, e.g. 64, 32"><input value={form.recurrentLayers} onChange={(event) => onForm({ recurrentLayers: event.target.value })} /></Field>
                <Field label="Layer count" hint="Between one and five"><input value={String(recurrentLayers.length)} disabled /></Field>
                <Field label="Gate activations" hint={isLstm ? "Fixed standard LSTM equations" : "Fixed reset-before GRU equations"}><input value="sigmoid · tanh" disabled /></Field>
              </> : <>
                <Field label="Hidden layers" hint="Comma-separated widths, e.g. 64, 32"><input value={form.mlpHiddenLayers} onChange={(event) => onForm({ mlpHiddenLayers: event.target.value })} /></Field>
                <Field label="Hidden activation"><select value={form.mlpActivation} onChange={(event) => onForm({ mlpActivation: event.target.value as WorkspaceForm["mlpActivation"] })}><option value="relu">ReLU</option><option value="tanh">tanh</option></select></Field>
              </>}
              <Field label="Output activation"><select value={form.mlpOutputActivation} onChange={(event) => onForm({ mlpOutputActivation: event.target.value as WorkspaceForm["mlpOutputActivation"] })}><option value="linear">Linear</option><option value="softplus">Softplus · non-negative</option></select></Field>
            </div>}
            <div className="inline-note"><Info size={16} /><span>Both input and target transformations are fitted before window construction using training rows only, then frozen for validation, test, and future forecasts.</span></div>
          </section>
        </div>
        <section className="card process-card neural-pathway-card">
          <div className="card-heading compact"><div><p className="section-kicker">Live structure</p><h2>{modelName} pathway</h2></div></div>
          <div className="process-diagram neural-process-diagram">
            <div className="process-source"><Database size={22} /><strong>Selected predictors</strong><span>{form.mlpFeatureRoles.length} direct + {form.laggedFeatures.length} shifted</span></div>
            <div className="process-line labeled"><span>{form.mlpLookback} antecedent days</span></div>
            <div className="neural-node"><strong>Training-fitted scaler</strong><small>No validation/test leakage</small></div>
            <div className="process-line" />
            {isRecurrent ? recurrentLayers.map((width, index) => <div className="neural-node hidden-layer" key={`${index}-${width}`}><strong>{modelName} recurrent layer {index + 1}</strong><small>{width} units · {isLstm ? "four learned gates" : "update and reset gates"}</small></div>) : isCnn ? form.cnnLayers.map((layer, index) => <div className="neural-node hidden-layer" key={`${index}-${layer.filters}-${layer.kernelSize}`}><strong>Causal Conv1D layer {index + 1}</strong><small>{layer.filters} filters · kernel {layer.kernelSize} · {layer.activation}</small></div>) : hiddenLayers.map((width, index) => <div className="neural-node hidden-layer" key={`${index}-${width}`}><strong>Hidden layer {index + 1}</strong><small>{width} units · {form.mlpActivation}</small></div>)}
            <div className="process-line" />
            <div className="process-output"><Waves size={22} /><strong>{isS2S ? "Q at every causal t" : "Q at t + 1"}</strong><span>{form.mlpOutputActivation} output</span></div>
          </div>
        </section>
      </div>
      <div className="step-footer"><span>{neuralValid ? "Neural structure and preprocessing are explicit." : `Select at least one feature and review the lookback and ${isRecurrent ? "recurrent width" : isCnn ? "convolutional stack" : "hidden layers"}.`}</span><button className="button button-primary" disabled={!neuralValid} onClick={onNext}>Set training <ArrowRight size={16} /></button></div>
    </>;
  }
  return (
    <>
      <PageHeader
        eyebrow="Scientific structure"
        title={`Configure ${modelName} explicitly`}
        description="Select a documented model variant. Every implementation exposes boundary states and interval fluxes and is tested for numerical mass conservation."
      />
      <div className="content-grid model-grid">
        <div className="stack">
          <section className="card model-identity">
            <div className="model-title-row">
              <div className="model-mark">{isHbv ? <Boxes size={25} /> : <Waves size={25} />}</div>
              <div>
                <p className="section-kicker">Conceptual model</p>
                <h2>{modelName}</h2>
              <span>{isHbv ? "Daily HBV response formulation" : "Daily EXP-HYDRO formulation"} · v1.0.0</span>
              </div>
              <span className="status-pill success"><CircleCheck size={14} /> Available</span>
            </div>
            <div className="capability-row">
              <span>3 forcings</span><span>{stateSummary} states</span><span>{fluxCount} fluxes</span><span>{calibratedParameterCount(form)} calibrated parameters</span>
            </div>
            <div className="model-note">
              <Info size={17} />
              <p>{isHbv ? <>This implementation is identified as <code>hbv.mass_conservative_daily.v1</code>. Its response discharge can remain unrouted or pass through the separately versioned, restartable <code>routing.maxbas.triangular.v1</code> process.</> : <>This implementation is identified as <code>exp_hydro.mass_conservative.v1</code> and preserves the daily EXP-HYDRO process structure with explicit interval water accounting.</>}</p>
            </div>
          </section>

          <section className="card">
            <div className="card-heading compact"><div><p className="section-kicker">Model library</p><h2>Conceptual structure</h2></div></div>
            <div className="choice-grid model-library-grid">
              <button className={`choice-card ${isExpHydro ? "selected" : ""}`} onClick={() => onForm({ modelId: "exp_hydro.mass_conservative.v1" })}><Waves size={20} /><strong>EXP-HYDRO</strong><span>Two storages · three parameters</span>{isExpHydro && <i><Check size={12} /></i>}</button>
              <button className={`choice-card ${isHbv ? "selected" : ""}`} onClick={() => onForm({ modelId: "hbv.mass_conservative_daily.v1" })}><Boxes size={20} /><strong>HBV</strong><span>Four storages · eight parameters</span>{isHbv && <i><Check size={12} /></i>}</button>
              <button className="choice-card" onClick={() => onForm({ modelId: "mlp.tensorflow_s2v.v1" })}><BrainCircuit size={20} /><strong>MLP</strong><span>Neural · one-step S2V</span></button>
              <button className="choice-card" onClick={() => onForm({ modelId: "lstm.tensorflow_stacked_s2v.v2" })}><BrainCircuit size={20} /><strong>LSTM</strong><span>Stacked recurrent · one-step S2V</span></button>
              <button className="choice-card" onClick={() => onForm({ modelId: "gru.tensorflow_stacked_s2v.v2" })}><BrainCircuit size={20} /><strong>GRU</strong><span>Stacked recurrent · one-step S2V</span></button>
              <button className="choice-card" onClick={() => onForm({ modelId: "cnn1d.tensorflow_s2v.v1" })}><Activity size={20} /><strong>CNN-1D</strong><span>Causal convolution · one-step S2V</span></button>
              <button className="choice-card" onClick={() => onForm({ modelId: "paml.tensorflow_prnn_cnn_s2s.v1" })}><BrainCircuit size={20} /><strong>PaML</strong><span>P-RNN + causal CNN</span></button>
              <button className="choice-card" onClick={() => onForm({
                modelId: "deepdiscover.numpy_instantaneous_s2s.v1",
                initialSoilStorage: 0,
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
                trainingEpochs: 600,
                trainingLearningRate: 0.01,
                trainingPatience: 75,
                trainingMinimumDelta: 0.000001,
                trainingGradientClipNorm: 5,
                seed: 200,
              })}><FlaskConical size={20} /><strong>DeepDiscover</strong><span>Process discovery · S2S</span></button>
            </div>
          </section>

          <section className="card">
            <div className="card-heading compact"><div><p className="section-kicker">Composable process</p><h2>Snow module</h2></div></div>
            <div className="choice-grid three-choices">
              {[
                ["snow.none.v1", CloudSun, "None", "All precipitation enters as rain."],
                ["snow.degree_day.hard.v1", Snowflake, "Degree-day", "Hard temperature thresholds."],
                ["snow.degree_day.smooth.v1", Activity, "Smooth", "Differentiable transitions."],
              ].map(([id, Icon, label, detail]) => (
                <button
                  key={id as string}
                  className={`choice-card ${form.snowModule === id ? "selected" : ""}`}
                  onClick={() => onForm({ snowModule: id as WorkspaceForm["snowModule"] })}
                >
                  <Icon size={20} />
                  <strong>{label as string}</strong>
                  <span>{detail as string}</span>
                  {form.snowModule === id && <i><Check size={12} /></i>}
                </button>
              ))}
            </div>
            {snowEnabled && (
              <div className="parameters-block snow-parameters">
                <div className="snow-parameters-heading"><div><strong>Parameter treatment</strong><span>Fix known values or estimate selected parameters within explicit bounds.</span></div><span className="status-pill neutral">{activeTreatments.filter((key) => form.snowCalibration[key].mode === "calibrate").length} estimated</span></div>
                <ParameterTreatmentControl label="Snowfall threshold" unit="°C" value={form.snowfallTemperature} treatment={form.snowCalibration.snowfallTemperature} onValue={(value) => onForm({ snowfallTemperature: value })} onTreatment={(update) => updateSnowTreatment("snowfallTemperature", update)} />
                <ParameterTreatmentControl label="Melt threshold" unit="°C" value={form.meltTemperature} treatment={form.snowCalibration.meltTemperature} onValue={(value) => onForm({ meltTemperature: value })} onTreatment={(update) => updateSnowTreatment("meltTemperature", update)} />
                <ParameterTreatmentControl label="Degree-day factor" unit="mm °C⁻¹ day⁻¹" value={form.degreeDayFactor} treatment={form.snowCalibration.degreeDayFactor} onValue={(value) => onForm({ degreeDayFactor: value })} onTreatment={(update) => updateSnowTreatment("degreeDayFactor", update)} />
                {form.snowModule === "snow.degree_day.smooth.v1" && <ParameterTreatmentControl label="Transition sharpness" unit="°C⁻¹" value={form.sharpness} treatment={form.snowCalibration.sharpness} onValue={(value) => onForm({ sharpness: value })} onTreatment={(update) => updateSnowTreatment("sharpness", update)} />}
              </div>
            )}
          </section>

          {isHbv && <section className="card">
            <div className="card-heading compact"><div><p className="section-kicker">Composable process</p><h2>Runoff routing</h2></div></div>
            <div className="choice-grid two-choices">
              <button className={`choice-card ${!routingEnabled ? "selected" : ""}`} onClick={() => onForm({ routingProcess: "routing.none.v1", initialRoutingState: [0, 0, 0, 0, 0, 0] })}><Waves size={20} /><strong>None</strong><span>Use the raw HBV response discharge.</span>{!routingEnabled && <i><Check size={12} /></i>}</button>
              <button className={`choice-card ${routingEnabled ? "selected" : ""}`} onClick={() => onForm({ routingProcess: "routing.maxbas.triangular.v1" })}><Activity size={20} /><strong>Triangular MAXBAS</strong><span>Distribute generated runoff over 1–7 daily intervals.</span>{routingEnabled && <i><Check size={12} /></i>}</button>
            </div>
            {routingEnabled && <div className="parameters-block routing-parameters">
              <div className="snow-parameters-heading"><div><strong>Routing parameter treatment</strong><span>MAXBAS is an integer number of days; calibration samples whole days only.</span></div><span className="status-pill neutral">{form.maxbasTreatment.mode === "calibrate" ? "1 estimated" : "fixed"}</span></div>
              <ParameterTreatmentControl label="MAXBAS" unit="days · integer" value={form.maxbas} treatment={form.maxbasTreatment} step={1} minimum={1} maximum={7} onValue={(value) => onForm({ maxbas: value })} onTreatment={(update) => onForm({ maxbasTreatment: { ...form.maxbasTreatment, ...update } })} />
            </div>}
          </section>}

          <section className="card">
            <div className="card-heading compact"><div><p className="section-kicker">Numerical setup</p><h2>Initial state & integration</h2></div></div>
            <div className="field-grid four-columns">
              <Field label="Initial snow storage" hint="mm at the first boundary">
                <input type="number" min="0" value={form.initialSnowStorage} disabled={!snowEnabled} onChange={(e) => onForm({ initialSnowStorage: Number(e.target.value) })} />
              </Field>
              <Field label="Initial soil storage" hint="mm at the first boundary">
                <input type="number" min="0" value={form.initialSoilStorage} onChange={(e) => onForm({ initialSoilStorage: Number(e.target.value) })} />
              </Field>
              {isHbv && <Field label="Initial upper zone" hint="SUZ in mm at the first boundary">
                <input type="number" min="0" value={form.initialUpperZoneStorage} onChange={(e) => onForm({ initialUpperZoneStorage: Number(e.target.value) })} />
              </Field>}
              {isHbv && <Field label="Initial lower zone" hint="SLZ in mm at the first boundary">
                <input type="number" min="0" value={form.initialLowerZoneStorage} onChange={(e) => onForm({ initialLowerZoneStorage: Number(e.target.value) })} />
              </Field>}
              {!isHbv && <Field label="Substeps per day" hint="Explicit mass-conservative integration">
                <input type="number" min="1" max="96" value={form.substeps} onChange={(e) => onForm({ substeps: Number(e.target.value) })} />
              </Field>}
            </div>
            {routingEnabled && <div className="routing-memory-block">
              <div><strong>Initial routing memory</strong><span>Pending release in the current and following five daily intervals. Leave all values at zero for a cold start.</span></div>
              <div className="field-grid routing-memory-grid">
                {form.initialRoutingState.map((value, position) => <Field key={position} label={`Pending slot ${position + 1}`} hint="mm">
                  <input type="number" min="0" step="0.01" value={value} onChange={(event) => {
                    const next = [...form.initialRoutingState] as WorkspaceForm["initialRoutingState"];
                    next[position] = Number(event.target.value);
                    onForm({ initialRoutingState: next });
                  }} />
                </Field>)}
              </div>
            </div>}
          </section>
        </div>

        <section className="card process-card">
          <div className="card-heading compact"><div><p className="section-kicker">Live structure</p><h2>Model pathway</h2></div></div>
          <div className="process-diagram">
            <div className="process-source"><CloudSun size={22} /><strong>Forcings</strong><span>P · T · PET</span></div>
            <div className="process-line" />
            <div className={`reservoir snow ${snowEnabled ? "" : "inactive"}`} data-state="snow_storage">
              <span className="reservoir-index">01</span>
              <span className="reservoir-label">Snow storage</span>
              <div className="water-level snow-level" />
              <small>{snowLabel(form.snowModule)}</small>
            </div>
            <div className="process-line labeled"><span>rain + melt</span></div>
            <div className="reservoir soil" data-state="soil_storage">
              <span className="reservoir-index">02</span>
              <span className="reservoir-label">Soil storage</span>
              <div className="water-level soil-level" />
              <small>{isHbv ? "FC-limited soil routine" : "Continuous state"}</small>
            </div>
            {isHbv ? (
              <>
                <div className="flow-split pathway-branches two-branch">
                  <div><span>ET</span><small>to atmosphere</small></div>
                  <div className="primary-branch"><span>Recharge</span><small>all soil excess included</small></div>
                </div>
                <div className="reservoir upper-zone" data-state="upper_zone_storage">
                  <span className="reservoir-index">03</span>
                  <span className="reservoir-label">Upper-zone storage</span>
                  <div className="water-level upper-zone-level" />
                  <small>Fast and intermediate response</small>
                </div>
                <div className="process-line labeled percolation-link"><span>bounded percolation</span></div>
                <div className="reservoir lower-zone" data-state="lower_zone_storage">
                  <span className="reservoir-index">04</span>
                  <span className="reservoir-label">Lower-zone storage</span>
                  <div className="water-level lower-zone-level" />
                  <small>Slow response reservoir</small>
                </div>
                <div className="flow-split pathway-branches">
                  <div><span>Q<sub>0</sub></span><small>fast above UZL</small></div>
                  <div><span>Q<sub>1</sub></span><small>intermediate</small></div>
                  <div><span>Q<sub>2</sub></span><small>baseflow</small></div>
                </div>
              </>
            ) : (
              <div className="flow-split pathway-branches">
                <div><span>ET</span><small>to atmosphere</small></div>
                <div><span>Q<sub>sub</sub></span><small>subsurface</small></div>
                <div><span>Q<sub>surf</sub></span><small>surface</small></div>
              </div>
            )}

            {routingEnabled && <>
              <div className="process-line" />
              <div className="routing-node" data-process="routing.maxbas.triangular.v1">
                <Activity size={19} /><strong>Triangular routing</strong><span>MAXBAS {form.maxbas} day{form.maxbas === 1 ? "" : "s"} · six-slot restart memory</span>
              </div>
              <div className="process-line" />
            </>}
            <div className="output-node"><Waves size={20} /><strong>Streamflow</strong><span>{isHbv ? (routingEnabled ? "MAXBAS-routed response" : "Q0 + Q1 + Q2 · unrouted") : "Qsub + Qsurf"}</span></div>
          </div>
          <div className="timing-legend">
            <span><i className="dot state" /> States at n + 1 boundaries</span>
            <span><i className="dot flux" /> Fluxes on n intervals</span>
          </div>
        </section>
      </div>
      <div className="step-footer"><span>{modelParametersValid ? "Model structure and parameter treatments are explicit." : "Review the process parameter values, integer bounds, and initial states."}</span><button className="button button-primary" disabled={!modelParametersValid} onClick={onNext}>Set calibration <ArrowRight size={16} /></button></div>
    </>
  );
}

export function CalibrationPanel({
  form,
  preview,
  quality,
  onForm,
  onNext,
}: {
  form: WorkspaceForm;
  preview: CsvPreview;
  quality: DataQualityReport | null;
  onForm: (update: Partial<WorkspaceForm>) => void;
  onNext: () => void;
}) {
  if (
    isMlpModel(form.modelId)
    || isRecurrentModel(form.modelId)
    || isCnnModel(form.modelId)
    || isPamlModel(form.modelId)
    || isDiscoveryModel(form.modelId)
  ) {
    const isPaml = isPamlModel(form.modelId);
    const isDiscovery = isDiscoveryModel(form.modelId);
    const correctorId = form.modelId;
    const isLstm = isLstmModel(correctorId);
    const isGru = isGruModel(correctorId);
    const isCnn = isCnnModel(correctorId);
    const isRecurrent = isLstm || isGru;
    const isSequenceModel = isPaml || isRecurrent || isCnn || isS2SNeuralModel(correctorId);
    const usesGradientClipping = isSequenceModel || isDiscovery;
    const testEnd = form.testEndIndex ?? preview.row_count;
    const trainRows = form.trainEndIndex - form.trainStartIndex;
    const validationRows = form.validationEndIndex - form.trainEndIndex;
    const testRows = testEnd - form.validationEndIndex;
    const hiddenLayers = form.mlpHiddenLayers.split(",").map((value) => Number(value.trim())).filter((value) => Number.isInteger(value) && value > 0);
    const longestLag = Math.max(0, ...form.laggedFeatures.map((item) => item.lagSteps));
    const activeNeuralWarmup = isPaml ? form.warmupSteps : isS2SNeuralModel(correctorId)
      ? form.s2sWarmup
      : form.mlpLookback;
    const featureCount = (isPaml ? 4 : form.mlpFeatureRoles.length) + form.laggedFeatures.length;
    const valid = form.trainStartIndex >= 0
      && trainRows > (isDiscovery ? form.warmupSteps : activeNeuralWarmup + longestLag)
      && validationRows > 0
      && testRows > 0
      && testEnd <= preview.row_count
      && form.trainingEpochs >= 1
      && (isDiscovery || form.trainingBatchSize >= 1)
      && form.trainingLearningRate > 0
      && form.trainingPatience >= 1
      && (isDiscovery || form.trainingStride >= 1)
      && (
        isDiscovery
        || !form.initializationEnsembleEnabled
        || (
          form.initializationEnsembleMembers >= 3
          && form.initializationEnsembleMembers <= 20
        )
      )
      && (!usesGradientClipping || form.trainingGradientClipNorm > 0);
    const selectedRows = testEnd - form.trainStartIndex;
    return <>
      <PageHeader eyebrow={isDiscovery ? "Constrained process discovery" : isPaml ? "Joint PaML learning" : "Supervised learning"} title="Separate training, validation, and test" description={isDiscovery ? "Full-sequence Adam learns the instantaneous mapping and bounded switches. Training-only scaling prevents leakage; validation selects the checkpoint and test remains untouched until final evaluation." : isPaml ? "The P-RNN and causal CNN are optimized jointly on training rows. Validation selects the checkpoint and test remains untouched until final evaluation." : "The partitions are chronological and disjoint. Scalers and weights learn from training rows only; validation selects the checkpoint and test remains untouched until final evaluation."} />
      <div className="content-grid calibration-grid">
        <div className="stack">
          <section className="card">
            <div className="card-heading"><div className="icon-tile purple"><Gauge size={19} /></div><div><h2>Chronological partitions</h2><p>Every boundary is a half-open row index.</p></div></div>
            <div className="field-grid four-columns">
              <Field label="First training row" hint="Included"><input type="number" min="0" max={preview.row_count - 3} value={form.trainStartIndex} onChange={(event) => onForm({ trainStartIndex: Number(event.target.value) })} /></Field>
              <Field label="Training end" hint="Exclusive"><input type="number" min="1" max={preview.row_count - 2} value={form.trainEndIndex} onChange={(event) => onForm({ trainEndIndex: Number(event.target.value) })} /></Field>
              <Field label="Validation end" hint="Exclusive"><input type="number" min="2" max={preview.row_count - 1} value={form.validationEndIndex} onChange={(event) => onForm({ validationEndIndex: Number(event.target.value) })} /></Field>
              <Field label="Test end" hint={`Exclusive · maximum ${preview.row_count}`}><input type="number" min="3" max={preview.row_count} value={form.testEndIndex ?? preview.row_count} onChange={(event) => onForm({ testEndIndex: Number(event.target.value) })} /></Field>
            </div>
            <div className="neural-partition-bar" aria-label={`${trainRows} training, ${validationRows} validation, ${testRows} test rows`}>
              <span className="partition-train" style={{ width: `${selectedRows > 0 ? 100 * trainRows / selectedRows : 0}%` }} />
              <span className="partition-validation" style={{ width: `${selectedRows > 0 ? 100 * validationRows / selectedRows : 0}%` }} />
              <span className="partition-test" />
            </div>
            <div className="period-legend"><span><i className="partition-train-key" /> {trainRows} training</span><span><i className="partition-validation-key" /> {validationRows} validation</span><span><i className="partition-test-key" /> {testRows} test</span></div>
            <div className="inline-note"><Info size={16} /><span>{isDiscovery ? "The discovered storage is propagated continuously through all partitions, while gradients and input-scaling statistics remain restricted to training. Validation and test targets never participate in fitting." : "Validation and test inputs may use antecedent forcing context from the preceding partition, but their targets never participate in fitting transformations or weights."}</span></div>
          </section>
          <section className="card">
            <div className="card-heading"><div className="icon-tile green"><Settings2 size={19} /></div><div><h2>Adam training</h2><p>The selected objective drives the data loss and validation checkpoint.</p></div></div>
            <div className="field-grid four-columns">
              <Field label="Training objective"><select value={form.objective} onChange={(event) => onForm({ objective: event.target.value as WorkspaceForm["objective"] })}>{objectiveOptions.map((option) => <option key={`training-${option.id}`} value={option.id}>{option.symbol} · {option.name}</option>)}</select></Field>
              <Field label="Maximum epochs"><input type="number" min="1" max="100000" value={form.trainingEpochs} onChange={(event) => onForm({ trainingEpochs: Number(event.target.value) })} /></Field>
              {!isDiscovery && <Field label="Batch size"><input type="number" min="1" value={form.trainingBatchSize} onChange={(event) => onForm({ trainingBatchSize: Number(event.target.value) })} /></Field>}
              <Field label="Learning rate"><input type="number" min="0.0000001" max="1" step="0.0001" value={form.trainingLearningRate} onChange={(event) => onForm({ trainingLearningRate: Number(event.target.value) })} /></Field>
              <Field label="Early-stopping patience"><input type="number" min="1" value={form.trainingPatience} onChange={(event) => onForm({ trainingPatience: Number(event.target.value) })} /></Field>
              <Field label="Minimum improvement"><input type="number" min="0" step="0.000001" value={form.trainingMinimumDelta} onChange={(event) => onForm({ trainingMinimumDelta: Number(event.target.value) })} /></Field>
              {!isDiscovery && !isPaml && <Field label={isS2SNeuralModel(correctorId) ? "Target stride" : "Window stride"}><input type="number" min="1" max="366" disabled={isS2SNeuralModel(correctorId)} value={isS2SNeuralModel(correctorId) ? 1 : form.trainingStride} onChange={(event) => onForm({ trainingStride: Number(event.target.value) })} /></Field>}
              {(isDiscovery || isPaml) && <Field label="Training warm-up" hint="Intervals excluded from training loss"><input type="number" min="0" max={Math.max(0, trainRows - 1)} value={form.warmupSteps} onChange={(event) => onForm({ warmupSteps: Number(event.target.value) })} /></Field>}
              {usesGradientClipping && <Field label="Gradient clipping" hint="Global norm"><input type="number" min="0.000001" step="0.1" value={form.trainingGradientClipNorm} onChange={(event) => onForm({ trainingGradientClipNorm: Number(event.target.value) })} /></Field>}
              <Field label="Random seed"><input type="number" value={form.seed} onChange={(event) => onForm({ seed: Number(event.target.value) })} /></Field>
            </div>
            {["nash_sutcliffe_efficiency", "kling_gupta_efficiency_2009", "range_normalized_root_mean_square_error"].includes(form.objective) && <div className="inline-note"><Info size={16} /><span>{objectiveOption(form.objective).symbol} depends on statistics fixed over the complete training series. Each epoch therefore uses the full training partition as one objective batch; the mini-batch value is ignored for gradient updates.</span></div>}
            {isDiscovery && <div className="inline-note"><Info size={16} /><span>The selected streamflow objective is combined with the declared process regularizers. Changing its numerical scale can change that trade-off, so review the regularization weights when switching objectives.</span></div>}
          </section>
          {!isDiscovery && !isPaml && <section className="card">
            <div className="card-heading"><div className="icon-tile purple"><Boxes size={19} /></div><div><h2>Initialization robustness</h2><p>Choose one checkpoint or repeat the same experiment from distinct neural seeds.</p></div></div>
            <div className="choice-grid two-choices">
              <button className={`choice-card horizontal ${!form.initializationEnsembleEnabled ? "selected" : ""}`} onClick={() => onForm({ initializationEnsembleEnabled: false })}>
                <BrainCircuit size={20} /><div><strong>Single initialization</strong><span>Fastest · one validation-selected checkpoint</span></div>{!form.initializationEnsembleEnabled && <i><Check size={12} /></i>}
              </button>
              <button className={`choice-card horizontal ${form.initializationEnsembleEnabled ? "selected" : ""}`} onClick={() => onForm({ initializationEnsembleEnabled: true })}>
                <Boxes size={20} /><div><strong>Initialization ensemble</strong><span>Recommended for robustness analysis · arithmetic mean</span></div>{form.initializationEnsembleEnabled && <i><Check size={12} /></i>}
              </button>
            </div>
            <div className="ensemble-configuration">
              {form.initializationEnsembleEnabled && <Field label="Neural members" hint="3 to 20 distinct deterministic seeds"><input type="number" min="3" max="20" value={form.initializationEnsembleMembers} onChange={(event) => onForm({ initializationEnsembleMembers: Number(event.target.value) })} /></Field>}
              <div className="inline-note"><Info size={16} /><span>Every member uses the same architecture, partitions, preprocessing, optimizer settings, and base experiment. When enabled, the displayed 10–90% member envelope measures initialization sensitivity; it is not a predictive interval.</span></div>
            </div>
          </section>}
        </div>
        <aside className="card calibration-summary">
          <p className="section-kicker">Before execution</p><h2>Training setup</h2>
          {isDiscovery ? <>
            <div className="summary-stat"><span>Temporal mapping</span><strong>Instantaneous S2S</strong></div>
            <div className="summary-stat"><span>Learned structure</span><strong>{form.discoveryProcessCount} processes · {form.discoveryHiddenUnits} hidden units</strong></div>
            <div className="summary-stat"><span>Optimizer scope</span><strong>Complete chronological sequence</strong></div>
          </> : <>
            <div className="summary-stat"><span>{isSequenceModel ? "Input sequence" : "Input width"}</span><strong>{isSequenceModel ? `${form.mlpLookback} × ${featureCount}` : form.mlpLookback * featureCount}</strong></div>
            <div className="summary-stat"><span>{isRecurrent ? "Recurrent stack" : isCnn ? "Convolutional stack" : "Hidden layers"}</span><strong>{isRecurrent ? `${form.recurrentLayers} ${isLstm ? "LSTM" : "GRU"} units` : isCnn ? form.cnnLayers.map((layer) => `${layer.filters}×k${layer.kernelSize}`).join(" → ") : hiddenLayers.join(" → ") || "—"}</strong></div>
          </>}
          <div className="summary-stat"><span>Checkpoint metric</span><strong>Validation {objectiveOption(form.objective).symbol}</strong></div>
          {!isDiscovery && <div className="summary-stat"><span>Neural initializations</span><strong>{form.initializationEnsembleEnabled ? `${form.initializationEnsembleMembers} members · mean` : "1 checkpoint"}</strong></div>}
          <div className="summary-stat"><span>Final metrics</span><strong>Test partition</strong></div>
          {isPaml && <div className="summary-stat"><span>Hybrid training</span><strong>P-RNN + causal CNN · joint optimization</strong></div>}
          <div className="summary-divider" />
          <p className="summary-copy">The test set is evaluated once after the best validation checkpoint has been restored. The archive stores the scalers, architecture, {isDiscovery ? "active inputs, raw and corrected processes, bounded switches, constraint terms, weights, " : isRecurrent ? "gate kernels, recurrent kernel, " : isCnn ? "causal convolution kernels, " : "weights, "}biases, seed, and complete epoch history.</p>
          <span className={`readiness ${valid ? "ready" : "not-ready"}`}>{valid ? <CircleCheck size={16} /> : <CircleAlert size={16} />}{valid ? "Ready for preflight validation" : "Review partition and optimizer settings"}</span>
        </aside>
      </div>
      <div className="step-footer"><span>{valid ? "Training setup is internally consistent." : "Training, validation, and test must be non-empty and ordered."}</span><button className="button button-primary" disabled={!valid} onClick={onNext}>Review experiment <ArrowRight size={16} /></button></div>
    </>;
  }
  const calibrationEnd = form.endIndex ?? Math.max(
    form.startIndex + 1,
    Math.floor(preview.row_count * 0.6),
  );
  const validationEnd = form.validationEndIndex;
  const testEnd = form.testEndIndex ?? preview.row_count;
  const calibrationRows = Math.max(0, calibrationEnd - form.startIndex);
  const validationRows = form.conceptualProtocol === "calibration_validation_test"
    ? Math.max(0, validationEnd - calibrationEnd)
    : 0;
  const testStart = form.conceptualProtocol === "calibration_validation_test"
    ? validationEnd
    : calibrationEnd;
  const testRows = form.conceptualProtocol === "legacy_calibration_only"
    ? 0
    : Math.max(0, testEnd - testStart);
  const selected = calibrationRows + validationRows + testRows;
  const scoreStart = Math.min(calibrationEnd, form.startIndex + form.warmupSteps);
  const scored = quality?.observation_valid
    ? quality.observation_valid.slice(scoreStart, calibrationEnd).filter(Boolean).length
    : Math.max(0, calibrationRows - form.warmupSteps);
  const parameterCount = calibratedParameterCount(form);
  const resolvedPopulation = form.dePopulationSize
    ?? Math.max(8, 10 * parameterCount);
  const evaluationBudget = calibrationEvaluationBudget(form);
  const valid = calibrationEnd <= preview.row_count
    && calibrationEnd > form.startIndex
    && form.warmupSteps < calibrationRows
    && (
      form.conceptualProtocol === "legacy_calibration_only"
      || (
        testEnd <= preview.row_count
        && testRows > 0
        && (
          form.conceptualProtocol === "calibration_test"
          || validationRows > 0
        )
      )
    )
    && calibrationSearchIsValid(form);
  const timestamps = quality?.timestamp_labels;
  const startDate = timestamps?.[form.startIndex]?.slice(0, 10) ?? "";
  const calibrationLastDate = timestamps?.[Math.max(form.startIndex, calibrationEnd - 1)]?.slice(0, 10) ?? "";
  const validationLastDate = timestamps?.[Math.max(calibrationEnd, validationEnd - 1)]?.slice(0, 10) ?? "";
  const testLastDate = timestamps?.[Math.max(testStart, testEnd - 1)]?.slice(0, 10) ?? "";

  return (
    <>
      <PageHeader eyebrow="Parameter estimation" title="Configure the calibration protocol" description="The warm-up remains in the simulation but is excluded from the objective. Every candidate is reproducible from the recorded seed." />
      <div className="content-grid calibration-grid">
        <div className="stack">
          <section className="card">
            <div className="card-heading"><div className="icon-tile purple"><Gauge size={19} /></div><div><h2>Chronological evaluation protocol</h2><p>Parameters learn from calibration only; holdout targets remain untouched.</p></div></div>
            <div className="choice-grid two-choices">
              <button className={`choice-card horizontal ${form.conceptualProtocol === "calibration_validation_test" ? "selected" : ""}`} onClick={() => onForm({ conceptualProtocol: "calibration_validation_test" })}><Activity size={20} /><div><strong>Calibration · validation · test</strong><span>Recommended · validation compares choices, test reports generalization</span></div>{form.conceptualProtocol === "calibration_validation_test" && <i><Check size={12} /></i>}</button>
              <button className={`choice-card horizontal ${form.conceptualProtocol === "calibration_test" ? "selected" : ""}`} onClick={() => onForm({ conceptualProtocol: "calibration_test" })}><Gauge size={20} /><div><strong>Calibration · test</strong><span>Minimum holdout protocol · no separate validation period</span></div>{form.conceptualProtocol === "calibration_test" && <i><Check size={12} /></i>}</button>
            </div>
            {form.conceptualProtocol === "legacy_calibration_only" && <div className="notice notice-warning"><CircleAlert size={18} /><div><strong>Legacy calibration-only project</strong><p>This archive has no held-out period. Choose a protocol above before treating performance as generalization.</p></div></div>}
            <div className="field-grid four-columns">
              <Field label="First calibration row" hint={startDate ? `Included · ${startDate}` : "Included"}><input type="number" min="0" max={preview.row_count - 2} value={form.startIndex} onChange={(event) => onForm({ startIndex: Number(event.target.value) })} /></Field>
              <Field label="Calibration end" hint={calibrationLastDate ? `Exclusive · ${calibrationLastDate}` : "Exclusive"}><input type="number" min="1" max={preview.row_count - 1} value={calibrationEnd} onChange={(event) => onForm({ endIndex: Number(event.target.value) })} /></Field>
              {form.conceptualProtocol === "calibration_validation_test" && <Field label="Validation end" hint={validationLastDate ? `Exclusive · ${validationLastDate}` : "Exclusive"}><input type="number" min={calibrationEnd + 1} max={preview.row_count - 1} value={form.validationEndIndex} onChange={(event) => onForm({ validationEndIndex: Number(event.target.value) })} /></Field>}
              {form.conceptualProtocol !== "legacy_calibration_only" && <Field label="Test end" hint={testLastDate ? `Exclusive · ${testLastDate}` : `Exclusive · maximum ${preview.row_count}`}><input type="number" min={testStart + 1} max={preview.row_count} value={testEnd} onChange={(event) => onForm({ testEndIndex: Number(event.target.value) })} /></Field>}
              <Field label="Warm-up steps" hint="Inside calibration · never scored"><input type="number" min="0" max={Math.max(0, calibrationRows - 1)} value={form.warmupSteps} onChange={(event) => onForm({ warmupSteps: Number(event.target.value) })} /></Field>
            </div>
            <div className="neural-partition-bar" aria-label={`${calibrationRows} calibration, ${validationRows} validation, ${testRows} test rows`}>
              <span className="partition-train" style={{ width: `${selected > 0 ? 100 * calibrationRows / selected : 0}%` }} />
              {validationRows > 0 && <span className="partition-validation" style={{ width: `${selected > 0 ? 100 * validationRows / selected : 0}%` }} />}
              {testRows > 0 && <span className="partition-test" />}
            </div>
            <div className="period-legend"><span><i className="partition-train-key" /> {calibrationRows} calibration</span>{validationRows > 0 && <span><i className="partition-validation-key" /> {validationRows} validation</span>}{testRows > 0 && <span><i className="partition-test-key" /> {testRows} test</span>}<span>{form.warmupSteps} warm-up · {scored} calibration observations scored</span></div>
            <div className="inline-note"><Info size={16} /><span>Simulation and model states remain continuous across every boundary. Validation and test streamflow never enter the parameter optimizer.</span></div>
          </section>

          <section className="card">
            <div className="card-heading"><div className="icon-tile amber"><Activity size={19} /></div><div><h2>Calibration objective</h2><p>The same registered objective is used across every model family.</p></div></div>
            <div className="choice-grid objective-choice-grid">
              {objectiveOptions.map((option) => <button key={option.id} className={`choice-card horizontal ${form.objective === option.id ? "selected" : ""}`} onClick={() => onForm({ objective: option.id })}><span className="metric-symbol">{option.symbol}</span><div><strong>{option.name}</strong><span>{option.guidance}</span></div>{form.objective === option.id && <i><Check size={12} /></i>}</button>)}
            </div>
            <div className="inline-note"><Info size={16} /><span>NSE, MSE, RMSE and NRMSE rank candidates identically on one fixed calibration support. KGE and MAE can select genuinely different parameter sets.</span></div>
          </section>

          <section className="card">
            <div className="card-heading"><div className="icon-tile green"><Settings2 size={19} /></div><div><h2>Search configuration</h2><p>Select a bounded and reproducible parameter optimizer.</p></div></div>
            <div className="choice-grid two-choices">
              <button className={`choice-card horizontal ${form.calibrationAlgorithm === "differential_evolution.rand1bin.v1" ? "selected" : ""}`} onClick={() => onForm({ calibrationAlgorithm: "differential_evolution.rand1bin.v1" })}>
                <Activity size={20} /><div><strong>Differential evolution</strong><span>Recommended · population search · DE/rand/1/bin</span></div>{form.calibrationAlgorithm === "differential_evolution.rand1bin.v1" && <i><Check size={12} /></i>}
              </button>
              <button className={`choice-card horizontal ${form.calibrationAlgorithm === "uniform_random_search.v1" ? "selected" : ""}`} onClick={() => onForm({ calibrationAlgorithm: "uniform_random_search.v1" })}>
                <FlaskConical size={20} /><div><strong>Uniform random search</strong><span>Reference baseline · independent candidates</span></div>{form.calibrationAlgorithm === "uniform_random_search.v1" && <i><Check size={12} /></i>}
              </button>
            </div>
            {form.calibrationAlgorithm === "differential_evolution.rand1bin.v1" ? <>
              <div className="field-grid three-columns">
                <Field label="Generations" hint="One complete population update"><input type="number" min="1" max="100000" value={form.deGenerations} onChange={(event) => onForm({ deGenerations: Number(event.target.value) })} /></Field>
                <Field label="Population size" hint={`Blank = automatic (${resolvedPopulation})`}><input type="number" min="4" max="100000" placeholder="Automatic" value={form.dePopulationSize ?? ""} onChange={(event) => onForm({ dePopulationSize: event.target.value === "" ? null : Number(event.target.value) })} /></Field>
                <Field label="Random seed" hint="Reproduces initialization and trials"><input type="number" value={form.seed} onChange={(event) => onForm({ seed: Number(event.target.value) })} /></Field>
              </div>
              <details className="advanced-settings">
                <summary><span><strong>Advanced DE controls</strong><small>Classic mutation and binomial crossover settings</small></span><ChevronDown size={17} /></summary>
                <div className="field-grid two-columns">
                  <Field label="Mutation factor F" hint="0 < F ≤ 2"><input type="number" min="0.000001" max="2" step="0.05" value={form.deMutationFactor} onChange={(event) => onForm({ deMutationFactor: Number(event.target.value) })} /></Field>
                  <Field label="Crossover probability CR" hint="0 ≤ CR ≤ 1"><input type="number" min="0" max="1" step="0.05" value={form.deCrossoverProbability} onChange={(event) => onForm({ deCrossoverProbability: Number(event.target.value) })} /></Field>
                </div>
              </details>
              <div className="inline-note"><Info size={16} /><span>Parameters evolve in a normalized unit cube, then return to their physical bounds for simulation. Integer parameters are rounded only at evaluation. The run uses a fixed generation budget and does not claim proof of a global optimum.</span></div>
            </> : <>
              <div className="field-grid two-columns">
                <Field label="Candidate evaluations" hint="1 to 1,000,000"><input type="number" min="1" max="1000000" value={form.evaluations} onChange={(event) => onForm({ evaluations: Number(event.target.value) })} /></Field>
                <Field label="Random seed" hint="Makes sampling reproducible"><input type="number" value={form.seed} onChange={(event) => onForm({ seed: Number(event.target.value) })} /></Field>
              </div>
              <div className="inline-note"><Info size={16} /><span>The declared default parameter set is evaluated first. This baseline samples every remaining candidate independently and is retained for legacy reproducibility.</span></div>
            </>}
          </section>
        </div>

        <aside className="card calibration-summary">
          <p className="section-kicker">Before execution</p><h2>Evaluation setup</h2>
          <div className="summary-stat"><span>Calibration rows</span><strong>{calibrationRows.toLocaleString()}</strong></div>
          <div className="summary-stat"><span>Validation rows</span><strong>{validationRows ? validationRows.toLocaleString() : "Not used"}</strong></div>
          <div className="summary-stat"><span>Test rows</span><strong>{testRows ? testRows.toLocaleString() : "Not available"}</strong></div>
          <div className="summary-stat"><span>Primary reported partition</span><strong>{testRows ? "Held-out test" : "Calibration only"}</strong></div>
          <div className="summary-stat"><span>Optimizer</span><strong>{form.calibrationAlgorithm === "differential_evolution.rand1bin.v1" ? "Differential evolution" : "Random baseline"}</strong></div>
          <div className="summary-stat"><span>Maximum simulations</span><strong>{evaluationBudget.toLocaleString()}</strong></div>
          <div className="summary-stat"><span>Parameter dimensions</span><strong>{parameterCount}</strong></div>
          <div className="summary-divider" />
          <p className="summary-copy">The optimizer sees {scored.toLocaleString()} finite calibration observations after warm-up. Every metric is then recomputed separately on calibration, validation when present, and held-out test.</p>
          <span className={`readiness ${valid ? "ready" : "not-ready"}`}>{valid ? <CircleCheck size={16} /> : <CircleAlert size={16} />}{valid ? "Ready for preflight validation" : "Review the period settings"}</span>
        </aside>
      </div>
      <div className="step-footer"><span>{valid ? "Calibration and held-out evaluation are chronologically consistent." : "Calibration, validation, and test boundaries must be ordered and non-empty."}</span><button className="button button-primary" disabled={!valid} onClick={onNext}>Review experiment <ArrowRight size={16} /></button></div>
    </>
  );
}

function RunPanel({
  form,
  preview,
  quality,
  validation,
  task,
  busy,
  onValidate,
  onStart,
  onCancel,
  onDownload,
}: {
  form: WorkspaceForm;
  preview: CsvPreview;
  quality: DataQualityReport | null;
  validation: ValidationSummary | null;
  task: TaskSnapshot | null;
  busy: boolean;
  onValidate: () => void;
  onStart: () => void;
  onCancel: () => void;
  onDownload: () => void;
}) {
  const event = task?.latest_event;
  const running = task !== null && !task.terminal;
  const [nowMilliseconds, setNowMilliseconds] = useState(() => Date.now());
  useEffect(() => {
    if (!running) return undefined;
    setNowMilliseconds(Date.now());
    const timer = window.setInterval(() => setNowMilliseconds(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [running, task?.task_id]);
  const timing = task ? liveTaskTiming(task, nowMilliseconds) : null;
  const fraction = event?.fraction ?? 0;
  const isMlp = isMlpModel(form.modelId);
  const isLstm = isLstmModel(form.modelId);
  const isGru = isGruModel(form.modelId);
  const isCnn = isCnnModel(form.modelId);
  const isPaml = isPamlModel(form.modelId);
  const isDiscovery = isDiscoveryModel(form.modelId);
  const isRecurrent = isLstm || isGru;
  const isTrainable = isMlp || isRecurrent || isCnn || isPaml || isDiscovery;
  const calibrationBudget = calibrationEvaluationBudget(form);
  const neuralMemberCount = (
    !isDiscovery && form.initializationEnsembleEnabled
      ? form.initializationEnsembleMembers
      : 1
  );
  return (
    <>
      <PageHeader
        eyebrow="Preflight & execution"
        title={running ? (isDiscovery ? "Process discovery is running" : isTrainable ? "Neural training is running" : "Calibration is running") : "Review the complete experiment"}
        description={running ? `The scientific engine is isolated from the interface. You can monitor or cancel at a safe ${isTrainable ? "epoch" : "candidate"} boundary.` : `HydroSoftware validates the data and experiment setup before ${isTrainable ? "training" : "any candidate is evaluated"}.`}
        actions={<button className="button button-secondary" onClick={onDownload}><Download size={16} /> Export configuration</button>}
      />
      <div className="content-grid run-grid">
        <div className="stack">
          <section className="card review-card">
            <div className="review-section"><span className="review-number">01</span><div><small>Data</small><strong>{preview.relative_path.split("/").at(-1)}</strong><p>{preview.row_count.toLocaleString()} rows · {preview.columns.length} mapped columns</p></div><CircleCheck size={18} /></div>
            <div className="review-section"><span className="review-number">02</span><div><small>Data quality</small><strong>{quality?.transformed_value_count.toLocaleString() ?? 0} transformed values</strong><p>{quality?.warning_count ?? 0} documented warnings · source fingerprint verified</p></div><CircleCheck size={18} /></div>
            <div className="review-section"><span className="review-number">03</span><div><small>Model</small><strong>{isDiscovery ? "DeepDiscover · conservative single-DU" : isPaml ? `PaML · ${form.pamlPhysicalModelId === "hbv.mass_conservative_daily.v1" ? "HBV" : "EXP-HYDRO"} P-RNN + causal CNN` : isLstm ? `LSTM · TensorFlow stacked ${isS2SNeuralModel(form.modelId) ? "causal S2S" : "one-step S2V"}` : isGru ? `GRU · TensorFlow stacked ${isS2SNeuralModel(form.modelId) ? "causal S2S" : "one-step S2V"}` : isCnn ? `CNN-1D · TensorFlow causal ${isS2SNeuralModel(form.modelId) ? "S2S" : "S2V"}` : isMlp ? `MLP · TensorFlow ${isS2SNeuralModel(form.modelId) ? "time-distributed S2S" : "one-step S2V"}` : form.modelId === "hbv.mass_conservative_daily.v1" ? "HBV · daily response" : "EXP-HYDRO · daily conceptual model"}</strong><p>{isDiscovery ? `${form.discoveryProcessCount} discovered processes · ${form.discoveryHiddenUnits} hidden units · exact analytical balance` : isPaml ? `${form.warmupSteps}-day training warm-up · ${form.cnnLayers[0]?.filters ?? 8} CNN filters` : isTrainable ? `${form.mlpLookback}-day ${isS2SNeuralModel(form.modelId) ? `sequence · ${form.s2sWarmup}-day causal warm-up` : "lookback"} · ${isRecurrent ? `${form.recurrentLayers} recurrent units by layer` : isCnn ? `${form.cnnLayers.length} causal convolutional layers` : `${form.mlpHiddenLayers} hidden units`}` : <>{snowLabel(form.snowModule)} · {calibratedParameterCount(form)} calibrated parameters{form.modelId === "exp_hydro.mass_conservative.v1" ? ` · ${form.substeps} substeps/day` : " · daily response"}</>}</p></div><CircleCheck size={18} /></div>
            <div className="review-section"><span className="review-number">04</span><div><small>{isTrainable ? "Training" : "Calibration"}</small><strong>{isDiscovery ? `Full-sequence Adam · validation ${objectiveOption(form.objective).symbol} checkpoint` : isPaml ? `Joint Adam · validation ${objectiveOption(form.objective).symbol} checkpoint` : isTrainable ? (neuralMemberCount === 1 ? `Adam · validation ${objectiveOption(form.objective).symbol} checkpoint` : `Adam · ${neuralMemberCount}-member ${objectiveOption(form.objective).symbol} ensemble`) : objectiveOption(form.objective).name}</strong><p>{isPaml ? `up to ${form.trainingEpochs.toLocaleString()} joint epochs` : isTrainable ? `${(form.trainingEpochs * neuralMemberCount).toLocaleString()} maximum member-epochs` : `${calibrationAlgorithmLabel(form)} · ${calibrationBudget.toLocaleString()} maximum simulations`} · base seed {form.seed}</p></div><CircleCheck size={18} /></div>
          </section>

          {validation && (
            <section className="notice notice-success validation-notice">
              <CircleCheck size={22} /><div><strong>Preflight validation passed</strong><p>{validation.scored_observations.toLocaleString()} observations belong to the primary {validation.primary_evaluation_partition} evaluation after {validation.warmup_steps} warm-up steps.</p><small>Configuration {validation.configuration_hash.slice(0, 16)}…</small></div>
            </section>
          )}

          {task && (
            <section className="card task-card">
              <div className="task-topline"><div><span className={`pulse ${task.terminal ? "terminal" : ""}`} /><p className="section-kicker">{task.state.replaceAll("_", " ")}</p><h2>{event?.message ?? "Task created"}</h2></div><span className="task-id">{task.task_id.slice(0, 18)}…</span></div>
              <div className="progress-track"><span style={{ width: `${Math.max(0, Math.min(100, fraction * 100))}%` }} /></div>
              <div className="task-metrics">
                <div><span>Progress</span><strong>{event?.total_units ? `${event.completed_units} / ${event.total_units}` : "Preparing"}</strong></div>
                <div><span>Best score</span><strong>{formatScore(event?.best_score)}</strong></div>
                <div><span>Elapsed</span><strong>{formatSeconds(timing?.elapsed)}</strong></div>
                <div><span>Remaining</span><strong>{formatSeconds(timing?.remaining)}</strong></div>
              </div>
            </section>
          )}
        </div>

        <aside className="card action-card">
          <div className={`action-orb ${validation ? "validated" : ""}`}>{validation ? <Check size={30} /> : <FlaskConical size={30} />}</div>
          <h2>{running ? "Experiment in progress" : validation ? (isTrainable ? "Ready to train" : "Ready to calibrate") : "Validate before running"}</h2>
          <p>{running ? (isTrainable ? `${isDiscovery ? "The current full-sequence gradient pass" : "The current mini-batch"} is allowed to finish; cancellation occurs before the next epoch.` : "Completed candidates are retained if you cancel. The active simulation is allowed to finish safely.") : validation ? "All inexpensive checks passed. Starting will create an asynchronous, cancellable task." : `This catches data, period, mapping, ${isTrainable ? "partition, architecture, and optimizer" : "objective, and initial-state"} problems before execution.`}</p>
          {!task && !validation && <button className="button button-primary button-large" onClick={onValidate} disabled={busy}>{busy ? "Validating…" : "Validate experiment"}<ArrowRight size={17} /></button>}
          {!task && validation && <button className="button button-primary button-large" onClick={onStart} disabled={busy}><Play size={17} fill="currentColor" /> {isDiscovery ? "Start discovery" : isTrainable ? "Start training" : "Start calibration"}</button>}
          {running && <button className="button button-danger button-large" onClick={onCancel}><Square size={15} fill="currentColor" /> Request cancellation</button>}
          {task?.terminal && task.state !== "completed" && <span className="readiness not-ready"><CircleAlert size={16} /> {task.state.replaceAll("_", " ")}</span>}
          <div className="action-footnote"><span>Local execution</span><span>Reproducible seed</span><span>Safe cancellation</span></div>
        </aside>
      </div>
    </>
  );
}

function ForecastDialog({
  sourceResult,
  sourceConfig,
  onClose,
  onComplete,
  onError,
}: {
  sourceResult: RunResult;
  sourceConfig: ExperimentConfig;
  onClose: () => void;
  onComplete: (result: RunResult) => void;
  onError: (error: StructuredError) => void;
}) {
  const continuingEnsemble = Boolean(sourceResult.forecast_ensemble);
  const previousMemberIds = sourceResult.forecast_ensemble?.member_ids ?? [];
  const isStandaloneNeural = isMlpModel(sourceConfig.model.model_id)
    || isRecurrentModel(sourceConfig.model.model_id)
    || isCnnModel(sourceConfig.model.model_id);
  const isPaml = isPamlModel(sourceConfig.model.model_id);
  const isDiscovery = isDiscoveryModel(sourceConfig.model.model_id);
  const isNeural = isStandaloneNeural || isPaml;
  const isTrained = isNeural || isDiscovery;
  const activeLookback = isPaml
    ? sourceConfig.model.corrector?.lookback ?? 1
    : sourceConfig.model.lookback ?? 1;
  const activeNeuralConfig = isPaml
    ? sourceConfig.model.corrector
    : sourceConfig.model;
  const activeIsS2S = sourceConfig.model.model_id.includes("_s2s.");
  const activeWarmup = activeIsS2S
    ? activeNeuralConfig?.s2s_warmup ?? 0
    : activeLookback;
  const laggedContext = activeNeuralConfig?.lagged_features ?? [];
  const requiredContextRoles = Array.from(new Set([
    ...(activeNeuralConfig?.feature_roles ?? []),
    ...laggedContext.map((item) => item.source),
  ]));
  const contextHistoryLength = activeLookback + Math.max(
    0,
    ...laggedContext.map((item) => item.lag_steps),
  );
  const requiredAdditional = sourceConfig.data.columns.additional_predictors ?? [];
  const [forecastMode, setForecastMode] = useState<"deterministic" | "ensemble">(
    continuingEnsemble ? "ensemble" : "deterministic",
  );
  const [preview, setPreview] = useState<CsvPreview | null>(null);
  const [ensemblePreviews, setEnsemblePreviews] = useState<CsvPreview[]>([]);
  const [ensembleMemberIds, setEnsembleMemberIds] = useState<string[]>([]);
  const [issueId, setIssueId] = useState("");
  const [spinupSteps, setSpinupSteps] = useState(0);
  const [mapping, setMapping] = useState<ColumnMapping>({
    timestamp: "",
    precipitation: "",
    temperature: "",
    potential_evapotranspiration: "",
    streamflow: "",
  });
  const [additionalMapping, setAdditionalMapping] = useState(
    () => requiredAdditional.map((item) => ({ ...item, column: "" })),
  );
  const [initialization, setInitialization] = useState<"source_final" | "configured">("source_final");
  const [precipitationUnit, setPrecipitationUnit] = useState(sourceConfig.data.processing.units.precipitation);
  const [temperatureUnit, setTemperatureUnit] = useState(sourceConfig.data.processing.units.temperature);
  const [petUnit, setPetUnit] = useState(sourceConfig.data.processing.units.potential_evapotranspiration);
  const [streamflowUnit, setStreamflowUnit] = useState(sourceConfig.data.processing.units.streamflow);
  const [busy, setBusy] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const assigned = [
    mapping.precipitation,
    mapping.temperature,
    mapping.potential_evapotranspiration,
    mapping.timestamp,
    mapping.streamflow,
    ...additionalMapping.map((item) => item.column),
  ].filter(Boolean);
  const activePreviews = forecastMode === "ensemble"
    ? ensemblePreviews
    : preview ? [preview] : [];
  const selectedColumns = assigned.filter(Boolean);
  const memberSchemasAligned = activePreviews.every(
    (item) => selectedColumns.every((column) => item.columns.includes(column)),
  ) && (
    activePreviews.length === 0
    || activePreviews.every((item) => item.row_count === activePreviews[0].row_count)
  );
  const mappingValid = Boolean(
    preview
    && mapping.precipitation
    && mapping.temperature
    && mapping.potential_evapotranspiration
    && additionalMapping.every((item) => item.column)
    && new Set(assigned).size === assigned.length
    && memberSchemasAligned
    && (
      forecastMode === "deterministic"
      || (
        ensemblePreviews.length >= 2
        && (!continuingEnsemble || (
          ensemblePreviews.length === previousMemberIds.length
          && ensembleMemberIds.length === previousMemberIds.length
          && new Set(ensembleMemberIds).size === previousMemberIds.length
          && previousMemberIds.every((memberId) => ensembleMemberIds.includes(memberId))
          && Boolean(mapping.timestamp)
        ))
        && spinupSteps < (preview?.row_count ?? 0)
        && /^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$/.test(issueId)
      )
    )
  );

  const importForcing = async (file: File) => {
    setBusy(true);
    try {
      const next = await importCsv(file);
      setPreview(next);
      setEnsemblePreviews([]);
      setEnsembleMemberIds([]);
      setMapping(autoMapColumns(next.columns));
      setAdditionalMapping(requiredAdditional.map((item) => ({
        ...item,
        column: next.columns.includes(item.column)
          ? item.column
          : next.columns.find((column) => column === item.name) ?? "",
      })));
    } catch (value) {
      onError(errorFrom(value));
    } finally {
      setBusy(false);
      if (input.current) input.current.value = "";
    }
  };

  const importForcingMembers = async (files: File[]) => {
    if (files.length < 2 || files.length > 50) {
      onError({
        code: "forecast_ensemble_member_count",
        message: "Choose between 2 and 50 meteorological member CSV files.",
      });
      return;
    }
    if (continuingEnsemble && files.length !== previousMemberIds.length) {
      onError({
        code: "forecast_ensemble_member_identity",
        message: `This cycle requires exactly ${previousMemberIds.length} files, one for each archived meteorological member.`,
      });
      return;
    }
    setBusy(true);
    try {
      const next = await Promise.all(files.map((file) => importCsv(file)));
      const mapped = autoMapColumns(next[0].columns);
      const firstTimestamp = mapped.timestamp
        ? next[0].rows[0]?.[mapped.timestamp] ?? ""
        : "";
      const inferredIssue = (
        firstTimestamp
        || files[0]?.name.replace(/\.[^.]+$/, "")
        || ""
      )
        .replace(/[^A-Za-z0-9._-]+/g, "-")
        .replace(/^-+|-+$/g, "")
        .slice(0, 128);
      setEnsemblePreviews(next);
      setEnsembleMemberIds(
        continuingEnsemble
          ? [...previousMemberIds]
          : next.map((_, index) => `member-${String(index + 1).padStart(2, "0")}`),
      );
      setPreview(next[0]);
      setMapping(mapped);
      setIssueId(inferredIssue);
      setAdditionalMapping(requiredAdditional.map((item) => ({
        ...item,
        column: next[0].columns.includes(item.column)
          ? item.column
          : next[0].columns.find((column) => column === item.name) ?? "",
      })));
    } catch (value) {
      onError(errorFrom(value));
    } finally {
      setBusy(false);
      if (input.current) input.current.value = "";
    }
  };

  const launchForecast = async () => {
    if (!preview || !mappingValid) return;
    const chainingCurrentForecast = !continuingEnsemble
      && sourceResult.run_kind === "forecast"
      && initialization === "source_final";
    const chainingStateful = chainingCurrentForecast && (!isNeural || isDiscovery);
    const chainingPaml = chainingCurrentForecast && isPaml;
    const contextSource = sourceResult.neural_context ?? sourceResult.forcings;
    const antecedentContext = Object.fromEntries(
      requiredContextRoles.map((name) => [
        name,
        (contextSource[name] ?? []).slice(-contextHistoryLength),
      ]),
    );
    const columns: ForecastRequest["data"]["columns"] = {
      precipitation: mapping.precipitation,
      temperature: mapping.temperature,
      potential_evapotranspiration: mapping.potential_evapotranspiration,
    };
    if (mapping.timestamp) columns.timestamp = mapping.timestamp;
    if (mapping.streamflow) columns.streamflow = mapping.streamflow;
    if (additionalMapping.length) {
      columns.additional_predictors = additionalMapping;
    }
    const request: ForecastRequest = {
      source_task_id: sourceResult.source_task_id ?? sourceResult.task_id,
      initialization: chainingStateful || chainingPaml
        ? "custom"
        : initialization,
      ...(chainingStateful || chainingPaml
        ? { initial_state: sourceResult.best_simulation.states.at(-1) }
        : {}),
      ...((isStandaloneNeural || isPaml) && chainingCurrentForecast
        ? { antecedent_forcings: antecedentContext }
        : {}),
      data: {
        ...sourceConfig.data,
        path: preview.relative_path,
        expected_sha256: preview.sha256,
        columns,
        processing: {
          ...sourceConfig.data.processing,
          units: {
            ...sourceConfig.data.processing.units,
            precipitation: precipitationUnit,
            temperature: temperatureUnit,
            potential_evapotranspiration: petUnit,
            streamflow: streamflowUnit,
          },
          timestamps: {
            ...sourceConfig.data.processing.timestamps,
            require_timestamp: Boolean(mapping.timestamp),
          },
        },
      },
    };
    setBusy(true);
    try {
      if (forecastMode === "ensemble") {
        const ensembleRequest: ForecastEnsembleRequest = {
          source_task_id: request.source_task_id,
          ...(continuingEnsemble
            ? { continuation_task_id: sourceResult.task_id }
            : {}),
          ...(issueId.trim() ? { issue_id: issueId.trim() } : {}),
          spinup_steps: spinupSteps,
          initialization: request.initialization,
          ...(request.initial_state ? { initial_state: request.initial_state } : {}),
          ...(request.antecedent_forcings
            ? { antecedent_forcings: request.antecedent_forcings }
            : {}),
          lower_quantile: 0.1,
          upper_quantile: 0.9,
          members: ensemblePreviews.map((memberPreview, index) => ({
            member_id: ensembleMemberIds[index]
              ?? `member-${String(index + 1).padStart(2, "0")}`,
            data: {
              ...request.data,
              path: memberPreview.relative_path,
              expected_sha256: memberPreview.sha256,
            },
          })),
        };
        onComplete(await startForecastEnsemble(ensembleRequest));
      } else {
        onComplete(await startForecast(request));
      }
    } catch (value) {
      onError(errorFrom(value));
    } finally {
      setBusy(false);
    }
  };

  const mappingField = (role: keyof ColumnMapping, label: string, optional = false) => (
    <Field label={label} hint={optional ? "Optional" : undefined}>
      <select value={mapping[role]} disabled={!preview} onChange={(event) => setMapping((current) => ({ ...current, [role]: event.target.value }))}>
        <option value="">{optional ? "Not provided" : "Select column"}</option>
        {preview?.columns.map((column) => <option key={column} value={column}>{column}</option>)}
      </select>
    </Field>
  );

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <section className="forecast-dialog" role="dialog" aria-modal="true" aria-labelledby="forecast-dialog-title" onMouseDown={(event) => event.stopPropagation()}>
        <div className="modal-heading">
          <div><p className="section-kicker">Frozen {isTrained ? "trained" : "calibrated"} model</p><h2 id="forecast-dialog-title">Run with new forcings</h2><p>The saved {isTrained ? "weights and transformations" : "parameters"} remain unchanged. Streamflow observations are optional.</p></div>
          <button className="notice-dismiss" onClick={onClose} aria-label="Close forecast dialog">×</button>
        </div>
        <div className="forecast-dialog-scroll">
          <section className="forecast-source-card">
            <div><strong>{sourceConfig.name}</strong><span>{sourceConfig.model.model_id}</span></div>
            <span className="status-pill neutral">{continuingEnsemble ? `Cycle ${sourceResult.forecast_ensemble?.cycle_index ?? 1}` : "Source"} {sourceResult.task_id}</span>
          </section>
          <section className="forecast-section">
            <div className="forecast-section-heading"><div><strong>Forecast forcing mode</strong><span>Use one forcing trajectory or aligned meteorological scenarios.</span></div></div>
            <div className="forecast-state-options forecast-mode-options" role="radiogroup" aria-label="Forecast forcing mode">
              <button type="button" disabled={continuingEnsemble} className={forecastMode === "deterministic" ? "selected" : ""} aria-pressed={forecastMode === "deterministic"} onClick={() => { setForecastMode("deterministic"); setEnsemblePreviews([]); setEnsembleMemberIds([]); }}><strong>Single trajectory</strong><span>{continuingEnsemble ? "An aggregate ensemble state cannot initialize one deterministic trajectory." : "One future forcing CSV produces one frozen-model trajectory."}</span><small>{continuingEnsemble ? "Return to the base run for this option" : "Deterministic forecast or hindcast"}</small></button>
              <button type="button" className={forecastMode === "ensemble" ? "selected" : ""} aria-pressed={forecastMode === "ensemble"} onClick={() => { setForecastMode("ensemble"); setPreview(ensemblePreviews[0] ?? null); }}><strong>Meteorological ensemble</strong><span>Two to 50 aligned CSV files produce a forcing-scenario ensemble.</span><small>CRPS and signed error by lead when Q is supplied</small></button>
            </div>
          </section>
          <section className="forecast-section">
            <div className="forecast-section-heading"><div><strong>1. Forecast data</strong><span>Daily standard forcings{requiredAdditional.length ? ` and ${requiredAdditional.length} saved additional predictor${requiredAdditional.length === 1 ? "" : "s"}` : ""} are required with one common schema.</span></div><button className="button button-secondary" disabled={busy} onClick={() => input.current?.click()}><Upload size={15} /> {activePreviews.length ? "Replace CSV files" : forecastMode === "ensemble" ? "Choose member CSVs" : "Choose CSV"}</button></div>
            {activePreviews.length > 0 && <div className={`forecast-file-list${forecastMode === "ensemble" ? " ensemble-files" : ""}`}>{activePreviews.map((item, index) => <div className="forecast-file-summary" key={item.sha256}><FileText size={17} /><div><strong>{item.relative_path.split("/").at(-1)}</strong><span>{item.row_count.toLocaleString()} rows · {item.columns.length} columns</span>{forecastMode === "ensemble" && (continuingEnsemble ? <label><small>Archived member identity</small><select aria-label={`Member identity for ${item.relative_path.split("/").at(-1)}`} value={ensembleMemberIds[index] ?? ""} onChange={(event) => setEnsembleMemberIds((current) => current.map((value, position) => position === index ? event.target.value : value))}>{previousMemberIds.map((memberId) => <option key={memberId} value={memberId}>{memberId}</option>)}</select></label> : <small>{ensembleMemberIds[index] ?? `member-${String(index + 1).padStart(2, "0")}`}</small>)}</div><CircleCheck size={17} /></div>)}</div>}
            <input ref={input} className="visually-hidden" type="file" multiple={forecastMode === "ensemble"} accept=".csv,text/csv" onChange={(event) => { const files = Array.from(event.target.files ?? []); if (forecastMode === "ensemble") { if (files.length) void importForcingMembers(files); } else if (files[0]) void importForcing(files[0]); }} />
            {forecastMode === "ensemble" && activePreviews.length > 0 && !memberSchemasAligned && <div className="inline-note warning"><CircleAlert size={16} /><span>Every member must have the same number of rows and contain all selected columns. Exact timestamp alignment is checked before simulation.</span></div>}
            {continuingEnsemble && activePreviews.length > 0 && <div className="inline-note"><Info size={16} /><span>Match every new file to its archived member identifier. The engine restores that member’s own final state and neural context, and requires the next consecutive daily timestamp.</span></div>}
            {forecastMode === "ensemble" && <Field label="Forecast-origin identifier" hint="Unique label used to assemble repeated hindcasts">
              <input value={issueId} maxLength={128} pattern="[A-Za-z0-9][A-Za-z0-9._-]*" onChange={(event) => setIssueId(event.target.value.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^[^A-Za-z0-9]+/, ""))} placeholder="e.g. hindcast-2026-04-01" />
            </Field>}
            {forecastMode === "ensemble" && <Field label="Shared state spin-up rows" hint="Excluded from forecast leads and verification">
              <input type="number" min="0" max={Math.max(0, (preview?.row_count ?? 1) - 1)} step="1" value={spinupSteps} onChange={(event) => {
                const next = Math.max(0, Math.min(Math.max(0, (preview?.row_count ?? 1) - 1), Number(event.target.value)));
                setSpinupSteps(next);
                const timestamp = mapping.timestamp ? preview?.rows[next]?.[mapping.timestamp] ?? "" : "";
                if (timestamp) setIssueId(timestamp.replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 128));
              }} />
            </Field>}
            {forecastMode === "ensemble" && spinupSteps > 0 && <div className="inline-note"><Info size={16} /><span>The first {spinupSteps} rows must contain identical observed antecedent forcings in every member. They update the model state but are not treated as forecast leads.</span></div>}
          </section>
          <section className="forecast-section">
            <div className="forecast-section-heading"><div><strong>2. Column mapping</strong><span>Observed streamflow is used only for comparison and metrics.</span></div></div>
            <div className="forecast-mapping-grid">
              {mappingField("timestamp", "Timestamp", true)}
              {mappingField("precipitation", "Precipitation")}
              {mappingField("temperature", "Temperature")}
              {mappingField("potential_evapotranspiration", "Potential ET")}
              {mappingField("streamflow", "Observed streamflow", true)}
              {additionalMapping.map((item, index) => <Field key={item.name} label={item.name.replaceAll("_", " ")} hint={`Required additional predictor · ${item.unit}`}>
                <select value={item.column} disabled={!preview} onChange={(event) => setAdditionalMapping((current) => current.map((candidate, position) => position === index ? { ...candidate, column: event.target.value } : candidate))}>
                  <option value="">Select column</option>
                  {preview?.columns.map((column) => <option key={column} value={column}>{column}</option>)}
                </select>
              </Field>)}
            </div>
          </section>
          <section className="forecast-section">
            <div className="forecast-section-heading"><div><strong>3. Source units</strong><span>Values are converted to the model's daily units.</span></div></div>
            <div className="forecast-unit-grid">
              <Field label="Precipitation"><select value={precipitationUnit} onChange={(event) => setPrecipitationUnit(event.target.value as typeof precipitationUnit)}><option value="mm/day">mm/day</option><option value="inch/day">inch/day</option></select></Field>
              <Field label="Temperature"><select value={temperatureUnit} onChange={(event) => setTemperatureUnit(event.target.value as typeof temperatureUnit)}><option value="degC">°C</option><option value="degF">°F</option><option value="K">K</option></select></Field>
              <Field label="Potential ET"><select value={petUnit} onChange={(event) => setPetUnit(event.target.value as typeof petUnit)}><option value="mm/day">mm/day</option><option value="inch/day">inch/day</option></select></Field>
              <Field label="Observed streamflow"><select value={streamflowUnit} disabled={!mapping.streamflow} onChange={(event) => setStreamflowUnit(event.target.value as typeof streamflowUnit)}><option value="mm/day">mm/day</option><option value="m3/s">m³/s</option></select></Field>
            </div>
          </section>
          <section className="forecast-section">
            <div className="forecast-section-heading"><div><strong>4. {isNeural ? "Antecedent context" : "Initial state"}</strong><span>Choose how the saved model enters the new forcing period.</span></div></div>
            {continuingEnsemble ? <div className="inline-note"><Info size={16} /><span>Initialization is fixed to member-wise continuation. The displayed mean state is never reused; each scenario resumes from its matching archived state and context.</span></div> : <div className="forecast-state-options" role="radiogroup" aria-label="Forecast initialization">
              <button type="button" className={initialization === "source_final" ? "selected" : ""} aria-pressed={initialization === "source_final"} onClick={() => setInitialization("source_final")}><strong>{isPaml ? "Continue physics and corrector context" : isNeural ? "Continue with recent predictors" : "Continue from final state"}</strong><span>{isPaml ? `Uses the final physical states plus the last ${activeLookback} corrector rows.` : isNeural ? `Uses the last ${activeLookback} predictor rows as antecedent context.` : "Uses the last saved storages and routing memory."}</span><small>Recommended for the next chronological period</small></button>
              <button type="button" className={initialization === "configured" ? "selected" : ""} aria-pressed={initialization === "configured"} onClick={() => setInitialization("configured")}><strong>{isPaml ? "Configured physics without saved context" : isNeural ? "No saved context" : "Configured initial state"}</strong><span>{isPaml ? `Restarts physical states; the first ${activeWarmup} rows condition the corrector and are not emitted.` : isNeural ? `The first ${activeWarmup} forecast rows provide causal context and have no prediction.` : "Restarts from the experiment's declared values."}</span><small>Useful for an independent hindcast</small></button>
            </div>}
          </section>
        </div>
        <div className="forecast-dialog-footer"><span>{mappingValid ? `${preview?.row_count.toLocaleString()} forcing intervals × ${activePreviews.length} ${activePreviews.length === 1 ? "trajectory" : "members"} ready` : forecastMode === "ensemble" ? "Import at least two aligned member files and map the required forcings." : "Import and map the three required forcings."}</span><div><button className="button button-secondary" onClick={onClose}>Cancel</button><button className="button button-primary" disabled={!mappingValid || busy} onClick={launchForecast}><CloudSun size={16} /> {busy ? "Simulating…" : forecastMode === "ensemble" ? "Run ensemble" : "Run forecast"}</button></div></div>
      </section>
    </div>
  );
}

export function ForecastPostprocessingResults({
  result,
  onBack,
}: {
  result: ForecastPostprocessing;
  onBack: () => void;
}) {
  return (
    <div className="hindcast-results">
      <div className="ensemble-results-heading"><div className="icon-tile cyan"><Gauge size={20} /></div><div><p className="section-kicker">Independent post-processing evaluation</p><h2>{result.split.training_case_count} fit + {result.split.evaluation_case_count} evaluate</h2><p>The oldest origins estimate lead-wise affine mean and spread coefficients. Every reported score below uses only later, untouched origins.</p></div><button className="button button-secondary" onClick={onBack}><RotateCcw size={15} /> Change cases</button></div>
      <div className="forcing-score-grid">
        <article><span>Raw mean CRPS</span><strong>{formatDecimal(result.aggregate.raw.mean_crps, 4)}</strong><small>Independent evaluation</small></article>
        <article><span>Adjusted mean CRPS</span><strong>{formatDecimal(result.aggregate.adjusted.mean_crps, 4)}</strong><small>Lower is better</small></article>
        <article><span>CRPSS</span><strong>{result.aggregate.crps_skill_score === null ? "—" : formatDecimal(result.aggregate.crps_skill_score, 3)}</strong><small>Positive means improvement</small></article>
        <article><span>Adjusted coverage</span><strong>{Math.round(result.aggregate.adjusted.interval_coverage * 100)}%</strong><small>{Math.round(result.aggregate.adjusted.nominal_interval_coverage * 100)}% nominal envelope</small></article>
      </div>
      <section className="forecast-section">
        <div className="forecast-section-heading"><div><strong>Raw versus adjusted by lead</strong><span>CRPSS = 1 − adjusted CRPS / raw CRPS on later origins only.</span></div></div>
        <div className="forcing-lead-table-wrap"><table className="forcing-lead-table hindcast-lead-table"><thead><tr><th>Lead</th><th>Cases</th><th>Raw CRPS</th><th>Adjusted CRPS</th><th>CRPSS</th><th>Raw bias</th><th>Adjusted bias</th><th>Adjusted coverage</th></tr></thead><tbody>{result.lead_time_diagnostics.map((lead) => <tr key={lead.lead_step}><td>+{lead.lead_step}</td><td>{lead.available_cases}</td>{lead.reported && lead.raw && lead.adjusted ? <><td>{formatDecimal(lead.raw.mean_crps, 4)}</td><td>{formatDecimal(lead.adjusted.mean_crps, 4)}</td><td>{lead.crps_skill_score === null || lead.crps_skill_score === undefined ? "—" : formatDecimal(lead.crps_skill_score, 3)}</td><td>{formatDecimal(lead.raw.mean_error_bias, 4)}</td><td>{formatDecimal(lead.adjusted.mean_error_bias, 4)}</td><td>{Math.round(lead.adjusted.interval_coverage * 100)}%</td></> : <td colSpan={6}>Insufficient independent cases</td>}</tr>)}</tbody></table></div>
      </section>
      <section className="forecast-section">
        <div className="forecast-section-heading"><div><strong>Fitted lead-wise coefficients</strong><span>Adjusted member = max(0, intercept + slope × raw mean + spread multiplier × member anomaly).</span></div></div>
        <div className="forcing-lead-table-wrap"><table className="forcing-lead-table hindcast-lead-table"><thead><tr><th>Lead</th><th>Fit cases</th><th>Intercept</th><th>Mean slope</th><th>Spread multiplier</th><th>Fit raw CRPS</th><th>Fit adjusted CRPS</th></tr></thead><tbody>{result.parameters.map((parameter) => <tr key={parameter.lead_step}><td>+{parameter.lead_step}</td><td>{parameter.training_case_count}</td><td>{formatDecimal(parameter.intercept, 4)}</td><td>{formatDecimal(parameter.mean_slope, 4)}</td><td>{formatDecimal(parameter.spread_multiplier, 4)}</td><td>{formatDecimal(parameter.training_raw_mean_crps, 4)}</td><td>{formatDecimal(parameter.training_adjusted_mean_crps, 4)}</td></tr>)}</tbody></table></div>
      </section>
      <div className="shap-warnings">{result.warnings.map((warning) => <div key={warning.code}><Info size={15} /><span>{warning.message}</span></div>)}</div>
    </div>
  );
}

export function ForecastEventVerificationResults({
  result,
  onBack,
}: {
  result: ForecastEventVerification;
  onBack: () => void;
}) {
  const rawReliability = result.aggregate.raw.reliability_bins.flatMap((bin) => (
    bin.count > 0
      && bin.mean_forecast_probability !== null
      && bin.observed_frequency !== null
      ? [{
        probability: bin.mean_forecast_probability,
        frequency: bin.observed_frequency,
        count: bin.count,
      }]
      : []
  ));
  const adjustedReliability = result.aggregate.adjusted.reliability_bins.flatMap((bin) => (
    bin.count > 0
      && bin.mean_forecast_probability !== null
      && bin.observed_frequency !== null
      ? [{
        probability: bin.mean_forecast_probability,
        frequency: bin.observed_frequency,
        count: bin.count,
      }]
      : []
  ));
  const valueRows = result.aggregate.raw.cost_loss_curve.map((row, index) => ({
    ratio: row.cost_loss_ratio,
    raw: row.relative_economic_value,
    adjusted: result.aggregate.adjusted.cost_loss_curve[index]?.relative_economic_value ?? null,
  }));
  const adjustedSkill = result.aggregate.adjusted_brier_skill_vs_raw;

  return (
    <div className="hindcast-results event-verification-results">
      <div className="ensemble-results-heading"><div className="icon-tile cyan"><CircleAlert size={20} /></div><div><p className="section-kicker">Independent threshold-event evaluation</p><h2>Q &gt; {formatDecimal(result.threshold.value, 3)} {result.threshold.unit}</h2><p>{result.split.training_case_count} older origins define the threshold, climatology and adjustment; {result.split.evaluation_case_count} later origins evaluate every score below.</p></div><button className="button button-secondary" onClick={onBack}><RotateCcw size={15} /> Change cases</button></div>
      <div className="forcing-score-grid">
        <article><span>Raw Brier score</span><strong>{formatDecimal(result.aggregate.raw.brier_score, 4)}</strong><small>Lower is better</small></article>
        <article><span>Adjusted Brier score</span><strong>{formatDecimal(result.aggregate.adjusted.brier_score, 4)}</strong><small>Same independent events</small></article>
        <article><span>Adjusted skill vs raw</span><strong>{adjustedSkill === null ? "—" : formatDecimal(adjustedSkill, 3)}</strong><small>Positive means improvement</small></article>
        <article><span>Training climatology</span><strong>{Math.round(result.climatology.training_event_probability * 100)}%</strong><small>Reference event probability</small></article>
      </div>
      <section className="forecast-section">
        <div className="forecast-section-heading"><div><strong>Event diagnostics by lead</strong><span>BSS uses the older-archive climatology; AUC is unavailable when a lead contains only events or only non-events.</span></div></div>
        <div className="forcing-lead-table-wrap"><table className="forcing-lead-table hindcast-lead-table"><thead><tr><th>Lead</th><th>Cases</th><th>Events</th><th>Raw BS</th><th>Adjusted BS</th><th>Skill vs raw</th><th>Raw BSS</th><th>Adjusted BSS</th><th>Adjusted AUC</th></tr></thead><tbody>{result.lead_time_diagnostics.map((lead) => <tr key={lead.lead_step}><td>+{lead.lead_step}</td><td>{lead.available_cases}</td>{lead.reported && lead.raw && lead.adjusted ? <><td>{lead.raw.event_count}</td><td>{formatDecimal(lead.raw.brier_score, 4)}</td><td>{formatDecimal(lead.adjusted.brier_score, 4)}</td><td>{lead.adjusted_brier_skill_vs_raw === null || lead.adjusted_brier_skill_vs_raw === undefined ? "—" : formatDecimal(lead.adjusted_brier_skill_vs_raw, 3)}</td><td>{lead.raw.brier_skill_score === null ? "—" : formatDecimal(lead.raw.brier_skill_score, 3)}</td><td>{lead.adjusted.brier_skill_score === null ? "—" : formatDecimal(lead.adjusted.brier_skill_score, 3)}</td><td>{lead.adjusted.roc_auc === null ? "—" : formatDecimal(lead.adjusted.roc_auc, 3)}</td></> : <td colSpan={7}>Insufficient later cases</td>}</tr>)}</tbody></table></div>
      </section>
      <div className="event-diagnostic-grid">
        <section className="forecast-section hindcast-rank-card">
          <div className="forecast-section-heading"><div><strong>Reliability diagram</strong><span>Points compare mean predicted probability with observed event frequency. Point size is not weighted visually; inspect bin counts in the tooltip.</span></div></div>
          <div className="hindcast-rank-chart"><ResponsiveContainer width="100%" height="100%"><ScatterChart margin={{ top: 14, right: 18, left: 2, bottom: 8 }}><CartesianGrid stroke="var(--chart-grid)" /><XAxis type="number" dataKey="probability" name="Forecast probability" domain={[0, 1]} /><YAxis type="number" dataKey="frequency" name="Observed frequency" domain={[0, 1]} /><Tooltip cursor={{ strokeDasharray: "3 3" }} /><Legend /><ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--text-muted)" strokeDasharray="5 5" /><Scatter name="Raw ensemble" data={rawReliability} fill="var(--warning)" line /><Scatter name="Adjusted ensemble" data={adjustedReliability} fill="var(--accent)" line /></ScatterChart></ResponsiveContainer></div>
        </section>
        <section className="forecast-section hindcast-rank-card">
          <div className="forecast-section-heading"><div><strong>Relative economic value</strong><span>Potential value is shown across cost–loss ratios using the older-archive climatology as the operational reference.</span></div></div>
          <div className="hindcast-rank-chart"><ResponsiveContainer width="100%" height="100%"><LineChart data={valueRows} margin={{ top: 14, right: 18, left: 2, bottom: 8 }}><CartesianGrid stroke="var(--chart-grid)" vertical={false} /><XAxis type="number" dataKey="ratio" domain={[0, 1]} /><YAxis domain={["auto", "auto"]} /><Tooltip /><Legend /><ReferenceLine y={0} stroke="var(--text-muted)" strokeDasharray="5 5" /><Line type="linear" dataKey="raw" name="Raw ensemble" stroke="var(--warning)" strokeWidth={2} dot={{ r: 2 }} connectNulls={false} /><Line type="linear" dataKey="adjusted" name="Adjusted ensemble" stroke="var(--accent)" strokeWidth={2} dot={{ r: 2 }} connectNulls={false} /></LineChart></ResponsiveContainer></div>
        </section>
      </div>
      <div className="shap-warnings">{result.warnings.map((warning) => <div key={warning.code}><Info size={15} /><span>{warning.message}</span></div>)}</div>
    </div>
  );
}

export function HindcastVerificationDialog({
  currentResult,
  onClose,
  onError,
}: {
  currentResult: RunResult;
  onClose: () => void;
  onError: (error: StructuredError) => void;
}) {
  const [forecasts, setForecasts] = useState<ForecastEnsembleArchiveSummary[]>([]);
  const [archives, setArchives] = useState<ForecastVerificationArchiveSummary[]>([]);
  const [postprocessingArchives, setPostprocessingArchives] = useState<ForecastPostprocessingArchiveSummary[]>([]);
  const [eventArchives, setEventArchives] = useState<ForecastEventVerificationArchiveSummary[]>([]);
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [verification, setVerification] = useState<ForecastVerification | null>(null);
  const [postprocessing, setPostprocessing] = useState<ForecastPostprocessing | null>(null);
  const [eventVerification, setEventVerification] = useState<ForecastEventVerification | null>(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [seed, setSeed] = useState(0);
  const [minimumCases, setMinimumCases] = useState(2);
  const [trainingCases, setTrainingCases] = useState(5);
  const [eventThresholdMode, setEventThresholdMode] = useState<"absolute" | "training_quantile">("training_quantile");
  const [eventThresholdValue, setEventThresholdValue] = useState(1);
  const [eventTrainingQuantile, setEventTrainingQuantile] = useState(0.9);
  const [eventProbabilityBins, setEventProbabilityBins] = useState(10);
  const [eventMinimumCases, setEventMinimumCases] = useState(2);

  useEffect(() => {
    let active = true;
    Promise.all([
      listForecastEnsembles(),
      listForecastVerifications(),
      listForecastPostprocessings(),
      listForecastEventVerifications(),
    ])
      .then(([available, saved, savedPostprocessings, savedEvents]) => {
        if (!active) return;
        const compatible = available.filter((item) => (
          item.source_task_id === currentResult.source_task_id
          && item.experiment_id === currentResult.experiment_id
          && item.model_id === currentResult.model_id
        ));
        setForecasts(compatible);
        setArchives(saved.filter((item) => (
          item.source_task_id === currentResult.source_task_id
          && item.experiment_id === currentResult.experiment_id
          && item.model_id === currentResult.model_id
        )));
        setPostprocessingArchives(savedPostprocessings.filter((item) => (
          item.source_task_id === currentResult.source_task_id
          && item.experiment_id === currentResult.experiment_id
          && item.model_id === currentResult.model_id
        )));
        setEventArchives(savedEvents.filter((item) => (
          item.source_task_id === currentResult.source_task_id
          && item.experiment_id === currentResult.experiment_id
          && item.model_id === currentResult.model_id
        )));
        if (compatible.some((item) => item.task_id === currentResult.task_id)) {
          setSelected(new Set([currentResult.task_id]));
        }
      })
      .catch((value) => onError(errorFrom(value)))
      .finally(() => {
        if (active) setLoading(false);
      });
    return () => {
      active = false;
    };
  }, [
    currentResult.experiment_id,
    currentResult.model_id,
    currentResult.source_task_id,
    currentResult.task_id,
    onError,
  ]);

  const toggle = (taskId: string) => {
    setSelected((current) => {
      const next = new Set(current);
      if (next.has(taskId)) next.delete(taskId);
      else next.add(taskId);
      return next;
    });
  };
  const runVerification = async () => {
    if (selected.size < 2) return;
    setBusy(true);
    try {
      const next = await verifyForecastEnsembles([...selected], {
        seed,
        minimumCasesPerLead: minimumCases,
      });
      setVerification(next);
      setArchives((await listForecastVerifications()).filter((item) => (
        item.source_task_id === currentResult.source_task_id
        && item.experiment_id === currentResult.experiment_id
        && item.model_id === currentResult.model_id
      )));
    } catch (value) {
      onError(errorFrom(value));
    } finally {
      setBusy(false);
    }
  };
  const runPostprocessing = async () => {
    if (selected.size < 7) return;
    setBusy(true);
    try {
      const next = await postprocessForecastEnsembles([...selected], {
        seed,
        trainingCaseCount: Math.min(
          Math.max(5, trainingCases),
          selected.size - 2,
        ),
        minimumTrainingCasesPerLead: 5,
      });
      setPostprocessing(next);
      setPostprocessingArchives(
        (await listForecastPostprocessings()).filter((item) => (
          item.source_task_id === currentResult.source_task_id
          && item.experiment_id === currentResult.experiment_id
          && item.model_id === currentResult.model_id
        )),
      );
    } catch (value) {
      onError(errorFrom(value));
    } finally {
      setBusy(false);
    }
  };
  const runEventVerification = async () => {
    if (selected.size < 7) return;
    const resolvedTrainingCases = Math.min(
      Math.max(5, trainingCases),
      selected.size - 2,
    );
    const evaluationCases = selected.size - resolvedTrainingCases;
    setBusy(true);
    try {
      const next = await verifyForecastEvents([...selected], {
        trainingCaseCount: resolvedTrainingCases,
        thresholdMode: eventThresholdMode,
        ...(eventThresholdMode === "absolute"
          ? { thresholdValue: eventThresholdValue }
          : {}),
        trainingQuantile: eventTrainingQuantile,
        probabilityBinCount: eventProbabilityBins,
        minimumEvaluationCasesPerLead: Math.min(
          Math.max(2, eventMinimumCases),
          evaluationCases,
        ),
      });
      setEventVerification(next);
      setEventArchives(
        (await listForecastEventVerifications()).filter((item) => (
          item.source_task_id === currentResult.source_task_id
          && item.experiment_id === currentResult.experiment_id
          && item.model_id === currentResult.model_id
        )),
      );
    } catch (value) {
      onError(errorFrom(value));
    } finally {
      setBusy(false);
    }
  };
  const reopen = async (taskId: string) => {
    setBusy(true);
    try {
      setVerification((await loadForecastVerification(taskId)).result);
    } catch (value) {
      onError(errorFrom(value));
    } finally {
      setBusy(false);
    }
  };
  const reopenPostprocessing = async (taskId: string) => {
    setBusy(true);
    try {
      setPostprocessing((await loadForecastPostprocessing(taskId)).result);
    } catch (value) {
      onError(errorFrom(value));
    } finally {
      setBusy(false);
    }
  };
  const reopenEventVerification = async (taskId: string) => {
    setBusy(true);
    try {
      setEventVerification((await loadForecastEventVerification(taskId)).result);
    } catch (value) {
      onError(errorFrom(value));
    } finally {
      setBusy(false);
    }
  };
  const rankRows = verification?.aggregate.rank_counts.map((count, index) => ({
    rank: String(index),
    count,
  })) ?? [];

  return (
    <div className="modal-backdrop" role="presentation" onMouseDown={onClose}>
      <section className="forecast-dialog hindcast-dialog" role="dialog" aria-modal="true" aria-labelledby="hindcast-dialog-title" onMouseDown={(event) => event.stopPropagation()}>
        <div className="modal-heading">
          <div><p className="section-kicker">Repeated forecast origins</p><h2 id="hindcast-dialog-title">Ensemble hindcast verification</h2><p>Verify continuous forecasts, recalibrate mean and spread, or evaluate threshold-event decisions without changing the frozen model or source archives.</p></div>
          <button className="notice-dismiss" onClick={onClose} aria-label="Close hindcast verification">×</button>
        </div>
        <div className="forecast-dialog-scroll hindcast-dialog-scroll">
          {!verification && !postprocessing && !eventVerification ? (
            <div className="hindcast-selection-layout">
              <section className="forecast-section">
                <div className="forecast-section-heading"><div><strong>Compatible forecast cases</strong><span>All selected cases must share the source model, member count, quantiles and horizon.</span></div><span className="status-pill neutral">{selected.size} selected</span></div>
                <div className="hindcast-case-list">
                  {loading ? <div className="library-empty"><Activity size={26} /><strong>Loading archived ensembles…</strong></div> : forecasts.length < 2 ? <div className="library-empty"><History size={27} /><strong>At least two ensemble origins are required</strong><span>Run another observed meteorological ensemble with the same frozen model and a different origin.</span></div> : forecasts.map((item) => {
                    const checked = selected.has(item.task_id);
                    return <label className={`hindcast-case ${checked ? "selected" : ""}`} key={item.task_id}><input type="checkbox" checked={checked} onChange={() => toggle(item.task_id)} /><span><strong>{item.issue_id}</strong><small>{item.member_count} members · {item.horizon} leads · {item.scored_leads} observed</small></span><em>{new Date(item.saved_at).toLocaleString()}</em></label>;
                  })}
                </div>
              </section>
              <aside className="forecast-section hindcast-settings">
                <div className="forecast-section-heading"><div><strong>Verification settings</strong><span>Applied identically to every lead.</span></div></div>
                <Field label="Tie-breaking seed" hint="Reproducible randomized ranks"><input type="number" min="0" step="1" value={seed} onChange={(event) => setSeed(Math.max(0, Number(event.target.value)))} /></Field>
                <Field label="Minimum cases per lead" hint="Leads below this count are not reported"><input type="number" min="2" max="500" step="1" value={minimumCases} onChange={(event) => setMinimumCases(Math.max(2, Math.min(500, Number(event.target.value))))} /></Field>
                <div className="inline-note"><Info size={16} /><span>Twenty or more independent origins are preferable for interpreting rank and coverage diagnostics. Smaller samples remain clearly flagged.</span></div>
                <div className="forecast-section-heading"><div><strong>Chronological post-processing</strong><span>Requires at least seven selected origins.</span></div></div>
                <Field label="Oldest origins used for fitting" hint="At least two later origins remain untouched for evaluation"><input type="number" min="5" max={Math.max(5, selected.size - 2)} step="1" value={trainingCases} onChange={(event) => setTrainingCases(Math.max(5, Number(event.target.value)))} /></Field>
                <div className="inline-note"><Info size={16} /><span>The affine correction is fitted independently at each lead. Raw and adjusted CRPS remain visible together; negative CRPSS means the correction degraded the independent evaluation.</span></div>
                <div className="forecast-section-heading"><div><strong>Threshold-event decisions</strong><span>Uses the same chronological split and compares raw with adjusted probabilities.</span></div></div>
                <Field label="Event threshold source"><select value={eventThresholdMode} onChange={(event) => setEventThresholdMode(event.target.value as typeof eventThresholdMode)}><option value="training_quantile">Quantile of older observations</option><option value="absolute">Physical discharge threshold</option></select></Field>
                {eventThresholdMode === "training_quantile" ? <Field label="Older-archive quantile" hint="Estimated before the later origins are inspected"><input type="number" min="0.5" max="0.995" step="0.005" value={eventTrainingQuantile} onChange={(event) => setEventTrainingQuantile(Math.max(0.5, Math.min(0.995, Number(event.target.value))))} /></Field> : <Field label="Discharge threshold" hint="mm/day · event when Q is strictly greater"><input type="number" min="0" step="0.01" value={eventThresholdValue} onChange={(event) => setEventThresholdValue(Math.max(0, Number(event.target.value)))} /></Field>}
                <Field label="Reliability probability bins"><input type="number" min="3" max="20" step="1" value={eventProbabilityBins} onChange={(event) => setEventProbabilityBins(Math.max(3, Math.min(20, Number(event.target.value))))} /></Field>
                <Field label="Minimum later cases per lead"><input type="number" min="2" max={Math.max(2, selected.size - Math.min(Math.max(5, trainingCases), Math.max(5, selected.size - 2)))} step="1" value={eventMinimumCases} onChange={(event) => setEventMinimumCases(Math.max(2, Number(event.target.value)))} /></Field>
                <div className="inline-note"><Info size={16} /><span>Brier score evaluates probability accuracy, AUC evaluates discrimination, and economic value is conditional on the selected cost–loss ratio. These diagnostics answer different questions.</span></div>
                {archives.length > 0 && <div className="hindcast-archive-list"><strong>Saved analyses</strong>{archives.slice(0, 6).map((item) => <button type="button" disabled={busy} key={item.task_id} onClick={() => void reopen(item.task_id)}><span>{item.case_count} cases × {item.horizon} leads</span><small>Mean CRPS {formatDecimal(item.mean_crps, 4)}</small></button>)}</div>}
                {postprocessingArchives.length > 0 && <div className="hindcast-archive-list"><strong>Saved post-processing</strong>{postprocessingArchives.slice(0, 6).map((item) => <button type="button" disabled={busy} key={item.task_id} onClick={() => void reopenPostprocessing(item.task_id)}><span>{item.training_case_count} fit + {item.evaluation_case_count} evaluate</span><small>CRPSS {item.crps_skill_score === null ? "—" : formatDecimal(item.crps_skill_score, 3)}</small></button>)}</div>}
                {eventArchives.length > 0 && <div className="hindcast-archive-list"><strong>Saved event decisions</strong>{eventArchives.slice(0, 6).map((item) => <button type="button" disabled={busy} key={item.task_id} onClick={() => void reopenEventVerification(item.task_id)}><span>Q &gt; {formatDecimal(item.threshold_value, 3)} · {item.evaluation_case_count} evaluate</span><small>Brier skill vs raw {item.adjusted_brier_skill_vs_raw === null ? "—" : formatDecimal(item.adjusted_brier_skill_vs_raw, 3)}</small></button>)}</div>}
              </aside>
            </div>
          ) : eventVerification ? (
            <ForecastEventVerificationResults result={eventVerification} onBack={() => setEventVerification(null)} />
          ) : postprocessing ? (
            <ForecastPostprocessingResults result={postprocessing} onBack={() => setPostprocessing(null)} />
          ) : verification ? (
            <div className="hindcast-results">
              <div className="ensemble-results-heading"><div className="icon-tile cyan"><Gauge size={20} /></div><div><p className="section-kicker">Raw ensemble verification</p><h2>{verification.case_count} origins × {verification.horizon} leads</h2><p>Scores retain the issue–lead structure. Repeated valid dates are forecast events, not silently deduplicated observations.</p></div><button className="button button-secondary" onClick={() => setVerification(null)}><RotateCcw size={15} /> Change cases</button></div>
              <div className="forcing-score-grid">
                <article><span>Mean CRPS</span><strong>{formatDecimal(verification.aggregate.mean_crps, 4)}</strong><small>Lower is better</small></article>
                <article><span>Mean signed error</span><strong>{formatDecimal(verification.aggregate.mean_error_bias, 4)}</strong><small>Forecast mean − observed</small></article>
                <article><span>Spread–skill ratio</span><strong>{verification.aggregate.spread_skill_ratio === null ? "—" : formatDecimal(verification.aggregate.spread_skill_ratio, 3)}</strong><small>RMS member spread / RMSE</small></article>
                <article><span>Interval coverage</span><strong>{Math.round(verification.aggregate.interval_coverage * 100)}%</strong><small>{Math.round(verification.aggregate.nominal_interval_coverage * 100)}% nominal envelope</small></article>
              </div>
              <section className="forecast-section">
                <div className="forecast-section-heading"><div><strong>Diagnostics by lead</strong><span>Each row aggregates independent issue cases available at one forecast step.</span></div></div>
                <div className="forcing-lead-table-wrap"><table className="forcing-lead-table hindcast-lead-table"><thead><tr><th>Lead</th><th>Cases</th><th>CRPS</th><th>Bias</th><th>RMSE</th><th>RMS spread</th><th>SSR</th><th>Coverage</th></tr></thead><tbody>{verification.lead_time_diagnostics.map((lead) => <tr key={lead.lead_step}><td>+{lead.lead_step}</td><td>{lead.available_cases}</td>{lead.reported ? <><td>{formatDecimal(lead.mean_crps ?? Number.NaN, 4)}</td><td>{formatDecimal(lead.mean_error_bias ?? Number.NaN, 4)}</td><td>{formatDecimal(lead.root_mean_square_error ?? Number.NaN, 4)}</td><td>{formatDecimal(lead.root_mean_ensemble_variance ?? Number.NaN, 4)}</td><td>{lead.spread_skill_ratio === null || lead.spread_skill_ratio === undefined ? "—" : formatDecimal(lead.spread_skill_ratio, 3)}</td><td>{Math.round((lead.interval_coverage ?? 0) * 100)}%</td></> : <td colSpan={6}>Insufficient finite cases</td>}</tr>)}</tbody></table></div>
              </section>
              <section className="forecast-section hindcast-rank-card">
                <div className="forecast-section-heading"><div><strong>Aggregate rank histogram</strong><span>Ranks 0 and {verification.member_count} represent observations below or above all members. A reliable exchangeable ensemble tends toward a flat histogram over many independent cases.</span></div></div>
                <div className="hindcast-rank-chart"><ResponsiveContainer width="100%" height="100%"><BarChart data={rankRows} margin={{ top: 10, right: 14, left: 0, bottom: 2 }}><CartesianGrid stroke="var(--chart-grid)" vertical={false} /><XAxis dataKey="rank" /><YAxis allowDecimals={false} /><Tooltip /><Bar dataKey="count" name="Forecast events" fill="var(--accent)" radius={[4, 4, 0, 0]} isAnimationActive={false} /></BarChart></ResponsiveContainer></div>
              </section>
              <div className="shap-warnings">{verification.warnings.map((warning) => <div key={warning.code}><Info size={15} /><span>{warning.message}</span></div>)}</div>
            </div>
          ) : null}
        </div>
        {!verification && !postprocessing && !eventVerification && <div className="forecast-dialog-footer"><span>{selected.size < 2 ? "Select at least two compatible observed ensemble cases." : selected.size < 7 ? `${selected.size} origins ready for raw verification; seven are required for post-processing and event decisions.` : `${selected.size} origins ready for continuous and threshold-event verification.`}</span><div><button className="button button-secondary" onClick={onClose}>Cancel</button><button className="button button-secondary" disabled={selected.size < 7 || busy} onClick={() => void runEventVerification()}><CircleAlert size={16} /> {busy ? "Evaluating…" : "Evaluate event decisions"}</button><button className="button button-secondary" disabled={selected.size < 7 || busy} onClick={() => void runPostprocessing()}><Gauge size={16} /> {busy ? "Processing…" : "Post-process chronologically"}</button><button className="button button-primary" disabled={selected.size < 2 || busy} onClick={() => void runVerification()}><Gauge size={16} /> {busy ? "Verifying…" : "Verify raw ensembles"}</button></div></div>}
      </section>
    </div>
  );
}

type TemporalRange = { start: number; end: number };

function TemporalNavigator({
  overview,
  range,
  timestamps,
  onRange,
}: {
  overview: Array<{ index: number; time: string; observed: number | null; simulated: number | null }>;
  range: TemporalRange;
  timestamps?: string[] | null;
  onRange: (range: TemporalRange) => void;
}) {
  const total = timestamps?.length ?? (overview.at(-1)?.index ?? 0) + 1;
  const maximum = Math.max(0, total - 1);
  const fullRange = range.start === 0 && range.end === maximum;
  const startPosition = Math.max(0, overview.findIndex((row) => row.index >= range.start));
  let endPosition = startPosition;
  for (let position = startPosition; position < overview.length; position += 1) {
    if (overview[position].index <= range.end) endPosition = position;
  }
  const setStart = (value: number) => onRange({
    start: Math.max(0, Math.min(range.end, value)),
    end: range.end,
  });
  const setEnd = (value: number) => onRange({
    start: range.start,
    end: Math.max(range.start, Math.min(maximum, value)),
  });
  const handleBrush = (selection: { startIndex?: number; endIndex?: number }) => {
    if (selection.startIndex === undefined || selection.endIndex === undefined) return;
    const start = overview[selection.startIndex]?.index;
    const end = overview[selection.endIndex]?.index;
    if (start === undefined || end === undefined) return;
    onRange({ start: Math.min(start, end), end: Math.max(start, end) });
  };

  return (
    <div className="temporal-navigator" aria-label="Temporal plot zoom">
      <div className="temporal-navigator-heading">
        <div>
          <strong>Temporal viewport</strong>
          <span>Drag the overview handles, or enter exact row bounds. The same window is used by every result plot.</span>
        </div>
        <div className="temporal-range-fields">
          <label><span>From row</span><input aria-label="First visible result row" type="number" min={1} max={total} value={range.start + 1} onChange={(event) => setStart(Number(event.target.value) - 1)} /></label>
          <label><span>To row</span><input aria-label="Last visible result row" type="number" min={1} max={total} value={range.end + 1} onChange={(event) => setEnd(Number(event.target.value) - 1)} /></label>
          <button className="button button-secondary temporal-reset" disabled={fullRange} onClick={() => onRange({ start: 0, end: maximum })}><RotateCcw size={15} /> Reset zoom</button>
        </div>
      </div>
      <div className="temporal-window-labels"><span>{timestamps?.[range.start] ?? `Row ${range.start + 1}`}</span><strong>{(range.end - range.start + 1).toLocaleString()} intervals visible</strong><span>{timestamps?.[range.end] ?? `Row ${range.end + 1}`}</span></div>
      <div className="temporal-overview">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={overview} margin={{ top: 2, right: 8, left: 8, bottom: 0 }}>
            <XAxis dataKey="index" type="number" domain={[0, maximum]} hide />
            <YAxis hide domain={["auto", "auto"]} />
            <Line type="linear" dataKey="simulated" stroke="var(--chart-simulated)" strokeWidth={1.4} dot={false} isAnimationActive={false} />
            <Brush dataKey="index" height={34} travellerWidth={10} startIndex={startPosition} endIndex={Math.max(startPosition, endPosition)} onChange={handleBrush} stroke="var(--accent)" fill="var(--surface-muted)" />
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

export function ResultsPanel({
  result,
  busy,
  onNewRun,
  onRerun,
  onForecast,
  onHindcastVerification,
  onCompare,
  onTemporalValidation,
}: {
  result: RunResult;
  busy: boolean;
  onNewRun: () => void;
  onRerun: () => void;
  onForecast?: () => void;
  onHindcastVerification?: () => void;
  onCompare?: () => void;
  onTemporalValidation?: () => void;
}) {
  const [processExpanded, setProcessExpanded] = useState(false);
  const [decimalPlaces, setDecimalPlaces] = useState(4);
  const [explanation, setExplanation] = useState<RunExplanation | null>(null);
  const [explanationBusy, setExplanationBusy] = useState(false);
  const [explanationError, setExplanationError] = useState<StructuredError | null>(null);
  const [uncertainty, setUncertainty] = useState<RunUncertainty | null>(null);
  const [uncertaintyBusy, setUncertaintyBusy] = useState(false);
  const [uncertaintyError, setUncertaintyError] = useState<StructuredError | null>(null);
  const [scientificVerification, setScientificVerification] = useState<RunScientificVerification | null>(null);
  const [scientificVerificationBusy, setScientificVerificationBusy] = useState(false);
  const [scientificVerificationError, setScientificVerificationError] = useState<StructuredError | null>(null);
  const [selectedExplanationRow, setSelectedExplanationRow] = useState(0);
  const isForecast = result.run_kind === "forecast";
  const isTraining = !isForecast && (result.run_kind === "training" || Boolean(result.training));
  const discoveryModelId = result.training?.model_id ?? result.model_id;
  const isDiscoveryResult = discoveryModelId !== undefined
    && isDiscoveryModel(discoveryModelId);
  const initializationEnsemble = result.training?.initialization_ensemble ?? null;
  const forcingEnsemble = result.forecast_ensemble ?? null;
  const isExplainable = !isForecast && Boolean(result.training) && !isDiscoveryResult;
  const selectedEpochRecord = result.training?.history.find(
    (item) => item.epoch === result.training?.best_epoch,
  );
  const calibrationConvergence = useMemo(() => {
    const calibration = result.calibration;
    if (!calibration) return [];
    let best: number | null = null;
    const complete = calibration.records.flatMap((record) => {
      if (!record.succeeded || record.score === null || !Number.isFinite(record.score)) {
        return [];
      }
      if (
        best === null
        || (
          calibration.objective_direction === "maximize"
            ? record.score > best
            : record.score < best
        )
      ) {
        best = record.score;
      }
      return [{
        evaluation: record.index,
        bestScore: best,
        generation: record.generation ?? null,
      }];
    });
    const stride = Math.max(1, Math.ceil(complete.length / 600));
    return complete.filter(
      (_, index) => index % stride === 0 || index === complete.length - 1,
    );
  }, [result.calibration]);
  const discoveryLossEntries = selectedEpochRecord ? [
    ["Data loss", selectedEpochRecord.data_loss],
    ["Decorrelation loss", selectedEpochRecord.decorrelation_loss],
    ["Mass-balance loss", selectedEpochRecord.mass_balance_loss],
    ["Smoothness loss", selectedEpochRecord.smoothness_loss],
    ["Binary-relaxation loss", selectedEpochRecord.binary_loss],
    ["Sparsity loss", selectedEpochRecord.sparsity_loss],
  ].filter((entry): entry is [string, number] => typeof entry[1] === "number") : [];
  const hasProcesses = result.best_simulation.state_names.length > 0 || result.best_simulation.flux_names.length > 0;
  const partitionMetrics = result.partition_metrics
    ?? result.training?.partition_metrics
    ?? {};
  const partitionNames = Object.keys(partitionMetrics);
  const primaryPartition = result.validation.primary_evaluation_partition
    ?? (partitionNames.includes("test") ? "test" : partitionNames[0]);
  const [selectedPartition, setSelectedPartition] = useState(
    primaryPartition ?? "test",
  );
  const availableMetrics = useMemo(() => {
    if (partitionMetrics[selectedPartition]?.length) {
      return partitionMetrics[selectedPartition];
    }
    if (result.evaluation_metrics?.length) return result.evaluation_metrics;
    if (isForecast || !result.calibration) return [];
    const option = objectiveOption(
      result.calibration.objective_name as WorkspaceForm["objective"],
    );
    return [{
      metric_id: result.calibration.objective_name,
      symbol: option.symbol,
      name: option.name,
      value: result.calibration.best_score,
      direction: result.calibration.objective_direction,
      ideal_value: result.calibration.objective_direction === "maximize" ? 1 : 0,
      unit: option.id === "mean_square_error"
        ? "(mm/day)^2"
        : option.id === "root_mean_square_error" || option.id === "mean_absolute_error"
          ? "mm/day"
          : "dimensionless",
      definition: "Stored calibration objective from a legacy result archive.",
      unavailable_reason: null,
    }];
  }, [isForecast, partitionMetrics, result, selectedPartition]);
  const [selectedMetricIds, setSelectedMetricIds] = useState<Set<string>>(
    () => new Set(availableMetrics.map((metric) => metric.metric_id)),
  );
  const [metricMenuOpen, setMetricMenuOpen] = useState(false);
  const metricPickerRef = useRef<HTMLDivElement>(null);
  const routingStatePositions = useMemo(
    () => result.best_simulation.state_names
      .map((name, position) => name.startsWith("routing_pending_day_") ? position : -1)
      .filter((position) => position >= 0),
    [result],
  );
  const displayStateNames = useMemo(
    () => [
      ...result.best_simulation.state_names.filter(
        (name) => !name.startsWith("routing_pending_day_"),
      ),
      ...(routingStatePositions.length ? ["routing_storage"] : []),
    ],
    [result, routingStatePositions],
  );
  const discoveryAnalysisVariables = useMemo(
    () => Object.entries(result.best_simulation.diagnostics)
      .filter(([name, values]) => (
        name.startsWith("analysis__")
        && Array.isArray(values)
        && values.length === result.best_simulation.outputs.length
      ))
      .map(([name, values]) => ({
        key: `analysis:${name}`,
        name: name.replace("analysis__", ""),
        values: values as number[],
        position: -1,
        kind: "analysis" as const,
        group: "Discovery analysis",
      })),
    [result],
  );
  const trajectoryVariables = useMemo(() => [
    ...result.best_simulation.output_names.map((name, position) => ({
      key: `output:${name}`,
      name,
      position,
      kind: "output" as const,
      group: "Model outputs",
    })),
    ...result.best_simulation.flux_names.map((name, position) => ({
      key: `flux:${name}`,
      name,
      position,
      kind: "flux" as const,
      group: "Intermediate fluxes",
    })),
    ...result.best_simulation.state_names
      .map((name, position) => ({ name, position }))
      .filter(({ name }) => !name.startsWith("routing_pending_day_"))
      .map(({ name, position }) => ({
        key: `state:${name}`,
        name,
        position,
        kind: "state" as const,
        group: "Internal states",
      })),
    ...(routingStatePositions.length ? [{
      key: "state:routing_storage",
      name: "routing_storage",
      position: -1,
      kind: "routing" as const,
      group: "Internal states",
    }] : []),
    ...discoveryAnalysisVariables,
  ], [discoveryAnalysisVariables, result, routingStatePositions]);
  const [selectedVariableKey, setSelectedVariableKey] = useState(
    () => trajectoryVariables[0]?.key ?? "output:streamflow",
  );
  const selectedVariable = trajectoryVariables.find(
    (variable) => variable.key === selectedVariableKey,
  ) ?? trajectoryVariables[0];
  const physicalOutputPosition = result.best_simulation.output_names.indexOf(
    "physical_streamflow",
  );
  const compareCorrectedAndPhysical = selectedVariable?.kind === "output"
    && selectedVariable.name === "streamflow"
    && physicalOutputPosition >= 0;
  const hasObservedSeries = selectedVariable?.kind === "output"
    && (selectedVariable.name === "streamflow" || selectedVariable.name === "physical_streamflow")
    && result.observations !== null
    && result.observations.some((value) => value !== null && Number.isFinite(value));
  const ensembleLower = Array.isArray(
    result.best_simulation.diagnostics.initialization_ensemble_lower,
  )
    ? result.best_simulation.diagnostics.initialization_ensemble_lower as number[]
    : null;
  const ensembleUpper = Array.isArray(
    result.best_simulation.diagnostics.initialization_ensemble_upper,
  )
    ? result.best_simulation.diagnostics.initialization_ensemble_upper as number[]
    : null;
  const forcingEnsembleLower = Array.isArray(
    result.best_simulation.diagnostics.forcing_ensemble_lower,
  )
    ? result.best_simulation.diagnostics.forcing_ensemble_lower as number[]
    : null;
  const forcingEnsembleUpper = Array.isArray(
    result.best_simulation.diagnostics.forcing_ensemble_upper,
  )
    ? result.best_simulation.diagnostics.forcing_ensemble_upper as number[]
    : null;
  const showInitializationBand = selectedVariable?.kind === "output"
    && selectedVariable.name === "streamflow"
    && ensembleLower?.length === result.best_simulation.outputs.length
    && ensembleUpper?.length === result.best_simulation.outputs.length;
  const showForcingBand = selectedVariable?.kind === "output"
    && selectedVariable.name === "streamflow"
    && forcingEnsembleLower?.length === result.best_simulation.outputs.length
    && forcingEnsembleUpper?.length === result.best_simulation.outputs.length;
  const showEnsembleBand = showForcingBand || showInitializationBand;
  const displayedEnsembleLower = showForcingBand
    ? forcingEnsembleLower
    : ensembleLower;
  const displayedEnsembleUpper = showForcingBand
    ? forcingEnsembleUpper
    : ensembleUpper;
  const nIntervals = result.best_simulation.outputs.length;
  const trajectory = useMemo(() => {
    const output = result.best_simulation.outputs;
    const states = result.best_simulation.states;
    const fluxes = result.best_simulation.fluxes;
    return Array.from({ length: nIntervals }, (_, index) => {
      let simulated: number | null = null;
      if (selectedVariable?.kind === "output") simulated = output[index]?.[selectedVariable.position] ?? null;
      if (selectedVariable?.kind === "flux") simulated = fluxes[index]?.[selectedVariable.position] ?? null;
      if (selectedVariable?.kind === "state") simulated = states[index + 1]?.[selectedVariable.position] ?? null;
      if (selectedVariable?.kind === "analysis") simulated = selectedVariable.values[index] ?? null;
      if (selectedVariable?.kind === "routing") {
        simulated = routingStatePositions.reduce(
          (total, position) => total + Number(states[index + 1]?.[position] ?? 0),
          0,
        );
      }
      const observed = hasObservedSeries ? result.observations?.[index] : null;
      const lower = showEnsembleBand ? displayedEnsembleLower?.[index] : null;
      const upper = showEnsembleBand ? displayedEnsembleUpper?.[index] : null;
      const row: { index: number; time: string; observed: number | null; simulated: number | null; physical: number | null; ensembleRange: [number, number] | null; [key: string]: string | number | null | [number, number] } = {
        index,
        time: result.timestamps?.[index] ?? String(index + 1),
        observed: observed !== null && observed !== undefined && Number.isFinite(observed) ? observed : null,
        simulated,
        physical: compareCorrectedAndPhysical
          ? output[index]?.[physicalOutputPosition] ?? null
          : null,
        ensembleRange: (
          lower !== null
          && lower !== undefined
          && upper !== null
          && upper !== undefined
          && Number.isFinite(lower)
          && Number.isFinite(upper)
            ? [lower, upper]
            : null
        ),
      };
      result.best_simulation.state_names.forEach((name, position) => {
        if (!name.startsWith("routing_pending_day_")) {
          row[name] = states[index + 1]?.[position] ?? null;
        }
      });
      if (routingStatePositions.length) {
        row.routing_storage = routingStatePositions.reduce(
          (total, position) => total + Number(states[index + 1]?.[position] ?? 0),
          0,
        );
      }
      return row;
    });
  }, [compareCorrectedAndPhysical, displayedEnsembleLower, displayedEnsembleUpper, hasObservedSeries, nIntervals, physicalOutputPosition, result, routingStatePositions, selectedVariable, showEnsembleBand]);
  const [visibleRange, setVisibleRange] = useState<TemporalRange>(() => ({
    start: 0,
    end: Math.max(0, nIntervals - 1),
  }));
  const visibleTrajectory = useMemo(
    () => trajectory.slice(visibleRange.start, visibleRange.end + 1),
    [trajectory, visibleRange],
  );
  const outputTrajectory = useMemo(
    () => downsampleForChart(
      visibleTrajectory,
      compareCorrectedAndPhysical
        ? ["observed", "physical", "simulated"]
        : ["observed", "simulated"],
      900,
    ),
    [compareCorrectedAndPhysical, visibleTrajectory],
  );
  const overviewTrajectory = useMemo(
    () => downsampleForChart(
      trajectory,
      compareCorrectedAndPhysical
        ? ["observed", "physical", "simulated"]
        : ["observed", "simulated"],
      320,
    ),
    [compareCorrectedAndPhysical, trajectory],
  );
  const stateTrajectory = useMemo(
    () => processExpanded
      ? downsampleForChart(visibleTrajectory, displayStateNames, 900)
      : [],
    [displayStateNames, processExpanded, visibleTrajectory],
  );
  useEffect(() => {
    setProcessExpanded(false);
    setExplanation(null);
    setExplanationBusy(false);
    setExplanationError(null);
    setUncertainty(null);
    setUncertaintyBusy(false);
    setUncertaintyError(null);
    setSelectedExplanationRow(0);
    setSelectedPartition(primaryPartition ?? "test");
    setVisibleRange({ start: 0, end: Math.max(0, nIntervals - 1) });
    setSelectedMetricIds(new Set(availableMetrics.map((metric) => metric.metric_id)));
    setMetricMenuOpen(false);
    setSelectedVariableKey(trajectoryVariables[0]?.key ?? "output:streamflow");
  }, [availableMetrics, nIntervals, primaryPartition, result.task_id, trajectoryVariables]);
  useEffect(() => {
    if (!metricMenuOpen) return;
    const closeOnPointerDown = (event: PointerEvent) => {
      if (!metricPickerRef.current?.contains(event.target as Node)) {
        setMetricMenuOpen(false);
      }
    };
    const closeOnKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMetricMenuOpen(false);
    };
    document.addEventListener("pointerdown", closeOnPointerDown);
    document.addEventListener("keydown", closeOnKeyDown);
    return () => {
      document.removeEventListener("pointerdown", closeOnPointerDown);
      document.removeEventListener("keydown", closeOnKeyDown);
    };
  }, [metricMenuOpen]);
  const objective = result.training?.objective?.objective_id
    ?? result.calibration?.objective_name;
  const displayedMetrics = availableMetrics.filter(
    (metric) => selectedMetricIds.has(metric.metric_id),
  );
  const toggleMetric = (metricId: string) => {
    setSelectedMetricIds((current) => {
      const next = new Set(current);
      if (next.has(metricId)) {
        if (next.size === 1) return current;
        next.delete(metricId);
      } else {
        next.add(metricId);
      }
      return next;
    });
  };
  const rawMassBalance = result.best_simulation.diagnostics.maximum_absolute_mass_balance_error;
  const massBalance = typeof rawMassBalance === "number" ? rawMassBalance : null;
  const stateColors = ["var(--snow)", "var(--accent)", "var(--purple)", "var(--warning)", "var(--accent-cyan)"];
  const selectedUnit = selectedVariable?.kind === "analysis"
    ? selectedVariable.name === "balance_residual" ? "mm / interval" : "relative value"
    : selectedVariable?.kind === "state" || selectedVariable?.kind === "routing"
    ? "mm"
    : "mm / interval";
  const tooltipFormatter = (value: unknown) => (
    Array.isArray(value) && value.length === 2
      ? `${formatDecimal(Number(value[0]), decimalPlaces)} – ${formatDecimal(Number(value[1]), decimalPlaces)}`
      : formatDecimal(Number(value), decimalPlaces)
  );
  const selectedLocalExplanation = explanation?.local[
    Math.min(selectedExplanationRow, Math.max(0, explanation.local.length - 1))
  ];
  const maximumFeatureImportance = Math.max(
    0,
    ...(explanation?.global_by_feature.map((item) => item.mean_absolute_shap) ?? []),
  );
  const maximumLagImportance = Math.max(
    0,
    ...(explanation?.global_by_lag.map((item) => item.mean_absolute_shap) ?? []),
  );
  const displayedLagImportance = explanation
    ? [...explanation.global_by_lag]
      .sort((left, right) => right.mean_absolute_shap - left.mean_absolute_shap)
      .slice(0, 24)
      .sort((left, right) => left.lag_steps - right.lag_steps)
    : [];
  const maximumLocalAttribution = Math.max(
    0,
    ...(selectedLocalExplanation?.attributions.map(
      (item) => Math.abs(item.shap_value),
    ) ?? []),
  );
  const computeExplanation = async () => {
    setExplanationBusy(true);
    setExplanationError(null);
    try {
      const next = await explainSavedRun(result.task_id);
      setExplanation(next);
      setSelectedExplanationRow(0);
    } catch (value) {
      setExplanationError(errorFrom(value));
    } finally {
      setExplanationBusy(false);
    }
  };
  const computeUncertainty = async () => {
    setUncertaintyBusy(true);
    setUncertaintyError(null);
    try {
      setUncertainty(await quantifyRunUncertainty([result.task_id]));
    } catch (value) {
      setUncertaintyError(errorFrom(value));
    } finally {
      setUncertaintyBusy(false);
    }
  };
  const computeScientificVerification = async () => {
    setScientificVerificationBusy(true);
    setScientificVerificationError(null);
    try {
      setScientificVerification(await verifySavedRunScientifically(result.task_id));
    } catch (value) {
      setScientificVerificationError(errorFrom(value));
    } finally {
      setScientificVerificationBusy(false);
    }
  };

  return (
    <>
      <PageHeader eyebrow={isForecast ? "Completed forecast" : isTraining ? "Completed training" : "Completed experiment"} title={forcingEnsemble ? "Meteorological ensemble forecast complete" : isForecast ? "Frozen-model forecast complete" : isTraining ? "Validation-selected checkpoint available" : "Simulation complete"} description={isForecast ? `New forcings were processed without changing the saved ${result.training ? "weights or transformations" : "calibrated parameters"}.` : isTraining ? "Test metrics, the complete prediction trajectory, training history, transformations, and learned weights are archived together." : "Outputs, internal states, calibrated parameters, and numerical diagnostics remain synchronized."} actions={<><button className="button button-secondary" onClick={onNewRun}><RotateCcw size={16} /> Adjust experiment</button>{!isForecast && <button className="button button-secondary" disabled={busy} onClick={onRerun}><Play size={16} /> {busy ? "Preparing…" : "Run unchanged"}</button>}{!isForecast && <button className="button button-secondary" disabled={busy} onClick={onCompare}><BarChart3 size={16} /> Compare runs</button>}{!isForecast && <button className="button button-secondary" disabled={busy} onClick={onTemporalValidation}><History size={16} /> Temporal validation</button>}{forcingEnsemble && <button className="button button-secondary" disabled={busy} onClick={onHindcastVerification}><Gauge size={16} /> Verify hindcasts</button>}<button className="button button-primary" disabled={busy} onClick={onForecast}><CloudSun size={16} /> {forcingEnsemble ? "Continue ensemble" : isForecast ? "New forecast" : "Use new forcings"}</button></>} />
      <section className="result-metrics" aria-labelledby="result-metrics-title">
        <div className="metric-selector-heading">
          <div><p className="section-kicker">{selectedPartition === "test" ? "Independent evaluation" : "Partition diagnostics"}</p><h2 id="result-metrics-title">Performance metrics</h2><p>{isForecast ? "Metrics use the finite streamflow observations supplied with these new forcings." : `Choose what to display. Every value below is recomputed on the ${selectedPartition} partition only.`}</p></div>
          <div className="result-chart-controls">
            {partitionNames.length > 1 && <label><span>Evaluation partition</span><select aria-label="Displayed evaluation partition" value={selectedPartition} onChange={(event) => setSelectedPartition(event.target.value)}>{partitionNames.map((name) => <option key={name} value={name}>{name[0].toUpperCase() + name.slice(1)} · {result.validation.partition_scored_observations?.[name] ?? "—"} observations</option>)}</select></label>}
            {availableMetrics.length > 0 && <span className="metric-count">{displayedMetrics.length} of {availableMetrics.length} shown</span>}
          </div>
        </div>
        {availableMetrics.length > 0 ? <><div className="metric-picker" ref={metricPickerRef}>
          <button type="button" className="metric-picker-trigger" aria-haspopup="true" aria-expanded={metricMenuOpen} aria-controls="displayed-metrics-menu" onClick={() => setMetricMenuOpen((current) => !current)}>
            <span><small>Displayed result metrics</small><strong>{displayedMetrics.map((metric) => metric.symbol).join(", ")}</strong></span>
            <span className="metric-picker-summary">{displayedMetrics.length} selected <ChevronDown className={metricMenuOpen ? "rotate-up" : ""} size={16} /></span>
          </button>
          <div id="displayed-metrics-menu" className={`metric-picker-menu${metricMenuOpen ? " open" : ""}`} hidden={!metricMenuOpen}>
            <div className="metric-picker-menu-heading"><div><strong>Choose metrics</strong><span>Values remain calculated in the archive when hidden.</span></div><button type="button" onClick={() => setSelectedMetricIds(new Set(availableMetrics.map((metric) => metric.metric_id)))}>Reset</button></div>
            <div className="metric-picker-options" role="group" aria-label="Displayed result metrics">
              {availableMetrics.map((metric) => {
                const selected = selectedMetricIds.has(metric.metric_id);
                const isObjective = !isForecast && metric.metric_id === objective;
                return <label key={metric.metric_id} className={`metric-picker-option${selected ? " selected" : ""}`} title={metric.definition}><input type="checkbox" checked={selected} onChange={() => toggleMetric(metric.metric_id)} /><span className="metric-picker-check">{selected && <Check size={13} />}</span><strong>{metric.symbol}</strong><span>{metric.name}</span>{isObjective && <small>Objective</small>}</label>;
              })}
            </div>
            <div className="metric-picker-footer"><button type="button" onClick={() => setSelectedMetricIds(new Set([objective && availableMetrics.some((metric) => metric.metric_id === objective) ? objective : availableMetrics[0].metric_id]))}>{objective ? "Objective only" : "First metric only"}</button><button type="button" className="button button-primary" onClick={() => setMetricMenuOpen(false)}>Done</button></div>
          </div>
        </div>
        <div className="metric-grid performance-metric-grid">
          {displayedMetrics.map((metric) => {
            const isObjective = !isForecast && metric.metric_id === objective;
            return <article key={metric.metric_id} className={`metric-card${isObjective ? " featured" : ""}`} title={metric.definition}><div className="metric-card-title"><span>{metric.symbol}</span>{isObjective && <small>{isTraining ? "Training objective" : "Calibration objective"}</small>}</div><strong>{formatMetricValue(metric.value, metric.unit, decimalPlaces)}</strong><small>{metric.unavailable_reason ?? metricGuidance(metric.direction, metric.ideal_value)}</small><p>{metric.name}{metric.unit !== "dimensionless" && metric.unit !== "%" ? ` · ${metric.unit}` : ""}</p></article>;
          })}
        </div></> : <div className="empty-metrics"><Info size={18} /><div><strong>No observed streamflow was supplied</strong><span>The forecast trajectory and internal processes remain available, but performance metrics require observations.</span></div></div>}
      </section>
      {!isForecast && result.calibration && calibrationConvergence.length > 1 && <section className="card calibration-convergence-card" aria-labelledby="calibration-convergence-title">
        <div className="calibration-convergence-heading">
          <div><p className="section-kicker">{result.training ? "Frozen physical stage" : "Parameter search"}</p><h2 id="calibration-convergence-title">Calibration convergence</h2><p>The line is the best objective value found up to each evaluation. A flat segment shows no improvement, not proof that the global optimum has been reached.</p></div>
          <div className="calibration-convergence-audit">
            <span>{result.calibration.algorithm_id === "differential_evolution.rand1bin.v1" ? "DE/rand/1/bin" : "Uniform random baseline"}</span>
            <strong>{result.calibration.records.length.toLocaleString()} evaluations</strong>
            {Boolean(result.calibration.generations_completed) && <small>{result.calibration.generations_completed} generations completed</small>}
          </div>
        </div>
        <div className="calibration-convergence-chart">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={calibrationConvergence} margin={{ top: 8, right: 16, left: 4, bottom: 4 }}>
              <CartesianGrid stroke="var(--chart-grid)" vertical={false} />
              <XAxis dataKey="evaluation" type="number" domain={["dataMin", "dataMax"]} tick={{ fontSize: 11 }} />
              <YAxis domain={["auto", "auto"]} width={72} tick={{ fontSize: 11 }} tickFormatter={(value) => formatDecimal(Number(value), Math.min(decimalPlaces, 4))} />
              <Tooltip formatter={(value) => [formatDecimal(Number(value), decimalPlaces), "Best score"]} labelFormatter={(value) => `Evaluation ${value}`} />
              <Line type="stepAfter" dataKey="bestScore" name="Best score" stroke="var(--accent)" strokeWidth={2.2} dot={false} isAnimationActive={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </section>}
      {!isForecast && <section className="card scientific-verification-card" aria-labelledby="scientific-verification-title">
        <div className="scientific-verification-heading">
          <div className="scientific-verification-heading-main">
            <div className="icon-tile green"><ShieldCheck size={20} /></div>
            <div><p className="section-kicker">Numerical checks</p><h2 id="scientific-verification-title">Scientific verification</h2><p>Replay the frozen archive, interrupt and restart it, perturb only future forcings, and inspect its physical numerical checks.</p></div>
          </div>
          <button className="button button-secondary" disabled={scientificVerificationBusy} onClick={computeScientificVerification}><ShieldCheck size={16} /> {scientificVerificationBusy ? "Verifying…" : scientificVerification ? "Run verification again" : "Verify this run"}</button>
        </div>
        {!scientificVerification && !scientificVerificationError && <div className="uncertainty-method-note"><Info size={17} /><div><strong>Integrity is not predictive skill</strong><span>Passing these checks shows that the archived computation is internally reproducible and causal. It does not establish that the model represents the catchment accurately.</span></div></div>}
        {scientificVerificationError && <div className="shap-error" role="alert"><CircleAlert size={18} /><div><strong>{scientificVerificationError.summary ?? "Scientific verification could not be completed"}</strong><span>{scientificVerificationError.explanation ?? scientificVerificationError.message ?? "Review the archived model and forcings, then retry."}</span></div></div>}
        {scientificVerification && <>
          <div className={`scientific-verification-summary ${scientificVerification.status}`}>
            {scientificVerification.status === "passed" ? <CircleCheck size={20} /> : <CircleAlert size={20} />}
            <div><strong>{scientificVerification.status === "passed" ? "All applicable checks passed" : `${scientificVerification.summary.failed} check${scientificVerification.summary.failed === 1 ? "" : "s"} failed`}</strong><span>{scientificVerification.checked_intervals.toLocaleString()} forcing intervals · {scientificVerification.summary.passed} passed · {scientificVerification.summary.not_applicable} not applicable</span></div>
          </div>
          <div className="scientific-check-grid">
            {scientificVerification.checks.map((check) => <article className={check.status} key={check.check_id}>
              <div>{check.status === "passed" ? <CircleCheck size={16} /> : check.status === "failed" ? <CircleAlert size={16} /> : <Info size={16} />}<strong>{check.name}</strong><span>{check.status.replace("_", " ")}</span></div>
              <p>{check.description}</p>
              {check.maximum_absolute_difference !== null && <small>Maximum difference: {check.maximum_absolute_difference.toExponential(2)}{check.tolerance !== null ? ` · tolerance ${check.tolerance.toExponential(1)}` : ""}</small>}
            </article>)}
          </div>
          <div className="scientific-source-info">
            <div><span>Implementation relationship</span><strong>{scientificVerification.implementation_source.relationship.replaceAll("_", " ")}</strong><p>{scientificVerification.implementation_source.statement}</p></div>
            <div><span>Historical source files</span><strong>{scientificVerification.implementation_source.historical_sources.length ? scientificVerification.implementation_source.historical_sources.join(" · ") : "HydroSoftware-native engine"}</strong><p>Numerical equivalence: {scientificVerification.implementation_source.equivalence_claim.replaceAll("_", " ")}</p></div>
          </div>
          <div className="uncertainty-warnings">{scientificVerification.limitations.map((limitation) => <div key={limitation}><Info size={15} /><span>{limitation}</span></div>)}</div>
        </>}
      </section>}
      {!isForecast && availableMetrics.length > 0 && <section className="card uncertainty-card" aria-labelledby="uncertainty-title">
        <div className="uncertainty-heading">
          <div className="uncertainty-heading-main">
            <div className="icon-tile cyan"><Gauge size={20} /></div>
            <div><p className="section-kicker">Evaluation robustness</p><h2 id="uncertainty-title">Sampling uncertainty</h2><p>Estimate how much each score changes when contiguous portions of the finite evaluation record are resampled.</p></div>
          </div>
          <button className="button button-secondary" disabled={uncertaintyBusy} onClick={computeUncertainty}><Gauge size={16} /> {uncertaintyBusy ? "Resampling…" : uncertainty ? "Recompute intervals" : "Compute intervals"}</button>
        </div>
        {!uncertainty && !uncertaintyError && <div className="uncertainty-method-note"><Info size={17} /><div><strong>Temporal blocks, not independent days</strong><span>The default uses 1,000 reproducible moving-block replicates and selects a block length from flow and residual autocorrelation.</span></div></div>}
        {uncertaintyError && <div className="shap-error" role="alert"><CircleAlert size={18} /><div><strong>{uncertaintyError.summary ?? "Sampling uncertainty could not be computed"}</strong><span>{uncertaintyError.explanation ?? uncertaintyError.message ?? "Review the finite evaluation support and retry."}</span></div></div>}
        {uncertainty && <><div className="uncertainty-audit-grid">
          <div><span>Scored observations</span><strong>{uncertainty.scored_observations.toLocaleString()}</strong></div>
          <div><span>Temporal block</span><strong>{uncertainty.block_length} day{uncertainty.block_length === 1 ? "" : "s"}</strong></div>
          <div><span>Replicates</span><strong>{uncertainty.bootstrap_replicates.toLocaleString()}</strong></div>
          <div><span>Interval</span><strong>{Math.round(uncertainty.confidence_level * 100)}% percentile</strong></div>
        </div>
        <div className="uncertainty-interval-grid">
          {uncertainty.runs[0].metrics.map((metric) => <article key={`interval-${metric.metric_id}`}>
            <div><span>{metric.symbol}</span><strong>{formatMetricValue(metric.value, metric.unit, decimalPlaces)}</strong></div>
            <p>{formatMetricInterval(metric.lower, metric.upper, metric.unit, decimalPlaces)}</p>
            <small>{metric.interval_unavailable_reason ?? `${metric.valid_replicates.toLocaleString()} finite bootstrap replicates`}</small>
          </article>)}
        </div>
        <div className="uncertainty-warnings">{uncertainty.warnings.map((warning) => <div key={warning.code}><Info size={15} /><span>{warning.message}</span></div>)}</div></>}
      </section>}
      <div className="metric-grid result-diagnostic-grid">
        <article className="metric-card"><span>{initializationEnsemble ? "Initialization ensemble" : result.training ? (isForecast ? "Frozen checkpoint" : "Best checkpoint") : isForecast ? "Parameters" : "Evaluations"}</span><strong>{initializationEnsemble ? `${initializationEnsemble.member_count_completed} members` : result.training ? `Epoch ${result.training.best_epoch}` : isForecast ? Object.keys(result.calibration?.best_parameters ?? {}).length : result.calibration?.records.length ?? 0}</strong><small>{initializationEnsemble ? `${result.training?.epochs_completed ?? 0} member-epochs · arithmetic mean` : result.training ? `${result.training.epochs_completed} epochs completed` : isForecast ? "Frozen from source calibration" : `${result.calibration?.records.filter((record) => record.succeeded).length ?? 0} successful`}</small></article>
        <article className="metric-card"><span>Scored observations</span><strong>{result.validation.scored_observations}</strong><small>{result.validation.warmup_steps} warm-up steps</small></article>
        <article className="metric-card"><span>{massBalance === null ? "Execution engine" : "Maximum balance error"}</span><strong>{massBalance === null ? String(result.best_simulation.diagnostics.engine_id ?? "—") : massBalance.toExponential(decimalPlaces)}</strong><small>{massBalance === null ? "Persisted with the result" : "mm per interval"}</small></article>
      </div>

      {initializationEnsemble && <section className="card ensemble-results-card" aria-labelledby="ensemble-results-title">
        <div className="ensemble-results-heading">
          <div className="icon-tile purple"><Boxes size={20} /></div>
          <div><p className="section-kicker">Initialization sensitivity</p><h2 id="ensemble-results-title">Neural ensemble diagnostics</h2><p>All members share the same data, architecture and fitting protocol. Only their derived initialization seed differs.</p></div>
          <span className="status-pill neutral">10–90% member envelope</span>
        </div>
        <div className="ensemble-member-grid">
          {initializationEnsemble.members.map((member) => <article key={member.member}>
            <div><span>Member {String(member.member).padStart(2, "0")}</span><small>seed {member.seed}</small></div>
            <strong>{(member.best_validation_score ?? member.best_validation_rmse) === null ? "—" : formatDecimal(Number(member.best_validation_score ?? member.best_validation_rmse), decimalPlaces)}</strong>
            <small>validation {result.training?.objective?.symbol ?? "RMSE"} · epoch {member.best_epoch}</small>
          </article>)}
        </div>
        <div className="ensemble-scope-note"><Info size={16} /><span>This spread measures sensitivity to neural weight initialization. It is not a calibrated probability interval and does not include forcing, observation, parameter or structural uncertainty.{initializationEnsemble.physical_calibration_reused ? " The PaML physical calibration is identical for every corrector member." : ""}</span></div>
      </section>}

      {forcingEnsemble && <section className="card ensemble-results-card forcing-ensemble-card" aria-labelledby="forcing-ensemble-title">
        <div className="ensemble-results-heading">
          <div className="icon-tile cyan"><CloudRain size={20} /></div>
          <div><p className="section-kicker">Meteorological scenarios · cycle {forcingEnsemble.cycle_index ?? 1}</p><h2 id="forcing-ensemble-title">Forcing-ensemble diagnostics</h2><p>{(forcingEnsemble.cycle_index ?? 1) > 1 ? "Each member resumes from its matching archived state and context." : "The frozen source model starts every member from the same hydrological state or antecedent context."} The plotted trajectory is the arithmetic member mean.</p></div>
          <span className="status-pill neutral">{Math.round(forcingEnsemble.lower_quantile * 100)}–{Math.round(forcingEnsemble.upper_quantile * 100)}% envelope</span>
        </div>
        <div className="forcing-score-grid">
          <article><span>Members</span><strong>{forcingEnsemble.member_count}</strong><small>aligned forcing scenarios</small></article>
          <article><span>Mean CRPS</span><strong>{forcingEnsemble.aggregate_scores.mean_crps === null ? "—" : formatDecimal(forcingEnsemble.aggregate_scores.mean_crps, decimalPlaces)}</strong><small>mm/day · lower is better</small></article>
          <article><span>Mean signed error</span><strong>{forcingEnsemble.aggregate_scores.mean_error_bias === null ? "—" : formatDecimal(forcingEnsemble.aggregate_scores.mean_error_bias, decimalPlaces)}</strong><small>forecast mean − observed · mm/day</small></article>
          <article><span>Mean spread</span><strong>{forcingEnsemble.aggregate_scores.mean_ensemble_spread === null ? "—" : formatDecimal(forcingEnsemble.aggregate_scores.mean_ensemble_spread, decimalPlaces)}</strong><small>population standard deviation · mm/day</small></article>
        </div>
        {forcingEnsemble.aggregate_scores.scored_leads > 0 && <div className="forcing-lead-table-wrap"><table className="forcing-lead-table">
          <thead><tr><th>Lead</th><th>Time</th><th>Observed</th><th>Mean</th><th>Spread</th><th>Signed error</th><th>CRPS</th></tr></thead>
          <tbody>{forcingEnsemble.lead_time_diagnostics.map((lead) => <tr key={lead.lead_step}><td>+{lead.lead_step}</td><td>{lead.timestamp ?? `Row ${lead.lead_step}`}</td><td>{lead.observation === null ? "—" : formatDecimal(lead.observation, decimalPlaces)}</td><td>{lead.ensemble_mean === null ? "—" : formatDecimal(lead.ensemble_mean, decimalPlaces)}</td><td>{lead.spread === null ? "—" : formatDecimal(lead.spread, decimalPlaces)}</td><td>{lead.signed_error === null ? "—" : formatDecimal(lead.signed_error, decimalPlaces)}</td><td>{lead.crps === null ? "—" : formatDecimal(lead.crps, decimalPlaces)}</td></tr>)}</tbody>
        </table></div>}
        <div className="ensemble-scope-note"><Info size={16} /><span>The envelope and CRPS reflect only the supplied meteorological members. They do not include parameter, structural or observation uncertainty. Internal states and fluxes shown below are member means, not a single physical scenario. Continuation uses the separately archived state and context of every matching member, never this displayed mean.</span></div>
      </section>}

      <section className="card chart-card">
        <div className="chart-heading result-chart-heading"><div><p className="section-kicker">Temporal trajectory</p><h2>{compareCorrectedAndPhysical ? "Observed, physical, and corrected streamflow" : showEnsembleBand ? "Observed streamflow and ensemble mean" : hasObservedSeries ? `Observed vs simulated ${variableLabel(selectedVariable?.name ?? "streamflow").toLowerCase()}` : variableLabel(selectedVariable?.name ?? "streamflow")}</h2></div><div className="result-chart-controls"><label><span>Variable</span><select aria-label="Displayed temporal variable" value={selectedVariableKey} onChange={(event) => setSelectedVariableKey(event.target.value)}>{["Model outputs", "Intermediate fluxes", "Internal states", "Discovery analysis"].map((group) => trajectoryVariables.some((variable) => variable.group === group) ? <optgroup key={group} label={group}>{trajectoryVariables.filter((variable) => variable.group === group).map((variable) => <option key={variable.key} value={variable.key}>{variableLabel(variable.name)}</option>)}</optgroup> : null)}</select></label><label><span>Decimal places</span><select aria-label="Displayed decimal places" value={decimalPlaces} onChange={(event) => setDecimalPlaces(Number(event.target.value))}>{Array.from({ length: 9 }, (_, value) => <option key={value} value={value}>{value}</option>)}</select></label><div className="chart-legend">{hasObservedSeries && <span><i className="observed" /> Observed</span>}{compareCorrectedAndPhysical && <span><i className="physical" /> Physical</span>}{showEnsembleBand && <span><i className="ensemble" /> {showForcingBand ? "Forcing spread" : "Initialization spread"}</span>}<span><i className="simulated" /> {showEnsembleBand ? "Ensemble mean" : compareCorrectedAndPhysical ? "Corrected" : "Simulated"}</span></div></div></div>
        <TemporalNavigator overview={overviewTrajectory} range={visibleRange} timestamps={result.timestamps} onRange={setVisibleRange} />
        <div className="chart-wrap">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={outputTrajectory} margin={{ top: 10, right: 18, left: 0, bottom: 4 }}>
              <CartesianGrid strokeDasharray="4 6" vertical={false} stroke="var(--chart-grid)" />
              <XAxis dataKey="index" type="number" domain={[visibleRange.start, Math.max(visibleRange.start + 1, visibleRange.end)]} allowDataOverflow tickFormatter={(value) => result.timestamps?.[Math.round(Number(value))] ?? String(Number(value) + 1)} minTickGap={34} tick={{ fill: "var(--text-muted)", fontSize: 12 }} tickLine={false} axisLine={false} />
              <YAxis tickFormatter={(value) => formatDecimal(Number(value), decimalPlaces)} tick={{ fill: "var(--text-muted)", fontSize: 11 }} tickLine={false} axisLine={false} width={64} label={{ value: selectedUnit, angle: -90, position: "insideLeft", fill: "var(--text-muted)", fontSize: 11 }} />
              <Tooltip formatter={tooltipFormatter} labelFormatter={(value) => result.timestamps?.[Math.round(Number(value))] ?? value} contentStyle={{ background: "var(--surface-elevated)", border: "1px solid var(--border)", borderRadius: 10 }} />
              {showEnsembleBand && <Area type="monotone" dataKey="ensembleRange" name={showForcingBand ? "Forcing-member envelope" : "Initialization 10–90%"} stroke="none" fill="var(--chart-band)" fillOpacity={0.34} connectNulls={false} isAnimationActive={false} />}
              {hasObservedSeries && <Line type="monotone" dataKey="observed" name="Observed" stroke="var(--chart-observed)" strokeWidth={2.4} dot={false} connectNulls={false} />}
              {compareCorrectedAndPhysical && <Line type="monotone" dataKey="physical" name="Physical" stroke="var(--chart-physical)" strokeWidth={2.1} strokeDasharray="7 4" dot={false} />}
              <Line type="monotone" dataKey="simulated" name={showEnsembleBand ? "Ensemble mean" : compareCorrectedAndPhysical ? "Corrected" : "Simulated"} stroke="var(--chart-simulated)" strokeWidth={2.2} dot={false} />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </section>

      <div className="content-grid result-detail-grid result-summary-grid">
        <section className="card parameter-card">
          <p className="section-kicker">{isForecast ? "Source model" : result.training ? "Best checkpoint" : "Best candidate"}</p><h2>{result.training ? isDiscoveryResult ? "Saved process-discovery model" : "Saved neural model" : isForecast ? "Frozen calibrated parameters" : "Calibrated parameters"}</h2>
          <div className="parameter-list">
            {result.training ? <>
              <div><span>Engine</span><strong>{result.training.engine_id}</strong></div>
              {result.training.physical_model_id && <div><span>Embedded physics</span><strong>{result.training.physical_model_id}</strong></div>}
              {result.training.corrector_model_id && <div><span>Corrector</span><strong>{result.training.corrector_model_id}</strong></div>}
              {result.training.correction_strategy && <div><span>Fusion</span><strong>{result.training.correction_strategy.replaceAll("_", " ")}</strong></div>}
              {result.training.lookback !== undefined && <div><span>Lookback</span><strong>{result.training.lookback} days</strong></div>}
              {result.training.feature_roles && <div><span>Features</span><strong>{result.training.feature_roles.length}</strong></div>}
              {result.training.process_count !== undefined && <div><span>Discovered processes</span><strong>{result.training.process_count}</strong></div>}
              {result.training.hidden_units !== undefined && <div><span>Hidden mapping</span><strong>{result.training.hidden_units} units</strong></div>}
              {result.training.snow_module_id && <div><span>Snow module</span><strong>{snowLabel(result.training.snow_module_id)}</strong></div>}
              {Object.entries(result.training.snow_parameters ?? {}).map(([name, value]) => <div key={`discovery-snow-${name}`}><span>{result.training?.calibrated_snow_parameters?.includes(name) ? "Estimated snow" : "Fixed snow"} · {name.replaceAll("_", " ")}</span><strong>{formatDecimal(value, decimalPlaces)}</strong></div>)}
              {result.training.input_switch_probabilities?.length ? <div><span>Input switches</span><strong>{result.training.input_switch_probabilities.map((value) => formatDecimal(value, decimalPlaces)).join(" · ")}</strong></div> : null}
              {result.training.process_switch_probabilities?.length ? <div><span>Process switches</span><strong>{result.training.process_switch_probabilities.map((value) => formatDecimal(value, decimalPlaces)).join(" · ")}</strong></div> : null}
              {isDiscoveryResult && discoveryLossEntries.map(([name, value]) => <div key={name}><span>Selected epoch · {name}</span><strong>{formatDecimal(value, decimalPlaces)}</strong></div>)}
              {result.training.recurrent_layers?.length ? <div><span>Recurrent stack</span><strong>{result.training.recurrent_layers.join(" → ")} {result.training.model_id?.startsWith("gru.") ? "GRU" : "LSTM"} units</strong></div> : result.training.recurrent_units ? <div><span>Recurrent layer</span><strong>{result.training.recurrent_units} {result.training.model_id?.startsWith("gru.") ? "GRU" : "LSTM"} units</strong></div> : null}
              {result.training.convolution_layers?.length ? <div><span>Convolutional stack</span><strong>{result.training.convolution_layers.map((layer) => `${layer.filters}×k${layer.kernel_size}`).join(" → ")}</strong></div> : null}
              {initializationEnsemble && <div><span>Initialization ensemble</span><strong>{initializationEnsemble.member_count_completed} members · arithmetic mean</strong></div>}
              {initializationEnsemble && <div><span>Member envelope</span><strong>{Math.round(initializationEnsemble.lower_quantile * 100)}–{Math.round(initializationEnsemble.upper_quantile * 100)}%</strong></div>}
              <div><span>Best validation {result.training.objective?.symbol ?? "RMSE"}</span><strong>{result.training.best_validation_score === null ? "—" : formatDecimal(result.training.best_validation_score, decimalPlaces)}</strong></div>
              {result.training.physical_model_id && Object.entries(result.calibration?.best_parameters ?? {}).map(([name, value]) => <div key={`physical-${name}`}><span>Physical · {name.replaceAll("_", " ")}</span><strong>{formatDecimal(value, decimalPlaces)}</strong></div>)}
            </> : Object.entries(result.calibration?.best_parameters ?? {}).map(([name, value]) => (
              <div key={name}><span>{name.replaceAll("_", " ")}</span><strong>{formatDecimal(value, decimalPlaces)}</strong></div>
            ))}
          </div>
          <div className="result-integrity"><CircleCheck size={18} /><div><strong>{result.training ? "Saved model verified" : "Numerical integrity verified"}</strong><span>{result.training ? isDiscoveryResult ? "Mapping weights, input and process switches, constraints, transformations, analysis tensors, and epoch history are persisted." : initializationEnsemble ? "Every member’s transformations, weights, seed and validation-selected checkpoint are persisted with the aggregation settings." : "Transformations, kernels, biases, architecture, and epoch history are persisted." : `${result.best_simulation.states.length} state boundaries for ${result.best_simulation.outputs.length} forcing intervals.`}</span></div></div>
        </section>
        {hasProcesses && <section className="card process-explorer-launch">
          <div className="icon-tile cyan"><Activity size={20} /></div>
          <div><p className="section-kicker">Progressive results</p><h2>Process explorer</h2><p>The full animation and storage chart are loaded only when requested, keeping the initial results view responsive.</p></div>
          <button className="button button-secondary" aria-expanded={processExpanded} onClick={() => setProcessExpanded((current) => !current)}>{processExpanded ? "Close process explorer" : "Open process explorer"}<ArrowRight className={processExpanded ? "rotate-up" : ""} size={16} /></button>
        </section>}
      </div>

      {isExplainable && <section className="card shap-card" aria-labelledby="shap-title">
        <div className="shap-heading">
          <div className="shap-heading-main">
            <div className="icon-tile purple"><BrainCircuit size={20} /></div>
            <div><p className="section-kicker">Post-training interpretation</p><h2 id="shap-title">SHAP explainability</h2><p>Attribute frozen test predictions to each predictor and antecedent lag, using empirical training windows as the reference distribution.</p></div>
          </div>
          <button className="button button-secondary" disabled={explanationBusy} onClick={computeExplanation}>
            <BrainCircuit size={16} /> {explanationBusy ? "Computing…" : explanation ? "Recompute SHAP" : "Compute SHAP"}
          </button>
        </div>
        {!explanation && !explanationError && <div className="shap-method-note"><Info size={17} /><div><strong>Model explanation, not causal attribution</strong><span>The analysis uses variable × lag players and the training partition only. Predictions and stored weights remain unchanged.</span></div></div>}
        {explanationError && <div className="shap-error" role="alert"><CircleAlert size={18} /><div><strong>{explanationError.summary ?? "SHAP could not be computed"}</strong><span>{explanationError.explanation ?? explanationError.message ?? "Review the saved model and retry."}</span></div></div>}
        {explanation && <>
          <div className="shap-audit-grid">
            <div><span>Reference</span><strong>{explanation.background_rows} training windows</strong></div>
            <div><span>Explained support</span><strong>{explanation.explained_rows} test predictions</strong></div>
            <div><span>Players</span><strong>{explanation.player_count} variable × lag cells</strong></div>
            <div><span>Maximum additivity error</span><strong>{explanation.maximum_additivity_error.toExponential(2)} mm/day</strong></div>
            <div><span>Maximum archive difference</span><strong>{explanation.maximum_archive_prediction_difference.toExponential(2)} mm/day</strong></div>
          </div>
          <div className="shap-global-grid">
            <div className="shap-panel">
              <div className="shap-panel-heading"><div><span>Global importance</span><strong>By predictor</strong></div><small>Top 12 · mean |SHAP| · mm/day</small></div>
              <div className="shap-bars">
                {explanation.global_by_feature.slice(0, 12).map((item) => <div className="shap-bar-row" key={item.feature}>
                  <span title={item.feature}>{variableLabel(item.feature)}</span>
                  <div><i style={{ width: `${maximumFeatureImportance > 0 ? 100 * item.mean_absolute_shap / maximumFeatureImportance : 0}%` }} /></div>
                  <strong>{formatDecimal(item.mean_absolute_shap, decimalPlaces)}</strong>
                </div>)}
              </div>
            </div>
            <div className="shap-panel">
              <div className="shap-panel-heading"><div><span>Antecedent importance</span><strong>By lag</strong></div><small>Top 24 · mean |SHAP| · mm/day</small></div>
              <div className="shap-bars">
                {displayedLagImportance.map((item) => <div className="shap-bar-row" key={item.lag_steps}>
                  <span>t − {item.lag_steps}</span>
                  <div><i className="lag" style={{ width: `${maximumLagImportance > 0 ? 100 * item.mean_absolute_shap / maximumLagImportance : 0}%` }} /></div>
                  <strong>{formatDecimal(item.mean_absolute_shap, decimalPlaces)}</strong>
                </div>)}
              </div>
            </div>
          </div>
          {selectedLocalExplanation && <div className="shap-local">
            <div className="shap-local-heading">
              <div><span>Local explanation</span><strong>{selectedLocalExplanation.label}</strong></div>
              <label><span>Test prediction</span><select aria-label="Explained test prediction" value={Math.min(selectedExplanationRow, explanation.local.length - 1)} onChange={(event) => setSelectedExplanationRow(Number(event.target.value))}>{explanation.local.map((item, index) => <option value={index} key={`${item.source_row}-${item.label}`}>{item.label}</option>)}</select></label>
            </div>
            <div className="shap-equation">
              {selectedLocalExplanation.physical_streamflow !== null && <div><span>Physical Q · held fixed</span><strong>{formatDecimal(selectedLocalExplanation.physical_streamflow, decimalPlaces)}</strong></div>}
              <div><span>SHAP reference prediction</span><strong>{formatDecimal(selectedLocalExplanation.baseline, decimalPlaces)}</strong></div>
              <div><span>Σ SHAP</span><strong>{formatDecimal(selectedLocalExplanation.attributions.reduce((total, item) => total + item.shap_value, 0), decimalPlaces)}</strong></div>
              <div className="shap-prediction"><span>Model prediction</span><strong>{formatDecimal(selectedLocalExplanation.prediction, decimalPlaces)} mm/day</strong></div>
              {selectedLocalExplanation.observation !== null && <div><span>Observed</span><strong>{formatDecimal(selectedLocalExplanation.observation, decimalPlaces)}</strong></div>}
            </div>
            <div className="shap-attribution-list">
              {selectedLocalExplanation.attributions.slice(0, 16).map((item) => {
                const magnitude = Math.abs(item.shap_value);
                return <div className={`shap-attribution${item.shap_value >= 0 ? " positive" : " negative"}`} key={`${item.feature}-${item.lag_steps}`}>
                  <div><strong>{variableLabel(item.feature)}</strong><span>t − {item.lag_steps} · SE {formatDecimal(item.standard_error, decimalPlaces)}</span></div>
                  <div className="shap-attribution-track"><i style={{ width: `${maximumLocalAttribution > 0 ? 100 * magnitude / maximumLocalAttribution : 0}%` }} /></div>
                  <strong>{item.shap_value >= 0 ? "+" : ""}{formatDecimal(item.shap_value, decimalPlaces)}</strong>
                </div>;
              })}
            </div>
          </div>}
          <div className="shap-warnings">{explanation.warnings.map((warning) => <div key={warning.code}><Info size={15} /><span>{warning.message}</span></div>)}</div>
        </>}
      </section>}

      {processExpanded && hasProcesses && <div className="deferred-results" aria-label="Process explorer details">
        <HydrologicAnimator result={result} decimalPlaces={decimalPlaces} />
        <section className="card chart-card compact-chart storage-chart-card">
          <div className="chart-heading"><div><p className="section-kicker">Boundary states</p><h2>Internal storages</h2></div><span className="render-note">Shared zoom · full data retained</span></div>
          <div className="chart-wrap small">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={stateTrajectory} margin={{ top: 10, right: 16, left: 0, bottom: 4 }}>
                <CartesianGrid strokeDasharray="4 6" vertical={false} stroke="var(--chart-grid)" />
                <XAxis dataKey="index" type="number" domain={[visibleRange.start, Math.max(visibleRange.start + 1, visibleRange.end)]} allowDataOverflow tickFormatter={(value) => result.timestamps?.[Math.round(Number(value))] ?? String(Number(value) + 1)} minTickGap={34} tick={{ fill: "var(--text-muted)", fontSize: 12 }} tickLine={false} axisLine={false} />
                <YAxis tickFormatter={(value) => formatDecimal(Number(value), decimalPlaces)} tick={{ fill: "var(--text-muted)", fontSize: 12 }} tickLine={false} axisLine={false} width={62} />
                <Tooltip formatter={tooltipFormatter} labelFormatter={(value) => result.timestamps?.[Math.round(Number(value))] ?? value} contentStyle={{ background: "var(--surface-elevated)", border: "1px solid var(--border)", borderRadius: 10 }} />
                <Legend />
                {displayStateNames.map((name, position) => <Line key={name} type="monotone" dataKey={name} name={name.replaceAll("_", " ")} stroke={stateColors[position % stateColors.length]} strokeWidth={2.2} dot={false} />)}
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>
      </div>}
    </>
  );
}

export default function App() {
  const [panel, setPanel] = useState<PanelId>("data");
  const [theme, setTheme] = useState<"light" | "dark">("light");
  const [form, setForm] = useState<WorkspaceForm>(defaultForm);
  const [preview, setPreview] = useState<CsvPreview | null>(null);
  const [quality, setQuality] = useState<DataQualityReport | null>(null);
  const [validation, setValidation] = useState<ValidationSummary | null>(null);
  const [task, setTask] = useState<TaskSnapshot | null>(null);
  const [result, setResult] = useState<RunResult | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<StructuredError | null>(null);
  const [service, setService] = useState<{ connected: boolean; version?: string }>({ connected: false });
  const [projectLibraryOpen, setProjectLibraryOpen] = useState(false);
  const [forecastOpen, setForecastOpen] = useState(false);
  const [comparisonOpen, setComparisonOpen] = useState(false);
  const [temporalValidationOpen, setTemporalValidationOpen] = useState(false);
  const [hindcastVerificationOpen, setHindcastVerificationOpen] = useState(false);
  const [savedProjects, setSavedProjects] = useState<SavedExperimentSummary[]>([]);
  const [saveState, setSaveState] = useState<"idle" | "saved">("idle");
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);

  useEffect(() => {
    health()
      .then((response) => setService({ connected: true, version: response.version }))
      .catch(() => setService({ connected: false }));
  }, []);

  useEffect(() => {
    if (!task?.task_id || result) return;
    const taskId = task.task_id;
    let active = true;
    let timer: number | undefined;
    const poll = async () => {
      try {
        const snapshot = await taskStatus(taskId);
        if (!active) return;
        setTask(snapshot);
        if (snapshot.terminal) {
          if (snapshot.state === "completed" && snapshot.has_result) {
            const completed = await runResult(snapshot.task_id);
            if (active) {
              setResult(completed);
              setPanel("results");
            }
          } else if (snapshot.error) {
            setError(snapshot.error);
          }
          return;
        }
        timer = window.setTimeout(poll, 350);
      } catch (value) {
        if (active) setError(errorFrom(value));
      }
    };
    timer = window.setTimeout(poll, 200);
    return () => {
      active = false;
      if (timer) window.clearTimeout(timer);
    };
  }, [task?.task_id, result]);

  const applyForm = (update: Partial<WorkspaceForm>) => {
    setForm((current) => {
      const next = { ...current, ...update };
      if (update.snowModule === "snow.none.v1") next.initialSnowStorage = 0;
      return next;
    });
    if ((Object.keys(update) as Array<keyof WorkspaceForm>).some((key) => dataQualityFields.has(key))) {
      setQuality(null);
    }
    setValidation(null);
    setTask(null);
    setResult(null);
  };

  const applyPreview = (next: CsvPreview) => {
    const trainEnd = Math.max(1, Math.min(next.row_count - 2, Math.floor(next.row_count * 0.6)));
    const validationEnd = Math.max(trainEnd + 1, Math.min(next.row_count - 1, Math.floor(next.row_count * 0.8)));
    setPreview(next);
    setForm((current) => {
      const additionalPredictors = current.additionalPredictors.filter(
        (item) => next.columns.includes(item.column),
      );
      const availableNames = new Set([
        "precipitation",
        "temperature",
        "potential_evapotranspiration",
        ...additionalPredictors.map((item) => item.name),
      ]);
      return {
        ...current,
        mapping: autoMapColumns(next.columns),
        additionalPredictors,
        mlpFeatureRoles: current.mlpFeatureRoles.filter((name) => availableNames.has(name)),
        laggedFeatures: current.laggedFeatures.filter(
          (item) => availableNames.has(item.source) || item.source.startsWith("physical_"),
        ),
        endIndex: trainEnd,
        conceptualProtocol: "calibration_validation_test",
        warmupSteps: Math.min(current.warmupSteps, Math.max(0, next.row_count - 1)),
        trainStartIndex: 0,
        trainEndIndex: trainEnd,
        validationEndIndex: validationEnd,
        testEndIndex: next.row_count,
      };
    });
    setQuality(null);
    setValidation(null);
    setTask(null);
    setResult(null);
  };

  const handleImport = async (file: File) => {
    setBusy(true);
    setError(null);
    try {
      applyPreview(await importCsv(file));
    } catch (value) {
      setError(errorFrom(value));
    } finally {
      setBusy(false);
      if (fileInput.current) fileInput.current.value = "";
    }
  };

  const loadExampleDataset = async () => {
    setBusy(true);
    setError(null);
    try {
      const next = await previewWorkspaceCsv("examples/data/synthetic_exp_hydro.csv");
      applyPreview(next);
      setForm((current) => ({ ...current, experimentId: "synthetic-exp-hydro", name: "Synthetic EXP-HYDRO calibration", evaluations: 25, seed: 2026 }));
    } catch (value) {
      setError(errorFrom(value));
    } finally {
      setBusy(false);
    }
  };

  const config = preview && mappingIsComplete(form.mapping, form.additionalPredictors)
    ? buildExperimentConfig(form, preview.relative_path, preview.sha256)
    : null;

  const runQualityChecks = async () => {
    if (!config) return;
    setBusy(true);
    setError(null);
    try {
      setQuality(await analyzeDataQuality(config));
    } catch (value) {
      setError(errorFrom(value));
    } finally {
      setBusy(false);
    }
  };

  const runValidation = async () => {
    if (!config) return;
    setBusy(true);
    setError(null);
    try {
      setValidation(await validateExperiment(config));
    } catch (value) {
      setError(errorFrom(value));
    } finally {
      setBusy(false);
    }
  };

  const launch = async () => {
    if (!config) return;
    setBusy(true);
    setError(null);
    try {
      const snapshot = await startRun(config);
      setTask(snapshot);
    } catch (value) {
      setError(errorFrom(value));
    } finally {
      setBusy(false);
    }
  };

  const persistCurrentExperiment = async () => {
    if (!config) return;
    setBusy(true);
    setError(null);
    try {
      await saveExperiment(config);
      setSaveState("saved");
      window.setTimeout(() => setSaveState("idle"), 2200);
    } catch (value) {
      setError(errorFrom(value));
    } finally {
      setBusy(false);
    }
  };

  const openProjectLibrary = async () => {
    setBusy(true);
    setError(null);
    try {
      setSavedProjects(await listSavedExperiments());
      setProjectLibraryOpen(true);
    } catch (value) {
      setError(errorFrom(value));
    } finally {
      setBusy(false);
    }
  };

  const openComparison = async () => {
    setBusy(true);
    setError(null);
    try {
      setSavedProjects(await listSavedExperiments());
      setComparisonOpen(true);
    } catch (value) {
      setError(errorFrom(value));
    } finally {
      setBusy(false);
    }
  };

  const openTemporalValidation = async () => {
    setBusy(true);
    setError(null);
    try {
      setSavedProjects(await listSavedExperiments());
      setTemporalValidationOpen(true);
    } catch (value) {
      setError(errorFrom(value));
    } finally {
      setBusy(false);
    }
  };

  const openSavedSetup = async (experimentId: string) => {
    setBusy(true);
    setError(null);
    try {
      const saved = await loadSavedExperiment(experimentId);
      const nextPreview = await previewWorkspaceCsv(saved.config.data.path);
      setPreview(nextPreview);
      setForm(workspaceFormFromConfig(saved.config));
      setQuality(await analyzeDataQuality(saved.config));
      setValidation(null);
      setTask(null);
      setResult(null);
      setPanel("data");
      setProjectLibraryOpen(false);
    } catch (value) {
      setError(errorFrom(value));
    } finally {
      setBusy(false);
    }
  };

  const openSavedResult = async (taskId: string) => {
    setBusy(true);
    setError(null);
    try {
      const archive = await loadRunArchive(taskId);
      const nextPreview = await previewWorkspaceCsv(archive.config.data.path);
      setPreview(nextPreview);
      setForm(workspaceFormFromConfig(archive.config));
      setQuality(await analyzeDataQuality(archive.config));
      setValidation(archive.result.validation);
      setTask(null);
      setResult(archive.result);
      setPanel("results");
      setProjectLibraryOpen(false);
    } catch (value) {
      setError(errorFrom(value));
    } finally {
      setBusy(false);
    }
  };

  const rerunUnchanged = async () => {
    if (!config) return;
    setBusy(true);
    setError(null);
    setTask(null);
    setResult(null);
    setPanel("run");
    try {
      const nextValidation = await validateExperiment(config);
      setValidation(nextValidation);
      setTask(await startRun(config));
    } catch (value) {
      setError(errorFrom(value));
    } finally {
      setBusy(false);
    }
  };

  const requestCancellation = async () => {
    if (!task) return;
    try {
      setTask(await cancelTask(task.task_id));
    } catch (value) {
      setError(errorFrom(value));
    }
  };

  const downloadConfig = () => {
    if (!config) return;
    const blob = new Blob([JSON.stringify(config, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `${config.experiment_id}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  const stepComplete = (id: PanelId): boolean => {
    if (id === "data") return preview !== null && mappingIsComplete(form.mapping, form.additionalPredictors);
    if (id === "quality" || id === "model") return quality?.ready ?? false;
    if (id === "calibration") return validation !== null || task !== null || result !== null;
    if (id === "run") return task !== null || result !== null;
    return result !== null;
  };

  const canVisit = (id: PanelId): boolean => {
    if (id === "data") return true;
    if (id === "quality") return preview !== null && mappingIsComplete(form.mapping, form.additionalPredictors);
    if (id === "model" || id === "calibration" || id === "run") return quality?.ready ?? false;
    return result !== null;
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand"><div className="brand-mark"><Waves size={23} /></div><div><strong>HydroSoftware</strong><span>Hydrological modelling</span></div></div>
        <div className="project-chip"><span>Current experiment</span><strong>{form.name || "Untitled experiment"}</strong><small>{form.modelId === "deepdiscover.numpy_instantaneous_s2s.v1" ? "DeepDiscover · single-DU S2S" : isPamlModel(form.modelId) ? "PaML · P-RNN + causal CNN" : isLstmModel(form.modelId) ? `LSTM · stacked recurrent ${isS2SNeuralModel(form.modelId) ? "S2S" : "S2V"}` : isGruModel(form.modelId) ? `GRU · stacked recurrent ${isS2SNeuralModel(form.modelId) ? "S2S" : "S2V"}` : isCnnModel(form.modelId) ? `CNN-1D · causal ${isS2SNeuralModel(form.modelId) ? "S2S" : "S2V"}` : isMlpModel(form.modelId) ? `MLP · neural ${isS2SNeuralModel(form.modelId) ? "S2S" : "S2V"}` : `${form.modelId === "hbv.mass_conservative_daily.v1" ? "HBV" : "EXP-HYDRO"} · lumped`}</small></div>
        <nav className="sidebar-nav" aria-label="Experiment workflow">
          {navigation.map((item, index) => {
            const Icon = item.icon;
            const enabled = canVisit(item.id);
            return (
              <button key={item.id} className={`${panel === item.id ? "active" : ""}`} disabled={!enabled} onClick={() => setPanel(item.id)} aria-label={`${item.label}: ${item.detail}`} title={`${item.label}: ${item.detail}`}>
                <span className="nav-index">{stepComplete(item.id) ? <Check size={13} /> : String(index + 1).padStart(2, "0")}</span>
                <Icon size={18} />
                <span><strong>{item.id === "calibration" && isTrainableModel(form.modelId) ? "Training" : item.label}</strong><small>{item.id === "model" && isTrainableModel(form.modelId) ? isDiscoveryModel(form.modelId) ? "Processes & constraints" : "Architecture & inputs" : item.id === "calibration" && isTrainableModel(form.modelId) ? "Partitions & optimizer" : item.detail}</small></span>
              </button>
            );
          })}
        </nav>
        <div className="sidebar-bottom">
          <div className="engine-state"><span className={service.connected ? "online" : "offline"} /><div><strong>{service.connected ? "Engine connected" : "Engine unavailable"}</strong><small>{service.connected ? `Local API v${service.version}` : "Start hydrosoftware-api"}</small></div></div>
          <span className="version-label">Version {service.version ?? interfaceVersion}</span>
        </div>
      </aside>

      <main className="workspace">
        <header className="topbar">
          <div className="breadcrumb"><span>HydroSoftware</span><ArrowRight size={13} /><strong>{navigation.find((item) => item.id === panel)?.label}</strong></div>
          <div className="topbar-actions">
            {saveState === "saved" && <span className="save-confirmation"><Check size={13} /> Saved</span>}
            <button className="button button-secondary topbar-button" disabled={!config || busy} onClick={persistCurrentExperiment} title="Save experiment"><Save size={15} /> <span className="topbar-button-label">Save experiment</span></button>
            <button className="button button-secondary topbar-button" disabled={busy} onClick={openProjectLibrary} title="Open saved experiments"><FolderOpen size={15} /> <span className="topbar-button-label">Open</span></button>
            <button className="icon-button" onClick={() => setTheme(theme === "light" ? "dark" : "light")} title={`Use ${theme === "light" ? "dark" : "light"} theme`}>{theme === "light" ? <Moon size={17} /> : <Sun size={17} />}</button>
          </div>
        </header>
        <div className="workspace-scroll">
          {error && <ErrorNotice error={error} onDismiss={() => setError(null)} />}
          {panel === "data" && <DataPanel preview={preview} form={form} busy={busy} onForm={applyForm} onMapping={(role, value) => applyForm({ mapping: { ...form.mapping, [role]: value } })} onImport={() => fileInput.current?.click()} onLoadExample={loadExampleDataset} onNext={() => setPanel("quality")} />}
          {panel === "quality" && <QualityPanel form={form} report={quality} busy={busy} onForm={applyForm} onAnalyze={runQualityChecks} onNext={() => setPanel("model")} />}
          {panel === "model" && <ModelPanel form={form} onForm={applyForm} onNext={() => setPanel("calibration")} />}
          {panel === "calibration" && preview && <CalibrationPanel form={form} preview={preview} quality={quality} onForm={applyForm} onNext={() => setPanel("run")} />}
          {panel === "run" && preview && <RunPanel form={form} preview={preview} quality={quality} validation={validation} task={task} busy={busy} onValidate={runValidation} onStart={launch} onCancel={requestCancellation} onDownload={downloadConfig} />}
          {panel === "results" && result && <ResultsPanel result={result} busy={busy} onNewRun={() => setPanel("calibration")} onRerun={rerunUnchanged} onForecast={() => setForecastOpen(true)} onHindcastVerification={() => setHindcastVerificationOpen(true)} onCompare={openComparison} onTemporalValidation={openTemporalValidation} />}
        </div>
      </main>
      <input ref={fileInput} className="visually-hidden" type="file" accept=".csv,text/csv" onChange={(event) => { const file = event.target.files?.[0]; if (file) void handleImport(file); }} />
      {projectLibraryOpen && <ProjectLibrary projects={savedProjects} busy={busy} onClose={() => setProjectLibraryOpen(false)} onOpenSetup={openSavedSetup} onOpenResult={openSavedResult} />}
      {comparisonOpen && result && <ComparisonDialog projects={savedProjects} currentTaskId={result.task_id} onClose={() => setComparisonOpen(false)} onError={setError} />}
      {temporalValidationOpen && result && <TemporalValidationDialog projects={savedProjects} currentExperimentId={result.experiment_id} onClose={() => setTemporalValidationOpen(false)} onError={setError} />}
      {hindcastVerificationOpen && result?.forecast_ensemble && <HindcastVerificationDialog currentResult={result} onClose={() => setHindcastVerificationOpen(false)} onError={setError} />}
      {forecastOpen && result && config && <ForecastDialog sourceResult={result} sourceConfig={config} onClose={() => setForecastOpen(false)} onError={(nextError) => setError(nextError)} onComplete={(forecastResult) => { setResult(forecastResult); setValidation(forecastResult.validation); setTask(null); setPanel("results"); setForecastOpen(false); }} />}
    </div>
  );
}
