import type {
  CsvPreview,
  DataQualityReport,
  ExperimentConfig,
  ForecastEnsembleArchive,
  ForecastEnsembleArchiveSummary,
  ForecastEnsembleRequest,
  ForecastEventVerification,
  ForecastEventVerificationArchive,
  ForecastEventVerificationArchiveSummary,
  ForecastPostprocessing,
  ForecastPostprocessingArchive,
  ForecastPostprocessingArchiveSummary,
  ForecastVerification,
  ForecastVerificationArchive,
  ForecastVerificationArchiveSummary,
  ForecastRequest,
  RunArchive,
  RunArchiveSummary,
  RunComparison,
  RunExplanation,
  RunResult,
  RunScientificVerification,
  RunUncertainty,
  SavedExperiment,
  SavedExperimentSummary,
  StructuredError,
  TaskSnapshot,
  TemporalValidation,
  TemporalValidationArchive,
  TemporalValidationArchiveSummary,
  TemporalValidationRequest,
  ValidationSummary,
} from "./types";

const API_BASE = import.meta.env.VITE_API_BASE ?? "/api/v1";

export class ApiError extends Error {
  detail: StructuredError;

  constructor(message: string, detail: StructuredError) {
    super(message);
    this.name = "ApiError";
    this.detail = detail;
  }
}

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers: {
      "Content-Type": "application/json",
      ...options?.headers,
    },
  });
  const payload = (await response.json()) as T & { detail?: StructuredError };
  if (!response.ok) {
    const detail = payload.detail ?? {
      code: "request_failed",
      message: `The local service returned HTTP ${response.status}.`,
    };
    throw new ApiError(
      detail.message ?? detail.explanation ?? detail.summary ?? "Request failed.",
      detail,
    );
  }
  return payload;
}

export async function health(): Promise<{ version: string; status: string }> {
  return request("/health");
}

export async function previewWorkspaceCsv(path: string): Promise<CsvPreview> {
  return request("/data/preview", {
    method: "POST",
    body: JSON.stringify({ path }),
  });
}

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  const chunkSize = 0x8000;
  let binary = "";
  for (let offset = 0; offset < bytes.length; offset += chunkSize) {
    binary += String.fromCharCode(...bytes.subarray(offset, offset + chunkSize));
  }
  return btoa(binary);
}

export async function importCsv(file: File): Promise<CsvPreview> {
  return request("/data/imports", {
    method: "POST",
    body: JSON.stringify({
      filename: file.name,
      content_base64: arrayBufferToBase64(await file.arrayBuffer()),
    }),
  });
}

export async function analyzeDataQuality(
  config: ExperimentConfig,
): Promise<DataQualityReport> {
  return request("/data/quality", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

export async function validateExperiment(
  config: ExperimentConfig,
): Promise<ValidationSummary> {
  return request("/experiments/validate", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

export async function startRun(config: ExperimentConfig): Promise<TaskSnapshot> {
  return request("/runs", {
    method: "POST",
    body: JSON.stringify(config),
  });
}

export async function startForecast(requestBody: ForecastRequest): Promise<RunResult> {
  return request("/forecasts", {
    method: "POST",
    body: JSON.stringify(requestBody),
  });
}

export async function startForecastEnsemble(
  requestBody: ForecastEnsembleRequest,
): Promise<RunResult> {
  return request("/forecast-ensembles", {
    method: "POST",
    body: JSON.stringify(requestBody),
  });
}

export async function listForecastEnsembles(): Promise<
  ForecastEnsembleArchiveSummary[]
> {
  const payload = await request<{
    forecast_ensembles: ForecastEnsembleArchiveSummary[];
  }>("/forecast-ensembles");
  return payload.forecast_ensembles;
}

export async function loadForecastEnsemble(
  taskId: string,
): Promise<ForecastEnsembleArchive> {
  return request(`/forecast-ensembles/${encodeURIComponent(taskId)}`);
}

export async function verifyForecastEnsembles(
  taskIds: string[],
  options: {
    seed?: number;
    minimumCasesPerLead?: number;
  } = {},
): Promise<ForecastVerification> {
  return request("/forecast-verifications", {
    method: "POST",
    body: JSON.stringify({
      task_ids: taskIds,
      seed: options.seed ?? 0,
      minimum_cases_per_lead: options.minimumCasesPerLead ?? 2,
    }),
  });
}

export async function listForecastVerifications(): Promise<
  ForecastVerificationArchiveSummary[]
> {
  const payload = await request<{
    forecast_verifications: ForecastVerificationArchiveSummary[];
  }>("/forecast-verifications");
  return payload.forecast_verifications;
}

export async function loadForecastVerification(
  taskId: string,
): Promise<ForecastVerificationArchive> {
  return request(`/forecast-verifications/${encodeURIComponent(taskId)}`);
}

export async function postprocessForecastEnsembles(
  taskIds: string[],
  options: {
    trainingCaseCount: number;
    seed?: number;
    minimumTrainingCasesPerLead?: number;
  },
): Promise<ForecastPostprocessing> {
  return request("/forecast-postprocessings", {
    method: "POST",
    body: JSON.stringify({
      task_ids: taskIds,
      training_case_count: options.trainingCaseCount,
      seed: options.seed ?? 0,
      minimum_training_cases_per_lead:
        options.minimumTrainingCasesPerLead ?? 5,
    }),
  });
}

export async function listForecastPostprocessings(): Promise<
  ForecastPostprocessingArchiveSummary[]
> {
  const payload = await request<{
    forecast_postprocessings: ForecastPostprocessingArchiveSummary[];
  }>("/forecast-postprocessings");
  return payload.forecast_postprocessings;
}

export async function loadForecastPostprocessing(
  taskId: string,
): Promise<ForecastPostprocessingArchive> {
  return request(`/forecast-postprocessings/${encodeURIComponent(taskId)}`);
}

export async function verifyForecastEvents(
  taskIds: string[],
  options: {
    trainingCaseCount: number;
    thresholdMode: "absolute" | "training_quantile";
    thresholdValue?: number;
    trainingQuantile?: number;
    probabilityBinCount?: number;
    minimumEvaluationCasesPerLead?: number;
  },
): Promise<ForecastEventVerification> {
  return request("/forecast-event-verifications", {
    method: "POST",
    body: JSON.stringify({
      task_ids: taskIds,
      training_case_count: options.trainingCaseCount,
      threshold_mode: options.thresholdMode,
      ...(options.thresholdMode === "absolute"
        ? { threshold_value: options.thresholdValue }
        : {}),
      training_quantile: options.trainingQuantile ?? 0.9,
      probability_bin_count: options.probabilityBinCount ?? 10,
      minimum_evaluation_cases_per_lead:
        options.minimumEvaluationCasesPerLead ?? 5,
    }),
  });
}

export async function listForecastEventVerifications(): Promise<
  ForecastEventVerificationArchiveSummary[]
> {
  const payload = await request<{
    forecast_event_verifications: ForecastEventVerificationArchiveSummary[];
  }>("/forecast-event-verifications");
  return payload.forecast_event_verifications;
}

export async function loadForecastEventVerification(
  taskId: string,
): Promise<ForecastEventVerificationArchive> {
  return request(`/forecast-event-verifications/${encodeURIComponent(taskId)}`);
}

export async function taskStatus(taskId: string): Promise<TaskSnapshot> {
  return request(`/tasks/${taskId}`);
}

export async function cancelTask(taskId: string): Promise<TaskSnapshot> {
  return request(`/tasks/${taskId}/cancel`, { method: "POST" });
}

export async function runResult(taskId: string): Promise<RunResult> {
  return request(`/tasks/${taskId}/result`);
}

export async function listSavedExperiments(): Promise<SavedExperimentSummary[]> {
  const payload = await request<{ experiments: SavedExperimentSummary[] }>("/experiments");
  return payload.experiments;
}

export async function saveExperiment(
  config: ExperimentConfig,
): Promise<SavedExperimentSummary> {
  return request(`/experiments/${encodeURIComponent(config.experiment_id)}`, {
    method: "PUT",
    body: JSON.stringify(config),
  });
}

export async function loadSavedExperiment(experimentId: string): Promise<SavedExperiment> {
  return request(`/experiments/${encodeURIComponent(experimentId)}`);
}

export async function listSavedRuns(experimentId: string): Promise<RunArchiveSummary[]> {
  const payload = await request<{ runs: RunArchiveSummary[] }>(
    `/experiments/${encodeURIComponent(experimentId)}/runs`,
  );
  return payload.runs;
}

export async function loadRunArchive(taskId: string): Promise<RunArchive> {
  return request(`/run-archives/${encodeURIComponent(taskId)}`);
}

export async function compareSavedRuns(taskIds: string[]): Promise<RunComparison> {
  return request("/comparisons", {
    method: "POST",
    body: JSON.stringify({ task_ids: taskIds }),
  });
}

export async function explainSavedRun(
  taskId: string,
  options: {
    explainedRows?: number;
    backgroundRows?: number;
    permutations?: number;
    seed?: number;
  } = {},
): Promise<RunExplanation> {
  return request("/explanations", {
    method: "POST",
    body: JSON.stringify({
      task_id: taskId,
      explained_rows: options.explainedRows ?? 8,
      background_rows: options.backgroundRows ?? 4,
      permutations: options.permutations ?? 8,
      seed: options.seed ?? 0,
    }),
  });
}

export async function verifySavedRunScientifically(
  taskId: string,
): Promise<RunScientificVerification> {
  return request("/scientific-verifications", {
    method: "POST",
    body: JSON.stringify({ task_id: taskId }),
  });
}

export async function quantifyRunUncertainty(
  taskIds: string[],
  options: {
    bootstrapReplicates?: number;
    confidenceLevel?: number;
    blockLength?: number | null;
    seed?: number;
    referenceTaskId?: string;
  } = {},
): Promise<RunUncertainty> {
  return request("/uncertainty", {
    method: "POST",
    body: JSON.stringify({
      task_ids: taskIds,
      bootstrap_replicates: options.bootstrapReplicates ?? 1000,
      confidence_level: options.confidenceLevel ?? 0.95,
      block_length: options.blockLength ?? null,
      seed: options.seed ?? 0,
      reference_task_id: options.referenceTaskId ?? taskIds[0],
    }),
  });
}

export async function startTemporalValidation(
  requestBody: TemporalValidationRequest,
): Promise<TaskSnapshot> {
  return request("/temporal-validations", {
    method: "POST",
    body: JSON.stringify(requestBody),
  });
}

export async function temporalValidationStatus(
  taskId: string,
): Promise<TaskSnapshot> {
  return request(
    `/temporal-validations/tasks/${encodeURIComponent(taskId)}`,
  );
}

export async function cancelTemporalValidation(
  taskId: string,
): Promise<TaskSnapshot> {
  return request(
    `/temporal-validations/tasks/${encodeURIComponent(taskId)}/cancel`,
    { method: "POST" },
  );
}

export async function temporalValidationResult(
  taskId: string,
): Promise<TemporalValidation> {
  return request(
    `/temporal-validations/tasks/${encodeURIComponent(taskId)}/result`,
  );
}

export async function listTemporalValidations(): Promise<
  TemporalValidationArchiveSummary[]
> {
  const payload = await request<{
    temporal_validations: TemporalValidationArchiveSummary[];
  }>("/temporal-validations");
  return payload.temporal_validations;
}

export async function loadTemporalValidation(
  taskId: string,
): Promise<TemporalValidationArchive> {
  return request(
    `/temporal-validations/archives/${encodeURIComponent(taskId)}`,
  );
}
