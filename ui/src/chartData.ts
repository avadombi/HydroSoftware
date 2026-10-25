export interface IndexedChartRow {
  index: number;
  [key: string]: unknown;
}

/**
 * Reduce large trajectories for SVG rendering while preserving local minima
 * and maxima of every requested scientific series. The full-resolution arrays
 * remain in the saved result and are still used by the process reader.
 */
export function downsampleForChart<T extends IndexedChartRow>(
  rows: T[],
  fields: string[],
  maximumPoints = 900,
): T[] {
  if (rows.length <= maximumPoints || maximumPoints < 4 || fields.length === 0) {
    return rows;
  }

  const selectionsPerBucket = Math.max(1, fields.length * 2);
  const bucketCount = Math.max(1, Math.floor((maximumPoints - 2) / selectionsPerBucket));
  const interiorLength = rows.length - 2;
  const selected = new Set<number>([0, rows.length - 1]);

  for (let bucket = 0; bucket < bucketCount; bucket += 1) {
    const start = 1 + Math.floor(bucket * interiorLength / bucketCount);
    const end = 1 + Math.floor((bucket + 1) * interiorLength / bucketCount);
    for (const field of fields) {
      let minimumIndex = -1;
      let maximumIndex = -1;
      let minimum = Number.POSITIVE_INFINITY;
      let maximum = Number.NEGATIVE_INFINITY;
      for (let index = start; index < end; index += 1) {
        const value = rows[index][field];
        if (typeof value !== "number" || !Number.isFinite(value)) continue;
        if (value < minimum) {
          minimum = value;
          minimumIndex = index;
        }
        if (value > maximum) {
          maximum = value;
          maximumIndex = index;
        }
      }
      if (minimumIndex >= 0) selected.add(minimumIndex);
      if (maximumIndex >= 0) selected.add(maximumIndex);
    }
  }

  return [...selected].sort((left, right) => left - right).map((index) => rows[index]);
}
