import { describe, expect, it } from "vitest";
import { downsampleForChart } from "./chartData";

describe("chart trajectory optimization", () => {
  it("keeps endpoints and local scientific extrema within the render budget", () => {
    const rows = Array.from({ length: 10_000 }, (_, index) => ({
      index,
      observed: index === 4_321 ? 999 : Math.sin(index / 20),
      simulated: index === 7_654 ? -500 : Math.cos(index / 25),
    }));

    const sampled = downsampleForChart(rows, ["observed", "simulated"], 500);
    expect(sampled.length).toBeLessThanOrEqual(500);
    expect(sampled[0].index).toBe(0);
    expect(sampled.at(-1)?.index).toBe(9_999);
    expect(sampled.some((row) => row.index === 4_321)).toBe(true);
    expect(sampled.some((row) => row.index === 7_654)).toBe(true);
  });
});
