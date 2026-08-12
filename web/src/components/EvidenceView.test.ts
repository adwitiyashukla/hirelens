import { describe, expect, it } from "vitest";
import { segmentize, type EvidenceSpan } from "./EvidenceView";

const TEXT = "Built payment services in Python and ran them on Kubernetes.";

const span = (
  start: number,
  end: number,
  requirementId: string,
  verified = true,
): EvidenceSpan => ({ start, end, requirementId, verified });

function reconstruct(text: string, spans: EvidenceSpan[]): string {
  const segments = segmentize(text, spans);
  let out = "";
  let cursor = 0;
  for (const segment of segments) {
    out += text.slice(cursor, segment.start);
    out += text.slice(segment.start, segment.end);
    cursor = segment.end;
  }
  return out + text.slice(cursor);
}

describe("segmentize", () => {
  it("returns nothing when there are no spans", () => {
    expect(segmentize(TEXT, [])).toEqual([]);
  });

  it("keeps a single span intact", () => {
    const segments = segmentize(TEXT, [span(6, 14, "r1")]);
    expect(segments).toHaveLength(1);
    expect(TEXT.slice(segments[0]!.start, segments[0]!.end)).toBe("payment ");
    expect(segments[0]!.requirementIds).toEqual(["r1"]);
  });

  it("leaves disjoint spans separate rather than merging across the gap", () => {
    const segments = segmentize(TEXT, [span(0, 5, "r1"), span(26, 32, "r2")]);
    expect(segments).toHaveLength(2);
    expect(segments[0]!.requirementIds).toEqual(["r1"]);
    expect(segments[1]!.requirementIds).toEqual(["r2"]);
  });

  it("splits overlapping spans into three parts and attributes the middle to both", () => {

    const segments = segmentize(TEXT, [span(6, 22, "r1"), span(14, 32, "r2")]);
    expect(segments).toHaveLength(3);
    expect(segments[0]!.requirementIds).toEqual(["r1"]);
    expect(segments[1]!.requirementIds.sort()).toEqual(["r1", "r2"]);
    expect(segments[2]!.requirementIds).toEqual(["r2"]);
  });

  it("attributes a fully nested span to both requirements", () => {
    const segments = segmentize(TEXT, [span(0, 30, "outer"), span(6, 13, "inner")]);
    const covering = segments.find((s) => s.requirementIds.length === 2);
    expect(covering).toBeDefined();
    expect(covering!.requirementIds.sort()).toEqual(["inner", "outer"]);
  });

  it("collapses duplicate citations of the same range from one requirement", () => {
    const segments = segmentize(TEXT, [span(6, 14, "r1"), span(6, 14, "r1")]);
    expect(segments).toHaveLength(1);
    expect(segments[0]!.requirementIds).toEqual(["r1"]);
  });

  it("marks a segment unverified if any covering citation failed verification", () => {
    const segments = segmentize(TEXT, [
      span(6, 22, "r1", true),
      span(14, 32, "r2", false),
    ]);
    expect(segments[1]!.verified).toBe(false);
    expect(segments[0]!.verified).toBe(true);
  });

  it("discards spans that fall outside the document", () => {

    expect(segmentize(TEXT, [span(5, 5000, "r1")])).toEqual([]);
    expect(segmentize(TEXT, [span(-4, 10, "r1")])).toEqual([]);
    expect(segmentize(TEXT, [span(10, 4, "r1")])).toEqual([]);
  });

  it("reproduces the source text exactly, whatever the spans", () => {
    const cases: EvidenceSpan[][] = [
      [span(0, 5, "a")],
      [span(0, 20, "a"), span(10, 30, "b")],
      [span(0, 60, "a"), span(5, 9, "b"), span(5, 9, "c"), span(40, 60, "d")],
      [span(59, 60, "a"), span(0, 1, "b")],
    ];
    for (const spans of cases) {
      expect(reconstruct(TEXT, spans)).toBe(TEXT);
    }
  });
});
