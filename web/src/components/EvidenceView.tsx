/**
 * Renders the resume text with cited spans highlighted.
 *
 * This is the view that makes the project's central claim inspectable. Every
 * score in the shortlist decomposes into requirement verdicts, and every
 * verdict carries character offsets into this exact text. Hovering a
 * requirement lights up the lines it was scored from. If the model had invented
 * an achievement, there would be nothing to light up, and the pipeline would
 * have rejected the assessment before it ever reached this component.
 *
 * The rendering problem is that citations overlap: two requirements often cite
 * the same bullet point. Naively wrapping each span in a `<mark>` produces
 * invalid nesting and lost text. So the spans are flattened into a
 * non-overlapping segment list first, by a boundary sweep, and each segment
 * remembers every citation covering it.
 */

import { useEffect, useMemo, useRef } from "react";
import type { RequirementResult } from "../api/types";

export interface EvidenceSpan {
  start: number;
  end: number;
  requirementId: string;
  verified: boolean;
}

interface Segment {
  start: number;
  end: number;
  requirementIds: string[];
  verified: boolean;
}

/** Flatten possibly overlapping spans into disjoint segments, in order. */
export function segmentize(text: string, spans: EvidenceSpan[]): Segment[] {
  const valid = spans.filter(
    (s) => s.start >= 0 && s.end > s.start && s.end <= text.length,
  );
  if (valid.length === 0) return [];

  // Every start and end is a boundary. Between two adjacent boundaries the set
  // of covering spans is constant, which is what makes the sweep correct.
  const boundaries = Array.from(
    new Set(valid.flatMap((s) => [s.start, s.end])),
  ).sort((a, b) => a - b);

  const segments: Segment[] = [];
  for (let i = 0; i < boundaries.length - 1; i += 1) {
    const start = boundaries[i]!;
    const end = boundaries[i + 1]!;
    const covering = valid.filter((s) => s.start <= start && s.end >= end);
    if (covering.length === 0) continue; // a gap between two disjoint spans

    segments.push({
      start,
      end,
      requirementIds: Array.from(new Set(covering.map((s) => s.requirementId))),
      // One unverified citation is enough to mark the segment as suspect. Better
      // to over-flag than to present unchecked text as confirmed.
      verified: covering.every((s) => s.verified),
    });
  }
  return segments;
}

export function EvidenceView({
  text,
  spans,
  activeRequirementId,
  onSelectRequirement,
}: {
  text: string;
  spans: EvidenceSpan[];
  activeRequirementId: string | null;
  onSelectRequirement: (requirementId: string) => void;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  const segments = useMemo(() => segmentize(text, spans), [text, spans]);

  // Bring the first highlight of the active requirement into view. Without
  // this, selecting a requirement on a three-page resume appears to do nothing,
  // because the evidence is below the fold.
  useEffect(() => {
    if (!activeRequirementId || !containerRef.current) return;
    const target = containerRef.current.querySelector(
      `[data-first="${CSS.escape(activeRequirementId)}"]`,
    );
    target?.scrollIntoView({ behavior: "smooth", block: "center" });
  }, [activeRequirementId]);

  if (segments.length === 0) {
    return (
      <div ref={containerRef} className="resume-text">
        {text}
      </div>
    );
  }

  const nodes: React.ReactNode[] = [];
  let cursor = 0;
  const seenFirst = new Set<string>();

  segments.forEach((segment, index) => {
    if (segment.start > cursor) {
      nodes.push(text.slice(cursor, segment.start));
    }

    const isActive =
      activeRequirementId !== null &&
      segment.requirementIds.includes(activeRequirementId);
    // When something is selected, everything else fades. With a dozen
    // requirements cited across one page, leaving them all lit is unreadable.
    const isDimmed = activeRequirementId !== null && !isActive;

    const firstFor = segment.requirementIds.find((id) => !seenFirst.has(id));
    if (firstFor) seenFirst.add(firstFor);

    const classes = [
      !segment.verified ? "unverified" : "",
      isActive ? "active" : "",
      isDimmed ? "dim" : "",
    ]
      .filter(Boolean)
      .join(" ");

    nodes.push(
      <mark
        key={`seg-${index}`}
        className={classes}
        {...(firstFor ? { "data-first": firstFor } : {})}
        title={
          segment.verified
            ? `Cited by ${segment.requirementIds.length} requirement(s). Click to focus.`
            : "This quote no longer matches the stored document text."
        }
        onClick={() => {
          const first = segment.requirementIds[0];
          if (first) onSelectRequirement(first);
        }}
      >
        {text.slice(segment.start, segment.end)}
      </mark>,
    );
    cursor = segment.end;
  });

  if (cursor < text.length) nodes.push(text.slice(cursor));

  return (
    <div ref={containerRef} className="resume-text">
      {nodes}
    </div>
  );
}

/** Collect every citation from an assessment into flat span records. */
export function spansFrom(requirements: RequirementResult[]): EvidenceSpan[] {
  return requirements.flatMap((requirement) =>
    requirement.citations.map((citation) => ({
      start: citation.start,
      end: citation.end,
      requirementId: requirement.requirement_id,
      verified: citation.verified,
    })),
  );
}
