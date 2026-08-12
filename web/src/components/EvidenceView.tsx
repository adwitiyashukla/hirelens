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

export function segmentize(text: string, spans: EvidenceSpan[]): Segment[] {
  const valid = spans.filter(
    (s) => s.start >= 0 && s.end > s.start && s.end <= text.length,
  );
  if (valid.length === 0) return [];

  const boundaries = Array.from(
    new Set(valid.flatMap((s) => [s.start, s.end])),
  ).sort((a, b) => a - b);

  const segments: Segment[] = [];
  for (let i = 0; i < boundaries.length - 1; i += 1) {
    const start = boundaries[i]!;
    const end = boundaries[i + 1]!;
    const covering = valid.filter((s) => s.start <= start && s.end >= end);
    if (covering.length === 0) continue;

    segments.push({
      start,
      end,
      requirementIds: Array.from(new Set(covering.map((s) => s.requirementId))),

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
