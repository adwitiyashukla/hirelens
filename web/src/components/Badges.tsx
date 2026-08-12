import type { Band, Verdict } from "../api/types";

const VERDICT_LABEL: Record<Verdict, string> = {
  strong: "Strong evidence",
  clear: "Clear evidence",
  partial: "Partial evidence",
  weak: "Weak evidence",
  none: "No evidence",
};

const BAND_TONE: Record<Band, string> = {
  "strong fit": "strong",
  "possible fit": "clear",
  "weak fit": "weak",
  "not a fit": "none",
  "missing a must-have": "danger",
};

export function VerdictBadge({ verdict }: { verdict: Verdict }) {
  return <span className={`badge ${verdict}`}>{VERDICT_LABEL[verdict]}</span>;
}

export function BandBadge({ band }: { band: Band }) {

  return <span className={`badge ${BAND_TONE[band] ?? "neutral"}`}>{band}</span>;
}

export function ScoreCell({
  score,
  low,
  high,
}: {
  score: number;
  low: number;
  high: number;
}) {
  const width = Math.max(high - low, 0);
  return (
    <div>
      <div className="score">
        <span className="value">{score.toFixed(1)}</span>
        <span className="interval">
          [{low.toFixed(0)} to {high.toFixed(0)}]
        </span>
      </div>
      <div className="ci-bar" title={`90% interval: ${low.toFixed(1)} to ${high.toFixed(1)}`}>
        <div className="range" style={{ left: `${low}%`, width: `${width}%` }} />
        <div className="point" style={{ left: `${score}%` }} />
      </div>
    </div>
  );
}

export function RateBadge({
  value,
  good = 0.9,
  label,
}: {
  value: number;
  good?: number;
  label?: string;
}) {
  const tone = value >= good ? "strong" : value >= good - 0.2 ? "partial" : "weak";
  return (
    <span className={`badge ${tone}`} title={label}>
      {(value * 100).toFixed(0)}%
    </span>
  );
}
