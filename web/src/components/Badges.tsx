import type { Band, Verdict } from "../api/types";

/**
 * The verdict labels shown to a user.
 *
 * The API speaks the ordinal scale (`strong` down to `none`); a recruiter reading
 * a table wants a sentence fragment, not an enum. The mapping lives here rather
 * than inline so that both the table and the detail view agree.
 */
const VERDICT_LABEL: Record<Verdict, string> = {
  strong: "Strong evidence",
  clear: "Clear evidence",
  partial: "Partial evidence",
  weak: "Weak evidence",
  none: "No evidence",
};

const BAND_LABEL: Record<Band, string> = {
  strong: "Strong fit",
  promising: "Promising",
  borderline: "Borderline",
  weak: "Weak fit",
};

export function VerdictBadge({ verdict }: { verdict: Verdict }) {
  return <span className={`badge ${verdict}`}>{VERDICT_LABEL[verdict]}</span>;
}

export function BandBadge({ band }: { band: Band }) {
  return <span className={`badge ${band}`}>{BAND_LABEL[band]}</span>;
}

/**
 * A score with its confidence interval, never the point estimate alone.
 *
 * The interval is the output of bootstrap resampling over the self-consistency
 * samples, and showing it is the point: a 61 that could be anywhere from 54 to
 * 68 should not be read as beating a 59. Hiding the width would invite exactly
 * the false precision this project exists to argue against.
 */
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

/** A 0..1 rate rendered as a percentage, with a threshold colour. */
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
