"""Render the fairness audit, and gate CI on it.

The markdown output is written to be publishable. NYC Local Law 144 requires an
annual bias audit of automated employment decision tools, and the EU AI Act
requires documentation for high-risk systems; this is not that document, but it is
the measurement such a document would be built on, and it is shaped accordingly:
method stated, control reported, numbers given whether or not they flatter the
system.

The gate fires on blind-mode drift only. Blind mode is the shipping configuration,
so that is what has to be safe. The sighted numbers are diagnostic, and a large
sighted drift is not a build failure, it is the finding that justifies blind mode
existing.
"""

from __future__ import annotations

from dataclasses import dataclass

from hirelens.audit.runner import AuditReport

_AXIS_LABELS = {
    "null": "control (nothing changed)",
    "gender": "gender-coded name",
    "ethnicity": "ethnicity-coded name",
    "university": "university prestige",
    "location": "location",
}


def _label(axis: str) -> str:
    return _AXIS_LABELS.get(axis, axis)


# ---------------------------------------------------------------------------
# Console
# ---------------------------------------------------------------------------


def to_console(report: AuditReport) -> str:
    out: list[str] = []
    add = out.append
    rule = "=" * 76

    add(rule)
    add(f"FAIRNESS AUDIT  {report.provider}/{report.model}  k={report.self_consistency_k}")
    add(
        f"                {report.profiles_tested} profiles x {report.variants_tested} variants, "
        f"job '{report.job_id}'"
    )
    add(rule)

    add("")
    add(f"noise floor (identical resume, re-run) : {report.noise_floor_max:.2f} pts")
    add(f"drift threshold                        : {report.threshold:.2f} pts above the floor")
    add("")

    for blind in (True, False):
        results = [a for a in report.axes if a.blind is blind and a.axis != "null"]
        if not results:
            continue

        heading = (
            "BLIND MODE ON (shipping configuration)" if blind else "BLIND MODE OFF (diagnostic)"
        )
        add(heading)
        add("-" * 76)
        add(f"{'axis':<26} {'max':>7} {'excess':>8} {'gap':>7} {'flips':>6}  verdict")

        for result in sorted(results, key=lambda a: -a.max_drift):
            excess = report.excess_drift(result)
            verdict = "ok" if excess <= report.threshold else "OVER THRESHOLD"
            add(
                f"{_label(result.axis):<26} {result.max_drift:>7.2f} {excess:>8.2f} "
                f"{result.group_gap:>7.2f} {result.rank_flips:>6}  {verdict}"
            )
        add("")

    blind_worst = report.worst_blind_axis
    sighted_worst = report.worst_sighted_axis

    if blind_worst and sighted_worst:
        add(
            f"blind mode removes {report.blind_mode_benefit:.2f} pts of worst-case drift "
            f"({sighted_worst.max_drift:.2f} -> {blind_worst.max_drift:.2f})"
        )

    # Only report a directional gap when there is one. With every group equal,
    # favoured and disfavoured are the same group and the sentence would read
    # "'female' scores 0.00 pts above 'female'".
    if sighted_worst and sighted_worst.group_gap > 0:
        add(
            f"largest sighted gap: '{sighted_worst.favoured_group}' scores "
            f"{sighted_worst.group_gap:.2f} pts above '{sighted_worst.disfavoured_group}' "
            f"on the {_label(sighted_worst.axis)} axis"
        )
    elif sighted_worst:
        add("no directional gap between groups on any sighted axis")

    add("")
    add(f"AUDIT {'PASSED' if report.passes else 'FAILED'}")

    if report.warnings:
        add("")
        add("warnings:")
        for warning in report.warnings:
            add(f"  ! {warning}")

    add("")
    add(f"{report.api_calls} API calls, {report.elapsed_s:.0f}s")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Markdown
# ---------------------------------------------------------------------------


def to_markdown(report: AuditReport) -> str:
    lines: list[str] = []
    add = lines.append

    add("# Bias audit")
    add("")
    add(
        f"Counterfactual audit of `{report.provider}/{report.model}`, "
        f"{report.profiles_tested} candidate profiles across {report.variants_tested} "
        f"demographic variants, scored against the `{report.job_id}` role."
    )
    add("")

    add("## Method")
    add("")
    add(
        "Each candidate resume is scored repeatedly with **only** the demographic fields "
        "changed: name, pronouns, institution, location. Every other character of the "
        "document is byte-identical across variants, which is enforced by construction "
        "(profiles are structured specs and only the demographic block is substituted). "
        "Any movement in the score is therefore *caused* by the demographic change rather "
        "than merely correlated with it."
    )
    add("")
    add(
        f"**A null control runs alongside.** The same unmodified resume is scored twice, "
        f"and the difference is the system's own run-to-run noise: **{report.noise_floor_max:.2f} "
        f"points**. Demographic drift is only meaningful above that floor, and it is reported "
        f"as excess over it. Without this control, ordinary sampling noise would be "
        f"indistinguishable from bias."
    )
    add("")
    add(
        "**Both modes are measured.** With blind mode on (how the system ships) identifying "
        "details are masked before the model sees them. With blind mode off, the same "
        "experiment measures the underlying model bias that masking exists to suppress. "
        "The difference between the two quantifies what the mitigation is worth."
    )
    add("")

    for blind in (True, False):
        results = [a for a in report.axes if a.blind is blind and a.axis != "null"]
        if not results:
            continue

        add(f"## {'Blind mode on (shipping)' if blind else 'Blind mode off (diagnostic)'}")
        add("")
        add("| Axis | Max drift | Excess over noise | Group gap | Rank flips | Must-have flips |")
        add("|---|---|---|---|---|---|")
        for result in sorted(results, key=lambda a: -a.max_drift):
            add(
                f"| {_label(result.axis)} | {result.max_drift:.2f} pts "
                f"| {report.excess_drift(result):.2f} pts | {result.group_gap:.2f} pts "
                f"| {result.rank_flips} | {result.must_have_flips} |"
            )
        add("")

    sighted = report.worst_sighted_axis
    if sighted and len(sighted.group_means) > 1 and sighted.group_gap > 0:
        add("### Group means on the worst sighted axis")
        add("")
        add(f"Axis: {_label(sighted.axis)}")
        add("")
        add("| Group | Mean score |")
        add("|---|---|")
        for group, mean in sorted(sighted.group_means.items(), key=lambda kv: -kv[1]):
            add(f"| {group} | {mean:.1f} |")
        add("")

    add("## Result")
    add("")
    add(
        f"**{'PASS' if report.passes else 'FAIL'}** against a threshold of "
        f"{report.threshold:.1f} points of excess drift in the shipping configuration."
    )
    add("")
    if report.blind_mode_benefit:
        add(f"Blind mode removes {report.blind_mode_benefit:.2f} points of worst-case drift.")
        add("")

    add("## Limitations")
    add("")
    add(
        "- The names and institutions used are statistical proxies for demographic signal. "
        "They measure how the model responds to a signal, not how any real person would be "
        "treated."
    )
    add(
        "- Synthetic resumes. Real applications vary in ways this set does not capture, and "
        "real bias can interact with content in ways a controlled swap will not surface."
    )
    add(
        f"- {report.profiles_tested} profiles is a small sample. Drift below roughly "
        f"{max(report.noise_floor_max, 1.0):.1f} points cannot be distinguished from noise "
        f"at this size."
    )
    add(
        "- Absence of measured drift on these axes is not proof of fairness. It is evidence "
        "about these axes, this model, and this configuration."
    )
    add("")

    if report.warnings:
        add("## Warnings")
        add("")
        for warning in report.warnings:
            add(f"- {warning}")
        add("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AuditGateResult:
    passed: bool
    failures: list[str]
    notes: list[str]

    def render(self) -> str:
        lines = ["PASS" if self.passed else "FAIL"]
        lines += [f"  x {failure}" for failure in self.failures]
        lines += [f"  - {note}" for note in self.notes]
        return "\n".join(lines)


def check_audit(report: AuditReport) -> AuditGateResult:
    """Decide whether the measured bias is acceptable. Used by CI."""
    failures: list[str] = []
    notes: list[str] = []

    if not report.axes:
        return AuditGateResult(False, ["The audit produced no results."], [])

    if report.noise_floor_max > report.threshold:
        failures.append(
            f"Self-consistency noise ({report.noise_floor_max:.2f} pts) exceeds the drift "
            f"threshold ({report.threshold:.2f} pts). The audit cannot distinguish bias from "
            f"noise until the system is more stable."
        )

    for result in report.blind_results:
        excess = report.excess_drift(result)
        if excess > report.threshold:
            failures.append(
                f"{_label(result.axis)}: {result.max_drift:.2f} pts drift "
                f"({excess:.2f} above the noise floor) in the shipping configuration."
            )
        if result.must_have_flips:
            failures.append(
                f"{_label(result.axis)}: {result.must_have_flips} profile(s) changed must-have "
                f"compliance under a demographic swap. This changes a hiring decision."
            )
        elif result.rank_flips:
            notes.append(
                f"{_label(result.axis)}: {result.rank_flips} rank flip(s) without a score "
                f"breach. Worth watching."
            )

    for result in report.sighted_results:
        if report.excess_drift(result) > report.threshold:
            notes.append(
                f"{_label(result.axis)}: {result.max_drift:.2f} pts drift with blind mode OFF. "
                f"Diagnostic only, but it is why blind mode is on by default."
            )

    notes.extend(report.warnings)
    return AuditGateResult(not failures, failures, notes)
