from __future__ import annotations

import math
import random
from dataclasses import dataclass

_BOOTSTRAP_SAMPLES = 2000

_BOOTSTRAP_SEED = 20260728


@dataclass(frozen=True, slots=True)
class Estimate:
    value: float
    low: float
    high: float
    n: int

    @property
    def width(self) -> float:
        return self.high - self.low

    def __str__(self) -> str:
        return f"{self.value:.3f} [{self.low:.2f}, {self.high:.2f}]"

    def format(self, digits: int = 3) -> str:
        return f"{self.value:.{digits}f} [{self.low:.{digits - 1}f}, {self.high:.{digits - 1}f}]"


def rank_with_ties(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    position = 0

    while position < len(order):
        end = position
        while end + 1 < len(order) and values[order[end + 1]] == values[order[position]]:
            end += 1
        average = (position + end) / 2 + 1
        for index in range(position, end + 1):
            ranks[order[index]] = average
        position = end + 1

    return ranks


def pearson(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n
    dx = [x - mean_x for x in xs]
    dy = [y - mean_y for y in ys]

    numerator = sum(a * b for a, b in zip(dx, dy, strict=True))
    denominator = math.sqrt(sum(a * a for a in dx) * sum(b * b for b in dy))
    return numerator / denominator if denominator else 0.0


def spearman(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    return pearson(rank_with_ties(xs), rank_with_ties(ys))


def kendall_tau_b(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0

    concordant = discordant = tied_x = tied_y = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            product = dx * dy
            if product > 0:
                concordant += 1
            elif product < 0:
                discordant += 1
            else:
                if dx == 0:
                    tied_x += 1
                if dy == 0:
                    tied_y += 1

    denominator = math.sqrt((concordant + discordant + tied_x) * (concordant + discordant + tied_y))
    return (concordant - discordant) / denominator if denominator else 0.0


def inversion_rate(xs: list[float], ys: list[float]) -> float:
    n = len(xs)
    comparable = inverted = 0

    for i in range(n):
        for j in range(i + 1, n):
            if ys[i] == ys[j]:
                continue
            comparable += 1
            if (xs[i] - xs[j]) * (ys[i] - ys[j]) < 0:
                inverted += 1

    return inverted / comparable if comparable else 0.0


def top_k_precision(system: list[float], human: list[float], k: int = 3) -> float:
    if not system or k <= 0:
        return 0.0

    k = min(k, len(system))
    ours = sorted(range(len(system)), key=lambda i: -system[i])[:k]
    cutoff = sorted(human, reverse=True)[k - 1]
    return sum(1 for i in ours if human[i] >= cutoff) / k


def mean_absolute_error(xs: list[float], ys: list[float]) -> float:
    if not xs:
        return 0.0
    return sum(abs(a - b) for a, b in zip(xs, ys, strict=True)) / len(xs)


def bootstrap(
    xs: list[float],
    ys: list[float],
    statistic,
    *,
    samples: int = _BOOTSTRAP_SAMPLES,
    confidence: float = 0.95,
    seed: int = _BOOTSTRAP_SEED,
) -> Estimate:
    n = len(xs)
    point = statistic(xs, ys)
    if n < 3:
        return Estimate(value=point, low=-1.0, high=1.0, n=n)

    rng = random.Random(seed)
    values: list[float] = []

    for _ in range(samples):
        indices = [rng.randrange(n) for _ in range(n)]
        resampled_x = [xs[i] for i in indices]
        resampled_y = [ys[i] for i in indices]
        if len(set(resampled_y)) < 2 or len(set(resampled_x)) < 2:
            continue
        values.append(statistic(resampled_x, resampled_y))

    if not values:
        return Estimate(value=point, low=-1.0, high=1.0, n=n)

    values.sort()
    tail = (1.0 - confidence) / 2
    low = values[max(0, int(tail * len(values)))]
    high = values[min(len(values) - 1, int((1 - tail) * len(values)))]
    return Estimate(value=point, low=low, high=high, n=n)


def spearman_ci(xs: list[float], ys: list[float], **kwargs) -> Estimate:
    return bootstrap(xs, ys, spearman, **kwargs)


def kendall_ci(xs: list[float], ys: list[float], **kwargs) -> Estimate:
    return bootstrap(xs, ys, kendall_tau_b, **kwargs)


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]

    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


@dataclass(frozen=True, slots=True)
class Distribution:
    mean: float
    p50: float
    p95: float
    minimum: float
    maximum: float
    n: int

    @classmethod
    def of(cls, values: list[float]) -> Distribution:
        if not values:
            return cls(0.0, 0.0, 0.0, 0.0, 0.0, 0)
        return cls(
            mean=sum(values) / len(values),
            p50=percentile(values, 0.5),
            p95=percentile(values, 0.95),
            minimum=min(values),
            maximum=max(values),
            n=len(values),
        )
