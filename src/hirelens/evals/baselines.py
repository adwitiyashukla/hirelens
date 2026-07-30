"""Trivial rankers to measure the real pipeline against.

A Spearman rho of 0.78 means nothing on its own. It only becomes evidence when you
know what a system with no intelligence in it would score on the same data. This
module is short and it is the most important honesty mechanism in the project.

Three baselines, weakest first:

**Random.** Averaged over many seeds, this is the floor. Any real system must
clear it by a wide margin, and if the confidence interval on the real system
overlaps the random band, the golden set is too small to support any claim at all.

**Keyword overlap.** Count how many job-description terms appear anywhere in the
resume. This is roughly what a naive applicant tracking system does, and it is
uncomfortably competitive on easy datasets, because a good candidate for a job
genuinely does tend to use that job's vocabulary.

**BM25 whole-document.** The job description as a query against the whole resume
as one document. A real information retrieval baseline: properly length-normalised
and IDF-weighted, no LLM anywhere.

If HireLens cannot beat BM25, the eight hundred lines of extraction, retrieval and
judging above it are not earning their place, and the right response is to say so
in the README rather than quietly omit the comparison. The whole reason to build
this module is to make that outcome visible if it happens.
"""

from __future__ import annotations

import random
from abc import ABC, abstractmethod

from hirelens.retrieve.hybrid import BM25, tokenize


class Baseline(ABC):
    """Ranks candidates for a job without using the pipeline."""

    name: str
    description: str

    @abstractmethod
    def score(self, job_text: str, resume_texts: list[str]) -> list[float]:
        """A comparable score per resume. Absolute scale does not matter, only order."""


class RandomBaseline(Baseline):
    """Scores at random. The floor any real system must clear."""

    name = "random"
    description = "Uniform random scores, averaged over 200 seeds"

    def __init__(self, trials: int = 200, seed: int = 7) -> None:
        self.trials = trials
        self.seed = seed

    def score(self, job_text: str, resume_texts: list[str]) -> list[float]:
        rng = random.Random(self.seed)
        return [rng.random() for _ in resume_texts]

    def expected_correlation(self, human: list[float], statistic) -> tuple[float, float]:
        """Mean and 95th percentile of the statistic under random ordering.

        The 95th percentile is the number that matters: it says how high a
        correlation pure chance produces on a set this small. On twelve
        candidates that is not close to zero, and quoting a result without
        knowing it is how a meaningless number gets published.
        """
        rng = random.Random(self.seed)
        values: list[float] = []
        for _ in range(self.trials):
            noise = [rng.random() for _ in human]
            if len(set(human)) > 1:
                values.append(statistic(noise, human))
        if not values:
            return (0.0, 0.0)
        values.sort()
        mean = sum(values) / len(values)
        return (mean, values[int(0.95 * (len(values) - 1))])


class KeywordOverlapBaseline(Baseline):
    """Fraction of distinct job-description terms present in the resume.

    Deliberately naive, and deliberately close to what a keyword-filtering
    applicant tracking system does. Candidate c04 in the golden set exists
    specifically to be over-scored by this: every required keyword is in the
    skills list with nothing behind any of them.
    """

    name = "keyword"
    description = "Share of distinct job-description terms appearing in the resume"

    def score(self, job_text: str, resume_texts: list[str]) -> list[float]:
        terms = set(tokenize(job_text))
        if not terms:
            return [0.0] * len(resume_texts)
        return [len(terms & set(tokenize(text))) / len(terms) for text in resume_texts]


class BM25Baseline(Baseline):
    """The job description as a query against each whole resume.

    The strongest baseline here and the one worth beating. It is a real IR method
    with length normalisation and IDF weighting, and it costs nothing to run.
    """

    name = "bm25"
    description = "BM25 with the job description as the query and each resume as a document"

    def score(self, job_text: str, resume_texts: list[str]) -> list[float]:
        if not resume_texts:
            return []
        return BM25(resume_texts).scores(job_text)


def all_baselines() -> list[Baseline]:
    return [RandomBaseline(), KeywordOverlapBaseline(), BM25Baseline()]
