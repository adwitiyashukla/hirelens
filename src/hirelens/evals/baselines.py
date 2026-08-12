from __future__ import annotations

import random
from abc import ABC, abstractmethod

from hirelens.retrieve.hybrid import BM25, tokenize


class Baseline(ABC):
    name: str
    description: str

    @abstractmethod
    def score(self, job_text: str, resume_texts: list[str]) -> list[float]:
        pass


class RandomBaseline(Baseline):
    name = "random"
    description = "Uniform random scores, averaged over 200 seeds"

    def __init__(self, trials: int = 200, seed: int = 7) -> None:
        self.trials = trials
        self.seed = seed

    def score(self, job_text: str, resume_texts: list[str]) -> list[float]:
        rng = random.Random(self.seed)
        return [rng.random() for _ in resume_texts]

    def expected_correlation(self, human: list[float], statistic) -> tuple[float, float]:
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
    name = "keyword"
    description = "Share of distinct job-description terms appearing in the resume"

    def score(self, job_text: str, resume_texts: list[str]) -> list[float]:
        terms = set(tokenize(job_text))
        if not terms:
            return [0.0] * len(resume_texts)
        return [len(terms & set(tokenize(text))) / len(terms) for text in resume_texts]


class BM25Baseline(Baseline):
    name = "bm25"
    description = "BM25 with the job description as the query and each resume as a document"

    def score(self, job_text: str, resume_texts: list[str]) -> list[float]:
        if not resume_texts:
            return []
        return BM25(resume_texts).scores(job_text)


def all_baselines() -> list[Baseline]:
    return [RandomBaseline(), KeywordOverlapBaseline(), BM25Baseline()]
