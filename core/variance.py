# -*- coding: utf-8 -*-

from typing import Iterable
from typing import Tuple

import numpy as np

from .vector import Vector


class IPooledVarianceCalculator:

    def calculate(self, vectors: Iterable[Vector]) -> float:
        raise NotImplementedError


class UnbiasedPooledVarianceCalculator(IPooledVarianceCalculator):
    # unbiased least square estimate of pooled sample variance

    @classmethod
    def make(cls) -> "UnbiasedPooledVarianceCalculator":
        # public constructor!
        return cls(
            SampleVarianceCalculator()
        )

    def __init__(self, calculator: "ISampleVarianceCalculator"):
        # private!
        self._calculator = calculator

    @property
    def calculator(self) -> "ISampleVarianceCalculator":
        # for testing only!
        return self._calculator

    def calculate(self, samples: Iterable[Vector]) -> float:
        # will raise if `samples` is empty,
        # or any sample in `samples` is empty
        frozen = tuple(samples)
        self._raise_if_no_samples(frozen)
        if all(sample.size == 1 for sample in frozen):  # corner case!
            return 0.
        return (
                sum(
                    (sample.size - 1) * self._calculator.calculate(sample)
                    for sample in frozen
                )
                / sum(sample.size - 1 for sample in frozen)
        )

    @staticmethod
    def _raise_if_no_samples(frozen: Tuple[Vector, ...]):
        if len(frozen) == 0:
            msg = (
                f'expecting at least one sample, received none'
            )
            raise ValueError(msg)


class ISampleVarianceCalculator:

    def calculate(self, sample: Vector) -> float:
        raise NotImplementedError


class SampleVarianceCalculator(ISampleVarianceCalculator):
    # sample variance with Bessel's correction

    def calculate(self, sample: Vector) -> float:
        # will raise if `sample` is empty
        self._raise_if_sample_is_empty(sample)
        if sample.size == 1:  # corner case!
            return 0.
        return (
                np.sum(np.power(sample.data - np.mean(sample.data), 2))
                / (sample.size - 1)
        )

    @staticmethod
    def _raise_if_sample_is_empty(sample: Vector):
        if sample.is_empty():
            msg = 'sample must be non-empty'
            raise ValueError(msg)
