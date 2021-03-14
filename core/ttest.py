# -*- coding: utf-8 -*-

from typing import Tuple

import numpy as np

from .variance import IPooledVarianceCalculator
from .variance import UnbiasedPooledVarianceCalculator
from .vector import Vector


class ITwoSampleTTestStatisticCalculator:

    def calculate(self, samples: Tuple[Vector, Vector]) -> float:
        raise NotImplementedError


class IndepEqualVarTTestStatisticCalculator(ITwoSampleTTestStatisticCalculator):
    # independent two sample t-test test statistic calculator
    # assuming equal variances

    @classmethod
    def make(cls) -> "IndepEqualVarTTestStatisticCalculator":
        # public constructor!
        return cls(
            UnbiasedPooledVarianceCalculator.make()
        )

    def __init__(self, calculator: IPooledVarianceCalculator):
        # private!
        self._calculator = calculator

    @property
    def calculator(self) -> IPooledVarianceCalculator:
        # for testing!
        return self._calculator

    def calculate(self, samples: Tuple[Vector, Vector]) -> float:
        # will raise if any sample in `samples` is empty,
        # or if the unbiased pooled variance of the two samples
        # is exactly zero
        variance = self._calculator.calculate(samples)
        self._raise_if_variance_is_zero(variance)
        a, b = samples
        return (
                (np.mean(a.data) - np.mean(b.data))
                / np.sqrt(variance * (1. / a.size + 1. / b.size))
        )

    @staticmethod
    def _raise_if_variance_is_zero(variance: float):
        if variance == 0.:
            msg = (
                'cannot compute t-test test statistic, unbiased pooled '
                'variance of provided samples is 0'
            )
            raise ValueError(msg)
