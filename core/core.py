# -*- coding: utf-8 -*-

import numpy as np

from .variance import IPooledVarianceCalculator
from .variance import UnbiasedPooledVarianceCalculator
from .vector import Vector


class TwoSampleTTestStatisticCalculator:
    # independent two sample t-test test statistic calculator
    # assuming equal variances

    @classmethod
    def make(cls) -> "TwoSampleTTestStatisticCalculator":
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

    def calculate(self, sample1: Vector, sample2: Vector) -> float:
        # will raise if `sample1` or `sample2` is empty,
        # or if the unbiased pooled variance of the two samples
        # is exactly zero
        variance = self._calculator.calculate((sample1, sample2))
        self._raise_if_variance_is_zero(variance)
        return (
                (np.mean(sample1.data) - np.mean(sample2.data))
                / np.sqrt(variance * (1. / sample1.size + 1. / sample2.size))
        )

    @staticmethod
    def _raise_if_variance_is_zero(variance: float):
        if variance == 0.:
            msg = (
                'cannot compute t-test test statistic, unbiased pooled '
                'variance of provided samples is 0'
            )
            raise ValueError(msg)
