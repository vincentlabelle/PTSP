# -*- coding: utf-8 -*-
# TODO tests

from typing import Tuple

import numpy as np

from .random import IRandomPermutator
from .ttest import ITwoSampleTTestStatisticCalculator
from .vector import Vector


class IOneSidedPermutationPValueCalculator:

    def calculate(
            self,
            number_of_permutations: int,
            samples: Tuple[Vector, Vector]
    ) -> float:
        raise NotImplementedError


class OneSidedPermutationPValueCalculator(
    IOneSidedPermutationPValueCalculator
):
    # Calculator of p-value for a one-sided permutation test on two
    # samples

    @classmethod
    def make(
            cls,
            calculator: ITwoSampleTTestStatisticCalculator,
            permutator: IRandomPermutator
    ) -> "OneSidedPermutationPValueCalculator":
        # public constructor!
        return cls(
            TwoSamplePermutator(
                calculator,
                permutator
            )
        )

    def __init__(self, permutator: "ITwoSamplePermutator"):
        # private!
        self._permutator = permutator

    @property
    def permutator(self) -> "ITwoSamplePermutator":
        # for testing!
        return self._permutator

    def calculate(
            self,
            number_of_permutations: int,
            samples: Tuple[Vector, Vector]
    ) -> float:
        # will raise if `number_of_permutations` is not strictly positive,
        # if it is impossible to compute the test statistic for `samples`
        # (ie. the test denominator is zero or any sample is empty),
        # or if it is impossible to compute the test statistic for all
        # permutations (unlikely)
        observed, permuted = self._permutator.permute(
            number_of_permutations,
            samples
        )
        return np.mean(permuted.data > observed)


class ITwoSamplePermutator:

    def permute(
            self,
            number_of_permutations: int,
            samples: Tuple[Vector, Vector]
    ) -> Vector:
        raise NotImplementedError


class TwoSamplePermutator(ITwoSamplePermutator):
    # Performs the permutations required by a permutation test
    # on two samples

    def __init__(
            self,
            calculator: ITwoSampleTTestStatisticCalculator,
            permutator: IRandomPermutator
    ):
        self._calculator = calculator
        self._permutator = permutator

    def permute(
            self,
            number_of_permutations: int,
            samples: Tuple[Vector, Vector]
    ) -> Tuple[float, Vector]:
        # will raise if `number_of_permutations` is not strictly positive,
        # if it is impossible to compute the test statistic for `samples`
        # (ie. the test denominator is zero or any sample is empty),
        # or if it is impossible to compute the test statistic for all
        # permutations (unlikely)
        self._raise_if_number_of_permutations_is_not_strictly_positive(
            number_of_permutations
        )
        observed = self._calculator.calculate(samples)
        permuted = self._do_permutations(number_of_permutations, samples)
        self._raise_runtime_if_permuted_is_empty(permuted)
        return observed, permuted

    def _do_permutations(
            self,
            number_of_permutations: int,
            samples: Tuple[Vector, Vector]
    ) -> Vector:
        concatenated = Vector.concatenate(samples)
        permuted = np.empty((number_of_permutations,), dtype=np.float_)
        for i in range(permuted.size):
            permuted[i] = self._try_to_calculate_permuted_statistic(
                concatenated,
                samples[0].size
            )
        return Vector(permuted[~np.isnan(permuted)])

    def _try_to_calculate_permuted_statistic(
            self,
            concatenated: Vector,
            size: int
    ) -> float:
        # returns nan if statistic cannot be computed as in R
        shuffled = self._permutator.permute(concatenated)
        try:
            permuted = self._calculator.calculate(shuffled.split(size))
        except ValueError:
            permuted = np.nan
        return permuted

    @staticmethod
    def _raise_if_number_of_permutations_is_not_strictly_positive(
            number_of_permutations: int
    ):
        if number_of_permutations <= 0:
            msg = 'number of permutations must be strictly positive'
            raise ValueError(msg)

    @staticmethod
    def _raise_runtime_if_permuted_is_empty(permuted: Vector):
        # unlikely! should we do something else? would verify in practice
        if permuted.is_empty():
            msg = (
                'unable to generate permutations with non-nan '
                't-test statistic'
            )
            raise RuntimeError(msg)
