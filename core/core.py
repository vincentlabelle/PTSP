# -*- coding: utf-8 -*-
# TODO tests

from typing import Tuple

import numpy as np
from numpy.random import PCG64

from .permutation import IOneSidedPermutationPValueCalculator
from .permutation import OneSidedPermutationPValueCalculator
from .random import INormalRandomGenerator
from .random import NumpyNormalGenerator
from .random import NumpyRandomPermutator
from .ttest import IndepEqualVarTTestStatisticCalculator
from .vector import Vector


class UnpairedOneSidedPermutationTestPowerSimulator:
    # Calculates power of one-sided permutation test on unpaired
    # samples with equal variance and equal number of observations

    @classmethod
    def make(
            cls,
            *,
            seed: int
    ) -> "UnpairedOneSidedPermutationTestPowerSimulator":
        # public constructor!
        # will raise if `seed` is negative
        cls._raise_if_is_negative(seed)
        generator = PCG64(seed=seed)
        return cls(
            OneSidedPermutationPValueCalculator.make(
                IndepEqualVarTTestStatisticCalculator.make(),
                NumpyRandomPermutator(generator)
            ),
            NumpyNormalGenerator(generator)
        )

    def __init__(
            self,
            calculator: IOneSidedPermutationPValueCalculator,
            generator: INormalRandomGenerator
    ):
        # private!
        self._calculator = calculator
        self._generator = generator

    @property
    def calculator(self) -> IOneSidedPermutationPValueCalculator:
        # for testing!
        return self._calculator

    @property
    def generator(self) -> INormalRandomGenerator:
        # for testing!
        return self._generator

    def simulate(
            self,
            *,
            number_of_simulations: int,
            number_of_permutations: int,
            number_of_observations: int,
            means: Tuple[float, float],
            scale: float,
            alpha: float
    ) -> float:
        # will raise if `number_of_simulations` or `number_of_permutations`
        # is not strictly positive,
        # if `number_of_observations` is not at least two (ie. 2),
        # if `alpha` is not in [0, 1],
        # if any mean in `means` or `scale` is not finite,
        # or if `scale` is negative
        self._raise_if_is_not_strictly_positive(number_of_simulations)
        self._raise_if_is_not_at_least_two(number_of_observations)
        self._raise_if_is_not_between_zero_and_one(alpha)
        simulated = self._do_simulations(
            number_of_simulations,
            number_of_permutations,
            number_of_observations,
            means,
            scale
        )
        return np.mean(simulated < alpha)

    def _do_simulations(
            self,
            number_of_simulations: int,
            number_of_permutations: int,
            number_of_observations: int,
            means: Tuple[float, float],
            scale: float
    ) -> np.ndarray:
        simulated = np.empty((number_of_simulations,), dtype=np.float_)
        for i in range(simulated.size):
            samples = self._generate_samples(
                number_of_observations,
                means,
                scale
            )
            simulated[i] = self._calculator.calculate(
                number_of_permutations,
                samples
            )
        return simulated

    def _generate_samples(
            self,
            number_of_observations: int,
            means: Tuple[float, float],
            scale: float
    ) -> Tuple[Vector, Vector]:
        return (
            self._generator.generate(
                size=number_of_observations,
                mean=means[0],
                scale=scale
            ),
            self._generator.generate(
                size=number_of_observations,
                mean=means[1],
                scale=scale
            )
        )

    @staticmethod
    def _raise_if_is_negative(seed: int):
        if seed < 0:
            msg = f'seed must be non-negative, was [{seed}]'
            raise ValueError(msg)

    @staticmethod
    def _raise_if_is_not_strictly_positive(number_of_simulations: int):
        if number_of_simulations <= 0:
            msg = (
                f'number_of_simulations must be strictly positive, '
                f'was [{number_of_simulations}]'
            )
            raise ValueError(msg)

    @staticmethod
    def _raise_if_is_not_at_least_two(number_of_observations: int):
        if number_of_observations <= 1:
            msg = (
                f'number_of_observations must be at least 2, '
                f'was [{number_of_observations}]'
            )
            raise ValueError(msg)

    @staticmethod
    def _raise_if_is_not_between_zero_and_one(alpha: float):
        if not 0. <= alpha <= 1.:
            msg = f'alpha must be in [0, 1], was [{alpha}]'
            raise ValueError(msg)
