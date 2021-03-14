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
        cls._raise_if_is_not_strictly_positive(seed, name='seed')
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
        self._raise_if_is_not_strictly_positive(
            number_of_simulations,
            name='number_of_simulations'
        )
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
            simulated[i] = self._calculator.calculate(
                number_of_permutations,
                self._generate_samples(number_of_observations, means, scale)
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
    def _raise_if_is_not_strictly_positive(value: int, *, name: str):
        if value <= 0:
            msg = f'{name} must be strictly positive, was [{value}]'
            raise ValueError(msg)

    @staticmethod
    def _raise_if_is_not_between_zero_and_one(alpha: float):
        if not 0. <= alpha <= 1.:
            msg = f'alpha must be in [0, 1], was [{alpha}]'
            raise ValueError(msg)
