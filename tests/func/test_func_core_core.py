# -*- coding: utf-8 -*-

import pytest

from core import UnpairedOneSidedPermutationTestPowerSimulator


def _almost_equal(result: float, expected: float, *, tolerance: float) -> bool:
    assert tolerance >= 0
    if expected == 0.:
        return abs(result - expected) <= tolerance
    return abs(result - expected) / expected <= tolerance


class TestUnpairedOneSidedPermutationTestPowerSimulator:

    @pytest.fixture(scope='function')
    def simulator(self):
        return UnpairedOneSidedPermutationTestPowerSimulator.make(
            seed=1234
        )  # stateful!

    def test(self, simulator):
        result = simulator.simulate(
            number_of_simulations=300,
            number_of_permutations=300,
            number_of_observations=50,
            means=(0.5, 0.),
            scale=1.,
            alpha=0.025
        )
        assert _almost_equal(result, 0.6968888, tolerance=2e-2)
