# -*- coding: utf-8 -*-

from itertools import permutations
from math import isfinite
from math import nan, inf

import numpy as np
import pytest
from numpy.random import PCG64

from core.random import NumpyNormalGenerator
from core.random import NumpyRandomPermutator
from core.vector import Vector


def _almost_equal(result: float, expected: float, *, tolerance: float) -> bool:
    assert tolerance >= 0
    if expected == 0.:
        return abs(result - expected) <= tolerance
    return abs(result - expected) / expected <= tolerance


class TestNumpyNormalGeneratorGenerate:

    @pytest.fixture(scope='function')
    def generator(self) -> NumpyNormalGenerator:
        return NumpyNormalGenerator(
            PCG64(seed=1234)  # stateful
        )

    @pytest.mark.parametrize('size', [-2, -1, 0, 1, 2])
    def test_size_validity(
            self,
            generator: NumpyNormalGenerator,
            size: int
    ):
        if size <= 0:
            with pytest.raises(
                    ValueError,
                    match='size must be strictly positive'
            ):
                generator.generate(size=size, mean=0., scale=1.)
        else:
            generator.generate(size=size, mean=0., scale=1.)  # does not raise

    @pytest.mark.parametrize('mean', [-inf, -1e2, 0., 1., 1e6, inf, nan])
    def test_mean_validity(
            self,
            generator: NumpyNormalGenerator,
            mean: float
    ):
        if not isfinite(mean):
            with pytest.raises(ValueError, match='mean must be finite'):
                generator.generate(size=1, mean=mean, scale=1.)
        else:
            generator.generate(size=1, mean=mean, scale=1.)  # does not raise

    @pytest.mark.parametrize('scale', [-1., 0., 1., 1e6, inf, nan])
    def test_scale_validity(
            self,
            generator: NumpyNormalGenerator,
            scale: float
    ):
        if scale < 0.:
            with pytest.raises(ValueError, match='scale must be non-negative'):
                generator.generate(size=1, mean=0., scale=scale)
        elif not isfinite(scale):
            with pytest.raises(ValueError, match='scale must be finite'):
                generator.generate(size=1, mean=0, scale=scale)
        else:
            generator.generate(size=1, mean=0, scale=scale)  # does not raise

    @pytest.mark.parametrize('size', [1, 2, 3])
    def test_size_parameter_passing(
            self,
            generator: NumpyNormalGenerator,
            size: int
    ):
        result = generator.generate(size=size, mean=0., scale=1.)
        assert result.size == size

    @pytest.mark.parametrize('mean', [-1., 0., 1.])
    def test_mean_parameter_passing(
            self,
            generator: NumpyNormalGenerator,
            mean: float
    ):
        result = generator.generate(size=int(1e6), mean=mean, scale=1.)
        mle = np.mean(result.data)
        assert _almost_equal(mle, mean, tolerance=1e-2)

    @pytest.mark.parametrize('scale', [0., 1., 2.])
    def test_scale_parameter_passing(
            self,
            generator: NumpyNormalGenerator,
            scale: float
    ):
        result = generator.generate(size=int(1e6), mean=0., scale=scale)
        mle = np.std(result.data, ddof=0)
        assert _almost_equal(mle, scale, tolerance=1e-2)


class TestNumpyRandomPermutatorPermute:

    @pytest.fixture(scope='function')
    def permutator(self) -> NumpyRandomPermutator:
        return NumpyRandomPermutator(
            PCG64(seed=1234)  # stateful!
        )

    def test_when_is_empty(self, permutator: NumpyRandomPermutator):
        vector = Vector.empty()
        result = permutator.permute(vector)
        assert result == vector

    def test_when_is_of_size_one(self, permutator: NumpyRandomPermutator):
        vector = Vector.from_sequence([1.])
        result = permutator.permute(vector)
        assert result == vector

    def test_when_is_of_size_greater_than_one(
            self,
            permutator: NumpyRandomPermutator
    ):
        vector = Vector.from_sequence([1., 2., 3.])
        all_permutations = tuple(
            Vector.from_sequence(permutation)
            for permutation in permutations(vector.data)
        )
        first = permutator.permute(vector)
        second = permutator.permute(vector)
        assert first in all_permutations
        assert second in all_permutations
        assert first != second
