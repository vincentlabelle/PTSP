# -*- coding: utf-8 -*-

import pytest

from core.ttest import IndepEqualVarTTestStatisticCalculator
from core.vector import Vector


def _almost_equal(result: float, expected: float, *, tolerance: float) -> bool:
    assert tolerance >= 0
    if expected == 0.:
        return abs(result - expected) <= tolerance
    return abs(result - expected) / expected <= tolerance


class TestIndepEqualVarTTestStatisticCalculator:

    @pytest.fixture(scope='class')
    def calculator(self):
        return IndepEqualVarTTestStatisticCalculator.make()

    def test_when_both_samples_are_empty(self, calculator):
        with pytest.raises(ValueError, match='must be non-empty'):
            calculator.calculate((
                Vector.empty(),
                Vector.empty()
            ))

    def test_when_sample1_is_empty(self, calculator):
        with pytest.raises(ValueError, match='must be non-empty'):
            calculator.calculate((
                Vector.from_sequence([1., 2., 3.]),
                Vector.empty()
            ))

    def test_when_sample_2_is_empty(self, calculator):
        with pytest.raises(ValueError, match='must be non-empty'):
            calculator.calculate((
                Vector.empty(),
                Vector.from_sequence([1., 2., 3.])
            ))

    def test_when_variance_is_zero(self, calculator):
        sample1 = Vector.from_sequence([1.])
        sample2 = Vector.from_sequence([2.])
        with pytest.raises(ValueError, match='pooled variance.*is 0'):
            calculator.calculate((sample1, sample2))

    def test_when_same_sample_size(self, calculator):
        sample1 = Vector.from_sequence([1., 2., 3.])
        sample2 = Vector.from_sequence([4., 5., 6.])
        result = calculator.calculate((sample1, sample2))
        expected = -3.6742346141747673
        assert _almost_equal(result, expected, tolerance=1e-8)

    def test_when_different_sample_size(self, calculator):
        sample1 = Vector.from_sequence([1., 2., 3.])
        sample2 = Vector.from_sequence([4., 5.])
        result = calculator.calculate((sample1, sample2))
        expected = -3.0
        assert _almost_equal(result, expected, tolerance=1e-8)
