# -*- coding: utf-8 -*-

from typing import Iterable

import pytest

from core.ttest import UnpairedSimilarVarTTestStatisticCalculator
from core.variance import IPooledVarianceCalculator
from core.variance import UnbiasedPooledVarianceCalculator
from core.vector import Vector


class TestUnpairedSimilarVarTTestStatisticCalculatorAlternativeConstructors:

    def test_make(self):
        calculator = UnpairedSimilarVarTTestStatisticCalculator.make()
        assert isinstance(
            calculator.calculator,
            UnbiasedPooledVarianceCalculator
        )


class TestUnpairedSimilarVarTTestStatisticCalculatorProperties:

    @pytest.fixture(scope='class')
    def sub(self) -> IPooledVarianceCalculator:
        return IPooledVarianceCalculator()

    @pytest.fixture(scope='class')
    def calculator(
            self,
            sub: IPooledVarianceCalculator
    ) -> UnpairedSimilarVarTTestStatisticCalculator:
        return UnpairedSimilarVarTTestStatisticCalculator(sub)

    def test_calculator(
            self,
            calculator: UnpairedSimilarVarTTestStatisticCalculator,
            sub: IPooledVarianceCalculator
    ):
        assert calculator.calculator is sub

    def test_set_calculator(
            self,
            calculator: UnpairedSimilarVarTTestStatisticCalculator,
            sub: IPooledVarianceCalculator
    ):
        with pytest.raises(AttributeError):
            calculator.calculator = sub


class _ZeroPooledVarianceCalculatorStub(IPooledVarianceCalculator):

    def calculate(self, samples: Iterable[Vector]) -> float:
        return 0.


class _NonZeroPooledVarianceCalculatorStub(IPooledVarianceCalculator):

    def calculate(self, samples: Iterable[Vector]) -> float:
        return 6.4


class TestUnpairedSimilarVarTTestStatisticCalculatorCalculate:

    def test_when_variance_is_zero(self):
        calculator = UnpairedSimilarVarTTestStatisticCalculator(
            _ZeroPooledVarianceCalculatorStub()
        )
        with pytest.raises(ValueError, match='pooled variance.*is 0'):
            calculator.calculate((
                Vector.from_sequence([1., 2.]),
                Vector.from_sequence([1.])
            ))

    def test_when_variance_is_non_zero(self):
        calculator = UnpairedSimilarVarTTestStatisticCalculator(
            _NonZeroPooledVarianceCalculatorStub()
        )
        result = calculator.calculate((
            Vector.from_sequence([5., 2., 3., 4., 5., 6., 7., 8.]),
            Vector.from_sequence([2., 2.])
        ))
        assert result == 1.5
