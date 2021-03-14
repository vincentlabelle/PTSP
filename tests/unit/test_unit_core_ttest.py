# -*- coding: utf-8 -*-

from typing import Iterable

import pytest

from core.ttest import IndepEqualVarTTestStatisticCalculator
from core.variance import IPooledVarianceCalculator
from core.variance import UnbiasedPooledVarianceCalculator
from core.vector import Vector


class TestIndepEqualVarTTestStatisticCalculatorAlternativeConstructors:

    def test_make(self):
        calculator = IndepEqualVarTTestStatisticCalculator.make()
        assert isinstance(
            calculator.calculator,
            UnbiasedPooledVarianceCalculator
        )


class TestIndepEqualVarTTestStatisticCalculatorProperties:

    @pytest.fixture(scope='class')
    def sub(self) -> IPooledVarianceCalculator:
        return IPooledVarianceCalculator()

    @pytest.fixture(scope='class')
    def calculator(
            self,
            sub: IPooledVarianceCalculator
    ) -> IndepEqualVarTTestStatisticCalculator:
        return IndepEqualVarTTestStatisticCalculator(sub)

    def test_calculator(
            self,
            calculator: IndepEqualVarTTestStatisticCalculator,
            sub: IPooledVarianceCalculator
    ):
        assert calculator.calculator is sub

    def test_set_calculator(
            self,
            calculator: IndepEqualVarTTestStatisticCalculator,
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


class TestIndepEqualVarTTestStatisticCalculatorCalculate:

    def test_when_variance_is_zero(self):
        calculator = IndepEqualVarTTestStatisticCalculator(
            _ZeroPooledVarianceCalculatorStub()
        )
        with pytest.raises(ValueError, match='pooled variance.*is 0'):
            calculator.calculate((
                Vector.from_sequence([1., 2.]),
                Vector.from_sequence([1.])
            ))

    def test_when_variance_is_non_zero(self):
        calculator = IndepEqualVarTTestStatisticCalculator(
            _NonZeroPooledVarianceCalculatorStub()
        )
        result = calculator.calculate((
            Vector.from_sequence([5., 2., 3., 4., 5., 6., 7., 8.]),
            Vector.from_sequence([2., 2.])
        ))
        assert result == 1.5
