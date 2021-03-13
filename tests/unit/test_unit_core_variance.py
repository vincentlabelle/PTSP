# -*- coding: utf-8 -*-

import pytest

from core.variance import ISampleVarianceCalculator
from core.variance import SampleVarianceCalculator
from core.variance import UnbiasedPooledVarianceCalculator
from core.vector import Vector


class TestUnbiasedPooledVarianceCalculatorAlternativeConstructors:

    def test_make(self):
        calculator = UnbiasedPooledVarianceCalculator.make()
        assert isinstance(calculator.calculator, SampleVarianceCalculator)


class TestUnbiasedPooledVarianceCalculatorProperties:

    @pytest.fixture(scope='class')
    def sub(self) -> ISampleVarianceCalculator:
        return ISampleVarianceCalculator()

    @pytest.fixture(scope='class')
    def calculator(
            self,
            sub: ISampleVarianceCalculator
    ) -> UnbiasedPooledVarianceCalculator:
        return UnbiasedPooledVarianceCalculator(sub)

    def test_calculator(
            self,
            calculator: UnbiasedPooledVarianceCalculator,
            sub: ISampleVarianceCalculator
    ):
        assert calculator.calculator is sub

    def test_set_calculator(
            self,
            calculator: UnbiasedPooledVarianceCalculator,
            sub: ISampleVarianceCalculator
    ):
        with pytest.raises(AttributeError):
            calculator.calculator = sub


class _SampleVarianceCalculatorStub(ISampleVarianceCalculator):

    def __init__(self):
        self._call_count = 0

    def calculate(self, sample: Vector) -> float:
        result = self._call_count
        self._call_count += 1
        return result


class TestUnbiasedPooledVarianceCalculatorCalculate:

    @pytest.fixture(scope='function')
    def calculator(self) -> UnbiasedPooledVarianceCalculator:
        return UnbiasedPooledVarianceCalculator(
            _SampleVarianceCalculatorStub()  # stateful!
        )

    def test_when_no_samples(
            self,
            calculator: UnbiasedPooledVarianceCalculator
    ):
        with pytest.raises(ValueError, match='at least one sample'):
            calculator.calculate(())

    def test_when_all_samples_are_of_size_one(
            self,
            calculator: UnbiasedPooledVarianceCalculator
    ):
        samples = iter((
            Vector.from_sequence([0.]),
            Vector.from_sequence([1.]),
            Vector.from_sequence([2.]),
            Vector.from_sequence([-1.])
        ))
        result = calculator.calculate(samples)
        assert result == 0.

    def test_when_any_sample_is_not_of_size_one(
            self,
            calculator: UnbiasedPooledVarianceCalculator
    ):
        samples = iter((
            Vector.from_sequence([0.]),
            Vector.from_sequence([1., 2., 3.]),
            Vector.from_sequence([4., 5., 6., 7.])
        ))
        result = calculator.calculate(samples)
        assert result == 1.6


class TestSampleVarianceCalculatorCalculate:

    @pytest.fixture(scope='class')
    def calculator(self) -> SampleVarianceCalculator:
        return SampleVarianceCalculator()

    def test_when_empty(self, calculator: SampleVarianceCalculator):
        sample = Vector.empty()
        with pytest.raises(ValueError, match='must be non-empty'):
            calculator.calculate(sample)

    @pytest.mark.parametrize('element', [-1., 0., 2.])
    def test_when_one_element(
            self,
            calculator: SampleVarianceCalculator,
            element: float
    ):
        sample = Vector.from_sequence([element])
        result = calculator.calculate(sample)
        assert result == 0.

    @pytest.mark.parametrize(
        'sample',
        [
            Vector.from_sequence([0., 0., 0.]),
            Vector.from_sequence([2., 2., 2.])
        ]
    )
    def test_when_variance_is_zero(
            self,
            calculator: SampleVarianceCalculator,
            sample: Vector
    ):
        result = calculator.calculate(sample)
        assert result == 0.

    def test_when_variance_is_non_zero(
            self,
            calculator: SampleVarianceCalculator
    ):
        sample = Vector.from_sequence([3., -2., 5.])
        result = calculator.calculate(sample)
        assert result == 13.
