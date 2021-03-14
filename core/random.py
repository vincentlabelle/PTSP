# -*- coding: utf-8 -*-

from math import isfinite

from numpy.random import BitGenerator, Generator

from .vector import Vector


class INormalRandomGenerator:

    def generate(
            self,
            *,
            size: int,
            mean: float,
            scale: float
    ) -> Vector:
        raise NotImplementedError


class NumpyNormalGenerator(INormalRandomGenerator):

    def __init__(self, generator: BitGenerator):
        self._generator: Generator = Generator(generator)

    def generate(
            self,
            *,
            size: int,
            mean: float,
            scale: float
    ) -> Vector:
        # will raise if `size` is not strictly positive,
        # if `mean` or `scale` is not finite,
        # or if `scale` is negative
        self._raise_if_size_is_not_strictly_positive(size)
        self._raise_if_mean_is_non_finite(mean)
        self._raise_if_scale_is_negative(scale)
        self._raise_if_scale_is_non_finite(scale)
        return Vector(
            self._generator.normal(loc=mean, scale=scale, size=size)
        )

    @staticmethod
    def _raise_if_size_is_not_strictly_positive(size: int):
        if size <= 0:
            msg = f'size must be strictly positive, was [{size}]'
            raise ValueError(msg)

    @staticmethod
    def _raise_if_mean_is_non_finite(mean: float):
        _raise_if_is_non_finite(mean, name='mean')

    @staticmethod
    def _raise_if_scale_is_negative(scale: float):
        if scale < 0.:
            msg = f'scale must be non-negative, was [{scale}]'
            raise ValueError(msg)

    @staticmethod
    def _raise_if_scale_is_non_finite(scale: float):
        _raise_if_is_non_finite(scale, name='scale')


def _raise_if_is_non_finite(value: float, *, name: str):
    if not isfinite(value):
        msg = f'{name} must be finite, was [{value}]'
        raise ValueError(msg)


class IRandomPermutator:

    def permute(self, vector: Vector) -> Vector:
        raise NotImplementedError


class NumpyRandomPermutator(IRandomPermutator):

    def __init__(self, generator: BitGenerator):
        self._generator: Generator = Generator(generator)

    def permute(self, vector: Vector) -> Vector:
        return Vector(
            self._generator.permutation(vector.data)
        )
