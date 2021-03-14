# -*- coding: utf-8 -*-

from typing import Iterable, Sequence
from typing import SupportsFloat
from typing import Tuple

import numpy as np
from numba import njit
from numpy import ndarray


class Vector:
    # wrapper of one-dimensional np.ndarray of finite floats

    @classmethod
    def concatenate(cls, vectors: Iterable["Vector"]) -> "Vector":
        frozen = tuple(vector.data for vector in vectors)
        if len(frozen) == 0:  # corner-case!
            return Vector.empty()
        return cls(np.concatenate(frozen))

    @classmethod
    def empty(cls) -> "Vector":
        return cls.from_sequence([])

    @classmethod
    def from_sequence(cls, sequence: Sequence[SupportsFloat]) -> "Vector":
        # will raise if `sequence` contains non-finite elements
        return cls(
            np.array(sequence, dtype=np.float_)
        )

    def __init__(self, data: ndarray):
        # will raise if `data` has not only and only one dimension,
        # if the data type of `data` is not one of floating point numbers,
        # or if any element in `data` is not finite
        self._data = data.copy()
        self._data.flags.writeable = False  # immutable!
        self._raise_if_is_not_one_dimension()
        self._raise_if_is_not_float()
        self._raise_if_is_not_finite()

    @property
    def data(self) -> ndarray:
        return self._data

    @property
    def size(self) -> int:
        return self._data.size

    def is_empty(self) -> int:
        return self.size == 0

    def split(self, index: int) -> Tuple["Vector", "Vector"]:
        a, b = self._split(self._data, index)
        return self.__class__(a), self.__class__(b)

    def __repr__(self) -> str:
        return f'<{self.__class__.__name__}{self}>'

    def __str__(self) -> str:
        return str(self._data.tolist())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Vector):
            return NotImplemented
        return np.array_equal(self._data, other._data)

    def __hash__(self) -> int:
        return hash(self.data.tobytes())

    def _raise_if_is_not_one_dimension(self):
        if self._data.ndim != 1:
            msg = (
                f'data must have one dimension, the number of dimension'
                f'of data was [{self._data.ndim}]'
            )
            raise ValueError(msg)

    def _raise_if_is_not_float(self):
        if self._data.dtype != np.float_:
            msg = (
                f'data must contain float elements, the data type of '
                f'data was [{self._data.dtype}]'
            )
            raise ValueError(msg)

    def _raise_if_is_not_finite(self):
        if not self._all_finite(self._data):
            msg = (
                f'data must contain finite elements, some elements in '
                f'data were not finite'
            )
            raise ValueError(msg)

    @staticmethod
    @njit(cache=True)
    def _all_finite(data: ndarray) -> bool:
        return np.all(np.isfinite(data))

    @staticmethod
    @njit(cache=True)
    def _split(data: ndarray, index: int) -> Tuple[ndarray, ndarray]:
        return np.split(data, (index,))
