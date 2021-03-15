# -*- coding: utf-8 -*-

import numpy as np
import pytest

from core.vector import Vector


class TestVectorInvariants:

    def test_immutable(self):
        data = np.ones((1,), dtype=np.float_)
        vector = Vector(data)
        data[0] = 0.
        assert vector.data != data

    @pytest.mark.parametrize('ndim', [0, 1, 2])
    def test_dimension(self, ndim: int):
        data = np.empty((1,) * ndim)
        if ndim != 1:
            with pytest.raises(ValueError, match='one dimension'):
                Vector(data)
        else:
            Vector(data)  # does not raise

    @pytest.mark.parametrize('dtype', [np.float_, np.float64, np.int_])
    def test_data_type(self, dtype):
        data = np.ones((3,), dtype=dtype)
        if dtype not in [np.float_, np.float64]:
            with pytest.raises(ValueError, match='float elements'):
                Vector(data)
        else:
            Vector(data)  # does not raise

    def test_when_all_finite(self):
        Vector(
            np.array([1., -1., 100., -0.25, 0.], dtype=np.float_)
        )  # does not raise

    @pytest.mark.parametrize('non_finite', [np.nan, np.inf, -np.inf])
    def test_when_one_non_finite(self, non_finite: float):
        with pytest.raises(ValueError, match='finite elements'):
            Vector(np.array([1., 2., non_finite], dtype=np.float_))

    def test_many_non_finite(self):
        with pytest.raises(ValueError, match='finite elements'):
            Vector(np.array([np.inf, 2., -np.nan], dtype=np.float_))


class TestVectorEqual:

    @pytest.fixture(scope='class')
    def vector(self) -> Vector:
        return Vector.from_sequence([1., 2., 3.])

    def test_when_equal(self, vector: Vector):
        other = Vector(vector.data)
        assert other == vector

    def test_when_different_size(self, vector: Vector):
        other = Vector.from_sequence([1., 2., 3., 4.])
        assert other != vector

    def test_when_different_values(self, vector: Vector):
        other = Vector.from_sequence([1., 2., 4.])
        assert other != vector

    def test_when_different_object(self, vector: Vector):
        assert vector != 'a'


class TestVectorHash:

    @pytest.fixture(scope='class')
    def vector(self) -> Vector:
        return Vector.from_sequence([1., 2., 3.])

    def test_when_equal(self, vector: Vector):
        other = Vector(vector.data)
        assert hash(other) == hash(vector)

    def test_when_different_size(self, vector: Vector):
        other = Vector.from_sequence([1., 2., 3., 4.])
        assert hash(other) != hash(vector)

    def test_when_different_values(self, vector: Vector):
        other = Vector.from_sequence([1., 2., 4.])
        assert hash(other) != hash(vector)


class TestVectorStringRepresentation:

    @pytest.fixture(scope='class')
    def vector(self) -> Vector:
        return Vector.from_sequence([1., 2., 3.])

    def test_str(self, vector: Vector):
        assert str(vector) == str(vector.data.tolist())

    def test_repr(self, vector: Vector):
        assert repr(vector) == f'<{vector.__class__.__name__}{vector}>'


class TestVectorAlternativeConstructors:

    def test_concatenate_when_empty(self):
        result = Vector.concatenate(())
        assert result == Vector.empty()

    def test_concatenate_when_one_empty(self):
        result = Vector.concatenate(
            iter((
                Vector.empty(),
            ))
        )
        assert result == Vector.empty()

    def test_concatenate_when_one_non_empty(self):
        to_concatenate = Vector.from_sequence([1., 2., 3.])
        result = Vector.concatenate(
            iter((
                to_concatenate,
            ))
        )
        assert result == to_concatenate

    def test_concatenate_when_multiple_and_all_empty(self):
        result = Vector.concatenate(
            iter((
                Vector.empty(),
                Vector.empty()
            ))
        )
        assert result == Vector.empty()

    def test_concatenate_when_multiple_and_any_non_empty(self):
        result = Vector.concatenate(
            iter((
                Vector.empty(),
                Vector.from_sequence([1., 2., 3.]),
                Vector.empty()
            ))
        )
        expected = Vector.from_sequence([1., 2., 3.])
        assert result == expected

    def test_concatenate_when_multiple_non_empty(self):
        result = Vector.concatenate(
            iter((
                Vector.empty(),
                Vector.from_sequence([1., 2., 3.]),
                Vector.empty(),
                Vector.from_sequence([4., 5.])
            ))
        )
        expected = Vector.from_sequence([1., 2., 3., 4., 5.])
        assert result == expected

    def test_empty(self):
        result = Vector.empty()
        expected = Vector.from_sequence(np.empty((0,), dtype=np.float_))
        assert result == expected

    def test_from_sequence_when_empty(self):
        result = Vector.from_sequence([])
        expected = Vector(np.array([], dtype=np.float_))
        assert result == expected

    def test_from_sequence_when_one_element(self):
        sequence = [1.]
        result = Vector.from_sequence(sequence)
        expected = Vector(np.array(sequence, dtype=np.float_))
        assert result == expected

    def test_from_sequence_when_many_elements(self):
        sequence = [1., 2., 3.]
        result = Vector.from_sequence(sequence)
        expected = Vector(np.array(sequence, dtype=np.float_))
        assert result == expected

    def test_from_sequence_when_supports_float(self):
        class _SupportsFloat:
            def __float__(self):
                return 1.

        result = Vector.from_sequence([_SupportsFloat()])
        expected = Vector(np.array([1.], dtype=np.float_))
        assert result == expected

    @pytest.mark.parametrize('non_finite', [np.nan, np.inf, -np.inf])
    def test_from_sequence_when_non_finite(self, non_finite: float):
        with pytest.raises(ValueError, match='finite elements'):
            Vector.from_sequence([1., non_finite])

    def test_from_sequence_when_tuple(self):
        sequence = (1.,)
        result = Vector.from_sequence(sequence)
        expected = Vector(np.array(sequence, dtype=np.float_))
        assert result == expected


class TestVectorProperties:

    @pytest.fixture(scope='class')
    def data(self) -> np.ndarray:
        return np.array([1., 2., 3.], dtype=np.float_)

    @pytest.fixture(scope='class')
    def vector(self, data) -> Vector:
        return Vector(data)

    def test_data(self, vector: Vector, data: np.ndarray):
        assert np.array_equal(vector.data, data)

    def test_set_data(self, vector: Vector, data: np.ndarray):
        with pytest.raises(AttributeError):
            vector.data = data

    def test_size(self, vector: Vector, data: np.ndarray):
        assert vector.size == data.size

    def test_set_size(self, vector: Vector, data: np.ndarray):
        with pytest.raises(AttributeError):
            vector.size = data.size


class TestVectorIsEmpty:

    def test_when_empty(self):
        vector = Vector.from_sequence([])
        assert vector.is_empty()

    def test_when_non_empty(self):
        vector = Vector.from_sequence([1.])
        assert not vector.is_empty()


class TestVectorSplit:

    @pytest.fixture(scope='class')
    def vector(self) -> Vector:
        return Vector.from_sequence([1., 2., 3.])

    @pytest.mark.parametrize('index', [-1, 0, 1])
    def test_when_empty(self, index: int):
        vector = Vector.empty()
        result = vector.split(index)
        expected = (
            Vector.empty(),
            Vector.empty()
        )
        assert result == expected

    def test_when_non_empty_and_index_negative_and_greater_than_size(
            self,
            vector: Vector
    ):
        result = vector.split(-5)
        expected = (
            Vector.empty(),
            Vector.from_sequence([1., 2., 3.])
        )
        assert result == expected

    def test_when_non_empty_and_index_negative_and_equal_to_size(
            self,
            vector: Vector
    ):
        result = vector.split(-3)
        expected = (
            Vector.empty(),
            Vector.from_sequence([1., 2., 3.])
        )
        assert result == expected

    def test_when_non_empty_and_index_negative_and_lower_than_size(
            self,
            vector: Vector
    ):
        result = vector.split(-1)
        expected = (
            Vector.from_sequence([1., 2.]),
            Vector.from_sequence([3.])
        )
        assert result == expected

    def test_when_non_empty_and_index_is_zero(self, vector: Vector):
        result = vector.split(0)
        expected = (
            Vector.empty(),
            vector
        )
        assert result == expected

    def test_when_non_empty_and_index_positive_and_lower_than_size(
            self,
            vector: Vector
    ):
        result = vector.split(1)
        expected = (
            Vector.from_sequence([1.]),
            Vector.from_sequence([2., 3.])
        )
        assert result == expected

    def test_when_non_empty_and_index_positive_and_equal_to_size(
            self,
            vector: Vector
    ):
        result = vector.split(3)
        expected = (
            vector,
            Vector.empty()
        )
        assert result == expected

    def test_when_non_empty_and_index_positive_and_greater_than_size(
            self,
            vector: Vector
    ):
        result = vector.split(4)
        expected = (
            vector,
            Vector.empty()
        )
        assert result == expected
