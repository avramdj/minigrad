import numpy as np

from minigrad.engine.tensor import Tensor


def test_add():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    c = a + b
    assert np.allclose(c.to_numpy(), [5, 7, 9])


def test_add_cuda():
    a = Tensor([1.0, 2.0, 3.0], device="cuda")
    b = Tensor([4.0, 5.0, 6.0], device="cuda")
    c = a + b
    assert np.allclose(c.to_numpy(), [5, 7, 9])
