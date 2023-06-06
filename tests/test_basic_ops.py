import numpy as np

from minigrad.core.tensor import Tensor


def test_add():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    c = a + b
    assert np.allclose(c.numpy(), [5, 7, 9])


def test_sub():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    c = a - b
    assert np.allclose(c.numpy(), [-3, -3, -3])


def test_mul():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    c = a * b
    assert np.allclose(c.numpy(), [4, 10, 18])


def test_div():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    c = a / b
    assert np.allclose(c.numpy(), [0.25, 0.4, 0.5])


def test_pow():
    a = Tensor([1.0, 2.0, 3.0])
    b = Tensor([4.0, 5.0, 6.0])
    c = a**b
    assert np.allclose(c.numpy(), [1, 32, 729])


def test_matmul():
    a = Tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    b = Tensor([[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])
    c = a @ b
    assert np.allclose(c.numpy(), [[40, 46], [94, 109]])


def test_neg():
    a = Tensor([1.0, 2.0, 3.0])
    c = -a
    assert np.allclose(c.numpy(), [-1, -2, -3])


def test_log():
    a = Tensor([1.0, 2.0, 3.0])
    c = a.log()
    assert np.allclose(c.numpy(), [0, np.log(2), np.log(3)])


def test_exp():
    a = Tensor([1.0, 2.0, 3.0])
    c = a.exp()
    assert np.allclose(c.numpy(), [np.exp(1), np.exp(2), np.exp(3)])


def test_sqrt():
    a = Tensor([1.0, 2.0, 3.0])
    c = a.sqrt()
    assert np.allclose(c.numpy(), [np.sqrt(1), np.sqrt(2), np.sqrt(3)])


def test_add_cuda():
    a = Tensor([1.0, 2.0, 3.0], device="cuda")
    b = Tensor([4.0, 5.0, 6.0], device="cuda")
    c = a + b
    assert np.allclose(c.numpy(), [5, 7, 9])


def test_sub_cuda():
    a = Tensor([1.0, 2.0, 3.0], device="cuda")
    b = Tensor([4.0, 5.0, 6.0], device="cuda")
    c = a - b
    assert np.allclose(c.numpy(), [-3, -3, -3])


def test_mul_cuda():
    a = Tensor([1.0, 2.0, 3.0], device="cuda")
    b = Tensor([4.0, 5.0, 6.0], device="cuda")
    c = a * b
    assert np.allclose(c.numpy(), [4, 10, 18])


def test_div_cuda():
    a = Tensor([1.0, 2.0, 3.0], device="cuda")
    b = Tensor([4.0, 5.0, 6.0], device="cuda")
    c = a / b
    assert np.allclose(c.numpy(), [0.25, 0.4, 0.5])


def test_pow_cuda():
    a = Tensor([1.0, 2.0, 3.0], device="cuda")
    b = Tensor([4.0, 5.0, 6.0], device="cuda")
    c = a**b
    assert np.allclose(c.numpy(), [1, 32, 729])


def test_matmul_cuda():
    aa = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    bb = np.array([[4.0, 5.0], [6.0, 7.0], [8.0, 9.0]])
    cc = aa @ bb
    a = Tensor(aa, device="cuda")
    b = Tensor(bb, device="cuda")
    c = a @ b
    assert np.allclose(c.numpy(), cc)


def test_neg_cuda():
    a = Tensor([1.0, 2.0, 3.0], device="cuda")
    c = -a
    assert np.allclose(c.numpy(), [-1, -2, -3])


def test_log_cuda():
    a = Tensor([1.0, 2.0, 3.0], device="cuda")
    c = a.log()
    assert np.allclose(c.numpy(), [0, np.log(2), np.log(3)])


def test_exp_cuda():
    a = Tensor([1.0, 2.0, 3.0], device="cuda")
    c = a.exp()
    assert np.allclose(c.numpy(), [np.exp(1), np.exp(2), np.exp(3)])


def test_sqrt_cuda():
    a = Tensor([1.0, 2.0, 3.0], device="cuda")
    c = a.sqrt()
    assert np.allclose(c.numpy(), [np.sqrt(1), np.sqrt(2), np.sqrt(3)])
