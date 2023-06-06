import numpy as np
import torch as t

from minigrad.engine.tensor import Tensor


def test_add():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True)
    b = Tensor([3.0, 3.0, 3.0], requires_grad=True)
    c = a * b
    d = c * 5
    d.mean().backward()
    a1 = t.tensor([1.0, 2.0, 3.0], requires_grad=True)
    b1 = t.tensor([3.0, 3.0, 3.0], requires_grad=True)
    c1 = a1 * b1
    d1 = c1 * 5
    d1.mean().backward()

    print(f"{a.grad} \t\t {a1.grad}")
    print(f"{b.grad} \t\t {b1.grad}")

    assert np.allclose(a.grad.detach().to_numpy(), a1.grad.detach().numpy())
    assert np.allclose(b.grad.detach().to_numpy(), b1.grad.detach().numpy())


# # Pytorch cuda error
# def test_add():
#     a = Tensor([1.0, 2.0, 3.0], requires_grad=True, device="cuda")
#     b = Tensor([3.0, 3.0, 3.0], requires_grad=True, device="cuda")
#     c = a * b
#     d = c * 5
#     d.mean().backward()
#     a1 = t.tensor([1.0, 2.0, 3.0], requires_grad=True, device="cuda")
#     b1 = t.tensor([3.0, 3.0, 3.0], requires_grad=True, device="cuda")
#     c1 = a1 * b1
#     d1 = c1 * 5
#     d1.mean().backward()

#     print(f"{a.grad} \t\t {a1.grad}")
#     print(f"{b.grad} \t\t {b1.grad}")

#     assert np.allclose(a.grad.detach().to_numpy(), a1.grad.detach().numpy())
#     assert np.allclose(b.grad.detach().to_numpy(), b1.grad.detach().numpy())
