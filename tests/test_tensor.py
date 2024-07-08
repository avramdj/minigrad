import minigrad.core.tensor as Tensor


def test_create_tensor():
    t = Tensor.Tensor([1, 2, 3])
    # assert t.data == [1, 2, 3]
    assert t.requires_grad == False
    assert t.grad == None


def test_create_cuda_tensor():
    t = Tensor.Tensor([1, 2, 3], device="cuda")
    # assert t.data == [1, 2, 3]
    assert t.requires_grad == False
    assert t.grad == None
