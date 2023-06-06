from typing import ClassVar


class zero_grad:
    _GLOBAL_GRAD: ClassVar[bool] = True

    def __enter__(self):
        _GLOBAL_GRAD = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        _GLOBAL_GRAD = True


def is_grad():
    return zero_grad._GLOBAL_GRAD
