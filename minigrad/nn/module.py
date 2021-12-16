from typing import Iterable, Union
from engine import Tensor


class Module:
    """
    Base class for all minigrad modules.
    Derived classes must implement forward(self, *args, **kwargs)
    """
    def __init__(self):
        self._params: list[Tensor] = []

    def params(self):
        return self._params

    def _register_params(self, params: Union[Iterable[Tensor], Tensor]):
        if isinstance(params, Tensor):
            params = [params]
        self._params.extend(params)

    def _register_modules(self, modules: Union[Iterable["Module"], "Module"]):
        if isinstance(modules, Module):
            modules = [modules]
        for module in modules:
            self._register_params(module.params())

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward not implemented in module")