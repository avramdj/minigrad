from typing import Iterable, List, Union

from minigrad import Parameter, Tensor


# noinspection PyProtectedMember
class _MetaModule(type):
    def __call__(cls, *args, **kwargs):
        module = super().__call__(*args, **kwargs)
        module._register_child_modules()
        module._register_child_params()
        return module


class Module(metaclass=_MetaModule):
    """
    Base class for all minigrad modules.
    Derived classes must implement forward(self, *args, **kwargs)
    """

    def __init__(self) -> None:
        self._params: List[Tensor] = []

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

    def _register_child_modules(self):
        for name, node in self.__dict__.items():
            if isinstance(node, Module):
                self._register_params(node.params())

    def _register_child_params(self):
        for name, node in self.__dict__.items():
            if isinstance(node, Parameter):
                self._register_params(node)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError("Forward not implemented in module")
