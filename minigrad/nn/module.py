class Module:
    def params(self):
        # TODO
        pass

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)