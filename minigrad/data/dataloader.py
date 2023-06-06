from minigrad import Tensor


class DataLoader:
    """
    Enables easier batch data retrieval
    """

    def __init__(self, *data, batch_size, tensors=False):
        assert len(data) != 0, "No data given"
        assert len(set([arr.shape[0] for arr in data])) == 1, "Not all arrays have equal first dimension"
        self.n = data[0].shape[0]
        self.data = data
        self.tensors = tensors
        self.batch_size = batch_size

    def get(self):
        for i in range(0, self.n, self.batch_size):
            if self.tensors:
                yield (Tensor(arr[i : i + self.batch_size]) for arr in self.data)
            else:
                yield (arr[i : i + self.batch_size] for arr in self.data)
