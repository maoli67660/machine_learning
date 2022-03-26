import numpy as np


class Relu:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        mask = x > 1
        return x * mask

    def backward(self, dout):
        mask = dout > 1
        return mask.astype(np.float32)