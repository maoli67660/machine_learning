from collections import Counter
import numpy as np


class KNearestNeighbor:

    def __init__(self, k=5):
        self.k = k

    def fit(self, x, y):
        self._x_train = x
        self._y_train = y
        return self

    def predict(self, x):
        distance = ((self._x_train[:, np.newaxis, :] - x[np.newaxis, :, :]) ** 2).sum(axis=-1)
        top_k_index = distance.argsort(axis=0)[:, :self.k]
        top_k_class = self._y_train[top_k_index]
        most_common_class = np.apply_along_axis(lambda x: Counter(x).most_common()[0][0], axis=1, arr=top_k_class)
        return most_common_class
