import numpy as np

def numerical_diff(func, x):
    h = 1e-6
    gradien = np.zeros(len(x))
    y = func(x)
    for idx, item in enumerate(x):
        hx = x.copy()
        hx[idx] += h
        diff = (func(hx) - y) / h
        gradien[idx] = diff
    return gradien
