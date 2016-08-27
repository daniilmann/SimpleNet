# -*- encoding: utf-8 -*-

import numpy as np
from scipy.stats import skew, kurtosis

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (np.array(cumsum[N:]) - np.array(cumsum[:-N])) / float(N)

def normilize(data, axis=0, type='O'):

    def _tanh():
        mn = data.min(axis)
        mx = data.max(axis)
        a = mx + mn
        b = mx - mn

        return (2. * data - a) / b

    def _zmean():
        mn = data.min(axis)
        mx = data.max(axis)

        return (data - mn) / (mx - mn)

    return {
        'O' : _tanh(),
        'ZM' : _zmean()
    }.get(type)

def mean(x, axis=1, keepdims=True):
    return x.mean(axis, keepdims=keepdims)

def mad(x, axis=1, keepdims=True):
    return np.absolute(x - x.mean(1, keepdims=True)).mean(axis, keepdims=keepdims)

def var(x, axis=1, keepdims=True):
    return x.var(axis, keepdims=keepdims)

def skew(x, axis=1, keepdims=True):
    if keepdims:
        return skew(x, axis=axis)[:, np.newaxis]
    else:
        return skew(x, axis=axis)

def kurtosis(x, axis=1, keepdims=True):
    if keepdims:
        return kurtosis(x, axis=axis)[:, np.newaxis]
    else:
        return kurtosis(x, axis=axis)