# -*- encode: utf-8 -*-

from os.path import expanduser, join, exists, sep
from os import listdir
import sys
from datetime import datetime as dt, date
from calendar import monthrange

import pandas as pd
import numpy as np

from scipy import signal, fftpack

import matplotlib.pyplot as plt
from matplotlib import colors

from dataworker import dataloader as dw
import wavelet as wv
import configs as cfg

def gen_seasonal_data(data, startMonth, endMonth, startYear=None, endYear=None, years=0, resample='B', includeSmall=False, index=None):
    if endMonth < startMonth and years == 0:
        raise Exception("last month must be greated or equal than first one")

    def resampler(rdata, rsmpl, sdt, edt, isml, cp):
        if rdata.shape[0] < cp and not isml:
            return None

        ix = pd.bdate_range(sdt, edt)
        rdata = rdata.reindex(ix, fill_value=0.0)
        my = min(rdata.index.year)
        if rsmpl == 'W':
            rdata = rdata.resample(rsmpl).sum()

        rdata.index = range(rdata.shape[0])

        return rdata

    if startYear is not None:
        data = data.loc[(data.index.year >= startYear)]

    if endYear is not None:
        data = data.loc[(data.index.year <= endYear)]

    obs = []
    lbls = []
    idxData = []
    cyear, myear = min(data.index.year), max(data.index.year)
    if (data.index.year[-1] < date.today().year) or (data.index.year[-1] == date.today().year and data.index.month[-1] < endMonth):
        myear -= 1
    critPeriods = {
                      'B': 15,
                      'W': 3
                  }[resample] * (years * 12 + endMonth - startMonth + 1)
    while cyear + years <= myear:
        try:
            startdt, enddt = dt(cyear, startMonth, 1), dt(cyear + years, endMonth, monthrange(cyear + years, endMonth)[1])
            tmp = data.loc[(data.index >= startdt) & (data.index <= enddt)]

            tmp = resampler(tmp, resample, startdt, enddt, includeSmall, critPeriods)

            if tmp is None:
                continue

            if index is not None:
                idx = index.loc[(index.index >= startdt) & (index.index <= enddt)]
                idx = resampler(idx, resample, startdt, enddt, includeSmall, critPeriods)
                if idx is None:
                    continue
                else:
                    idxData.append(idx[idx.columns[0]].tolist())
                    obs.append(tmp[tmp.columns[0]].tolist())
            else:
                obs.append(tmp[tmp.columns[0]].tolist())

            lbls.append(cyear)
        finally:
            cyear += max(1, years)

    ml = max([len(v) for v in obs])
    obs = np.vstack([np.append(np.repeat(0, ml - len(v)), v) for v in obs])
    if index is not None:
        ml = max([len(v) for v in idxData])
        idxData = np.vstack([np.append(np.repeat(0, ml - len(v)), v) for v in idxData])
    else:
        idxData = None

    return obs, idxData, lbls

gsd = gen_seasonal_data

def plot(ptype, xdata, ydata=None, labels=None, **params):


    figsize = params.get('figsize') if params.get('figsize') is not None else (15, 8)
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    if params.get('ticker') is not None:
        ax.set_title(params.get('ticker'), fontsize=14)

    if ptype == 'bar':
        ax.bar(xdata, ydata)
    elif ptype == 'chart':
        if labels is not None:
            for row, label in zip(xdata, labels):
                ax.plot(row, label=label)
        else:
            ax.plot(xdata)

    if params.get('save') is None:
        if labels is not None:
            ax.legend(loc='lower center', ncol=5, bbox_to_anchor=(0.5, 0.01))
        plt.show()
    else:
        if labels is not None:
            handles, labels = ax.get_legend_handles_labels()
            lgd = ax.legend(handles, labels, ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.05))
            fig.savefig(params.get('save'), bbox_extra_artists=(lgd,), bbox_inches='tight')
        else:
            fig.savefig(params.get('save'))

def cumret(data, index=None, labels=None, wave=None):

    data = np.copy(data)

    if wave is not None:
        data = data[:, -wave[0]:]
        data = wv.smooth(data, wave[1], wave[2])

    data = np.cumsum(data, 1)

    if labels is None:
        labels = range(data.shape[0])

    if index is not None:
        index = np.copy(index)
        if wave is not None:
            index = index[:, -wave[0]:]
            index = wv.smooth(index, wave[1], wave[2])
        index = np.cumsum(index, 1)

        lbls = map(lambda i: '%d | %.3f(%.3f)' % (labels[i], data[i][-1], data[i][-1] - index[i][-1]), range(len(labels)))

        return data - index, lbls
    else:
        lbls = map(lambda i: '%d | %.3f' % (labels[i], data[i][-1]), range(len(labels)))
        return data, lbls

def dependencies(data, index, dtype, years=None, **params):

    data = np.copy(data)
    index = np.copy(index)

    if params.get('wave') is not None:
        wave = params.get('wave')
        data = data[-wave[0]:]
        index = index[-wave[0]:]

        data = wv.smooth(data, wave[1], wave[2])
        index = wv.smooth(index, wave[1], wave[2])

    if dtype == 'corr':
        res = np.array([np.corrcoef(data[i], index[i])[0,1] for i in range(data.shape[0])])
    elif dtype == 'beta':
        res = np.array([np.cov(data[i], index[i])[0,1] / index.var() for i in range(data.shape[0])])

    if params.get('in_np'):
        return res
    else:
        if years is None:
            years = range(len(res))
        return pd.DataFrame(pd.Series(res), index=[int(v) for v in years], columns=['Corr'])

def find_seasonality():
    pass

ts = []
for i in range(100):
    ts.append(np.hstack((np.random.randint(1, 10, 4*30), np.random.randint(300, 500, 30), np.random.randint(1, 10, 7*30))))
    #ts.append(np.hstack((np.random.randint(1, 10, 2), np.random.randint(30,50,1), np.random.randint(1, 10, 1))))
    #ts.append(np.hstack((np.random.randint(1, 10, 10), np.random.randint(30, 50, 1), np.random.randint(1, 10, 1))))
ts = np.array(ts)
nts = (ts - ts.mean()) / (ts.max() - ts.min())
nts = nts - nts.mean()
# plt.plot(ts[:360])
# plt.show()

Fs = 44100
f = 5
sample = 100
x = np.arange(sample)
ts = []
for i in range(100):
    ts.append(32767. * np.sin(x * np.pi * 2. * 1. / 250.))# + np.sin(.75*x*np.pi) + np.random.randint(-100, 100, 360) / 100.)
nts = np.vstack(ts)
# plt.plot(32767. * np.sin(x * np.pi * 2. * 100. / 44100.),'r')
# plt.plot(32767. * np.sin(x * np.pi * 2. * 100. / 8000.), 'b')
# plt.show()


#fi, mi = signal.periodogram(nts, fs=1., window='flattop', nfft=360, detrend=False)
fi, mi = signal.welch(nts, fs=1., window='flattop', scaling='spectrum', nperseg=30, noverlap=3, nfft=100, detrend=False)
# plt.bar(fi, mi.mean(0), width=.5/len(fi))
# plt.show()

print fi[np.argmax(mi.mean(0))]
print 1. / fi[np.argmax(mi.mean(0))]