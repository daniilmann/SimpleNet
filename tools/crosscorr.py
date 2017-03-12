# -*- encode: utf-8 -*-

from os.path import expanduser, join
from os import listdir

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

import wavelet as wv
from dataworker import dataworker as dw

path = expanduser('~/investparty/ideas/portfolio/092016/gold/data')
files = filter(lambda n: n[-3:] == 'csv', listdir(path))

def get_frame(f, format, columns, dcolumns):
    try:
        return dw.load_frame(join(path, f), 'Date', format, columns=columns, pre_delete=dcolumns)
    except:
        return None

formats = [
    [ '%Y-%m-%d',
      ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Adj Close'],
      ['Open', 'High', 'Low', 'Volume', 'Adj Close']
    ],
    [ '%Y-%m-%d',
      ['Date', 'Open', 'High', 'Low', 'Close', 'Volume'],
      ['Open', 'High', 'Low', 'Volume']
    ],
    [
        '%d/%m/%y',
        ['Date', 'TIME', 'OPEN', 'HIGH', 'LOW', 'Close', 'VOL'],
        ['TIME', 'OPEN', 'HIGH', 'LOW', 'VOL']
    ],
    [
        '%d.%m.%y',
        ['Date', 'Close'],
        None
    ]
]
frames = []
for f in files:
    frame = None
    for form in formats:
        frame = get_frame(f, *form)
        if frame is not None:
            break
    frames.append(frame)


data = frames[0]
for f in frames[1:]:
    data = pd.concat((data, f), axis=1)
data.columns = map(lambda f: f[:-4], files)
data = data.dropna()
data = data.resample('W').last()
data = data.dropna()
ret = np.log(data / data.shift())[1:]
#ret = ret.cumsum(0)
#ret.to_csv(join(path, 'ret.csv'), index=False)

dts = []
#dts.append(ret)
#dts.append(ret.ix[np.intersect1d(np.argwhere(ret.index.year < 2007), np.argwhere(ret.index.year >= 2000))])
dts.append(ret.loc[ret.index.year > 2013])
crls, bts = [], []
for dt in dts:
    dt = dt.fillna(0.)
    smooth = wv.smooth(dt.as_matrix().transpose()[:, -np.power(2, np.floor(np.log2(dt.shape[0]))):], 2, 20)
    # for l in [0,1,2,6,7]:
    #     plt.plot(smooth[l])
    # plt.show()
    crls.append(np.corrcoef(smooth))
    betas = []
    for i in range(smooth.shape[0]):
        bs = []
        for j in range(smooth.shape[0]):
            if i != j:
                bs.append(.33 + .66 * np.cov(smooth[i], smooth[j])[0, 1] / np.var(smooth[j]))
            else:
                bs.append(1.)
        betas.append(bs)
    bts.append(np.vstack(betas))

corrs, betas = crls[0], bts[0]
for i in range(1, len(crls)):
    corrs = np.add(corrs, crls[i])
    betas = np.add(betas, bts[i])

corrs, betas = corrs / len(crls), betas / len(bts)


# clrs = ['y', 'r', 'b', 'm', 'k']
# plt.colors()
# for t in range(0, 5):
#     plt.plot(np.cumsum(smooth[t][-52:]), clrs[t])
# plt.show()

print pd.DataFrame(corrs, columns=data.columns, index=data.columns)
#pd.DataFrame(corrs, columns=data.columns, index=data.columns).to_csv(join(path, 'corrD'), sep=';', dec=',')
print pd.DataFrame(betas, columns=data.columns, index=data.columns)
#pd.DataFrame(betas, columns=data.columns, index=data.columns).to_csv(join(path, 'betaD'), sep=';', dec=',')