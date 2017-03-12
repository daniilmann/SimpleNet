# -*- encode: utf-8 -*-

from os.path import expanduser, join, exists, sep
from os import listdir
import sys

import pandas as pd
import numpy as np

from scipy.stats import kruskal, levene, ttest_ind

import matplotlib.pyplot as plt

import wavelet as wv
from dataworker import dataloader as dl, quandlwrap as qn
import configs as cfg

path = expanduser('~/investparty/ideas/maxday/data')
exch = 'nyse'
exchf = 'NYSE.csv'


data = dl.load_frame(join(cfg.curr_path, 'XAUUSD.csv'), columns = ['Date', 'TIME', 'OPEN', 'HIGH', 'LOW', 'Close', 'VOL'], pre_delete = ['TIME', 'OPEN', 'HIGH', 'LOW', 'VOL'], ix_columns='Date', ix_format='%d/%m/%y', logret={'Close':1})
pixs, nixs = np.argwhere(data.CloseLogRet1 >= .03).squeeze(), np.argwhere(data.CloseLogRet1 <= -.03).squeeze()

pret, pstd = [], []
nret, nstd = [], []
print '\tDay\t|\tMin\t\t|\tMedian\t|\tMean\t|\tMax\t\t|\tStd'
for npv in range(1,16,1):
    nppret, nnpret = np.array([np.sum(data.ix[i+1:i+npv+1].CloseLogRet1) for i in pixs if np.sum(data.ix[i+1:i+npv+1].CloseLogRet1) > 0]), np.array([np.sum(data.ix[i+1:i+npv+1].CloseLogRet1) for i in pixs if np.sum(data.ix[i+1:i+npv+1].CloseLogRet1) <= 0])
    npnret, nnnret = np.array([np.sum(data.ix[i+1:i+npv+1].CloseLogRet1) for i in nixs if np.sum(data.ix[i+1:i+npv+1].CloseLogRet1) > 0]), np.array([np.sum(data.ix[i+1:i+npv+1].CloseLogRet1) for i in nixs if np.sum(data.ix[i+1:i+npv+1].CloseLogRet1) <= 0])

    # pret.append(np.round(nppret.mean(), 3))
    # pstd.append(np.round(nppret.std(), 3))
    #
    # nret.append(np.round(nnpret.mean(), 3))
    # nstd.append(np.round(nnpret.std(), 3))

    #print '\t%d\t|\t%.3f\t|\t%.3f\t|\t%.3f' % (npv, np.round(np.append(-nnpret, npnret).sum(), 3), np.round(np.append(-nppret, nnnret).sum(), 3), np.round(np.append(-nnpret, npnret).sum() + np.append(-nppret, nnnret).sum(), 3))

    print npv,  npnret.mean(), nnnret.mean()
    print ttest_ind(npnret, np.absolute(nnnret))
    #t = ttest_ind(np.append(nppret, np.absolute(nnnret)), np.append(np.absolute(nnpret), npnret))
    #k = kruskal(np.append(nppret, np.absolute(nnnret)), np.append(np.absolute(nnpret), npnret))
    #l = levene(np.append(nppret, np.absolute(nnnret)), np.append(np.absolute(nnpret), npnret))
    #print '\t%.3f (%.3f)\t|\t%.3f (%.3f)\t|\t%.3f (%.3f)\t' % (np.round(t[0], 2), np.round(t[1], 2), np.round(k[0], 2), np.round(k[1], 2), np.round(l[0], 2), np.round(l[1], 2))
    t = nppret
    #print '\t%d\t|\t%0.3f\t|\t%0.3f\t|\t%0.3f\t|\t%0.3f\t|\t%0.3f' % (npv, np.round(t.min(), 3), np.round(np.median(t), 3), np.round(t.mean(), 3), np.round(t.max(), 3), np.round(t.std(), 3))
    #print '\t%d\t|\t%d%%\t|\t%d\t|\t%0.3f\t|\t%d\t|\t%0.3f' % (npv, np.round(float(len(nnpret)) / len(pixs) * 100), len(nppret), np.round(nppret.mean(), 3), len(nnpret), np.round(nnpret.mean(), 3))
    # print 'Day ', npv, ' || ', np.round(float(len(npnret)) / len(nixs) * 100), '%'
    # print '\t+\t\t:', len(npnret), '\t\t-\t\t:', len(nnnret)
    # #print float(len(nnpret)) / float(len(pixs)), float(len(npnret)) / float(len(nixs))
    # #print nppret.mean(), nnpret.mean()
    # #print nppret.sum(), nnpret.sum()
    # print '\tmean +\t:', np.round(npnret.mean(), 3), '\tmean -\t:', np.round(nnnret.mean(), 3)
    # print '\tmed +\t:', np.round(np.median(npnret), 3), '\tmed -\t:', np.round(np.median(nnnret), 3)
    #print npnret.sum(), nnnret.sum()

# plt.bar(range(1,16), pret, yerr=pstd, width=.5, color='b')
# plt.bar(range(1,16), nret, yerr=nstd, width=.5, color='r')
# plt.xlabel("Days")
# plt.ylabel('Return')
# plt.show()
# data = None
# if exists(join(path, exch + '_ret.csv')) and exists(join(path, exch + '_price.csv')):
#     prices = pd.read_csv(join(path, exch + '_price.csv'))
#     prices = prices.set_index(pd.DatetimeIndex(pd.to_datetime(prices[prices.columns[0]], format='%Y-%m-%d')))
#     del(prices[prices.columns[0]])
#     ret = pd.read_csv(join(path, exch + '_ret.csv'))
#     ret = ret.set_index(pd.DatetimeIndex(pd.to_datetime(ret[ret.columns[0]], format='%Y-%m-%d')))
#     del(ret[ret.columns[0]])
# else:
#     if len(listdir(join(path, 'nyse'))) == 1:
#         qn.load(pd.read_csv(join(path, exchf)), join(path, exch), '2000-01-01', n_jobs=2)
#
#     files = filter(lambda n: n[-3:] == 'csv', listdir(join(path, exch)))
#     files = map(lambda x: join(path, exch, x), files)
#
#     prices = dw.load_frame(files[0], 'Date', '%Y-%m-%d', columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], pre_delete=['Open', 'High', 'Low', 'Volume'])
#     ret = np.log(prices / prices.shift())[1:]
#     i0 = files[0].rindex(sep) + 1
#     prices.columns = [files[0][i0:-4]]
#     ret.columns = [files[0][i0:-4]]
#     for f in files[1:]:
#         tmp = dw.load_frame(f, 'Date', '%Y-%m-%d', columns=['Date', 'Open', 'High', 'Low', 'Close', 'Volume'], pre_delete=['Open', 'High', 'Low', 'Volume'])
#         tmpr = np.log(tmp / tmp.shift())[1:]
#         tmp.columns = [f[i0:-4]]
#         tmpr.columns = [f[i0:-4]]
#         prices = pd.concat((prices, tmp), axis=1)
#         ret = pd.concat((ret, tmpr), axis=1)
#
#     prices.to_csv(join(path, exch + '_price.csv'))
#     ret.to_csv(join(path, exch + '_ret.csv'))
#
# prices = prices.loc[prices.index.year >= 1990]
# prices = prices.resample('W').last()
# ret = ret.loc[ret.index.year >= 1990]
# ret = ret.resample('W').sum()
# tmp = np.absolute(ret.fillna(0).as_matrix())
# #tmp = data.fillna(0).as_matrix()
# crit = pd.DataFrame(np.round(tmp.mean(0) + 6 * tmp.std(0), 3), columns=['Crit'], index=ret.columns)
# critdict = crit.Crit.to_dict()
#
# rets = []
# idx = []
# assets = {}
# counter = 0
# a = 1
# for row in ret.index:
#     counter += 1
#     tmp = np.absolute(ret.ix[row].tolist())
#     #clmns = np.intersect1d(ret.columns[np.argwhere(tmp > crit)], ret.columns[np.argwhere(np.array(ret.ix[row].tolist()) < -.1)])
#     #clmns = np.intersect1d(ret.columns[np.argwhere(tmp > crit)], ret.columns[np.argwhere(tmp > .15)])
#     clmns = np.array(ret.columns[np.argwhere(tmp > crit.Crit)])
#     if len(clmns) > 0:
#         if len(clmns.squeeze().shape) == 0:
#             idxmax = clmns[0]
#         else:
#             clmns = clmns.squeeze()[(np.absolute(ret.ix[row][clmns.squeeze()].tolist()) / [critdict[c] for c in clmns.squeeze()]).argsort()][:min(len(clmns), a)]
#             idxmax = clmns.tolist()
#             #idxmax = data.ix[row][clmns.squeeze()].argsort().head(10).tolist()
#
#         if idxmax:
#             try:
#                 r = np.array(np.nan_to_num(ret.ix[row][idxmax].tolist()))
#                 row1 = row + 1 * pd.tseries.offsets.Week()
#                 tmp, tmp1 = np.array(prices.ix[row][idxmax].tolist()), np.array(prices.ix[row1][idxmax].tolist())
#                 v = []
#                 if len(tmp.shape) == 0:
#                     s = np.sign(r)
#                     if s != 0 and np.sign(tmp1) != 0. and np.sign(tmp) != 0.:
#                         #c = np.round(5000. / tmp)
#                         if c > 0:
#                             #t = np.nan_to_num(np.round(-s * (tmp1 - tmp + s * .04) * c - 1, 4))
#                             #t = np.nan_to_num(np.round(s * (tmp1 - tmp - s * .04) * c - 1, 4))
#                             v.append(t)
#                             #v.append(np.sign(t) * min(abs(t), 500))
#                     else:
#                         v.append(0.)
#                 else:
#                     for i in range(len(tmp)):
#                         s = np.sign(r[i])
#                         if s != 0 and np.sign(tmp1[i]) != 0. and np.sign(tmp[i]) != 0.:
#                             c = np.round(1000. / tmp[i])
#                             if c > 0:
#                                 t = np.nan_to_num(np.round(-s * (tmp1[i] - tmp[i] + s * .04) * c - 1, 4))
#                                 #t = np.nan_to_num(np.round(s * (tmp1[i] - tmp[i] - s * .04) * c - 1, 4))
#                                 if abs(t) < 1000:
#                                     v.append(t)
#                                 else:
#                                     v.append(0.)
#                                 #v.append(np.sign(t) * min(abs(t), 200))
#                             else:
#                                 v.append(0.)
#                         else:
#                             v.append(0.)
#                 p = np.sum(v)
#                 #p = np.sign(p) * min(abs(p), 100 * a)
#                 #s = np.sum(s)
#                 if p != 0.:
#                     clmns = [idxmax] if isinstance(idxmax, str) else idxmax
#                     for i in range(len(clmns)):
#                         if clmns[i] in assets.keys():
#                             assets[clmns[i]].append(v[i])
#                         else:
#                             assets[clmns[i]] = [v[i]]
#                     rets.append(p)
#                     idx.append(row1)
#                     #print('\r%5d / %5d || %5d :: %f' % (row, ret.shape[0], len(rets), np.sum(rets)))
#                     sys.stdout.write('\r%5d / %5d || %5d :: %f' % (counter, ret.shape[0], len(rets), np.sum(rets)))
#             except Exception:
#                 sys.stdout.write('\r%5d / %5d || %5d :: %f' % (counter, ret.shape[0], len(rets), np.sum(rets)))
#
# print ''
# print np.mean(rets), np.std(rets)
# print np.min(rets), np.max(rets)
# print np.mean(np.absolute(rets)), np.std(np.absolute(rets))
# plt.plot(idx, np.cumsum(rets))
# plt.show()
