# -*- encode: utf-8 -*-

from os.path import expanduser, join
from os import listdir, remove
import sys
import multiprocessing as mp
from functools import partial

import pandas as pd
import numpy as np
import quandl
import urllib

quandl.ApiConfig.api_key = '7fhzpCA_h2GZobLUKwsT'
quandl.ApiConfig.api_version = '2015-04-09'

def _load(params, lock, n, mx, start_date, path):
    ticker, dataset, name = params
    res = 'SUCCESS'
    try:
        # fl = join(path, ticker + '.csv')
        # urllib.urlretrieve('https://www.quandl.com/api/v3/datasets/{dataset}.csv?'
        #                    'api_key=7fhzpCA_h2GZobLUKwsT&'
        #                    'start_date={start_date}'.format(dataset=dataset, start_date=start_date),
        #                    filename=fl)
        # data = pd.read_csv(fl)
        # if data.Volume.mean() < 1E+6:
        #     remove(fl)
        data = quandl.get(dataset, ticker=ticker, start_date=start_date, api_key='7fhzpCA_h2GZobLUKwsT')
        if data.Volume.mean() >= 1E+6:
            data.to_csv(join(path, ticker + '.csv'))
    except Exception:
        res = 'ERROR'
    with lock:
        n.value += 1
        print('\r %s %5d / %5d || %s' % (res, n.value, mx, name))

def load(tickers, path, start_date=None, n_jobs=2):
    n = 0
    tickers = tickers.as_matrix()
    if n_jobs == 1:
        for t in tickers:
            try:
                n += 1
                data = quandl.get(t[1], ticker=t[0], start_date=start_date)
                if data.Volume.mean() >= 1E+6:
                    data.to_csv(join(path, t[0] + '.csv'))
            except Exception:
                print('ERROR \r%5d / %5d || %s' % (n, tickers.shape[0], t[-1]))
                continue
            print('SUCCESS \r%5d / %5d || %s' % (n, tickers.shape[0], t[-1]))
    else:
        m = mp.Manager()
        pload = partial(_load, start_date=start_date, path=path, mx=tickers.shape[0], n=m.Value('i', 0), lock=m.Lock())
        pool = mp.Pool(n_jobs)
        #res = pool.map_async(pcaus, tickers)
        res = pool.map(pload, tickers)
        pool.close()
        pool.join()