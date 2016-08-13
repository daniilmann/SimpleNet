# -*- encoding: utf-8 -*-

import pandas as pd
import numpy as np
from os.path import expanduser, join
from os import listdir
import re
from itertools import izip as zip
import pickle
from functools import partial
from scipy.signal import argrelextrema
from scipy.special import erf
from scipy.stats import kruskal, levene, ttest_ind

import wavelet as wv
import manifold as mf

data_path = expanduser('~/investparty/data/futures')

def set_data_index(data, column, form='{:06d}'):
    '''

    :param pd.DataFrame data:
    :param str column:
    :return:
    '''
    try:
        del (data['Time'])
    except Exception:
        pass
    data[column] = map(form.format, data[column])
    data = data.set_index(pd.DatetimeIndex(pd.to_datetime(data[column], format='%d%m%y')))
    del (data[column])
    data = data.sort_index(ascending=True)
    return data

def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (np.array(cumsum[N:]) - np.array(cumsum[:-N])) / float(N)

def futures_path(symbol=None):
    if symbol:
        names = listdir(data_path)
        if symbol in names:
            cpath = join(data_path, symbol)
        else:
            return None
        regexp = re.compile(symbol + '[A-Z][0-9].csv')
        symbols = filter(lambda f: regexp.search(f), listdir(cpath))
        if symbols:
            symbols = [(s[:-4], join(cpath, s)) for s in symbols]
        else:
            return None
        return symbols
        pass
    return None

def load_data(path, columns=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']):
    try:
        dat = pd.read_csv(path, names=columns, header=0, index_col=6)
        # dat['Date'] = pd.to_datetime(dat['Date'], format='%m-%d-%y')
        dat = set_data_index(dat, 'Date')
        return dat
    except Exception:
        return None

def normilize_to_one(obs):
    '''

    :param pd.DataFrame obs:
    :return:
    '''
    def _norm(row):
        row = row.tolist()
        return pd.Series(map((lambda x: x / row[29]), row))
    return obs.apply(_norm, 1)

def data_stat(data, period=2, file_name='stats.obs', matrix=True, verbose=True):

    def _mean(x):
        if verbose:
            print 'Mean ', x.shape
        return map(np.mean, x)

    def _mad(x):
        if verbose:
            print 'MAD ', x.shape
        return map(lambda x: np.mean(np.abs(x - np.mean(x))), x)

    def _variance(x):
        if verbose:
            print 'Variance ', x.shape
        def _var(x):
            v = np.var(x)
            m = np.mean(x)
            return (np.exp(v) - 1) * np.exp(2 * m + v)

        return map(_var, x)

    def _skewness(x):
        if verbose:
            print 'Skewness ', x.shape
        def _skew(x):
            v = np.var(x)
            return (np.exp(4 * v) + 2) * np.sqrt(np.exp(v) - 1)

        return map(_skew, x)

    def _kurtosis(x):
        if verbose:
            print 'Kurtosis ', x.shape
        def _kurt(x):
            v = np.var(x)
            return np.exp(4 * v) + 2 * np.exp(3 * v) + 3 * np.exp(2 * v) - 6

        return map(_kurt, x)

    def _extrema(x):
        if verbose:
            print 'Extrema ', x.shape
        n = np.array(map(lambda x: len(argrelextrema(np.array(x), np.greater)[0]), x))
        n += np.array(map(lambda x: len(argrelextrema(np.array(x), np.less)[0]), x))
        return n.tolist()

    def _shannon(x):
        if verbose:
            print 'Entropy ', x.shape
        return map(lambda x: np.log2(np.std(x) * np.exp(np.mean(x) + .5) * 2.5066282746310002), x)

    def _volat(x):
        if verbose:
            print 'Volatility ', x.shape
        per = 2 if period + 1 < 2 else period + 1
        lmbd = .94

        def _vol(x):
            def vol(x):
                if len(x) != per:
                    return lmbd * vol(x[:-1]) + (1 - lmbd) * np.power(x[-1], 2)
                else:
                    return np.var(x) + (1 - lmbd) * np.power(x[-1], 2)

            try:
                return vol(x)
            except Exception, err:
                print err.message
                return np.repeat(np.nan, len(x))

        volat = map(lambda x: _vol(x), x)
        return volat

    def _sum(x):
        if verbose:
            print 'Sum ', x.shape
        return map(np.sum, x)

    def _levels(x):
        pass

    def _data(x):
        if verbose:
            print 'data generation'
        dat = pd.DataFrame([
            _mean(x),
            _sum(x),
            _mean(np.absolute(x)),
            _mad(x),
            _variance(x),
            _skewness(x),
            _kurtosis(x),
            _extrema(x),
            _shannon(x),
            _volat(x)
        ]).transpose()
        dat.columns = ['Mean', 'Sum', 'MAM', 'MAD', 'Variance', 'Skew', 'Kurtosis', 'Extrema', 'Entropy', 'Volatility']
        return dat

    stats = _data(data)

    if file_name is not None:
        if verbose:
            print 'dumping'
        path = join(data_path, str(data.shape[-1]) + file_name)
        pickle.dump(stats, open(path, 'wb'))

    if matrix:
        return stats.as_matrix()
    else:
        return stats

def data_wave(data, level, db, file_name='daubechies.waves'):

    waves = wv.filter(data, level=level, db=db)

    if file_name is not None:
        path = join(data_path, file_name)
        pickle.dump(waves, open(path, 'wb'))

    return waves

def _data_3D(data_X, data_Y, sign, vsplit = .9, tsplit=.8):
    if sign:
        if len(data_Y.shape) == 3:
            data_Y = data_Y.squeeze()
            pos, neg = np.where(data_Y[:, 0, :1] > 0)[0], np.where(data_Y[:, 0, :1] < 0)[0]
        elif len(data_Y.shape) == 2:
            pos, neg = np.where(data_Y > 0)[0], np.where(data_Y < 0)[0]
    else:
        pos, neg = np.where(data_X[:,-1,-1] < data_Y[:,0,0])[0], np.where(data_X[:,-1,-1] > data_Y[:,0,0])[0]

    for _ in range(8):
        np.random.shuffle(pos)
        np.random.shuffle(neg)

    pos_X, pos_Y = data_X[pos, :, :], data_Y[pos, :]
    neg_X, neg_Y = data_X[neg, :, :], data_Y[neg, :]

    np.random.seed(123)

    order = range(pos_X.shape[0])
    np.random.shuffle(order)
    pos_X, pos_Y = pos_X[order], pos_Y[order]

    order = range(neg_X.shape[0])
    np.random.shuffle(order)
    neg_X, neg_Y = neg_X[order], neg_Y[order]

    vsplit = int(min(len(pos_X), len(neg_X)) * vsplit)
    tsplit = int(vsplit + int(min(len(pos_X), len(neg_X)) - vsplit) * tsplit)

    learn_X = np.vstack((pos_X[:vsplit, :, :], neg_X[:vsplit, :, :]))
    learn_Y = np.vstack((pos_Y[:vsplit, :], neg_Y[:vsplit, :]))
    valid_X = np.vstack((pos_X[vsplit:tsplit, :, :], neg_X[vsplit:tsplit, :, :]))
    valid_Y = np.vstack((pos_Y[vsplit:tsplit, :], neg_Y[vsplit:tsplit, :]))
    test_X = np.vstack((pos_X[tsplit:, :, :], neg_X[tsplit:, :, :]))
    test_Y = np.vstack((pos_Y[tsplit:, :], neg_Y[tsplit:, :]))

    order = range(len(learn_X))
    np.random.shuffle(order)
    learn_X, learn_Y = learn_X[order, :, :], learn_Y[order, :]

    order = range(len(valid_X))
    np.random.shuffle(order)
    valid_X, valid_Y = valid_X[order, :, :], valid_Y[order, :]

    # learn_Y = np.array([v for v in learn_Y[:, (y-1):y].tolist()])
    # valid_Y = np.array([v for v in valid_Y[:, (y-1):y].tolist()])
    # test_Y = np.array([v for v in test_Y[:, (y-1):y].tolist()])

    return learn_X, learn_Y, valid_X, valid_Y, test_X, test_Y

def _order_posneg(y, sign=None, vsplit=.8, tsplit=.4, seed=123):

    np.random.seed(seed)

    pos, neg = np.where(y < 0)[0], np.where(y > 0)[0]

    for _ in range(8):
        np.random.shuffle(pos)
        np.random.shuffle(neg)

    vsplit = int(min(len(pos), len(neg)) * vsplit)
    tsplit = int(vsplit + int(min(len(pos), len(neg)) - vsplit) * tsplit)

    learn = np.append(pos[:vsplit], neg[:vsplit])
    valid = np.append(pos[vsplit:tsplit], neg[vsplit:tsplit])
    test = np.append(pos[tsplit:], neg[tsplit:])

    for _ in range(8):
        np.random.shuffle(learn)
        np.random.shuffle(valid)
        np.random.shuffle(test)

    return learn, valid, test

def _order_dist(y, crit_stat=.01, max_iter=1000, vsplit=.8, tsplit=.4, verbose=1):

    # TODO add skewness and kurtosis comparisson

    vsplit = int(len(y) * vsplit)
    tsplit = min(int(vsplit + (len(y) - vsplit) * tsplit), len(y))
    ix = range(len(y))

    stats = {'kruskal' : [100, .001], 'levene' : [100, .001], 'ttestL' : [100, .001], 'ttestV' : [100, .001]}
    bix = None
    i = 0
    while stats['kruskal'][0] > crit_stat and stats['levene'][0] > crit_stat and abs(stats['ttestL'][0]) > crit_stat and abs(stats['ttestV'][0]) > crit_stat and i < max_iter:
        i += 1
        np.random.shuffle(ix)
        l, v = y[ix[:vsplit]], y[ix[vsplit:tsplit]]
        cs = kruskal(y, l, v)
        lv = levene(y, l, v)
        ttl = ttest_ind(y, l, equal_var=np.round(lv[0]) < .1)
        ttv = ttest_ind(y, v, equal_var=np.round(lv[0]) < .1)
        # if tsplit != len(y):
        #     t = y[ix[tsplit:]]
        #     cs = kruskal(y, l, v, t)
        #     lv = levene(y, l, v, t)
        # else:
        #     cs = kruskal(y, l, v)
        #     lv = levene(y, l, v)
        if verbose == 2:
            print cs, '\n', lv, '\n', ttl, '\n', ttv
        if cs[0] < stats['kruskal'][0] and lv[0] < stats['levene'][0] and abs(stats['ttestL'][0]) > abs(ttl[0]) and abs(stats['ttestV'][0]) > abs(ttv[0]):
            stats['kruskal'] = cs
            stats['levene'] = lv
            stats['ttestL'] = ttl
            stats['ttestV'] = ttv
            bix = ix

    if verbose > 0:
        print 'Best stat:\n\t', stats['kruskal'], '\n\t', stats['levene'], '\n\t', stats['ttestL'], '\n\t', stats['ttestV']

    if tsplit != len(y):
        return bix[:vsplit], bix[vsplit:tsplit], bix[tsplit:]
    else:
        return bix[:vsplit], bix[vsplit:tsplit]

def norm_tanh(data, a=None, b=None):
    if a is None or b is None:
        mx, mn = np.max(data), np.min(data)
        a, b = (mx + mn) / 2, (mx - mn) / 2

    return (data - a) / b #, a, b
    # if len(data.shape) == 2 and data.shape[-1] != 1:
    #     return np.array(map(lambda x: (2 * x - (np.max(x) + np.min(x))) / (np.max(x) - np.min(x)), data))
    # return None

def chong_net(paths, columns=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'], units=256, punits=1, period='D', db=20, vsplit=.4, tsplit=.65, norm=False, lvl=-1, dump_path=None):
    '''

    :param data[pd.DataFrame]:
    :param units:
    :param period:
    :param vsplit:
    :param tsplit:
    :return:
    '''

    if not isinstance(paths, list):
        raise Exception('paths is not a list')

    tss = []
    for path in paths:

        data = pd.read_csv(path, names=columns, header=0)

        data = set_data_index(data, 'Date')
        data = data.drop(data.index[np.argwhere(data.index.weekday == 5).squeeze()])
        data = data[['Close']]
        data = np.log(data / data.shift())[1:]
        data.columns = ['Ret']
        tss.append(data)

    if isinstance(data, pd.DataFrame):
        if not isinstance(data.index, pd.DatetimeIndex) and period != 'D':
            raise Exception('Data does not contain datetime index')
    else:
        raise Exception('Data is not a pandas DataFrame')

    if float(units) % 2 != 0:
        raise Exception('units length is not a power of 2')

    # if period == 'M':
    #     pass
    # elif period == 'W':
    #     data = data.resample('W-FRI').sum()
    #     data = data.set_index(pd.Series(data.index - pd.Timedelta('4 days')))
    # elif period == 'D':
    #     pass

    if len(tss) == 0:
        units = units + 1 if period == 'D' else units + 5
        i = 0
        obs = []
        while len(data) >= units + i:
            if period == 'D':
                obs.append(data.ix[i:(units + i), 0].tolist())
            else:
                tmp = data.ix[i:(units + i - 5), 0].tolist()
                tmp.append(np.sum(data.ix[(units + i - 5):(units + i + 5), 0].tolist()))
                obs.append(tmp)
            i += 1

        obs = np.array(obs)
        obs_x, obs_y = obs[:, :-1], obs[:, -1]
    else:
        def grouped(x, n, fn=None):
            "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
            #return zip(*[iter(x)] * n)
            #mn = n - 1
            return np.vstack(filter(lambda obs: len(obs) == n * 6, map(lambda i: np.hstack(map(lambda s: s[1].tolist(), x[i:i+n])), range(len(x) - n))))
        mtss = []
        for ts in tss:
            tmp_x, tmp_y = [], []
            ix = pd.date_range(ts.index.min(), ts.index.max())
            ix = ix[ix.weekday != 5]
            ts = ts.reindex(ix, fill_value=np.nan)
            ix = list(ts.resample('W-FRI').Ret)
            if ix[-1][1].shape != 6:
                ix = ix[1:len(ix) - 1]
            else:
                ix = ix[1:]
            mtss.append(ix)
        obs_x, obs_y = [], []
        if period == 'DW':
            n = int(np.ceil(units / 6.) + punits)
            for ts in mtss:
                obs_x.append(grouped(ts, n))
            obs_x = np.vstack(obs_x)[:, np.ceil(units / 6.) * 6 - units:]
            ix = np.argwhere(np.sum(np.isnan(obs_x[:, units:]), axis=1) <= punits)
            obs_x = obs_x[ix].squeeze()
            obs_x = np.nan_to_num(obs_x)
            obs_y = np.hstack(map(lambda x: np.sum(x, axis=1)[:, np.newaxis], np.hsplit(obs_x[:, units:], np.arange(1, punits, 1) * 6 if punits > 1 else 1)))
            obs_x = obs_x[:, :units]
        elif period == 'W':
            n = int(units + punits)
            for ts in mtss:
                obs_x.append(grouped(ts, n))
            obs_x = np.vstack(obs_x)
            ix = np.argwhere(np.sum(np.isnan(obs_x[:, units * 6:]), axis=1) <= punits)
            obs_x = obs_x[ix].squeeze()
            obs_x = np.nan_to_num(obs_x)
            obs_y = np.hstack(map(lambda x: np.sum(x, axis=1)[:, np.newaxis], np.hsplit(obs_x[:, units * 6:], np.arange(1, punits, 1) * 6 if punits > 1 else 1)))
            obs_x = np.hstack(map(lambda x: np.sum(x, axis=1)[:, np.newaxis], np.hsplit(obs_x[:, :units * 6], np.arange(1, units, 1) * 6)))
        elif period == 'D':
            n = int(np.ceil((units + punits) / 6.))
            for ts in mtss:
                obs_x.append(grouped(ts, n))
            obs_x = np.vstack(obs_x)
            crit = np.round(punits * .15)
            ix = np.argwhere(np.sum(np.isnan(obs_x[:, units:]), axis=1) <= crit)
            obs_x = obs_x[ix].squeeze()
            obs_x = np.nan_to_num(obs_x)
            obs_y = obs_x[:, units:units+punits]
            obs_x = obs_x[:, :units]

    if len(obs_y.shape) == 1:
        obs_y = obs_y[:, np.newaxis]

    print 'x shape ', obs_x.shape
    print 'y shape ',  obs_y.shape
    #print x[0][-1] == y[0]
    waves_all = wv.filter(obs_x, lvl, db)

    wave_A = waves_all[-1]['A']
    if norm:
        wave_A = norm_tanh(wave_A)

    waves = []
    waves.append(wave_A)
    for lvl in range(len(waves_all)):
        if norm:
            waves.append(norm_tanh(waves_all[lvl]['D']))
        else:
            waves.append(waves_all[lvl]['D'])

    stats = data_stat(obs_x, file_name=None)

    ex = ~np.isnan(stats).any(axis=1)
    obs_x = obs_x[ex]
    obs_y = obs_y[ex]
    stats = stats[ex]

    if norm:
        stats = np.apply_along_axis(norm_tanh, 0, stats)

    if obs_y.shape[-1] == 1:
        oy = obs_y
    else:
        oy = obs_y.sum(axis=1)
    lix, vix, tix = _order_dist(oy, .01, 1000, vsplit, tsplit, 1)

    # == learn ==
    learn_x = obs_x[lix]
    learn_y = obs_y[lix]
    learn_w = []
    for wave in waves:
        learn_w.append(wave[lix])
    learn_s = stats[lix]
    learn = (learn_x, learn_w, learn_s, learn_y)
    # ===========

    # == valid ==
    valid_x = obs_x[vix]
    valid_y = obs_y[vix]
    valid_w = []
    for wave in waves:
        valid_w.append(wave[vix])
    valid_s = stats[vix]
    valid = (valid_x, valid_w, valid_s, valid_y)
    # ===========

    # == learn ==
    test_x = obs_x[tix]
    test_y = obs_y[tix]
    test_w = []
    for wave in waves:
        test_w.append(wave[tix])
    test_s = stats[tix]
    test = (test_x, test_w, test_s, test_y)
    # ===========

    if dump_path is not None:
        pickle.dump((learn, valid, test), open(dump_path, 'wb'))

    return (learn, valid, test)

def smooth_net(paths, columns=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'], units=256, punits=1, period='D', db=20, vsplit=.8, tsplit=.8, lvl=-1, dump_path=None, mult=2.84):
        '''

        :param data[pd.DataFrame]:
        :param units:
        :param period:
        :param vsplit:
        :param tsplit:
        :return:
        '''

        if not isinstance(paths, list):
            raise Exception('paths is not a list')

        tss = []
        for path in paths:
            data = pd.read_csv(path, names=columns, header=0)

            data = set_data_index(data, 'Date')
            data = data.drop(data.index[np.argwhere(data.index.weekday == 5).squeeze()])
            data = data[['Close']]
            data = np.log(data / data.shift())[1:]
            data.columns = ['Ret']
            tss.append(data)

        if isinstance(data, pd.DataFrame):
            if not isinstance(data.index, pd.DatetimeIndex) and period != 'D':
                raise Exception('Data does not contain datetime index')
        else:
            raise Exception('Data is not a pandas DataFrame')

        if float(units) % 2 != 0:
            raise Exception('units length is not a power of 2')

        # if period == 'M':
        #     pass
        # elif period == 'W':
        #     data = data.resample('W-FRI').sum()
        #     data = data.set_index(pd.Series(data.index - pd.Timedelta('4 days')))
        # elif period == 'D':
        #     pass

        if len(tss) == -1:
            units = units + 1 if period == 'D' else units + 5
            i = 0
            obs = []
            while len(data) >= units + i:
                if period == 'D':
                    obs.append(data.ix[i:(units + i), 0].tolist())
                else:
                    tmp = data.ix[i:(units + i - 5), 0].tolist()
                    tmp.append(np.sum(data.ix[(units + i - 5):(units + i + 5), 0].tolist()))
                    obs.append(tmp)
                i += 1

            obs = np.array(obs)
            obs_x, obs_y = obs[:, :-1], obs[:, -1]
        else:
            def grouped(x, n, fn=None):
                "s -> (s0,s1,s2,...sn-1), (sn,sn+1,sn+2,...s2n-1), (s2n,s2n+1,s2n+2,...s3n-1), ..."
                # return zip(*[iter(x)] * n)
                # mn = n - 1
                return np.vstack(filter(lambda obs: len(obs) == n * 6,
                                        map(lambda i: np.hstack(map(lambda s: s[1].tolist(), x[i:i + n])),
                                            range(len(x) - n))))

            mtss = []
            for ts in tss:
                tmp_x, tmp_y = [], []
                ix = pd.date_range(ts.index.min(), ts.index.max())
                ix = ix[ix.weekday != 5]
                ts = ts.reindex(ix, fill_value=np.nan)
                ix = list(ts.resample('W-FRI').Ret)
                if ix[-1][1].shape != 6:
                    ix = ix[1:len(ix) - 1]
                else:
                    ix = ix[1:]
                mtss.append(ix)
            obs_x, obs_y = [], []
            if period == 'DW':
                n = int(np.ceil(units / 6.) + punits)
                for ts in mtss:
                    obs_x.append(grouped(ts, n))
                obs_x = np.vstack(obs_x)[:, np.ceil(units / 6.) * 6 - units:]
                ix = np.argwhere(np.sum(np.isnan(obs_x[:, units:]), axis=1) <= punits)
                obs_x = obs_x[ix].squeeze()
                obs_x = np.nan_to_num(obs_x)
                obs_y = np.hstack(map(lambda x: np.sum(x, axis=1)[:, np.newaxis],
                                      np.hsplit(obs_x[:, units:], np.arange(1, punits, 1) * 6 if punits > 1 else 1)))
                obs_x = obs_x[:, :units]
            elif period == 'W':
                n = int(units + punits)
                for ts in mtss:
                    obs_x.append(grouped(ts, n))
                obs_x = np.vstack(obs_x)
                ix = np.argwhere(np.sum(np.isnan(obs_x[:, units * 6:]), axis=1) <= punits)
                obs_x = obs_x[ix].squeeze()
                obs_x = np.nan_to_num(obs_x)
                obs_y = np.hstack(map(lambda x: np.sum(x, axis=1)[:, np.newaxis], np.hsplit(obs_x[:, units * 6:],
                                                                                            np.arange(1, punits,
                                                                                                      1) * 6 if punits > 1 else 1)))
                obs_x = np.hstack(map(lambda x: np.sum(x, axis=1)[:, np.newaxis],
                                      np.hsplit(obs_x[:, :units * 6], np.arange(1, units, 1) * 6)))
            elif period == 'D':
                n = int(np.ceil((units + punits) / 6.))
                for ts in mtss:
                    obs_x.append(grouped(ts, n))
                obs_x = np.vstack(obs_x)
                crit = np.round(punits * .15)
                ix = np.argwhere(np.sum(np.isnan(obs_x[:, units:]), axis=1) <= crit)
                obs_x = obs_x[ix].squeeze()
                obs_x = np.nan_to_num(obs_x)
                obs_y = obs_x[:, units:units + punits]
                obs_x = obs_x[:, :units]

        if len(obs_y.shape) == 1:
            obs_y = obs_y[:, np.newaxis]

        print 'x shape ', obs_x.shape
        print 'y shape ', obs_y.shape

        ixs = np.arange(obs_y.shape[0])
        oy = obs_y.sum(1)
        tix = np.append(np.argwhere(oy > oy.mean() + 2.5 * oy.std()), np.argwhere(oy < oy.mean() - 2.5 * oy.std()))
        ixs = np.delete(ixs, tix)

        my = oy[ixs] / obs_x[ixs, 12:].std(1)
        pix, zix, nix = np.argwhere(my >= .5).squeeze(), np.argwhere(abs(my) < .5).squeeze(), np.argwhere(my <= -.5).squeeze()
        count = min(len(pix), len(nix))

        if len(pix) == count:
            np.random.shuffle(nix)
            tix = np.append(tix, ixs[nix[count:]])
            nix = nix[:count]
        else:
            np.random.shuffle(pix)
            tix = np.append(tix, ixs[pix[count:]])
            pix = pix[:count]

        plix, pvix, ptix = _order_dist(my[pix], vsplit=.45, tsplit=.7, verbose=1)
        nlix, nvix, ntix = _order_dist(my[nix], vsplit=.45, tsplit=.7, verbose=1)
        zlix, zvix, ztix = _order_dist(my[zix], vsplit=.45, tsplit=.7, verbose=1)

        plix, pvix, ptix = ixs[pix[plix]], ixs[pix[pvix]], ixs[pix[ptix]]
        nlix, nvix, ntix = ixs[nix[nlix]], ixs[nix[nvix]], ixs[nix[ntix]]
        zlix, zvix, ztix = ixs[zix[zlix]], ixs[zix[zvix]], ixs[zix[ztix]]

        lix = np.hstack((plix, zlix, nlix))
        vix = np.hstack((pvix, zvix, nvix))
        tix = np.hstack((ptix, ztix, ntix, tix))

        np.random.shuffle(lix)
        np.random.shuffle(vix)
        np.random.shuffle(tix)

        ry = obs_y.sum(1, keepdims=True)
        obs_y = obs_x.sum(1, keepdims=True) + obs_y.cumsum(1)

        wave = wv.filter(obs_x, lvl, db)
        wave = np.cumsum(wave[-1]['A'] * mult, 1)

        learn_x, learn_w, learn_y, learn_ry = obs_x[lix], wave[lix], obs_y[lix], ry[lix]
        valid_x, valid_w, valid_y, valid_ry = obs_x[vix], wave[vix], obs_y[vix], ry[vix]
        test_x, test_w, test_y, test_ry = obs_x[tix], wave[tix], obs_y[tix], ry[tix]

        learn, valid, test = (learn_x, learn_w, learn_y, learn_ry), (valid_x, valid_w, valid_y, valid_ry), (test_x, test_w, test_y, test_ry)

        if dump_path is not None:
            pickle.dump((learn, valid, test), open(dump_path, 'wb'))

        return (learn, valid, test)

def reduced_class(path, columns=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'], from_units=256, to_units=10, lr=100, iters=200, period='D', vsplit=.8, tsplit=.8, dump_path=None):
    data = load_data(path, columns)

    data = data[['Close']]
    data = np.log(data / data.shift())[1:]
    data.columns = ['Ret']

    if period == 'M':
        pass
    elif period == 'W':
        data = data.resample('W-FRI').sum()
        data = data.set_index(pd.Series(data.index - pd.Timedelta('4 days')))
    elif period == 'D':
        pass

    i = 0
    obs = []
    while len(data) >= from_units + i:
        obs.append(data.ix[i:(from_units + i), 0].tolist())
        i += 1

    obs = np.array(obs)
    x, y = obs[:, :-1], obs[:, -1]
    x = wv.downsampling(wv.daubechies(x, 20, 'low', 'dec'))

    stats = data_stat(x)

    mx, mn = np.max(x), np.min(x)
    a, b = (mx + mn) / 2, (mx - mn) / 2
    x = (x - a) / b

    or_y = y
    y = (np.sign(y) + 1) / 2

    lix, vix, tix = _data_order(or_y, True, vsplit, tsplit)

    learn_x, learn_y, learn_stat, orly = x[lix], y[lix], stats[lix], or_y[lix]
    valid_x, valid_y, valid_stat, orvy = x[vix], y[vix], stats[vix], or_y[vix]
    test_x, test_y, test_stat, orty = x[tix], y[tix], stats[tix], or_y[tix]

    learn_obs_x = mf.tsne(learn_x, components=to_units, lr=lr, iter=iters, method='exact', verbose=2)

    valid_obs_x = []
    while len(valid_obs_x) < valid_x.shape[0]:
        print 'valid: ', len(valid_obs_x), ':', valid_x.shape[0]
        valid_obs_x.extend(mf.tsne(learn_x, predict=valid_x[len(valid_obs_x):min(len(valid_obs_x) + 1000, valid_x.shape[0])], components=to_units, lr=lr, iter=iters, method='exact'))
    valid_obs_x = np.array(valid_obs_x)

    test_obs_x = []
    while len(test_obs_x) < test_x.shape[0]:
        print 'test: ', len(test_obs_x), ':', test_x.shape[0]
        test_obs_x.extend(mf.tsne(learn_x, predict=valid_x[len(test_obs_x):min(len(test_obs_x) + 1000, test_x.shape[0])], components=to_units, lr=lr, iter=iters, method='exact'))
    test_obs_x = np.array(test_obs_x)

    learn, valid, test = (learn_obs_x, learn_y, learn_stat, learn_x, orly), (valid_obs_x, valid_y, valid_stat, valid_x, orvy), (test_obs_x, test_y, test_stat, test_x, orty)

    if dump_path is not None:
        pickle.dump((learn, valid, test), open(dump_path, 'wb'))

    return (learn, valid, test)

def prepare(paths, columns=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'], **params):

    # units=256, punits=1, period='D', db=20, vsplit=.4, tsplit=.65, norm=False, lvl=-1, dump_path=None

    '''

    :param data[pd.DataFrame]:
    :param units:
    :param period:
    :param vsplit:
    :param tsplit:
    :return:
    '''

    units = params['units']
    punits = params['punits']
    period = params['period']
    if params.get('order'):
        vsplit = params['vsplit']
        tsplit = params['tsplit']

    if not isinstance(paths, list):
        paths = [paths]

    tss = []
    for path in paths:

        data = pd.read_csv(path, names=columns, header=0)

        data = set_data_index(data, 'Date')
        data = data.drop(data.index[np.argwhere(data.index.weekday == 5).squeeze()])
        data = data[['Close']]
        data['Ret'] = np.log(data / data.shift())
        tss.append(data)

    def grouped(x, n):
        return np.vstack(filter(lambda obs: len(obs) == n * 6, map(lambda i: np.hstack(map(lambda s: s[1].tolist(), x[i:i+n])), range(0, len(x) - n, 1))))

    mtss = []
    price_key = 'price'
    ret_key = 'ret'
    for ts in tss:
        if period != 'ED':
            ix = pd.date_range(ts.index.min(), ts.index.max())
            ix = ix[ix.weekday != 5]
            ts = ts.reindex(ix, fill_value=np.nan)
            datas = {price_key : np.array(list(ts.resample('W-FRI').Close)), ret_key : np.array(list(ts.resample('W-FRI').Ret))}
            if datas.get(price_key)[-1][1].shape != 6:
                datas[price_key] = datas[price_key][1:datas[price_key].shape[0] - 1]
                datas[ret_key] = datas[ret_key][1:datas[ret_key].shape[0] - 1]
            else:
                datas[price_key] = datas[price_key][1:]
                datas[ret_key] = datas[ret_key][1:]
            assert datas[price_key].shape == datas[ret_key].shape
            mtss.append(datas)
        else:
            mtss.append({
                price_key : ts.Close.tolist()[1:],
                ret_key : ts.Ret.tolist()[1:]
            })
    xs, ys = {price_key : [], ret_key : []}, {price_key : [], ret_key : []}
    if period == 'DW':
        n = int(np.ceil(units / 6.) + punits)
        for ts in mtss:
            for k in xs.keys():
                xs[k].append(grouped(ts.get(k), n))
        for k in xs.keys():
            xs[k] = np.vstack(xs[k])[:, np.ceil(units / 6.) * 6 - units:]
            xs[k] = xs[k][range(0, xs[k].shape[0], punits)]
        ix = np.intersect1d(range(xs[price_key].shape[0]), np.argwhere(np.sum(np.isnan(xs[ret_key][:, :units]), axis=1) <= units / 3))
        ix = np.intersect1d(np.argwhere(np.sum(np.isnan(xs[ret_key][:, units:]), axis=1) <= punits), ix)
        for k in xs.keys():
            xs[k] = xs[k][ix].squeeze()
        # return
        xs[ret_key] = np.nan_to_num(xs[ret_key])
        ys[ret_key] = np.hstack(map(lambda x: np.sum(x, axis=1)[:, np.newaxis], np.hsplit(xs[ret_key][:, units:], np.arange(1, punits, 1) * 6 if punits > 1 else 1)))
        xs[ret_key] = xs[ret_key][:, :units]
        # prices
        i = 1
        while np.isnan(xs[price_key][:, 0]).any():
            xs[price_key][np.isnan(xs[price_key][:, 0]), 0] = xs[price_key][np.isnan(xs[price_key][:, 0]), i]
            i += 1
            if i == xs[price_key].shape[-1]:
                break

        for c in range(1, xs[price_key].shape[-1], 1):
            xs[price_key][np.isnan(xs[price_key][:, c]), c] = xs[price_key][np.isnan(xs[price_key][:, c]), c - 1]

        ys[price_key] = xs[price_key][:, range(units + 5, xs[price_key].shape[-1], 6)]
        xs[price_key] = xs[price_key][:, :units]

    elif period == 'W':
        n = int(units + punits)
        for ts in mtss:
            for k in xs.keys():
                xs[k].append(grouped(ts.get(k), n))
        for k in xs.keys():
            xs[k] = np.vstack(xs[k])
            xs[k] = xs[k][range(0, xs[k].shape[0], punits)]
        ix = np.intersect1d(range(xs[price_key].shape[0]), np.argwhere(np.sum(np.isnan(xs[ret_key][:, :units * 6]), axis=1) <= units * 2))
        ix = np.intersect1d(np.argwhere(np.sum(np.isnan(xs[ret_key][:, units * 6:]), axis=1) <= punits), ix)
        for k in xs.keys():
            xs[k] = xs[k][ix].squeeze()

        # return
        xs[ret_key] = np.nan_to_num(xs[ret_key])
        ys[ret_key] = np.hstack(map(lambda x: np.sum(x, axis=1)[:, np.newaxis], np.hsplit(xs[ret_key][:, units * 6:], np.arange(1, punits, 1) * 6 if punits > 1 else 1)))
        xs[ret_key] = np.hstack(map(lambda x: np.sum(x, axis=1)[:, np.newaxis], np.hsplit(xs[ret_key][:, :units * 6], np.arange(1, punits, 1) * 6 if punits > 1 else 1)))

        # prices
        i = 1
        while np.isnan(xs[price_key][:, 0]).any():
            xs[price_key][np.isnan(xs[price_key][:, 0]), 0] = xs[price_key][np.isnan(xs[price_key][:, 0]), i]
            i += 1
            if i == xs[price_key].shape[-1]:
                break

        for c in range(1, xs[price_key].shape[-1], 1):
            xs[price_key][np.isnan(xs[price_key][:, c]), c] = xs[price_key][np.isnan(xs[price_key][:, c]), c - 1]

        xs[price_key] = xs[price_key][:, range(5, n * 6, 6)]
        ys[price_key] = xs[price_key][:, units:]
        xs[price_key] = xs[price_key][:, :units]

    elif period == 'D':
        n = int(np.ceil((units) / 6.) + np.ceil((punits) / 6.))
        for ts in mtss:
            for k in xs.keys():
                xs[k].append(grouped(ts.get(k), n))
        for k in xs.keys():
            xs[k] = np.vstack(xs[k])[:, np.ceil(units / 6.) * 6 - units:np.ceil((units) / 6.) * 6 + punits]
            xs[k] = xs[k][range(0, xs[k].shape[0], int(np.ceil(punits / 6.)))]
        ix = np.intersect1d(range(xs[price_key].shape[0]), np.argwhere(np.sum(np.isnan(xs[ret_key][:, :units]), axis=1) <= units / 3))
        ix = np.intersect1d(np.argwhere(np.sum(np.isnan(xs[ret_key][:, units:]), axis=1) <= punits), ix)
        for k in xs.keys():
            xs[k] = xs[k][ix].squeeze()

        # return
        xs[ret_key] = np.nan_to_num(xs[ret_key])
        ys[ret_key] = np.hstack(map(lambda x: np.sum(x, axis=1)[:, np.newaxis], np.hsplit(xs[ret_key][:, units:], np.arange(1, punits, 1) * 6 if punits > 1 else 1)))
        xs[ret_key] = xs[ret_key][:, :units]

        # prices
        i = 1
        while np.isnan(xs[price_key][:, 0]).any():
            xs[price_key][np.isnan(xs[price_key][:, 0]), 0] = xs[price_key][np.isnan(xs[price_key][:, 0]), i]
            i += 1
            if i == xs[price_key].shape[-1]:
                break

        for c in range(1, xs[price_key].shape[-1], 1):
            xs[price_key][np.isnan(xs[price_key][:, c]), c] = xs[price_key][np.isnan(xs[price_key][:, c]), c - 1]

        ys[price_key] = xs[price_key][:, units:]
        xs[price_key] = xs[price_key][:, :units]
    elif period == 'ED':
        n = units + punits
        for ts in mtss:
            for k in xs.keys():
                x = ts.get(k)
                xs[k].append([x[i:i+n] for i in range(0, len(x) - n + 1, 1)])
        for k in xs.keys():
            xs[k] = np.vstack(xs[k])

        # return
        ys[ret_key] = xs[ret_key][:, -punits:]
        xs[ret_key] = xs[ret_key][:, :units]

        # prices
        ys[price_key] = xs[price_key][:, units:]
        xs[price_key] = xs[price_key][:, :units]

    if params.get('ysum'):
        ys[ret_key] = ys[ret_key].sum(1, keepdims=True)

    print 'x shape ', xs[price_key].shape, xs[ret_key].shape
    print 'y shape ',  ys[price_key].shape, ys[ret_key].shape

    if params.get('waves'):
        assert None not in [params.get('db'), params.get('lvl')]
        waves_all = wv.filter(xs[ret_key], params.get('lvl'), params.get('db'))

        cwaves_all = []
        mults = [1.42, 2., 2.828, 4., 5.655]
        for i in range(params.get('lvl')):
            cwaves_all.append(np.cumsum(waves_all[i]['A'] * mults[i], 1))

    if params.get('stats'):
        stats = data_stat(xs[ret_key], file_name=None, matrix=False)

        ex = ~np.isnan(stats).any(axis=1)
        for k in xs.keys():
            xs[k] = xs[k][np.where(ex)]
            ys[k] = ys[k][np.where(ex)]
        stats = stats[ex]

    if(params.get('order')):
        if ys[ret_key].shape[-1] == 1:
            oy = ys[ret_key].squeeze()
        else:
            oy = ys[ret_key].sum(axis=1).squeeze()

        ixs = np.arange(oy.shape[0])
        mult_std = params.get('mult_std') if isinstance(params.get('mult_std'), float) else 3.
        tix = np.append(np.argwhere(oy > oy.mean() + mult_std * oy.std()), np.argwhere(oy < oy.mean() - mult_std * oy.std()))
        ixs = np.delete(ixs, tix)

        var_length = params.get('var_length') if isinstance(params.get('var_length'), float) else min(12, xs[ret_key].shape[-1])
        my = oy[ixs] / xs[ret_key][ixs, var_length:].std(1)
        pix, zix, nix = np.argwhere(np.round(my, 1) >= .5).squeeze(), np.argwhere(abs(np.round(my, 1)) < .5).squeeze(), np.argwhere(
            np.round(my, 1) <= -.5).squeeze()
        count = min(len(pix), len(nix))

        if len(pix) == count:
            np.random.shuffle(nix)
            tix = np.append(tix, ixs[nix[count:]])
            nix = nix[:count]
        else:
            np.random.shuffle(pix)
            tix = np.append(tix, ixs[pix[count:]])
            pix = pix[:count]

        split_iter = params.get('split_iter') if isinstance(params.get('split_iter'), float) or isinstance(params.get('split_iter'), int) else 1000
        split_crit = params.get('split_crit') if isinstance(params.get('split_crit'), float) else .001
        print 'Positive'
        plix, pvix, ptix = _order_dist(my[pix], split_crit, split_iter, vsplit=vsplit, tsplit=tsplit, verbose=1)
        print 'Negative'
        nlix, nvix, ntix = _order_dist(my[nix], split_crit, split_iter, vsplit=vsplit, tsplit=tsplit, verbose=1)
        print 'Zero'
        zlix, zvix, ztix = _order_dist(my[zix], split_crit, split_iter, vsplit=vsplit, tsplit=tsplit, verbose=1)

        plix, pvix, ptix = ixs[pix[plix]], ixs[pix[pvix]], ixs[pix[ptix]]
        nlix, nvix, ntix = ixs[nix[nlix]], ixs[nix[nvix]], ixs[nix[ntix]]
        zlix, zvix, ztix = ixs[zix[zlix]], ixs[zix[zvix]], ixs[zix[ztix]]

        lix = np.hstack((plix, zlix, nlix))
        vix = np.hstack((pvix, zvix, nvix))
        tix = np.hstack((ptix, ztix, ntix, tix))

        for _ in range(10):
            np.random.shuffle(lix)
            np.random.shuffle(vix)
            np.random.shuffle(tix)

        data = {}
        # == learn ==
        learn = {
            'Price' : {'x' : xs[price_key][lix], 'y' : ys[price_key][lix]},
            'Return' : {'x' : xs[ret_key][lix], 'y' : ys[ret_key][lix]},
            'CReturn' : {'x' : xs[ret_key][lix].cumsum(1), 'y' : xs[ret_key][lix].sum(1, keepdims=True) + ys[ret_key][lix].cumsum(1)},
        }
        if params.get('waves'):
            learn['Waves'] = []
            learn['CWaves'] = []
            for w in waves_all:
                learn['Waves'].append({
                    'A' : w['A'][lix],
                    'D': w['D'][lix]
                })
            for w in cwaves_all:
                learn['CWaves'].append(w[lix])
        if params.get('stats'):
            learn['Stats'] = stats.ix[lix, :].to_dict()

        data['learn'] = learn
        # ===========

        # == valid ==
        valid = {
            'Price' : {'x' : xs[price_key][vix], 'y' : ys[price_key][vix]},
            'Return' : {'x' : xs[ret_key][vix], 'y' : ys[ret_key][vix]},
            'CReturn' : {'x' : xs[ret_key][vix].cumsum(1), 'y' : xs[ret_key][vix].sum(1, keepdims=True) + ys[ret_key][vix].cumsum(1)},
        }
        if params.get('waves'):
            valid['Waves'] = []
            valid['CWaves'] = []
            for w in waves_all:
                valid['Waves'].append({
                    'A' : w['A'][vix],
                    'D': w['D'][vix]
                })
            for w in cwaves_all:
                valid['CWaves'].append(w[vix])
        if params.get('stats'):
            valid['Stats'] = stats.ix[vix, :].to_dict()

        data['valid'] = valid
        # ===========

        # == test ==
        test = {
            'Price' : {'x' : xs[price_key][tix], 'y' : ys[price_key][tix]},
            'Return' : {'x' : xs[ret_key][tix], 'y' : ys[ret_key][tix]},
            'CReturn' : {'x' : xs[ret_key][tix].cumsum(1), 'y' : xs[ret_key][tix].sum(1, keepdims=True) + ys[ret_key][tix].cumsum(1)},
        }
        if params.get('waves'):
            test['Waves'] = []
            test['CWaves'] = []
            for w in waves_all:
                test['Waves'].append({
                    'A' : w['A'][tix],
                    'D': w['D'][tix]
                })
            for w in cwaves_all:
                test['CWaves'].append(w[tix])
        if params.get('stats'):
            test['Stats'] = stats.ix[tix, :].to_dict()

        data['test'] = test
        # ===========
    else:
        data = {}
        learn = {
            'Price' : {'x' : xs[price_key], 'y' : ys[price_key]},
            'Return' : {'x' : xs[ret_key], 'y' : ys[ret_key]},
            'CReturn' : {'x' : xs[ret_key].cumsum(1), 'y' : xs[ret_key].sum(1, keepdims=True) + ys[ret_key].cumsum(1)},
        }
        if params.get('waves'):
            learn['Waves'] = []
            learn['CWaves'] = []
            for w in waves_all:
                learn['Waves'].append({
                    'A' : w['A'],
                    'D': w['D']
                })
            for w in cwaves_all:
                learn['CWaves'].append(w)
        if params.get('stats'):
            learn['Stats'] = stats.to_dict()

        data['full'] = learn

    if isinstance(params.get('dump_path'), basestring):
        pickle.dump(data, open(gen_path(**params), 'wb'))

    return data

def gen_path(**params):
    dpath = params.get('dump_path')
    if data_path[-1] != '_':
        dpath += '_'
    dpath += ''.join(map(str, [params.get('units'), params.get('period'), params.get('punits')]))
    if params.get('waves'):
        dpath += '_' + str(params.get('db')) + 'db' + str(params.get('lvl'))
    so = ''
    if params.get('stats'):
        so += 'S'
    if params.get('order'):
        so += 'O'
    if len(so) > 0:
        dpath += '_' + so

    return dpath + '.data'