# -*- encoding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle
from functools import partial
import multiprocessing as mp

from scipy.stats import kruskal, levene, ttest_ind, kendalltau
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as mse

import wavelet as wv

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

def order_sign(y, vsplit=.8, tsplit=.4):

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

def order_dist(y, crit_stat=.01, max_iter=1000, vsplit=.8, tsplit=.4, verbose=1):

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

def smooth_net(obs_paths, target_paths, columns=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'], units=256, punits=1, period='D', db=20, vsplit=.8, tsplit=.8, lvl=-1, dump_path=None, **params):
        '''

        :param data[pd.DataFrame]:
        :param units:
        :param period:
        :param vsplit:
        :param tsplit:
        :return:
        '''

        if not isinstance(obs_paths, list) or not isinstance(target_paths, list):
            raise Exception('paths is not a list')

        oss = []
        for path in obs_paths:
            data = pd.read_csv(path, names=columns, header=0)

            data = set_data_index(data, 'Date')
            data = data.drop(data.index[np.argwhere(data.index.weekday == 5).squeeze()])
            data = data[['Close']]
            data['Ret'] = np.log(data / data.shift())[1:]
            oss.append(data)

        tss = []
        for path in obs_paths:
            data = pd.read_csv(path, names=columns, header=0)

            data = set_data_index(data, 'Date')
            data = data.drop(data.index[np.argwhere(data.index.weekday == 5).squeeze()])
            data = data[['Close']]
            data['Ret'] = np.log(data / data.shift())[1:]
            tss.append(data)

        price_key = 'price'
        ret_key = 'ret'
        oxs, oys = gen_obs(oss, units, punits, period, ret_key, price_key)
        txs, tys = gen_obs(tss, units, punits, period, ret_key, price_key)

        print 'ox shape ', oxs[price_key].shape, oxs[ret_key].shape
        print 'tx shape ', txs[price_key].shape, txs[ret_key].shape
        print 'oy shape ', oys[price_key].shape, oys[ret_key].shape
        print 'ty shape ', tys[price_key].shape, tys[ret_key].shape

        xwave = wv.smooth(oxs, lvl, db)
        twave = wv.smooth(txs, lvl, db)

        def correlations(target, obs):
            return np.array(map(lambda x: kendalltau(target, x)[0], obs))
        def svr(target, obs):
            model = SVR(kernel='poly', degree=5, C=100.)
            err = []
            target = target.cumsum()
            obs = obs.cumsum(1)[:, :, np.newaxis]
            for x in obs:
                model.fit(x, target)
                err.append(np.sqrt(mse(target, model.predict(x))))
            return np.array(err)

        partial_corr = partial(correlations, obs=xwave)
        if params.get('process'):
            processes = params.get('process')
        else:
            processes = 2
        pool = mp.Pool(processes)
        corr_map = pool.map_async(partial_corr, twave[:5])

        if dump_path is not None:
            pickle.dump((learn, valid, test), open(dump_path, 'wb'))

        return (learn, valid, test)

def prepare(paths, columns=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'], **params):

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

    price_key = 'price'
    ret_key = 'ret'
    xs, ys = gen_obs(tss, units, punits, period, ret_key, price_key)

    if params.get('ysum'):
        ys[ret_key] = ys[ret_key].sum(1, keepdims=True)

    print 'x shape ', xs[price_key].shape, xs[ret_key].shape
    print 'y shape ',  ys[price_key].shape, ys[ret_key].shape

    if params.get('waves'):
        assert None not in [params.get('db'), params.get('lvl')]
        waves_all = wv.filter(xs[ret_key], params.get('lvl'), params.get('db'))

        cwaves_all = []
        if params.get('mult'):
            mults = [1.42, 2., 2.828, 4., 5.655]
        else:
            mults = np.ones(5)
        for i in range(params.get('lvl')):
            cwaves_all.append(np.cumsum(waves_all[i]['A'] * mults[i], 1))

    if(params.get('order')):
        lix, vix, tix = _obs_order(xs, ys, ret_key, price_key, **params)

        data = {}
        # == learn ==
        learn = {
            'Price' : {'x' : xs[price_key][lix], 'y' : ys[price_key][lix]},
            'Return' : {'x' : xs[ret_key][lix], 'y' : ys[ret_key][lix]},
            'CReturn' : {'x' : xs[ret_key][lix].cumsum(1), 'y' : xs[ret_key][lix, :].sum(1, keepdims=True) + ys[ret_key][lix, :].cumsum(1)},
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

        data['learn'] = learn
        # ===========

        # == valid ==
        valid = {
            'Price' : {'x' : xs[price_key][vix], 'y' : ys[price_key][vix]},
            'Return' : {'x' : xs[ret_key][vix], 'y' : ys[ret_key][vix]},
            'CReturn' : {'x' : xs[ret_key][vix].cumsum(1), 'y' : xs[ret_key][vix, :].sum(1, keepdims=True) + ys[ret_key][vix, :].cumsum(1)},
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

        data['valid'] = valid
        # ===========

        # == test ==
        test = {
            'Price' : {'x' : xs[price_key][tix], 'y' : ys[price_key][tix]},
            'Return' : {'x' : xs[ret_key][tix], 'y' : ys[ret_key][tix]},
            'CReturn' : {'x' : xs[ret_key][tix].cumsum(1), 'y' : xs[ret_key][tix, :].sum(1, keepdims=True) + ys[ret_key][tix, :].cumsum(1)},
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

        data['full'] = learn

    if isinstance(params.get('dump_path'), basestring):
        pickle.dump(data, open(gen_path(**params), 'wb'))

    return data

def _obs_order(xs, ys, ret_key, price_key, vsplit, tsplit, **params):
    if ys[ret_key].shape[-1] == 1:
        oy = ys[ret_key].squeeze() + xs[ret_key].sum(1)
    else:
        oy = ys[ret_key].sum(axis=1).squeeze() + xs[ret_key].sum(1)

    ixs = np.arange(oy.shape[0])
    mult_std = params.get('mult_std') if isinstance(params.get('mult_std'), float) else 3.
    tix = np.append(np.argwhere(oy > oy.mean() + mult_std * oy.std()),
                    np.argwhere(oy < oy.mean() - mult_std * oy.std()))
    ixs = np.delete(ixs, tix)

    # var_length = params.get('var_length') if isinstance(params.get('var_length'), float) else min(12, xs[ret_key].shape[-1])
    my = oy[ixs]  # / xs[ret_key][ixs, var_length:].std(1)
    pix, nix = np.argwhere(my > 0).squeeze(), np.argwhere(my < 0).squeeze()
    count = min(len(pix), len(nix))

    if len(pix) == count:
        np.random.shuffle(nix)
        tix = np.append(tix, ixs[nix[count:]])
        nix = nix[:count]
    else:
        np.random.shuffle(pix)
        tix = np.append(tix, ixs[pix[count:]])
        pix = pix[:count]

    split_iter = params.get('split_iter') if isinstance(params.get('split_iter'), float) or isinstance(
        params.get('split_iter'), int) else 1000
    split_crit = params.get('split_crit') if isinstance(params.get('split_crit'), float) else .001
    print 'Positive'
    plix, pvix, ptix = order_dist(my[pix], split_crit, split_iter, vsplit=vsplit, tsplit=tsplit, verbose=1)
    print 'Negative'
    nlix, nvix, ntix = order_dist(my[nix], split_crit, split_iter, vsplit=vsplit, tsplit=tsplit, verbose=1)

    plix, pvix, ptix = ixs[pix[plix]], ixs[pix[pvix]], ixs[pix[ptix]]
    nlix, nvix, ntix = ixs[nix[nlix]], ixs[nix[nvix]], ixs[nix[ntix]]
    # zlix, zvix, ztix = ixs[zix[zlix]], ixs[zix[zvix]], ixs[zix[ztix]]

    lix = np.hstack((plix, nlix))
    vix = np.hstack((pvix, nvix))
    tix = np.hstack((ptix, ntix, tix))

    for _ in range(10):
        np.random.shuffle(lix)
        np.random.shuffle(vix)
        np.random.shuffle(tix)

    return lix, vix, tix

def gen_obs(tss, units, punits, period, ret_key, price_key):

    def grouped(x, n):
        return np.vstack(filter(lambda obs: len(obs) == n * 6, map(lambda i: np.hstack(map(lambda s: s[1].tolist(), x[i:i+n])), range(0, len(x) - n, 1))))

    mtss = []
    for ts in tss:
        if period != 'ED':
            ix = pd.date_range(ts.index.min(), ts.index.max())
            ix = ix[ix.weekday != 5]
            ts = ts.reindex(ix, fill_value=np.nan)
            datas = {price_key: np.array(list(ts.resample('W-FRI').Close)),
                     ret_key: np.array(list(ts.resample('W-FRI').Ret))}
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
                price_key: ts.Close.tolist()[1:],
                ret_key: ts.Ret.tolist()[1:]
            })
    xs, ys = {price_key: [], ret_key: []}, {price_key: [], ret_key: []}
    if period == 'DW':
        n = int(np.ceil(units / 6.) + punits)
        for ts in mtss:
            for k in xs.keys():
                xs[k].append(grouped(ts.get(k), n))
        for k in xs.keys():
            xs[k] = np.vstack(xs[k])[:, np.ceil(units / 6.) * 6 - units:]
            xs[k] = xs[k][range(0, xs[k].shape[0], punits)]
        ix = np.intersect1d(range(xs[price_key].shape[0]),
                            np.argwhere(np.sum(np.isnan(xs[ret_key][:, :units]), axis=1) <= units / 3))
        ix = np.intersect1d(np.argwhere(np.sum(np.isnan(xs[ret_key][:, units:]), axis=1) <= punits), ix)
        for k in xs.keys():
            xs[k] = xs[k][ix].squeeze()
        # return
        xs[ret_key] = np.nan_to_num(xs[ret_key])
        ys[ret_key] = np.hstack(map(lambda x: np.sum(x, axis=1)[:, np.newaxis], np.hsplit(xs[ret_key][:, units:],
                                                                                          np.arange(1, punits,
                                                                                                    1) * 6 if punits > 1 else 1)))
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
        ix = np.intersect1d(range(xs[price_key].shape[0]),
                            np.argwhere(np.sum(np.isnan(xs[ret_key][:, :units * 6]), axis=1) <= units * 2))
        ix = np.intersect1d(np.argwhere(np.sum(np.isnan(xs[ret_key][:, units * 6:]), axis=1) <= punits), ix)
        for k in xs.keys():
            xs[k] = xs[k][ix].squeeze()

        # return
        xs[ret_key] = np.nan_to_num(xs[ret_key])
        ys[ret_key] = np.hstack(map(lambda x: np.sum(x, axis=1)[:, np.newaxis], np.hsplit(xs[ret_key][:, units * 6:],
                                                                                          np.arange(1, punits,
                                                                                                    1) * 6 if punits > 1 else 1)))
        xs[ret_key] = np.hstack(map(lambda x: np.sum(x, axis=1)[:, np.newaxis], np.hsplit(xs[ret_key][:, :units * 6],
                                                                                          np.arange(1, punits,
                                                                                                    1) * 6 if punits > 1 else 1)))

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
        ix = np.intersect1d(range(xs[price_key].shape[0]),
                            np.argwhere(np.sum(np.isnan(xs[ret_key][:, :units]), axis=1) <= units / 3))
        ix = np.intersect1d(np.argwhere(np.sum(np.isnan(xs[ret_key][:, units:]), axis=1) <= punits), ix)
        for k in xs.keys():
            xs[k] = xs[k][ix].squeeze()

        # return
        xs[ret_key] = np.nan_to_num(xs[ret_key])
        ys[ret_key] = np.hstack(map(lambda x: np.sum(x, axis=1)[:, np.newaxis], np.hsplit(xs[ret_key][:, units:],
                                                                                          np.arange(1, punits,
                                                                                                    1) * 6 if punits > 1 else 1)))
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
                xs[k].append([x[i:i + n] for i in range(0, len(x) - n + 1, 1)])
        for k in xs.keys():
            xs[k] = np.vstack(xs[k])

        # return
        ys[ret_key] = xs[ret_key][:, -punits:]
        xs[ret_key] = xs[ret_key][:, :units]

        # prices
        ys[price_key] = xs[price_key][:, units:]
        xs[price_key] = xs[price_key][:, :units]

    return xs, ys

def gen_path(**params):
    dpath = params.get('dump_path')
    if dpath[-1] != '_':
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