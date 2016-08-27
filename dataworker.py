# -*- encoding: utf-8 -*-

import pandas as pd
import numpy as np
import pickle
from functools import partial
import multiprocessing as mp

from scipy.stats import kruskal, levene, ttest_ind, kendalltau, pearsonr
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error as mse

import wavelet as wv
import operators as ops

def load_frame(path, ix_columns, ix_format, columns=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume'], **params):
    data = pd.read_csv(path, names=columns, header=0)
    columns = set(columns)
    if params.get('pre_delete'):
        params['pre_delete'] = set(params['pre_delete'])
        if set(params.get('pre_delete')).difference(columns):
            raise Exception('pre_delete columns don\'t exist in DataFrame')
        for column in params.get('pre_delete'):
            columns.remove(column)
            del data[column]

    if params.get('preformat'):

        if not isinstance(params.get('preformat'), dict):
            raise Exception('preformat is not a <columns : format> dict')
        elif set(params.get('preformat').keys()).difference(columns):
            raise Exception('preformat columns don\'t exist in DataFrame')

        for c, f in params.get('preformat').items():
            data[c] = map(f.format, data[c])

    if not hasattr(ix_columns, '__iter__') and isinstance(ix_columns, str):
        ix_columns = [ix_columns]

    if set(ix_columns).difference(columns):
        raise Exception('There is no such ix columns in DataFrame')
    if not isinstance(ix_format, str):
        raise Exception('ix_format must be string pattern')

    for column in ix_columns:
        data[column] = data[column].apply(str)

    data['dtix'] = map(''.join, data[ix_columns].as_matrix())
    data = data.set_index(pd.DatetimeIndex(pd.to_datetime(data['dtix'], format=ix_format)))
    del data['dtix']

    for column in ix_columns:
        del data[column]

    if params.get('drop_wd'):
        if not hasattr(params.get('drop_wd'), '__iter__'):
            params['drop_wd'] = [params['drop_wd']]
        params['drop_wd'] = set(params['drop_wd'])
        for v in params['drop_wd']:
            if not isinstance(isinstance(v, int), int):
                raise Exception('drop weekdays must be 0 <= x < 6 integers')
            elif v < 0 or v >= 6:
                raise Exception('drop weekdays must be 0 <= x < 6 integers')
            else:
                data = data.drop(data.index[np.argwhere(data.index.weekday == v).squeeze()])

    if params.get('logret'):
        if not isinstance(params.get('logret'), dict):
            raise Exception('logret is not a <column : levels> dict')
        elif set(params.get('preformat').keys()).difference(columns):
            raise Exception('logret columns don\'t exist in DataFrame')

        for column in params.get('logret').keys():
            if not hasattr(params.get('logret')[column], '__iter__'):
                params['logret'][column] = [params['logret'][column]]

            for v in params.get('logret')[column]:
                if not isinstance(isinstance(v, int), int):
                    raise Exception('drop weekdays must be 0 <= x < 6 integers')
                else:
                    data[column + 'LogRet' + str(v)] = np.log(data[column] / data[column].shift(v))

    if params.get('post_delete'):
        params['post_delete'] = set(params['post_delete'])
        if set(params.get('post_delete')).difference(columns):
            raise Exception('post_delete columns don\'t exist in DataFrame')
        for column in params.get('post_delete'):
            columns.remove(column)
            del data[column]

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

# def prepare(path, units, punits, period, columns=['Date', 'Time', 'Open', 'High', 'Low', 'Close', 'Volume']  **params):
#
#     # keys validation
#     assert len(set(params.keys()).difference([
#         'order',
#         'vsplit',
#         'tsplit',
#         'operators',
#         'dump_path'
#     ]))
#
#     if params.get('order'):
#         vsplit = params['vsplit']
#         tsplit = params['tsplit']
#
#     frame = load_frame(path, columns)
#     ckeys = frame.columns
#     xs, ys = gen_obs(frame, units, punits, period)
#
#     if params.get('ysum'):
#         ys[ret_key] = ys[ret_key].sum(1, keepdims=True)
#
#     print 'x shape ', xs[price_key].shape, xs[ret_key].shape
#     print 'y shape ',  ys[price_key].shape, ys[ret_key].shape
#
#     if params.get('operators'):
#         if not isinstance(params.get('operators'), dict):
#             raise Exception('Operators must be <"name" : "alias"|function> like dict')
#
#         ids = {}
#         for cop_key, cop_value in params.get('operators'):
#             try:
#                 if isinstance(cop_value, str):
#                     op_call = getattr(ops, cop_value)
#                 elif hasattr(cop_value, '__call__'):
#                     op_call = cop_value
#
#                 ids[cop_key] = op_call(xs)
#
#             except Exception, e:
#                 print e.message
#
#
#
#     if params.get('waves'):
#         assert None not in [params.get('db'), params.get('lvl')]
#         waves_all = wv.filter(xs[ret_key], params.get('lvl'), params.get('db'))
#
#         cwaves_all = []
#         if params.get('mult'):
#             mults = [1.42, 2., 2.828, 4., 5.655]
#         else:
#             mults = np.ones(5)
#         for i in range(params.get('lvl')):
#             cwaves_all.append(np.cumsum(waves_all[i]['A'] * mults[i], 1))
#
#     if(params.get('order')):
#         lix, vix, tix = _obs_order(xs, ys, ret_key, price_key, vslpit, tsplit, **params)
#
#         data = {}
#         # == learn ==
#         learn = {
#             'Price' : {'x' : xs[price_key][lix], 'y' : ys[price_key][lix]},
#             'Return' : {'x' : xs[ret_key][lix], 'y' : ys[ret_key][lix]},
#             'CReturn' : {'x' : xs[ret_key][lix].cumsum(1), 'y' : xs[ret_key][lix, :].sum(1, keepdims=True) + ys[ret_key][lix, :].cumsum(1)},
#         }
#         if params.get('waves'):
#             learn['Waves'] = []
#             learn['CWaves'] = []
#             for w in waves_all:
#                 learn['Waves'].append({
#                     'A' : w['A'][lix],
#                     'D': w['D'][lix]
#                 })
#             for w in cwaves_all:
#                 learn['CWaves'].append(w[lix])
#
#         data['learn'] = learn
#         # ===========
#
#         # == valid ==
#         valid = {
#             'Price' : {'x' : xs[price_key][vix], 'y' : ys[price_key][vix]},
#             'Return' : {'x' : xs[ret_key][vix], 'y' : ys[ret_key][vix]},
#             'CReturn' : {'x' : xs[ret_key][vix].cumsum(1), 'y' : xs[ret_key][vix, :].sum(1, keepdims=True) + ys[ret_key][vix, :].cumsum(1)},
#         }
#         if params.get('waves'):
#             valid['Waves'] = []
#             valid['CWaves'] = []
#             for w in waves_all:
#                 valid['Waves'].append({
#                     'A' : w['A'][vix],
#                     'D': w['D'][vix]
#                 })
#             for w in cwaves_all:
#                 valid['CWaves'].append(w[vix])
#
#         data['valid'] = valid
#         # ===========
#
#         # == test ==
#         test = {
#             'Price' : {'x' : xs[price_key][tix], 'y' : ys[price_key][tix]},
#             'Return' : {'x' : xs[ret_key][tix], 'y' : ys[ret_key][tix]},
#             'CReturn' : {'x' : xs[ret_key][tix].cumsum(1), 'y' : xs[ret_key][tix, :].sum(1, keepdims=True) + ys[ret_key][tix, :].cumsum(1)},
#         }
#         if params.get('waves'):
#             test['Waves'] = []
#             test['CWaves'] = []
#             for w in waves_all:
#                 test['Waves'].append({
#                     'A' : w['A'][tix],
#                     'D': w['D'][tix]
#                 })
#             for w in cwaves_all:
#                 test['CWaves'].append(w[tix])
#
#         data['test'] = test
#         # ===========
#     else:
#         data = {}
#         learn = {
#             'Price' : {'x' : xs[price_key], 'y' : ys[price_key]},
#             'Return' : {'x' : xs[ret_key], 'y' : ys[ret_key]},
#             'CReturn' : {'x' : xs[ret_key].cumsum(1), 'y' : xs[ret_key].sum(1, keepdims=True) + ys[ret_key].cumsum(1)},
#         }
#         if params.get('waves'):
#             learn['Waves'] = []
#             learn['CWaves'] = []
#             for w in waves_all:
#                 learn['Waves'].append({
#                     'A' : w['A'],
#                     'D': w['D']
#                 })
#             for w in cwaves_all:
#                 learn['CWaves'].append(w)
#
#         data['full'] = learn
#
#     if isinstance(params.get('dump_path'), basestring):
#         pickle.dump(data, open(gen_path(**params), 'wb'))
#
#     return data

def _obs_order(xs, ys, ret_key, price_key, vslpit, tsplit, **params):
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

def gen_obs(frame, units, punits, period, **params):

    assert period in ['DW', 'W', 'D', 'ED']
    if period in ['DW', 'W', 'D'] and not params.get('resample'):
        raise Exception('resample must be specified for [DW, W, D]')
    if period == 'D' and not params.get('day_number') and 0 <= int(params.get('day_number')) <= 6:
        raise Exception('For D period day_number must be in [0, 1, 2, 3, 4, 5, 6]')
    if not params.get('order_nan'):
        params['order_nan'] = dict()
    else:
        if not isinstance(params['order_nan'], dict):
            raise Exception('order_nan must be a dict')
        elif set(params['order_nan'].keys()).difference(frame.columns):
            raise Exception('Unknown columns in order_nan')

    def grouped(x, w, n):
        return np.vstack(filter(lambda obs: len(obs) == n * w, map(lambda i: np.hstack(map(lambda s: s[1].tolist(), x[i:i+n])), range(0, len(x) - n, 1))))

    def _set_nan_obs(ys, key, w):
        if period == 'W':
            ys[key] = np.hstack(map(lambda x: np.sum(x, axis=1)[:, np.newaxis], np.hsplit(xs[key][:, units * w:], np.arange(1, punits, 1) * w if punits > 1 else 1)))
            xs[key] = np.hstack(map(lambda x: np.sum(x, axis=1)[:, np.newaxis], np.hsplit(xs[key][:, :units * w], np.arange(1, units, 1) * w if units > 1 else 1)))
        elif period == 'DW':
            ys[key] = np.hstack(map(lambda x: np.sum(x, axis=1)[:, np.newaxis], np.hsplit(xs[key][:, units:], np.arange(1, punits, 1) * w if punits > 1 else 1)))
            xs[key] = xs[key][:, :units]
        elif period == 'D':
            dn = int(params['day_number']) - 1
            ys[key] = np.hstack(map(lambda x: x[:, dn], np.hsplit(xs[key][:, units:], np.arange(1, punits, 1) * w if punits > 1 else 1)))
            xs[key] = xs[key][:, :units]
        elif period == 'ED':
            ys[key] = xs[key][:, units:]
            xs[key] = xs[key][:, :units]

    def _set_prev_obs(ys, key, w):
        if period == 'W':
            xs[key] = xs[key][:, range(w-1, (units + punits) * w, w)]
            ys[key] = xs[key][:, units:]
            xs[key] = xs[key][:, :units]
        elif period == 'DW':
            ys[key] = xs[key][:, units:][:, range(w-1, punits * w, w)]
            xs[key] = xs[key][:, :units]
        elif period == 'D':
            dn = int(params['day_number'])
            ys[key] = xs[key][:, units:][:, range(dn, punits * w, w)]
            xs[key] = xs[key][:, :units]
        elif period == 'ED':
            ys[key] = xs[key][:, units:]
            xs[key] = xs[key][:, :units]

    def _fix_nan(n, w):
        ys = {}
        for key in xs.keys():
            if key in params.get('order_nan').keys():
                if params.get('order_nan')[key] == 'nan':
                    xs[key] = np.nan_to_num(xs[key])

                    _set_nan_obs(ys, key, w)
                elif params.get('order_nan')[key] == 'previous':
                    i = 1
                    while np.isnan(xs[key][:, 0]).any():
                        xs[key][np.isnan(xs[key][:, 0]), 0] = xs[key][np.isnan(xs[key][:, 0]), i]
                        i += 1
                        if i == xs[key].shape[-1]:
                            break

                    for c in range(1, xs[key].shape[-1], 1):
                        xs[key][np.isnan(xs[key][:, c]), c] = xs[key][np.isnan(xs[key][:, c]), c - 1]

                    _set_prev_obs(ys, key, w)
            elif (xs[key] <= 0).any():
                xs[key] = np.nan_to_num(xs[key])

                _set_nan_obs(ys, key, w)
            else:
                i = 1
                while np.isnan(xs[key][:, 0]).any():
                    xs[key][np.isnan(xs[key][:, 0]), 0] = xs[key][np.isnan(xs[key][:, 0]), i]
                    i += 1
                    if i == xs[key].shape[-1]:
                        break

                for c in range(1, xs[key].shape[-1], 1):
                    xs[key][np.isnan(xs[key][:, c]), c] = xs[key][np.isnan(xs[key][:, c]), c - 1]

                _set_prev_obs(ys, key, w)

        return xs, ys

    xs, ys = {}, {}
    if period == 'ED':
        n = units + punits
        for column in frame.columns:
            x = frame[column].tolist()
            xs[column] = np.vstack([x[i:i + n] for i in range(0, len(x) - n + 1, 1)])

        ix = np.arange(xs[xs.keys()[-1]].shape[0])
        for key in xs.keys():
            ix = np.intersect1d(ix, np.argwhere(np.sum(np.isnan(xs[key][:, :units]), axis=1) <= units / 3))

        for key in xs.keys():
            xs[key] = xs[key][ix]

        xs, ys = _fix_nan(n, None)
    elif params.get('resample'):
        data = {}
        rsmpl = 'W-' + params.get('resample')
        ix = pd.date_range(frame.index.min(), frame.index.max())
        frame = frame.reindex(ix, fill_value=np.nan)
        weekends = params.get('weekends') if params.get('weekends') else [5,6]
        for w in weekends:
            frame = frame.ix[frame.index.weekday != w]
        c = np.argwhere(frame.index.weekday == {
            'MON' : 1,
            'TUE' : 2,
            'WED' : 3,
            'THU' : 4,
            'FRI' : 5,
            'SAT' : 6,
            'SUN' : 0
        }[params.get('resample')])[0][0]
        frame = frame.ix[c:]
        c = np.argwhere(frame.index.weekday == 4)[-1][0]
        frame = frame.ix[:c+1]
        w = np.unique(frame.index.weekday).shape[0]
        resampler = frame.resample(rsmpl)
        for column in frame.columns:
            data[column] = np.array(list(resampler[column]))

        if period == 'DW':
            n = int(np.ceil(units / 6.) + punits)
            for column in frame.columns:
                xs[column] = np.vstack(grouped(data[column], w, n))[:, np.ceil(units / float(w)) * w - units:]
                xs[column] = xs[column][range(0, xs[column].shape[0], punits)]

            ix = np.arange(xs[xs.keys()[-1]].shape[0])
            for key in  xs.keys():
                ix = np.intersect1d(ix, np.argwhere(np.sum(np.isnan(xs[key][:, :units]), axis=1) <= units / 3))

            for key in xs.keys():
                xs[key] = xs[key][ix]

            xs, ys = _fix_nan(n, w)

        elif period == 'W':
            n = int(units + punits)
            for column in frame.columns:
                xs[column] = np.vstack(grouped(data[column], w, n))
                xs[column] = xs[column][range(0, xs[column].shape[0], punits)]

            ix = np.arange(xs[xs.keys()[-1]].shape[0])
            for key in xs.keys():
                ix = np.intersect1d(ix, np.argwhere(np.sum(np.isnan(xs[key][:, :units]), axis=1) <= units))

            for key in xs.keys():
                xs[key] = xs[key][ix]

            xs, ys = _fix_nan(n, w)

        elif period == 'D':
            n = int(np.ceil((units) / 6.) + np.ceil((punits) / 6.))
            for column in frame.columns:
                xs[column] = np.vstack(grouped(data[column], w, n))[:, np.ceil(units / float(w)) * w - units:]
                xs[column] = xs[column][range(0, xs[column].shape[0], punits)]

            ix = np.arange(xs[xs.keys()[-1]].shape[0])
            for key in xs.keys():
                ix = np.intersect1d(ix, np.argwhere(np.sum(np.isnan(xs[key][:, :units]), axis=1) <= units / 3))

            for key in xs.keys():
                xs[key] = xs[key][ix]

            xs, ys = _fix_nan(n, w)

    else:
        raise Exception('Specify ED or S period or resample for other periods')

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

def order_similarity(x_obs, x_target, processes=2):

    metrics = []
    if len(x_target.shape) == 1:
        x_target = x_target[:, np.newaxis]

    def corr(obs, target):
        ppearsonr = partial(pearsonr, y=target)
        pool = mp.Pool(processes)
        mapres = pool.map_async(ppearsonr, obs)
        while not mapres.ready():
            mapres.wait(1.)
        res = mapres.get()
        pool.close()
        pool.join()
        return [v[0] for v in res]

    for target in x_target[:10]:
        metrics.append(corr(x_obs, target))

    print np.vstack(metrics)