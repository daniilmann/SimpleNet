# -*- encoding: utf-8 -*-

import multiprocessing as mp
import pickle
import sys
import warnings
from functools import partial
from os import mkdir, rmdir, listdir
from os.path import exists, join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.pyplot import savefig
from scipy.stats import kruskal, levene, ttest_ind, kendalltau, pearsonr, randint
from sklearn.grid_search import RandomizedSearchCV as RSCV
from sklearn.metrics import mean_squared_error as mse, classification_report, confusion_matrix
from sklearn.preprocessing import MaxAbsScaler, PolynomialFeatures
from sklearn.svm import SVR, SVC

import configs as cfg
import operators as ops
import wavelet as wv
from classifiers import trees as cs

warnings.filterwarnings('ignore')

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

def prepare(frame_params, obs_params, **params):

    # keys validation
    assert len(set(params.keys()).difference([
        'order',
        'operators'
    ]))

    if params.get('order'):
        vsplit = params['vsplit']
        tsplit = params['tsplit']

    frame = load_frame(**frame_params)
    xs, ys = gen_obs(frame, **obs_params)

    for key in xs.keys():
        print key, '  x: ', xs[key], ' || y: ', ys[key].shape

    if params.get('operators'):
        if not isinstance(params.get('operators'), dict):
            raise Exception('Operators must be <"name" : (column, "alias"|function)>-like dict')

        ids = {}
        for cop_key, cop_value in params.get('operators'):
            try:
                if isinstance(cop_value, str):
                    op_call = getattr(ops, cop_value)
                elif hasattr(cop_value, '__call__'):
                    op_call = cop_value

                ids[cop_key] = op_call(xs)

            except Exception, e:
                print e.message

    data = {}
    data['origin'] = {'x': dict(), 'y': dict()}
    for key in xs.keys():
        data['origin']['x'][key] = xs[key]
    for key in ys.keys():
        data['origin']['y'][key] = ys[key]

    if(params.get('order')):
        lix, vix, tix = _obs_order(ys, **params)

        # == learn ==
        learn = {'x' : dict(), 'y' : dict()}
        for key in xs.keys():
            learn['x'][key] = xs[key][lix]
        for key in ys.keys():
            learn['y'][key] = ys[key][lix]
        data['learn'] = learn
        # ===========

        # == valid ==
        valid = {'x' : dict(), 'y' : dict()}
        for key in xs.keys():
            valid['x'][key] = xs[key][vix]
        for key in ys.keys():
            valid['y'][key] = ys[key][vix]
        data['valid'] = valid
        # ===========

        # == test ==
        if len(lix) > 0:
            test = {'x': dict(), 'y': dict()}
            for key in xs.keys():
                test['x'][key] = xs[key][tix]
            for key in ys.keys():
                test['y'][key] = ys[key][tix]
            data['test'] = test
        # ===========

    if params.get('dump_name'):
        # TODO: clean folder before mk
        path = join(cfg.dump_path, 'obs', params['dump_name'])
        if exists(path):
            rmdir(path)
        mkdir(path)

        # origin
        opath = join(path, 'origin')
        xpath = join(opath, 'x')
        mkdir(xpath)
        ypath = join(opath, 'y')
        mkdir(ypath)
        for key in xs.keys():
            pickle.dump(data['origin']['x'][key], join(xpath, key + '.pkl'))
        for key in ys.keys():
            pickle.dump(data['origin']['y'][key], join(ypath, key + '.pkl'))

        # ordered
        opath = join(path, 'ordered')
        xpath = join(opath, 'x')
        mkdir(xpath)
        ypath = join(opath, 'y')
        mkdir(ypath)
        for key in xs.keys():
            pickle.dump(data['ordered']['x'][key], join(xpath, key + '.pkl'))
        for key in ys.keys():
            pickle.dump(data['ordered']['y'][key], join(ypath, key + '.pkl'))

    return data

def _obs_order(ys, vslpit, tsplit, **params):
    y = ys[params['y']]

    if len(y.shape) != 1:
        y = y.squeeze()
    if len(y.shape) != 1:
        raise Exception('y could not be squeezed')


    ixs = np.arange(y.shape[0])
    mult_std = params.get('mult_std') if isinstance(params.get('mult_std'), float) else 3.
    tix = np.append(np.argwhere(oy > oy.mean() + mult_std * oy.std()),
                    np.argwhere(oy < oy.mean() - mult_std * oy.std()))
    ixs = np.delete(ixs, tix)

    my = y[ixs]
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

    split_iter = int(params.get('split_iter')) if isinstance(params.get('split_iter'), float) or isinstance(
        params.get('split_iter'), int) else 1000
    split_crit = params.get('split_crit') if isinstance(params.get('split_crit'), float) else .001
    print 'Positive'
    plix, pvix, ptix = order_dist(my[pix], split_crit, split_iter, vsplit=vsplit, tsplit=tsplit, verbose=1)
    print 'Negative'
    nlix, nvix, ntix = order_dist(my[nix], split_crit, split_iter, vsplit=vsplit, tsplit=tsplit, verbose=1)

    plix, pvix, ptix = ixs[pix[plix]], ixs[pix[pvix]], ixs[pix[ptix]]
    nlix, nvix, ntix = ixs[nix[nlix]], ixs[nix[nvix]], ixs[nix[ntix]]

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
    if params.get('tools'):
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

def _causality(frame, mx, tlen, counter, lock):
    ticker, frame = frame
    def cr2(tmp, kernel, pf):
        learn = tmp[c1].as_matrix()
        if len(learn.shape) == 1:
            learn = learn[:, np.newaxis]
        if pf:
            learn = PolynomialFeatures(degree=max(2, len(c1)), include_bias=False).fit_transform(learn)
            if len(c1) > 1:
                learn = np.hstack((learn, tmp[c1].as_matrix().sum(1, keepdims=True)))
        tmp[c2] = np.sign(tmp[c2])
        params = {
            'C': [.00001, .0001, .001, .01, .1, 1., 10., 100., 1000., 10000.],
            'kernel': [kernel],
            'class_weight': [None, 'balanced']
        }
        if kernel != 'linear':
            params['gamma'] = [.1e-4, 1e-3, 1e-2, 1e-1, .5, .9]
        elif kernel == 'poly':
            params['degree'] = randint(1, 11)
        elif kernel == 'sigmoid':
            tmp[c2] = (tmp[c2] + 1) / 2.
        model = SVC(max_iter=100)
        opt = RSCV(model, param_distributions=params, scoring='accuracy', verbose=0, n_iter=5, n_jobs=2, pre_dispatch=2)
        opt.fit(learn, tmp[c2].tolist())
        return np.sum(opt.predict(learn) == tmp[c2].tolist()) / float(tmp.shape[0])

    def rr2(tmp, kernel, pf=False):
        learn = tmp[c1].as_matrix()
        if len(learn.shape) == 1:
            learn = learn[:, np.newaxis]
        if pf:
            learn = PolynomialFeatures(degree=max(2, len(c1)), include_bias=False).fit_transform(learn)
            if len(c1) > 1:
                learn = np.hstack((learn, tmp[c1].as_matrix().sum(1, keepdims=True)))
        params = {
            'C': [.00001, .0001, .001, .01, .1, 1., 10., 100., 1000., 10000.],
            'kernel': [kernel]
        }
        if kernel != 'linear':
            params['gamma'] = [.1e-4, 1e-3, 1e-2, 1e-1, .5, .9]
        elif kernel == 'poly':
            params['degree'] = randint(1, 11)
        model = SVR(max_iter=100)
        opt = RSCV(model, param_distributions=params, scoring='mean_squared_error', verbose=0, n_iter=5, n_jobs=2,
                   pre_dispatch=2)
        opt.fit(learn, tmp[c2].tolist())
        return mse(tmp[c2].tolist(), opt.predict(learn))

    def corr(tmp):
        clmns = tmp.columns
        tmp['Ret1'] = tmp.Ret.shift()
        tmp = tmp.ix[1:]
        res = np.corrcoef(tmp[clmns[0]], tmp.Ret)[0, 1]
        a = np.pi / 8.
        r = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
        rframe = np.vstack(map(lambda x: np.dot(r, x.transpose()), tmp[[clmns[0], 'Ret1']].as_matrix()))
        c = np.corrcoef(rframe[:, 0], tmp.Ret)[0, 1]
        res = c if abs(c) > abs(res) else res
        a = np.pi / 4.
        rframe = np.vstack(map(lambda x: np.dot(r, x.transpose()), tmp[[clmns[0], 'Ret1']].as_matrix()))
        c = np.corrcoef(rframe[:, 0], tmp.Ret)[0, 1]
        a = np.pi / 2.
        rframe = np.vstack(map(lambda x: np.dot(r, x.transpose()), tmp[[clmns[0], 'Ret1']].as_matrix()))
        c = np.corrcoef(rframe[:, 0], tmp.Ret)[0, 1]
        res = c if abs(c) > abs(res) else res
        return res

    params = {
        'NDiffRClass': {
            'c1'    : 'NDiff',
            'c2'    : 'Ret',
            'err'   : 'rcr2',
        },
        'MDiffRClass': {
            'c1': ['NDiff', 'NDiff1'],
            'c2': 'Ret',
            'err': 'rcr2',
        },
        'NDiffPFRReg': {
            'c1': 'NDiff',
            'c2': 'Ret',
            'err': 'rrr2',
        },
        'MDiffPFRReg': {
            'c1': ['NDiff', 'NDiff1'],
            'c2': 'Ret',
            'err': 'rrr2',
        },
        'ValueCorr': {
            'c1': 'value',
            'c2': 'Ret',
            'err': 'corr',
        },
        'DiffCorr': {
            'c1': 'Diff',
            'c2': 'Ret',
            'err': 'corr',
        },
        'NDiffCorr': {
            'c1': 'NDiff',
            'c2': 'Ret',
            'err': 'corr',
        },
        'LDiffCorr': {
            'c1': 'LDiff',
            'c2': 'Ret',
            'err': 'corr',
        },
        'NLDiffCorr': {
            'c1': 'NLDiff',
            'c2': 'Ret',
            'err': 'corr',
        },
        'DiffSign': {
            'c1': 'Diff',
            'c2': 'Ret',
            'err': 'sign',
        },
        'LDiffSign': {
            'c1': 'LDiff',
            'c2': 'Ret',
            'err': 'sign',
        },
        'DiffNSign': {
            'c1': 'Diff',
            'c2': 'Ret',
            'err': 'nsign',
        },
        'LDiffNSign': {
            'c1': 'LDiff',
            'c2': 'Ret',
            'err': 'nsign',
        },
        'DiffRet': {
            'c1': 'Diff',
            'c2': 'Ret',
            'err': 'ret',
        },
        'DiffNRet': {
            'c1': 'Diff',
            'c2': 'Ret',
            'err': 'nret',
        },
        'LDiffRet': {
            'c1': 'LDiff',
            'c2': 'Ret',
            'err': 'ret',
        },
        'LDiffNRet': {
            'c1': 'LDiff',
            'c2': 'Ret',
            'err': 'nret',
        },
        'NDiffRet': {
            'c1': 'NDiff',
            'c2': 'Ret',
            'err': 'ret',
        },
        'NDiffNRet': {
            'c1': 'NDiff',
            'c2': 'Ret',
            'err': 'nret',
        },
    }
    result = {}
    for name, prms in params.items():
        c1 = prms['c1']
        c2 = prms['c2']
        op = prms['err']
        pf = True
        try:
            if isinstance(c1, str):
                tmp = frame[[c1, c2]]
                tmp = tmp.ix[np.isfinite(tmp.as_matrix()).all(1)]
                tmp = tmp.ix[tmp[c1] != 0.]
                c1 = [c1]
            else:
                c = c1
                c.append(c2)
                tmp = frame[c]
                tmp = tmp.ix[np.isfinite(tmp.as_matrix()).all(1)]
                tmp = tmp.ix[(tmp[c1].as_matrix() != 0.).all(1)]

            res = None
            if op == 'corr':
                res = corr(tmp)
            elif op == 'ret':
                res = np.sum(np.sign(tmp[c1[0]]) * tmp[c2])
            elif op == 'nret':
                res = np.sum(-np.sign(tmp[c1[0]]) * tmp[c2])
            elif op == 'sign':
                res = np.sum(np.sign(tmp[c1[0]]) == np.sign(tmp[c2])) / float(tmp.shape[0])
            elif op == 'nsign':
                res = np.sum(-np.sign(tmp[c1[0]]) == np.sign(tmp[c2])) / float(tmp.shape[0])
            elif op == 'lcr2':
                res = cr2(tmp, 'linear', pf)
            elif op == 'rcr2':
                res = cr2(tmp, 'rbf', pf)
            elif op == 'pcr2':
                res = cr2(tmp, 'poly', pf)
            elif op == 'scr2':
                res = cr2(tmp, 'sigmoid', pf)
            elif op == 'rrr2':
                res = rr2(tmp, 'rbf', pf) / (float(tmp.shape[0]) / tlen)
            elif op == 'prr2':
                res = rr2(tmp, 'poly', pf) / (float(tmp.shape[0]) / tlen)

            if op in ['rrr2', 'prr2']:
                result[name] = res * frame.shape[0]
            else:
                result[name] = (float(tmp.shape[0]) / tlen) * res
        except Exception, err:
            result[name] = np.nan

    with lock:
        counter.value += 1
        sys.stdout.write('\r%4s / %4s' % (counter.value, mx))
    sys.stdout.flush()

    return {ticker : result}

def causality(target, path, exclude, dname, extension='csv'):
    print 'Causality start'
    target['Ret'] = np.log(target.Value / target.Value.shift())
    target = target.ix[2:]

    files = listdir(path)
    files = filter(lambda f: f[-3:] == 'csv', listdir(path))
    files.remove(exclude)
    frames = {}
    for f in files:#['BAMLH0A0HYM2.csv', 'DGS10.csv']: #
        frame = load_frame(join(path, f), 'date', '%Y-%m-%d', ['date', 'value'])
        frame = frame.dropna()
        frame['Diff'] = frame.value - frame.value.shift()
        frame['NDiff'] = frame.Diff
        frame.NDiff.ix[~np.isfinite(frame.Diff)] = 0.
        frame['NDiff'] = MaxAbsScaler().fit_transform(frame.NDiff)
        frame['NDiff1'] = frame.NDiff.shift()
        frame['Dir'] = np.sign(frame.Diff)
        if not (frame.value <= 0).any():
            frame['LDiff'] = np.round(np.log(frame.value / frame.value.shift()), 4)
            frame['NLDiff'] = frame.LDiff
            frame.NLDiff.ix[~np.isfinite(frame.LDiff)] = 0.
            frame['NLDiff'] = MaxAbsScaler().fit_transform(frame.NLDiff)
        frame = frame.shift()
        if pd.concat((frame, target), axis=1).dropna().shape[0] / float(target.shape[0]) >= .5:
            frames[f[:-4]] = frame

    print 'Frames count: ', len(frames.keys())
    results = {}
    obs = {}
    for name, index in frames.items():
        obs[name] = pd.concat((index, target), axis=1)
    print 'Start processing'
    m = mp.Manager()
    pcaus = partial(_causality, mx=len(obs), tlen=float(target.shape[0]), counter=m.Value('i', 0), lock=m.Lock())
    pool = mp.Pool(2)
    res = pool.map_async(pcaus, obs.items())
    pool.close()
    pool.join()

    for r in res.get():
        results.update(r)
    results = pd.DataFrame.from_dict(results).transpose()
    results.to_csv(join(cfg.dump_path, 'indcomp', dname))
    print '\nEnd processing'

def fetature_significance(x, y, ixs=None, n_f=10, opt=True, n_estimators=10, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, bootstrap=False, oob_score=False, n_jobs=1, random_state=None, verbose=0, warm_start=False, class_weight=None):
    print 'Test feature significance'
    if ixs is None:
        lix, vix = order_dist(y, vsplit=.7, tsplit=1.)

    sy = np.sign(y)
    learn_y, valid_y = sy[lix], sy[vix]

    scaler = MaxAbsScaler()

    best_features = set(x.columns.tolist())

    logf = join(cfg.dump_path, 'indcomp', 'fs.csv')
    with open(logf, 'wb') as f:
        f.write('feature,value\n')

    bm = 0.
    lp = 0
    while len(best_features) > n_f:
        columns = x.columns.tolist()
        data_x = x.as_matrix()
        scaler.fit(data_x[lix])
        data_x = scaler.transform(data_x)
        n = int(data_x.shape[1] / n_f) + 1
        for i in range(n):
            l, r = i * n_f, min(x.shape[1], (i + 1) * n_f)
            l = l if r - l >= n_f else r - n_f
            new_data_x = data_x[:, l:r]
            learn_x, valid_x = new_data_x[lix], new_data_x[vix]
            model, pred_y = cs.tree_class((learn_x, learn_y), (valid_x, valid_y), generations_number=5, population_size=50)

            importances = model.best_estimator_.feature_importances_
            std = np.std([tree.feature_importances_ for tree in model.best_estimator_.estimators_], axis=0)
            indices = np.argsort(importances)[::-1]
            ixs = columns[l:r]

            cbest_features = []
            writed = False
            wattmpt = 0
            while not writed and wattmpt < 10:
                try:
                    with open(logf, 'a') as fl:
                        for f in range(len(ixs)):
                            cbest_features.append(ixs[indices[f]])
                            fl.write("%s,%f\n" % (ixs[indices[f]], importances[indices[f]]))
                        fl.write('\n')
                        fl.write(classification_report(sy[vix], pred_y))
                        fl.write(str(np.sum(np.sign(pred_y) * y[vix])))
                        fl.write('\n%d / %d\n' % (lp, i))
                        fl.write('=' * 10)
                        fl.write('\n')
                    writed = True
                except Exception, err:
                    wattmpt+=1
                    print err.message

            writed = False
            wattmpt = 0
            while not writed and wattmpt < 10:
                try:
                    # Plot the feature importances of the forest
                    plt.figure()
                    plt.title("Feature importances")
                    bars = plt.bar(range(len(ixs)), importances[indices],
                            color="r", yerr=std[indices], align="center")
                    plt.xticks(range(len(ixs)), indices)
                    plt.xlim([-1, len(ixs)])
                    #plt.legend(bars, np.array(ixs)[indices])
                    savefig(join(cfg.dump_path, 'indcomp', 'fs_' + str(lp) + '_' + str(i) + '.png'))
                    writed = True
                except Exception, err:
                    wattmpt+=1
                    print err.message

            m = np.mean(importances)
            bm = bm if bm > m else m
            if len(cbest_features) >= 2:
                worst = np.array(cbest_features)
                worst = worst[np.argwhere(importances < bm).squeeze()]
                if len(worst) > 0:
                    best_features.difference_update(worst)
                else:
                    worst = cbest_features[max(0, int(len(cbest_features) / 2.) - 1):]
                    best_features.difference_update(worst)
            #for c in worst:
             #   del(x[c])

        x = x[list(best_features)]
        lp += 1

    data_x = x.as_matrix()
    learn_x, valid_x = data_x[lix], data_x[vix]
    model, pred_y = cs.tree_class((learn_x, learn_y), (valid_x, valid_y))

    print confusion_matrix(valid_y, pred_y)
    print classification_report(valid_y, pred_y)
    print np.sum(np.sign(pred_y) == np.sign(valid_y)) / float(len(valid_y))

    importances = model.best_estimator_.feature_importances_
    std = np.std([tree.feature_importances_ for tree in model.best_estimator_.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    ixs = x.columns.tolist()

    try:
        itocsv = np.array(['tmp', 0])
        for f in range(len(ixs)):
            itocsv = np.vstack((itocsv, [ixs[f], importances[indices[f]]]))
        itocsv = pd.DataFrame(itocsv[1:], columns=['feature', 'significance'])
        itocsv.to_csv(join(cfg.dump_path, 'indcomp', 'feature_significance.csv'))
    except Exception, err:
        print err.message

    for f in range(len(ixs)):
        print("%d. %s (%f)" % (f + 1, ixs[f], importances[indices[f]]))

    # Plot the feature importances of the forest
    plt.figure()
    plt.title("Feature importances")
    plt.bar(range(len(ixs)), importances[indices],
            color="r", yerr=std[indices], align="center")
    plt.xticks(range(len(ixs)), indices)
    plt.xlim([-1, len(ixs)])
    savefig(join(cfg.dump_path, 'indcomp', 'feature_significance.png'))
