# -*- encoding: utf-8 -*-

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima_model import ARIMA

from keras.models import Sequential, Model
from keras.layers.recurrent import LSTM, GRU
from keras.layers import Reshape, Highway, Dense, Dropout, BatchNormalization, MaxoutDense, Input, merge, Activation, GaussianDropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.regularizers import l1l2, activity_l1l2, l2, activity_l2
from keras.optimizers import RMSprop, Adam, Adamax, SGD, Adadelta
from keras.layers.advanced_activations import PReLU, ParametricSoftplus
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils.visualize_util import plot
from keras import backend as K

from sklearn.metrics import confusion_matrix, classification_report, r2_score
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances
from scipy.signal import argrelextrema
from scipy.stats import kurtosis, skew, entropy
from scipy.special import erf
from scipy.spatial.distance import cosine

from os.path import join, expanduser, exists
from os import mkdir, listdir
from shutil import copyfile
from itertools import izip as zip
import pickle, json
from functools import partial
import sys

import dataworker as dw
import wavelet as wv

def model_ARIMA(data, xshape, batch_size=8, sign=True, nb_epoch=100, opt='adam', lr=.0001, loss='mae'):

    learn_X, learn_Y, valid_X, valid_Y, test_X, test_Y = _data_2D(data)

    learn = np.hstack((learn_X, learn_Y))
    valid = np.hstack((valid_X, valid_Y))
    test = np.hstack((test_X, test_Y))

    # learn = np.array((map(np.vstack, zip(
    #     learn[:, :-3],
    #     learn[:, 1:-2],
    #     learn[:, 2:-1],
    #     learn[:, 3:]
    # ))))
    # valid = np.array((map(np.vstack, zip(
    #     valid[:, :-3],
    #     valid[:, 1:-2],
    #     valid[:, 2:-1],
    #     valid[:, 3:]
    # ))))
    # test = np.array((map(np.vstack, zip(
    #     test[:, :-3],
    #     test[:, 1:-2],
    #     test[:, 2:-1],
    #     test[:, 3:]
    # ))))
    #
    # learn = learn[:,np.newaxis,:]
    # valid = valid[:,np.newaxis,:]
    # test = test[:,np.newaxis,:]
    #
    #
    # learn_X = learn[:, :, -(xshape+1):-1]
    # learn_Y = learn[:, :, -xshape:]
    # valid_X = valid[:, :, -(xshape+1):-1]
    # valid_Y = valid[:, :, -xshape:]
    # test_X = test[:, :, -(xshape+1):-1]
    # test_Y = test[:, :, -xshape:]
    #
    # learn_X = np.squeeze(learn_X)
    # valid_X = np.squeeze(valid_X)
    # test_X = np.squeeze(test_X)
    #
    # learn_Y = np.squeeze(learn_Y)
    # valid_Y = np.squeeze(valid_Y)
    # test_Y = np.squeeze(test_Y)

    input0 = Input(shape=(1, xshape), name='input_0')
    input1 = Input(shape=(xshape, ), name='input_1')
    input2 = Input(shape=(xshape, ), name='input_2')
    input3 = Input(shape=(xshape, ), name='input_3')

    gru1 = GRU(output_dim=xshape,
                  W_regularizer=l1l2(.05, .05),
                  U_regularizer=l1l2(.05, .05),
                  b_regularizer=l1l2(.05, .05),
                  unroll=False,
                  init='normal',
                  inner_activation='sigmoid',
                  activation='tanh',
               #return_sequences=True
                  )(input0)

    merge10 = merge([gru1, input1], mode='sum', name='merge10')
    merge11 = merge([gru1, merge10], mode='concat', concat_axis=1, name='merge11')
    out1    = Reshape((2, xshape))(merge11)

    gru2 = GRU(output_dim=xshape,
                  W_regularizer=l1l2(.05, .05),
                  U_regularizer=l1l2(.05, .05),
                  b_regularizer=l1l2(.05, .05),
                  unroll=False,
                  init='normal',
                  inner_activation='sigmoid',
                  activation='tanh',
               #return_sequences=True
                  )(out1)

    merge20 = merge([gru2, input2], mode='sum', name='merge20')
    merge21 = merge([gru2, merge20], mode='concat', concat_axis=1, name='merge21')
    out2    = Reshape((2, xshape))(merge21)

    gru3 = GRU(output_dim=xshape,
               W_regularizer=l1l2(.05, .05),
               U_regularizer=l1l2(.05, .05),
               b_regularizer=l1l2(.05, .05),
               unroll=False,
               init='normal',
               inner_activation='sigmoid',
               activation='tanh',
               #return_sequences=True
               )(out2)

    merge30 = merge([gru3, input3], mode='sum', name='merge30')
    merge31 = merge([gru3, merge30], mode='concat', concat_axis=1, name='merge31')
    out3    = Reshape((2, xshape))(merge31)

    output = GRU(output_dim=1,
               W_regularizer=l1l2(.05, .05),
               U_regularizer=l1l2(.05, .05),
               b_regularizer=l1l2(.05, .05),
               unroll=False,
               init='normal',
               inner_activation='sigmoid',
               activation='linear',
               #return_sequences=True
               )(out3)

    last = learn[:, 3:] - np.hstack((np.zeros(learn[:,3:-1].shape), learn[:, np.newaxis,-1]))
    model = Model(input=[input0, input1, input2, input3], output=output)
    if opt == 'adam':
        model.compile(optimizer=Adam(lr=lr), loss=loss)
    elif opt == 'adadelta':
        model.compile(optimizer=Adadelta(lr=lr), loss=loss)
    elif opt == 'adamax':
        model.compile(optimizer=Adamax(lr=lr), loss=loss)
    elif opt == 'rmsprop':
        model.compile(optimizer=RMSprop(lr=lr), loss=loss)
    model.fit([learn[:, np.newaxis, :-3], -learn[:, 1:-2], -learn[:, 2:-1], -last], learn_Y, nb_epoch=nb_epoch,
              callbacks=[EarlyStopping(patience=20)]
              )
    pred = model.predict([valid[:, np.newaxis, :-3], -valid[:, 1:-2], -valid[:, 2:-1], -last])

    print pred[:10]
    print pred.shape

    y = np.squeeze(valid_Y)
    p = np.squeeze(pred)
    print 'mse ', np.mean((y - p) ** 2)
    print 'rmse ', np.sqrt(np.mean((y - p) ** 2))
    print 'mae ', np.mean(abs(y - p))

    plt.plot(y, p, 'ro')
    plt.show()

    # model.add(BatchNormalization(input_shape=(learn_X.shape[1],)))
    # model.add(Dense(learn_X.shape[1]))
    # for _ in range(19):
    #     model.add(Highway(W_regularizer=l1l2(.005, .005),
    #                       activity_regularizer=activity_l1l2(.005, .005),
    #                       b_regularizer=l1l2(.005, .005),
    #                       init='normal',
    #                       activation='relu',
    #                       transform_bias=-3
    #                       ))
    # model.add(GRU(output_dim=learn_X.shape[2],
    #               #input_shape=(learn_X.shape[1], learn_X.shape[2]),
    #               W_regularizer=l1l2(.05, .05),
    #               U_regularizer=l1l2(.05, .05),
    #               b_regularizer=l1l2(.05, .05),
    #               unroll=False,
    #               init='normal',
    #               inner_activation='hard_sigmoid',
    #               activation='linear',
    #               return_sequences=True
    #               ))
    # model.add(GRU(output_dim=learn_X.shape[2],
    #               #input_shape=(learn_X.shape[1], learn_X.shape[2]),
    #               W_regularizer=l1l2(.05, .05),
    #               U_regularizer=l1l2(.05, .05),
    #               b_regularizer=l1l2(.05, .05),
    #               unroll=False,
    #               init='normal',
    #               inner_activation='hard_sigmoid',
    #               activation='linear',
    #               return_sequences=True
    #               ))
    # model.add(GRU(output_dim=learn_X.shape[2],
    #               #input_shape=(learn_X.shape[1], learn_X.shape[2]),
    #               W_regularizer=l1l2(.05, .05),
    #               U_regularizer=l1l2(.05, .05),
    #               b_regularizer=l1l2(.05, .05),
    #               unroll=False,
    #               init='normal',
    #               inner_activation='hard_sigmoid',
    #               activation='linear',
    #               #return_sequences=True
    #               ))

    # if opt == 'adam':
    #     model.compile(optimizer=Adam(lr=lr), loss=loss)
    # elif opt == 'adadelta':
    #     model.compile(optimizer=Adadelta(lr=lr), loss=loss)
    # elif opt == 'adamax':
    #     model.compile(optimizer=Adamax(lr=lr), loss=loss)
    # elif opt == 'rmsprop':
    #     model.compile(optimizer=RMSprop(lr=lr), loss=loss)
    # model.fit(learn_X, learn_Y, nb_epoch=nb_epoch, batch_size=batch_size, verbose=2, validation_data=(valid_X, valid_Y),
    #           callbacks=[EarlyStopping(patience=20)]
    #           )
    #
    # prediction = model.predict(test_X, batch_size=batch_size)
    #
    # d = cosine_distances(test_Y, prediction)
    # s = cosine_similarity(test_Y, prediction)
    # print d
    # print s
    # print test_Y[:5]
    # print prediction[:5]
    # print test_Y[-5:]
    # print prediction[-5:]

def levels(x):
    n = 1
    for symbol in ['CL', 'GC', 'SP', 'ZW', 'NG', 'HG', 'SB']:
        data = _data(symbol)
        data = _to_week(data)
        data = data.sort_index()
        cdata = np.cumsum(data)
        close = cdata.Close
        gc = close.ix[argrelextrema(close.as_matrix(), np.greater)]
        lc = close.ix[argrelextrema(close.as_matrix(), np.less)]
        # i = 1
        # while i < 6:
        #     x = np.array([[v] for v in np.hstack((gc, lc))])
        #     if x.shape[0] <= 200:
        #         break
        #     gc = gc.ix[argrelextrema(gc.as_matrix(), np.greater)]
        #     lc = lc.ix[argrelextrema(lc.as_matrix(), np.less)]
        #     i = i + 1
        # obs = [np.array([[v] for v in close])]
        obs = []
        score = []
        # afp = AffinityPropagation()
        # lbls = afp.fit_predict(obs[0])
        # score.append(silhouette_score(obs[0], lbls))
        # print symbol, 0, len(afp.cluster_centers_), score[-1]
        gc = close.ix[argrelextrema(close.as_matrix(), np.greater)]
        lc = close.ix[argrelextrema(close.as_matrix(), np.less)]
        i = 1
        while i < 5:
            x = np.array([[v] for v in np.hstack((gc, lc))])
            if len(x) <= 2:
                break;
            obs.append(x)
            gc = gc.ix[argrelextrema(gc.as_matrix(), np.greater)]
            lc = lc.ix[argrelextrema(lc.as_matrix(), np.less)]
            afp = AffinityPropagation()
            lbls = afp.fit_predict(x)
            score.append(silhouette_score(x, lbls))
            print symbol, i, len(x), len(afp.cluster_centers_), score[-1]
            if len(afp.cluster_centers_) >= 15:
                obs.pop()
                score.pop()
            i = i + 1
        print '=' * 30
        afp = AffinityPropagation()
        cobs = obs[np.argmin(score)]
        afp.fit(cobs)
        plt.figure(n)
        plt.title(symbol)
        plt.plot(cdata.Close, 'r')
        mx = max(cobs)
        cntrs = afp.cluster_centers_.tolist()
        if abs(mx - max(cntrs)) >= .06:
            cntrs.append(mx)
        mn = min(cobs)
        if abs(mx - min(cntrs)) >= .06:
            cntrs.append(mn)
        map(plt.axhline, cntrs)
        # plt.draw()
        plt.savefig(symbol + 'AFP2.png')
        n = n + 1
        # sc = []
        # for c in range(5, 11, 1):
        #     km = KMeans(n_clusters=c)
        #     lbls = km.fit_predict(x)
        #     sc.append(silhouette_score(x, lbls))
        #
        # c = np.argmin(sc) + 5
        # print symbol, i, c, min(sc)
        # km = KMeans(n_clusters=c)
        # km.fit(x)
        # plt.figure(n)
        # plt.title(symbol)
        # plt.plot(cdata.Close, 'r')
        # map(plt.axhline, km.cluster_centers_)
        # plt.savefig(symbol + '.png')
        # n = n + 1

        # plt.show()

def _lcreator(layer):
    insh = {'input_shape' : layer.get('input_shape')} if layer.get('input_shape') is not None else {}

    if layer['layer'] == 'dense':
        return Dense(output_dim             = int(np.round(layer['output_dim'])),
                     W_regularizer          = l2(layer.get('w_reg')) if layer.get('w_reg') is not None else None,
                     activity_regularizer   = activity_l2(layer.get('a_reg')) if layer.get('a_reg') is not None else None,
                     b_regularizer          = l2(layer.get('b_reg')) if layer.get('b_reg') is not None else None,
                     init                   = layer.get('init') if layer.get('init') is not None else 'lecun_uniform',
                     bias                   = layer.get('bias') is True,
                     **insh
                     )
    elif layer['layer'] == 'maxout':
        return MaxoutDense(output_dim           = int(np.round(layer['output_dim'])),
                           nb_feature           = int(np.round(layer.get('nb_feature'))) if isinstance(layer.get('nb_feature'), int) else 4,
                           W_regularizer        = l2(layer.get('w_reg')) if layer.get('w_reg') is not None else None,
                           activity_regularizer = activity_l2(layer.get('a_reg')) if layer.get('b_reg') is not None else None,
                           b_regularizer        = l2(layer.get('b_reg')) if layer.get('b_reg') is not None else None,
                           init                 = layer.get('init') if layer.get('init') is not None else 'lecun_uniform',
                           bias                 = layer.get('bias') is True,
                           **insh
                           )
    elif layer['layer'] == 'gru':
        return GRU(output_dim               = int(np.round(layer['output_dim'])),
                   activation               = layer.get('activation') if layer.get('activation') is not None else 'tanh',
                   inner_activation         = layer.get('inn_activation') if layer.get('inn_activation') is not None else 'hard_sigmoid',
                   W_regularizer            = l2(layer.get('w_reg')) if layer.get('w_reg') is not None else None,
                   U_regularizer            = l2(layer.get('u_reg')) if layer.get('u_reg') is not None else None,
                   b_regularizer            = l2(layer.get('b_reg')) if layer.get('b_reg') is not None else None,
                   init                     = layer.get('init') if layer.get('init') is not None else 'lecun_uniform',
                   inner_init               = layer.get('inner_init') if layer.get('inner_init') is not None else 'orthogonal',
                   dropout_W                = layer.get('w_drop') if layer.get('w_drop') is not None else 0.,
                   dropout_U                = layer.get('u_drop') if layer.get('u_drop') is not None else 0.,
                   return_sequences         = layer.get('return_seq') is True,
                   unroll                   = layer.get('unroll') is True,
                   **insh
                    )
    elif layer['layer'] == 'highway':
        return Highway(W_regularizer            = l2(layer.get('w_reg')) if layer.get('w_reg') is not None else None,
                       activity_regularizer     = activity_l2(layer.get('a_reg')) if layer.get('a_reg') is not None else None,
                       b_regularizer            = l2(layer.get('b_reg')) if layer.get('b_reg') is not None else None,
                       init                     = layer.get('init') if layer.get('init') is not None else 'lecun_uniform',
                       transform_bias           = layer.get('trans_b') if isinstance(layer.get('trans_b'), float) else -3.,
                       **insh
                       )
    elif layer['layer'] == 'norm':
        return BatchNormalization(mode = layer.get('mode') if layer.get('mode') in [0, 1, 2] else 2,
                                  **insh
                                  )
    elif layer['layer'] == 'noise':
        return GaussianNoise(layer['noise'],
                             **insh
                             )
    elif layer['layer'] == 'reshape':
        return Reshape(target_shape=layer.get('tshape'))
    elif layer['layer'] == 'drop':
        return GaussianDropout(layer['drop'])
    elif layer['layer'] == 'activation':
        return Activation(layer['activation'])

def simple_net(data, layers, **params):

    if params.get('log'):
        stdout = sys.stdout
        stdout.flush()
        sys.stdout = open('tmp/net.log', 'w')


    x_tag = params['x_tag']
    x = None
    if x_tag == 'Price':
        y_tag = 'Price'
    elif x_tag in ['CWaves', 'Waves']:
        x = params.get('x_lvl') if params.get('x_lvl') is not None else -1
        y_tag = params['y_tag']
        learn_X = data['learn'][x_tag][x]
        valid_X = data['valid'][x_tag][x]
        test_X = data['test'][x_tag][x]
    elif params.get('y_tag') is not None:
        y_tag = params['y_tag']
    else:
        raise Exception('y_tag was not assigned')

    if x_tag not in ['CWaves', 'Waves']:
        learn_X = data['learn'][x_tag]['x']
        valid_X = data['valid'][x_tag]['x']
        test_X = data['test'][x_tag]['x']

    learn_Y = data['learn'][y_tag]['y']
    valid_Y = data['valid'][y_tag]['y']
    test_Y = data['test'][y_tag]['y']

    if params.get('xlen'):
        learn_X = learn_X[:, -params.get('xlen'):]
        valid_X = valid_X[:, -params.get('xlen'):]
        test_X = test_X[:, -params.get('xlen'):]

    if layers[0].get('layer') == 'gru': #np.array([l.get('layer') == 'gru' for l in layers]).any():
        if len(learn_X.shape) != 3:
            learn_X = learn_X[:, np.newaxis, :]
        if len(valid_X.shape) != 3:
            valid_X = valid_X[:, np.newaxis, :]
        if len(test_X.shape) != 3:
            test_X = test_X[:, np.newaxis, :]

    layers[0]['input_shape'] = (learn_X.shape[1:])

    if params.get('pretrain'):
        def sampling(args):
            z_mean, z_log_var = args
            epsilon = K.random_normal(shape=K.shape(z_mean), mean=0.)
            return z_mean + K.exp(z_log_var / 2) * epsilon

        tmpl = []
        for layer in layers:
            l = {}
            if 'input_shape' in layer.keys():
                l['input_shape'] = layer.get('input_shape')
            elif 'output_dim' in layer.keys():
                l['input_shape'] = tmpl[-1].get('output_dim')
            else:
                continue
            l['output_dim'] = layer.get('output_dim')
            l['return_seq'] = layer.get('return_seq')
            l['layer']      = layer.get('layer')
            tmpl.append(l)
        for layer in tmpl:
            if 'output_dim' in layer.keys():
                vae = Sequential()
                if layer.get('layer') in ['gru', 'conv1d']:
                    ishp = (1, layer.get('input_shape'),)
                    oshp = (1, layer.get('output_dim'),)
                else:
                    ishp = (layer.get('input_shape'),)
                    oshp = (layer.get('output_dim'),)
                inp = Input(shape=ishp)
                vae.add(inp)
                layer['input_shape'] = None
                z_mean = _lcreator(layer)(inp)
                z_log_var = _lcreator(layer)(inp)
                z = Lambda(sampling, output_shape=oshp)([z_mean, z_log_var])
                vae.add(z)
                outp = _lcreator({'layer' : layer.get('layer'), 'output_dim' : ishp[-1]})
                vae.add(outp)

                vae.compile(optimizer=Adadelta(lr=1.), loss='cosine_proximity')
                vae.fit(learn_X, learn_X,  nb_epoch=params['nb_epoch'], batch_size=params['batch_size'], verbose=params['verbose'],
                      validation_data=(valid_X, valid_X),
                      shuffle=True,
                      callbacks=[EarlyStopping(patience=params.get('early') if params.get('early') is not None else params['nb_epoch'])])
                layer['weights'] = z.get_weights()

    model = Sequential()

    for layer in layers:
        model.add(_lcreator(layer))

    opt = {
        'adam': Adam(lr=params['lr']),
        'adadelta': Adadelta(lr=params['lr']),
        'adamax': Adamax(lr=params['lr']),
        'rmsprop': RMSprop(lr=params['lr'])
    }[params['opt']]

    def rmse(y_real, y_pred):
        return K.sqrt(K.mean(K.square(y_real - y_pred)))

    if params.get('loss') == 'rmse':
        loss = rmse
    else:
        loss = params.get('loss')

    model.compile(optimizer=opt, loss=loss)

    callbacks = []
    if params.get('early'):
        early = params['early'] if params['early'] is not None else params['nb_epoch']
        callbacks.append(EarlyStopping(patience=early))
    if params.get('checkpoint'):
        callbacks.append(ModelCheckpoint('tmp/model.hdf5',
                                         monitor='val_loss',
                                         verbose=1 if params['verbose'] > 0 else 0,
                                         save_best_only=True,
                                         mode='auto'))
    model.fit(learn_X, learn_Y, nb_epoch=params['nb_epoch'], batch_size=params['batch_size'], verbose=params['verbose'],
              validation_data=(valid_X, valid_Y),
              shuffle=True,
              callbacks=callbacks
              )

    if params.get('checkpoint'):
        model.load_weights('tmp/model.hdf5')

    pred = (model.predict(valid_X, batch_size=params['batch_size']).squeeze(), model.predict(test_X, batch_size=params['batch_size']).squeeze())
    real = (data['valid'][y_tag]['y'].squeeze(), data['test'][y_tag]['y'].squeeze())

    if len(learn_X.shape) == 3:
        prices = (data['valid']['Price']['x'].squeeze(), data['test']['Price']['x'].squeeze())
    else:
        prices = (data['valid']['Price']['x'], data['test']['Price']['x'])

    if y_tag == 'Price':
        real_ret = (np.log(real[0] / prices[0][:, -1]), np.log(real[1] / prices[1][:, -1]))
        pred_ret = (np.log(pred[0] / prices[0][:, -1]), np.log(pred[1] / prices[1][:, -1]))
    elif y_tag == 'CReturn':
        real_ret = (np.log(np.exp(real[0]) * prices[0][:, 0] / prices[0][:, -1]),
                    np.log(np.exp(real[1]) * prices[1][:, 0] / prices[1][:, -1]))
        pred_ret = (np.log(np.exp(pred[0]) * prices[0][:, 0] / prices[0][:, -1]),
                    np.log(np.exp(pred[1]) * prices[1][:, 0] / prices[1][:, -1]))
    else:
        real_ret = real
        pred_ret = pred

    if params.get('stats'):
        actions     = (np.sign(pred_ret[0]), np.sign(pred_ret[1]))

        if params.get('examples'):
            print '\n== Examples ', '=' * 10, '\nValidation set'
            print np.round(np.vstack((pred[1], real[1])).transpose(), 2)[:5]
            print '\n Test set'
            print np.round(np.vstack((real_ret[1], pred_ret[1])).transpose(), 3)[:5]
        
        stats = {
            'test' : {
                'mse': np.mean(np.square(pred_ret[1] - real_ret[1])),
                'rmse': np.sqrt(np.mean(np.square(pred_ret[1] - real_ret[1]))),
                'mae': np.mean(np.absolute(pred_ret[1] - real_ret[1])),
                'std': np.std(pred_ret[1] - real_ret[1]),
                'target': np.sum(np.absolute(real_ret[1])),
                'return': np.sum(actions[1] * real_ret[1]),
                'optgt': np.round(np.sum(np.absolute(real_ret[1])) / test_Y.shape[0], 3),
                'opret': np.round(np.sum(actions[1] * real_ret[1]) / test_Y.shape[0], 3),
                'acc': np.round(np.sum(np.sign(actions[1]) == np.sign(real_ret[1])) * 100. / len(actions[1]))
            },
            'valid': {
                'mse': np.mean(np.square(pred_ret[0] - real_ret[0])),
                'rmse': np.sqrt(np.mean(np.square(pred_ret[0] - real_ret[0]))),
                'mae': np.mean(np.absolute(pred_ret[0] - real_ret[0])),
                'std': np.std(pred_ret[0] - real_ret[0]),
                'target': np.sum(np.absolute(real_ret[0])),
                'return': np.sum(actions[0] * real_ret[0]),
                'optgt': np.round(np.sum(np.absolute(real_ret[0])) / valid_Y.shape[0], 3),
                'opret': np.round(np.sum(actions[0] * real_ret[0]) / valid_Y.shape[0], 3),
                'acc': np.round(np.sum(np.sign(actions[0]) == np.sign(real_ret[0])) * 100. / len(actions[0]))
            }
        }

        print '\n== Shapes ', '=' * 10
        print 'Learn ', learn_X.shape[0]
        print 'Valid ', valid_X.shape[0]
        print 'Test  ', test_X.shape[0]
        print '\nModel ', learn_X.shape[-1], ' -> ', learn_Y.shape[-1]
        print '== Prices ', '=' * 10
        print '      Valid  ||  Test'
        print 'mse   %3.3f  ||  %3.3f' % (stats.get('valid').get('mse'), stats.get('test').get('mse'))
        print 'rmse  %3.3f  ||  %3.3f' % (stats.get('valid').get('rmse'), stats.get('test').get('rmse'))
        print 'mae   %3.3f  ||  %3.3f' % (stats.get('valid').get('mae'), stats.get('test').get('mae'))
        print 'std   %3.3f  ||  %3.3f' % (stats.get('valid').get('std'), stats.get('test').get('std'))
        print 'acc   %3.3f  ||  %3.3f' % (stats.get('valid').get('acc'), stats.get('test').get('acc'))
        print '\n== Return ', '=' * 10
        print '      Valid     ||  Test'
        print 'targ  %3.3f (%3.3f)  ||  %3.3f (%3.3f)' % (stats.get('valid').get('target'), stats.get('valid').get('optgt'), stats.get('test').get('target'), stats.get('test').get('optgt'))
        print 'ret   %3.3f (%3.3f)  ||  %3.3f (%3.3f)' % (stats.get('valid').get('return'), stats.get('valid').get('opret'), stats.get('test').get('return'), stats.get('test').get('opret'))
        print '\n== Classification ', '=' * 10, '\nValidation set:'
        print confusion_matrix(np.sign(real_ret[0]), actions[0])
        print classification_report(np.sign(real_ret[0]), actions[0])
        print '\nTest set:'
        print confusion_matrix(np.sign(real_ret[1]), actions[1])
        print classification_report(np.sign(real_ret[1]), actions[1])

    if params.get('log'):
        sys.stdout.flush()
        sys.stdout = stdout

    if params.get('plot'):
        plt.plot(pred[0], real[0], 'bh')
        plt.plot(pred[1], real[1], 'ro')
        plt.show()

    if params.get('dump_model'):
        mpath = join(dw.data_path, 'models')
        count = len(listdir(mpath))
        mpath = join(mpath, str(count) + '_' + str(np.round(np.sum(actions[1] * real_ret[1]), 2)))
        mkdir(mpath)
        model.save_weights(join(mpath, 'weights.hf5'))
        with open(join(mpath, 'model.json'), 'wb') as f:
            f.write(model.to_json())
        plot(model, to_file=join(mpath, 'model.png'), show_shapes=True)
        json.dump(params, open(join(mpath, 'params.json'), 'wb'), indent=3)
        json.dump(layers, open(join(mpath, 'layers.json'), 'wb'), indent=3)
        if(params.get('stats')):
            json.dump(stats, open(join(mpath, 'stats.json'), 'wb'), indent=3)
        if params.get('log'):
            copyfile('tmp/net.log', join(mpath, 'net.log'))


    return model

def evaluate(model, data, **params):

    key = data.keys()[-1] if params.get('key') is None else params.get('key')
    data = data[key]
    obs_y = None
    if params.get('x_tag') in ['Waves', 'CWaves']:
        if isinstance(params.get('lvl'), int):
            obs_x = data.get(params.get('x_tag'))[params.get('lvl')]
        else:
            raise Exception('Expected int lvl, received ' + str(type(params.get('lvl'))))
    elif params.get('x_tag') == 'Price':
        obs_x = data.get(params.get('x_tag'))['x']
        obs_y = data.get('Price')['y']
    else:
        obs_x = data.get(params.get('x_tag'))['x']
        obs_y = data.get(params.get('y_tag'))['y']

    if obs_y == None:
        obs_y = data.get(params.get('y_tag'))['y']

    pred = predict(model, obs_x)

    if len(obs_y.shape) == 2:
        real = obs_y.squeeze()
    else:
        real = obs_y

    if params.get('y_tag') == 'Price':
        real_ret = np.log(real / data.get('Price').get('x')[:, -1])
        pred_ret = np.log(pred / data.get('Price').get('x')[:, -1])
        pred_price = pred
    elif params.get('y_tag') == 'CReturn':
        real_ret = np.log(np.exp(real) * data.get('Price').get('x')[:, 0] / data.get('Price').get('x')[:, -1])
        pred_ret = np.log(np.exp(pred) * data.get('Price').get('x')[:, 0] / data.get('Price').get('x')[:, -1])
        pred_price = np.exp(pred) * data.get('Price').get('x')[:, 0]
    else:
        real_ret = real
        pred_ret = pred
        pred_price = np.exp(pred) * data.get('Price').get('x')[:, -1]

    actions = np.sign(pred_ret)

    logstr = ''
    if params.get('examples'):
        logstr += '\n== Results ' + '=' * 10
        logstr += '\n' + str(np.round(np.vstack((pred[1], real[1])).transpose(), 2))
        logstr += '\n\n'

    stats = {
        'mse': np.mean(np.square(pred_ret - real_ret)),
        'rmse': np.sqrt(np.mean(np.square(pred_ret - real_ret))),
        'mae': np.mean(np.absolute(pred_ret - real_ret)),
        'std': np.std(pred_ret - real_ret),
        'target': np.sum(np.absolute(real_ret)),
        'return': np.sum(actions * real_ret),
        'optgt': np.round(np.sum(np.absolute(real_ret)) / len(pred), 3),
        'opret': np.round(np.sum(actions * real_ret) / len(pred), 3)
    }

    logstr += '== Shape ' + '=' * 10
    logstr += '\nData \n' + str(obs_x.shape)
    logstr += '\n\nModel \n' + str(obs_x.shape[-1]) + ' -> ' + str(obs_y.shape[-1]) if len(obs_y.shape) == 2 else str(1)
    logstr += '\n== Prices ' + '=' * 10
    logstr += '\n      Score'
    logstr += '\nmse   %3.3f' % stats.get('mse')
    logstr += '\nrmse  %3.3f' % stats.get('rmse')
    logstr += '\nmae   %3.3f' % stats.get('mae')
    logstr += '\nstd   %3.3f' % stats.get('std')
    logstr += '\n\n== Return ' + '=' * 10
    logstr += '\n      Score'
    logstr += '\ntarg  %3.3f (%3.3f)' % (stats.get('target'), stats.get('optgt'))
    logstr += '\nret   %3.3f (%3.3f)' % (stats.get('return'), stats.get('opret'))
    logstr += '\n\n== Classification ' + '=' * 10
    logstr += '\n' + str(confusion_matrix(np.sign(real_ret), actions))
    logstr += '\n' + str(classification_report(np.sign(real_ret), actions))

    if not params.get('log'):
        print logstr

    if params.get('plot') == 'accuracy':
        plt.plot(pred, real, 'bh')
        plt.show()
    elif params.get('plot') == 'recon':
        plt.plot(data.get('Price').get('y')[:, 0], 'r')
        plt.plot(pred_price, 'b')
        plt.show()

    ret_key = params.get('return') if params.get('return') is not None else 'rmse'
    
    if params.get('log'):
        return (logstr, stats.get(ret_key))
    else:
        return stats.get(ret_key)


def predict(model, obs, keepdims=False):

    if len(model.layers[0].input_shape) == 3 and len(obs.shape) == 2:
        obs = obs[:, np.newaxis, :]
        if model.layers[0].input_shape[-1] < obs.shape[-1]:
            obs = obs[:, :, -model.layers[0].input_shape[-1]:]
    elif model.layers[0].input_shape[-1] < obs.shape[-1]:
            obs = obs[:, -model.layers[0].input_shape[-1]:]


    pred = model.predict(obs)

    if keepdims:
        return pred
    else:
        return pred.squeeze()



