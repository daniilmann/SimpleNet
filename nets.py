# -*- encoding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt

from keras.models import Sequential
from keras.layers.recurrent import GRU
from keras.layers import Reshape, Highway, Dense, MaxoutDense, Input, Activation, GaussianDropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.noise import GaussianNoise
from keras.regularizers import l2, activity_l2
from keras.optimizers import RMSprop, Adam, Adamax, Adadelta
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from keras.models import model_from_json

from sklearn.metrics import confusion_matrix, classification_report

from os.path import join, exists
from os import mkdir, listdir
from shutil import copyfile
import json
import sys

import configs as cfg

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
                   consume_less             = layer.get('consume_less') if layer.get('consume_less') in ['cpu', 'mem', 'gpu'] else 'cpu',
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
    elif x_tag == 'CWaves':
        x = params.get('x_lvl') if params.get('x_lvl') is not None else -1
        y_tag = params['y_tag']
        learn_X = data['learn'][x_tag][x]
        valid_X = data['valid'][x_tag][x]
        test_X = data['test'][x_tag][x]
    elif x_tag == 'Waves':
        x = params.get('x_lvl') if params.get('x_lvl') is not None else -1
        p = params.get('wpart') if params.get('wpart') is not None else 'A'
        y_tag = params['y_tag']
        learn_X = data['learn'][x_tag][x][p]
        valid_X = data['valid'][x_tag][x][p]
        test_X = data['test'][x_tag][x][p]
    elif params.get('y_tag') is not None:
        y_tag = params['y_tag']
    else:
        raise Exception('y_tag was not assigned')

    if x_tag == 'Stats':
        learn_X = data['learn'][x_tag]
        valid_X = data['valid'][x_tag]
        test_X = data['test'][x_tag]
    elif x_tag not in ['CWaves', 'Waves']:
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
        if not exists('tmp'):
            mkdir('tmp')
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
        real_ret = (real[0] - data['valid']['CReturn']['x'][:, -1], real[1] - data['test']['CReturn']['x'][:, -1])
        pred_ret = (pred[0] - data['valid']['CReturn']['x'][:, -1], pred[1] - data['test']['CReturn']['x'][:, -1])
        # real_ret = (np.log(np.exp(real[0]) * prices[0][:, 0] / prices[0][:, -1]),
        #             np.log(np.exp(real[1]) * prices[1][:, 0] / prices[1][:, -1]))
        # pred_ret = (np.log(np.exp(pred[0]) * prices[0][:, 0] / prices[0][:, -1]),
        #             np.log(np.exp(pred[1]) * prices[1][:, 0] / prices[1][:, -1]))
    else:
        real_ret = real
        pred_ret = pred

    if params.get('stats'):
        actions     = (np.sign(pred_ret[0]), np.sign(pred_ret[1]))

        if params.get('examples'):
            print '\n== Examples ', '=' * 10, '\nValidation set'
            print np.round(np.vstack((real[1], pred[1])).transpose(), 3)[:5]
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
        mpath = join(cfg.data_path, 'models')
        if not exists(mpath):
            mkdir(mpath)
        count = len(listdir(mpath))
        mpath = join(mpath, str(count) + '_' + str(np.round(np.sum(actions[1] * real_ret[1]), 2)))
        mkdir(mpath)
        model.save_weights(join(mpath, 'weights.hf5'))
        with open(join(mpath, 'model.json'), 'wb') as f:
            f.write(model.to_json())
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
    elif params.get('x_tag') == 'Stats':
        obs_x = data.get(params.get('x_tag'))
        obs_y = data.get(params.get('y_tag'))['y']
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
        real_ret = real - data['CReturn']['x'][:, -1]
        pred_ret = pred - data['CReturn']['x'][:, -1]
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

def load(path):
    with open(join(path, 'model.json'), 'rb') as f:
        model = model_from_json(f.read())
    model.load_weights(join(path, 'weights.hf5'))
    return model


