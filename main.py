# -*- encode: utf-8 -*-

# frame_params = {
#     'ix_columns'    : ['Date'],
#     'ix_format'     : '%d%m%y',
#     'preformat'     : {'Date': '{:06d}'},
#     'pre_delete'    : ['Time', 'Volume'],
#     'logret'        : {'Close': [1]},
#     'post_delete'   : ['Open', 'High', 'Low']
# }
#
# obs_params = {
#     'units'     : 64,
#     'punits'    : 1,
#     'period'    : 'ED',
#     'resample'  : 'SAT',
#     'weekends'  : [5]
# }
#
# params = {
#
# }

from keras.optimizers import Adadelta
from keras.regularizers import l2
from keras.layers.recurrent import GRU
from keras.models import Sequential
from keras.callbacks import EarlyStopping

import wavelet as wv
from dataworker import dataworker as dw;from os.path import expanduser, join;import numpy as np

layers = [
    {
        'layer':'gru',
        'output_dim':1,
        'activation':'linear',
        'inn_activation':'linear',
        'b_reg':10.
    }
]

path = expanduser('~/Downloads')
frame = dw.load_frame(join(path, 'EURUSD.csv'), 'DT', '%Y%m%d %H%M%S', columns=['DT', 'Close','Volume'], pre_delete=['Volume'], sep=';')
frame['Ret'] = np.log(frame.Close / frame.Close.shift())

data = frame.tail(10000)
obs = dw.gen_obs(data, 32, 5, 'ED')
xs, ys = (obs[0]['Ret'] * 100).cumsum(1), (obs[1]['Ret'] * 100).sum(1)
sxs = wv.smooth(xs, 3, 5)[:, np.newaxis, :]
xs = xs[:, np.newaxis, :]
ys = ys[:, np.newaxis]
learn, test = (xs[:-1000], ys[:-1000], sxs[:-1000]), (xs[-1000:], ys[-1000:], sxs[-1000:])

model = Sequential()
model.add(GRU(output_dim=1, input_shape=(1, 32), activation='linear', inner_activation='linear', b_regularizer=l2(10.)))
model.compile(Adadelta(lr=1.), loss='mse')
model.fit(learn[2], learn[1], batch_size=8, nb_epoch=10, validation_split=.3, verbose=2, callbacks=[EarlyStopping(patience=10)])

pys = model.predict(test[2]).squeeze()
print np.sum(np.sign(pys) * test[1].squeeze())
print np.sum(np.sign(pys) == np.sign(test[1].squeeze())) / float(test[1].shape[0])