# -*- encode: utf-8 -*-

from os.path import join
import pickle

import configs as cfg
import dataworker as dw
import nets as nn

paths = []
for s in ['GC', 'SP', 'HG']:
    paths.append(join(cfg.data_path, s + '/' + s + '.csv'))
data_params = {
    'units':64,
    'punits':1,
    'period':'ED',
    'ysum':False,
    'order':False,
    'vsplit':.45,
    'tsplit':.65,
    'split_iter':5000,
    'waves':True,
    'lvl':2,
    'db':20,
    'stats':False,
    'dump_path' : join(cfg.data_path, 'reduced_ECL_')
}
dw.prepare(paths, **data_params)
data = pickle.load(open(dw.gen_path(**data_params), 'rb'))

lg = [
    {
        'layer':'dense',
        'output_dim':9,
    },
    {
        'layer':'dense',
        'output_dim':1,
        'bias':False
    }

]

params = {
    'batch_size'    : 3,
    'lr'            : 1.,
    'opt'           : 'adadelta',
    'loss'          : 'rmse',
    'nb_epoch'      : 300,
    'early'         : 10,
    'verbose'       : 2,
    'metrics'       : None,
    'x_tag'           : 'Waves', # Price, Return, CReturn, Waves, CWaves
    'y_tag'        : 'CReturn', # Price - Price, * - *
    'x_lvl'         : -1,
    'wpart'         : 'D', # A - D
    'stats'         : True,
    'plot'          : False,
    'examples'      : True,
    'dump_model'    : True,
    'log'           : False,
    'checkpoint'   : False
}

data_params = {
    'units':64,
    'punits':1,
    'period':'ED',
    'ysum':True,
    'waves':True,
    'stats':False,
    'order':False,
    'lvl':2,
    'db':20,
    'dump_path' : join(dw.data_path, 'reduced_CL_')
}
dw.prepare([join(cfg.data_path, 'CL/CL.csv')], **data_params)
test = pickle.load(open(dw.gen_path(**data_params), 'rb'))

model = nn.simple_net(data, lg, **params)
nn.evaluate(model, test, x_tag=params['x_tag'], y_tag=params['y_tag'], lvl=params['x_lvl'])