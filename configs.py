# -*- encoding: utf-8 -*-

from os.path import expanduser, join
from os import listdir

data_path = expanduser('~/investparty/data')

list_path = join(data_path, 'lists')
list_files = filter(lambda f: f[-4:] == '.csv', listdir(list_path))

ind_path = join(data_path, 'indices')
shares_path = join(data_path, 'shares')
curr_path = join(data_path, 'currencies')
comm_path = join(data_path, 'commodities')
fut_path = join(data_path, 'futures')
rates_path = join(data_path, 'rates')
any_path = join(data_path, 'any')

seasonality_path = join(data_path, 'seasonality')

predef_assets = {
    'EURUSD'    :   join(curr_path, 'EURUSD.csv'),
    'USDRUB'    :   join(curr_path, 'USDRUB.csv'),
    'XAUUSD'    :   join(comm_path, 'XAUUSD.csv'),
    'XAUEUR'    :   join(comm_path, 'XAUEUR.csv'),
    'XAGUSD'    :   join(comm_path, 'XAGUSD.csv'),
    'XAGEUR'    :   join(comm_path, 'XAGEUR.csv'),
    'SPX'       :   join(ind_path, 'SPX.csv'),
    'NDJI'      :   join(ind_path, 'NDJI')
}

def predefAsset(name):
    return predef_assets.get(name)