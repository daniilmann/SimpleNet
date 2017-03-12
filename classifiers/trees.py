# -*- coding: utf-8 -*-

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from evolutionary_search import EvolutionaryAlgorithmSearchCV as EASCV

def tree_class(learn, valid, opt=True, **params):
    learn_x, learn_y = learn
    valid_x, valid_y = valid

    if opt:
        uparams = set(params.keys()).difference(
            [
                'scoring',
                'cv',
                'population_size',
                'gene_mutation_prob',
                'gene_crossover_prob',
                'tournament_size',
                'generations_number',
                'n_jobs',
                'estimator',
                'params_grig'
            ]
        )
        if uparams:
            raise Exception('Unrecognized parameters: ' + ', '.join(uparams))

        params_grig = {
            'n_estimators': np.arange(300, 1003, 100),
            'max_depth': np.arange(3, 31),
            'min_samples_split': [10, 30, 60, 90, 120, 160],
            'min_samples_leaf': [10, 30, 60, 90, 120, 160],
            'min_weight_fraction_leaf': [.1, .3, .5],
            'max_features': np.arange(2, int(np.ceil(learn_x.shape[1] / 2.))),
            'max_leaf_nodes': np.arange(300, 1001, 100)
        }

        est = ExtraTreesClassifier(criterion='gini', class_weight=None, bootstrap=True, warm_start=False)
        model = EASCV(estimator=est,
                      params=params_grig,
                      scoring=params.get('scoring') if isinstance(params.get('scoring'), str) else 'f1_weighted',
                      cv=params.get('cv') if isinstance(params.get('cv'), int) else 3,
                      verbose=params.get('verbose') if isinstance(params.get('verbose'), int) else 10,
                      population_size=params.get('population_size') if isinstance(params.get('population_size'), int) else 50,
                      gene_mutation_prob=params.get('gene_mutation_prob') if isinstance(params.get('gene_mutation_prob'), float) else .1,
                      gene_crossover_prob=params.get('gene_crossover_prob') if isinstance(params.get('gene_crossover_prob'), float) else .5,
                      tournament_size=params.get('tournament_size') if isinstance(params.get('tournament_size'), int) else 5,
                      generations_number=params.get('generations_number') if isinstance(params.get('generations_number'), int) else 1,
                      n_jobs=params.get('n_jobs') if isinstance(params.get('n_jobs'), int) else 4)
        # model = RSCV(est, params, 200, scoring='f1_weighted', n_jobs=4, verbose=10)
    # else:
    #     model = ExtraTreesClassifier(n_estimators=n_estimators,
    #                                  criterion=criterion,
    #                                  max_depth=max_depth,
    #                                  min_samples_split=min_samples_split,
    #                                  min_samples_leaf=min_samples_leaf,
    #                                  min_weight_fraction_leaf=min_weight_fraction_leaf,
    #                                  max_features=max_features,
    #                                  max_leaf_nodes=max_leaf_nodes,
    #                                  bootstrap=bootstrap,
    #                                  oob_score=oob_score,
    #                                  n_jobs=n_jobs,
    #                                  random_state=random_state,
    #                                  verbose=verbose,
    #                                  warm_start=warm_start,
    #                                  class_weight=class_weight)

    model.fit(learn_x, learn_y)
    pred_y = model.predict(valid_x)

    return model, pred_y