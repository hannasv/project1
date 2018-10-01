# -*- coding: utf-8 -*-
#
# model_comparison.py
#
# The module is part of model_comparison.
#

"""
The model comparison framework.
"""

__author__ = 'Hanna Svennevik', 'Paulina Tedesco'
__email__ = 'hanna.svennevik@fys.uio.no', 'paulinatedesco@gmail.com'


import algorithms
import numpy as np
from model_selection0 import GridSearchNew

def model_comparison0(models, param_grid, X, z, split_size=0.2, verbose=True):
    """Perform the model comparison experiment.

    Args:
        models (dict):
        param_grid (dict):
        X (array-like):
        y (array-like):
        random_state (array-like)

    Returns:
        (dict): Model scores of r2 and mse.
    """

    mse = {
        'ridge': [],
        'lasso': [],
        "ols": []
    }

    r2 = {
        'ridge': [],
        'lasso': [],
        "ols": []
    }

    z_pred_best = {
        'ridge': [],
        'lasso': [],
        "ols": []
    }

    for name, estimator in models.items():

        if verbose:
            print('Testing model: {}'.format(name))

        grid = GridSearchNew(estimator, param_grid[name], name)
        grid.fit(X, z, split_size=0.2)

        # store the scores for each model
        mse[name].append(grid.mse)
        r2[name].append(grid.r2)

        # find best mse
        mn, idx = min((grid.mse[i], i) for i in range(len(grid.mse)))
        print(idx)
        z_pred_best[name] = grid.z_pred[idx]

    results = {"mse": mse, "r2": r2}

    return results, z_pred_best
