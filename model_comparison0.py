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
        (dict): Model scores.
    """

    mse = {
        'ridge': [],
        'lasso': [],
        "ols":[]
    }

    r2 = {
        'ridge': [],
        'lasso': [],
        "ols": []
    }
    for name, estimator in models.items():

        if verbose:
            print('Testing model: {}'.format(name))

        #avg_train_scores_mse, avg_test_scores_mse = [], []
        #avg_train_scores_r2, avg_test_scores_r2 = [], []

        # bias_models = []  # store the bias of each model
        # var_model = []  # store the variance of each model (this is the covariance)

        grid = GridSearchNew(estimator, param_grid[name], name)
        grid.fit(X, z, split_size=0.2)

        # store the scores for each model
        mse[name].append(grid.mse)
        r2[name].append(grid.r2)

        """
        if verbose:
            print('Best mse score: {}'.format(grid.best_mse),
                  'Best lambda: {}'.format(grid.best_param_mse))
            print('Best r2 score: {}'.format(grid.best_r2),
                  'Best lambda: {}'.format(grid.best_param_r2))

        results[name] = {
            'Best mse': grid.best_mse,
            'Best mse lambda': grid.best_param_mse,
            'Best r2': grid.best_r2,
            'Best r2 lambda': grid.best_param_r2,
            'Bias for best MSE': best_bias_mse,
            'Bias for best r2': best_bias_r2
        }"""

    results = {"mse": mse, "r2": r2}

    return results
