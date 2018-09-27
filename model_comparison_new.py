# -*- coding: utf-8 -*-
#
# model_comparison.py
#
# The module is part of model_comparison.
#¨
#add to master

"""
The model comparison framework.
"""

__author__ = 'Hanna Svennevik', 'Paulina Tedesco'
__email__ = 'hanna.svennevik@fys.uio.no', 'paulinatedesco@gmail.com'


import algorithms
import numpy as np
from model_selection_new import GridSearchNew


def model_comparison_new(models, param_grid, X, z, split_size=0.2, verbose=True):
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

    results = {}
    for name, estimator in models.items():

        if verbose:
            print('Testing model: {}'.format(name))

        #avg_train_scores_mse, avg_test_scores_mse = [], []
        #avg_train_scores_r2, avg_test_scores_r2 = [], []


        # bias_models = []  # store the bias of each model
        # var_model = []  # store the variance of each model (this is the covariance)
        mse_all_models = []
        r2_all_models = []
        grid = GridSearchNew(estimator, param_grid[name], name)
        grid.fit(X, z, split_size=0.2)

        # store the scores for each model
        mse_all_models.append(grid.mse)
        r2_all_models.append(grid.r2)

        # Bias of best lambda calculated for each model ---> check maths!!
        best_bias_mse = (np.mean(z) - grid.best_avg_z_pred_mse)**2
        best_bias_r2 = (np.mean(z) - grid.best_avg_z_pred_mse) ** 2

        # # Variance of best lambda calculated for each model ---> check maths!!
        nrep = len(grid.best_z_pred_mse)
        mean_z_rep = np.squeeze((np.repeat(np.mean(z), nrep)))
        best_var_mse = (np.sum((mean_z_rep - grid.best_z_pred_mse)**2))/grid.nboots
        best_var_r2 = (np.sum((mean_z_rep - grid.best_z_pred_r2)**2))/grid.nboots

        # print('mean_z_rep', (mean_z_rep), 'grid.best_z_pred_r2', (grid.best_z_pred_r2), 'diff', (np.sum((mean_z_rep - grid.best_z_pred_r2)**2))/grid.nboots)

        # print('r2_boot', grid.test) # test dimensions




        # TODO: confidence intervals

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
            'Bias for best r2': best_bias_r2,
            'Variance for best MSE': best_var_mse,
            'Variance for best r2': best_var_r2

        }

    return results
