# -*- coding: utf-8 -*-
#
# model_comparison.py
#
# The module is part of model_comparison.
#

"""
The model comparison framework.
"""

__author__ = 'author 1', 'author 2'
__email__ = 'email1', 'email2'


import algorithms
import numpy as np
from model_selection import GridSearch


def model_comparison(models, param_grid, X, y, random_states):
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

    train_scores, test_scores = {}, {}
    for name, estimator in models.items():

        train_scores[name], test_scores[name] = [], []
        for random_state in random_states:

            # Generate data (bootstrap sampling in your case).
            X_train, X_test = train_test_split(X, random_state=random_state)
            y_train, y_test = train_test_split(y, random_state=random_state)

            # Pass algorithm + corresponding params to grid searcher and
            # determine optimal alpha param.
            grid = GridSearch(estimator, param_grid[name], random_state)
            grid.fit(X, y)

            # Store svg score values for each model.
            train_scores[name].append(np.mean(grid.train_scores))
            test_scores[name].append(np.mean(grid.test_scores))

    return train_scores, test_scores

if __name__ == '__main__':
    # Demo run

    # A collection of algorithm name : algorithm.
    models = {
        'ridge': algorithms.Ridge,
        'lasso': algorithms.Lasso
    }
    param_grid = {
        # Ridge alpha params.
        'ridge': [0.01, 0.1, 1.0, 10.0],
        # Lasso alpha params.
        'lasso': [0.01, 0.1, 1.0, 5.0]
    }
    random_states = np.arange(40)
    # Perform experiment and collect results.
    model_results = model_comparison(
        models, param_grid, X, y, random_states
    )
    # NOTE:
    # This procedure you can extent to return std of score and create a validation curve:
    # such as in https://chrisalbon.com/machine_learning/model_evaluation/plot_the_validation_curve/
