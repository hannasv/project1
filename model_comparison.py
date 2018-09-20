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
from model_selection import GridSearch

# Google gridsearch sklearn
    # finne lmd verdien som er best for hvert dataset.


def bootstrap(x):
        bootVec = np.random.choice(x, len(x))
        return bootVec, x  # returns x_train, x_test


def generateDesignmatrix(p, x, y):
    m = int((p**2+3*p+2)/2)  # returnerer heltall for p = [1:5]
    X = np.zeros((len(x), m))
    X[:, 0] = 1
    counter = 1
    for i in range(1, p+1):
        for j in range(i+1):
            X[:, counter] = x**(i-j) * y**j
            counter += 1
    return X


def frankeFunction(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def model_comparison(models, param_grid, p, x, y, random_states):
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
            print(x.shape)
            x_train, x_test = bootstrap(x, random_state=random_state)
            y_train, y_test = bootstrap(y, random_state=random_state)

            X_train = generateDesignmatrix(p, x_train, y_train)
            X_test = generateDesignmatrix(p, x_test, y_test)

            z_train = frankeFunction(x_train, y_train)
            z_test = frankeFunction(x_test, y_test)

            # Pass algorithm + corresponding params to grid searcher and
            # determine optimal alpha param.
            grid = GridSearch(estimator, param_grid[name], random_state)


            grid.fit(X_train, X_test, z_train, z_test)

            # Store svg score values for each model.
            train_scores[name].append(np.mean(grid.train_scores))
            test_scores[name].append(np.mean(grid.test_scores))

    return train_scores, test_scores



"""
if __name__ == '__main__':
    # Demo run

    # A collection of algorithm name : algorithm.
    models = {
        "ols": algorithms.OLS,
        'ridge': algorithms.Ridge,
        #'lasso': sklearn.Lasso
        "lasso": algorithms.Lasso
    }
    param_grid = {
        "ols": [0]
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
"""
