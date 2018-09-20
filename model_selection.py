# -*- coding: utf-8 -*-
#
# model_selection.py
#
# The module is part of model_comparison.
#

"""
The hyperparameter grid search framework.
"""

__author__ = 'Hanna Svennevik', 'Paulina Tedesco'
__email__ = 'email1', 'email2'


import numpy as np


class GridSearch:
    """Determines optimal hyperparameter for given algorithm."""

    def __init__(self, model, params, name, random_state=None):

        self.model = model
        self.params = params
        self.random_state = random_state
        self.name = name

        # NOTE: Attribuets modified with instance.
        self.best_score = None
        self.best_param = None
        self.train_scores = None
        self.test_scores = None
    """
    @property --> func to variable
    def best_score(self):
        return self._best_score

    @best_score.setter
    def best_score(self, value):

        if value is None:
            return
        else:
            return float(value)
    """

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """Computes the Mean Squared Error score metric."""
        mse = np.square(np.subtract(y_true, y_pred)).mean()
        # In case of matrix.
        if mse.ndim == 2:
            return np.sum(mse)
        else:
            return mse

    # TODO: Creating a R2-square fuction: Skriv denne som above
    @staticmethod # forteller klassen at den ikke trenger self.
    def r2(y, y_predict):

        C = y-y_predict
        val = sum(sum((y-y_predict))**2)/sum(sum((y-np.mean(y))**2))
        return 1 - val


    def fit(self, X_train, X_test, y_train, y_test):
        """Searches for the optimal hyperparameter combination."""
        # model and params are now lists --> sende med navn istedenfor.
        # Setup
        self.results = {self.name: []}
        self.train_scores, self.test_scores = [], []

        self.best_score = 0.0
        # looper over all lamda values
        for num, param in enumerate(self.params):

            # Create new model instance.
            estimator = self.model(lmd=param, random_state=self.random_state)

            # Train a model for this alpha value.
            estimator.fit(X_train, y_train)
            # Aggregate predictions to determine how `good` the model is.
            y_pred = estimator.predict(X_test)
            # Compute score.
            score = self.mean_squared_error(y_test, y_pred)
            # Lag en dictionary med r2 score ogsaa

            # Save best alpha and best score:
            if score > self.best_score:
                self.best_score = score
                self.best_param = param

            # Store both train and test scores to evaluate overfitting.
            # If train scores >> test scores ==> overfitting.
            self.train_scores.append(estimator.predict(X_train))
            self.test_scores.append(score)

            return self

"""
if __name__ == '__main__':
    # Demo run
    import algorithms
    import numpy as np

    models = {'ridge': algorithms.Ridge, "ols": algorithms.OLS, "lasso": Algorithmns.Lasso}
    param_grid = {'ridge': [0.01, 0.1, 1.0, 10.0], "ols": 0, 'lasso': [0.01, 0.1, 1.0, 10.0]}

    random_states = np.arange(40)
    for random_state in random_states:

        # Set random seed for reproducibility:
        np.random.seed(0)

        # Generate data (bootstrap sampling in your case).
        X_dummy = np.random.random((10, 5))
        y_dummy = np.random.random((10, 1))

        # Determine optimal alpha param.
        grid = GridSearch(estimator, param_grid[name], random_state)
        grid.fit(X, y)

        # Print the optimal selected alpha value.
        print(grid.best_param)

        # Print the score of of the corresponing alpha value.
        print(grid.best_score)
"""
