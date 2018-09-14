# -*- coding: utf-8 -*-
#
# algorithms.py
#
# The module is part of model_comparison.
#

"""
Representations of algorithms.
"""

__author__ = 'author 1', 'author 2'
__email__ = 'email1', 'email2'

class OLS:
    """The ordinary least squares algorithm"""

    def __init__(self, random_state = None):

        self.random_state = random_state
        # lagre beta her?
        self.beta = beta


    def fit(self, X, y):
        """Train the model"""
        self.beta = np.linalg.inv(X.T @ X)@ X.T @ y

    def predict(self, X):
        """Aggregate model predictions """
        return X @ beta




class Ridge:
    """The Ridge algorithm."""

    def __init__(self, lmd, random_state=None):

        self.lmd = lmd
        self.random_state = random_state
        self.beta = beta

    def fit(self, X, y):
        """Train the model."""
        self.beta = np.linalg.inv(X.T @ X - lmd*np.eyes(len(y)))@ X.T @ y


    def predict(self, X):
        """Aggregate model predictions."""
        return X @ beta


class Lasso:
    """The LASSO algorithm."""
    # sklearn.linear_model.Lasso
    # when the first column is 1 the use fit_intercept = True.

    def __init__(self, lmd, random_state=None):

        self.lmd = lmd
        self.random_state = random_state
        self.model = model

    def fit(self, X, y):
        """Train the model."""
        # from sklearn.linear_model import Lasso
        model = Lasso(self.lmd)
        model.fit(X, y)

    def predict(self, X):
        """Aggregate model predictions."""
        return model.predict(X)
