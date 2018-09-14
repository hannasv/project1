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


class Ridge:
    """The Ridge algorithm."""

    def __init__(self, alpha, random_state=None):

        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        """Train the model."""

        pass

    def predict(self, X):
        """Aggregate model predictions."""

        pass


class Lasso:
    """The LASSO algorithm."""

    def __init__(self, alpha, random_state=None):

        self.alpha = alpha
        self.random_state = random_state

    def fit(self, X, y):
        """Train the model."""

        pass

    def predict(self, X):
        """Aggregate model predictions."""

        pass
