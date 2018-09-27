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
__email__ = 'hanna.svennevik@fys.uio.no', 'paulinatedesco@gmail.com'


import numpy as np
from functions import bootstrap, mean_squared_error, train_test_split, r2_score

class GridSearchNew:
    """
    Determines optimal hyperparameter for given algorithm, without resampling.
    Returns dict = {
                    "rigde": [mse, r2, var]
                    "OLS": [mse, r2, var]
                    "lasso":[mse, r2, var]
    }
    """

    def __init__(self, model, params, name):
        self.model = model
        self.params = params
        self.name = name

        # NOTE: Attribuets modified with instance.
        # self.train_scores_mse = None
        # self.test_scores_mse = None
        # self.train_scores_r2 = None
        # self.test_scores_r2 = None
        self.best_mse = None
        self.best_r2 = None
        self.best_param_mse = None
        self.best_param_r2 = None
        self.best_avg_z_pred_mse = None
        self.avg_z_pred = None
        self.best_avg_z_pred_r2 = None
        self.mse = None
        self.r2 = None
        print("created a estimator")

    def fit(self, X, z, split_size):
        """Searches for the optimal hyperparameter combination."""
        # model and params are now lists --> sende med navn istedenfor.
        # Setup
        self.results = {self.name: []}
        self.train_scores_mse, self.test_scores_mse = [], []
        self.train_scores_r2, self.test_scores_r2 = [], []
        self.best_mse = 50
        self.best_r2 = -50

        # Splitting our original dataset into test and train.
        X_train, X_test, z_train, z_test = train_test_split(
            X, z, split_size=split_size
        )

        " Returning these dictionaries to plot mse vs model"
        self.mse = []
        #{"ridge":[], "ols": [], "lasso":[]}
        self.r2 = []
        #{"ridge":[], "ols": [], "lasso":[]}
        self.z_pred = []
        print(self.params)
        # For en model tester vi alle parameterne og returnerer denne.
        for param in self.params:
            print("lambda: " + str(param))
            estimator = self.model(lmd = param)
            # Train a model for this pair of lambda and random state
            estimator.fit(X_train, z_train)
            temp = estimator.predict(X_test)
            self.mse.append(mean_squared_error(z_test, temp))
            self.r2.append(r2_score(z_test, temp))
            self.z_pred.append(temp)

            print(len(self.mse))

            "Les gjennom dette igjen..."
            """
            if self.mse < self.best_mse: # the best mse score is close to zero
                self.best_mse = self.mse
                self.best_param_mse = param
                self.best_avg_z_pred_mse = self.avg_z_pred

            if self.r2 > self.best_r2:
                # the best r2 scor is close to 1.
                self.best_r2 = self.r2
                self.best_param_r2 = param
                self.best_avg_z_pred_r2 = self.avg_z_pred
            """
        return self
