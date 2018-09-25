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
from functions import bootstrap
from functions import train_test_split


class GridSearchNew:
    """Determines optimal hyperparameter for given algorithm."""

    def __init__(self, model, params, name):

        self.model = model
        self.params = params
        #self.random_state = random_state
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
        self.z_pred = None
        self.avg_z_pred = None
        self.best_avg_z_pred_r2 = None
        self.mse = None
        self.r2 = None
        self.best_z_pred_mse = None
        self.best_z_pred_r2 = None
        self.nboots = None
        self.test = None


    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """Computes the Mean Squared Error score metric."""
        mse = np.square(np.subtract(y_true, y_pred)).mean()
        # In case of matrix.
        if mse.ndim == 2:
            return np.sum(mse)
        else:
            return mse

    @staticmethod # forteller klassen at den ikke trenger self.
    def r_squared(y_true, y_pred):
        val = np.square(np.subtract(y_true, y_pred)).sum()/np.square(np.subtract(y_true, y_true.mean())).sum()
        return 1 - val



    def fit(self, X, z, split_size):
        """Searches for the optimal hyperparameter combination."""
        # model and params are now lists --> sende med navn istedenfor.
        # Setup
        self.results = {self.name: []}
        self.train_scores_mse, self.test_scores_mse = [], []
        self.train_scores_r2, self.test_scores_r2 = [], []
        self.best_mse = 500
        self.best_r2 = -500

        # looper over all lamda values
        # avg_z_pred = []  # this is the average of z_pred. Each element corresponds to one lambda
        for num, param in enumerate(self.params):

            self.nboots = 4
            random_states = np.arange(self.nboots)  # number of boots



            X_boot_std, X_boot_mean = [], []
            self.z_pred = []  # this has the length of random_states
            mse_boot = []  # this has the length of random_states
            r2_boot = []  # this has the length of random_states

            for num, random_state in enumerate(random_states):
                # Create new model instance.    ----> here or outside the loop?
                estimator = self.model(lmd=param)

                # Generate data with bootstrap
                X_subset, z_subset = bootstrap(X, z, random_state)
                # Pass algorithm + corresponding params to grid searcher and
                # determine optimal alpha param.
                X_train, X_test, z_train, z_test = train_test_split(
                    X_subset, z_subset, split_size=split_size
                )
                # TODO: May need to change axis
                X_boot_mean.append(np.mean(X_subset, axis=1))

                # Train a model for this pair of lambda and random state
                estimator.fit(X_train, z_train)
                # store mean prediction for this bootstrap selection in order to calculate the bias later
                self.z_pred.append(np.mean(estimator.predict(X_test)))
                # calculate the mse* for each loop and store the values
                mse_boot.append(self.mean_squared_error(z_test, estimator.predict(X_test)))
                r2_boot.append(self.r_squared(z_test, estimator.predict(X_test)))

                self.test = r2_boot  # test dimensions


            # For each lambda, save the average over boots (random_states)
            # of z_pred (which is already an average)
            self.avg_z_pred = (np.mean(self.z_pred))
            # also store the mse calculated as the mean of mse_boot
            self.mse = np.sum(mse_boot)/self.nboots
            self.r2 = np.sum(r2_boot)/self.nboots

            # # Compute score
            # score_mse = self.mean_squared_error(y_test, y_pred)
            # score_r2 = self.r2(y_test, y_pred)


            #----
            # TODO: skrive bias og covariance


            # for each model , save the best score (mse or r2),
            #  and the corresponding lambda and z_pred
            if self.mse < self.best_mse:
                self.best_mse = self.mse
                self.best_param_mse = param
                self.best_avg_z_pred_mse = self.avg_z_pred
                self.best_z_pred_mse = self.z_pred

            if self.r2 > self.best_r2:
                self.best_r2 = self.r2
                self.best_param_r2 = param
                self.best_avg_z_pred_r2 = self.avg_z_pred
                self.best_z_pred_r2 = self.z_pred


            # # Store both train and test scores to evaluate overfitting.
            # # Vi trenger egentlig ikke train score
            # # If train scores >> test scores ==> overfitting.
            # self.train_scores_mse.append(estimator.predict(X_train))  # This is not the score, it's y_pred ???
            # self.test_scores_mse.append(score_mse)
            #
            # self.train_scores_r2.append(estimator.predict(X_train))
            # self.test_scores_r2.append(score_r2)

            #print("best fit lamda " + self.name + " %0.2f  ", self.best_param_mse)
            #print("best fit lamda " + self.name + " %0.2f ", self.best_param_r2)
            return self

