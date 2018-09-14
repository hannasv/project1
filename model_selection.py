# -*- coding: utf-8 -*-
#
# model_selection.py
#
# The module is part of model_comparison.
#

"""
The hyperparameter grid search framework.
"""

__author__ = 'author 1', 'author 2'
__email__ = 'email1', 'email2'


class GridSearch:

    def __init__(self, model, param_grid, verbose=True, random_state=None):

        self.model = model
        self.param_grid = param_grid
        self.verbose = True
        self.random_state = random_state

        # NOTE: Attribuets modified with instance.
        self.best_score = None
        self.best_param = None
        self.train_scores = None
        self.test_scores = None

# funker denne for baae vektor og matrise???
    def mean_squared_error(self, y_true, y_pred):
        """Computes the Mean Squared Error score metric.""""
        return np.square(np.subtract(y_true, y_pred)).mean()

        # Creating a R2-square fuction:
    def R2(y, y_predict):
        C = y-y_predict
        val = sum(sum((y-y_predict))**2)/sum(sum((y-np.mean(y))**2))
        return 1 - val

    # Creating a mean square error function:
    def MSE(y, y_predict):
        C = y-y_predict
        [n, m] = C.shape
        return sum(sum((C)**2))/(n*m)


    def fit(self, X_train, X_test, y_train, y_test):
        """Searches for the optimal hyperparameter combination."""

        name, params = self.param_grid.items()

        # Setup
        self.results = {name: []}
        self.train_scores, self.test_scores = [], []

        for num, param in enumerate(params):

            # Prints an update on each round.
            if self.verbose:
                print('Grid search round: {}'.format(num + 1))

            # Create new model instance.
            estimator = model(alpha=param, random_state=self.random_state)
            # Train a model for this alpha value.
            estimator.fit(X_train, y_train)
            # Aggregate predictions to determine how `good` the model is.
            y_pred = estimator.predict(X_test)
            # Compute score.
            score = self.mean_squared_error(y_test, y_pred)

            # Save best alpha and best score:
            if score > self.best_score:
                self.best_score = score
                self.best_param = alpha

                # Prints nre bets score with param name and value.
                if self.verbose:
                    print('New best score: {} with param {} value: {}'
                          ''.format(score, name, param))

            # Store both train and test scores to evaluate overfitting.
            # If train scores >> test scores ==> overfitting.
            self.test_scores.append(estimator.predict(X_train))
            self.train_scores.append(score)

            return self


if __name__ == '__main__':
    # Demo run
    import algorithms
    import numpy as np

    models = {'ridge': algorithms.Ridge,}
    param_grid = {'ridge': [0.01, 0.1, 1.0, 10.0]}

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
