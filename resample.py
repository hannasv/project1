# send in dictionaries with their best lmd values.
from utils import bootstrap, train_test_split, variance, mean_squared_error, r2_score, ci, bias, model_variance
import numpy as np
import pandas as pd


def resample(models, lmd, X, z, nboots, split_size=0.2):

    """ lmd is a dictionary of regression methods (name) with their corresponding best hyperparam

      Returns dict = {
                    "mse_avg": { "ridge": , "ols": , "lasso":  },
                    "r2_avg": { "ridge": , "ols": , "lasso":  },
                    "bias_model": { "ridge": , "ols": , "lasso":  },
                    "model variance" : { "ridge": , "ols": , "lasso":  }
                    "  regression coefficient   " :  {"0": [], "1":[] ,...., for all boots}
    }"""

    np.random.seed(2018)

    # Spilt the data in tran and split
    X_train, X_test, z_train, z_test = train_test_split(
        X, z, split_size=split_size
    )

    for name, model in models.items():
        # creating a model with the previosly known best lmd
        estimator = model(lmd[name])
        # Train a model for this pair of lambda and random state
        """  Keeping information for test """
        estimator.fit(X_train, z_train)
        z_pred = np.empty((z_test.shape[0], nboots))
        for i in range(nboots):
            X_, z_ = bootstrap(X_train, z_train, i)  # i is now also the random state for the bootstrap

            estimator.fit(X_, z_)
            # Evaluate the new model on the same test data each time.
            z_pred[:, i] = estimator.predict(X_test)

        z_test = z_test.reshape((z_test.shape[0], 1))
        error = np.mean(np.mean((z_test - z_pred) ** 2, axis=1, keepdims=True))
        bias = np.mean((z_test - np.mean(z_pred, axis=1, keepdims=True)) ** 2)
        variance = np.mean(np.var(z_pred, axis=1, keepdims=True))
        print('Error:', error)
        print('Bias^2:', bias)
        print('Var:', variance)
        print('{} >= {} + {} = {}'.format(error, bias, variance, bias + variance))


