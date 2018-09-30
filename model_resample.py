from functions import bootstrap, train_test_split, variance, mean_squared_error, r2_score, ci
import numpy as np
import pandas as pd

# send in dictionaries with their best lmd values.
def model_resample(models, lmd, X, z, nboots, split_size = 0.2):
    """ lmd is a dictionarie of regression methods (name) with their corresponding best hyperparam
      Returns dict = {
                    "rigde": [mse, r2, var]
                    "OLS": [mse, r2, var]
                    "lasso":[mse, r2, var]
    }"""

    z_true_mean = z.mean()

    random_states = np.arange(nboots)  # number of boots
    mse = {"ridge": [], "lasso": [], "ols": []}
    r2 = {"ridge": [], "lasso": [], "ols": []}
    z_pred = {"ridge": [], "lasso": [], "ols": []}  # mean av alle b0
    reg_coeffs = {}

    for random_state in random_states:

        # Generate data with bootstrap
        X_subset, z_subset = bootstrap(X, z, random_state)

        X_train, X_test, z_train, z_test = train_test_split(
            X_subset, z_subset, split_size=split_size
        )

        reg_coeffs[random_state] = {}
        for name, model in models.items():
            estimator = model(lmd[name])
            # Train a model for this pair of lambda and random state
            estimator.fit(X_train, z_train)
            temp = estimator.predict(X_test)
            reg_coeffs[random_state][name] = estimator.coef_
            """  Keeping information for each model  """
            mse[name].append(mean_squared_error(z_test, temp))
            r2[name].append(r2_score(z_test, temp))
            z_pred[name].append(temp)

    mse_avg = {"ridge": np.array(mse["ridge"]).mean(),"lasso": np.array(mse["lasso"]).mean(),"ols": np.array(mse["ols"]).mean() }
    r2_avg = {"ridge": np.array(r2["ridge"]).mean(),"lasso": np.array(r2["lasso"]).mean(),"ols": np.array(r2["ols"]).mean() }

    return mse_avg, r2_avg, reg_coeffs, z_pred
