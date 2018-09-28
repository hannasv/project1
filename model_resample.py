from functions import bootstrap, train_test_split, variance, ci
import numpy as np

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
    beta = {"ridge": [], "lasso": [], "ols": []}
    beta0_mean = {"ridge": [], "lasso": [], "ols": []}

    for random_state in random_states:

        # Generate data with bootstrap
        X_subset, z_subset = bootstrap(X, z, random_state)
        X_train, X_test, z_train, z_test = train_test_split(
            X_subset, z_subset, split_size=split_size
        )

        for name, estimator in models.items():
            estimator = self.model(lmd[name])
            # Train a model for this pair of lambda and random state
            estimator.fit(X_train, z_train)
            temp = estimator.predict(X_test)

            """  Keeping information for each model  """
            mse[name].append(mean_squared_error(z_test, temp))
            r2[name].append(r2_score(z_test, temp))
            z_pred[name].append(temp)
            beta[name].append(estimator.coef_)

    mse_avg = {"ridge": mse["ridge"].mean(),"lasso": mse["lasso"].mean(),"ols": mse["ols"].mean() }
    r2_avg = {"ridge": r2["ridge"].mean(),"lasso": r2["lasso"].mean(),"ols": r2["ols"].mean() }

    z_pred_mean = [z.mean() for z in z_pred[name]]  # this is te mean for all the boots
    bias = abs(z_true_mean - z_pred_mean.mean())
    model_variance = np.sum(z_pred[name] - z_pred_mean)/nboots

    beta0_mean["rigde"] = np.mean(beta["rigde"][0, :])
    beta0_mean["lasso"] = np.mean(beta["lasso"][0, :])
    beta0_mean["ols"] = np.mean(beta["ols"][0, :])
    # bruker variancen av alle Beta0
    ci = {"ridge": ci(beta0_mean["ridge"]), "lasso": ci(beta0_mean["lasso"]), "ols": ci(beta0_mean["ols"])}

    return mse_avg, r2_avg, bias, model_variance
