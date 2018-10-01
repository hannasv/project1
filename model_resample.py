<<<<<<< HEAD
from functions import bootstrap, train_test_split, variance, mean_squared_error, r2_score, ci
=======
from utils import bootstrap, train_test_split, variance, mean_squared_error, r2_score, ci, bias, model_variance
>>>>>>> cat
import numpy as np
import pandas as pd

# send in dictionaries with their best lmd values.
def model_resample(models, lmd, X, z, nboots, split_size = 0.2):
    """ lmd is a dictionary of regression methods (name) with their corresponding best hyperparam

      Returns dict = {
                    "mse_avg": { "ridge": , "ols": , "lasso":  },
                    "r2_avg": { "ridge": , "ols": , "lasso":  },
                    "bias_model": { "ridge": , "ols": , "lasso":  },
                    "model variance" : { "ridge": , "ols": , "lasso":  }
                    "  regression coefficient   " :  {"0": [], "1":[] ,...., for all boots}
    }"""

    z_true_mean = z.mean() # need this to calculate the variance.
    random_states = np.arange(nboots)  # number of boots

    """ Dictionaires to keep track of the results  """
    mse = {"ridge": [], "lasso": [], "ols": []}
    r2 = {"ridge": [], "lasso": [], "ols": []}
    z_pred = {"ridge": [], "lasso": [], "ols": []}  # mean av alle b0
<<<<<<< HEAD
    bias = {"ridge": [], "lasso": [], "ols": []}
    model_variance = {"ridge": [], "lasso": [], "ols": []}
    beta = {"ridge": [], "lasso": [], "ols": []}
    beta0_mean = {"ridge": [], "lasso": [], "ols": []}
=======
    reg_coeffs = {}
>>>>>>> cat

    for random_state in random_states:

        # Resample and split data.
        X_subset, z_subset = bootstrap(X, z, random_state)

        X_train, X_test, z_train, z_test = train_test_split(
            X_subset, z_subset, split_size=split_size
        )

        reg_coeffs[random_state] = {}
        for name, model in models.items():
            # creating a model with the previosly known best lmd
            estimator = model(lmd[name])
            # Train a model for this pair of lambda and random state
            estimator.fit(X_train, z_train)
            temp = estimator.predict(X_test)
            reg_coeffs[random_state][name] = estimator.coef_
            """  Keeping information for each model  """
            mse[name].append(mean_squared_error(z_test, temp))
            r2[name].append(r2_score(z_test, temp))
            z_pred[name].append(temp)

            """  Lagre test coeff ogsaa -->  """

    """   Calulations done to get the information on correct format    """
    mse_avg = {"ridge": np.array(mse["ridge"]).mean(),"lasso": np.array(mse["lasso"]).mean(),"ols": np.array(mse["ols"]).mean() }
    r2_avg = {"ridge": np.array(r2["ridge"]).mean(),"lasso": np.array(r2["lasso"]).mean(),"ols": np.array(r2["ols"]).mean() }
    bias_model = {"ridge": bias(z_true_mean, z_pred["ridge"]), "ols":bias(z_true_mean, z_pred["ols"]), "lasso": bias(z_true_mean, z_pred["lasso"])}
    mv = {"ridge": model_variance(z_pred["ridge"], nboots), "ols":model_variance(z_pred["ols"], nboots), "lasso":model_variance(z_pred["lasso"], nboots)}

<<<<<<< HEAD
    # z_pred_mean = [np.array(z).mean() for z in z_pred[name]]  # this is te mean for all the boots
    # bias = abs(z_true_mean - np.array(z_pred_mean).mean())
    # model_variance = np.sum(np.array(z_pred_mean) - np.array(z_pred_mean).mean())/nboots

    for name, model in models.items():
        print(name)
        z_pred_mean = [np.array(z).mean() for z in z_pred[name]]
        # z_pred_mean = np.array(z).mean()
        bias[name] = abs(z_true_mean - np.array(z_pred_mean).mean())
        model_variance[name] = np.sum(np.array(z_pred_mean) - np.array(z_pred_mean).mean()) / nboots

    # print(np.array(z_pred_mean))
    # print(z_pred[name])




    # print(np.array(z_pred_mean).shape)
    # print(np.array(z_pred_mean).mean().shape)
    # print(z_true_mean.shape)

    beta0_mean["ridge"] = np.array(beta["ridge"]).mean(axis=0)
    beta0_mean["lasso"] = np.array(beta["lasso"]).mean(axis=0)
    beta0_mean["ols"] = np.array(beta["ols"]).mean(axis=0)
    # bruker variancen av alle Beta0

    print(np.array(beta0_mean["ols"]).shape)

    ci_coefs = {"ridge": ci(np.array(beta0_mean["ridge"])), "lasso": ci(np.array(beta0_mean["lasso"])), "ols": ci(np.array(beta0_mean["ols"]))}

    return mse_avg, r2_avg, bias, model_variance, ci_coefs
=======
    return mse_avg, r2_avg, reg_coeffs, bias_model, mv
>>>>>>> cat
