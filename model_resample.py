from utils import bootstrap, train_test_split, variance, mean_squared_error, r2_score, ci, bias, model_variance
import numpy as np

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
    mse_test = {"ridge": [], "lasso": [], "ols": []}
    r2_test = {"ridge": [], "lasso": [], "ols": []}
    z_pred_test = {"ridge": [], "lasso": [], "ols": []}  # mean av alle b0
    reg_coeffs = {}
    "       ----------------------"
    mse_train = {"ridge": [], "lasso": [], "ols": []}
    r2_train = {"ridge": [], "lasso": [], "ols": []}
    z_pred_train = {"ridge": [], "lasso": [], "ols": []}  # mean av alle b0

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
            """  Keeping information for test """
            estimator.fit(X_train, z_train)
            temp = estimator.predict(X_test)
            reg_coeffs[random_state][name] = estimator.coef_
            mse_test[name].append(mean_squared_error(z_test, temp))
            r2_test[name].append(r2_score(z_test, temp))
            z_pred_test[name].append(temp)


            """Lagre information about training """
            temp = estimator.predict(X_train)
            mse_train[name].append(mean_squared_error(z_train, temp))
            r2_train[name].append(r2_score(z_train, temp))
            z_pred_train[name].append(temp)

    """   Calulations done to get the information on correct format    """
    mse_avg_test = {"ridge": np.array(mse_test["ridge"]).mean(),"lasso": np.array(mse_test["lasso"]).mean(),"ols": np.array(mse_test["ols"]).mean() }
    r2_avg_test = {"ridge": np.array(r2_test["ridge"]).mean(),"lasso": np.array(r2_test["lasso"]).mean(),"ols": np.array(r2_test["ols"]).mean() }
    bias_model_test = {"ridge": bias(z_true_mean, z_pred_test["ridge"]), "ols":bias(z_true_mean, z_pred_test["ols"]), "lasso": bias(z_true_mean, z_pred_test["lasso"])}
    mv_test = {"ridge": model_variance(z_pred_test["ridge"], nboots), "ols":model_variance(z_pred_test["ols"], nboots), "lasso":model_variance(z_pred_test["lasso"], nboots)}

    mse_train = {"ridge": np.array(mse_train["ridge"]).mean(),"lasso": np.array(mse_train["lasso"]).mean(),"ols": np.array(mse_train["ols"]).mean() }
    r2_train = {"ridge": np.array(r2_train["ridge"]).mean(),"lasso": np.array(r2_train["lasso"]).mean(),"ols": np.array(r2_train["ols"]).mean() }
    bias_model_train = {"ridge": bias(z_true_mean, z_pred_train["ridge"]), "ols":bias(z_true_mean, z_pred_train["ols"]), "lasso": bias(z_true_mean, z_pred_train["lasso"])}
    mv_train = {"ridge": model_variance(z_pred_train["ridge"], nboots), "ols":model_variance(z_pred_train["ols"], nboots), "lasso":model_variance(z_pred_train["lasso"], nboots)}
<<<<<<< HEAD

    return mse_avg_test, r2_avg_test, reg_coeffs, bias_model_test, mv_test, mse_train, r2_train,  bias_model_train,  mv_train
=======
    
    return mse_avg_test, r2_avg_test, reg_coeffs, bias_model_test, mv_test, mse_train, r2_train,  bias_model_train,  mv_train, z_pred_test, z_pred_train, z_test, z_train
>>>>>>> dog
