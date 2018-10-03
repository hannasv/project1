from utils import bootstrap, train_test_split, variance, mean_squared_error, r2_score, ci, bias_square, model_variance, error
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

    mse_train = {"ridge": [], "lasso": [], "ols": []}
    r2_train = {"ridge": [], "lasso": [], "ols": []}
    z_pred_train = {"ridge": [], "lasso": [], "ols": []}  # mean av alle b0

    # Splitting the data in order to test all of the resamples against the same test data.
    X_train, X_test, z_train, z_test = train_test_split(
        X, z, split_size=split_size
    )

    for random_state in random_states:
        # Resample the training data.
        X_, z_ = bootstrap(X_train, z_train, random_state)

        reg_coeffs[random_state] = {}
        for name, model in models.items():
            # creating a model with the previosly known best lmd
            estimator = model(lmd[name])
            # Train a model for this pair of lambda and random state
            """  Keeping information for test """
            estimator.fit(X_, z_)
            pred  = estimator.predict(X_test)
            reg_coeffs[random_state][name] = estimator.coef_
            mse_test[name].append(mean_squared_error(z_test, pred))
            r2_test[name].append(r2_score(z_test, pred))
            z_pred_test[name].append(pred)

            """Lagre information about training """
            pred = estimator.predict(X_)
            mse_train[name].append(mean_squared_error(z_train, pred))
            r2_train[name].append(r2_score(z_train, pred))
            z_pred_train[name].append(pred)

    """   Calulations done to get the information on correct format    """
    mse_avg_test = {"ridge": np.array(mse_test["ridge"]).mean(),"lasso": np.array(mse_test["lasso"]).mean(),"ols": np.array(mse_test["ols"]).mean() }
    r2_avg_test = {"ridge": np.array(r2_test["ridge"]).mean(),"lasso": np.array(r2_test["lasso"]).mean(),"ols": np.array(r2_test["ols"]).mean() }
    bias_model_test = {"ridge": bias_square(z_true_mean, z_pred_test["ridge"]), "ols":bias_square(z_true_mean, z_pred_test["ols"]), "lasso": bias_square(z_true_mean, z_pred_test["lasso"])}
    mv_test = {"ridge": model_variance(z_pred_test["ridge"], nboots), "ols":model_variance(z_pred_test["ols"], nboots), "lasso":model_variance(z_pred_test["lasso"], nboots)}

    mse_train = {"ridge": np.array(mse_train["ridge"]).mean(),"lasso": np.array(mse_train["lasso"]).mean(),"ols": np.array(mse_train["ols"]).mean() }
    r2_train = {"ridge": np.array(r2_train["ridge"]).mean(),"lasso": np.array(r2_train["lasso"]).mean(),"ols": np.array(r2_train["ols"]).mean() }
    bias_model_train = {"ridge": bias_square(z_true_mean, z_pred_train["ridge"]), "ols":bias_square(z_true_mean, z_pred_train["ols"]), "lasso": bias_square(z_true_mean, z_pred_train["lasso"])}
    mv_train = {"ridge": model_variance(z_pred_train["ridge"], nboots), "ols":model_variance(z_pred_train["ols"], nboots), "lasso":model_variance(z_pred_train["lasso"], nboots)}

    err_test = {"ols": error(z_test, z_pred_test["ols"]),
            "ridge": error(z_test, z_pred_test["ridge"]),
            "lasso": error(z_test, z_pred_test["lasso"])}

    err_train = {"ols": error(z_train, z_pred_train["ols"]),
                 "ridge": error(z_train, z_pred_train["ridge"]),
                 "lasso": error(z_train, z_pred_train["lasso"])}


    return mse_avg_test, r2_avg_test, reg_coeffs, bias_model_test, mv_test, mse_train, r2_train,  bias_model_train,  mv_train, err_test, err_train
