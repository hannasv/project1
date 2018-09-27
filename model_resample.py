from function import bootstrap, train_test_split, variance, ci


# send in dictionaries with their best lmd values.

def model_resample(models, lmd, X, z, nboots, split_size = 0.2):
""" lmd is a dictionarie of regression methods (name) with their corresponding best hyperparam """

    z_true_mean = z.mean()

    random_states = np.arange(nboots)  # number of boots
    mse = { "ridge":[], "lasso":[], "ols":[]}
    r2 = { "ridge":[], "lasso":[], "ols":[]}
    z_pred = { "ridge":[], "lasso":[], "ols":[]} # mean av alle b0

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
            mse[name].append(mean_squared_error(z_test, temp))
            r2[name].append(r2_score(z_test, temp))
            z_pred[name].append(temp)

    mse_avg = {"ridge": mse["ridge"].mean(),"lasso": mse["lasso"].mean(),"ols": mse["ols"].mean() }
    r2_avg = {"ridge": r2["ridge"].mean(),"lasso": r2["lasso"].mean(),"ols": r2["ols"].mean() }

    # for hver boot --> calculate mean betas.
    for z in z_pred[name]:
        z_pred_mean = z.mean()

    bias = abs(z_true_mean - z_pred_mean.mean())
    model_variance =

    """ For hver boot tar vi z_pred -z_pred.mean() --> summerer over alle bots/nrBoots """

    # Calculate variance of all
    #variance = {"ridge": var(z_pred["ridge"].mean()),"lasso": mse["lasso"].mean(),"ols": mse["ols"].mean() }

    # bruker variancen av alle Beta0
    #ci = {"ridge": r2["ridge"].mean(),"lasso": r2["lasso"].mean(),"ols": r2["ols"].mean() }

    return mse_avg, r2_avg, bias
