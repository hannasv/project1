

def model_resample(model, lmd, X, y, nboots):
    random_states = np.arange(nboots)  # number of boots

    # Generate data with bootstrap
    X_subset, z_subset = bootstrap(X, z, random_state)
    X_train, X_test, z_train, z_test = train_test_split(
        X_subset, z_subset, split_size=split_size
    )


    # For each lambda, save the average over boots (random_states)
    # of z_pred (which is already an average)
    self.avg_z_pred = (np.mean(z_pred)) # kan vi ikke bare sette denne rett i if-testen?
    # also store the mse calculated as the mean of mse_boot
    self.mse = np.sum(mse_boot)/nboots
    self.r2 = np.sum(r2_boot)/nboots

    # # Compute score
    # score_mse = self.mean_squared_error(y_test, y_pred)
    # score_r2 = self.r2(y_test, y_pred)


    return 
