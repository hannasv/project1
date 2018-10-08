import numpy as np
from scipy import stats
import scipy.stats as st
import matplotlib.pyplot as plt

def generateDesignmatrix(p, x, y):
    m = int((p**2+3*p+2)/2)  # returnerer heltall for p = [1:5]
    X = np.zeros((len(x), m))
    X[:, 0] = 1
    counter = 1
    for i in range(1, p+1):
        for j in range(i+1):
            X[:, counter] = x**(i-j) * y**j
            counter += 1
    return X


def franke_function(x, y):
    term1 = 0.75*np.exp(-(0.25*(9*x-2)**2) - 0.25*((9*y-2)**2))
    term2 = 0.75*np.exp(-((9*x+1)**2)/49.0 - 0.1*(9*y+1))
    term3 = 0.5*np.exp(-(9*x-7)**2/4.0 - 0.25*((9*y-3)**2))
    term4 = -0.2*np.exp(-(9*x-4)**2 - (9*y-7)**2)
    return term1 + term2 + term3 + term4


def bootstrap(X, z, random_state):

    # For random randint
    rgen = np.random.RandomState(random_state)

    nrows, ncols = np.shape(X)

    selected_rows = np.random.randint(
        low=0, high=nrows, size=nrows
    )

    z_subset = z[selected_rows]
    X_subset = X[selected_rows, :]

    return X_subset, z_subset


def train_test_split(X, z, split_size=0.2, random_state=None):

    # For random choice.
    np.random.seed(random_state)

    # Extract number of rows and columns in data matrix.
    nrows, ncols = np.shape(X)

    # Determine the proportion of training and test samples
    # from the data matrix size-
    ntest_samples = int(nrows * split_size)
    ntrain_samples = int(nrows - ntest_samples)
    # Randomly select indices for training and test samples
    # without replacement.
    row_samples = np.arange(nrows)
    selected_train_samples = np.random.choice(
        row_samples, ntrain_samples, replace=False
    )
    selected_test_samples = [
        sample for sample in row_samples
        if sample not in selected_train_samples
    ]

    X_train = X[selected_train_samples, :]
    X_test = X[selected_test_samples, :]
    z_train = z[selected_train_samples]
    z_test = z[selected_test_samples]

    return X_train, X_test, z_train, z_test


def mean_squared_error(y_true, y_pred):
    """Computes the Mean Squared Error score metric."""
    mse = np.square(np.subtract(y_true, y_pred)).mean()
    # In case of matrix.
    if mse.ndim == 2:
        return np.sum(mse)
    else:
        return mse

def r2_score(y_true, y_pred):
    numerator = np.square(np.subtract(y_true, y_pred)).sum()
    denominator = np.square(np.subtract(y_true, np.average(y_true))).sum()
    val = numerator/denominator
    return 1 - val

def variance(x):
    n = len(x)
    mu = np.sum(x)/n
    var = np.sum((x - mu)**2)/(n-1)
    return var

def ci(x):
    """  Calculating the confidence intervals of regression coefficients  """
    n = len(x)
    mu = np.sum(x)/n
    sigma = np.sqrt(variance(x))
    se = sigma/np.sqrt(n)
    p = 0.025
    t_val = stats.t.ppf(1-p, n-1)
    ci_up = mu + t_val*se
    ci_low = mu - t_val*se
    return ci_low, ci_up

def clean_reg_coeff(X, reg_coeff, nboots):
    """  This is a helpfunction to clean up the dictionary of coefficients so its easier to plot.  """
    nrCoeff = X.shape[1]
    B_r = np.zeros((nrCoeff, nboots))
    B_l = np.zeros((nrCoeff, nboots))
    B_o = np.zeros((nrCoeff, nboots))

    for i in range(nboots):
        B_r[:, i] = reg_coeff[i]["ridge"]
        B_l[:, i] = reg_coeff[i]["lasso"]
        B_o[:,i] = reg_coeff[i]["ols"]

    m = np.array([B_r[i,:].mean() for i in range(nrCoeff)])
    h = np.array([ci(B_r[i,:])[1] for i in range(nrCoeff)])
    l = np.array([ci(B_r[i,:])[0] for i in range(nrCoeff)])

    ml = np.array([B_l[i,:].mean() for i in range(nrCoeff)])
    hl = np.array([ci(B_l[i,:])[1] for i in range(nrCoeff)])
    ll = np.array([ci(B_l[i,:])[0] for i in range(nrCoeff)])

    mo = np.array([B_o[i,:].mean() for i in range(nrCoeff)])
    ho = np.array([ci(B_o[i,:])[1] for i in range(nrCoeff)])
    lo = np.array([ci(B_o[i,:])[0] for i in range(nrCoeff)])

    return m,l,h, ml, ll, hl, mo,lo,ho


def plotCI(X,m,l,h, ml,ll,hl, mo,lo,ho):
    """  Plot confidence intervals. """
    fig, ax = plt.subplots(figsize = (8,6))
    nrCoeff = X.shape[1]
    x = np.arange(nrCoeff)

    ax.fill_between(x, l, h, where = h>l, alpha = 0.3, interpolate = True)
    ax.fill_between(x, ll, hl, where = hl>ll,facecolor = "red", alpha = 0.3, interpolate = True)

    ax.fill_between(x, lo, ho, where = ho>lo,facecolor = "green", alpha = 0.3, interpolate = True)
    ax.plot(x, mo, c='g',  alpha=0.8,  label = "mean, ols")
    ax.plot(x, ho, c='g', alpha=0.6,   label = "95 percentile, ols")
    ax.plot(x, lo, c='g', alpha=0.6, label = "5 percentile, ols")

    # Outline of the region we've filled in
    ax.plot(x, ml, c='r',  alpha=0.8,  label = "mean, lasso")
    ax.plot(x, hl, c='r', alpha=0.6,   label = "95 percentile, lasso")
    ax.plot(x, ll, c='r', alpha=0.6, label = "5 percentile, lasso")

    # Outline of the region we've filled in
    ax.plot(x, m, c='b', alpha=0.8, label = "mean, ridge")
    ax.plot(x, h, c='b', alpha=0.6, label = "95 percentile, ridge")
    ax.plot(x, l, c='b', alpha=0.6, label = "5 percentile, ridge")
    plt.xlabel("Coefficent number, beta_i", fontsize = 15)
    plt.ylabel("y", fontsize = 15)
    plt.legend()
    return

def bias_square(z_true_mean, z_pred):
    """ Calculating model bias  """
    z_pred_mean = np.array([np.array(z).mean() for z in z_pred])
<<<<<<< HEAD
    val = np.mean(  np.square(  z_true_mean -  np.mean(z_pred_mean) ))
    return val
=======
    val = np.sum(  np.square(  z_true_mean -  np.mean(z_pred_mean) ))
    return np.sqrt(val)   # returner val
>>>>>>> dog

def model_variance(z_pred, nboots):
    val = [ (z_pred[i]  -  np.mean(z_pred[i])) for i in range(len(z_pred))  ]
    return np.mean(val)/nboots

def error(y_test, y_pred):
    square_diff =  [np.square(y_test - y_pred[i]) for i in range(np.shape(y_pred)[0])]
    return np.mean( np.mean(square_diff ))
