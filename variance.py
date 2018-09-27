import numpy as np
from scipy import stats
import scipy.stats as st
# pass an array (?)

x = np.linspace(1, 5, 5)
n = len(x)
# First we need the sample mean
mu = np.sum(x)/n

# Now the unbiased variance
variance = (np.sum((x - mu)**2))/(n-1)

# Test for the variance score
test_variance=(x.var(ddof=1))


# Confidence interval
sigma = np.sqrt(variance)
se = sigma/np.sqrt(n)
p = 0.025
t_val = stats.t.ppf(1-p, n-1)
ci_up = mu + t_val*se
ci_low = mu - t_val*se

# Test intervals
st.t.interval(0.95, n-1, loc=mu, scale=st.sem(x))

