import pytest
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_squared_log_error, mean_absolute_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import algorithms
import model_selection_new
from model_comparison_new import model_comparison_new
from functions import generateDesignmatrix, franke_function, train_test_split

np.random.seed(1000)
m = 30  # m defines the size of the meshgrid.
x = np.random.rand(m, )
y = np.random.rand(m, )
z = franke_function(x, y)

def convert(X):
    m = len(X)
    n = len(X[0])
    new = np.zeros((n,m))
    for i in range(m): #3
        for j in range(n): # 30
            new[j][i] = X[i][j]
    return new

def test_design():
    # TODO: make seperate for all p's??
    for p in range(1,6):
        if (p==1):
            X = [np.ones(len(x)), x, y]
            assert np.all(abs(convert(X)- generateDesignmatrix(p,x,y))<1e8)

        elif(p==2):
            X = [np.ones(len(x)), x, y, x**2, x*y, y**2]
            assert np.all(abs(convert(X)- generateDesignmatrix(p,x,y))<1e8)

        elif(p==3):
            X = [np.ones(len(x)), x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3]
            assert np.all(abs(convert(X)- generateDesignmatrix(p,x,y))<1e8)

        elif(p==4):
            X = [np.ones(len(x)), x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3, x**4, x**3*y, x**2*y**2, x*y**3, y**4]
            assert np.all(abs(convert(X)- generateDesignmatrix(p,x,y))<1e8)

        elif(p==5):
            X = [np.ones(len(x)), x, y, x**2, x*y, y**2, x**3, x**2*y, x*y**2, y**3, x**4, x**3*y, x**2*y**2, x*y**3, y**4,
                x**5, y*x**4, x**3*y**2, x**2*y**3, x*y**4, y**5]
            assert np.all(abs(convert(X)- generateDesignmatrix(p,x,y))<1e8)

def test_mse():
    assert abs(mean_squared_error(x,y) - model_selection_new.GridSearchNew.mean_squared_error(x, y)) < 1e-8

def test_r2():
    assert abs( r2_score(x, y) -  model_selection_new.GridSearchNew.r_squared(x, y) ) < 1e-8

p = 2
X = generateDesignmatrix(p,x,y)

def test_ols():
    our_ols = algorithms.OLS()
    our_ols.fit(X,z)
    our_betas = our_ols.beta

    # Create linear regression object
    scikit_ols = LinearRegression(fit_intercept=False)
    # Train the model using the training sets
    scikit_ols.fit(X, z)
    assert np.all(abs(our_betas - scikit_ols.coef_[:])<1e8)


def test_ridge():
    our_ridge = algorithms.Ridge(lmd = 0.1)
    our_ridge.fit(X,z)
    our_betas = our_ridge.beta
    scikit_ridge=linear_model.RidgeCV(alphas=[0.1], fit_intercept=False)
    scikit_ridge.fit(X, z)
    assert  np.all(abs( our_betas == scikit_ridge.coef_[:])<1e8)
