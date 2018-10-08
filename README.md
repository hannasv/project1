# Project 1 - Regression anaysis with resampling
This cite contains all the material for our solution of project 1 in the course fys-stk4155 at UiO. 

Describe the content (which files are here) and how to run them.

## Abstract
In this project we show that applying linear regression analysis to terrain data allows us to represent the main characteristics of the smoothed topography, although we are not able to describe all the features of the terrain with these simple models. 

We fit the terrain data with polynomials of degree, p where p in [1,5]. As expected, an increase in the polynomial degree, $p$ leads to a better fit and a better prediction. This is quantified with a lower MSE and R-squared score closer to 1. However, the error depends strongly on the characteristics of the topography, for instance, the slope and the regularity of the terrain, whether there are large elevation differences in a small areas. 

On the other hand, an increase in the penalisation parameter for both Ridge- and LASSO regressions does not improve the quality of the fit, demonstrating that the Simple Linear Regression OLS, a particular case of these two types of regression with a regularisation parameter equal to zero, is the best model for fitting terrain data. 

After comparing the error to the bias and the variance, we can affirm that the optimal polynomial degree is higher than 5. We are aware of that the main reason why we cannot reproduce all the characteristics of the elevation is that we use polynomials to predict a surface that does not have a polynomial behaviour. We suspect that the fit could be improved by using other functions, or methods that are non-linear. This would be implemented in a future work.
