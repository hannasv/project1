# Project 1 - Regression anaysis with resampling
This cite contains all the material for our answar to the assignment project 1 in the course fys-stk4155 at UiO. 

Each regression method implemented as a class in the file algorithms.py, with their own class functions fit and predict. The script utils.py contains functions accessed from the notebooks and code used in this report. All the functions in utils.py where tested in the file test_project1.py. In order to execute the tests you make the following call in your command line. 

pytest test_project1.py


We have structured the solution into two parts. The first part consists of finding the optimal hyperparameter $\lambda$. This is done with a call to the model_comparison function, which creates one instance of a gridsearch object for each regression method given as an input. The gridsearch object fits all of the methods and returns itself. The gridsearch object now contains all the properties of the regression. These can now easily be accessed in model_comparison which returns a dictionary of the performance metrics. This is done in the Project1_FYSSTK4155.ipynb. 

Folder with results contain results not shown but reffered to in the pdf.

## Abstract
In this project we show that applying linear regression analysis to terrain data allows us to represent the main characteristics of the smoothed topography, although we are not able to describe all the features of the terrain with these simple models. 

We fit the terrain data with polynomials of degree, p where p in [1,5]. As expected, an increase in the polynomial degree, $p$ leads to a better fit and a better prediction. This is quantified with a lower MSE and R-squared score closer to 1. However, the error depends strongly on the characteristics of the topography, for instance, the slope and the regularity of the terrain, whether there are large elevation differences in a small areas. 

On the other hand, an increase in the penalisation parameter for both Ridge- and LASSO regressions does not improve the quality of the fit, demonstrating that the Simple Linear Regression OLS, a particular case of these two types of regression with a regularisation parameter equal to zero, is the best model for fitting terrain data. After comparing the error to the bias and the variance, we can affirm that the optimal polynomial degree is higher than 5. We are aware of that the main reason why we cannot reproduce all the characteristics of the elevation is that we use polynomials to predict a surface that does not have a polynomial behaviour. We suspect that the fit could be improved by using other functions, or methods that are non-linear. This would be implemented in a future work.
