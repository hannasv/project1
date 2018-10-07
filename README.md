# project1
Project 1 in fys-stk4155 at UiO

In this article we show that applying linear regression analysis to terrain data allows us to represent the main characteristics of the smoothed topography. Nevertheless, we are not able to describe all the features of the terrain with these simple models. The main reason why we cannot reproduce all the characteristics of the elevation is that we use polynomials to predict a surface that does not have a polynomial behaviour. We suspect that the fit could be improved by using other functions, or methods that are not linear.


As expected, an increase in the polynomial order (p \textit{write ",p" instead of (p)} ) leads to a better fit and a better prediction. This is quantified with a lower MSE and R-squared score closer to 1 for $p \in [1, 5 ]$.
%in the range $1<p<5$. 
On the other hand, an increase in the penalisation parameter for both Ridge and the LASSO regressions does not improve the quality of the fit, demonstrating that the Simple Linear Regression OLS, a particular case of Ridge regression (\textit{er det ikke lasso også om den lmd = 0?}) with a regularisation parameter equal to zero, is the best model for fitting terrain data. 
