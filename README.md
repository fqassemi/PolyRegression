# PolyRegression
Finding polynomial regression using simple NN. Following Horner's method to represent apolynomial of degree n as a composition of n linear function as following: p(x) = a_0+a_1 x+a_2 x^2+...+a_n x^n = a_0+x(a_1+x(a_2+x(...(a_n-1+a_n x))). Horner's method is proven to be quite useful for handling and computing polynomial expressions.  

This simple form provides a unique insight to generate a deep NN with m layers with Linear activation function (ax+b), however, mulliplying each node with the input, x. 
