#X*, x test
#X train

import numpy as np
import math
from sklearn.gaussian_process import GaussianProcess
from matplotlib import pyplot as pl

np.random.seed(1)

def matrix (X1, X2): 
    matr=[]
    for row in xrange(len(X1)): matr += [[0]*len(X2)]
   
    for i in range(len(X1)):
        for l in range(len(X2)):
            matr[i][l] = cov(X1[i], X2[l])
    return np.array(matr)    
        

def cov(x1, x2):
	return math.exp ((abs(x1-x2)*abs(x1-x2))/(-2.0))

def f(x):
    """The function to predict."""
    return x * np.sin(x)

#  First the noiseless case
X = np.array([1, 3, 4, 6.5])

# Observations
y = f(X)

# Mesh the input space for evaluations of the real function, the prediction and
# its MSE
x = np.array([2, 3.5])

#matr =  matrix([1, 2, 3], [1, 2, 3])

#for m in matr:
#    print m

#K = np.array([[2, 1], [1, 4]])
#L = np.linalg.cholesky(K)

mean = np.dot(matrix(x, X), (np.linalg.inv(matrix(X, X))))*y
covar = matrix(x, x) -np.dot(np.dot(matrix(x, X), np.linalg.inv(matrix(X, X))), matrix(X, x))

print mean

a, b = np.random.multivariate_normal(mean,covar,5000).T
pl.plot(a,b,'a'); plt.axis('equal'); plt.show()






