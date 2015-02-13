import numpy as np
import math, random
from matplotlib import pyplot as pl

RANGE = 100         # the range of the function (0 to RANGE)
N = 100             # number of training points
n = 50              # number of test points

def matrix (X1, X2): # covariance matrix
    matr = np.zeros((len(X1), len(X2)))
   
    for i in range(len(X1)):
        for l in range(len(X2)):
            matr[i][l] = cov(X1[i], X2[l])
    return np.array(matr) 
        
def cov(x1, x2): # covariance function
	return math.exp ((abs(x1-x2)*abs(x1-x2))/(-2.0))

def func(x):
    #The function to predict.
    return x * np.sin(x)

randList = []
for i in range(N):
    randList.append(random.random()*RANGE)
       
X = np.sort(np.array(randList)).reshape(N,1)
#X = np.array([1,1.5,3,4,6.5]).reshape(5,1) # training points
y = func(X) # targets

#x = np.array([2,  7]).reshape(2,1) # test point
randList = []
for i in range(n):
    randList.append(random.random()*RANGE)
    
x = np.sort(np.array(randList)).reshape(n,1)

L = np.linalg.cholesky(matrix(X, X))
alpha = np.linalg.solve(np.transpose(L), np.linalg.solve(L, y))

predictions = []

for i in x:
    k = matrix(i, X)
    f = np.dot(k, alpha) # mean
    predictions.append(f)
    
predictions = np.array(predictions).reshape(n,1)

line, = pl.plot(X, y, 'b-', linewidth=1)
line2, = pl.plot(x, predictions, 'ro', linewidth=1)

pl.legend([line, line2],['Training data','Test points'], loc=2)
pl.show()


