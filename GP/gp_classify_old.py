import numpy as np
from scipy.optimize import newton
import random
from matplotlib import pyplot as pl

RANGE = 10         # the range of the function (0 to RANGE)
N = 10             # number of training points
n = 1             # number of test points

def matrix (X1, X2): # covariance matrix
    matr = np.zeros((len(X1), len(X2)))
    for i in range(len(X1)):
        for l in range(len(X2)):
            matr[i][l] = cov(X1[i], X2[l])
    return np.array(matr) 
        
def cov(x1, x2): # covariance function
    #return math.exp ((abs(x1-x2)*abs(x1-x2))/(-2.0))
    return np.exp ((np.sqrt((x1[0]-x2[0])**2 + (x1[1]-x2[1])**2)**2)/(-2.0)) 

def likelihood(f):    
    return 1.0/(1.0+np.exp(-1.0*f))
    #return -np.log(1+ np.exp(-y*f));

def second_derivative(f):
    w = np.zeros_like(f)
    for i in range(len(f)):
        p = likelihood(f[i])
        w[i]= -p*(1.0-p)
    return w
    
def first_derivative(y, f):
    w = np.zeros_like(f)
    t = (y+1)/2
    
    for i in range(len(f)):
        p = likelihood(f[i])
        w[i] = t[i] - p
        
    return w
    
def log_probability(y, f):
    w = np.zeros_like(f)
    
    for i in range(len(f)):
        w[i] = -np.log(1+np.exp(-y[i]*f[i]))
        
    return w

randList = []
for i in range(N):
    randList.append(random.random()*RANGE)

X = np.array([[1.1, 3.2],[2.2, 4.5],[2.7, 5.2],[3.5, 0.7],[4.2, 1.9],[5.6, 2.3]])
y = np.array([-1,-1,-1,1,1,1])
f = np.zeros(6)
K = matrix(X,X)

magnitude = 100
while magnitude >0:    
    W = -second_derivative(f)
    L = np.linalg.cholesky(np.identity(len(X))+np.dot(W**(1/2), K)*W**(1/2))
    
    b = W*f + first_derivative(y, f)
    a = b - np.linalg.solve (W**(1/2)*(L.T),  np.linalg.solve(L, np.dot(W**(1/2), K)*b))
    
    f_old = f
    f = np.dot(K, a)
    magnitude = np.linalg.norm(f-f_old)

print (a.T)*f/-2.0+log_probability(y, f) 













