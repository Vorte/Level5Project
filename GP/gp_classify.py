import numpy as np
from matplotlib import pyplot as pl

def vector(x, X):
    vector = np.zeros((len(X), 1))
    for i in range(len(vector)):
        vector[i] = cov(x, X[i])
        
    return vector

def matrix (X1, X2): # covariance matrix    
    matr = np.zeros((len(X1), len(X2)))
    
    for i in range(len(X1)):
        for l in range(len(X2)):
            matr[i][l] = cov(X1[i], X2[l])
    return matr
        
def cov(x1, x2): # covariance function
    #if (is_numeric(x1)):
    #    return math.exp ((abs(x1-x2)*abs(x1-x2))/(-2.0))
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
        w[i] = -np.log(1+np.exp(-y[i]*f[i][0]))
        
    return w

X = np.array([[1.1, 3.2],[2.2, 4.5],[2.7, 5.2],[3.5, 0.7],[4.2, 1.9],[5.6, 2.3]]).reshape(6, 2)
x = np.array([4.7, 1.8]).reshape(2,1)
y = np.array([-1,-1,-1,1,1,1]).reshape(6, 1)
f = np.zeros(6).reshape(6,1)
#k = matrix(x, X)
K = matrix(X,X)

magnitude = 100
while magnitude > 0.0:    
    W = -second_derivative(f)
    L = np.linalg.cholesky(np.identity(len(X))+W**(1/2)*K*W**(1/2))
    #print L
    #print np.identity(6).reshape(6,6,1)

    b = W*f + first_derivative(y, f)    
    a = b - np.linalg.solve(W**(1/2)*(L.T),  np.linalg.solve(L, np.dot(W**(1/2)*K, b)))
    
    f_old = f
    f = np.dot(K, a)
    #print W.shape, L.shape, b.shape, a.shape, f.shape
    #break
    magnitude = np.linalg.norm(f-f_old)
    #print magnitude
    #break

#print f
#print (a)*f/-2.0+log_probability(y, f) 
W = -second_derivative(f)
L = np.linalg.cholesky(np.identity(len(X))+W**(1/2)*K*W**(1/2))
k = vector(x, X)

mean = np.dot((k.T), first_derivative(y, f))
v = np.linalg.solve(L, W**(1/2)*k)
V = cov(x, x) - np.dot(v.T, v)

print mean, V
likelihoods = []
for i in range(100):
    likelihoods.append(likelihood(np.random.normal(mean, V)))

print (np.sum(likelihoods)/100.0)


#axis = np.take(X, [0, 1,2,3,4,5], axis=1)
axis1 = X[:,0]
axis2 = X[:,1]

line, = pl.plot(axis1,axis2, 'bo', linewidth=1)
line2, = pl.plot(x[0], x[1], 'ro', linewidth=1)
pl.show()












