import numpy as np
from tmean import *
from generate_test_data import *
from libxml2 import predefinedEntity

def dist(X,Y,Z):
    
    nx = X.shape[0]
    ny = Y.shape[0]

    distance= (np.sum(np.dot((X**2), Z), 1).reshape(nx,1)*np.ones(ny)+
              np.ones((nx,1))*(np.sum(np.dot((Y**2), Z), 1).conj().T) 
               - 2*(np.dot(np.dot(X, Z), Y.conj().T))) 
           
    return distance

class VBGP:
    def __init__(self):
        self.X = None
        self.t = None
        self.Theta = None
        self.iK = None
        self.Y = None
        self.C = None
        self.nos_samps_tg = 1000
        
    def fit(self, X, t, theta=None, nos_its = 10, thresh=0.1):
        
        self.X = X
        self.t = t
        if theta==None:
            theta = 0.01*np.ones(X.shape[1])

        self.Theta = np.diag(theta)
        self.C = np.unique(self.t).size
        N = self.X.shape[0]
        self.Y = np.random.randn(N, self.C)
        
        M = np.random.rand(N, self.C)
        In = np.eye(N)
        diff = 1e100
        its = 0
    
        K = np.exp(-dist(self.X,self.X,self.Theta)) + np.eye(N)*1e-10
        
        self.iK = np.linalg.inv(K+In)
        Ki = np.dot(K, self.iK)
        
        LOWER_BOUND = [-1e-3]
        constant = (-0.5*self.C*np.trace(np.dot(K, self.iK))
                    -0.5*self.C*np.trace(self.iK)
                    -0.5*self.C*safelogsc(abs(np.linalg.det(K)))
                    +0.5*self.C*safelogsc(abs(np.linalg.det(np.dot(K, self.iK))))
                    -0.5*N*self.C*safelogsc(2*math.pi)
                    +0.5*N*self.C + 0.5*N*safelogsc(2*math.pi))
        
        while its<nos_its and diff>thresh:
            its += 1
            
            for k in range(0, self.C):
                M[:,[k]] = np.dot(Ki, self.Y[:,[k]])
      
            lower_bound = 0    
            for n in range(0, N):
                a, z = tmean(M[[n],].conj().T, self.t[n], self.nos_samps_tg)
                self.Y[n] = a
                lower_bound = lower_bound + safelogsc(z)
            
            if its == 2:
                LOWER_BOUND[0] = LOWER_BOUND[1]
    
            lower_bound = (lower_bound + constant -0.5*np.sum(np.diag(np.dot(np.dot(M.conj().T, self.iK), M))))
            LOWER_BOUND.append(lower_bound) 
            
            diff = abs(100*(lower_bound - LOWER_BOUND[-2])/LOWER_BOUND[-2])
            
    def optimize(self, theta, cv=10, nos_its = 10, thresh=0.1):
        indices = np.random.permutation(self.X.shape[0])
        
        fold_size = self.X.shape[0]/10
        score = []
        for i in range(cv):
            test_index =  indices[i*fold_size:(i+1)*fold_size]
            train_index = np.setdiff1d(indices, test_index, True)
            X_train, X_test = self.X[train_index], self.X[test_index]
            t_train, t_test = self.t[train_index], self.t[test_index]
            
            fold_gp = VBGP()
            fold_gp.fit(X_train, t_train, theta, nos_its, thresh)
            prob = fold_gp.predict(X_test)
            
            pred = np.argmax(prob, axis=1)
            pred = pred.reshape(pred.shape[0])
            t_test = t_test.reshape(t_test.shape[0])
            
            print pred
            print t_test
            diff = pred - t_test
            print diff
            no_points = 1.0*t_t.shape[0]
            error = np.nonzero(diff)[0].shape[0]/no_points
            score.append(error)
            
        return score            
                
            
    def predict(self, X_test):
        n_test = X_test.shape[0]
        K_test = np.exp(-dist(self.X,X_test,self.Theta))
        K_test_self = np.exp(-dist(X_test,X_test,self.Theta))
    
        S = (np.diag(K_test_self) - np.diag(np.dot(np.dot(K_test.conj().T, self.iK), K_test))).conj().T
    
        res = (np.dot(np.dot(self.Y.conj().T, self.iK), K_test)).conj().T
        P_test = np.ones((n_test, self.C))
        u = np.random.randn(self.nos_samps_tg, 1)
    
        for n in range(0, n_test):
            for i in range(0, self.C):
                pp = np.ones((self.nos_samps_tg, 1))
                for j in range(0, self.C):
                    if j!=i:
                        pp = pp*safenormcdf(u+(res[n,i]-res[n,j])/(math.sqrt(1+S[n])))
    
                P_test[n,i] = np.mean(pp)
            P_test[n] = P_test[n]/np.sum(P_test[n])
    
        return P_test
        


def VarMultProbRegGP(X, t, X_test=0, t_test=0, theta=0, theta_estimate=0, 
                     nos_its=0, kernel_Type=0, poly_kernel_power=0, thresh=0):
    
    SMALL_NOS = 1e-10
    nos_samps_tg = 1000

    C = np.unique(t).size
    N, D = X.shape
    Y = np.random.randn(N, C)
    M = np.random.rand(N, C)
    Theta = np.diag(theta)
    In = np.eye(N)

    diff = 1e100
    its = 0

    K = np.exp(-dist(X,X,Theta)) + np.eye(N)*SMALL_NOS; # TODO: implement create kernel function
    
    iK = np.linalg.inv(K+In)
    Ki = np.dot(K, iK)
    
    LOWER_BOUND = [-1e-3]
    constant = (-0.5*C*np.trace(np.dot(K, iK))
                -0.5*C*np.trace(iK)
                -0.5*C*safelogsc(abs(np.linalg.det(K)))
                +0.5*C*safelogsc(abs(np.linalg.det(np.dot(K, iK))))
                -0.5*N*C*safelogsc(2*math.pi)
                +0.5*N*C + 0.5*N*safelogsc(2*math.pi))
    
    while its<nos_its and diff>thresh:
        its += 1
        
        for k in range(0, C):
            M[:,[k]] = np.dot(Ki, Y[:,[k]])
  
        lower_bound = 0    
        for n in range(0, N):
            a, z = tmean(M[[n],].conj().T, t[n], nos_samps_tg)
            Y[n] = a
            lower_bound = lower_bound + safelogsc(z)
        
        if its == 2:
            LOWER_BOUND[0] = LOWER_BOUND[1]

        lower_bound = (lower_bound + constant -0.5*np.sum(np.diag(np.dot(np.dot(M.conj().T, iK), M))))
        LOWER_BOUND.append(lower_bound) 
        
        diff = abs(100*(lower_bound - LOWER_BOUND[-2])/LOWER_BOUND[-2])
        
    ####### predict ########

    n_test = X_test.shape[0]
    K_test = np.exp(-dist(X,X_test,Theta))
    K_test_self = np.exp(-dist(X_test,X_test,Theta))

    S = (np.diag(K_test_self) - np.diag(np.dot(np.dot(K_test.conj().T, iK), K_test))).conj().T

    res = (np.dot(np.dot(Y.conj().T, iK), K_test)).conj().T
    P_test = np.ones((n_test, C))
    u = np.random.randn(nos_samps_tg, 1)

    for n in range(0, n_test):
        for i in range(0, C):
            pp = np.ones((nos_samps_tg, 1))
            for j in range(0, C):
                if j!=i:
                    pp = pp*safenormcdf(u+(res[n,i]-res[n,j])/(math.sqrt(1+S[n])))

            P_test[n,i] = np.mean(pp)
        P_test[n] = P_test[n]/np.sum(P_test[n])

    return P_test
        

# X 2d, t 1d
# theta.size == X.shape[1]
# X_test.shape[1]==X.shape[1]
X, t = generate(500)
X_t, t_t = generate(5000)
theta = np.array([1.1348e+01, 8.8135e+00, 1.7372e-03, 2.2966e-03, 1.7476e-03, 
                   1.6263e-03, 1.8975e-03, 1.2296e-03,  2.4512e-03, 1.6161e-03])

#theta = np.array([0.01 for x in range(10)])
#prob = VarMultProbRegGP(X, t, theta=theta, nos_its=50, thresh = 0.1, X_test=X_t)    
vbgp = VBGP()
vbgp.fit(X, t, theta, nos_its = 50, thresh=0.1)
print vbgp.optimize(theta, cv =3)
prob = vbgp.predict(X_t)
print prob

pred = np.argmax(prob, axis=1)
pred = pred.reshape(pred.shape[0])
t_t = t_t.reshape(t_t.shape[0])

error = pred - t_t
print t_t.shape
print np.nonzero(error)[0].shape


