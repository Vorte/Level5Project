import normcdfM, normpdfM, math
import numpy as np

def tmean(m, index_max, n_samps):
    
    K = m.shape[0]
    u = np.random.randn(n_samps,1)

    t = m[index_max]*np.ones((K, 1)) - m 
    tr = t
    t = np.delete(t, index_max, 0)
    
    s = np.tile(u, (1, K-1)) + np.tile(t,(1, n_samps)).conj().T
    z = np.mean(np.prod(safenormcdf(s.conj().T), 0))

    tm = np.zeros(m.shape[0])
    for r in range(0, K):
        sr = np.tile(u, (1, K)) + np.tile(tr, (1, n_samps)).conj().T
        sr.take([r, index_max], axis=1)
        
        nr = np.mean(safenormpdf(u.conj().T + m[index_max] 
                                 - m[r])*np.prod(safenormcdf(sr.conj().T), 0))
        
        if r == index_max:
            tm[r] = 0.0
        else:
            tm[r] = m[r] - nr/z
            
    tm[index_max] = np.sum(m, axis=0) - np.sum(tm, axis=0)
    tm = tm.conj().T                    
            
    return tm, z
                                                               

def safenormcdf(x):
    thresh=-10;
    x[np.nonzero(x<thresh)] = thresh
    return normcdfM.normcdfM(x)

def safenormpdf(x):
    x=x[0]
    thresh=35;
    x[np.nonzero(x<-thresh)] = -thresh
    x[np.nonzero(x>thresh)] = thresh
    return normpdfM.normpdfM(x)

def safelogsc(x):
    if x<1e-300:
        x = 1e-200
    elif x>1e300:
        x = 1e300
    return math.log(x)

# def safelog(x):
#     x[np.nonzero(x<1e-300)] = 1e-200
#     x[np.nonzero(x>1e300)] = 1e300
#     return math.log(x)
        
        
        
        
        