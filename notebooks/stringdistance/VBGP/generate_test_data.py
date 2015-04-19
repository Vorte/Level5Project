import numpy as np

def generate(n):

    u = 4*np.random.rand(n,2)-2    
    
    i = np.nonzero( (u[:,0]**2 + u[:,1]**2 > 0.1) & (u[:,0]**2 + u[:,1]**2 < 0.5) )
    j = np.nonzero( (u[:,0]**2 + u[:,1]**2 > .6) & (u[:,0]**2 + u[:,1]**2 < 1) )
    X = u[np.concatenate((i[0],j[0])),:]

    t = np.zeros((i[0].shape[0],1), dtype=int)
    t = np.concatenate((t, np.ones((j[0].shape[0],1), dtype=int)))
    x = 0.1*np.random.randn(i[0].shape[0],2)
    
    k = np.nonzero(x[:,0]**2 + x[:,1]**2 < 0.1)
    X = np.concatenate((X, x[k[0],:]))
    t = np.concatenate((t, 2*np.ones((k[0].shape[0],1), dtype=int)))
    X = np.concatenate((X, np.random.randn(X.shape[0],8)), 1)

    
    return X, t
