from scipy.special import erfc
import math

def normcdfM(x, m=None, s=None):
    if m==None and s==None:
        z = x
    elif s==None:
        z = x-m
    else:
        z = (x-m)/s
        
    return 0.5*erfc(-z/math.sqrt(2))
    