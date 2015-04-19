import math

def normpdfM(x, m=0, s=1):    
    o = 1.0/math.sqrt(2.0*math.pi*(s**2))
    
    return o*math.exp((-((x-m)**2)/(2*(s**2)))[0])
    