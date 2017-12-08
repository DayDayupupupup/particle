import matplotlib.pyplot as mpl
import numpy as np
import scipy as sp

def getdata():
    a = np.zeros(10,np.double)
    b = np.zeros(10,np.double)
    for i in range(len(a)):
        a[i] = np.random.uniform(-255,255)
        b[i] = np.random.uniform(-255,255)
        print '(%f,%f)' %(a[i],b[i]),
    print
    return a,b

a = np.array([0,10,20,30,40,50,60,70,80,90,100,110,120],np.float32)
b = np.array([5,1,7.5,3,4.5,8.8,15.5,6.5,-5,-10,-2,4.5,7],np.float32)
def Larange(x,y,a):
    ans = 0.0
    for i in range(len(y)):
        t = y[i]
        for j in range(len(y)):
            if i != j:
                t *= (a-x[j])/(x[i]-x[j])
        ans += t
    return ans

y2 = Larange(a,b,65)
print y2
mpl.plot(a,b,'*')
mpl.scatter(65,y2,c='red')
mpl.show()
