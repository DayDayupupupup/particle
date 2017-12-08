# coding=UTF-8
from scipy import stats
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
import matplotlib.cm as cm
import math
import copy
X = []#横坐标
Y = []#纵坐标
M = []
def show(X, Y, M):
    ax = plt.subplot(111, projection='3d')
    f = interpolate.interp2d(X, Y, M, kind='linear')
    #f = interpolate.interp2d(X, Y, M, kind='linear')
    #f = interpolate.interp2d(X, Y, M, kind='cubic')
    xnew = np.arange(0, max(X), 0.1)
    ynew = np.arange(0, max(Y), 0.1)
    znew = f(xnew, ynew)
    x, y = np.meshgrid(xnew, ynew)
    ax.plot_surface(x, y, znew, rstride=1, cstride=1, antialiased=True, cmap='rainbow', linewidth=0.2)
    #ax.plot_surface(x, y, znew, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0.2, antialiased=True)

    plt.show()

f = open("oldmap", "r")
count = 0
while True:
    line = f.readline()
    if line :
        line = line.strip()
        point_info = line.split()
        if point_info[-1].startswith('x'):
            temp = point_info[-1].split(";")
            x,y,z = [i.split(":")[-1] for i in temp]
            mod = math.sqrt(float(x)**2 + float(y)**2 + float(z)**2)
            M.append(mod)
        else:
            point_x, point_y = point_info[-1].strip(";").split(",")
            point_x, point_y = point_x[-1], point_y[-1]
            X.append(float(point_x))
            Y.append(float(point_y))
    else:
        break
f.close()
#show(X,Y,M)
#print M


x =[]
file = open("magnetic.csv","r")
while True:
    line = file.readline()
    if line :
        line = line.strip()
        point_info = line.split(",")
        m = pow((pow(float(point_info[0]),2)+pow(float(point_info[1]),2)+pow(float(point_info[2]),2)),0.5)
        #x.append(round(m,0))
        x.append(round(float(point_info[0]), 4))
    else:
        break
file.close()

#x = np.random.normal(1.75, 0.2, 5000)
mean = np.mean(x)
dev = np.std(x)
print mean,dev
print stats.kstest(x,'norm',(mean,dev))
fig1 = plt.figure(1)
#rects =plt.bar(x = (1,2,3,4,5,6,7,8,9,10),height = (0.49,0.32,0.15,0.38,0.25,0.36,0.31,0.37,0.19,0.28),width = 0.2,align="center",yerr=0.000001)
rects =plt.bar(x = (1,2,3,4,5,6,7,8,9,10),height = (0.011,0.00192,0.0067,0.0018,0.0025,0.0036,0.0033,0.0047,0.0039,0.0048),width = 0.2,align="center",yerr=0.000001)
plt.title('P value')
plt.show()
plt.savefig('figure/test.png')
