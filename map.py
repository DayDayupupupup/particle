import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
from scipy import interpolate
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import csv

List = []
with open('sqrtData.csv','r') as csvfile:
    data = csv.DictReader(csvfile)
    for row in data:
        sqrtMag = float(row['sqrtMag'])
        list = [row['posX'],row['posY'],row['sqrtMag']]
        List.append(list)

x = []
y = []
z = []

for row in List:
    x.append(row[0])
    y.append(row[1])
    z.append(row[2])


def fun(x, y):
    for row in List:
        if int(row[0]) == x and int(row[1]) == y:
            return float(row[2])
# print(x)
# print(z)
# X-Y轴分为7*10的网格
x = np.arange(0, 8)
y = np.arange(0, 11)
X, Y = np.meshgrid(x, y)  # 20*20的网格数据

zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
Z = zs.reshape(X.shape)
fig = plt.figure(figsize=(9, 6))
ax = plt.subplot(1, 2, 1, projection='3d')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

surf = ax.plot_surface(X, Y, Z, rstride=2, cstride=2, cmap=cm.coolwarm, linewidth=0.5, antialiased=True)
plt.colorbar(surf, shrink=0.5, aspect=5)  # 标注
#
# # 二维插值
newfunc = interpolate.interp2d(X, Y, Z, kind='cubic')  # newfunc为一个函数
#
out = open('fingerPrint.csv', 'w', newline='')
# csv_writer = csv.writer(out)
label = ['posX','posY','magFinger']
# csv_writer.writerow(label)
out.write('posX,posY,magFinger\n')
# # 计算56*80的网格上的插值
xnew = np.linspace(0, 7, 56)  # x
ynew = np.linspace(0, 10, 80)  # y
fnew = newfunc(xnew, ynew)  # 仅仅是y值   100*100的值  np.shape(fnew) is 100*100
print('-------fnew-------')
#print(fnew[1][3])
print('-------------')
# for i in range(0,len(xnew)):
#     for j in range(0,len(ynew)):
#         fnew = newfunc(xnew[i], ynew[j])
#         item = [xnew[i]*0.8, ynew[j]*0.8, fnew[0]]
#         # print("%0.1f %0.1f %0.2f" % (xnew[i]*0.8, ynew[j]*0.8, fnew[0]))
#         # csv_writer.writerow("%0.1f %0.1f %0.2f" % (xnew[i]*0.8, ynew[j]*0.8, fnew[0]))
#         # out.write("%0.1f %0.1f %0.2f\n" % (xnew[i]*0.8, ynew[j]*0.8, fnew[0]))
#         out.write("%0.1f" % float(xnew[i] * 0.8) + ','+"%0.1f" % float(ynew[j] *0.8) + ','+ "%0.2f" % fnew[0]+'\n')
#         print("%0.1f" % float(xnew[i] * 0.8) + ','+"%0.1f" % float(ynew[j] *0.8) + ','+ "%0.2f" % fnew[0]+'\n')
#print(len(fingerPrint))
#print(fnew[0][0])
xnew, ynew = np.meshgrid(xnew, ynew)
ax2 = plt.subplot(1, 2, 2, projection='3d')
surf2 = ax2.plot_surface(xnew, ynew, fnew, rstride=2, cstride=2, cmap=cm.coolwarm, linewidth=0.5, antialiased=True)
ax2.set_xlabel('xnew')
ax2.set_ylabel('ynew')
ax2.set_zlabel('fnew(x, y)')
#plt.colorbar(surf2, shrink=0.5, aspect=5)  # 标注
#
plt.show()
#print(fnew)


