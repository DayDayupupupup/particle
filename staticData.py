# coding=UTF-8
from scipy import stats
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import interpolate
import matplotlib.cm as cm
import math
import copy
import matplotlib.patches as patches
import matplotlib.path as path
import algorithm1

#同一位置不同高度的地磁数据变化
def height():
    m1 = algorithm1.getMByUrl('data/1m-magnetic.csv')
    m2 = algorithm1.getMByUrl('data/0.5m-magnetic.csv')
    m3 = algorithm1.getMByUrl('data/0.3m-magnetic.csv')
    m4 = algorithm1.getMByUrl('data/0m-magnetic.csv')
    plt.figure(1)
    plt.plot(np.arange(0,100,1),m1[100:200],color='red',marker='.',label='1m')
    plt.plot(np.arange(0,100,1),m2[100:200],color='blue',marker='>',label='0.5m')
    plt.plot(np.arange(0,100,1),m3[100:200],color='green',marker='1',label='0.3m')
    plt.plot(np.arange(0,100,1),m4[100:200],color='yellow',marker='*',label='0m')
    plt.legend(loc='upper left')
    plt.xlabel('time(s)')
    plt.ylabel("magnetic field(ut)")
    plt.ylim(48,64)
    plt.show()



m1 = algorithm1.getMByUrl('data/train-dormitory-leftpath.csv')
m2 = algorithm1.getMByUrl('data/train-dormitory-leftcenterpath.csv')
m3 = algorithm1.getMByUrl('data/train-dormitory.csv')
m4 = algorithm1.getMByUrl('data/train-dormitory-rightcenterpath.csv')
m5 = algorithm1.getMByUrl('data/train-dormitory-rightpath.csv')

def getAverage(m):
    averageArray = []
    step = len(m)/28
    for i in range(28):
        start=i*step
        array = m[start:step+start]
        average = float(sum(array)/len(array))
        averageArray.append(average)
    return averageArray


data = []
data.append(getAverage(m1))
data.append(getAverage(m2))
data.append(getAverage(m3))
data.append(getAverage(m4))
data.append(getAverage(m5))
data=np.array(data)
print data.shape
np.random.seed(19680801)
#X = np.random.rand(3,2)
X = data.T
print X

fig, ax = plt.subplots()
ax.imshow(X, interpolation='nearest')
numrows, numcols = X.shape
# def format_coord(x, y):
#     col = int(x +1)
#     row = int(y + 1)
#     if col >= 0 and col < numcols and row >= 0 and row < numrows:
#         z = X[row, col]
#         return 'x=%1.4f, y=%1.4f, z=%1.4f' % (x, y, z)
#     else:
#         return 'x=%1.4f, y=%1.4f' % (x, y)
#
# ax.format_coord = format_coord
#plt.show()
N = 5
meizu_means = []
meizu_std = []
steps = [[11, 10, 12, 11, 10],
         [15, 14, 16, 13, 15],
         [21, 20, 18, 21, 20],
         [25, 23, 22, 21, 25],
         [27, 28, 30, 27, 32]]
for i in range(len(steps)):
    data = steps[i]
    mean = 10 + i * N
    sum = 0.0
    for s in range(N):
        sum = sum + abs(data[s] - mean)
    meizu_means.append(1-float(sum/(N*mean)))
    meizu_std.append(np.std(data))

print meizu_means
ind = np.arange(N)  # the x locations for the groups
width = 0.35       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, tuple(meizu_means), width, color='r')

samsung_means = []
samsung_std = []
steps = [[8, 9, 10, 8, 10],
         [13, 14, 15, 13, 12],
         [19, 18, 18, 20, 18],
         [25, 20, 22, 23, 22],
         [27, 28, 30, 27, 27]]
for i in range(len(steps)):
    data = steps[i]
    mean = 10 + i * N
    sum = 0.0
    for s in range(N):
        sum = sum + abs(data[s] - mean)
    samsung_means.append(1-float(sum/(N*mean)))
    samsung_std.append(np.std(data))

print samsung_std
rects2 = ax.bar(ind + width, tuple(samsung_means), width, color='y')

# add some text for labels, title and axes ticks
ax.set_ylabel('precision')
ax.set_xlabel('step')
ax.set_title('Step Detection Precision')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('10', '15', '20', '25', '30'))

ax.legend((rects1[0], rects2[0]), ('MeiZu', 'Samsung'))


def autolabel(rects):
    """
    Attach a text label above each bar displaying its height
    """
    for rect in rects:
        height = rect.get_height()
        ax.text(rect.get_x() + rect.get_width()/2., 1.05*height,
                '%.2f' %float(height),
                ha='center', va='bottom')

autolabel(rects1)
autolabel(rects2)
plt.ylim(0,1.2)
plt.show()

#步长估计的图
fig = plt.figure()
ax = fig.add_subplot(111)

# Fixing random state for reproducibility
np.random.seed(19680801)

# histogram our data with numpy
data = np.random.randn(1000)
print data
n, bins = np.histogram(data, 100)
print bins


# get the corners of the rectangles for the histogram
left = np.array(bins[:-1])
right = np.array(bins[1:])
bottom = np.zeros(len(left))
top = bottom + n
nrects = len(left)

nverts = nrects*(1+3+1)
verts = np.zeros((nverts, 2))
codes = np.ones(nverts, int) * path.Path.LINETO
codes[0::5] = path.Path.MOVETO
codes[4::5] = path.Path.CLOSEPOLY
verts[0::5,0] = left
verts[0::5,1] = bottom
verts[1::5,0] = left
verts[1::5,1] = top
verts[2::5,0] = right
verts[2::5,1] = top
verts[3::5,0] = right
verts[3::5,1] = bottom

barpath = path.Path(verts, codes)
patch = patches.PathPatch(barpath, facecolor='green', edgecolor='yellow', alpha=0.5)
ax.add_patch(patch)

ax.set_xlim(left[0], right[-1])
ax.set_ylim(bottom.min(), top.max())

plt.show()
