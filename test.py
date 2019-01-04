import matplotlib.pyplot as plt
import matplotlib
import numpy as np

zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
plt.xlabel("距离(单位：m)",fontproperties=zhfont1)
plt.ylabel("定位误差(单位：m)",fontproperties=zhfont1)
''' 4
X = [3, 5, 7, 9, 11, 13, 15, 17, 19]
Y1 = [0.5, 0.3, 0.8, 1.0, 1.2, 0.9, 0.9, 0.9, 1.1]
Y2 = [1, 2, 3, 3.2, 3.1, 2.7, 2.8, 2.9, 2.1]
'''
''' 3
X = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33]
Y1 = [2, 1.8, 1.5, 2.7, 1.8, 1.3, 1.1, 1, 2.5, 1.6, 1, 0.9, 1.3, 0.8, 0.9, 1]
Y2 = [0.4, 1.6, 2.8, 4.5, 3.6, 3.1, 3.8, 2.7, 2.5, 2.9, 4.5, 4, 4.5, 4.4, 3.5, 4]
'''
''' 3
X = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29, 31, 33]
Y1 = [2, 1.8, 1.3, 2.7, 1.8, 1.3, 1.1, 1, 2.3, 1.5, 1, 0.9, 1.2, 0.8, 0.9, 1]
Y2 = [1.6, 2.8, 3.9, 4.2, 4.5, 4, 3.5, 3.7, 4.8, 3.9, 4.2, 4.8, 5.5, 4.2, 5.1, 5.2]
'''

X = [3, 5, 7, 9, 11, 13, 15, 17, 19]
Y1 = [0.5, 0.3, 0.8, 1.0, 1.1, 0.9, 0.9, 0.8, 1]
Y2 = [1, 2, 2.8, 3, 2.6, 2.7, 2, 2.5, 3.5]

plt.plot(X, Y1, label='LmsLoc', color='red', marker='.')
plt.plot(X, Y2, label='SmartPDR', color='gree', marker='*')
for i in range(len(X)):
    plt.scatter(X[i], Y1[i], marker='>')
    plt.scatter(X[i], Y2[i], marker='.')
plt.xlim(0,int(22))
plt.ylim(0, 8)
plt.legend(loc='upper left', prop=zhfont1)
plt.savefig('图2.png', format='png')
plt.show()
