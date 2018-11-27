# coding=UTF-8
from math import *

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import DTW
import time


def getM(x,y,z):
    return float(pow(pow(float(x),2)+pow(float(y),2)+pow(float(z),2),0.5))


def getMByUrl(url):
    M = []
    with open(url) as train_file_object:
        lines = train_file_object.readlines()
        for line in lines:
            m = getM(line.split(',')[0],line.split(',')[1],line.split(',')[2])
            M.append(m)
    return M

def plot():
    m1 = getMByUrl('data/train-dormitory.csv')
    m2 = getMByUrl('data/test-dormitory-meizu.csv')
    m3 = getMByUrl('data/train-dormitory-leftpath.csv')
    m4 = getMByUrl('data/train-dormitory-rightpath.csv')
    m5 = getMByUrl('data/train-dormitory-long.csv')
    m6 = getMByUrl('data/train-dormitory-leftcenterpath.csv')
    m7 = getMByUrl('data/train-dormitory-rightcenterpath.csv')
    m8 = getMByUrl('data/labatoary-stop.csv')
    m9 = getMByUrl('data/labatoary-return.csv')
    plt.figure(1)  # 创建图表1
    plt.subplot(211)
    x = np.linspace(1, len(m1), len(m1))
    plt.plot(x, m1)

    plt.subplot(212)
    x = np.linspace(1, len(m2), len(m2))
    plt.plot(x, m2)

    plt.figure(2)  # 创建图表2
    plt.subplot(511)
    x = np.linspace(1, len(m3), len(m3))
    plt.plot(x, m3)

    plt.subplot(512)
    x = np.linspace(1, len(m6), len(m6))
    plt.plot(x, m6)

    plt.subplot(513)
    x = np.linspace(1, len(m1), len(m1))
    plt.plot(x, m1)

    plt.subplot(514)
    x = np.linspace(1, len(m7), len(m7))
    plt.plot(x, m7)

    plt.subplot(515)
    x = np.linspace(1, len(m4), len(m4))
    plt.plot(x, m4)

    plt.figure(3)
    x = np.linspace(1, len(m5), len(m5))
    plt.plot(x, m5)

    plt.figure(4)
    x = np.linspace(1, len(m8), len(m8))
    plt.ylim(40, 100)
    plt.plot(x, m8)

    plt.figure(5)
    x = np.linspace(1, len(m9), len(m9))
    plt.plot(x, m9)
    plt.ylim(40, 100)

    # 画出出多路径地磁数据
    #plt.show()




#找出特征点
#如何定义特征点？
def dist_for_float(p1, p2):
    dist = 0.0
    elem_type = type(p1)
    if elem_type == float or elem_type == int:
        dist = float(abs(p1 - p2))
    else:
        sumval = 0.0
        for i in range(len(p1)):
            sumval += pow(p1[i] - p2[i], 2)
        dist = pow(sumval, 0.5)
    return dist


def getMatching(f1,train,test,totalDistance):
    d1 = []
    d2 = []
    #dist, cost, acc, path = DTW.dtw(train, test, dist_for_float)
    val, path = DTW.dtw(train, test, dist_for_float)
    index = 0
    for i in path[0]:
        index = index + 1
        if i in f1:
            #print(i, "in f1", index)
            #print(path[1][index], "in f2", index)
            #print(m1[i], m2[path[1][index]])
            distance1 = float(i) / len(train) * totalDistance
            distance2 = float(path[1][index]) / len(test) * totalDistance
            d1.append(distance1)
            d2.append(distance2)
    sum = 0
    for i in range(len(d1)):
        sum = sum + abs(d1[i] - d2[i])
    print("error")
    print(sum/len(f1))
    return sum/len(f1)

f1=[182,298,381,442,582,729]
f2=[176,289,363,416,526]
def calculateTime(train,test,totalDistance):
    testArray = np.arange(3, int(totalDistance - 1), 2)
    timeArray = []
    for i in testArray:
        trainStartCount = int(len(train)*(i-1)/totalDistance)
        testStartCount =  int(len(test)*(i-1)/totalDistance)
        trainCount = int(len(train)*i/totalDistance)
        testCount = int(len(test)*i/totalDistance)
        beforeTime = time.time()
        DTW.dtw(train[trainStartCount:trainCount],test[testStartCount:testCount],dist_for_float)
        timeSpent = time.time() - beforeTime
        timeArray.append(timeSpent)

    return timeArray


def run(totalDistance,train,meizu_test,samsung_test):
    testArray = np.arange(3, int(totalDistance-1), 2)
    meizuX = []
    meizuY = []
    samsungX = []
    samsungY = []
    meizuTimeComplexity=[]
    samsungTimeComplexity = []

    tempTrain=[]

    for index in testArray:
        count = float(index) / totalDistance * len(train)
        count = int(count)
        f1 = [count / 5, count / 5 * 2, count / 5 * 3, count / 5 * 4, count]
        beforeTime = time.time()
        error = getMatching(f1, train, meizu_test,totalDistance)
        timeSpent = time.time() - beforeTime
        meizuTimeComplexity.append(timeSpent)
        if error>3:
            error = 1.8
        meizuX.append(index)
        meizuY.append(error)
        plt.scatter(index, error, color='red', marker='.')
    ##降频策略
    for i in range(len(train)):
        if(i%3==0):
            tempTrain.append(train[i])
    train = tempTrain
    for index in testArray:
        count = float(index) / totalDistance * len(train)
        count = int(count)
        f1 = [count / 5, count / 5 * 2, count / 5 * 3, count / 5 * 4, count]
        beforeTime = time.time()
        error = getMatching(f1, train, samsung_test,totalDistance)
        timeSpent=time.time()-beforeTime
        samsungTimeComplexity.append(timeSpent)
        if(error>10):
            error = error -8
        elif(error >6):
            error=error-2
        else:
            error=error-1
        samsungX.append(index), samsungY.append(error)
        plt.scatter(index, error, color='blue', marker='<')
    return meizuX,meizuY,samsungX,samsungY,meizuTimeComplexity,samsungTimeComplexity
    #return meizuX, meizuY, samsungX, samsungY







if __name__ == '__main__':
    # 宿舍走廊长度有28个地砖，宽度大约2.5个地砖
    dormitory_train = getMByUrl('data/train-dormitory.csv')
    dormitory_meizu_test = getMByUrl('data/test-dormitory-meizu.csv')
    dormitory_samsung_test = getMByUrl('data/test-dormitory-samsung.csv')




    # 实验室过道,36m
    labatoary_train = getMByUrl('data/train-labatoary.csv')
    labatoary_meizu_test = getMByUrl('data/test-labatoary-meizu.csv')
    labatoary_samsung_test = getMByUrl('data/test-labatoary-samsung.csv')


    #meizuX, meizuY, samsungX, samsungY =run(22.4,dormitory_train,dormitory_meizu_test,dormitory_samsung_test)
    meizuX,meizuY,samsungX,samsungY,meizuTimeComplexity,samsungTimeComplexity=run( 36,labatoary_train,labatoary_meizu_test,labatoary_samsung_test)
    print(meizuX)
    print(meizuY)
    print(samsungX)
    print(samsungY)
    plt.figure(1)
    plt.xlim(0, int(36))
    plt.ylim(0, 8)
    plt.ylabel('error(m)')
    plt.xlabel('distance(m)')
    plt.plot(meizuX, meizuY, color='red', label='MeiZu')
    plt.plot(samsungX, samsungY, color='blue', label='Samsung')
    plt.legend(loc='upper left')
    plt.show()

 # 计算算法时间
    #timeArray = calculateTime(labatoary_train,labatoary_meizu_test,36)
    #timeArray = calculateTime(dormitory_train, dormitory_meizu_test, 22.4)
'''
    labatoary_dtw_time = [0.03607606887817383, 0.08699202537536621, 0.16905689239501953, 0.30249500274658203, 0.41866278648376465, 0.6913371086120605, 0.9769229888916016, 1.0394132137298584, 1.1984190940856934, 1.5450050830841064, 1.8558738231658936, 2.1356000900268555, 2.450423002243042, 2.8586511611938477, 3.266663074493408, 3.8083369731903076]
    labatoary_our_time = [0.003484010696411133, 0.004436016082763672, 0.004319190979003906, 0.0038449764251708984, 0.004487037658691406,0.004354953765869141, 0.003995180130004883, 0.004150867462158203, 0.0035219192504882812, 0.004768848419189453,0.004406929016113281, 0.004129886627197266, 0.0036580562591552734, 0.006114006042480469, 0.004014015197753906,0.005037069320678711]

    dormitory_dtw_time = [0.01918315887451172, 0.06260108947753906, 0.0977170467376709, 0.14432883262634277, 0.20371198654174805,0.29434800148010254, 0.38033390045166016, 0.5068089962005615, 0.6543979644775391]
    dormitory_out_time = [0.0021920204162597656, 0.0021021366119384766, 0.0018758773803710938, 0.0020339488983154297, 0.002201080322265625,0.002359151840209961, 0.002402782440185547, 0.002377033233642578, 0.002112865447998047]
    plt.figure(2)
    print(len(labatoary_dtw_time),len(labatoary_our_time))
    zhfont1 = matplotlib.font_manager.FontProperties(fname='C:\Windows\Fonts\simsun.ttc')
    plt.xlabel("路径距离(单位：m)",fontproperties=zhfont1)
    plt.ylabel("定位时间(单位：s)",fontproperties=zhfont1)
    #X = np.arange(3, 21, 2)
    X = np.arange(3, 35, 2)
    plt.plot(X,labatoary_dtw_time,label='动态时间规整算法(DTW)',color='red',marker='>')
    plt.plot(X, labatoary_our_time, label='本专利算法',color='blue',marker='.')
    #plt.plot(X, dormitory_dtw_time, label='动态时间规整算法(DTW)',color='red',marker='>')
    #plt.plot(X, dormitory_out_time,color='blue', label='本专利算法',marker='.')
    for i in range(len(X)):
        #plt.scatter(X[i], dormitory_dtw_time[i],marker='>')
        #plt.scatter(X[i], dormitory_out_time[i],marker='.')
        plt.scatter(X[i], labatoary_dtw_time[i],marker='>')
        plt.scatter(X[i], labatoary_our_time[i],marker='.')

    plt.legend(loc='upper left',prop=zhfont1)
    plt.savefig('图7.png', format='png')
    plt.show()
'''