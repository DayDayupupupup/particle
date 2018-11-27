# coding=UTF-8
from numpy.random import uniform, randn, random, seed
import numpy as np
#from filterpy.monte_carlo import multinomial_resample
import scipy.stats
import csv
import math

seed(7)

N = 2000


def create_uniform_particles(x_range, y_range, h_range, N):
    """均匀分布，这里的粒子状态设置为（坐标x，坐标y，运动方向）"""
    particles = np.empty((N, 3))
    particles[:, 0] = uniform(x_range[0], x_range[1], N)
    particles[:, 1] = uniform(y_range[0], y_range[1], N)
    particles[:, 2] = uniform(h_range[0], h_range[1], N)
    particles[:, 2] %= 2 * np.pi
    return particles


def create_gaussian_particles(mean, std, N):
    """高斯分布，设置均值和标准差"""
    particles = np.empty((N, 3))
    particles[:, 0] = mean[0] + (randn(N) * std[0])
    particles[:, 1] = mean[1] + (randn(N) * std[1])
    particles[:, 2] = mean[2] + (randn(N) * std[2])
    particles[:, 2] %= 2 * np.pi
    return particles


def predict(particles, h_change, g_noise, length):
    """ move according to control input u (heading change, velocity)
    with gaussian noise g_noise(heading noise, length noise)`"""

    N = len(particles)
    # update heading 更新朝向
    particles[:, 2] += h_change[0] + (randn(N) * g_noise[0])
    particles[:, 2] %= 2 * np.pi

    # 更新粒子坐标
    dist = length + (randn(N) * g_noise[1])
    particles[:, 0] += np.cos(particles[:, 2]) * dist
    particles[:, 1] += np.sin(particles[:, 2]) * dist


def update_particles(particles, weights, obv, d_std):
    """粒子权重更新，根据观测结果中得到的位置pdf信息来更新权重，这里简单地假设是真实位置到观测位置的距离为高斯分布"""
    # weights.fill(1.)
    distances = np.linalg.norm(particles[:, 0:2] - obv, axis=1)
    weights *= scipy.stats.norm(0, d_std).pdf(distances)
    weights += 1.e-300       # avoid round-off to zero
    weights /= sum(weights)  # 归一化


def estimate(particles, weights):
    """估计位置"""
    print(np.average(particles, weights=weights, axis=0))
    return np.average(particles, weights=weights, axis=0)



def neff(weights):
    """用来判断当前要不要进行重采样"""
    return 1. / np.sum(np.square(weights))


def resample_from_index(particles, weights, indexes):
    """根据指定的样本进行重采样"""
    particles[:] = particles[indexes]
    weights[:] = weights[indexes]
    weights /= np.sum(weights)


def run_pf(particles, weights, z, x_range, y_range):
    """迭代一次粒子滤波，返回状态估计"""
    x_range, y_range = (0, 7), (0, 10)
    predict(particles, 0.0, 0.01, 0.8)  # 1. 预测
    update_particles(particles, weights, z, 4)  # 2. 更新
    if neff(weights) < len(particles) / 2:  # 3. 重采样
        indexes = multinomial_resample(weights)
        resample_from_index(particles, weights, indexes)
    return estimate(particles, weights)  # 4. 状态估计

particles = create_uniform_particles((0,1), (0,1), (0,np.pi*2), 1000)
weights = np.array([.25]*1000)
estimate(particles, weights)

fingerData = []
fingerMap = [[0 for i in range(81)] for i in range(57)]
onlineData = []

x=[]
y=[]
z=[]
rowindex = 0
colindex = 0
with open('fingerPrint.csv','r') as csvfile:
    data = csv.DictReader(csvfile)
    for row in data:
        list = [row['posX'], row['posY'], row['magFinger']]
        print(list)
        fingerMap[rowindex][colindex] = float(row['magFinger'])
        colindex = colindex + 1
        if colindex == 80 :
            colindex = 0
            rowindex = rowindex + 1

        fingerData.append(list)
print("----------map------------")
print(fingerMap[0][80])
print("-------------------------")
for row in fingerData:
    x.append(row[0])
    y.append(row[1])
    z.append(row[2])
X, Y = np.meshgrid(x, y)

def fun(x, y):
    for row in fingerData:
        if (row[0]) == x and (row[1]) == y:
            print(row[2])
            return row[2]

#zs = np.array([fun(x,y) for x,y in zip(np.ravel(X), np.ravel(Y))])
# Z = zs.reshape(X.shape)


with open('online.csv','r') as csvfile:
    data = csv.DictReader(csvfile)
    for row in data:
        squareMagX = math.pow(float(row['magX']), 2)
        squareMagY = math.pow(float(row['magY']), 2)
        squareMagZ = math.pow(float(row['magZ']), 2)
        sqrtMag = round(math.sqrt(squareMagX + squareMagY + squareMagZ),2)
        x= float(row['posX'])*0.8
        y= round(float(row['posY'])*0.8,1)
        item = [x, y, sqrtMag]
        onlineData.append(item)

print(onlineData)
def weight(particles, obv):
    x = particles[:, 0]
    y = particles[:, 1]
    x = float(("%.1f" % x))
    y = float(("%.1f" % y))
    distance = obv - fingerMap[int(x/0.1)][int(y/0.1)]


