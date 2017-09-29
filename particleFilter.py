from numpy.random import uniform, randn, random, seed
import numpy as np
from filterpy.monte_carlo import multinomial_resample
import scipy.stats

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
    weights += 1.e-300
    weights /= sum(weights)  # 归一化


def update(particles, weights, z, R, landmarks):
    weights.fill(1.)

    distance = np.linalg.norm(particles[:, 0:2] - landmark, axis=1)
    weights *= scipy.stats.norm(distance, R).pdf(z[i])

    weights += 1.e-300       # avoid round-off to zero
    weights /= sum(weights)  # normalize


def estimate(particles, weights):
    """估计位置"""
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
    x_range, y_range = [0, 20], [0, 15]
    predict_particles(particles, 0.5, 0.01, x_range, y_range)  # 1. 预测
    update_particles(particles, weights, z, 4)  # 2. 更新
    if neff(weights) < len(particles) / 2:  # 3. 重采样
        indexes = multinomial_resample(weights)
        resample_from_index(particles, weights, indexes)
    return estimate(particles, weights)  # 4. 状态估计