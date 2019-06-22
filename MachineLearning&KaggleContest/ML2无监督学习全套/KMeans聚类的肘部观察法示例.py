# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 20:59:15 2017

@author: CY_XYZ
"""

# 导入matplotlib.pyplot绘图工具包
import matplotlib.pyplot as plt
# 用肘部观察法粗略估计KMeans模型相对合理的类簇个数
# 导入numpy科学计算工具包
import numpy as np
# 从scipy.spatial.distance导入cdist
from scipy.spatial.distance import cdist
# 从sklearn.cluster导入KMeans模块
from sklearn.cluster import KMeans

# 使用均匀分布函数随机产生三个簇，每个簇周围50个数据样本
# 产生的样本数据：二维坐标，数组格式：2行50列
# 均匀分布函数用法：
#            np.random.uniform( low = 0.0, high = 1.0, size = [ 3, 10 ] )
# 参数说明：low限定最小值，high限定最大值
#          size为数据的规模，size = [ 3, 10 ]表示10个样本，每个样本有3个维度的数据
cluster_I = np.random.uniform(low=0.5, high=1.0, size=(2, 50))
cluster_II = np.random.uniform(low=5.5, high=6.0, size=(2, 50))
cluster_III = np.random.uniform(low=3.0, high=4.0, size=(2, 50))
# 绘制50*3个数据点阵的分布图象
# 将3类点阵数据合并顺序连接到一起，并转置为150行2列的数组
# np.hstack(  )方法只接受1个参数，因而用[ ...]或者( ... )
X = np.hstack([cluster_I, cluster_II, cluster_III]).T
plt.figure('Original Data')
plt.scatter(X[:, 0], X[:, 1])
plt.xlabel('x_label')
plt.ylabel('y_label')
plt.title('Original Data')
plt.grid(True)
# plt.show(  )

# 测试聚类中心数量分别为1，2，3，4，5，6，7，8，9共9种情况下的聚类质量
K = range(1, 10)
meandistortions = []

for i in K:
    k_means = KMeans(n_clusters=i)
    k_means.fit(X)
    meandistortions.append(sum(np.min(cdist(X, k_means.cluster_centers_, 'euclidean'), axis=1)) / X.shape[0])
plt.figure('Selecting K with ElbowMethod')
plt.plot(K, meandistortions, 'o-m')
plt.grid(True)
plt.xlabel('value of K')
plt.ylabel('Average Dispersion')
plt.title('Selecting K with the ElbowMethod')
# plt.show()

"""


"""
