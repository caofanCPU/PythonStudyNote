# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 20:57:43 2017

@author: CY_XYZ
"""

# import pandas as pd
import matplotlib.pyplot as plt
##############################################
##############################################
##############################################
# 对于测试集未给出真实结果，使用轮廓系数评价KMeans模型，这里给出一个示例
# 使用轮廓系数(Silhouette Coefficient)指标，评价KMeans模型
# 轮廓系数能兼顾聚类的凝聚度(Cohesion)和分离度(Separation)
# 轮廓系数取值范围为[-1, 1]，数值【越大】，聚类性能就越好
# '''
# 导入numpy、pandas、matplotlib工具包，分别用于科学计算、数据分析、作图
import numpy as np
# 从sklearn.cluster导入KMeans模型
from sklearn.cluster import KMeans
# 从sklearn.metrics导入suhouette_score模块
from sklearn.metrics import silhouette_score

# 给定原始数据点阵

'''
数组函数的操作
numpy.array( [ 数组1, 数组2, ... ,数组n ], dtype = None, ndim = 2 )
    参数说明：多个数组(或列表)保证长度一样，
             dtype为数据类型，常用'int32'，'float64'，complex
             ndim为多少行，数组相当于列向扩展；ndim指定多个数组组成多少行

Numpy的主要数据类型是ndarray，即多维数组。它有以下几个属性：
ndarray.ndim：数组的维数 
ndarray.shape：数组每一维的大小 
ndarray.size：数组中全部元素的数量 
ndarray.dtype：数组中元素的类型（numpy.int32, numpy.int16, and numpy.float64等） 
ndarray.itemsize：每个元素占几个字节
'''
Ox = np.array([1, 2, 3, 1, 5, 6, 5, 5, 6, 7, 8, 9, 7, 9])
Oy = np.array([1, 3, 2, 2, 8, 6, 7, 6, 7, 1, 2, 1, 1, 3])
O_X = np.array([Ox, Oy], dtype=None).transpose().reshape(len(Ox), 2)
"""                                          
经验：如何构建3维空间点阵坐标数据集？
     第1步，将每一维坐标数据构建为数组
                    X = np.array( [ ... ] )
                    Y = np.array( [ ... ] )
                    Z = np.array( [ ... ] )
     第2步，将每个维度合并起来构建新的数组
                    np.array( [ X, Y, Z ] )
     第3步，把第2步的结果转置
                    np.array( [ X, Y, Z ] ).transpose(  )
     第4步，对第3步的结果形状重组为 ( len( X ), 3 )的“矩阵数组”
                    np.array( [ X, Y, Z ] ).transpose(  ).reshape( len( X ), 3 )
把第2~4步合并起来书写，即为：
                    np.array( [ X, Y, Z ] )\
                                           .transpose(  )\
                                           .reshape( len( X ), 3 )
"""
# 分割出3*2=6个子图，并先在1号子图画出原始数据点阵的分布
plt.figure('OriginalData')
plt.subplot(3, 2, 1)
plt.xlim([0, 10])
plt.ylim([0, 10])
plt.title('OriginalData')
plt.scatter(Ox, Oy)
plt.grid(True)

# 设置聚类的类别种类数目，将原始数据分别分为2类、3类、4类、5类、8类
clusters = [2, 3, 4, 5, 8]
# 设置不同聚类点阵绘制颜色
colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'b']
# 设置不同聚类点阵绘制标识
markers = ['o', 's', 'D', 'v', '^', 'p', '*', '+']
# 变量subplot_counter为绘图子图序号标识器，已绘制子图1，因而初始化为1
subplot_counter = 1
# 变量sc_scores为KMeans系数(即KMeans模型的评分)，初始化为空列表
sc_scores_result = []

# 使用循环绘制子图2聚为2类，子图聚为3类，子图聚为4类，子图5聚为5类，子图6聚为8类
# 循环变量t在clusters中迭代取值，依次取为2、3、4、5、8
for t in clusters:
    # 子图序号标识依次加1，依次为2，3，4，5，6
    subplot_counter += 1
    # 依次在子图2，3，4，5，6绘制图形
    plt.subplot(3, 2, subplot_counter)
    # 依次以t的值作为分类数目，即依次为2，3，4，5，8类训练KMeans模型
    k_means_model = KMeans(n_clusters=t).fit(O_X)

    # 每次t循环内，需要绘制出KMeans模型聚类的结果，用另外一个循环绘制不同类别的点阵
    # 此处循环参数意义不懂，需要深入学习了解
    # 猜想：对于每一次KMeans模型训练结果，用不同的颜色、标识绘制不同聚类的点阵
    for i, k in enumerate(k_means_model.labels_):
        plt.plot(Ox[i], Oy[i], color=colors[k], marker=markers[k], ls='None')
    # 每次绘制完点阵后，标识坐标轴及子图的标题    
    plt.xlim([0, 10])
    plt.ylim([0, 10])
    plt.grid(True)
    # 每次都计算KMeans模型的轮廓系数
    sc_scores = silhouette_score(O_X, k_means_model.labels_, metric='euclidean')
    sc_scores_result.append(sc_scores)
    # 绘制轮廓系数与不同类簇数量的直观显示图
    plt.title('K = %s, SilhouetteCoefficient = %0.03f' % (t, sc_scores))

# 绘制轮廓系数与不同类簇数量的关系曲线
plt.figure('轮廓曲线与不同类簇数量的关系曲线')
plt.plot(clusters, sc_scores_result, 's-m')
plt.grid(True)
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Coefficient Score')  # 如果是在Ipython的Console里绘图，就需要plt.show(  )函数显示图形
# 如果设置了新开窗口显示绘图，就不需要plt.show(  )方法了
# plt.show(  )
