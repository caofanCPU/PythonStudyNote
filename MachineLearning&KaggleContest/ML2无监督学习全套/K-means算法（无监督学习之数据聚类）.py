# -*- coding: utf-8 -*-
"""
Created on Sat Jan 14 14:57:28 2017

@author: CY_XYZ
"""

import os

# 导入numpy、pandas、matplotlib工具包，分别用于科学计算、数据分析、作图
import numpy as np
import pandas as pd

# import matplotlib as plt
# 使用pandas分别读取手写体数字完整的训练集、测试集
# 得到的训练集保存至变量digits_train，数据规模DataFrame( 3823, 65 )
# 列名为0,1,2,3,...,64；前64列为64维度的像素特征列，第65列为1维度的数字目标列
# 得到的测试集保存至变量digits_test，数据规模DataFrame( 1797, 65 )
# 列名为0,1,2,3,...,64；前64列为64维度的像素特征列，第65列为1维度的数字目标列
# 强调说明：测试集中第65列是手写体数据的标记，即给出了真实结果
# 如果测试集给出了真实结果，用ARI指标可以评估聚类准确性，兼顾类簇无法和分类一一对应的问题
# 如果测试集没有给出真实结果，使用轮廓系数来度量聚类结果的质量
#####对于本例，使用ARI指标评价KMeans模型
'''第一次在网上抓取，此后从本地读取
# 直接在互联网上爬取数据，注意网址的绝对正确
digits_train = pd.read_csv( 'https://archive.ics.uci.edu/'\
                            'ml/machine-learning-databases/'\
                            'optdigits/optdigits.tra', header = None )
digits_test = pd.read_csv( 'https://archive.ics.uci.edu/'\
                           'ml/machine-learning-databases/'\
                           'optdigits/optdigits.tes', header = None )
# 将手写体数字完整训练集、测试集分别保存文件至本地
digits_train.to_csv( 'G:/P_Anaconda3-4.2.0/PA-WORK/AI-ML机器学习/'\
                     'P机器学习及实践&Kaggle竞赛之路/ML2无监督学习全套/'\
                     'optdigits_train.csv',
                     index = False )
digits_test.to_csv( 'G:/P_Anaconda3-4.2.0/PA-WORK/AI-ML机器学习/'\
                     'P机器学习及实践&Kaggle竞赛之路/ML2无监督学习全套/'\
                     'optdigits_test.csv',
                     index = False )
'''

# '''
# 本地读取手写体完整训练集、测试集
inputPar = os.path.dirname(__file__)
inputTrain = inputPar + os.path.sep + 'optdigits_train.csv'
inputTest = inputPar + os.path.sep + 'optdigits_test.csv'
digits_train = pd.read_csv(inputTrain)
digits_test = pd.read_csv(inputTest)
print(inputTest)
print(digits_test.shape)
# '''

# 从训练集、测试集上都分离出64维的像素特征与1维的数字目标
X_train = digits_train[np.arange(64)]
y_train = digits_train['64']

X_test = digits_test[np.arange(64)]
y_test = digits_test['64']

# 从sklearn.cluster导入KMeans模型
from sklearn.cluster import KMeans

# 初始化KMeans模型，设置聚类中心数量为10(因为数字目标为0~9共10类)
k_means = KMeans(n_clusters=10)
# ??此处并没有涉及目标变量的训练过程，经简要确认：只需要X_train
k_means.fit(X_train)
# 逐条判断每个测试图像(即每行记录)所属的聚类中心
k_means_y_predict = k_means.predict(X_test)

# 使用ARI(Adjusted Rand Index)指标，类似于分类问题的准确性，评价KMeans模型
# 从sklearn导入度量函数库metrics中的adjusted_rand_score模块
from sklearn.metrics import adjusted_rand_score

# 使用ARI进行KMeans聚类性能评估
print('----KMeans模型的ARI指标评价----')
print(adjusted_rand_score(y_test, k_means_y_predict))

"""
运行结果：
----KMeans模型的ARI指标评价----
0.6630577949326525

----KMeans模型的ARI指标评价----
0.6630577949326525
"""
