# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 16:53:13 2017

@author: CY_XYZ
"""

import os

# 导入pandas工具包，并更名为pd
import pandas as pd

# 调用pandas工具包的read_csv函数/模块，传入训练文件地址参数，获得返回数据并保存至变量df_train
# 输出本脚本文件的绝对路径
# print(os.path.dirname(__file__))
inputPara = os.path.dirname(__file__) + os.path.sep
inputTrain = inputPara + 'breast-cancer-train.csv'
# print(inputTrain)
'''
df_train = pd.read_csv( 'G:/P_Anaconda3/PA-WORK/AI-ML机器学习/'\
                        'P机器学习及实践&Kaggle竞赛之路/ML1全套/'\
                        'breast-cancer-train.csv' )
'''
df_train = pd.read_csv(inputTrain)

# 同理，读取测试数据文件并保存至变量df_test
inputTest = inputPara + 'breast-cancer-test.csv'
df_test = pd.read_csv(inputTest)
'''
df_test = pd.read_csv( 'G:/P_Anaconda3/PA-WORK/AI-ML机器学习/'\
                       'P机器学习及实践&Kaggle竞赛之路/ML1全套/'\
                       'breast-cancer-test.csv' )
'''
# 选取''与''作为特征，构建测试集的正负(良性恶性)分类样本
###正则表达式，以'Type'为逻辑判断条件，判断后的结果只提取'Clump Thickness'、'Cell Size'两列
df_test_positive = df_test.loc[df_test['Type'] == 0][['Clump Thickness', 'Cell Size']]
df_test_negative = df_test.loc[df_test['Type'] == 1][['Clump Thickness', 'Cell Size']]
############


# 导入matplotlib工具包中的pyplot并简化命名plt
import matplotlib.pyplot as plt

# 新建画图窗口figure('图1-1')
plt.figure('图1-1')
# 绘制图1-1中良性肿瘤样本点，标记为红色的o，并且保存绘图句柄Clump_Thickness
Clump_Thickness = plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='o', s=200, c='red')
# 绘制图1-1中恶性肿瘤样本点，标记为黑色的x，并且保存绘图句柄Cell_Size
Cell_Size = plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='x', s=150, c='black')
plt.rcParams['font.sans-serif'] = ['SimHei']  # 正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像中负号'-'变为□■的问题
plt.xlabel(u'肿块厚度')
plt.ylabel(u'细胞尺寸')
plt.title(u'图1-1')
plt.legend()

################
# 使用绘图显示中文的办法，导入
from matplotlib.font_manager import FontProperties

# 调用window系统自带字体，这需要确保字体文件存在且在指定路径
font = FontProperties(fname=r"C:\\WINDOWS\\Fonts\\simsun.ttc", size=14)
##显示中文的格式u'中文'，保证在文件第一行：  # -*- coding: utf-8 -*-
#################
'''
# 绘制x、y轴的说明
plt.xlabel( u'肿块厚度', fontproperties = font )
plt.ylabel( u'细胞尺寸', fontproperties = font )
# 显示图1-1
plt.title( u'图1-1', fontproperties = font )
####图例标注不支持中文，除非修改配置文件
# plt.legend( [Clump_Thickness, Cell_Size], ["Clump Thickness", "Cell_Size"] )
# plt.legend()    #使用空的参数会有问题
# plt.legend( loc = 'best' )    #在最合适的位置放图例
plt.show(  )
###############
'''

# 导入numpy工具包，重命名为np
import numpy as np

# 利用numpy中的random函数随机采样直线的截距和系数
intercept = np.random.random([1])
coef = np.random.random([2])
lx = np.arange(0, 12)
# 分类方程(二维平面)：(lx * coef[0]) + (ly * coef[1]) + intercept = 0
# 映射到二维平面后的直线解析式
ly = (-intercept - lx * coef[0]) / coef[1]

# 绘制图1-2
plt.figure('图1-2')
# 绘制一条随机直线
plt.plot(lx, ly, c='magenta')
# 绘制图1-2中良性肿瘤样本点，标记为红色的o，并且保存绘图句柄Clump_Thickness
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='o', s=200, c='red')
# 绘制图1-2中恶性肿瘤样本点，标记为黑色的x，并且保存绘图句柄Cell_Size
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='x', s=150, c='black')
plt.xlabel(u'肿块厚度', fontproperties=font)
plt.ylabel(u'细胞尺寸', fontproperties=font)
plt.title(u'图1-2', fontproperties=font)
plt.show()

# 导入sklearn的逻辑斯蒂回归分类器
from sklearn.linear_model import LogisticRegression

lr = LogisticRegression()

# 使用前10条训练样本学习直线的系数和截距，学习(求解)结果保存在对象lr中
lr.fit(df_train[['Clump Thickness', 'Cell Size']][:10], df_train['Type'][:10])
print('Testing accuracy (10 train_set samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type']))

Train_10_intercept = lr.intercept_
Train_10_coef = lr.coef_[0, :]
Train_10_lx = np.arange(0, 12)  # 与前述lx一样
Train_10_ly = (-Train_10_intercept - Train_10_lx * Train_10_coef[0]) / Train_10_coef[1]
# 绘制图1-3
plt.figure('图1-3')
plt.plot(Train_10_lx, Train_10_ly, c='green')
# 绘制图1-3中良性肿瘤样本点，标记为红色的o，并且保存绘图句柄Clump_Thickness
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='o', s=200, c='red')
# 绘制图1-3中恶性肿瘤样本点，标记为黑色的x，并且保存绘图句柄Cell_Size
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='x', s=150, c='black')
plt.xlabel(u'肿块厚度', fontproperties=font)
plt.ylabel(u'细胞尺寸', fontproperties=font)
plt.title(u'图1-3', fontproperties=font)
plt.show()

# 使用全部训练样本学习直线的系数和截距，学习(求解)结果保存在对象lr中
lr.fit(df_train[['Clump Thickness', 'Cell Size']][:], df_train['Type'][:])
# 上一句代码也可写为
# lr.fit( df_train[['Clump Thickness', 'Cell Size']], df_train['Type'] )

print('Testing accuracy (all train_set samples):', lr.score(df_test[['Clump Thickness', 'Cell Size']], df_test['Type']))

Train_all_intercept = lr.intercept_
Train_all_coef = lr.coef_[0, :]
Train_all_lx = np.arange(0, 12)  # 与前述lx一样
Train_all_ly = (-Train_all_intercept - Train_all_lx * Train_all_coef[0]) / Train_all_coef[1]
# 绘制图1-4
plt.figure('图1-4')
plt.plot(Train_all_lx, Train_all_ly, c='blue')
# 绘制图1-4中良性肿瘤样本点，标记为红色的o，并且保存绘图句柄Clump_Thickness
plt.scatter(df_test_positive['Clump Thickness'], df_test_positive['Cell Size'], marker='o', s=200, c='red')
# 绘制图1-4中恶性肿瘤样本点，标记为黑色的x，并且保存绘图句柄Cell_Size
plt.scatter(df_test_negative['Clump Thickness'], df_test_negative['Cell Size'], marker='x', s=150, c='black')
plt.xlabel(u'肿块厚度', fontproperties=font)
plt.ylabel(u'细胞尺寸', fontproperties=font)
plt.title(u'图1-4', fontproperties=font)
plt.show()
