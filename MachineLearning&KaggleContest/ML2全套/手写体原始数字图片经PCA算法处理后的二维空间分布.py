# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 23:47:28 2017

@author: CY_XYZ
"""
import os

import numpy as np
# 导入pandas、numpy工具包
import pandas as pd

# 获得手写数字图像数据框，保存在变量digits_train中

#######################################
'''
# 数据集1(原书)互联网读取：
digits_train = pd.read_csv( 'https://archive.ics.uci.edu/'\
                            'ml/machine-learning-databases/'\
                            'optdigits/optdigits.tra', header = None )
digits_test = pd.read_csv( 'https://archive.ics.uci.edu/'\
                            'ml/machine-learning-databases/'\
                            'optdigits/optdigits.tes', header = None )
# X_digits = digits_train[np.arange(64)]
# y_digits = digits_train[64]
'''
########################################
'''
# 数据集2(非原书)互联网读取：
digits_train = pd.read_csv( 'https://archive.ics.uci.edu/'\
                            'ml/machine-learning-databases/'\
                            'pendigits/pendigits.tra', header = None )
X_digits = digits_train[np.arange(16)]
y_digits = digits_train[16]
'''
#########################################
# 数据集1、2的本地读取方法
# 数据集1(原书)
inputPar = os.path.dirname(__file__)
inputTrain = inputPar + os.path.sep + 'optdigits_train.tra'
inputTest = inputPar + os.path.sep + 'optdigits_test.txt'
print(inputTrain)
digits_train = pd.read_csv(inputTrain, header=None)
digits_test = pd.read_csv(inputTest, header=None)

'''数据集2(非原书)
digits_train = pd.read_csv( 'G:/P_Anaconda3-4.2.0/PA-WORK/AI-ML机器学习/'\
                            'P机器学习及实践&Kaggle竞赛之路/ML2全套/pendigits.tra',
                            header = None )
X_digits = digits_train[np.arange(16)]
y_digits = digits_train[16]
'''
#########################################

# 输出训练集、测试集的数据规模和维度信息
print(digits_train.shape)
print(digits_test.shape)

#########################################
#########################################

# 将原始64维用PCA压缩重建为2维，并绘图表现出来，仅供简要直观理解PCA使用
# 保存digits_train数据到文件
# np.save( 'Add_Digits_Source_data.npy', digits_train )

X_digits = digits_train[np.arange(64)]
y_digits = digits_train[64]

# 导入sklearn.decomposition保重的PCA模块
from sklearn.decomposition import PCA

estimator = PCA(n_components=2)
X_pca = estimator.fit_transform(X_digits)
# 导入matplotlib包中的pyplot模块，为绘图做准备
from matplotlib import pyplot as plt


# Python3中xrange(  )函数报错解决方法：
# 不是取消了xrange，而是取消了range，同时将xrange重新命名成range;
# 如果希望生成原来的列表形式的range，只要用list(range(...))就可以了。

# 自定义一个函数plot_pca_scatter(  )函数
def plot_pca_scatter():
    # 注释也要注意缩进，否则报错！！！
    # 设定10种颜色white的小圆圈分别标记0、1、2、3、4、5、6、7、8、9共10个数字结果
    # 黑色'black'标记   0
    # 蓝色'blue'标记    1
    # 紫色'purple'标记  2
    # 黄色'yellow'标记  3
    # 白色'white'标记   4
    # 红色'red'标记     5
    # 浅绿'lime'标记    6
    # 青色'cyan'标记    7
    # 橙色'orange'标记  8
    # 灰色'grey'标记    9
    colors = ['black', 'blue', 'purple', 'yellow', 'white', 'red', 'lime', 'cyan', 'orange', 'grey']
    # 新建画图窗口并命名为'手写体数字PCA二维空间分布'
    plt.figure('手写体数字PCA二维空间分布')
    ##########################################    
    '''
    # 中途尝试过使用sklearn.datasets内置数据集
    from sklearn.datasets import load_digits
    digits_train = load_digits(  )
    X_digits = digits_train.data[np.arange(64)]
    y_digits = digits_train.data[64]
    # 原始数据读取后得到的是数据框，数据框.as_matrix()可行
    # 内置数据集读取后得到的是数组，数组没有as_matrix()属性
    # y_digits.as_matrix()会引起报错，改为y_digits即可
    # 但是会遗漏数据，所以这里的代码可以完全忽略，制作为扯淡尝试
    '''
    ###########################################
    # 使用for循环+range(  )迭代器
    for i in range(len(colors)):
        # y_digits是一个序列对象，可用.as_matrix()或者.values取到值
        # 且y_digits的值已经为0~9共10个整数
        # [ y_digits.as_matrix(  ) == i ]用于提取0~9对应的数据值索引
        # X_pca[:,0]用于提取第一列
        # 合起来即为提取出全为i结果的数据的横坐标px，纵坐标py
        px = X_pca[:, 0][y_digits.as_matrix() == i]
        py = X_pca[:, 1][y_digits.values == i]
        # 绘制手写数字图片PCA结果为i的散点图
        plt.scatter(px, py, c=colors[i])

    # 通过for循环，成功绘制了0~9结果的10种叠加散点图

    # 按照绘制顺序给绘图窗口添加'0'~'9'共10个图例
    plt.legend(np.arange(0, 10).astype(str))
    # x轴标注
    plt.xlabel('First Principal Component')
    # y轴标注
    plt.ylabel('Second Principal Component')
    # 显示绘图窗口
    plt.show()


# 自定义函数结束

# 直接调用自定义函数，不带任何参数   
plot_pca_scatter()

##################################
##################################
# 使用原始数据像素特征和经过PCA压缩重建(特征降维)后的低维特征
# 在相同配置下的支持向量机(分类)模型上分别进行手写体数字(图像)识别
# 对训练集、测试集进行特征向量、目标变量的分割
X_train = digits_train[np.arange(64)]
y_train = digits_train[64]
X_test = digits_test[np.arange(64)]
y_test = digits_test[64]

# 从sklearn.svm导入基于线性核的支持向量机分类模型LinearSVC
from sklearn.svm import LinearSVC

# 使用默认配置初始化LinearSVC，对原始64维像素特征的训练数据进行建模，并作出预测
linear_svc = LinearSVC()
linear_svc.fit(X_train, y_train)
linear_svc_y_predict = linear_svc.predict(X_test)

# 使用PCA将原始64维数据压缩重建为20维数据
# 导入sklearn.decomposition保重的PCA模块
from sklearn.decomposition import PCA

estimator = PCA(n_components=2)
# 训练集原始特征【适应转化】至20个正交方向
# 测试集原始特征【转化】至20个正交方向
pca_X_train = estimator.fit_transform(X_train)
pca_X_test = estimator.transform(X_test)
# 使用默认配置初始化LinearSVC，对压缩重建后的20维训练数据进行建模，并作出预测
pca_svc = LinearSVC()
pca_svc.fit(pca_X_train, y_train)
pca_svc_y_predict = pca_svc.predict(pca_X_test)

# 对于基于线性核的支持向量机分类模型，在原始64维度数据、压缩重建后的20维度数据识别性能评估
# 从sklearn.metrics导入classification_report进行详细性能分析
from sklearn.metrics import classification_report

print('----线性核SVC在原始64维度数据的性能----')
print(linear_svc.score(X_test, y_test))
print(classification_report(y_test, linear_svc_y_predict, target_names=np.arange(10).astype(str)))
print('----线性核SVC在压缩重建后20维度数据的性能----')
print(pca_svc.score(pca_X_test, y_test))
print(classification_report(y_test, pca_svc_y_predict, target_names=np.arange(10).astype(str)))

"""
运行结果：
runfile('G:/P_Anaconda3-4.2.0/PA-WORK/AI-ML机器学习/P机器学习及实践&Kaggle竞赛之路/ML2全套/手写体原始数字图片经PCA算法处理后的二维空间分布.py', wdir='G:/P_Anaconda3-4.2.0/PA-WORK/AI-ML机器学习/P机器学习及实践&Kaggle竞赛之路/ML2全套')
(3823, 65)
(1797, 65)
----线性核SVC在原始64维度数据的性能----
0.931552587646
             precision    recall  f1-score   support

          0       0.99      0.98      0.99       178
          1       0.96      0.84      0.89       182
          2       0.98      0.97      0.97       177
          3       0.93      0.93      0.93       183
          4       0.95      0.98      0.96       181
          5       0.91      0.97      0.94       182
          6       0.99      0.98      0.98       181
          7       0.98      0.91      0.94       179
          8       0.88      0.82      0.85       174
          9       0.78      0.94      0.86       180

avg / total       0.94      0.93      0.93      1797

----线性核SVC在压缩重建后20维度数据的性能----
0.919309961046
             precision    recall  f1-score   support

          0       0.98      0.96      0.97       178
          1       0.88      0.85      0.86       182
          2       0.93      0.97      0.95       177
          3       0.99      0.89      0.94       183
          4       0.95      0.96      0.95       181
          5       0.86      0.98      0.92       182
          6       0.98      0.97      0.98       181
          7       0.96      0.88      0.92       179
          8       0.92      0.82      0.87       174
          9       0.79      0.92      0.85       180

avg / total       0.92      0.92      0.92      1797

##################################################
----线性核SVC在原始64维度数据的性能----
0.939899833055
             precision    recall  f1-score   support

          0       0.99      0.98      0.99       178
          1       0.90      0.93      0.92       182
          2       0.99      0.98      0.98       177
          3       0.97      0.91      0.94       183
          4       0.95      0.97      0.96       181
          5       0.89      0.97      0.93       182
          6       0.99      0.98      0.98       181
          7       0.98      0.91      0.94       179
          8       0.91      0.86      0.88       174
          9       0.85      0.91      0.88       180

avg / total       0.94      0.94      0.94      1797

----线性核SVC在压缩重建后20维度数据的性能----
0.927100723428
             precision    recall  f1-score   support

          0       0.98      0.97      0.97       178
          1       0.87      0.88      0.88       182
          2       0.97      0.98      0.97       177
          3       0.98      0.91      0.94       183
          4       0.95      0.95      0.95       181
          5       0.88      0.98      0.93       182
          6       0.97      0.96      0.97       181
          7       0.96      0.91      0.93       179
          8       0.87      0.87      0.87       174
          9       0.85      0.86      0.86       180

avg / total       0.93      0.93      0.93      1797

#####################################################
----线性核SVC在原始64维度数据的性能----
0.939899833055
             precision    recall  f1-score   support

          0       0.99      0.98      0.99       178
          1       0.90      0.93      0.92       182
          2       0.99      0.98      0.98       177
          3       0.97      0.91      0.94       183
          4       0.95      0.97      0.96       181
          5       0.89      0.97      0.93       182
          6       0.99      0.98      0.98       181
          7       0.98      0.91      0.94       179
          8       0.91      0.86      0.88       174
          9       0.85      0.91      0.88       180

avg / total       0.94      0.94      0.94      1797

----线性核SVC在压缩重建后10维度数据的性能----
0.883695047301
             precision    recall  f1-score   support

          0       0.98      0.96      0.97       178
          1       0.77      0.86      0.81       182
          2       0.96      0.87      0.91       177
          3       0.91      0.87      0.89       183
          4       0.93      0.93      0.93       181
          5       0.84      0.98      0.90       182
          6       0.99      0.91      0.95       181
          7       0.94      0.87      0.90       179
          8       0.80      0.76      0.78       174
          9       0.78      0.81      0.79       180

avg / total       0.89      0.88      0.88      1797

#####################################################
----线性核SVC在原始64维度数据的性能----
0.939899833055
             precision    recall  f1-score   support

          0       0.99      0.98      0.99       178
          1       0.90      0.93      0.92       182
          2       0.99      0.98      0.98       177
          3       0.97      0.91      0.94       183
          4       0.95      0.97      0.96       181
          5       0.89      0.97      0.93       182
          6       0.99      0.98      0.98       181
          7       0.98      0.91      0.94       179
          8       0.91      0.86      0.88       174
          9       0.85      0.91      0.88       180

avg / total       0.94      0.94      0.94      1797

----线性核SVC在压缩重建后5维度数据的性能----
0.808013355593
             precision    recall  f1-score   support

          0       0.95      0.98      0.96       178
          1       0.74      0.82      0.78       182
          2       0.81      0.82      0.82       177
          3       0.86      0.74      0.80       183
          4       0.83      0.86      0.85       181
          5       0.77      0.84      0.80       182
          6       0.99      0.91      0.95       181
          7       0.83      0.92      0.87       179
          8       0.74      0.40      0.52       174
          9       0.61      0.78      0.68       180

avg / total       0.81      0.81      0.80      1797

####################################################
----线性核SVC在原始64维度数据的性能----
0.939899833055
             precision    recall  f1-score   support

          0       0.99      0.98      0.99       178
          1       0.90      0.93      0.92       182
          2       0.99      0.98      0.98       177
          3       0.97      0.91      0.94       183
          4       0.95      0.97      0.96       181
          5       0.89      0.97      0.93       182
          6       0.99      0.98      0.98       181
          7       0.98      0.91      0.94       179
          8       0.91      0.86      0.88       174
          9       0.85      0.91      0.88       180

avg / total       0.94      0.94      0.94      1797

----线性核SVC在压缩重建后2维度数据的性能----
0.500278241514
             precision    recall  f1-score   support

          0       0.54      0.90      0.68       178
          1       0.27      0.41      0.32       182
          2       0.32      0.20      0.24       177
          3       0.42      0.78      0.55       183
          4       0.57      0.96      0.71       181
          5       0.00      0.00      0.00       182
          6       0.93      0.85      0.89       181
          7       0.79      0.68      0.73       179
          8       0.45      0.12      0.19       174
          9       0.18      0.11      0.13       180

avg / total       0.45      0.50      0.44      1797
"""
