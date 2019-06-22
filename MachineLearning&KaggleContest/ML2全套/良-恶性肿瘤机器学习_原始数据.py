# -*- coding: utf-8 -*-
"""
Created on Tue Jan  3 21:56:25 2017

# 控制台运行命令runfile( '文件路径' )；文件路径所带的\必须换为/
runfile( 'G:/P_Anaconda3-4.2.0/PA-WORK/AI-ML机器学习/'\
         'P机器学习及实践&Kaggle竞赛之路/ML2全套/良-恶性肿瘤机器学习_原始数据.py',
         wdir='G:/P_Anaconda3-4.2.0/PA-WORK/AI-ML机器学习/'\
              'P机器学习及实践&Kaggle竞赛之路/ML2全套' )

# Python续行符为\，对于文件路径，它是一个字符串,使用'A'\【Enter】'B'\【Enter】续行
#                 对于网址，它也是一个字符串，处理方式一样
#                 对于赋值表达式，=\【Enter】另起一行
#                 但注意，Windows文件路径默认用\分隔，因而首先将全部路径\换为/，再处理
@author: CY_XYZ
"""

import os

# 导入numpy与pandas工具包
import numpy as np
import pandas as pd

# 根据原始数据的描述文件，创建特征列表
column_names = ['Sample code number', 'Clump Thickness', 'Uniformity of Cell Size', 'Uniformity of Cell Shape', 'Marginal Adhesion', 'Single Epithelial Cell Size', 'Bare Nuclei', 'Bland Chromatin', 'Normal Nucleoli', 'Mitoses', 'Class']
# 使用pandas.read_csv函数从互联网读取指定数据，在想换行的地方加上'\
'''
html_Sdata = pd.read_csv( 'https://archive.ics.uci.edu/ml/'\
                          'machine-learning-databases/'\
                          'breast-cancer-wisconsin/'\
                          'breast-cancer-wisconsin.data',
                          names = column_names )
'''
# 若网速受限，建议先下载到本地；文件路劲太长，分字符串'\；读取原始数据到变量Sdata
inputPar = os.path.dirname(__file__)
inputFile = inputPar + os.path.sep + 'breast-cancer-wisconsin.txt'
Sdata = pd.read_csv(inputFile, names=column_names)
# 将缺失值标记'?'替换为'NA'；原始数据的缺失值标记必须查看原始数据说明文档
data = Sdata.replace(to_replace='?', value=np.nan)
# 替换原始数据的缺失值后，查看缺失值所在行的信息,以及多少行带有缺失值
NaN_data = data[data.isnull().values == True]
NaN_length = len(NaN_data)

# 剔除带有缺失值的数据，只要有一个维度缺失就剔除
ok_data = data.dropna(how='any')
# 输出ok_data数据量和维度
print(data.shape)
print(NaN_data.shape)
print(ok_data.shape)

# 将ok_data数据的25%作为测试集，其余75%作为训练集
# 使用sklearn.cross_validation里的train_test_split模块用于分个数据
from sklearn.cross_validation import train_test_split

# 随机采样25%的数据用于测试集，其余75%作为训练集
X_train, X_test, y_train, y_test = train_test_split(ok_data[column_names[1:10]], ok_data[column_names[10]], test_size=0.25, random_state=33)
'''
上句代码解读：ok_data[column_names[1:10]] 是选取第1列~第9列作为X
            ok_data[column_names[10]]   是选取第10列作为y
这样就构建出了表达式关系
'''

# 检查训练集的数量和类型分布
'''
TrainSet_result = y_train.value_counts()
TestSet_result = y_test.value_counts()
'''

# 训练集、测试集结果如下
"""
y_train.value_counts()
Out[30]: 
2    344    #训练集344条良性肿瘤记录，168条恶性肿瘤记录，一共512条
4    168
Name: Class, dtype: int64

y_test.value_counts()
Out[31]: 
2    100    #测试集100条良性肿瘤记录，71条恶性肿瘤记录，一共171条
4     71
Name: Class, dtype: int64
"""

# 使用线性分类模型进行良-恶性肿瘤预测任务
# 从sklearn.preprocessing导入StandardScaler
from sklearn.preprocessing import StandardScaler
# 从sklearn.linear_model导入LogisticRegressing和SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

# 标准化数据，保证每个维度的特征数据方差为1，均值为0
# 使得预测结果不会被某些维度过大的特征值所主导，注意训练集和测试集标准化时的区别
ss = StandardScaler()
ss_X_train = ss.fit_transform(X_train)
ss_X_test = ss.transform(X_test)

# 初始化LogisticRegression和SGDClassifier
lr = LogisticRegression()
sgdc = SGDClassifier()
# 调用LogisticRegression中的fit函数/模块来训练模型参数
lr.fit(ss_X_train, y_train)
# 使用训练好的模型lr对X_test进行预测，结果保存至变量lr_y_predict
lr_y_predict = lr.predict(ss_X_test)
# 调用SGDClassifier中的fit函数/模块来训练模型参数
sgdc.fit(ss_X_train, y_train)
# 使用训练好的模型sgdc对X_test进行预测，结果保存至sgdc_y_predict
sgdc_y_predict = sgdc.predict(ss_X_test)

# 序列Series对象才具有.loc()方法、.values和.index属性
# 单独提取序列Series对象的index和values
# I_y_test = y_test.index
# C_y_test = y_test.values

######正负分类样本后回打破原有数据的序列性质(排列性)，因而直接绘制比较总的结果
'''
# 构建测试集的正负(良性恶性)分类样本
y_test_positive = y_test.loc[y_test.values == 2].values
y_test_negative = y_test.loc[y_test.values == 4].values

# 构建[LogisticRegression回归分类]分类样本，序列化函数pandas.Series()
S_lr_y_predict = pd.Series( lr_y_predict )
lr_y_predict_positive = S_lr_y_predict.loc[S_lr_y_predict.values == 2].values
lr_y_predict_negative = S_lr_y_predict.loc[S_lr_y_predict.values == 4].values
# 构建[SGDClassifier随机梯度]分类样本
S_sgdc_y_predict = pd.Series( sgdc_y_predict )
sgdc_y_predict_positive = S_sgdc_y_predict.loc[S_sgdc_y_predict.values == 2].values
sgdc_y_predict_negative = S_sgdc_y_predict.loc[S_sgdc_y_predict.values == 4].values

# 绘制测试集和预测集的散点图
# 由于前述操作只提取了结果列y，需要构建一个序号列数组x
# np.arange( 1, len( y_test )+1 )

'''

# 导入matplotlib工具包中的pyplot并简化命名plt
import matplotlib.pyplot as plt

#
arange_x = np.arange(1, len(y_test) + 1)
plt.scatter(arange_x, y_test.values, marker='o', s=2, c='red')
plt.scatter(arange_x, lr_y_predict, marker='*', s=2, c='green')
plt.scatter(arange_x, sgdc_y_predict, marker='+', s=2, c='blue')
plt.show()

# 以准确性Accuracy、精确率Precision、召回率Recall、F1综合4个指标来评价模型
# 从sklearn.metrics导入classification_report模块
from sklearn.metrics import classification_report

# 使用逻辑斯蒂回归分类模型自带评分函数score获得逻辑斯蒂回归分类模型的准确性结果
print('Accuracy of LR Classifier:', lr.score(ss_X_test, y_test))
# 使用classification_report模块获得LogisticRegression的其他三个指标结果
print(classification_report(y_test, lr_y_predict, target_names=['Benign', 'Malignant']))
# 使用随机梯度下降分类模型自带评分函数score获得随机梯度下降分类模型的准确性结果
print('Accuracy of SGDC Classifier:', sgdc.score(ss_X_test, y_test))
# 使用classification_report模块获得SGDClassifier的其他三个指标结果
print(classification_report(y_test, sgdc_y_predict, target_names=['Benign', 'Malignant']))
