# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 23:54:58 2017

@author: CY_XYZ
"""

####################################################################
####################################################################
##              XGBoost模型：集大成者
##  隶属集成模型，主要思想：把成百上千个分类准确率较低的树模型组合起来
##                       成为一个准确率很高的模型
##  XGBoost工具最大的特点：自动利用CPU的多线程并行，提高迭代的速度和结果的精度

# 导入pandas用于数据分析
import pandas as pd

# 利用pandas的read_csv模块直接从互联网爬下泰坦尼克号乘客信息数据
titanic = pd.read_csv('http://biostat.mc.vanderbilt.edu/' \
                      'wiki/pub/Main/DataSets/titanic.txt')
# 人工选取pclass、age、sex作为判断乘客生还能力的特征依据
X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
# 对于缺失数据，使用全体乘客的平均值代替，保证正确预测的同时尽可能不影响预测任务
X['age'].fillna(X['age'].mean(), inplace=True)

# 对原始数据进行分割，25%的数据用于测试，75%的数据用来训练
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
# 从sklearn.feature_extraction导入DictVectorizer
from sklearn.feature_extraction import DictVectorizer

# 初始化DictVectorizer
vec = DictVectorizer(sparse=False)
# 对原始数据进行特征向量化处理
# 请注意训练集使用.fit_transform(  )
#      测试集使用.transform(  )
X_train_vec = vec.fit_transform(X_train.to_dict(orient='record'))
X_test_vec = vec.transform(X_test.to_dict(orient='record'))
# 采用默认配置的随机森林分类器进行参数训练
# 从sklearn.ensemble导入RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

# 初始化随机森林分类器
rfc = RandomForestClassifier()
# 进行随机森林参数训练
rfc.fit(X_train_vec, y_train)
# 输出XGBClassifier训练的精确度
print('The accuracy of RandomForestClassifier on testing set is:', rfc.score(X_test_vec, y_test))
# 
# 从xgboost导入XGBClassifier模型
from xgboost import XGBClassifier

# 初始化XGBClassifier
xgbc = XGBClassifier()
# 进行集成分类模型训练
xgbc.fit(X_train_vec, y_train)
# 输出XGBClassifier训练的精确度
print('The accuracy of XGBoostClassifier on testing set is:', xgbc.score(X_test_vec, y_test))
'''
运行结果：
The accuracy of RandomForestClassifier on testing set is: 0.781155015198
The accuracy of XGBoostClassifier on testing set is: 0.787234042553
'''
#####################################################################
#####################################################################
##               tensorflow框架
##  tensorflow是一个完整的编码框架，在其内部有自己所定义的常量、变量、数据操作
##  特点：tensorflow使用图(Graph)表示计算任务，使用会话(Session)来执行图
##       tensorflow像搭积木一样将各个不同的计算模块拼接成流程图

# 导入tensorflow工具包
import tensorflow as tf

# import numpy as np
# 初始化一个Tensorflow常量：Hello Google Tensorflow!'
#                    并命名为greeting作为一个计算模块
greeting = tf.constant('Hello Google Tensorflow!')
# 启动一个会话
sess = tf.Session()
# 使用会话执行greeting计算模块
result = sess.run(greeting)
# 输出会话执行结果
print(result)
# 关闭会话
sess.close()

'''
运行结果：
b'Hello Google Tensorflow!'
'''
##########################################
##使用tensorflow执行一次线性函数的计算，并在一个隐式会话中执行
# 导入tensorflow工具包
import tensorflow as tf

# 声明matrix1为tensorflow的一个1 * 2的行向量
matrix1 = tf.constant([[3., 3.]])
# 声明matrix2为tensorflow的一个2 * 1的列向量
matrix2 = tf.constant([[2.], [2.]])
# 将上述两个算子相乘，作为新的算例【计算模块】product
product = tf.matmul(matrix1, matrix2)
# 将product与一个标量2.0求和拼接，作为最终的linear算例
linear = tf.add(product, tf.constant(2.0))
# 直接在会话中执行linear算例，并使用python特有的with功能将会话隐式执行
# 这相当于将上面所有的单独算例拼接成流程图执行
with tf.Session() as sess:
    result_linear = sess.run(linear)
    print('The linear-run-result is:', result_linear)
'''
运行结果：
The linear-run-result is: [[ 14.]]
'''
