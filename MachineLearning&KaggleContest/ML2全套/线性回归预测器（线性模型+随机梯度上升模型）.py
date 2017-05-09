# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 20:14:39 2017

@author: CY_XYZ
"""

# 从sklearn.datasets导入波士顿房价数据读取器
from sklearn.datasets import load_boston
# 读取房价信息数据存储至变量boston中
boston = load_boston(  )
# 输出数据描述
print( boston.DESCR )
# 查看boston数据说明信息后，确定特征变量和预测变量y
X = boston.data
y = boston.target
# 从sklearn.cross_validation导入数据分割器
from sklearn.cross_validation import train_test_split
# 随机采样25%的数据构造房价预测的测试集，其余75%作为训练集
X_train, X_test, y_train, y_test = train_test_split( X,
                                                     y,
                                                     test_size = 0.25,
                                                     random_state = 33 )
# 分析回归目标值的差异
'''
# 未对数据做处理
# 导入numpy科学计算工具包
import numpy as np
print( 'The max target value is:', np.max( boston.target ) )
print( 'The min target value is:', np.min( boston.target ) )
print( 'The average target value is:', np.mean( boston.target ) )
'''
# 由于预测目标房价之间的差异较大，因而需要对特征值及目标值进行标准化处理
# 训练集数据、测试集数据的标准化处理
# 从sklearn.preprocessing导入数据标准化模块
from sklearn.preprocessing import StandardScaler
# 分别初始化特征值、目标值的标准化器
ss_X = StandardScaler(  )
ss_y = StandardScaler(  )
# 分别对训练集数据、测试集数据的特征值、目标值进行标准化处理
s_X_train = ss_X.fit_transform( X_train )
s_X_test = ss_X.transform( X_test )

s_y_train = ss_y.fit_transform( y_train )
s_y_test = ss_y.transform( y_test )

# 使用线性回归模型(LinearRegression)和梯度上升模型(SGDRegressor)分别进行回归预测
# 从sklear.linear_model导入LinearRegression
from sklearn.linear_model import LinearRegression
# 使用默认配置初始化线性回归器(LinearRegression)
lr = LinearRegression(  )
# 进行线性回归预测的参数训练
lr.fit( s_X_train, s_y_train )
# 进行线性回归预测
lr_y_predict = lr.predict( s_X_test )

# 从sklearn.linear_model导入SGDRegression
from sklearn.linear_model import SGDRegressor
# 使用默认配置初始化梯度上升回归器
sgdr = SGDRegressor(  )
# 进行梯度上升回归预测的参数训练
sgdr.fit( s_X_train, s_y_train )
# 进行梯度上升回归预测
sgdr_y_predict = sgdr.predict( s_X_test )

# 对于回归问题，需要考量回归值与真实值的差异，也要兼顾问题本身真实值的波动
# 使用3种回归问题的评价机制，用两种方法调用R-squared评价模块
# 使用LinearRegression模型自带的评估模块，输出评价结果
print( 'The value od default measurement of LinearRegression is:',
       lr.score( s_X_test, s_y_test ) )
# 使用SGDRegressor模型自带的评估模块，输出评价结果
print( 'The value od default measurement of SGDRegressor is:',
       sgdr.score( s_X_test, s_y_test ) )
# 从sklearn.metrics导入r2_score、mean_squared_error、mean_absolute_erron模块
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
# 使用r2_score模块进行评估
print( 'The value of R-squared of LinearRegression is:',
       r2_score( s_y_test, lr_y_predict ) )
print( 'The value of R-squared of SGDRegressor is:',
       r2_score( s_y_test, sgdr_y_predict ) )
# 使用均方差模块进行评估
print( 'The mean-squqred-error of LinearRegression is:',
       mean_squared_error( ss_y.inverse_transform( s_y_test ),
                           ss_y.inverse_transform( lr_y_predict ) ) )
print( 'The mean-squqred-error of SGDRegressor is:',
       mean_squared_error( ss_y.inverse_transform( s_y_test ),
                           ss_y.inverse_transform( sgdr_y_predict ) ) )
# 使用平均绝对误差进行评估
print( 'The mean-absolute-error of LinearRegression is:',
       mean_absolute_error( ss_y.inverse_transform( s_y_test ),
                            ss_y.inverse_transform( lr_y_predict ) ) )
print( 'The mean-absolute-error of SGDRegressor is:',
       mean_absolute_error( ss_y.inverse_transform( s_y_test ),
                            ss_y.inverse_transform( sgdr_y_predict ) ) )

"""
运行结果：
The value od default measurement of LinearRegression is: 0.6763403831
The value od default measurement of SGDRegressor is: 0.655618578635
The value of R-squared of LinearRegression is: 0.6763403831
The value of R-squared of SGDRegressor is: 0.655618578635
The mean-squqred-error of LinearRegression is: 25.0969856921
The mean-squqred-error of SGDRegressor is: 26.703781236
The mean-absolute-error of LinearRegression is: 3.5261239964
The mean-absolute-error of SGDRegressor is: 3.51216811309
# 模型自带的评分函数与r2_score模块得到的结果是分别相等的
"""








