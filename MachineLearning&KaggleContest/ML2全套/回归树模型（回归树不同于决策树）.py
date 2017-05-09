# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 22:59:35 2017

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
# 回归树(及树模型擅长处理非线性问题，因而不要求数据特征化及统一量化)

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


'''
s_X_train = X_train
s_X_test = X_test
s_y_train = y_train
s_y_test = y_test
# 若不进行数据标准化，容易得到以下运行结果：
----回归树模型----
R-squared value of DecisionTreeRegressor is: 0.673154595999
MSE value of DecisionTreeRegressor is: 2182.51203845
MAE value of DecisionTreeRegressor is: 29.5712778909
'''
#################################################################

# 从sklearn.tree导入DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
# 使用默认配置初始化回归树
dtr = DecisionTreeRegressor(  )
# 将波士顿房价数据构建为回归树
dtr.fit( s_X_train, s_y_train )
# 使用默认配置的单一回归树对测试集数据进行预测
dtr_y_predict = dtr.predict( s_X_test )

# 对回归树模型进行评价
# 从sklearn.metrics导入mean_squared_error、mean_absolute_error模块
from sklearn.metrics import mean_squared_error, mean_absolute_error
print( '----回归树模型----' )
print( 'R-squared value of DecisionTreeRegressor is:',
       dtr.score( s_X_test, s_y_test ) )
print( 'MSE value of DecisionTreeRegressor is:',
       mean_squared_error( ss_y.inverse_transform( s_y_test ),
                           ss_y.inverse_transform( dtr_y_predict ) ) )
print( 'MAE value of DecisionTreeRegressor is:',
       mean_absolute_error( ss_y.inverse_transform( s_y_test ),
                           ss_y.inverse_transform( dtr_y_predict ) ) )

"""
运行结果：
----回归树模型----
R-squared value of DecisionTreeRegressor is: 0.659779959684
MSE value of DecisionTreeRegressor is: 26.3811023622
MAE value of DecisionTreeRegressor is: 3.22204724409

----回归树模型----
R-squared value of DecisionTreeRegressor is: 0.699975966358
MSE value of DecisionTreeRegressor is: 23.2642519685
MAE value of DecisionTreeRegressor is: 3.08031496063

----回归树模型----
R-squared value of DecisionTreeRegressor is: 0.504829756211
MSE value of DecisionTreeRegressor is: 38.3961417323
MAE value of DecisionTreeRegressor is: 3.63858267717

----回归树模型----
R-squared value of DecisionTreeRegressor is: 0.690706838707
MSE value of DecisionTreeRegressor is: 23.982992126
MAE value of DecisionTreeRegressor is: 3.24724409449
"""
