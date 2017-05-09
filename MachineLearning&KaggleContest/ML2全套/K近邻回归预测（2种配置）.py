# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 22:12:23 2017
K近邻回归、K近邻分类都是无参数训练模型，它根据测试样本在训练样本中的分布做出决策
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
#################################################################

# 使用2种不同配置下的K近邻回归模型，训练及预测波士顿房价
# 从sklearn.neighbors导入KNeighborsRegressor(K近邻回归器)
from sklearn.neighbors import KNeighborsRegressor
# 初始化K近邻回归器，配置预测方式为平均回归：weights = 'uniform'
uni_knr = KNeighborsRegressor( weights = 'uniform' )
uni_knr.fit( s_X_train, s_y_train )
uni_knr_y_predict = uni_knr.predict( s_X_test )

# 再重置初始化K近邻回归器，配置预测方式为根据距离加权回归：weights = 'distance'
dis_knr = KNeighborsRegressor( weights = 'distance' )
dis_knr.fit( s_X_train, s_y_train )
dis_knr_y_predict = dis_knr.predict( s_X_test )

# 对平均回归、根据距离加权回归2种K近邻回归模型进行评估
# 导入MSE、MAE评估模块
from sklearn.metrics import mean_squared_error, mean_absolute_error
print( '-----平均回归方式----' )
print( 'R-squared value of uniform-weights_KNeighborsRegressor is:',
       uni_knr.score( s_X_test, s_y_test ) )
print( 'MSE value of uniform-weights_KNeighborsRegressor is:',
       mean_squared_error( ss_y.inverse_transform( s_y_test ),
                           ss_y.inverse_transform( uni_knr_y_predict ) ) )
print( 'MSE value of uniform-weights_KNeighborsRegressor is:',
       mean_absolute_error( ss_y.inverse_transform( s_y_test ),
                            ss_y.inverse_transform( uni_knr_y_predict ) ) )

print( '----根据距离加权回归-方式---' )
print( 'R-squared value of distance-weights_KNeighborsRegressor is:',
       dis_knr.score( s_X_test, s_y_test ) )
print( 'MSE value of distance-weights_KNeighborsRegressor is:',
       mean_squared_error( ss_y.inverse_transform( s_y_test ),
                           ss_y.inverse_transform( dis_knr_y_predict ) ) )
print( 'MAE value of distance-weights_KNeighborsRegressor is:',
       mean_absolute_error( ss_y.inverse_transform( s_y_test ),
                            ss_y.inverse_transform( dis_knr_y_predict ) ) )

"""
运行结果
-----平均回归方式----
R-squared value of uniform-weights_KNeighborsRegressor is: 0.690345456461
MSE value of uniform-weights_KNeighborsRegressor is: 24.0110141732
MSE value of uniform-weights_KNeighborsRegressor is: 2.96803149606
----根据距离加权回归-方式---
R-squared value of distance-weights_KNeighborsRegressor is: 0.719758997016
MSE value of distance-weights_KNeighborsRegressor is: 21.7302501609
MAE value of distance-weights_KNeighborsRegressor is: 2.80505687851
"""
