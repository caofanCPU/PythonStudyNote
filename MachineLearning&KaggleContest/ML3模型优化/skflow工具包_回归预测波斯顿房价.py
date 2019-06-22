# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 20:42:29 2017

@author: CY_XYZ
"""

#######################################################################
##      使用skflow内置LinearRegressor、DNN(DeepNeuralNetwork)进行回归预测
##      skflow是对tensorflow的进一步封装，与sklearn的使用接口类似
# 
from sklearn import datasets, metrics, preprocessing, cross_validation

#
boston = datasets.load_boston()
# 获取房屋数据特征以及对应的房价
X, y = boston.data, boston.target
# 分割数据，随机采样25%作为测试样本
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=33)
# 对数据特征进行标准化处理
scaler = preprocessing.StandardScaler()
X_train_scaler = scaler.fit_transform(X_train)
X_test_scaler = scaler.transform(X_test)
#
import skflow

# 使用skflow的LinearRegressor线性回归模型
# skflow是对tensorflow的进一步封装
tf_lr = skflow.TensorFlowLinearRegressor(steps=10000, learning_rate=0.01, batch_size=50)
tf_lr.fit(X_train_scaler, y_train)
tf_lr_y_predict = tf_lr.predict(X_test_scaler)
# 输出skflow中TensorFlowLinearRegressor模型的回归性能
print('--------skflow内置的线性回归模型--------')
print('The mean-absolute-error of skflow.TensorFlowLinearRegressor\n' \
      'on boston datasets is:', metrics.mean_absolute_error(tf_lr_y_predict, y_test))
print('The mean-squared-error of skflow.TensorFlowLinearRegressor\n' \
      'on boston datasets is:', metrics.mean_squared_error(tf_lr_y_predict, y_test))
print('The R-squared of skflow.TensorFlowLinearRegressor\n' \
      'on boston datasets is:', metrics.r2_score(tf_lr_y_predict, y_test))
# 使用skflow内置的DNNRegreseor深度学习神经网络回归模型
tf_dnn_regressor = skflow.TensorFlowDNNRegressor(hidden_units=[100, 40], steps=10000, learning_rate=0.01, batch_size=50)
tf_dnn_regressor.fit(X_train_scaler, y_train)
tf_dnn_regressor_y_predict = tf_dnn_regressor.predict(X_test_scaler)
# 输出skflow中TensorFlowDNNRegressor模型的回归性能
print('--------skflow内置的深度学习神经网络回归模型--------')
print('The mean-absolute-error of skflow.TensorFlowDNNRegressor\n' \
      'on boston datasets is:', metrics.mean_absolute_error(tf_dnn_regressor_y_predict, y_test))
print('The mean-squared-error of skflow.TensorFlowDNNRegressor\n' \
      'on boston datasets is:', metrics.mean_squared_error(tf_dnn_regressor_y_predict, y_test))
print('The R-squared of skflow.TensorFlowDNNRegressor\n' \
      'on boston datasets is:', metrics.r2_score(tf_dnn_regressor_y_predict, y_test))
# 使用sklearn.ensemble中的RandomForestRegressor随机森林回归模型作对比
from sklearn.ensemble import RandomForestRegressor

rfr = RandomForestRegressor()
rfr.fit(X_train_scaler, y_train)
rfr_y_predict = rfr.predict(X_test_scaler)
# 输出sklearn中RandomForestRegressor模型的回归性能
print('------------随机森林回归模型------------')
print('The mean-absolute-error of RandomForestRegressor on boston datasets is:', metrics.mean_absolute_error(rfr_y_predict, y_test))
print('The mean-squared-error of RandomForestRegressor on boston datasets is:', metrics.mean_squared_error(rfr_y_predict, y_test))
print('The R-squared of RandomForestRegressor on boston datasets is:', metrics.r2_score(rfr_y_predict, y_test))

'''
运行结果：


------------随机森林回归模型------------
The mean-absolute-error of RandomForestRegressor on boston datasets is: 
    2.57692913386
The mean-squared-error of RandomForestRegressor on boston datasets is: 
    15.3417425197
The R-squared of RandomForestRegressor on boston datasets is: 
    0.784707599313
'''
