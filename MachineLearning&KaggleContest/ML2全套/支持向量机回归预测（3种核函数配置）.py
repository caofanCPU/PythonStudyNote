# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 21:38:01 2017

@author: CY_XYZ
"""

# 从sklearn.datasets导入波士顿房价数据读取器
from sklearn.datasets import load_boston

# 读取房价信息数据存储至变量boston中
boston = load_boston()
# 输出数据描述
print(boston.DESCR)
# 查看boston数据说明信息后，确定特征变量和预测变量y
X = boston.data
y = boston.target
# 从sklearn.cross_validation导入数据分割器
from sklearn.cross_validation import train_test_split

# 随机采样25%的数据构造房价预测的测试集，其余75%作为训练集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
# 由于预测目标房价之间的差异较大，因而需要对特征值及目标值进行标准化处理
# 训练集数据、测试集数据的标准化处理
# 从sklearn.preprocessing导入数据标准化模块
from sklearn.preprocessing import StandardScaler

# 分别初始化特征值、目标值的标准化器
ss_X = StandardScaler()
ss_y = StandardScaler()
# 分别对训练集数据、测试集数据的特征值、目标值进行标准化处理
s_X_train = ss_X.fit_transform(X_train)
s_X_test = ss_X.transform(X_test)

s_y_train = ss_y.fit_transform(y_train)
s_y_test = ss_y.transform(y_test)
#################################################################
# 使用3种不同核函数配置的支持向量机回归模型，进行训练，并预测波士顿房价
# 从sklearn.svm种导入支持向量机(回归)模型
from sklearn.svm import SVR

# 使用线性核函数配置的支持向量机模型进行训练及预测
linear_svr = SVR(kernel='linear')
linear_svr.fit(s_X_train, s_y_train)
linear_svr_y_predict = linear_svr.predict(s_X_test)
# 使用多项式核函数配置的支持向量机模型进行训练和预测
poly_svr = SVR(kernel='poly')
poly_svr.fit(s_X_train, s_y_train)
poly_svr_y_predict = poly_svr.predict(s_X_test)
# 使用径向基核函数配置的支持向量机模型进行训练和预测
rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(s_X_train, s_y_train)
rbf_svr_y_predict = rbf_svr.predict(s_X_test)

# 使用R-squared、MSE、MAE指标分别评价三种核函数配置的支持向量机模型
from sklearn.metrics import mean_squared_error, mean_absolute_error

print('----线性核函数SVR-----')
print('R-squared value of linear-SVR is:', linear_svr.score(s_X_test, s_y_test))
print('MSE value of linear-SVR is:', mean_squared_error(ss_y.inverse_transform(s_y_test), ss_y.inverse_transform(linear_svr_y_predict)))
print('MAE value of linear-SVR is:', mean_absolute_error(ss_y.inverse_transform(s_y_test), ss_y.inverse_transform(linear_svr_y_predict)))

print('----多项式核函数SVR-----')
print('R-squared value of poly-SVR is:', poly_svr.score(s_X_test, s_y_test))
print('MSE value of poly-SVR is:', mean_squared_error(ss_y.inverse_transform(s_y_test), ss_y.inverse_transform(poly_svr_y_predict)))
print('MAE value of poly-SVR is:', mean_absolute_error(ss_y.inverse_transform(s_y_test), ss_y.inverse_transform(poly_svr_y_predict)))

print('----径向基核函数SVR-----')
print('R-squared value of rbf-SVR is:', rbf_svr.score(s_X_test, s_y_test))
print('MSE value of rbf-SVR is:', mean_squared_error(ss_y.inverse_transform(s_y_test), ss_y.inverse_transform(rbf_svr_y_predict)))
print('MAE value of rbf-SVR is:', mean_absolute_error(ss_y.inverse_transform(s_y_test), ss_y.inverse_transform(rbf_svr_y_predict)))

"""
运行结果：
----线性核函数SVR-----
R-squared value of linear-SVR is: 0.65171709743
MSE value of linear-SVR is: 27.0063071393
MAE value of linear-SVR is: 3.42667291687
----多项式核函数SVR-----
R-squared value of poly-SVR is: 0.404454058003
MSE value of poly-SVR is: 46.179403314
MAE value of poly-SVR is: 3.75205926674
----径向基核函数SVR-----
R-squared value of rbf-SVR is: 0.756406891227
MSE value of rbf-SVR is: 18.8885250008
MAE value of rbf-SVR is: 2.60756329798
"""
