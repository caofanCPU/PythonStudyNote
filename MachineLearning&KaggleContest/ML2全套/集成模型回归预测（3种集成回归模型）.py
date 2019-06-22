# -*- coding: utf-8 -*-
"""
Created on Fri Jan 13 23:19:54 2017

@author: CY_XYZ
"""

# 从sklearn.datasets导入波士顿房价数据读取器
from sklearn.datasets import load_boston

# 读取房价信息数据存储至变量boston中
boston = load_boston()
# 输出数据描述
# print( boston.DESCR )
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
##################################################################

# 从sklearn.ensemble导入三种集成回归模块
# RandomForestRegressor、ExtraTreeGressor、GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, GradientBoostingRegressor

# 使用RandomForestRegressor训练模型，并对测试集做出预测
rfr = RandomForestRegressor()
rfr.fit(s_X_train, s_y_train)
rfr_y_predict = rfr.predict(s_X_test)
# 使用ExtraTreeGressor训练模型，并对测试集做出预测
etr = ExtraTreesRegressor()
etr.fit(s_X_train, s_y_train)
etr_y_predict = etr.predict(s_X_test)
# 使用GradientBoostingRegressor训练模型，并对测试集做出预测
gbr = GradientBoostingRegressor()
gbr.fit(s_X_train, s_y_train)
gbr_y_predict = gbr.predict(s_X_test)

# 使用R-squared、MSE、MAE三个指标对3种集成回归模型进行评价
# 从sklearn.metrics导入mean_squared_error、mean_absolute_error模块
from sklearn.metrics import mean_squared_error, mean_absolute_error

print('----随机森林回归模型----')
print('R-squared value of RandomForestRegressor is:', rfr.score(s_X_test, s_y_test))
print('MSE value of RandomForestRegressor is:', mean_squared_error(ss_y.inverse_transform(s_y_test), ss_y.inverse_transform(rfr_y_predict)))
print('MAE value of RandomForestRegressor is:', mean_absolute_error(ss_y.inverse_transform(s_y_test), ss_y.inverse_transform(rfr_y_predict)))

print('----极端回归森林模型----')
print('R-squared value of ExtraTreesRegressor is:', etr.score(s_X_test, s_y_test))
print('MSE value of ExtraTreesRegressor is:', mean_squared_error(ss_y.inverse_transform(s_y_test), ss_y.inverse_transform(etr_y_predict)))
print('MAE value of ExtraTreeGressor is:', mean_absolute_error(ss_y.inverse_transform(s_y_test), ss_y.inverse_transform(etr_y_predict)))
# 利用训练好的极端森林回归模型，输出每种特征对预测目标的贡献度
# 导入numpy科学计算工具包

# 导入pandas工具包，进行DataFrame数据框化操作
import pandas as pd
# 导入numpy工具包，进行int64强制类型转换操作
import numpy as np

print('极端森林模型种每种特征对预测目标的贡献度如下：')
# 将数组转变为数据框对象，转化后feature_IMP只有唯一一列，列名为0
# 如果要操作该列，用feature_IMP[0]
feature_IMP = pd.DataFrame(etr.feature_importances_)
# 给feature_IMP新增一列，列名为1，该列的值用boston.feature_names特征名称对应填充
feature_IMP[1] = pd.DataFrame(boston.feature_names)
# 再将feature_IMP的列名人性化处理
feature_IMP.columns = ['feature_IMP', 'feature_names']

# 按照某一列、升序(True)或降序(False)排序整个数据框
# 优先使用.sort_values(  )方法；.sort(  )方法过时啦
feature_IMP = feature_IMP.sort_values('feature_IMP', ascending=True)
# feature_IMP = feature_IMP.sort( columns = 'feature_IMP', ascending = False )
# 将首列'index'人性化依次编号
feature_IMP.index = np.int64([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13])
print(feature_IMP)
"""这种方法导致错误的结果：特征名与贡献度未匹配
# 将预测目标贡献度数组结果先变为了列表对象，并进行降序排列，保存至变量feature_IMP
feature_IMP = list( etr.feature_importances_ )
feature_IMP.sort( reverse = True )
feature_names = list( boston.feature_names ) 
# 将贡献度数据与对应的特征名对应连接起来
feature_IMP_result = [ [ i, j ]\
                       for i,j in zip( feature_IMP, feature_names ) ]

print( feature_IMP_result )
"""
# 列表list对象有自己的排序方法，XXX.sort( key = len, reverse = True )，默认升序
# 参数key = len对于字符串排序按照字符长度排序；参数reverse = True代表降序排序
# [ [a,b] for a,b in zip(list1,list2) ]
# np.set_printoptions(threshold='nan')  #全部输出，但是可能会导致莫名错误，慎用

print('----梯度提升回归模型----')
print('R-squared value of GradientBoostingRegressor is:', gbr.score(s_X_test, s_y_test))
print('MSE value of GradientBoostingRegressor is:', mean_squared_error(ss_y.inverse_transform(s_y_test), ss_y.inverse_transform(gbr_y_predict)))
print('MAE value of GradientBoostingRegressor is:', mean_absolute_error(ss_y.inverse_transform(s_y_test), ss_y.inverse_transform(gbr_y_predict)))

"""
运行结果：
----随机森林回归模型----
R-squared value of RandomForestRegressor is: 0.831811465536
MSE value of RandomForestRegressor is: 13.0415566929
MAE value of RandomForestRegressor is: 2.44070866142
----极端回归森林模型----
R-squared value of ExtraTreesRegressor is: 0.763517664231
MSE value of ExtraTreesRegressor is: 18.3371464567
MAE value of ExtraTreeGressor is: 2.56031496063
极端森林模型种每种特征对预测目标的贡献度如下：
    feature_IMP feature_names
1      0.003427            ZN
2      0.010197           RAD
3      0.012009             B
4      0.016817           AGE
5      0.020269          CRIM
6      0.021507          CHAS
7      0.030086           NOX
8      0.032708           DIS
9      0.039520         INDUS
10     0.061118       PTRATIO
11     0.064657           TAX
12     0.283295         LSTAT
13     0.404390            RM
----梯度提升回归模型----
R-squared value of GradientBoostingRegressor is: 0.843161766033
MSE value of GradientBoostingRegressor is: 12.1614396987
MAE value of GradientBoostingRegressor is: 2.27168590131

----随机森林回归模型----
R-squared value of RandomForestRegressor is: 0.798428216913
MSE value of RandomForestRegressor is: 15.6301370079
MAE value of RandomForestRegressor is: 2.45244094488
----极端回归森林模型----
R-squared value of ExtraTreesRegressor is: 0.789558338048
MSE value of ExtraTreesRegressor is: 16.3179188976
MAE value of ExtraTreeGressor is: 2.53850393701
极端森林模型种每种特征对预测目标的贡献度如下：
    feature_IMP feature_names
1      0.003813            ZN
2      0.013170             B
3      0.013389           RAD
4      0.014906          CHAS
5      0.022453           TAX
6      0.023266           DIS
7      0.023687           NOX
8      0.031082           AGE
9      0.032451         INDUS
10     0.034568          CRIM
11     0.044168       PTRATIO
12     0.302735         LSTAT
13     0.440314            RM
----梯度提升回归模型----
R-squared value of GradientBoostingRegressor is: 0.841115619183
MSE value of GradientBoostingRegressor is: 12.3201005743
MAE value of GradientBoostingRegressor is: 2.26816767903

----随机森林回归模型----
R-squared value of RandomForestRegressor is: 0.795125227045
MSE value of RandomForestRegressor is: 15.8862551181
MAE value of RandomForestRegressor is: 2.53102362205
----极端回归森林模型----
R-squared value of ExtraTreesRegressor is: 0.791785985222
MSE value of ExtraTreesRegressor is: 16.145184252
MAE value of ExtraTreeGressor is: 2.4262992126
极端森林模型种每种特征对预测目标的贡献度如下：
    feature_IMP feature_names
1      0.004898            ZN
2      0.006989           RAD
3      0.009248          CHAS
4      0.016428             B
5      0.017969           AGE
6      0.019914         INDUS
7      0.028486           NOX
8      0.029819           DIS
9      0.037515       PTRATIO
10     0.050960          CRIM
11     0.056227           TAX
12     0.287848            RM
13     0.433699         LSTAT
----梯度提升回归模型----
R-squared value of GradientBoostingRegressor is: 0.84116802972
MSE value of GradientBoostingRegressor is: 12.3160365935
MAE value of GradientBoostingRegressor is: 2.26928272628
"""
