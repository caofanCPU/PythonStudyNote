# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 23:43:26 2017
特征筛选：与PCA不同，不存在修改特征值，而是侧重于那些对模型性能提升较大的少量特征
执行结果提供的信息：
1.经过初步的特征抽取后，最终训练集、测试集均有474个维度的特征
2.使用全部特征，即474个维度，决策树预测性能为81.16%
3.使用前20%比例的特征，约95个维度，决策树预测性能为82.37%，略高于全部特征筛选
4.通过使用交叉验证的算法，将特征筛选比例与决策树模型性能的关系绘图，找到最佳筛选比例前7%
5.使用前7%比例的特征，约33个维度，决策树预测性能为85.71%，比全部特征筛选高出接近4%

也就是说：全部特征琳琅满目，反而干扰决策；而特征筛选类似'找重点'，一针见血
@author: CY_XYZ
"""

# 导入pandas工具包
import pandas as pd
import os
# 从互联网爬下titanic数据
'''
titanic = pd.read_csv( 'http://biostat.mc.vanderbilt.edu/'\
                       'wiki/pub/Main/DataSets/titanic.txt' )
'''
#方法2，本地读取对应的txt文件
inputPar = os.path.dirname(__file__)
inputFile = inputPar + os.path.sep + 'titanic.txt'
# print(inputFile)
titanic = pd.read_csv( inputFile )

# 查看数据规模，titanic数据规模为DataFrame( 1313, 11 )
print( titanic.shape )
# 分离特征与预测目标
# 分离后，特征变量X数据规模DataFrame( 1313, 8 )，为titanic去除(.drop)所得
X = titanic.drop( ['row.names', 'name', 'survived'], axis = 1 )
# 分离后，预测目标变量y数据规模Series( 1313 )
y = titanic['survived']

# 对缺失数据进行补充
X['age'].fillna( X['age'].mean(  ), inplace = True )
X.fillna( 'UNKNOWN', inplace = True )

# 分割数据集，25%作为测试集，其余75%作为训练集
# 从sklearn.cross_validation导入train_test_split模块
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test =\
train_test_split( X,
                  y,
                  test_size = 0.25,
                  random_state = 33 )
# 类别型特征向量化
# 从sklearn.feature_extraction导入DictVectorizer模块
from sklearn.feature_extraction import DictVectorizer
# 初始化DictVectorizer特征抽取器
vec = DictVectorizer(  )
# 以【标定方向】orient = 'record'【记录】的方式将训练集、测试集【字典】特征化
X_train_vec = vec.fit_transform( X_train.to_dict( orient = 'record' ) )
X_test_vec = vec.transform( X_test.to_dict( orient = 'record' ) )

# 查看特征抽取后特征向量的维度
print( len( vec.feature_names_ ) ) 
# 
# 使用决策树模型依靠所有特征进行预测，并作性能评估
# 从sklearn.tree导入DecisionTreeClassifier模块
from sklearn.tree import DecisionTreeClassifier
# 将决策树模型的【评判】criterion = 'entropy'【平均】，初始化决策树
# 全部特征的决策树模型对象为dtc
dtc = DecisionTreeClassifier( criterion = 'entropy' )
# 使用决策树模型进行参数训练
# 应用全部特征决策树对象dtc，训练参数为( X_train_vec, y_train )
dtc.fit( X_train_vec, y_train )
# 输出决策树模型依靠所有特征后的性能评分
print( 'The accuracy of DecisionTreeClassifier[100%] is:',
       dtc.score( X_test_vec, y_test ) )

# 从sklearn导入feature_selection特征筛选器模块
from sklearn import feature_selection
# 筛选前20%的特征，使用相同配置的决策树模型进行参数训练及预测，并作性能评估
fs = feature_selection.SelectPercentile( feature_selection.chi2,
                                         percentile = 20 )
# 将X_train_vec特征训练集【筛选】:【适应转化】20%的结果保存至X_train_fs
X_train_fs = fs.fit_transform( X_train_vec, y_train )
# 将X_test_vec特征训练集【筛选】:【转化】20%的结果保存至X_test_fs
X_test_fs = fs.transform( X_test_vec )
# 使用相同的决策树配置，初始化决策树
# 20%【部分】特征的决策树模型对象为dtc_fs
dtc_fs = DecisionTreeClassifier( criterion = 'entropy' )
# 应用部分特征决策树对象dtc_fs，训练参数为( X_train_fs, y_train )
dtc_fs.fit( X_train_fs, y_train )
# 输出决策树模型依靠筛选前20%特征后的性能评分
print( 'The accuracy of DecisionTreeClassifier[20%] is:',
       dtc_fs.score( X_test_fs, y_test ) )

#################################################################
# 通过交叉验证的方法，按照固定间隔百分比的方式筛选特征，并作图展示性能随特征筛选比例的关系
# 交叉验证：将全部数据随机分割平均数量的分组，每次都迭代使用其中一组
#          保证所有数据有被训练、测试的机会
# 从sklearn.cross_validation导入cross_val_score模块
from sklearn.cross_validation import cross_val_score
# 导入科学计算工具包numpy
import numpy as np
# 特征筛选按照固定间隔2%进行
# range( 起始值【包括】, 终点值【不包括】, 步长 )，返回的对象是range对象，视作等差序列
# 本例percentiles的长度为50
# percenttiles[0] = 1
# percenttiles[1] = 3
# percenttiles[2] = 5
# percenttiles[3] = 7
# percenttiles[4] = 9
# percenttiles[5] = 11
# ...
# percenttiles[49] = 99
percentiles = range( 1, 100, 2 )
# 如果需要查看percentiles的数值内容，在导入numpy工具包的前提下，执行下一行代码
# print( np.array( percentiles ) )
# 所有特征筛选比例下决策树模型的性能评分结果保存在results中，此处初始化results
results = []
# 使用相同的决策树配置，初始化决策树
dtc_fs_percentiles = DecisionTreeClassifier( criterion = 'entropy' )
# 在循环中迭代特征比例，并依次进行模型训练及性能评估
for i in percentiles:
    # 在特征筛选比例依次为i的条件下，创造特征筛选比例对象fs_percentiles
    fs_percentiles = feature_selection.SelectPercentile( feature_selection.chi2,
                                                         percentile = i )
    # 依次应用特征筛选比例对象fs_percentiles，适应转化训练集特征向量
    X_train_fs_percentiles = fs_percentiles.fit_transform( X_train_vec,
                                                           y_train )
    # 决策树模型对象为dtc_fs_percentiles
    # 在决策树模型对象中，进行参数训练，并采用5折交叉验证
    # 此处训练参数为( X_train_fs_percentiles, y_train )
    scores = cross_val_score( dtc_fs_percentiles,
                              X_train_fs_percentiles,
                              y_train,
                              cv = 5 )
    # 对上述scores的结果取平均值(决策树模型并不稳定，多做几次取平均不失为良策)，并保存
    results = np.append( results, scores.mean(  ) )
# 输出结果results
print( results )

# 找到最佳性能的特征筛选百分比
opt = np.where( results == results.max(  ) )[0]
# 得到的opt变量为numpy.ndarray类型，type(opt)
# 获取其值：opt[0]
# 输出最佳特征筛选百分比
print( 'Optimal number of features [%d%%]' % percentiles[opt[0]] )
# 绘图
import matplotlib.pylab as plt
plt.plot( percentiles, results, 'o-m' )
plt.xlabel( 'percentiles of features' )
plt.ylabel( 'The accuracy of DecisionTreeClassifier' )
plt.grid( True )
# plt.show(  )
#############################################################
# 使用最佳筛选比例的特征，利用相同配置的决策树模型进行参数训练及预测，并进行性能评估
fs_opt_percentiles =\
feature_selection.SelectPercentile( feature_selection.chi2,
                                    percentile = percentiles[opt[0]] )
X_train_fs_opt_percentiles = fs_opt_percentiles.fit_transform( X_train_vec,
                                                               y_train )
X_test_fs_opt_percentiles = fs_opt_percentiles.transform( X_test_vec )
# 使用相同的决策树配置，初始化决策树
dtc_fs_opt_percentiles = DecisionTreeClassifier( criterion = 'entropy' )
dtc_fs_opt_percentiles.fit( X_train_fs_opt_percentiles, y_train )
# 输出最佳特征筛选比例的决策树模型性能得分
print( 'The accuracy of DecisionTreeClassifier[%d%%] is:' % percentiles[opt[0]],
       dtc_fs_opt_percentiles.score( X_test_fs_opt_percentiles,
                                     y_test ) )

'''
运行结果：
(1313, 11)
474
The accuracy of DecisionTreeClassifier[100%] is: 0.811550151976
The accuracy of DecisionTreeClassifier[20%] is: 0.823708206687
[ 0.85063904  0.85673057  0.87501546 ...,  0.86489384  0.85878169
  0.86184292]
Optimal number of features [7%]
The accuracy of DecisionTreeClassifier[7%] is: 0.857142857143
##########################
# 将percentiles改为:percentiles = range( 1, 100, 1 )，则
Optimal number of features [8%]
The accuracy of DecisionTreeClassifier[8%] is: 0.860182370821
'''
