# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 19:24:56 2017

@author: CY_XYZ
"""

########################################################
##      使用Tensorflow自定义一个线性分类器
##      对良/恶性肿瘤进行预测

import tensorflow as tf
import numpy as np
import pandas as pd
import os
# 从本地导入肿瘤训练集、测试集数据
inputPra = os.path.dirname(__file__)
inputTrain = inputPra + os.path.sep + 'breast-cancer-train.csv'
inputTest = inputPra + os.path.sep + 'breast-cancer-test.csv'
train = pd.read_csv( inputTrain )
test = pd.read_csv( inputTest )
# 分割特征与分类目标
X_train = np.float32( train[['Clump Thickness', 'Cell Size']].T )
y_train = np.float32( train['Type'].T )
X_test = np.float32( test[['Clump Thickness', 'Cell Size']].T )
y_test = np.float32( test['Type'].T )
# 定义一个tensorflow变量b作为线性模型的截距，初始化为1.0
b = tf.Variable( tf.zeros( [1.0] ) )
# 定义一个tensorflow变量W作为线性模型的系数，初始化为-1.0至1.0之间均匀分布的随机数
W = tf.Variable( tf.random_uniform( [ 1, 2 ],
                                    -1.0,
                                    1.0 ) )
# 显示定义线性函数的算例
y = tf.matmul( W, X_train ) + b
# 使用tensorflow中的reduce_mean取得训练集上的均方误差
loss = tf.reduce_mean( tf.square( y - y_train ) )
# 使用梯度下降法估计参数W、b，设置迭代步长为0.01，与sklearn中的SGDRegressor类似
optimizer = tf.train.GradientDescentOptimizer( 0.01 ) 
# 以最小二乘法损失为优化目标
train_opt = optimizer.minimize( loss )
# 初始化所有变量，第一句较老，IPython推荐第二句
# init = tf.initialize_all_variables(  )
init = tf.global_variables_initializer(  )
# 开启tensorflow会话
sess = tf.Session(  )
# 执行变量初始化操作
sess.run( init )
# 迭代1000次，训练参数
for step in range( 0, 1000 ):
    sess.run( train_opt )
    # step依次迭代0，1，...，111，...，222，...，888，...，999
    # step为999正是最后一次的结果
    if  step % 111 == 0:
        print( '第', step, '次：',
               '系数W：', sess.run( W ),
               '截距b：', sess.run( b ) )
# 准备测试样本
test_negative =\
test.loc[test['Type'] == 0][ ['Clump Thickness', 'Cell Size'] ]
test_positive =\
test.loc[test['Type'] == 1][ ['Clump Thickness', 'Cell Size'] ]
# 以最终的参数绘图
import matplotlib.pyplot as plt
plt.scatter( test_negative['Clump Thickness'],
             test_negative['Cell Size'],
             marker = 'o',
             s = 200,
             c = 'red' )
plt.scatter( test_positive['Clump Thickness'],
             test_positive['Cell Size'],
             marker = 'x',
             s = 150,
             c = 'black' )
plt.xlabel( 'Clump Thickness' )
plt.ylabel( 'Cell Size' )
plt.grid( True )
# 绘制线性分类线选取的x轴数据
lx = np.arange( 0, 12 )
# 以0.5作为分界面，计算方式如下：
ly =  ( 0.5 - sess.run( b ) - lx * sess.run( W )[0][0] )\
      /\
      ( sess.run( W )[0][1] )
plt.plot( lx, ly, color = 'green', linewidth = 4 )
# plt.show(  )
'''
运行结果：
第 0 次： 系数W： [[-0.38910568  0.61156416]] 截距b： [-0.03212841]
第 111 次： 系数W： [[-0.03635213  0.17082274]] 截距b： [-0.09092805]
第 222 次： 系数W： [[ 0.03994784  0.09597905]] 截距b： [-0.09238077]
第 333 次： 系数W： [[ 0.05442158  0.08128043]] 截距b： [-0.08968401]
第 444 次： 系数W： [[ 0.05718761  0.0782861 ]] 截距b： [-0.0880695]
第 555 次： 系数W： [[ 0.05772378  0.07763766]] 截距b： [-0.08735312]
第 666 次： 系数W： [[ 0.05783049  0.07748399]] 截距b： [-0.0870645]
第 777 次： 系数W： [[ 0.05785273  0.07744329]] 截距b： [-0.08695292]
第 888 次： 系数W： [[ 0.05785771  0.07743123]] 截距b： [-0.08691055]
第 999 次： 系数W： [[ 0.05785896  0.07742734]] 截距b： [-0.08689464]
'''
