# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 02:47:43 2017

@author: CY_XYZ
"""
# Pizza饼已知训练数据集
X_train = [ [6], [8], [10], [14], [18] ]
y_train = [ [7], [9], [13], [17.5], [18] ]

# 使用线性回归模型预测
from sklearn.linear_model import LinearRegression
lr = LinearRegression(  )
lr.fit( X_train, y_train )

import numpy as np
# 在x轴上从0到26(不包含)均匀采样100个数据点作为测试
xx = np.linspace( 0, 26, 100 )
xx = xx.reshape( xx.shape[0], 1 )

yy_predict = lr.predict( xx )
######################################################
# 使用2次多项式回归模型预测
from sklearn.preprocessing import PolynomialFeatures
poly2 = PolynomialFeatures( degree = 2 )
X_train_poly2 = poly2.fit_transform( X_train )
lr_poly2 = LinearRegression(  )
lr_poly2.fit( X_train_poly2, y_train )
# 重新映射绘图用x轴采样数据
xx_poly2 = poly2.transform( xx )
yy_poly2_predict = lr_poly2.predict( xx_poly2 )
######################################################
# 使用4次多项式回归模型预测
# from sklearn.preprocessing import PolynomialFeatures
poly4 = PolynomialFeatures( degree = 4 )
X_train_poly4 = poly4.fit_transform( X_train )
lr_poly4 = LinearRegression(  )
lr_poly4.fit( X_train_poly4, y_train )
# 重新映射绘图用x轴采样数据
xx_poly4 = poly4.transform( xx )
yy_poly4_predict = lr_poly4.predict( xx_poly4 )
######################################################
# 绘图函数plt.plot常用参数
'''
颜色（color 简写为 c）：

蓝色： 'b' (blue)
绿色： 'g' (green)
红色： 'r' (red)
蓝绿色(墨绿色)： 'c' (cyan)
红紫色(洋红)： 'm' (magenta)
黄色： 'y' (yellow)
黑色： 'k' (black)
白色： 'w' (white)
灰度表示： e.g. 0.75 ([0,1]内任意浮点数)
RGB表示法： e.g. '#2F4F4F' 或 (0.18, 0.31, 0.31)
任意合法的html中的颜色表示： e.g. 'red', 'darkslategray'
线型（linestyle 简写为 ls）：

实线： '-'
虚线： '--'
虚点线： '-.'
点线： ':'
点： '.' 
点型（标记marker）：

像素： ','
圆形： 'o'
上三角： '^'
下三角： 'v'
左三角： '<'
右三角： '>'
方形： 's'
加号： '+' 
叉形： 'x'
棱形： 'D'
细棱形： 'd'
三脚架朝下： '1'（就是丫）
三脚架朝上： '2'
三脚架朝左： '3'
三脚架朝右： '4'
六角形： 'h'
旋转六角形： 'H'
五角形： 'p'
垂直线： '|'
水平线： '_'
gnuplot 中的steps： 'steps' （只能用于kwarg中）
标记大小（markersize 简写为 ms）： markersize： 实数 
标记边缘宽度（markeredgewidth 简写为 mew）：markeredgewidth：实数
标记边缘颜色（markeredgecolor 简写为 mec）：markeredgecolor：颜色选项中的任意值
标记表面颜色（markerfacecolor 简写为 mfc）：markerfacecolor：颜色选项中的任意值
透明度（alpha）：alpha： [0,1]之间的浮点数
线宽（linewidth）：linewidth： 实数
'''
import matplotlib.pyplot as plt
plt.figure( 'Figure of Train Data' )
plt.scatter( X_train, y_train, s = 150, c = 'k')
plt.grid( True )
plt1, = plt.plot( xx, yy_predict, label = 'Degree = 1', linewidth = 3 )
plt2, = plt.plot( xx, yy_poly2_predict, label = 'Degree = 2', linewidth = 3 )
plt4, = plt.plot( xx, yy_poly4_predict, label = 'Degree = 4', linewidth = 3 )
# 标注坐标轴：plt.axis( xmin, xmax, ymin, ymax )
plt.axis( [0, 25, 0, 25] )
plt.xlabel( 'Diameter of Pizza' )
plt.ylabel( 'Price of Pizza' )
plt.legend( handles = [plt1, plt2, plt4] )
# plt.show(  )
print( 'The R-squared value of LinearRegression performing'\
       'on the training data is:', lr.score( X_train, y_train ) )
print( 'The R-squared value of LinearRegression performing'\
       'on the training data is:', lr_poly2.score( X_train_poly2, y_train ) )
print( 'The R-squared value of LinearRegression performing'\
       'on the training data is:', lr_poly4.score( X_train_poly4, y_train ) )
'''
运行结果：
The R-squared value of LinearRegression performingon the training data is: 
    0.910001596424
The R-squared value of LinearRegression performingon the training data is: 
    0.98164216396
The R-squared value of LinearRegression performingon the training data is: 
    1.0
The R-squared value of LinearRegression performingon the test data is: 
    0.809726832467
The R-squared value of LinearRegression performingon the test data is: 
    0.867544365635
The R-squared value of LinearRegression performingon the test data is: 
    0.809588079577
'''

# 评估3中回归模型在测试集数据上的性能
# 事实上，赛事往往给出训练集【含预测变量的真实结果】，测试集【预测变量真是结果？？】
# 然后需要你取寻找模型、训练参数，最后提交代码【可能会只允许提交一次】
# 赛事主办方拿着真实结果评估你的模型性能
# 整个流程类似本例
# 准备测试数据
X_test = [ [6], [8], [11], [16] ]
y_test = [ [8], [12], [15], [18] ]
plt.figure( 'Figure of Test Data' )
plt.scatter( X_test, y_test, s = 150, c = 'y')
plt.grid( True )
plt1, = plt.plot( xx, yy_predict, label = 'Degree = 1', linewidth = 3 )
plt2, = plt.plot( xx, yy_poly2_predict, label = 'Degree = 2', linewidth = 3 )
plt4, = plt.plot( xx, yy_poly4_predict, label = 'Degree = 4', linewidth = 3 )
# 标注坐标轴：plt.axis( xmin, xmax, ymin, ymax )
plt.axis( [0, 25, 0, 25] )
plt.xlabel( 'Diameter of Pizza' )
plt.ylabel( 'Price of Pizza' )
plt.legend( handles = [plt1, plt2, plt4] )
# plt.show(  )
print( 'The R-squared value of LinearRegression performing'\
       'on the test data is:', lr.score( X_test, y_test ) )
X_test_poly2 = poly2.transform( X_test )
print( 'The R-squared value of LinearRegression performing'\
       'on the test data is:', lr_poly2.score( X_test_poly2, y_test ) )
X_test_poly4 = poly4.transform( X_test )
print( 'The R-squared value of LinearRegression performing'\
       'on the test data is:', lr_poly4.score( X_test_poly4, y_test ) )
print( '-----------------------------------------------------------------' )
########################################################################
########################################################################
# 本例中，1次线性回归属于欠拟合
#        4次多项式回归属于过拟合
# 使用模型正则化方法，用L1范数(Lasso模型)、L2范数(Ridge模型)，纠正过拟合
# L1范数使得大部分特征失去对目标的贡献，让有效特征构建成稀疏矩阵
# L2范数使得参数向量中大部分元素变得很小，压制了参数之间的差异
#####L1范数正则化前后对比
from sklearn.linear_model import Lasso
lasso_poly4 = Lasso(  )
lasso_poly4.fit( X_train_poly4, y_train )
print( 'The R-squared value of [Lasso] LinearRegression performing'\
       'on the test data is:', lasso_poly4.score( X_test_poly4, y_test ) )
# 对比L1范数正则化前后4次多项式得参数列表
print( 'Before Lasso:\n', lr_poly4.coef_ )
print( 'After Lasso:\n', lasso_poly4.coef_ )
#####L2范数正则化前后对比
from sklearn.linear_model import Ridge
ridge_poly4 = Ridge(  )
ridge_poly4.fit( X_train_poly4, y_train )
print( '-----------------------------------------------------------------' )
print( 'The R-squared value of LinearRegression performing'\
       'on the test data is:', lr_poly4.score( X_test_poly4, y_test ) )
print( 'The R-squared value of [Ridge] LinearRegression performing'\
       'on the test data is:', ridge_poly4.score( X_test_poly4, y_test ) )
# 对比L1范数正则化前后4次多项式得参数列表
print( 'Before Lasso:\n', lr_poly4.coef_,
       '\nSum-Squared value is:', np.sum( lr_poly4.coef_ ** 2 ) )
print( 'After Lasso:\n', ridge_poly4.coef_,
       '\nSum-Squared value is:', np.sum( ridge_poly4.coef_ ** 2 ) )
'''
运行结果：
The R-squared value of LinearRegression performingon the training data is: 
    0.910001596424
The R-squared value of LinearRegression performingon the training data is: 
    0.98164216396
The R-squared value of LinearRegression performingon the training data is: 
    1.0
The R-squared value of LinearRegression performingon the test data is: 
    0.809726832467
The R-squared value of LinearRegression performingon the test data is: 
    0.867544365635
-----------------------------------------------------------------
The R-squared value of LinearRegression performingon the test data is: 
    0.809588079577
The R-squared value of [Lasso] LinearRegression performingon the test data is:
    0.83889268736
Before Lasso:
 [[  0.00000000e+00  -2.51739583e+01   3.68906250e+00  -2.12760417e-01
    4.29687500e-03]]
After Lasso:
 [  0.00000000e+00   0.00000000e+00   1.17900534e-01   5.42646770e-05
  -2.23027128e-04]
-----------------------------------------------------------------
The R-squared value of LinearRegression performingon the test data is: 
    0.809588079577
The R-squared value of [Ridge] LinearRegression performingon the test data is:
    0.837420175937
Before Lasso:
 [[  0.00000000e+00  -2.51739583e+01   3.68906250e+00  -2.12760417e-01
    4.29687500e-03]] 
Sum-Squared value is: 647.382645737
After Lasso:
 [[ 0.         -0.00492536  0.12439632 -0.00046471 -0.00021205]] 
Sum-Squared value is: 0.0154989652036    
'''
