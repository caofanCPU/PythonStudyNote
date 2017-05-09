# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 05:10:58 2017

@author: CY_XYZ
"""

# 导入numpy科学计算工具包
import numpy as np
# 从sklearn.datasets导入新闻数据抓取器fetch_20newgroups
from sklearn.datasets import fetch_20newsgroups
# 使用fetch_20newsgroups即时从互联网下载数据
news = fetch_20newsgroups( subset = 'all' )
# 查验数据规模和细节
print( len( news.data ) )
# print( news.data[ 0 ] )
# 从sklearn.cross_validation导入train_test_split
from sklearn.cross_validation import train_test_split
# 取新闻数据前3000条随机采样25%作为测试集，其余75%作为训练集
X_train, X_test, y_train, y_test = \
train_test_split( news.data[:3000],
                  news.target[:3000],
                  test_size = 0.25,
                  random_state = 33 )
# 从sklearn.svm导入支持向量机(分类)SVC模块
from sklearn.svm import SVC
# 对新闻数据进行特征抽取
# 从sklearn.feature_extraction.text导入TfidfVectorizer模块
from sklearn.feature_extraction.text import TfidfVectorizer
# 从sklearn.pipeline导入Pipeline模块简化搭建系统流程，将文本抽取与分类器模型串联起来
from sklearn.pipeline import Pipeline
clf = Pipeline( [( 'vect',
                   TfidfVectorizer( stop_words = 'english',
                                    analyzer = 'word' ) ),
                   ( 'svc', SVC(  ) )] )
# 此处需要实验得2个超参数得个数分别是4个、3个
parameters = { 'svc__gamma':np.logspace( -2, 1, 4 ),
               'svc__C':np.logspace( -1, 1, 3 ) }
# 从sklearn.grid_search中导入网格搜索模块GridSearchCV
from sklearn.grid_search import GridSearchCV
# 将12组参数组合以及初始化的Pipeline、3折交叉验证得要求告知网格搜索模块
# 参数refit = True代表以交叉验证训练集得到的最佳超参数
# 重新对所有可用训练集与开发集进行训练，作为最终用于性能评估得最佳模型得参数
###############################################################
# 使用超参数搜索之网格搜索方法
gs = GridSearchCV( clf,
                   parameters,
                   verbose = 2,
                   refit = True,
                   cv = 3 )
# 执行单线程网格搜索
gs.fit( X_train, y_train )
# 下一句代码只能在Console控制台单独运行
# %time _ = gs.fit( X_train, y_train )
print( gs.best_params_, gs.best_score_ )
print( gs.score( X_test, y_test ) )
###############################################################
# 使用超参数搜索之并行搜索方法
gs_B = GridSearchCV( clf,
                     parameters,
                     verbose = 2,
                     refit = True,
                     cv = 3,
                     n_jobs = -1 )
# 执行单线程网格搜索
gs_B.fit( X_train, y_train )
# 下一句代码只能在Console控制台单独运行
# %time _ = gs.fit( X_train, y_train )
print( '-----------------------------------------------------' )
print( gs_B.best_params_, gs_B.best_score_ )
print( gs_B.score( X_test, y_test ) )

'''

Fitting 3 folds for each of 12 candidates, totalling 36 fits
[CV] svc__C=0.1, svc__gamma=0.01 .....................................
[CV] ............................ svc__C=0.1, svc__gamma=0.01 -   6.5s
[CV] svc__C=0.1, svc__gamma=0.01 .....................................
[CV] ............................ svc__C=0.1, svc__gamma=0.01 -   6.4s
[CV] svc__C=0.1, svc__gamma=0.01 .....................................
[CV] ............................ svc__C=0.1, svc__gamma=0.01 -   6.6s
[CV] svc__C=0.1, svc__gamma=0.1 ......................................
[CV] ............................. svc__C=0.1, svc__gamma=0.1 -   6.5s
[CV] svc__C=0.1, svc__gamma=0.1 ......................................
[CV] ............................. svc__C=0.1, svc__gamma=0.1 -   6.8s
[CV] svc__C=0.1, svc__gamma=0.1 ......................................
[CV] ............................. svc__C=0.1, svc__gamma=0.1 -   6.5s
[CV] svc__C=0.1, svc__gamma=1.0 ......................................
[CV] ............................. svc__C=0.1, svc__gamma=1.0 -   6.5s
[CV] svc__C=0.1, svc__gamma=1.0 ......................................
[CV] ............................. svc__C=0.1, svc__gamma=1.0 -   6.6s
[CV] svc__C=0.1, svc__gamma=1.0 ......................................
[CV] ............................. svc__C=0.1, svc__gamma=1.0 -   6.6s
[CV] svc__C=0.1, svc__gamma=10.0 .....................................
[CV] ............................ svc__C=0.1, svc__gamma=10.0 -   6.9s
[CV] svc__C=0.1, svc__gamma=10.0 .....................................
[CV] ............................ svc__C=0.1, svc__gamma=10.0 -   6.6s
[CV] svc__C=0.1, svc__gamma=10.0 .....................................
[CV] ............................ svc__C=0.1, svc__gamma=10.0 -   6.6s
[CV] svc__C=1.0, svc__gamma=0.01 .....................................
[CV] ............................ svc__C=1.0, svc__gamma=0.01 -   6.4s
[CV] svc__C=1.0, svc__gamma=0.01 .....................................
[CV] ............................ svc__C=1.0, svc__gamma=0.01 -   6.5s
[CV] svc__C=1.0, svc__gamma=0.01 .....................................
[CV] ............................ svc__C=1.0, svc__gamma=0.01 -   6.7s
[CV] svc__C=1.0, svc__gamma=0.1 ......................................
[CV] ............................. svc__C=1.0, svc__gamma=0.1 -   6.3s
[CV] svc__C=1.0, svc__gamma=0.1 ......................................
[CV] ............................. svc__C=1.0, svc__gamma=0.1 -   6.3s
[CV] svc__C=1.0, svc__gamma=0.1 ......................................
[CV] ............................. svc__C=1.0, svc__gamma=0.1 -   6.4s
[CV] svc__C=1.0, svc__gamma=1.0 ......................................
[CV] ............................. svc__C=1.0, svc__gamma=1.0 -   6.5s
[CV] svc__C=1.0, svc__gamma=1.0 ......................................
[CV] ............................. svc__C=1.0, svc__gamma=1.0 -   6.8s
[CV] svc__C=1.0, svc__gamma=1.0 ......................................
[CV] ............................. svc__C=1.0, svc__gamma=1.0 -   6.5s
[CV] svc__C=1.0, svc__gamma=10.0 .....................................
[CV] ............................ svc__C=1.0, svc__gamma=10.0 -   6.4s
[CV] svc__C=1.0, svc__gamma=10.0 .....................................
[CV] ............................ svc__C=1.0, svc__gamma=10.0 -   6.8s
[CV] svc__C=1.0, svc__gamma=10.0 .....................................
[CV] ............................ svc__C=1.0, svc__gamma=10.0 -   6.9s
[CV] svc__C=10.0, svc__gamma=0.01 ....................................
[CV] ........................... svc__C=10.0, svc__gamma=0.01 -   6.3s
[CV] svc__C=10.0, svc__gamma=0.01 ....................................
[CV] ........................... svc__C=10.0, svc__gamma=0.01 -   6.5s
[CV] svc__C=10.0, svc__gamma=0.01 ....................................
[CV] ........................... svc__C=10.0, svc__gamma=0.01 -   6.4s
[CV] svc__C=10.0, svc__gamma=0.1 .....................................
[CV] ............................ svc__C=10.0, svc__gamma=0.1 -   6.4s
[CV] svc__C=10.0, svc__gamma=0.1 .....................................
[CV] ............................ svc__C=10.0, svc__gamma=0.1 -   6.6s
[CV] svc__C=10.0, svc__gamma=0.1 .....................................
[CV] ............................ svc__C=10.0, svc__gamma=0.1 -   6.5s
[CV] svc__C=10.0, svc__gamma=1.0 .....................................
[CV] ............................ svc__C=10.0, svc__gamma=1.0 -   6.5s
[CV] svc__C=10.0, svc__gamma=1.0 .....................................
[CV] ............................ svc__C=10.0, svc__gamma=1.0 -   6.6s
[CV] svc__C=10.0, svc__gamma=1.0 .....................................
[CV] ............................ svc__C=10.0, svc__gamma=1.0 -   6.6s
[CV] svc__C=10.0, svc__gamma=10.0 ....................................
[CV] ........................... svc__C=10.0, svc__gamma=10.0 -   6.5s
[CV] svc__C=10.0, svc__gamma=10.0 ....................................
[CV] ........................... svc__C=10.0, svc__gamma=10.0 -   6.6s
[CV] svc__C=10.0, svc__gamma=10.0 ....................................
[CV] ........................... svc__C=10.0, svc__gamma=10.0 -   6.6s
[Parallel(n_jobs=1)]: Done  36 out of  36 | elapsed:  4.0min finished
{'svc__C': 10.0, 'svc__gamma': 0.10000000000000001} 0.790666666667
0.822666666667
-----------------------------------------------------
Fitting 3 folds for each of 12 candidates, totalling 36 fits
[Parallel(n_jobs=-1)]: Done  36 out of  36 | elapsed:  1.5min finished
{'svc__C': 10.0, 'svc__gamma': 0.10000000000000001} 0.790666666667
0.822666666667
'''


















