# -*- coding: utf-8 -*-
"""
Created on Wed Jan  4 20:38:01 2017

@author: CY_XYZ
"""

# 从sklearn.datasets导入手写体数字加载器
from sklearn.datasets import load_digits
# 从通过数据加载器获得手写体数字的数码图像数据并储存在digits变量中
digits = load_digits(  )
# 检视数据规模和特征维度
print( digits.data.shape )
# 从sklearn.cross_validation导入train_test_split用于数据分割
from sklearn.cross_validation import train_test_split
# 随机选取75%的数据作为训练集，剩余25%的数据作为测试集
X_train, X_test, y_train, y_test = train_test_split( digits.data,
                                                     digits.target,
                                                     test_size = 0.25,
                                                     random_state = 33 )
'''
上句代码解读：digits.data   是作为X
            digits.target 是作为y
            从digits的数据结构得到的，sklearn.datasets内部数据集已做好了处理
这样就构建出了表达式关系
'''

# 检视训练集、测试集的规模
print( '手写体数字识别训练集规模：', y_train.shape )
print( '手写体数字识别测试集规模：', y_test.shape )
# 从sklearn.preprocessing导入数据标准化模块
from sklearn.preprocessing import StandardScaler
# 仍然需要对训练和测试数据集的特征数据进行标准化，注意训练集和测试集标准化时的区别
ss = StandardScaler(  )
S_X_train = ss.fit_transform( X_train )
S_X_test = ss.transform( X_test )
# 从sklearn.svm导入基于线性假设的支持向量机分类器LinearSVC
from sklearn.svm import LinearSVC
# 初始化线性假设的支持向量机分类器LinearSVC
lsvc = LinearSVC(  )
# 进行模型训练
lsvc.fit( S_X_train, y_train )
# 利用训练好的模型对测试样本的数字类别进行预测，预测结果保存在变量
lsvc_y_predict = lsvc.predict( S_X_test )
# 使用LSVC模型自带的lsvc.score函数进行准确性测评
print( 'The Accuracy of Linear SVC is', lsvc.score( S_X_test, y_test ) )
# 仍然使用sklearn.metrics中classification_report模块对线性支持向量机模型结果进行详细分析
from sklearn.metrics import classification_report
print( classification_report( y_test,
                              lsvc_y_predict,
                              target_names = digits.target_names.astype( str ) ) )

"""
运行结果：
(1797, 64)
手写体数字识别训练集规模： (1347,)
手写体数字识别测试集规模： (450,)
The Accuracy of Linear SVC is 0.953333333333
             precision    recall  f1-score   support

          0       0.92      1.00      0.96        35
          1       0.96      0.98      0.97        54
          2       0.98      1.00      0.99        44
          3       0.93      0.93      0.93        46
          4       0.97      1.00      0.99        35
          5       0.94      0.94      0.94        48
          6       0.96      0.98      0.97        51
          7       0.92      1.00      0.96        35
          8       0.98      0.84      0.91        58
          9       0.95      0.91      0.93        44

avg / total       0.95      0.95      0.95       450
"""



