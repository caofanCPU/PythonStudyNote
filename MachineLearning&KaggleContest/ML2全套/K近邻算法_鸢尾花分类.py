# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 22:39:19 2017
本例使用sklearn.datasets自带数据集iris进行K近邻算法分类的机器学习，
K近邻算法的特点：“近朱者赤，近墨者黑”；
属于无参数模型中的建议模型，该模型每处理一个测试样本，都需要对所有
预先加载在内存中的训练样本进行遍历，逐一计算相似度、排序并且选取K个
最近邻训练样本的标记，进而做出分类决策；
算法时间复杂度：平方级别，因而对于大规模数据就需要仔细权衡时间代价。
@author: CY_XYZ
"""

print( '========本程序使用说明========' )
print( __doc__ )
print( '============================' )
# 从sklearn.datasets导入iris数据集
from sklearn.datasets import load_iris
# 使用加载器读取数据并且存入变量iris中
iris = load_iris(  )
# 检查数据规模和维度，利用数据的shape属性
print( iris.data.shape )
# 查看数据集说明(保持好习惯)
# print( iris.DESCR )

# 对iris数据集进行分割
# 从sklearn.cross_validation导入train_test_split用于数据分割
from sklearn.cross_validation import train_test_split
# 使用train_test_split分割数据集，并借助random_state采集25%作为测试集
X_train, X_test, y_train, y_test = \
train_test_split( iris.data,
                  iris.target,
                  test_size = 0.25,
                  random_state = 33 )
# 从sklearn.preprocessing导入标准模块
from sklearn.preprocessing import StandardScaler
# 从sklearn.neighbors导入KNeighborsClassifier
from sklearn.neighbors import KNeighborsClassifier
# 对训练样本和测试样本的特征进行标准化
# 初始化标准对象ss
ss = StandardScaler(  )
# 对训练集数据X_train进行训练适应特征化处理，
K_X_train = ss.fit_transform( X_train )
# 对测试集书籍X_test进行标准化处理
K_X_test = ss.transform( X_test )
# 使用K近邻分类器对测试数据进行类别预测，预测结果存储在变量K_y_predic中
# 初始化K紧邻分类器对象Knc
Knc = KNeighborsClassifier(  )
# 使用K紧邻分类器对训练集数据X_train，y_train进行适应性训练
Knc.fit( K_X_train, y_train )
# 训练后，进行测试集数据X_test的结果预测
K_y_predict = Knc.predict( K_X_test )
# 对K近邻分类器在鸢尾花( iris )数据集上的预测性能进行评估
print( 'The accuracy of K-Neighbor Classifier is:',
       Knc.score( K_X_test, y_test ) )
# 依然使用sklearn.metrics里面的classfication_report模块详细评价预测结果
from sklearn.metrics import classification_report
print( classification_report( y_test,
                              K_y_predict,
                              target_names = iris.target_names ) )
