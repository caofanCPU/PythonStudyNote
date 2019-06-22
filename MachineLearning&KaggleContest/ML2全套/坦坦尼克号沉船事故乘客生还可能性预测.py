# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 00:35:24 2017
本例使用泰坦尼克号乘客信息数据，人工选择了3个特征作为生还能力的判断依据，
比较单一决策树、随机增林分类器、梯度提升决策树三种模型的预测结果，
结果表明：梯度提升决策树[优于]随机森林分类器[优于]单一决策树，
其中，随机森林经常作为分类模型的基线系统。
一般结论：集成模型受参数估计的影响，具有一定的不确定性，
         在训练参数过程中耗费更多的时间，
         但是集成模型往往性能好、稳定性高。
编程经验积累：将数据保存为.npy文件，再导入内存，转化为数据框时出现莫名BUG，
            所以还是网上爬或者本地读。
@author: CY_XYZ
"""

print(__doc__)

# 导入pandas用于数据分析
import pandas as pd

##############################
# '''方法1
# 利用pandas的read_csv模块直接从互联网爬下泰坦尼克号乘客信息数据
dataURL = 'http://biostat.mc.vanderbilt.edu/' \
          'wiki/pub/Main/DataSets/titanic.txt'
titanic = pd.read_csv(dataURL)

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# '''
##############################
'''步骤2
# 导入numpy用于数据保存和科学计算
import numpy as np
# 保存泰坦尼克号乘客信息数据
np.save( 'G:/P_Anaconda3-4.2.0/PA-WORK/'\
         'AI-ML机器学习/P机器学习及实践&Kaggle竞赛之路/'\
         'ML2全套/Titanic.npy', titanic )
'''
##############################
##############################
'''
# 泰坦尼克号乘客信息数据已在本地.npy文件中，使用numpy的load方法导入数据
import numpy as np
# 利用numpy的load模块本地读取泰坦尼克号乘客信息数据
titanic = np.load( 'G:/P_Anaconda3-4.2.0/PA-WORK/'\
                       'AI-ML机器学习/P机器学习及实践&Kaggle竞赛之路/'\
                       'ML2全套/Titanic.npy' )
# 网上原始数据为DataFrame格式
# 本地保存导入的.npy文件数据为数组对象，因而需要将数组对象转化为数据据框
column_names = [ 'row.names',
                 'pclass',
                 'survived',
                 'name',
                 'age',
                 'embarked',
                 'home.dest',
                 'room',
                 'ticket',
                 'boat',
                 'sex' ]
# pandas的DataFrame方法的属性columns可以设置数据框的列名
titanic = pd.DataFrame( titanic, columns = column_names )
'''
##############################
# =============================
'''
说明：本地保存数据为npy文件再导入，转化为数据框后会出现不明BUG，所以不推荐!!
本地保存.npy读取的方法复杂，实际中若网速胶好，数据文件不大的话可以直接爬虫
                                 或者先下载到本地，在读取
'''
# =============================
################################
'''该文件第一行自带列名，因而除文件名的其他参数均默认
####默认header = 0，表示文件自带列名，如果文件没有列名，必须设置header = None
""""
column_names = [ 'pclass',
                 'survived',
                 'name',
                 'sex'
                 'age',
                 'sibsp',
                 'parch',
                 'ticket',
                 'fare',
                 'cabin',
                 'embarked',
                 'boat',
                 'body',
                 'home.dest' ]
""""
titanic3 = pd.read_csv( 'G:/P_Anaconda3-4.2.0/PA-WORK/'\
                        'AI-ML机器学习/P机器学习及实践&Kaggle竞赛之路/'\
                        'ML2全套/titanic3.csv' )
'''
# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&

'''
#方法2，本地读取对应的txt文件
titanic = pd.read_csv( 'G:/P_Anaconda3-4.2.0/PA-WORK/'\
                       'AI-ML机器学习/P机器学习及实践&Kaggle竞赛之路/'\
                       'ML2全套/titanic.txt' )
'''
################################

# 人工选取pclass、age、sex作为判断乘客生还能力的特征依据
X = titanic[['pclass', 'age', 'sex']]
y = titanic['survived']
# 对于缺失数据，使用全体乘客的平均值代替，保证正确预测的同时尽可能不影响预测任务
X['age'].fillna(X['age'].mean(), inplace=True)

# 对原始数据进行分割，25%的数据用于测试，75%的数据用来训练
from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
# 对类别进行特征化，转化为特征向量
from sklearn.feature_extraction import DictVectorizer

vec = DictVectorizer(sparse=False)
v_X_train = vec.fit_transform(X_train.to_dict(orient='record'))
v_X_test = vec.transform(X_test.to_dict(orient='record'))
# 使用单一决策树(Decision Tree)进行模拟训练及预测分析
from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier()
dtc.fit(v_X_train, y_train)
dtc_y_predict = dtc.predict(v_X_test)
# 使用随机森林分类器(Random Forest Classifier)进行集成模型的训练和预测分析
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
rfc.fit(v_X_train, y_train)
rfc_y_predict = rfc.predict(v_X_test)
# 使用梯度提升决策树(Gradient Tree Boosting)进行集成模型的训练和预测分析
from sklearn.ensemble import GradientBoostingClassifier

gbc = GradientBoostingClassifier()
gbc.fit(v_X_train, y_train)
gbc_y_predict = gbc.predict(v_X_test)

# 使用准确性、召回率、F1指标、精确率共4个指标评价比较以上3种模型
from sklearn.metrics import classification_report

# 输出单一决策树在测试集上的分类准确性、召回率、F1指标、精确率
print('The accurary of DecisionTree is:', dtc.score(v_X_test, y_test))
print(classification_report(dtc_y_predict, y_test))
# 输出随机森林分类器在测试集上的分类准确性、召回率、F1指标、精确性
print('The accurary of RandomForestClassifier is:', rfc.score(v_X_test, y_test))
print(classification_report(rfc_y_predict, y_test))
# 输出梯度提升决策树在测试集上的分类准确性、召回率、F1指标、精确率
print('The accurary of GradientTreeBoosing is:', gbc.score(v_X_test, y_test))
print(classification_report(gbc_y_predict, y_test))
