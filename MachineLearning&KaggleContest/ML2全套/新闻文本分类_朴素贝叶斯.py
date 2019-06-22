# -*- coding: utf-8 -*-
"""
Created on Wed Jan 11 20:11:30 2017

@author: CY_XYZ
"""

# 从sklearn.datasets导入新闻数据抓取器fetch_20newgroups
from sklearn.datasets import fetch_20newsgroups

# 使用fetch_20newsgroups即时从互联网下载数据
news = fetch_20newsgroups(subset='all')
# 查验数据规模和细节
print(len(news.data))
print(news.data[0])
# 从sklearn.cross_validation导入train_test_split
from sklearn.cross_validation import train_test_split

# 随机采样25%作为测试集，其余75%作为训练集
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)
# 再将文本转化为特征向量，利用朴素贝叶斯训练数据中估计参数
# 从sklearn.feature_extraction.text导入文本特征向量转化模块
from sklearn.feature_extraction.text import CountVectorizer

# 初始化文本特征模块
t_vec = CountVectorizer()
t_vec_X_train = t_vec.fit_transform(X_train)
t_vec_X_test = t_vec.transform(X_test)
# 下面一行代码为IPython特有的魔术指令，只能在从之泰单独运行
# %time pow( pi, pi )
# 从sklearn.naive_bayes导入朴素贝叶斯模型
from sklearn.naive_bayes import MultinomialNB

# 使用默认配置初始化贝叶斯模型
mnb = MultinomialNB()
# 利用训练数据对模型参数进行估计
mnb.fit(t_vec_X_train, y_train)
# 对测试样本进行类别预测
mnb_y_predict = mnb.predict(t_vec_X_test)
# 使用准确性、召回率、F1指标、精确率共4个指标评价朴素贝叶斯模型
# 从sklearn.metrics导入classification_report用于详细分类性能评价
from sklearn.metrics import classification_report

print('The accurary of NaiveBayesClassifier is:', mnb.score(t_vec_X_test, y_test))
print(classification_report(y_test, mnb_y_predict, target_names=news.target_names))
'''
运行结果：
The accurary of NaiveBayesClassifier is: 0.839770797963
                          precision    recall  f1-score   support

             alt.atheism       0.86      0.86      0.86       201
           comp.graphics       0.59      0.86      0.70       250
 comp.os.ms-windows.misc       0.89      0.10      0.17       248
comp.sys.ibm.pc.hardware       0.60      0.88      0.72       240
   comp.sys.mac.hardware       0.93      0.78      0.85       242
          comp.windows.x       0.82      0.84      0.83       263
            misc.forsale       0.91      0.70      0.79       257
               rec.autos       0.89      0.89      0.89       238
         rec.motorcycles       0.98      0.92      0.95       276
      rec.sport.baseball       0.98      0.91      0.95       251
        rec.sport.hockey       0.93      0.99      0.96       233
               sci.crypt       0.86      0.98      0.91       238
         sci.electronics       0.85      0.88      0.86       249
                 sci.med       0.92      0.94      0.93       245
               sci.space       0.89      0.96      0.92       221
  soc.religion.christian       0.78      0.96      0.86       232
      talk.politics.guns       0.88      0.96      0.92       251
   talk.politics.mideast       0.90      0.98      0.94       231
      talk.politics.misc       0.79      0.89      0.84       188
      talk.religion.misc       0.93      0.44      0.60       158

             avg / total       0.86      0.84      0.82      4712
'''
