# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 21:02:20 2017
本程序验证的结论：
TfidfVectorizer模型比CountVectorizer模型的特征抽取、特征量化性能更具备优势；
同时，TfidfVectorizer模型比CountVectorizer模型，去除英文停用词比未去用英文词停用词
     性能高出3%~4%
@author: CY_XYZ
"""

#######################################
'''
# 特征抽取与特征量化样例
# 定义一组字典列表，每个字典代表一个数据样本
measurements = [ {'city':'Dubai', 'temperature':33.},
                 {'city':'London', 'temperature':12.},
                 {'city':'San Fransisco', 'temperature':18.} ]

# 从sklearn.feature_extraction导入DictVectorizer
from sklearn.feature_extraction import DictVectorizer
# 初始化DictVectorizer特征抽取器
vec = DictVectorizer(  )
# 输出经DictVectorizer特征抽取后的特征矩阵
print( vec.fit_transform( measurements ).toarray(  ) )
# 输出各个维度的特征含义
print( vec.get_feature_names(  ) )
'''
#######################################

# 从sklearn.datasets导入20类新闻文本数据抓取器
from sklearn.datasets import fetch_20newsgroups

# 从互联网爬下新闻样本，参数条件subset = 'all'表示下载全部近2万条新闻文本
news = fetch_20newsgroups(subset='all')
# 查验数据规模和细节
print(len(news.data))
# print( news.data[ 0 ] )

# 从sklearn.cross_validation导入train_test_split模块分割数据集
from sklearn.cross_validation import train_test_split

# 对news中的数据data进行分割，25%作为测试集，其余75%作为训练集
X_train, X_test, y_train, y_test = train_test_split(news.data, news.target, test_size=0.25, random_state=33)
#############################################################
###################################################
##使用CountVectorizer，并且【不去掉英文停用词】
##对20条新闻文本数据进行量化，并用朴素贝叶斯分类测试性能
###################################################
# 从sklearn.feature_extraction.text导入CountVectorizer模块
from sklearn.feature_extraction.text import CountVectorizer

# 采用默认配置对CountVectorizer进行初始化，默认配置不会去除英文停用词
count_vec = CountVectorizer()
# 只使用词频统计的方式将训练集、测试集的文本数据转化为特征向量
# 训练集：模型名.fit_transform( X_train )
# 测试集：模型名.transform( X_test )
# 这是第4次栽跟头了！！！！
# 栽跟头错误反馈信息：关于测试集y类变量的'维度不匹配'
X_train_count_vec = count_vec.fit_transform(X_train)
X_test_count_vec = count_vec.transform(X_test)
# 从sklearn.naive_bayes导入朴素贝叶斯模块
from sklearn.naive_bayes import MultinomialNB

# 使用默认配置对朴素贝叶斯分类器进行初始化
mnb_count_vec = MultinomialNB()
# 使用朴素贝叶斯分类器对CountVectorizer(未去除停用词)后的训练样本进行参数训练
mnb_count_vec.fit(X_train_count_vec, y_train)
# 输出CountVectorizer模型准确性结果
print('The accuracy of classifying 20newsgroups using NaiveBayes' \
      '(CountVectorizer without filting stopwords):', mnb_count_vec.score(X_test_count_vec, y_test))
# 将分类预测结果保存在变量y_count_vec_predict中
y_count_vec_predict = mnb_count_vec.predict(X_test_count_vec)
# 从sklearn.metrics导入classification_report模块
from sklearn.metrics import classification_report

# 输出更加详细的其他评价分类性能的指标
print(classification_report(y_test, y_count_vec_predict, target_names=news.target_names))
###################################################################
###################################################
##使用TfidfVectorizer，并且【不去掉英文停用词】
##对20条新闻文本数据进行量化，并用朴素贝叶斯分类测试性能
###################################################
# 从sklearn.feature_extrction.text导入TfidfVectorizer模块
from sklearn.feature_extraction.text import TfidfVectorizer

# 采用默认配置对TfidfVectorizer进行初始化，即未去除英文停用词
tfidf_vec = TfidfVectorizer()
# 使用TfdifVectorizer将训练集、测试集文本转化为特征向量
# 请注意训练集、测试集转化所用函数的不同
X_train_tfidf_vec = tfidf_vec.fit_transform(X_train)
X_test_tfidf_vec = tfidf_vec.transform(X_test)
# 依然使用默认配置的朴素贝叶斯分类器，对TfidfVectorizer特征化的方式进行性能评估
# MultinomialNB模块在前述已经导入
mnb_tfidf_vec = MultinomialNB()
# 使用朴素贝叶斯分类器对TfidfVectorizer(未去除停用词)后的训练样本进行参数训练
mnb_tfidf_vec.fit(X_train_tfidf_vec, y_train)
# 输出TfidfVectorizer模型准确性结果
print('The accuracy of classifying 20newsgroups using NaiveBayes' \
      '(TfidfVectorizer without filting stopwords):', mnb_tfidf_vec.score(X_test_tfidf_vec, y_test))
# 将分类预测结果保存在变量y_tfidf_vec_predict中
y_tfidf_vec_predict = mnb_tfidf_vec.predict(X_test_tfidf_vec)
# 在前述已经导入classification_report模块
# 输出更加详细的其他评价分类性能的指标
print(classification_report(y_test, y_tfidf_vec_predict, target_names=news.target_names))
###################################################################
#############################################################
###################################################
##使用CountVectorizer、TfidfVectorizer，并且【去掉英文停用词】
##对20条新闻文本数据进行量化，并用朴素贝叶斯分类测试性能
###################################################
# 继续沿用前述导入的模块、获得的数据
# 使用停用词过滤配置初始化CountVectorizer、TfidfVectorzer模型
count_filter_vec, tfidf_filter_vec = CountVectorizer(analyzer='word', stop_words='english'), TfidfVectorizer(analyzer='word', stop_words='english')

# 使用带有停用词过滤的CountVectorizer、TfidfVectorizer模型对训练集、测试集进行特征化
X_train_count_filter_vec = count_filter_vec.fit_transform(X_train)
X_test_count_filter_vec = count_filter_vec.transform(X_test)

X_train_tfidf_filter_vec = tfidf_filter_vec.fit_transform(X_train)
X_test_tfidf_filter_vec = tfidf_filter_vec.transform(X_test)
# 使用默认配置初始化朴素贝叶斯分类器
mnb_count_filter_vec = MultinomialNB()
# 进行CountVectorizer模型参数训练
mnb_count_filter_vec.fit(X_train_count_filter_vec, y_train)
# 输出CountVectorizer模型准确性结果及详细性能评估
print('The accuracy of classifying 20newsgroups using NaiveBayes' \
      '(CountVectorizer with filting stopwords):', mnb_count_filter_vec.score(X_test_count_filter_vec, y_test))
# 将分类预测结果保存在变量y_count_filter_vec_predict中
y_count_filter_vec_predict = mnb_count_filter_vec.predict(X_test_count_filter_vec)
# 在前述已经导入classification_report模块
# 输出更加详细的其他评价分类性能的指标
print(classification_report(y_test, y_count_filter_vec_predict, target_names=news.target_names))
##############
# 再使用默认配置初始化朴素贝叶斯分类器
mnb_tfidf_filter_vec = MultinomialNB()
# 进行TfidfVectorizer模型参数训练
mnb_tfidf_filter_vec.fit(X_train_tfidf_filter_vec, y_train)
# 输出TfidfVectorizer模型准确性结果及详细性能评估
print('The accuracy of classifying 20newsgroups using NaiveBayes' \
      '(TfidfVectorizer with filting stopwords):', mnb_tfidf_filter_vec.score(X_test_tfidf_filter_vec, y_test))
# 将分类预测结果保存在变量y_count_filter_vec_predict中
y_tfidf_filter_vec_predict = mnb_tfidf_filter_vec.predict(X_test_tfidf_filter_vec)
# 在前述已经导入classification_report模块
# 输出更加详细的其他评价分类性能的指标
print(classification_report(y_test, y_tfidf_filter_vec_predict, target_names=news.target_names))
####################################################################
'''
运行结果：【单独使用CountVectorizer(未去除英文停用词)】
18846
The accuracy of classifying 20newsgroups using NaiveBayes(CountVectorizer
 without filting stopwords): 0.839770797963

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
指标注释：accuracy：准确率
        precision：平均进度
        recall：召回率
        f1-score：F1指标
             
             
运行结果：【单独使用TfidfVectorizer(未去除英文停用词)】
The accuracy of classifying 20newsgroups using NaiveBayes(TfidfVectorizer
 without filting stopwords): 0.846349745331
                          precision    recall  f1-score   support

             alt.atheism       0.84      0.67      0.75       201
           comp.graphics       0.85      0.74      0.79       250
 comp.os.ms-windows.misc       0.82      0.85      0.83       248
comp.sys.ibm.pc.hardware       0.76      0.88      0.82       240
   comp.sys.mac.hardware       0.94      0.84      0.89       242
          comp.windows.x       0.96      0.84      0.89       263
            misc.forsale       0.93      0.69      0.79       257
               rec.autos       0.84      0.92      0.88       238
         rec.motorcycles       0.98      0.92      0.95       276
      rec.sport.baseball       0.96      0.91      0.94       251
        rec.sport.hockey       0.88      0.99      0.93       233
               sci.crypt       0.73      0.98      0.83       238
         sci.electronics       0.91      0.83      0.87       249
                 sci.med       0.97      0.92      0.95       245
               sci.space       0.89      0.96      0.93       221
  soc.religion.christian       0.51      0.97      0.67       232
      talk.politics.guns       0.83      0.96      0.89       251
   talk.politics.mideast       0.92      0.97      0.95       231
      talk.politics.misc       0.98      0.62      0.76       188
      talk.religion.misc       0.93      0.16      0.28       158

             avg / total       0.87      0.85      0.84      4712

运行结果：【使用CountVectorizer、TfidfVectorizer(去除英文停用词)】
The accuracy of classifying 20newsgroups using NaiveBayes(CountVectorizer
 with filting stopwords): 0.863752122241
                          precision    recall  f1-score   support

             alt.atheism       0.85      0.89      0.87       201
           comp.graphics       0.62      0.88      0.73       250
 comp.os.ms-windows.misc       0.93      0.22      0.36       248
comp.sys.ibm.pc.hardware       0.62      0.88      0.73       240
   comp.sys.mac.hardware       0.93      0.85      0.89       242
          comp.windows.x       0.82      0.85      0.84       263
            misc.forsale       0.90      0.79      0.84       257
               rec.autos       0.91      0.91      0.91       238
         rec.motorcycles       0.98      0.94      0.96       276
      rec.sport.baseball       0.98      0.92      0.95       251
        rec.sport.hockey       0.92      0.99      0.95       233
               sci.crypt       0.91      0.97      0.93       238
         sci.electronics       0.87      0.89      0.88       249
                 sci.med       0.94      0.95      0.95       245
               sci.space       0.91      0.96      0.93       221
  soc.religion.christian       0.87      0.94      0.90       232
      talk.politics.guns       0.89      0.96      0.93       251
   talk.politics.mideast       0.95      0.98      0.97       231
      talk.politics.misc       0.84      0.90      0.87       188
      talk.religion.misc       0.91      0.53      0.67       158

             avg / total       0.88      0.86      0.85      4712

The accuracy of classifying 20newsgroups using NaiveBayes(TfidfVectorizer
 with filting stopwords): 0.882640067912
                          precision    recall  f1-score   support

             alt.atheism       0.86      0.81      0.83       201
           comp.graphics       0.85      0.81      0.83       250
 comp.os.ms-windows.misc       0.84      0.87      0.86       248
comp.sys.ibm.pc.hardware       0.78      0.88      0.83       240
   comp.sys.mac.hardware       0.92      0.90      0.91       242
          comp.windows.x       0.95      0.88      0.91       263
            misc.forsale       0.90      0.80      0.85       257
               rec.autos       0.89      0.92      0.90       238
         rec.motorcycles       0.98      0.94      0.96       276
      rec.sport.baseball       0.97      0.93      0.95       251
        rec.sport.hockey       0.88      0.99      0.93       233
               sci.crypt       0.85      0.98      0.91       238
         sci.electronics       0.93      0.86      0.89       249
                 sci.med       0.96      0.93      0.95       245
               sci.space       0.90      0.97      0.93       221
  soc.religion.christian       0.70      0.96      0.81       232
      talk.politics.guns       0.84      0.98      0.90       251
   talk.politics.mideast       0.92      0.99      0.95       231
      talk.politics.misc       0.97      0.74      0.84       188
      talk.religion.misc       0.96      0.29      0.45       158

             avg / total       0.89      0.88      0.88      4712

'''
