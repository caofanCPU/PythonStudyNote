# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 20:05:32 2017

@author: CY_XYZ
"""

##############################################################
# 使用词袋法(Bag-of-Words)对示例文本进行特征向量化
sent1 = 'The cat is walking in the bedroom'
sent2 = 'A dog was running across the kitchen'
# 从sklearn.feature_extraction.text导入CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer

# 初始化CountVwctorizer
count_vec = CountVectorizer()
sentences = [sent1, sent2]
# 输出特征向量化的表示
print(count_vec.fit_transform(sentences).toarray())
# 输出特征向量各个维度的特征含义
print(count_vec.get_feature_names())

# 使用NLTK(自然语言处理工具包)对示例文本进行语言学分析
import nltk

# 对示例句子进行词汇分割和正规化，有些情况如aren't需要分割为are和n't；
#                                    或者I'm要分割为I和'm
tokens_sent1 = nltk.word_tokenize(sent1)
tokens_sent2 = nltk.word_tokenize(sent2)
# 输出经nltk.word_tokenize(  )方法词汇分割呵呵正规化的结果
print(tokens_sent1)
print(tokens_sent2)
# 整理两句的词表，按照ASCII码顺序输出
vocab_sent1 = sorted(tokens_sent1)
vocab_sent2 = sorted(tokens_sent2)
print(vocab_sent1)
print(vocab_sent2)

# 初始化stemmer寻找各个词汇最原始的词根
stemmer = nltk.stem.PorterStemmer()
stemmer_sent1 = [stemmer.stem(t) for t in tokens_sent1]
stemmer_sent2 = [stemmer.stem(t) for t in tokens_sent2]
print(stemmer_sent1)
print(stemmer_sent2)

# 初始化词性标注器，对每个词汇进行标注
pos_tag_sent1 = nltk.pos_tag(tokens_sent1)
pos_tag_sent2 = nltk.pos_tag(tokens_sent2)
print(pos_tag_sent1)
print(pos_tag_sent2)

'''
运行结果：
[[0 1 1 ..., 2 1 0]
 [1 0 0 ..., 1 0 1]]
['across', 'bedroom', 'cat', 'dog', 'in', 'is', 'kitchen', 'running',
 'the', 'walking', 'was']
-------------------------------------------------------------------
['The', 'cat', 'is', 'walking', 'in', 'the', 'bedroom']
['A', 'dog', 'was', 'running', 'across', 'the', 'kitchen']
['The', 'bedroom', 'cat', 'in', 'is', 'the', 'walking']
['A', 'across', 'dog', 'kitchen', 'running', 'the', 'was']
['The', 'cat', 'is', 'walk', 'in', 'the', 'bedroom']
['A', 'dog', 'wa', 'run', 'across', 'the', 'kitchen']
------------------------------------------------------------------
[('The', 'DT'), ('cat', 'NN'), ('is', 'VBZ'), ('walking', 'VBG'),
 ('in', 'IN'), ('the', 'DT'), ('bedroom', 'NN')]
[('A', 'DT'), ('dog', 'NN'), ('was', 'VBD'), ('running', 'VBG'),
 ('across', 'IN'), ('the', 'DT'), ('kitchen', 'NN')]
'''
