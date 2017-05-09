# -*- coding: utf-8 -*-
"""
Created on Wed Jan 18 21:29:30 2017

@author: CY_XYZ
"""

# 从sklearn.datasets导入20类新闻文本抓取器fetch_20newsgroups
from sklearn.datasets import fetch_20newsgroups
# 使用fetch_20newsgroups抓取器从互联网上爬下18846条新闻数据
# news为数据集对象，新闻文本数据在news.data列表对象中
#                 新闻文本对应的分类标签在news.target数组对象中
news = fetch_20newsgroups( subset = 'all' )
# print( len( news.data ) )
# 从news中获得【新闻文本数据列表】及对应的【新闻类别标签】，分别保存至变量X，y中
X, y = news.data, news.target
# 从bs导入BeautifulSoup文本处理工具包
from bs4 import BeautifulSoup
# 导入自然语言处理工具包nltk，正则表达式工具包re
import nltk, re
# 定义函数：news_to_sentences，将每条新闻中的句子逐一剥离出来，并返回一个句子的列表
# news_to_sentences的输入参数为一条新闻文本【全文】
#                    输出结果为该条新闻文本的所有句子
def news_to_sentences( news ):
    # news_text为单条新闻文本数据的【文本内容】
    news_text = BeautifulSoup( news ).get_text(  )
    # tokenizer为英文文本处理工具对象
    tokenizer = nltk.data.load( 'tokenizers/punkt/english.pickle' )
    # 对news_text进行词汇分割，结果保存在raw_sentences中
    raw_sentences = tokenizer.tokenize( news_text )
    # 此处是函数内的局部【临时】变量sentences，每次都初始化为[]
    sentences = []
    # 正则表达式，处理？？？？？？
    for sent in raw_sentences:
        sentences.append( re.sub( '[^a-zA-Z]',
                                  ' ',
                                  sent.lower(  ).strip(  ) )\
                                                            .split(  ) )
    # 返回局部变量sentences的内容
    return sentences

# 此处是全局变量sentences，初始化为空
sentences = []
# 使用for循环迭代，依次对X中的18846条新闻文本，依次转化为句子
for x in X:
    # 以添加方式保存在全局变量sentences中，调用news_to_sentences函数
    sentences += news_to_sentences( x )

# 配置词向量的维度
num_features = 300
# 保证被考虑的词汇频度
min_word_count = 20
# 设定并行化训练使用CPU计算核心的数量，多核使用
num_workers = 2
# 定义并行化训练词向量上下文窗口大小
context_window = 5
# 
downsampling = 1e-3
# 从gensim.models导入word2vec
from gensim.models import word2vec
# 训练词向量模型
model = word2vec.Word2Vec( sentences,
                           workers = num_workers,
                           size = num_features,
                           min_count = min_word_count,
                           window = context_window,
                           sample = downsampling )
# 设定代表当前训练好的词向量为最终版，也可以加快模型的训练速度
model.init_sims( replace = True )
# 利用训练好的模型，寻找文本中与moning最相关的10个词汇并输出
print( model.most_similar( 'morning' ) )
print( model.most_similar( 'email' ) )

'''
运行结果：
[('afternoon', 0.8327237963676453),
 ('weekend', 0.7910783290863037),
 ('saturday', 0.7617961764335632),
 ('evening', 0.7528278827667236),
 ('night', 0.7443951368331909),
 ('friday', 0.7291523218154907),
 ('sunday', 0.7074136734008789),
 ('newspaper', 0.6682775020599365),
 ('tuesday', 0.6414511203765869),
 ('monday', 0.6391574740409851)]
[('mail', 0.7408841252326965),
 ('contact', 0.7049922943115234),
 ('replies', 0.6903711557388306),
 ('address', 0.6560025215148926),
 ('compuserve', 0.6533834934234619),
 ('mailed', 0.6466802358627319),
 ('request', 0.6310352683067322),
 ('sas', 0.6255717277526855),
 ('subscription', 0.6158285737037659),
 ('send', 0.6069809198379517)]
'''
