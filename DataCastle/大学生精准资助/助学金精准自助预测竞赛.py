# -*-coding:utf-8 -*-
"""
Created on Wed Jan 11 20:11:30 2017
赛题 
    大数据时代的来临，为创新资助工作方式提供了新的理念和技术支持，
    也为高校利用大数据推进快速、便捷、高效精准资助工作带来了新的机遇。
    基于学生每天产生的一卡通实时数据，利用大数据挖掘与分析技术、
    数学建模理论帮助管理者掌握学生在校期间的真实消费情况、
    学生经济水平、发现“隐性贫困”与疑似“虚假认定”学生，从而实现精准资助，
    让每一笔资助经费得到最大价值的发挥与利用，帮助每一个贫困大学生顺利完成学业。
    因此，基于学生在校期间产生的消费数据运用大数据挖掘与分析技术实现
    贫困学生的精准挖掘具有重要的应用价值。
    教育算法资格赛采用某高校2014、2015两学年的助学金获取情况作为标签，
    2013~2014、2014~2015两学年的学生在校行为数据作为原始数据，
    包括消费数据、图书借阅数据、寝室门禁数据、图书馆门禁数据、
    学生成绩排名数据，并以助学金获取金额作为结果数据进行模型优化和评价。
本次竞赛需利用学生在2013/09~2014/09的数据，预测学生在2014年的助学金获得情况；
利用学生在2014/09~2015/09的数据，预测学生在2015年的助学金获得情况。
虽然所有数据在时间上混合在了一起，即训练集和测试集中的数据都有
2013/09~2015/09的数据，但是学生的行为数据和助学金数据是对应的。
    此竞赛赛程分为两个阶段，以测试集切换为标志，2017年2月13日切换。
数据
*注 : 报名参赛或加入队伍后，可获取数据下载权限。
1）数据总体概述
    数据分为两组，分别是训练集和测试集，每一组都包含大约1万名学生的信息纪录：
    图书借阅数据borrow_train.txt和borrow_test.txt、
    一卡通数据card_train.txt和card_test.txt、
    寝室门禁数据dorm_train.txt和dorm_test.txt、
    图书馆门禁数据library_train.txt和library_test.txt、
    学生成绩数据score_train.txt和score_test.txt
    助学金获奖数据subsidy_train.txt和subsidy_test.txt
    训练集和测试集中的学生id无交集，详细信息如下。注：数据中所有的记录均为
    “原始数据记录”直接经过脱敏而来，可能会存在一些重复的或者是异常的记录，
    请参赛者自行处理。
2）数据详细描述
（1）图书借阅数据borrow*.txt（*代表_train和_test）
    注：有些图书的编号缺失。字段描述和示例如下（第三条记录缺失图书编号）：
    学生id，借阅日期，图书名称，图书编号
    9708,2014/2/25,"我的英语日记/ (韩)南银英著 (韩)卢炫廷插图","H315 502"
    6956,2013/10/27,"解读联想思维: 联想教父柳传志","K825.38=76 547"
    9076,2014/3/28,"公司法 gong si fa = = Corporation law / 范健, 王建文著 eng"
（2）一卡通数据card*.txt
    字段描述和示例如下：
    学生id，消费类别，消费地点，消费方式，消费时间，消费金额，剩余金额
    1006,"POS消费","地点551","淋浴","2013/09/01 00:00:32","0.5","124.9"
    1406,"POS消费","地点78","其他","2013/09/01 00:00:40","0.6","373.82"
    13554,"POS消费","地点6","淋浴","2013/09/01 00:00:57","0.5","522.37"
（3）寝室门禁数据dorm*.txt
    字段描述和示例如下：
    学生id，具体时间，进出方向(0进寝室，1出寝室) 
    13126,"2014/01/21 03:31:11","1"
    9228,"2014/01/21 10:28:23","0"
（4）图书馆门禁数据library*.txt
    图书馆的开放时间为早上7点到晚上22点，门禁编号数据在2014/02/23之前只有“编号”信息，
    之后引入了“进门、出门”信息，还有些异常信息为null，请参赛者自行处理。
    字段描述和示例如下：
    学生id，门禁编号，具体时间
    3684,"5","2013/09/01 08:42:50"
    7434,"5","2013/09/01 08:50:08"
    8000,"进门2","2014/03/31 18:20:31"
    5332,"小门","2014/04/03 20:11:06"
    7397,"出门4","2014/09/04 16:50:51"
（5）学生成绩数据score*.txt。
    注：成绩排名的计算方式是将所有成绩按学分加权求和，然后除以学分总和，
    再按照学生所在学院排序。
    学生id,学院编号,成绩排名
    0,9,1
    1,9,2
    8,6,1565
    9,6,1570
（6）助学金数据（训练集中有金额，测试集中无金额）subsidy*.txt
    字段描述和示例如下：
    学生id,助学金金额（分隔符为半角逗号）
    10,0
    22,1000
    28,1000
    64,1500
    650,2000
@author: CY_XYZ
"""

# 导入科学计算工具numpy，自命名为np
import numpy as np
# 导入数据文本读取与存储的工具包pandas，自命名为pd
import pandas as pd
# 从sklearn.ensemble工具包中导入梯度提升分类器GradientBoostingClassifier
from sklearn.ensemble import GradientBoostingClassifier
# 从sklearn.ensemble工具包中导入随机森林分类器模型RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier

##########################################################
####            数据获取、数据清洗、数据特征化
####            训练集数据、测试集数据准备
##########################################################

# train_test
# 读取指定训练集数据文件subsidy_train.txt文件(97KB)
# 得到的训练集数据保存在变量train中，数据规模：DataFrame( 10885, 2 )
# 此处，train数据框的2列名称：'0'，'1'，首列'index'为数据框默认
train = pd.read_table('G:/P_Anaconda3-4.2.0/PA-WORK/' \
                      '助学金精准自助预测竞赛/train/subsidy_train.txt', sep=',', header=-1)
# 将读取到的subsidy_train数据的两列：第一列'id'；第二列'money'
# 此处，train数据框的2列名称：'id'，'money'，首列'index'为数据框默认
train.columns = ['id', 'money']

# 读取指定测试集数据文件studentID_test.txt文件(71KB)
# 得到的测试集数据保存在变量test中，数据规模：DataFrame( 10783, 1 )
# 此处，test数据框的1列名称：'0'，，首列'index'为数据框默认
test = pd.read_table('G:/P_Anaconda3-4.2.0/PA-WORK/' \
                     '助学金精准自助预测竞赛/test/studentID_test.txt', sep=',', header=-1)
# 将读取到的studentID_test数据的唯一一列重命名为'id'
# 此处，test数据框的1列名称：'id'，首列'index'为数据框默认
test.columns = ['id']
# 给studentID_test新增一列：命名为'money'，该列的值全部初始化为nan
# 得到的测试集数据保存在变量test中，数据规模：DataFrame( 10783, 2 )
# 此处，test数据框的1列名称：'id'，'money'，首列'index'为数据框默认
test['money'] = np.nan

# 将上述训练集数据train、测试集数据test合并连接，保存在变量train_test中
# 得到的数据train在前，test在后，数据规模DataFrame( 21668, 2 )
# 此处，train_test数据框的2列名称：'id'，'money'
# 首列'index'为数据框默认，且索引由0~10884再0~10782
train_test = pd.concat([train, test])

# score
'''
（5）学生成绩数据score*.txt。
    注：成绩排名的计算方式是将所有成绩按学分加权求和，然后除以学分总和，
    再按照学生所在学院排序。
    学生id,学院编号,成绩排名
    0,9,1
    1,9,2
    8,6,1565
    9,6,1570
'''

# 读取指定训练数据集score_train.txt文件(111KB)
# 得到的训练集数据保存在变量score_train中，数据规模：DataFrame( 9130, 3 )
# 此处，score_train数据框的3列名称：'0'，'1'，'2'，首列'index'为数据框默认
score_train = pd.read_table('G:/P_Anaconda3-4.2.0/PA-WORK/' \
                            '助学金精准自助预测竞赛/train/score_train.txt', sep=',', header=-1)
# 将score_train的3列重命名为：'id'，'college'，'score'
# 此处，score_train数据框的3列名称：'id'，'college'，'score'
# 首列'index'为数据框默认
score_train.columns = ['id', 'college', 'score']
# 读取指定测试集数据score_test.txt文件(109KB)
# 得到的测试集数据保存在变量score_test中，数据规模：DataFrame( 9000, 3 )
# 此处，score_test数据框的3列名称：'0'，'1'，'2'，首列'index'为数据框默认
score_test = pd.read_table('G:/P_Anaconda3-4.2.0/PA-WORK/' \
                           '助学金精准自助预测竞赛/test/score_test.txt', sep=',', header=-1)
# 将score_test的3列重命名为：'id'，'college'，'score'
# 此处，score_test数据框的3列名称：'id'，'college'，'score'
# 首列'index'为数据框默认
score_test.columns = ['id', 'college', 'score']

# 将上述训练集数据score_train，测试集数据score_test合并连接
# 保存在变量score_train_test中，数据规模：DataFrame( 18130, 3 )
# 此处，score_train_test数据框的3列名称：'id'，'college'，'score'
# 首列'index'为数据框默认，且索引由0~9129再0~8999
score_train_test = pd.concat([score_train, score_test])

# 获取学院编号及各学院的总人数(此处默认：成绩排名最大值为学院总人数)
# 保存在变量college中，数据规模：DataFrame( 19, 1 )
# 此处，college数据框的1列名称：'sore'
# 首列'index'非数据框默认，是'college'充当的
college = pd.DataFrame(score_train_test.groupby(['college'])['score'].max())
# 将学院编号与其对应总人数写入college.csv文件
# college.csv文件的2列名称：'college'，'score'
college.to_csv('G:/P_Anaconda3-4.2.0/PA-WORK/' \
               '助学金精准自助预测竞赛/save_output/college.csv', index=True)
# 再将刚才保存的college.csv的数据读取到college变量中
# 此处，college数据框变为两列：'college'，'score'，数据规模DataFrame( 19, 2 )
# 首列'index'为数据框默认，由0~18
college = pd.read_csv('G:/P_Anaconda3-4.2.0/PA-WORK/' \
                      '助学金精准自助预测竞赛/save_output/college.csv')
# 更改college的列名为'college'，'num'
# 此处，college的2列名称为：'college'，'num'，数据规模DataFrame( 19, 2 )
# 首列'index'为数据框默认，由0~18
college.columns = ['college', 'num']

# 使用pandas.merge合并函数将score_train_test、college有效信息合并起来
# 参数解释：score_train_test在左(前)，college在右(后)
#          how指定合并方式：'left'左合并，'right'右合并，
#                         'inner'内合并，'outer'外合并
# 关于合并的理解需要数据库合并、连接概念(笛卡儿积)的支持
# 参数on指定为'college'，就是选择'college'列中的值，只要'college'值相同
#            就将college值对应的score值合并到score_train_test中
# 执行合并操作前后，score_train_test由3列变为4列：'id'，'college'，'score'，'num'
# 且后两列名称：score(学生成绩排名)，num(学生所在学院总人数)
# 此处，score_train_test变量，数据规模DataFrame( 18130, 4 )
score_train_test = pd.merge(score_train_test, college, how='left', on='college')
# 给score_train_test新增1列：'order'
# 'order'列的值 = 'score'列的值 / 'num'列的值
# 此处，score_train_test变为5列：'id'，'college'，'score'，'num'，'order'
# 数据规模DataFrame( 18130, 5 )，首列'index'为数据框默认，由0~18129
score_train_test['order'] = score_train_test['score'] / score_train_test['num']

# 再将train_test、score_train_test的有效信息合并
# 合并结果保存到变量train_test中
# 执行合并操作前后，train_test由2列变为6列
# 合并过程中，train_test中存在'id'，而score_train_test中不存在，合并结果用nan代替
# 此处，train_test的6列列名：'id'，'money'，'cillege'，'score'，'num'，'order'
# 合并后，train_test数据规模DataFrame( 21668, 6 )
# 首列'index'为数据框默认，由0~21667
train_test = pd.merge(train_test, score_train_test, how='left', on='id')

# card
'''
（2）一卡通数据card*.txt
    字段描述和示例如下：
    学生id，消费类别，消费地点，消费方式，消费时间，消费金额，剩余金额
    1006,"POS消费","地点551","淋浴","2013/09/01 00:00:32","0.5","124.9"
    1406,"POS消费","地点78","其他","2013/09/01 00:00:40","0.6","373.82"
    13554,"POS消费","地点6","淋浴","2013/09/01 00:00:57","0.5","522.37"
'''
# 读取训练集数据card_train.txt文件(900286KB)
# 得到的训练集数据保存在变量card_train中，数据规模DataFrame( 12455558, 7 )
# 此处，card_train数据框的7列名称：'0'，'1'，'2'，'3'，'4'，'5'，'6'
# 首列'index'为数据框默认，由0~12455557
card_train = pd.read_table('G:/P_Anaconda3-4.2.0/PA-WORK/' \
                           '助学金精准自助预测竞赛/train/card_train.txt', sep=',', header=-1)
# 按照数据原始描述，将列名改为：'id'，'consume'，'where'，'how'，
#                           'time'，'amount'，'remainder'
# 分别代表：学生id，消费类别，消费地点，消费时间，消费金额，剩余金额
card_train.columns = ['id', 'consume', 'where', 'how', 'time', 'amount', 'remainder']

# 读取测试集数据card_test.txt文件(895627KB)
# 得到的测试集数据保存在变量card_test中，数据规模DataFrame( 12392844, 7 )
# 此处，card_test数据框的7列名称：'0'，'1'，'2'，'3'，'4'，'5'，'6'
# 首列'index'为数据框默认，由0~12392843
card_test = pd.read_table('G:/P_Anaconda3-4.2.0/PA-WORK/' \
                          '助学金精准自助预测竞赛/test/card_test.txt', sep=',', header=-1)
"""
card_test.columns = [ 'id',
                      'consume',
                      'where',
                      'how',
                      'time',
                      'amount',
                      'remainder' ]
"""
# 同样将card_test的列名更改为:'id'，'consume'，'where'，'how'，
#                           'time'，'amount'，'remainder'
card_test.columns = card_train.columns

# 将训练集数据card_train、测试集数据card_test连接合并
# 保存在card_train_test变量中，数据规模DataFrame( 24848402, 7 )
# 此处，card_train_test的7列名称：'id'，'consume'，'where'，'how'，
#                               'time'，'amount'，'remainder'
# 首列'index'为数据框默认，由0~24848401
card_train_test = pd.concat([card_train, card_test])

# 对card_train_test以'id'列的值分组，统计每个学生消费的总次数
# 得到的数据结果保存在变量card中，数据规模DataFrame( 21631, 1 )
# 此处，card的唯一一列名称：'consume'
# 首列'index'为数据框默认，由0~21630
card = pd.DataFrame(card_train_test.groupby(['id'])['consume'].count())
# 对card_train_test以'id'列的值分组，统计每个学生：消费总额
#                                              消费平均额
#                                              消费最大额
#                                              余额总计
#                                              余额平均值
#                                              余额最大值
# 得到的数据结果添加到变量card新增列'consumesum'、'consumeavg'、'consumemax'
#                               'remaindersum'、'remainderavg'、'remaindermax'
# 此处，card的7列名称：'consume'，'consumesum''consumeavg'、'consumemax'
#                    'remaindersum'、'remainderavg'、'remaindermax'
# 首列'index'非数据框默认，是'id'列充当的
card['consumesum'] = card_train_test.groupby(['id'])['amount'].sum()
card['consumeavg'] = card_train_test.groupby(['id'])['amount'].mean()
card['consumemax'] = card_train_test.groupby(['id'])['amount'].max()
card['remaindersum'] = card_train_test.groupby(['id'])['remainder'].sum()
card['remainderavg'] = card_train_test.groupby(['id'])['remainder'].mean()
card['remaindermax'] = card_train_test.groupby(['id'])['remainder'].max()
# 将card数据结果保存到card.csv文件中
# card.csv文件的8列名称：'id'，'consume'，'consumesum''consumeavg'、'consumemax'
#                      'remaindersum'、'remainderavg'、'remaindermax'
card.to_csv('G:/P_Anaconda3-4.2.0/PA-WORK/' \
            '助学金精准自助预测竞赛/save_output/card.csv', index=True)
# 再将card.csv的8列数据读入card变量中，相当于在原始card还原'id'列，再增添默认索引
# 执行读取操作后，变量card的数据规模变为DataFrame( 21631, 8 )
# 此处，首列'index'为数据框默认，由0~21630
card = pd.read_csv('G:/P_Anaconda3-4.2.0/PA-WORK/' \
                   '助学金精准自助预测竞赛/save_output/card.csv')
# 再将train_test、card的有效信息合并
# 合并结果保存到变量train_test中
# 执行合并操作前后，train_test由6列变为13列
# 合并过程中，train_test中存在'id'，而card中不存在，合并结果用nan代替
# 此处，train_test的13列列名：'id'，'money'，'cillege'，'score'，'num'，'order'
#                           'consume'，'consumesum''consumeavg'、'consumemax'
#                           'remaindersum'、'remainderavg'、'remaindermax'
# 合并后，train_test数据规模DataFrame( 21668, 13 )
# 首列'index'为数据框默认，由0~21667 
train_test = pd.merge(train_test, card, how='left', on='id')

# 前面是将训练集和测试集进行同规化操作，目的是构建训练集、测试集的同规多维特征向量
# 根据初始训练集和测试集的连接合并处，可确定'money'列不为nan，即为训练集；否则，为测试集
# 在train_test里，将训练集与测试集剥离开来，分别保存在变量C_train、C_test中
C_train = train_test[train_test['money'].notnull()]
C_test = train_test[train_test['money'].isnull()]
# 在剥离后的训练集C_train、测试集C_test中，存在许多nan值，用数值-1替换
C_train = C_train.fillna(-1)
C_test = C_test.fillna(-1)

# 'money'列为因变量，设置为target变量
target = 'money'
# 除'money'列外的其他12列的变量均为自变量，设置为predictors变量(组)
predictors = [x for x in C_train.columns if x not in [target]]

# 问题转化为:在训练集中，根据自变量(组)和因变量之间的关系(数据)特征，训练模型参数
#           在测试集中，给定自变量(组)，预测因变量(值)

# C_train

# 在训练集中统计出得到助学金1000、1500、2000的学生的数据信息
Oversampling1000 = C_train.loc[C_train.money == 1000]
Oversampling1500 = C_train.loc[C_train.money == 1500]
Oversampling2000 = C_train.loc[C_train.money == 2000]
# 扩展训练集,CE_train作为C_train的先锋，以此保留下C_train的数据
CE_train = C_train
# 根据Oversampling1000值为741，Oversampling1500值为465，Oversampling2000值为354
# 使用循环冗余扩展训练数据集，使得3类助学金类别的训练样本比例接近1:1:1
for i in range(5):
    CE_train = CE_train.append(Oversampling1000)
for j in range(8):
    CE_train = CE_train.append(Oversampling1500)
for k in range(10):
    CE_train = CE_train.append(Oversampling2000)

##########################################################
####            模型训练及预测
####
##########################################################
# 使用梯度上升模型进行参数训练及预测
gbc = GradientBoostingClassifier(n_estimators=200, random_state=2016)
gbc = gbc.fit(CE_train[predictors], CE_train[target])
gbc_result = gbc.predict(C_test[predictors])
##########################################################

##########################################################
####            输出目标结果及保存结果至文件
####
##########################################################
# 测试结果保存在数据框test_XXX_result中
# 首列为数据框默认'index'，第1列为'studentID'(学生ID)，'subsidy'(助学金金额)
test_gbc_result = pd.DataFrame(columns=["studentID", "subsidy"])
# 将测试集中学生ID填入数据框test_XXX_result的'studentID'(学生ID)列
test_gbc_result.studentID = C_test['id'].values
# 将模型预测结果填入数据框test_XXX_result的'subsidy'(助学金金额)列
test_gbc_result.subsidy = gbc_result
# 将'subdsidy'列float64型转化为int型整数
test_gbc_result.subsidy = test_gbc_result.subsidy.apply(lambda x: int(x))

################
# 输出梯度提升模型结果
print('1000--' + str(len(test_gbc_result[test_gbc_result.subsidy == 1000])) + ':741')
print('1500--' + str(len(test_gbc_result[test_gbc_result.subsidy == 1500])) + ':465')
print('2000--' + str(len(test_gbc_result[test_gbc_result.subsidy == 2000])) + ':354')

test_gbc_result.to_csv('G:/P_Anaconda3-4.2.0/PA-WORK/助学金精准自助预测竞赛/' \
                       'save_output/test_clf_result2016.csv', index=False)
####################################################
######由于两种模型与random设置有关，因而必须分开运行
######
# 使用随机森林模型进行参数训练及预测
rfc = RandomForestClassifier(n_estimators=500, random_state=2016)
rfc = rfc.fit(CE_train[predictors], CE_train[target])
rfc_result = rfc.predict(C_test[predictors])
# 测试结果保存在数据框test_XXX_result中
# 首列为数据框默认'index'，第1列为'studentID'(学生ID)，'subsidy'(助学金金额)
test_rfc_result = pd.DataFrame(columns=["studentID", "subsidy"])
# 将测试集中学生ID填入数据框test_XXX_result的'studentID'(学生ID)列
test_rfc_result.studentID = C_test['id'].values
# 将模型预测结果填入数据框test_XXX_result的'subsidy'(助学金金额)列
test_rfc_result.subsidy = rfc_result
# 将'subdsidy'列float64型转化为int型整数
test_rfc_result.subsidy = test_rfc_result.subsidy.apply(lambda x: int(x))
# 输出随机森林模型结果
print('1000--' + str(len(test_rfc_result[test_rfc_result.subsidy == 1000])) + ':741')
print('1500--' + str(len(test_rfc_result[test_rfc_result.subsidy == 1500])) + ':465')
print('2000--' + str(len(test_rfc_result[test_rfc_result.subsidy == 2000])) + ':354')

test_rfc_result.to_csv('G:/P_Anaconda3-4.2.0/PA-WORK/助学金精准自助预测竞赛/' \
                       'save_output/test_rfc_result2016.csv', index=False)
#####################################################
# 运行结果
"""训练集原始比例1000:1500:2000 = 2.09:1.31:1.00
   梯度上升模型，训练集扩展后，预测比例1000:1500:2000 = 2.58:1.27:1.00
1000--1353:741
1500--666:465
2000--525:354
   梯度上升模型，训练集扩展前，预测比例1000:1500:2000 = 2.82:2.00:1.00
1000--48:741
1500--34:465
2000--17:354
   随机森林模型，训练集扩展后，预测比例1000:1500:2000 = 5.18:1.69:1.00
1000--166:741
1500--54:465
2000--32:354
   随机森林模型，训练集扩展前，预测比例1000:1500:2000 = 9.75:2.75:1.00
1000--39:741
1500--11:465
2000--4:354
"""
