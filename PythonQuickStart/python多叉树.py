# -*- coding: utf-8 -*-
"""
Created on Sat Feb 11 20:36:19 2017

@author: CY_XYZ
"""


# ============================================================
# =======  节点类，构造节点，主要属性：节点值，子结点值列表，父节点值
# =======                  主要方法：添加子节点，获取子结点
# ============================================================
class Node:
    """节点类"""

    # 初始化：节点的数据值属性self._data
    #        节点的子结点属性self._children，初始化为空
    #        节点的父节点属性self._parent，初始化为空
    def __init__(self, data):
        self._data = data
        self._children = []
        self._parent = []

    # 获取节点的值
    def getdata(self):
        return self._data

    # 获取节点的子结点
    def getchildren(self):
        return self._children

    # 在当前节点添加子节点，添加顺序为从左至右
    def add(self, node):
        self._children.append(node)
        node._parent.append(self)


# =======================================================
# =======  字典数据结构，存储多叉树的全部节点
# =======
# =======================================================
nodeDic = {'node_a': Node('A'), 'node_b': Node('B'), 'node_c': Node('C'), 'node_d': Node('D'), 'node_e': Node('E'), 'node_f': Node('F'), 'node_g': Node('G'), 'node_h': Node('H'), 'node_i': Node('I'), 'node_j': Node('J'), 'node_k': Node('K'), 'node_l': Node('L'), 'node_n': Node('N'), 'node_o': Node('O'), 'node_p': Node('P'), 'node_q': Node('Q'), 'node_r': Node('R'), 'node_s': Node('S'), 'node_t': Node('T')}

# =======================================================
# =======  构建一颗多叉树样例
# =======  输出多叉树的形式：节点及其子节点，父节点
# =======================================================
# 指定根结点'A'的父节点为其自身
nodeDic['node_a']._parent.append(nodeDic['node_a'])
# 输出结点'A'的值
print(nodeDic['node_a']._data)
# 添加结点'A'的子节点'B'、'C'、'D'
nodeDic['node_a'].add(nodeDic['node_b'])
nodeDic['node_a'].add(nodeDic['node_c'])
nodeDic['node_a'].add(nodeDic['node_d'])
# 输出节点'A'的子结点
print([childrenList._data for childrenList in nodeDic['node_a']._children])
# 输出节点'A'的父节点的值
print(nodeDic['node_a']._parent[0]._data)
print('------------------')
# 对节点'B'、'D'、'F'、'H'、'N'进行同样操作
print(nodeDic['node_b']._data)
nodeDic['node_b'].add(nodeDic['node_e'])
nodeDic['node_b'].add(nodeDic['node_f'])
nodeDic['node_b'].add(nodeDic['node_g'])
print([childrenList._data for childrenList in nodeDic['node_b']._children])
print(nodeDic['node_b']._parent[0]._data)
print('------------------')
print(nodeDic['node_d']._data)
nodeDic['node_d'].add(nodeDic['node_h'])
nodeDic['node_d'].add(nodeDic['node_i'])
nodeDic['node_d'].add(nodeDic['node_j'])
print([childrenList._data for childrenList in nodeDic['node_d']._children])
print(nodeDic['node_d']._parent[0]._data)
print('------------------')
print(nodeDic['node_f']._data)
nodeDic['node_f'].add(nodeDic['node_k'])
nodeDic['node_f'].add(nodeDic['node_l'])
nodeDic['node_f'].add(nodeDic['node_n'])
print([childrenList._data for childrenList in nodeDic['node_f']._children])
print(nodeDic['node_f']._parent[0]._data)
print('------------------')
print(nodeDic['node_h']._data)
nodeDic['node_h'].add(nodeDic['node_o'])
nodeDic['node_h'].add(nodeDic['node_p'])
nodeDic['node_h'].add(nodeDic['node_q'])
print([childrenList._data for childrenList in nodeDic['node_h']._children])
print(nodeDic['node_h']._parent[0]._data)
print('------------------')
print(nodeDic['node_n']._data)
nodeDic['node_n'].add(nodeDic['node_r'])
nodeDic['node_n'].add(nodeDic['node_s'])
nodeDic['node_n'].add(nodeDic['node_t'])
print([childrenList._data for childrenList in nodeDic['node_n']._children])
print(nodeDic['node_n']._parent[0]._data)
print('------------------')

# =======================================================
# =======  输入模块，起始节点nodeStart，终止节点nodeEnd
# =======           使用while循环对用户输入进行一定的容错处理
# =======================================================
# 输入起始节点nodeStart和终止节点nodeEnd
flag = 1
while flag:
    print('请输入起始节点编号，输入范围a~t,A~T，不包括m和M：')
    nodeStart = input().lower()
    print('请输入终止节点编号，输入范围a~t,A~T，不包括m和M：')
    nodeEnd = input().lower()
    if nodeStart == nodeEnd:
        print('起始节点与终止节点相同，最短路径为0！')
        print('请尝试起始节点不同于终止节点的情况')
    else:
        flag = 0
print('==============================')
# 对用户输入的正确节点进行多叉树字典键的匹配处理
nodeStart = 'node_' + nodeStart
nodeEnd = 'node_' + nodeEnd
startPath = [nodeStart[5].upper()]
endPath = [nodeEnd[5].upper()]


# =======================================================
# =======  递归搜索模块，保存起始节点到根结点的路径startPath；
# =======              保存终止节点到根结点的路径endPath；
# =======  显然startPath、endPath之间存在【重复路径】       
# ======================================================= 
def search2Root(nodeDic, node, path):
    # 递归结束条件：搜索到根结点'A'
    if nodeDic[node]._parent[0]._data == 'A':
        path.append('A')
        print('node_a')
        # print( path )
        return
    else:
        # 先把当前节点保存至搜索路径path中
        path.append(nodeDic[node]._parent[0]._data)
        # 再移至其父节点，进行递归搜索
        node = 'node_' + nodeDic[node]._parent[0]._data.lower()
        print(node)
        search2Root(nodeDic, node, path)


# =======================================================
# =======  输出模块，去除递归搜索模块中的重复路径
# =======  并对根结点'A'的父节点为其自身的假设作善后处理        
# ======================================================= 
print('节点：' + nodeStart)
print('经过如下节点到达根节点：')
search2Root(nodeDic, nodeStart, startPath)
print('节点：' + nodeEnd)
print('经过如下节点到达根节点：')
search2Root(nodeDic, nodeEnd, endPath)
print('起始节点全搜索路径：')
print(startPath)
print('终止节点全搜索路径')
print(endPath)
# 整合路径，输出结果
sPath = startPath
ePath = endPath
# 将列表元素顺序颠倒过来
sPath.reverse()
ePath.reverse()
L = min(len(sPath), len(ePath))
# 在输入模块，限制起始节点和终止节点不相同
# 从而，消除重复路径只需要考虑以下情况
for i in range(L):
    if i < L - 1:
        if (sPath[i] == ePath[i]) and (sPath[i + 1] != ePath[i + 1]):
            shortRoot = i
shortPath = sPath[shortRoot:]
shortPath.reverse()
for i in ePath[(shortRoot + 1):]:
    shortPath.append(i)
# 对于起始节点、终止节点中包含根结点的情况，应剔除重复元素
realPath = []
[realPath.append(i) for i in shortPath if not i in realPath]

# 输出节点间最短路径
print("最短路径：")
print(realPath)

print("------目标最短路径为------")
print('\"', end='')
for i in realPath:
    if i != realPath[len(realPath) - 1]:
        print(i + '->', end='')
    else:
        print(i + '\"', end='')

"""
测试运行结果：
runfile('G:/P_Anaconda3-4.2.0/PA-WORK/python多叉树.py', wdir='G:/P_Anaconda3-4.2.0/PA-WORK')
A
['B', 'C', 'D']
A
------------------
B
['E', 'F', 'G']
A
------------------
D
['H', 'I', 'J']
A
------------------
F
['K', 'L', 'N']
B
------------------
H
['O', 'P', 'Q']
D
------------------
N
['R', 'S', 'T']
F
------------------
请输入起始节点编号，输入范围a~t,A~T，不包括m和M：

G
请输入终止节点编号，输入范围a~t,A~T，不包括m和M：

R
==============================
节点：node_g
经过如下节点到达根节点：
node_b
node_a
节点：node_r
经过如下节点到达根节点：
node_n
node_f
node_b
node_a
起始节点全搜索路径：
['G', 'B', 'A']
终止节点全搜索路径
['R', 'N', 'F', 'B', 'A']
最短路径：
['G', 'B', 'F', 'N', 'R']
------目标最短路径为------
"G->B->F->N->R"

runfile('G:/P_Anaconda3-4.2.0/PA-WORK/python多叉树.py', wdir='G:/P_Anaconda3-4.2.0/PA-WORK')
A
['B', 'C', 'D']
A
------------------
B
['E', 'F', 'G']
A
------------------
D
['H', 'I', 'J']
A
------------------
F
['K', 'L', 'N']
B
------------------
H
['O', 'P', 'Q']
D
------------------
N
['R', 'S', 'T']
F
------------------
请输入起始节点编号，输入范围a~t,A~T，不包括m和M：

A
请输入终止节点编号，输入范围a~t,A~T，不包括m和M：

H
==============================
节点：node_a
经过如下节点到达根节点：
node_a
节点：node_h
经过如下节点到达根节点：
node_d
node_a
起始节点全搜索路径：
['A', 'A']
终止节点全搜索路径
['H', 'D', 'A']
最短路径：
['A', 'D', 'H']
------目标最短路径为------
"A->D->H"

runfile('G:/P_Anaconda3-4.2.0/PA-WORK/python多叉树.py', wdir='G:/P_Anaconda3-4.2.0/PA-WORK')
A
['B', 'C', 'D']
A
------------------
B
['E', 'F', 'G']
A
------------------
D
['H', 'I', 'J']
A
------------------
F
['K', 'L', 'N']
B
------------------
H
['O', 'P', 'Q']
D
------------------
N
['R', 'S', 'T']
F
------------------
请输入起始节点编号，输入范围a~t,A~T，不包括m和M：

H
请输入终止节点编号，输入范围a~t,A~T，不包括m和M：

A
==============================
节点：node_h
经过如下节点到达根节点：
node_d
node_a
节点：node_a
经过如下节点到达根节点：
node_a
起始节点全搜索路径：
['H', 'D', 'A']
终止节点全搜索路径
['A', 'A']
最短路径：
['H', 'D', 'A']
------目标最短路径为------
"H->D->A"

"""
