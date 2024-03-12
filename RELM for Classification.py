# -*- coding: utf-8 -*-
"""
Created on Tue Aug 29 15:03:07 2023

@author: fenghuan12
"""


import numpy as np
from sklearn.preprocessing import LabelEncoder #用于label编码
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split  #数据集的分割函数
from sklearn.preprocessing import StandardScaler      #数据预处理
from sklearn import metrics                           #引入包含数据验证方法的包
import pandas as pd
import time
from sklearn.utils import shuffle


class RELM_HiddenLayer:

    """
        正则化的极限学习机
        :param x: 初始化学习机时的训练集属性X
        :param num: 学习机隐层节点数
        :param C: 正则化系数的倒数
    """

    def __init__(self, x, num, C=10):                 #x:输入矩阵    num:隐含层神经网络
        row = x.shape[0]
        columns = x.shape[1]
        rnd = np.random.RandomState()                 #random.RandomState用于生成随机数
        # 权重w
        self.w = rnd.uniform(-1, 1, (columns, num))   
        # 偏置b
        self.b = np.zeros([row, num], dtype=float)    #随机设定隐含层神经元阈值，即bi的值 
        for i in range(num):
            rand_b = rnd.uniform(-0.4, 0.4)            #随机产生-0.4到0.4之间的数
            for j in range(row):
                self.b[j, i] = rand_b                   #设定输入层与隐含层的连接权值
        self.H0 = np.matrix(self.softmax(np.dot(x, self.w) + self.b))
        self.C = C
        self.P = (self.H0.H * self.H0 + len(x) / self.C).I 
        #.T:转置矩阵,.H:共轭转置,.I:逆矩阵

    @staticmethod
    def sigmoid(x):
        """
            激活函数sigmoid
            :param x: 训练集中的X
            :return: 激活值
        """
        return 1.0 / (1 + np.exp(-x))

    @staticmethod
    def softplus(x):
        """
            激活函数 softplus
            :param x: 训练集中的X
            :return: 激活值
        """
        return np.log(1 + np.exp(x))

    @staticmethod
    def tanh(x):
        """
            激活函数tanh
            :param x: 训练集中的X
            :return: 激活值
        """
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))
    
    @staticmethod
    def softmax(x):
        """
           激活函数 softmax
           :param x: 训练集中的x
           :return：激活值
        """
        return np.exp(x)/np.sum(np.exp(x), axis=0)

    # 分类问题 训练
    def classifisor_train(self, T):
        """
            初始化了学习机后需要传入对应标签T
            :param T: 对应属性X的标签T
            :return: 隐层输出权值beta
        """
        if len(T.shape) > 1:
            pass
        else:
            self.en_one = OneHotEncoder()
            T = self.en_one.fit_transform(T.reshape(-1, 1)).toarray() #独热编码之后一定要用toarray()转换成正常的数组
            pass
        all_m = np.dot(self.P, self.H0.H)
        #print(T.shape)
        #print(all_m.shape)
        self.beta = np.dot(all_m, T)
        #print(self.beta.shape)
        return self.beta

    # 分类问题 测试
    def classifisor_test(self, test_x):
        """
            传入待预测的属性X并进行预测获得预测值
            :param test_x:被预测标签的属性X
            :return: 被预测标签的预测值T
        """
        b_row = test_x.shape[0]  #shape[0]读取矩阵第一维度的长度，即数组的行数
        h = self.softplus(np.dot(test_x, self.w) + self.b[:b_row, :])
        result1 = np.dot(h, self.beta)
        result2 =np.argmax(result1,axis=1)
        return result1, result2
    
    # 分类问题 测试
    def predict_train(self, train_x):
        """
            传入待预测的属性X并进行预测获得预测值
            :param test_x:被预测标签的属性X
            :return: 被预测标签的预测值T
        """
        b_row = train_x.shape[0]  #shape[0]读取矩阵第一维度的长度，即数组的行数
        h = self.softmax(np.dot(train_x, self.w) + self.b[:b_row, :])
        result1 = np.dot(h, self.beta)
        result2 =np.argmax(result1,axis=1)
        return result1, result2
# In[]
#数据读取及划分
data1 = pd.read_csv('D:\\桌面\\Project of RELM\\Data prepare\\well logging data.csv', sep=',')
Y = data1.Lithology
Y = LabelEncoder().fit_transform(Y)
data2=np.array(data1)
#data=shuffle(data2)
X_data=data2[:,2:11]
Y=np.array(Y)
labels=np.asarray(pd.get_dummies(Y),dtype=np.uint8)  #pd.get_dummies是pandas库中一个函数，用于将一个包含分类变量的DataFrame或Series转换为哑变量矩阵。哑变量矩阵是一种二进制矩阵，用于表示分类变量的取值情况。

num_train=0.3
X_train,X_test,Y_train,Y_test=train_test_split(X_data,labels,test_size=num_train,random_state=20)

# In[]
#数据标准化处理
stdsc = StandardScaler() 
X_train=stdsc.fit_transform(X_train)
X_test=stdsc.fit_transform(X_test)
Y_true_test=np.argmax(Y_test,axis=1)
Y_true_train=np.argmax(Y_train,axis=1)

#开始记时
start_time = time.time()
# In[]

#开始记时
start_time = time.time()
'''
#不同隐藏层结果对比
result=[] 
for j in range(1,500,50):
    a = RELM_HiddenLayer(X_train,j)
    a.classifisor_train(Y_train)
    result1, result2 = a.classifisor_test(X_test)
    acc=metrics.accuracy_score(Y_true_test,result2)
    #pre=metrics.recall_score(Y_true,result2, average='macro')
    #result.append(pre)
    print('hidden- %d,acc：%f'%(j,acc))
'''

result=[] 
for j in range(1,500,50):
    a = RELM_HiddenLayer(X_train,j)
    a.classifisor_train(Y_train)
    result1, result2 = a.predict_train(X_train)
    acc=metrics.accuracy_score(Y_true_train,result2)
    #pre=metrics.recall_score(Y_train, result2, average='macro')
    #result.append(pre)
    print('hidden- %d,acc：%f'%(j,acc))   
    
    
#结束记时
end_time = time.time()

#输出执行时间
execution_time = end_time - start_time
print(execution_time)

