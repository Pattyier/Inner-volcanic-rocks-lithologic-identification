# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 20:43:14 2023

@author: fenghuan12
"""


import numpy as np
import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder #用于label编码
from sklearn.model_selection import train_test_split  #数据集的分割函数
from sklearn.preprocessing import StandardScaler      #数据预处理

#引入包含数据验证方法的包
from sklearn import metrics

class SingeHiddenLayer(object):
    
    def __init__(self,X,y,num_hidden):
        self.data_x = np.atleast_2d(X)       #判断输入训练集是否大于等于二维; 把x_train()取下来
        self.data_y = np.array(y).flatten()  #a.flatten()把a放在一维数组中，不写参数默认是“C”，也就是先行后列的方式，也有“F”先列后行的方式； 把 y_train取下来
        self.num_data = len(self.data_x)  #训练数据个数
        self.num_feature = self.data_x.shape[1];  #shape[] 读取矩阵的长度，比如shape[0]就是读取矩阵第一维度的长度 (120行，4列，所以shape[0]==120,shapep[1]==4)
        self.num_hidden = num_hidden;  #隐藏层节点个数
        
        #随机生产权重（从-1，到1，生成（num_feature行,num_hidden列））
        self.w = np.random.uniform(-1, 1, (self.num_feature, self.num_hidden)) 
        
        #随机生成偏置，一个隐藏层节点对应一个偏置
        for i in range(self.num_hidden):
            b = np.random.uniform(-0.6, 0.6, (1, self.num_hidden))
            self.first_b = b
   
        #生成偏置矩阵，以隐藏层节点个数4为行，样本数120为列
        for i in range(self.num_data-1):
            b = np.row_stack((b, self.first_b))  #row_stack 以叠加行的方式填充数组
        self.b = b
    #定义sigmoid函数  
    def sigmoid(self,x):
        return 1.0 / (1 + np.exp(-x))
    
    #定义softmax函数
    def softmax(self,x):
        return np.exp(x)/np.sum(np.exp(x), axis=0)
    
    #定义softplus函数
    def softplus(self,x):
        return np.log(1 + np.exp(x))
    
    #定义tanh函数
    def tanh(self,x):
        return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
    
        
    def train(self,x_train,y_train,classes):
        mul = np.dot(self.data_x, self.w)    #输入乘以权重 
        add = mul + self.b                        #加偏置
        H = self.softmax(add)                #激活函数

        H_ = np.linalg.pinv(H)               #求广义逆矩阵
        #print(type(H_.shape))
        
        #将只有一列的Label矩阵转换，Lithology_data的label中共有七个值，则转换为七列，以行为单位，label值对应位置标记为1，其它位置标记为0
        self.train_y = np.zeros((self.num_data,classes))  #初始化一个xx行，7列的全0矩阵
        for i in range(0,self.num_data):
            self.train_y[i,y_train[i]] = 1   #对应位置标记为1
        
        self.out_w = np.dot(H_,self.train_y)  #求输出权重
        
    def predict(self,x_test):
        self.t_data = np.atleast_2d(x_test)    #测试数据集
        self.num_tdata = len(self.t_data)      #测试集的样本数
        self.pred_Y = np.zeros((x_test.shape[0]))  #初始化
        
        b = self.first_b
        
        #扩充偏置矩阵，以隐藏层节点个数4为行，样本数30为列
        for i in range(self.num_tdata-1):
            b = np.row_stack((b, self.first_b))  #以叠加行的方式填充数组
          
         #预测  
        self.pred_Y = np.dot(self.softmax(np.dot(self.t_data,self.w)+b),self.out_w)
       
        
        #取输出节点中值最大的类别作为预测值 
        self.predy = []
        for i in self.pred_Y:
            L = i.tolist()
            self.predy.append(L.index(max(L)))
            
            return self.predy

    def score(self,y_test):
        print("准确率：")
        #使用准确率方法验证
        print(metrics.accuracy_score(y_true=y_test,y_pred=self.predy))
        
#数据读取及划分
data1 = pd.read_csv('D:\\第二篇\\Project of ELM\\Data prepare\\well logging data.csv', sep=',')
Y = data1.Lithology
Y = LabelEncoder().fit_transform(Y)
X_data=data1.iloc[:,2:11]
num_train=0.3
X_train,X_test,Y_train,Y_test=train_test_split(X_data,Y,test_size=num_train,random_state=20)

stdsc = StandardScaler()   #StandardScaler类,利用接口在训练集上计算均值和标准差，以便于在后续的测试集上进行相同的缩放
X_train=stdsc.fit_transform(X_train)
X_test=stdsc.fit_transform(X_test)


#开始记时
start_time = time.time()
#不同隐藏层结果对比
for j in range(1, 500, 50):
    ELM = SingeHiddenLayer(X_train,Y_train, 450)#训练数据集，训练集的label，隐藏层节点个数
    ELM.train(X_train,Y_train,7)
    score = ELM.predict(X_test)
    ELM.score(Y_test)
    print('hidden- %d'%j)

#结束记时
end_time = time.time()

#输出执行时间
execution_time = end_time - start_time
print(execution_time)
