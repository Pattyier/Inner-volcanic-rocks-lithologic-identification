# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 10:26:09 2023

@author: fenghuan12
"""

import numpy as np
from sklearn.preprocessing import LabelEncoder #用于label编码
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split  #数据集的分割函数
from sklearn.preprocessing import StandardScaler      #数据预处理
from sklearn import metrics                           #引入包含数据验证方法的包
import pandas as pd
import matplotlib.pyplot as plt

class RELM_HiddenLayer():

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
        h = self.softmax(np.dot(test_x, self.w) + self.b[:b_row, :])
        result1 = np.dot(h, self.beta)
        #result2 =np.argmax(result1,axis=1)
        return result1
    

# In[]
#数据读取及划分
data1 = pd.read_csv('D:\\桌面\\Project of RELM\\Data prepare\\well logging data.csv', sep=',')
X1 = data1.iloc[:, 2:11]
Y1 = data1.Lithology
num_train = 0.3
x1_train, x1_test, y1_train, y1_test = train_test_split(X1, Y1, test_size=num_train, random_state=20) 
y2_train = np.array(y1_train)
y2_test = np.array(y1_test)
y_train = LabelEncoder().fit_transform(y1_train)
y_test = LabelEncoder().fit_transform(y1_test)
x_train = np.array(x1_train)
x_test = np.array(x1_test)
Y_train = np.asarray(pd.get_dummies(y_train),dtype=np.uint8)  #pd.get_dummies是pandas库中一个函数，用于将一个包含分类变量的DataFrame或Series转换为哑变量矩阵。哑变量矩阵是一种二进制矩阵，用于表示分类变量的取值情况。
Y_test = np.asarray(pd.get_dummies(y_test),dtype=np.uint8)

#数据标准化处理
stdsc = StandardScaler() 
X_train = stdsc.fit_transform(x_train)
X_test = stdsc.fit_transform(x_test)
Y_true=np.argmax(Y_test,axis=1)


# In[]

a = RELM_HiddenLayer(X_train,75)
a.classifisor_train(Y_train)
#y_score = a.classifisor_test(X_test)


y_score = pd.read_csv("D:\桌面\Project of RELM\ROC曲线\y_score test.csv")
y_score = np.array(y_score)

# In[]
n_classes = len(np.unique(Y1))
target_names = ["Altered Andesite", "Andesite", "Basalt", "Tuff", "Tuffaceous Conglomerate", "Tuffaceous Sandstone", "Volcanic Breccia"]
target_names = np.array(target_names)

from sklearn.preprocessing import LabelBinarizer

label_binarizer = LabelBinarizer().fit(y1_test)
y_onehot_test = label_binarizer.transform(y1_test)

b = y_onehot_test.ravel()
c = y_score.ravel().reshape(-1,1)


from sklearn.metrics import auc, roc_curve
# store the fpr, tpr, and roc_auc for all averaging strategies
fpr, tpr, roc_auc = dict(), dict(), dict()
# Compute micro-average ROC curve and ROC area
fpr["micro"], tpr["micro"], _ = roc_curve(y_onehot_test.ravel(),y_score.ravel().reshape(-1,1))
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

print(f"Micro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['micro']:.2f}")

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_onehot_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr_grid = np.linspace(0.0, 1.0, 1000)

# Interpolate all ROC curves at these points
mean_tpr = np.zeros_like(fpr_grid)

for i in range(n_classes):
    mean_tpr += np.interp(fpr_grid, fpr[i], tpr[i])  # linear interpolation

# Average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = fpr_grid
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

print(f"Macro-averaged One-vs-Rest ROC AUC score:\n{roc_auc['macro']:.2f}")



from sklearn.metrics import RocCurveDisplay

from itertools import cycle

from matplotlib.colors import LinearSegmentedColormap

# Create a figure and axis
fig, ax = plt.subplots(figsize=(7, 6), dpi=1000)
# Define a custom colormap with shades of light blue
#colorss = [(100, 149, 237), (255, 255, 255)]  # Light blue to white
#cmap = LinearSegmentedColormap.from_list('custom_cmap', colorss, N=256)

# Create a gradient from light blue to white
#gradient = np.linspace(0, 1, 256)
#gradient = np.vstack((gradient, gradient))

# Show the gradient as the background using the custom colormap
#ax.imshow(gradient, aspect='auto', cmap=cmap, extent=[-0.02, 1.02, -0.02, 1.02])

# Set the background color to white
#ax.set_facecolor('#D6F0FF')

plt.plot(
    fpr["micro"],
    tpr["micro"],
    label=f"micro-average ROC curve (AUC = {roc_auc['micro']:.2f})",
    color="deeppink",
    linestyle=":",
    linewidth=2,
)

plt.plot(
    fpr["macro"],
    tpr["macro"],
    label=f"macro-average ROC curve (AUC = {roc_auc['macro']:.2f})",
    color="navy",
    linestyle=":",
    linewidth=2,
)

#colors = cycle(["#B3242A", "#1B64A4", "#A61C5D", "#4048A4", "#7668BA", "#F47368", "#69A3DD"])
#colors = cycle(["#0d5b26", "#c94733", "#fddf8b", "#3fab47", "#52b9d8", "#2e5fa1", "#e5086a"])
colors = cycle(["#2d2f2f", "#c84948", "#dd8950", "#ead064", "#6bbd75", "#8dc1dc", "#7b7bab"])
for class_id, color in zip(range(n_classes), colors):
    RocCurveDisplay.from_predictions(
        y_onehot_test[:, class_id],
        y_score[:, class_id],
        name=f"ROC curve for {target_names[class_id]}",
        color=color,
        ax=ax,
        linewidth=1.5,
    )
plt.plot([0, 1], [0, 1], "k--", label="Chance level (AUC = 0.5)", linewidth=1)
plt.axis("square")
plt.xlim(-0.02,1.02)
plt.ylim(-0.02,1.02)
plt.tick_params(axis='x', direction='in', which='both')
plt.tick_params(axis='y', direction='in', which='both')
plt.xticks(fontsize=12,fontweight='heavy',family='Arial')
plt.yticks(fontsize=12,fontweight='heavy',family='Arial')
plt.xlabel("False Positive Rate",fontsize=12,fontweight='heavy',family='Arial')
plt.ylabel("True Positive Rate",fontsize=12,fontweight='heavy',family='Arial')
plt.title("Test set One-vs-Rest multiclass ROC curves", fontsize=12,fontweight='heavy',family='Arial')
font = {'family': 'Arial','size':10,"weight":'heavy'}
plt.legend(prop=font, frameon=False)
plt.text(-0.1, 1.08, '(a)', fontsize=12,fontweight='heavy',family='Arial')
plt.show()
