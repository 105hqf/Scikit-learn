#!/usr/bin/python
#-*- coding: UTF-8 -*-
import sklearn.datasets as sk_datasets
from sklearn.model_selection import train_test_split
import sklearn.svm as sk_svm
import matplotlib.pyplot as plt

"导入数据，从python 的数据仓库中"
digits = sk_datasets.load_digits()

"数据分隔为测试数据和训练数据集"
xtrain,xtest,ytrain,ytest = train_test_split(digits.data,digits.target,test_size=0.20,random_state=2)
clf = sk_svm.SVC(gamma=0.001,C= 100.)
clf.fit(xtrain,ytrain)
print xtest
ypred = clf.predict(xtest)
print ypred
print clf.score(xtest,ytest)

"实验模型的预测情况"
fig,axes = plt.subplots(
    8,8,figsize = (8,8)
)
fig.subplots_adjust(hspace = 0.1,wspace = 0.1)
"for循环，遍历axes中的所有手写字体"
for i,ax in enumerate(axes.flat):
    "显示手写字"
    ax.imshow(xtest[i].reshape(8,8),cmap=plt.cm.gray_r,interpolation='nearest')
    "左下角显示算法预测的数字，如果和正确答案相符，字体为绿色，如果不是，则为红色"
    ax.text(0.05,0.05,str(ypred[i]),fontsize = 32,transform = ax.transAxes,color = 'green' if ypred[i] ==ytest[i] else 'red')
    "在右下角，用黑色字体显示一个正确答案"
    ax.text(0.8,0.05,str(ytest[i]),fontsize = 32,transform=ax.transAxes,color='black')
    ax.set_xticks([])
    ax.set_xticks([])
plt.show()
