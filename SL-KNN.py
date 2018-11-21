#!/usr/bin/python
#-*- coding: UTF-8 -*-
import sklearn.datasets as sk_datasets
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.svm as sk_svm
import matplotlib.pyplot as plt
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import KNeighborsClassifier
"基于python2.7.0"
"KNN（K近邻算法）部分"

"生成数据"
centers = [[-2,2],[2,2],[0,4]]
X,y = make_blobs(n_samples=60,centers = centers,random_state=0,cluster_std=0.60)

"画出数据图"
plt.figure(figsize=(16,10),dpi = 144)
c = np.array(centers)
plt.scatter(X[:,0],X[:,1],c=y,s=100,cmap='cool')
plt.scatter(c[:,0],c[:,1],s = 100, marker='^',c='red')
"显示数据图"
plt.show()

"模型训练"
k = 5
clf = KNeighborsClassifier(n_neighbors=k)
clf.fit(X,y)
X_sample = [0,2]
print X_sample
X_sample = np.array(X_sample).reshape(-1,1)
Y_sample = clf.predict(X_sample)
neighbors = clf.kneighbors(X_sample,return_distance=False)

"画出数据"
plt.figure(figsize=(16,10),dpi=144)
plt.scatter(X[:,0],X[:,1],c=y,s=100,cmap='cool')
plt.scatter(c[:,0],c[:,1],s=100,marker='^',c='k')
plt.scatter(X_sample[0],X_sample[1],marker="x",c=Y_sample,s=100,cmap='cool')

for i in neighbors[0]:
    plt.plot([X[i][0],X_sample[0]],[X[i][1],X_sample[1]],'k--',linewidth=0.6)
plt.show()


