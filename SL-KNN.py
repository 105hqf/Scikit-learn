#!/usr/bin/python
#-*- coding: UTF-8 -*-
import numpy as np
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
# 进行预测
X_sample = [0, 2]
X_sample = np.array(X_sample).reshape(1, -1)
y_sample = clf.predict(X_sample);
neighbors = clf.kneighbors(X_sample, return_distance=False);

# 画出示意图
plt.figure(figsize=(16, 10))
plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap='cool')    # 样本
plt.scatter(c[:, 0], c[:, 1], s=100, marker='^', c='k')   # 中心点
plt.scatter(X_sample[0][0], X_sample[0][1], marker="x",
            s=100, cmap='cool')    # 待预测的点

for i in neighbors[0]:
    # 预测点与距离最近的 5 个样本的连线
    plt.plot([X[i][0], X_sample[0][0]], [X[i][1], X_sample[0][1]],
             'k--', linewidth=0.6);


