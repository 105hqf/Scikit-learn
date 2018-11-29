#!/usr/bin/python
#-*- coding: UTF-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from matplotlib.figure import SubplotParams

"梯度下降算法"
n_dots = 200
"划定定义域"
X = np.linspace(-20,20, n_dots)
"随机生成几个在定义域上的正弦函数上的点，并加入一些噪音"
Y = np.sin(X) + 0.2 * np.random.rand(n_dots) - 0.1
"reshape()函数用来生成为numpy所能接受的数据格式"
X = X.reshape(-1, 1)
Y = Y.reshape(-1, 1)
"用PolynomialFeatures和pipeline创建一个多项式的拟合模型"
def polynomial_model(degree=1):
    polynomial_features = PolynomialFeatures(degree=degree,
                                             include_bias=False)
    linear_regression = LinearRegression(normalize=True)
    pipeline = Pipeline([("polynomial_features", polynomial_features),
                         ("linear_regression", linear_regression)])
    return pipeline
"用10、20、30、40阶多项式来拟合数据集"
degrees = [10, 20, 30,40]
results = []
"算出每个模型的拟合评分，用mean_squared_error算出均方根误差"
for d in degrees:
    model = polynomial_model(degree=d)
    model.fit(X, Y)
    train_score = model.score(X, Y)
    mse = mean_squared_error(Y, model.predict(X))
    results.append({"model": model, "degree": d, "score": train_score, "mse": mse})
for r in results:
    print("degree: {}; train score: {}; mean squared error: {}".format(r["degree"], r["score"], r["mse"]))

plt.figure(figsize=(12, 6), dpi=200, subplotpars=SubplotParams(hspace=0.3))
for i, r in enumerate(results):
    fig = plt.subplot(2, 2, i+1)
    plt.xlim(-8, 8)
    plt.title("LinearRegression degree={}".format(r["degree"]))
    plt.scatter(X, Y, s=5, c='b', alpha=0.5)
    plt.plot(X, r["model"].predict(X), 'r-')
    plt.show()