# 支持向量机主要用于分类和回归问题。线性支持向量机是SVM中最简单也是最基础的形式之一。
# 核心原理：线性支持向量机的目标是找到一个最优的超平面来分割数据空间，使得两个不同类别的数据点能够被正确分类，并且保持分类边界到最近的数据点的距离（即间隔）最大化。
# 核心公式：分类决策函数和超平面的方程

'''
特点：
线性支持向量机具有良好的泛化性能。
适用于处理高维数据和数据样本数量不平衡的情况。
可以通过核函数扩展到非线性问题。
对于稀疏数据的表现很好。

适用场景：
二分类问题
数据维度高、样本数量较小的情况
稀疏数据集
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC

# 生成样本数据
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)

# 训练线性支持向量机模型
svm = SVC(kernel='linear')
svm.fit(X, y)

# 绘制决策边界
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

# 创建网格以绘制决策边界
xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 50),
                     np.linspace(ylim[0], ylim[1], 50))
Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])

# 绘制决策边界和支持向量
Z = Z.reshape(xx.shape)
plt.contour(xx, yy, Z, colors='k', levels=[-1, 1], alpha=0.5,
            linestyles=['--', '-', '--'])
plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1], s=100,
            linewidth=1, facecolors='none', edgecolors='k')
plt.show()

# 线性支持向量机是一种有效的分类算法，通过寻找最优的超平面来划分数据空间，并在保持分类准确性的同时最大化分类间隔。它适用于二分类问题，尤其在高维数据和数据样本不平衡的情况下表现良好。