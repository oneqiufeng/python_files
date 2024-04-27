# 多类别支持向量机（Multiclass Support Vector Machine）是支持向量机在处理多类别分类问题时的扩展。
# 核心原理：多类别支持向量机通过将多个二分类支持向量机组合来处理多类别分类问题。常见的方法包括一对一（One-vs-One）和一对其余（One-vs-Rest）。
# 核心公式：One-vs-One + One-vs-Rest
'''
特点：
可以处理多类别分类问题。
可以使用与二分类支持向量机相同的技术和核函数。
一对一方法通常需要更多的模型训练和内存开销，但在某些情况下可能更有效。
一对其余方法通常更简单直观，但可能存在类别不平衡问题。

适用场景：
多类别分类问题。
适用于各类别之间相互独立的情况。
数据集较小的情况下，一对一方法可能更适用；数据集较大的情况下，一对其余方法可能更有效。
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC
from itertools import combinations

# 生成样本数据
X, y = make_classification(n_samples=100, n_features=2, n_classes=3, n_clusters_per_class=1, n_informative=2, n_redundant=0)

# 训练多类别支持向量机模型（一对一方法）
classifiers = {}
class_pairs = list(combinations(np.unique(y), 2))
for pair in class_pairs:
    X_pair, y_pair = X[np.where(np.isin(y, pair))], y[np.where(np.isin(y, pair))]
    svm = SVC(kernel='linear')
    svm.fit(X_pair, y_pair)
    classifiers[pair] = svm

# 绘制决策边界
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')

# 绘制决策边界
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 200),
                     np.linspace(ylim[0], ylim[1], 200))
for pair, clf in classifiers.items():
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, colors='k', levels=[-1, 1], alpha=0.5,
                linestyles=['--', '-', '--'])

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Multiclass SVM Classification (One-vs-One)')
plt.show()

# 多类别支持向量机通过扩展二分类支持向量机来处理多类别分类问题。常用的方法包括一对一和一对其余。可以使用不同的核函数和技术来实现多类别支持向量机。适用于多类别分类问题，选择适当的方法取决于数据集的大小和特性。