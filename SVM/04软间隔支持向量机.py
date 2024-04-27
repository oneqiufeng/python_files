# 软间隔支持向量机（Soft Margin Support Vector Machine）是一种在训练数据中允许一定程度的分类错误的支持向量机。
# 核心原理：在硬间隔支持向量机中，要求所有的训练样本都被正确分类，但在实际情况中，数据可能存在噪音或异常点。软间隔支持向量机通过引入松弛变量（slack variables）来容忍一些错误，从而提高模型的鲁棒性。
# 核心公式：原始优化问题和约束条件
'''
特点：
允许一定程度的分类错误，提高模型的鲁棒性。
适用于训练数据中存在噪音或异常点的情况。
超参数控制了松弛变量的惩罚力度，可以通过调整来平衡间隔的最大化和错误的最小化。

适用场景：
数据集中存在噪音或异常点的情况。
对于硬间隔支持向量机无法完全分离的数据集。
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC

# 生成样本数据
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)

# 训练软间隔支持向量机模型
svm = SVC(kernel='linear', C=1)
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

# 软间隔支持向量机通过引入松弛变量来容忍一定程度的分类错误，提高了模型的鲁棒性。适用于数据集中存在噪音或异常点的情况，通过调整正则化参数C来平衡最大化间隔和最小化分类错误。在实际应用中，软间隔支持向量机更灵活，可以适应更多的复杂情况。