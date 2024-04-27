# 非线性支持向量机
# 非线性支持向量机（Nonlinear Support Vector Machine）通过引入核函数来处理非线性可分的数据。
# 核心原理：非线性支持向量机的主要思想是将输入空间中的数据通过一个非线性映射（核函数）映射到一个高维特征空间，在高维特征空间中寻找一个线性超平面来进行分类。
# 核心公式：决策函数和核函数

'''
特点：
能够处理非线性可分的数据。
引入核函数后可以在高维特征空间中进行线性分类。
避免了直接对高维空间进行计算，节省了计算成本。

适用场景：
当数据不是线性可分时，使用非线性支持向量机能够获得更好的分类效果。
数据集维度较高，线性分类效果不佳时。
需要灵活选择核函数以适应不同的数据分布情况。
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.svm import SVC

# 生成非线性可分的样本数据
X, y = make_circles(n_samples=100, noise=0.1, factor=0.4, random_state=42)

# 训练非线性支持向量机模型
svm = SVC(kernel='rbf', gamma=1)
svm.fit(X, y)

# 绘制决策边界
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')

# 绘制支持向量
plt.scatter(svm.support_vectors_[:, 0], svm.support_vectors_[:, 1],
            s=200, facecolors='none', edgecolors='k')

# 绘制决策边界
ax = plt.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

xx, yy = np.meshgrid(np.linspace(xlim[0], xlim[1], 200),
                     np.linspace(ylim[0], ylim[1], 200))
Z = svm.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, levels=np.linspace(Z.min(), Z.max(), 10), alpha=0.3, cmap=plt.cm.coolwarm)

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Nonlinear SVM Classification')
plt.show()

# 非线性支持向量机通过引入核函数来处理非线性可分的数据，能够在高维特征空间中寻找一个线性超平面来进行分类，从而取得更好的分类效果。适用于处理非线性可分的数据和高维特征空间下的分类问题。通过选择不同的核函数可以适应不同的数据分布情况。