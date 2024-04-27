# 径向基函数（Radial Basis Function, RBF）支持向量机是很常用的核函数支持向量机，它通过径向基函数来将数据映射到高维特征空间。
# 核心原理：径向基函数支持向量机通过使用径向基函数将数据映射到高维特征空间，并在该空间中寻找一个线性超平面来进行分类。其核心思想是通过计算数据点与支持向量之间的距离来判断其类别。
# 决策函数+高斯核函数
'''特点：

能够处理非线性可分的数据。
通过调整高斯核函数的带宽参数可以控制决策边界的复杂度。
对于高维数据和复杂数据集具有较好的分类性能。
适用场景：

非线性可分的数据集。
高维特征空间下的分类问题。
对分类性能要求较高的应用场景。'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC

# 生成非线性可分的样本数据
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

# 训练径向基函数支持向量机模型
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
plt.title('RBF SVM Classification')
plt.show()

# 径向基函数支持向量机通过使用高斯核函数将数据映射到高维特征空间，并在该空间中寻找一个线性超平面来进行分类。适用于处理非线性可分的数据集和高维特征空间下的分类问题。调整高斯核函数的带宽参数可以控制决策边界的复杂度，从而适应不同的数据分布情况。