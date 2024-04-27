# 核函数支持向量机（Kernel Support Vector Machine）是支持向量机的扩展，通过引入核函数来处理非线性可分的数据。其原理和核心公式如下所述。
# 核心原理：核函数支持向量机通过将输入空间中的数据点映射到一个高维特征空间，在高维特征空间中寻找一个线性超平面来进行分类，从而实现对非线性可分数据的分类。
'''
特点：
能够处理非线性可分的数据。
引入核函数后可以在高维特征空间中进行线性分类。
避免了直接对高维空间进行计算，节省了计算成本。
适用场景：
数据集中存在非线性关系的情况。
高维特征空间下的分类问题。
数据集维度较高，线性分类效果不佳时。
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC

# 生成非线性可分的样本数据
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

# 训练核函数支持向量机模型（使用高斯径向基函数RBF作为核函数）
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
plt.title('Kernel SVM Classification (RBF Kernel)')
plt.show()

# 核函数支持向量机通过引入核函数来处理非线性可分的数据，能够在高维特征空间中寻找一个线性超平面来进行分类。适用于处理非线性可分的数据和高维特征空间下的分类问题。选择合适的核函数对模型的性能影响较大，需要根据具体问题选择合适的核函数。