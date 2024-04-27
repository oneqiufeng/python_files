# 稀疏支持向量机（Sparse Support Vector Machine）是支持向量机的一种变体，其目的是通过限制支持向量的数量来提高模型的可解释性和计算效率。
'''核心原理
稀疏支持向量机通过引入正则化项，同时最小化模型的损失函数和支持向量的数量，从而使得模型在保持较高分类性能的同时具有更少的支持向量。
优化目标

稀疏支持向量机的优化目标是：

稀疏性惩罚

在优化目标中，通过调整正则化参数来平衡间隔的最大化和错误的最小化，从而控制支持向量的数量，实现模型的稀疏性。
'''

'''特点：

通过限制支持向量的数量，提高模型的可解释性和计算效率。
适用于大规模数据集和高维特征空间下的分类问题。
通过调整正则化参数可以灵活控制模型的稀疏程度。
适用场景：

对模型的可解释性和计算效率要求较高的情况。
大规模数据集和高维特征空间下的分类问题。
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.svm import SVC

# 生成样本数据
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)

# 训练稀疏支持向量机模型
svm = SVC(kernel='linear', C=0.1)
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
plt.title('Sparse SVM Classification')
plt.show()

# 稀疏支持向量机通过限制支持向量的数量来提高模型的可解释性和计算效率。适用于大规模数据集和高维特征空间下的分类问题，对模型的可解释性和计算效率要求较高的情况。通过调整正则化参数可以灵活控制模型的稀疏程度。