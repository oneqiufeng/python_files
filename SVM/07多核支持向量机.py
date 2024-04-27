# 多核支持向量机（Multiple Kernel Support Vector Machine）是支持向量机的一种扩展，允许在训练过程中使用多个不同的核函数。
# 多核支持向量机通过结合多个核函数的计算结果来构建分类模型，以提高模型的鲁棒性和性能。通过学习各个核函数的权重来确定最终的决策函数。
# 核心公式：决策函数+多核函数（多核支持向量机允许使用多个不同的核函数，常用的核函数包括线性核、多项式核、高斯径向基函数（RBF）等。）+权重学习（通过训练数据学习各个核函数的权重，以使得模型在训练集上的性能达到最佳。）
'''特点：

能够充分利用多个核函数的优势，提高分类性能。
可以根据具体问题选择合适的核函数组合。
允许在训练过程中动态地学习核函数的权重。
适用场景：

数据集具有复杂的非线性结构。
对分类性能要求较高的情况。
需要根据具体问题选择合适的核函数组合。'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.svm import SVC

# 生成非线性可分的样本数据
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

# 训练多核支持向量机模型
svm = SVC(kernel='linear', C=1)
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
plt.title('Multiple Kernel SVM Classification')
plt.show()

# 多核支持向量机允许在训练过程中使用多个不同的核函数，通过结合多个核函数的计算结果来构建分类模型，提高模型的性能和鲁棒性。适用于处理具有复杂非线性结构的数据集，并且对分类性能要求较高的情况。在实际应用中，可以根据具体问题选择合适的核函数组合来构建多核支持向量机模型。