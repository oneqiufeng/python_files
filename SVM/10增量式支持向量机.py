# 增量式支持向量机（Incremental Support Vector Machine，ISVM）是一种能够在线学习的支持向量机算法，可以在不断接收新样本时更新模型参数。以下是详细的原理、核心公式、特点、适用场景、代码案例演示。

'''核心原理
增量式支持向量机通过逐步学习新样本并调整模型参数，实现对新数据的快速适应，而无需重新训练整个模型。其核心思想是利用新样本的信息来更新原有模型，从而提高模型的性能。
'''

'''核心公式
更新规则

增量式支持向量机的更新规则通常是基于梯度下降或增量优化方法来更新模型参数。更新规则的具体形式取决于选择的优化算法。

损失函数

增量式支持向量机通常采用与传统支持向量机相同的损失函数，例如Hinge Loss。损失函数的选择影响了模型的性能和收敛速度。
'''

'''特点：

能够在线学习，适用于连续接收新数据的场景。
可以快速地适应新样本，无需重新训练整个模型。
可以有效地处理大规模数据集。
适用场景：

需要实时更新模型以适应新数据的应用场景。
处理数据量大、维度高的数据集。
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import SGDClassifier

# 生成样本数据
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1)

# 初始化增量式支持向量机模型
svm = SGDClassifier(loss='hinge', penalty='l2', alpha=0.001, max_iter=100)

# 逐步接收新样本并更新模型
for i in range(0, len(X), 10):
    X_batch = X[i:i+10]
    y_batch = y[i:i+10]
    svm.partial_fit(X_batch, y_batch, classes=np.unique(y))

# 绘制决策边界
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, s=50, edgecolors='k')

# 绘制支持向量
plt.scatter(svm.coef_[0, 0], svm.coef_[0, 1], s=200, facecolors='none', edgecolors='k')

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
plt.title('Incremental SVM Classification')
plt.show()

'''这段代码画出的是二维特征空间中的数据点以及基于增量学习的线性支持向量机（SVM）的决策边界。具体来说：

数据点：生成了包含两个特征的二维数据集，并且使用散点图将这些数据点在特征空间中可视化。不同类别的数据点使用不同的颜色表示。

决策边界：通过对特征空间中的网格点进行预测，然后绘制等高线来显示SVM的决策边界。等高线表示决策函数的值，在决策函数为0的地方就是决策边界。

支持向量：在决策边界附近绘制了支持向量。这些是在训练过程中被SVM选中作为最接近决策边界的训练样本点。

通过观察上述图像，可以了解到SVM如何在特征空间中划分不同类别的数据点，并且理解SVM的决策边界是如何确定的。

最后说说增量式支持向量机，它是能够在线学习并逐步更新模型，适用于连续接收新数据的场景。通过不断接收新样本并更新模型参数，从而有效地提高模型的性能，并且无需重新训练整个模型。适用于处理大规模数据集和需要实时更新模型的应用场景。
'''