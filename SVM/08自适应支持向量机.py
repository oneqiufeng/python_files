# 自适应支持向量机（Adaptive Support Vector Machine，ASVM）是一种能够自适应地调整核函数参数的支持向量
# 核心原理：自适应支持向量机通过优化核函数参数，使得模型在训练集上的性能达到最佳。通常采用交叉验证等技术来选择最优的核函数参数。
# 核心公式：自适应支持向量机的核心公式与传统的支持向量机相同，但核函数参数需要经过优化得到。常用的核函数参数优化方法包括网格搜索、贝叶斯优化等。
'''特点：

能够自适应地调整核函数参数，提高模型的性能。
具有一定的鲁棒性，能够适应不同的数据分布情况。
可以根据具体问题选择合适的核函数类型和参数。
适用场景：

数据集具有复杂的非线性结构。
对分类性能要求较高的情况。
需要在训练过程中动态地调整核函数参数的情况。'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# 生成非线性可分的样本数据
X, y = make_moons(n_samples=100, noise=0.1, random_state=42)

# 使用网格搜索选择最优的核函数参数
param_grid = {'C': [0.1, 10, 100], 'gamma': [0.1, 10, 100]}
grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5)
grid_search.fit(X, y)
best_params = grid_search.best_params_

# 训练自适应支持向量机模型
svm = SVC(kernel='rbf', C=best_params['C'], gamma=best_params['gamma'])
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
plt.title('Adaptive SVM Classification')
plt.show()

'''这段代码画出的是二维特征空间中的数据点以及基于增量学习的线性支持向量机（SVM）的决策边界。具体来说：

数据点：生成了包含两个特征的二维数据集，并且使用散点图将这些数据点在特征空间中可视化。不同类别的数据点使用不同的颜色表示。

决策边界：通过对特征空间中的网格点进行预测，然后绘制等高线来显示SVM的决策边界。等高线表示决策函数的值，在决策函数为0的地方就是决策边界。

支持向量：在决策边界附近绘制了支持向量。这些是在训练过程中被SVM选中作为最接近决策边界的训练样本点。

通过观察上述图像，可以了解到SVM如何在特征空间中划分不同类别的数据点，并且理解SVM的决策边界是如何确定的。

最后说说增量式支持向量机，它是能够在线学习并逐步更新模型，适用于连续接收新数据的场景。通过不断接收新样本并更新模型参数，从而有效地提高模型的性能，并且无需重新训练整个模型。适用于处理大规模数据集和需要实时更新模型的应用场景。'''