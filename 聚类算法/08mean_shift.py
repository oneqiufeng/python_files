# Mean Shift 算法是一种基于密度的非参数聚类算法。其核心思想是通过迭代过程寻找数据点密度的峰值。这个算法不需要预先指定簇的数量，它通过数据本身的分布特性来确定簇的数量。

'''
算法概述
1. 选择带宽（Bandwidth）：带宽确定了搜索窗口的大小，对算法的结果有显著影响。
2. 迭代过程：对每个数据点，计算其在带宽范围内邻近点的均值，然后将数据点移动到这个均值位置。
3. 收敛：重复上述过程直到数据点的移动非常小或达到预定的迭代次数。
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift
from sklearn.datasets import make_blobs

# 生成样本数据
centers = [[1, 1], [5, 5], [3, 10]]
X, _ = make_blobs(n_samples=300, centers=centers, cluster_std=0.7)

# 应用 Mean Shift 算法
ms = MeanShift()
ms.fit(X)
labels = ms.labels_
cluster_centers = ms.cluster_centers_

# 可视化结果
plt.scatter(X[:, 0], X[:, 1], c=labels, marker='o')
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='x')
plt.title('Mean Shift Clustering')
plt.show()

# 这段代码首先生成一些样本数据，然后应用 Mean Shift 算法对数据进行聚类，并将结果可视化。每个聚类的中心用红色的 'x' 标记。