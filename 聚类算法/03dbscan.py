# Database Scan
# DBSCAN（Density-Based Spatial Clustering of Applications with Noise）是一种基于密度的聚类算法，特别适用于具有噪声的数据集和能够发现任意形状簇的情况。它不需要事先指定簇的数量，能有效处理异常点。

# 算法步骤

''' 
1. 标记所有点为核心点、边界点或噪声点。
2. 删除噪声点。
3. 为剩余的核心点创建簇，如果一个核心点在另一个核心点的邻域内，则将它们放在同一个簇中。
4. 将每个边界点分配给与之关联的核心点的簇。
 '''
from sklearn.cluster import DBSCAN
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

# 再次使用之前的模拟数据
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 应用DBSCAN算法
dbscan = DBSCAN(eps=0.5, min_samples=5)
# 邻域大小（eps=0.5）和最小点数（min_samples=5）
clusters = dbscan.fit_predict(X)

# 绘制聚类结果
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', marker='o', s=50)
plt.title("DBSCAN Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

''' 
上图展示了使用 DBSCAN 算法对模拟数据进行的聚类结果。在这个图中，不同颜色的点表示不同的簇，而相同颜色的点属于同一个簇。

在 DBSCAN 算法中，我设置了邻域大小（eps=0.5）和最小点数（min_samples=5）。算法能够识别出密度不同的簇，并且有效地区分出噪声点（通常用特殊颜色或标记表示，但在此图中未显示）。

DBSCAN 的优势在于它不需要事先指定簇的数量，可以识别任意形状的簇，并且对噪声数据具有良好的鲁棒性。然而，选择合适的 eps 和 min_samples 参数对于获得好的聚类结果至关重要。 
'''