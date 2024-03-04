# K-means 是一种广泛使用的聚类算法，它的目标是将数据点分组到 K 个簇中，以使簇内的点尽可能相似，而簇间的点尽可能不同。它的核心思想是通过迭代优化簇中心的位置，以最小化簇内的平方误差总和。

# 算法步骤
# 01 初始化：随机选择 K 个数据点作为初始簇中心。
# 02 分配：将每个数据点分配给最近的簇中心。
# 03 更新：重新计算每个簇的中心（即簇内所有点的均值）。
# 04 迭代：重复步骤 2 和 3 直到簇中心不再发生变化或达到预设的迭代次数。

# 使用 Python 的 sklearn 库来实现 K-means 算法

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# 生成模拟数据
np.random.seed(0)
X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 应用K-means算法
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# 绘制数据点和聚类中心
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.5)
plt.title("K-means Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 上图展示了使用 K-means 算法对模拟数据进行聚类的结果。图中的彩色点表示数据点，它们根据所属的簇被着色。红色的大点表示每个簇的中心。

# 在这个示例中，我们设定了四个簇（n_clusters=4），K-means 算法成功地将数据点分配到了这四个簇中，并计算出了每个簇的中心。

# K-means 算法简单高效，广泛应用于各种场景，特别是在需要快速、初步的数据分组时。然而，它也有局限性，比如对初始簇中心的选择敏感，可能会陷入局部最优，且假设簇是凸形的，对于复杂形状的数据可能不适用。