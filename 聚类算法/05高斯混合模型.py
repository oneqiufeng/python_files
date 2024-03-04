# 高斯混合模型（GMM）是一种基于概率模型的聚类算法，它假设所有数据点都是从有限个高斯分布的混合生成的。与K-means等硬聚类算法不同，GMM 属于软聚类算法，它为每个数据点提供了属于各个簇的概率。

from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 生成模拟数据
X, _ =make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 应用高斯混合模型聚类算法
gmm = GaussianMixture(n_components=4)
gmm.fit(X)
clusters = gmm.predict(X)

# 绘制聚类结果
plt.scatter(X[:, 0],X[:, 1], c = clusters, cmap='viridis', marker='o', s=50)
plt.title("Gaussian Mixture Model Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

'''
上图展示了使用高斯混合模型（GMM）对模拟数据进行的聚类结果。在这个图中，不同颜色的点表示不同的簇，而相同颜色的点属于同一个簇。

在这个示例中，GMM 被设置为将数据分成四个簇（n_components=4）。GMM 算法不仅为每个点分配了一个簇，而且还可以提供关于每个点属于各个簇的概率信息。

GMM 的优势在于它是一个基于概率的方法，提供了比 K-means 更丰富的信息，并且可以模拑非球形的簇。它通过期望最大化（EM）算法迭代地优化参数，以最大化数据的似然概率。不过，选择合适的簇数量和协方差类型对于获得好的聚类结果至关重要。此外，GMM 对于初始化参数比较敏感，可能会陷入局部最优。
'''