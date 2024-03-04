# 使用数据的相似性矩阵来进行聚类，特别适用于复杂形状的数据集。

# 谱聚类是一种基于图论的聚类方法，特别适用于发现复杂形状的簇和非球形簇。与传统的聚类算法（如K-means）不同，谱聚类依赖于数据的相似性矩阵，并利用数据的谱（即特征向量）来进行降维，进而在低维空间中应用如K-means的聚类方法。

# 算法步骤

'''
1. 构建相似性矩阵：基于数据点之间的距离或相似度。
2. 计算图的拉普拉斯矩阵：常用的是归一化拉普拉斯矩阵。
3. 计算拉普拉斯矩阵的特征向量和特征值。
4. 基于前k个特征向量的新特征空间，应用传统聚类算法（如K-means）。
'''

# 下面，使用 Python 的 sklearn 库中的 SpectralClustering 类来实现谱聚类。

from sklearn.cluster import SpectralClustering
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 生成模拟数据
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 应用谱聚类算法
spectral_clustering = SpectralClustering(n_clusters=4, affinity='nearest_neighbors')
clusters = spectral_clustering.fit_predict(X)

# 绘制聚类结果
plt.scatter(X[:, 0], X[:, 1], c=clusters, cmap='viridis', marker='o', s=50)
plt.title("Spectral Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 上图展示了使用谱聚类算法对模拟数据进行的聚类结果。在这个图中，不同颜色的点表示不同的簇，而相同颜色的点属于同一个簇。

# 在这个示例中，谱聚类被设置为将数据分成四个簇（n_clusters=4），并使用最近邻方法（affinity='nearest_neighbors'）来构建相似性矩阵。然而，警告信息表明，生成的图可能不是完全连接的，这可能影响聚类结果。

# 谱聚类的一个关键优势是能够发现任意形状的簇，这使得它特别适用于那些传统聚类算法（如K-means）难以处理的数据集。不过，选择合适的相似性度量和参数对于获得好的聚类结果至关重要。此外，谱聚类的计算复杂度比一些其他聚类算法高，特别是在处理大型数据集时。