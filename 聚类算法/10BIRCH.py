# BIRCH（平衡迭代式规约和聚类使用层次方法）是一种用于大数据集的聚类算法，特别适用于具有噪声的大规模数据集。BIRCH算法的核心思想是通过构建一个名为CF Tree（聚类特征树）的内存中的数据结构来压缩数据，该数据结构以一种方式保存数据，使得聚类可以高效地进行。

'''
算法步骤
1. 构建CF Tree：读取数据点，更新CF Tree。如果新数据点可以合并到现有聚类中而不违反树的定义，则进行合并；否则，创建新的叶子节点。
2. 凝聚步骤：可选步骤，用于进一步压缩CF Tree，通过删除距离较近的子聚类并重新平衡树。
3. 全局聚类：使用其他聚类方法（如K-Means）对叶子节点中的聚类特征进行聚类。
'''

from sklearn.cluster import Birch
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成模拟数据
X, _ = make_blobs(n_samples=500, centers=10, cluster_std=0.5, random_state=0)

# 应用BIRCH算法
brc = Birch(n_clusters=4)
brc.fit(X)

# 预测聚类标签
labels = brc.predict(X)

# 绘制结果
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', marker='o')
plt.title("BIRCH Clustering")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.colorbar(label='Cluster Label')
plt.show()

# 上图展示了使用BIRCH算法对模拟数据进行聚类的结果。在这个例子中，我们生成了1000个数据点，分布在4个中心点周围。使用BIRCH算法，我们能够有效地将这些点分成四个不同的聚类，如不同颜色所示。

# 在实际应用中，BIRCH算法特别适合于处理大规模数据集，并且当数据集中存在噪声时，它通常也能表现良好。通过调整算法参数，例如树的深度和分支因子，可以优化聚类的性能和准确性。

# 参考来源：https://mp.weixin.qq.com/s/DlT4LAIQdD8mc4yjD9VMjQ