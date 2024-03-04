# 层次聚类是一种常用的聚类方法，它通过构建数据点之间的层次结构来进行聚类。层次聚类不需要预先指定簇的数量，并且结果可以表示为树状图（称为树状图或层次树），提供了数据点之间关系的丰富视图。

# 类型
# 凝聚型（Agglomerative）：从每个点作为单独的簇开始，逐渐合并最近的簇。
# 分裂型（Divisive）：从所有数据作为一个簇开始，逐步分裂为更小的簇。

# 算法步骤（以凝聚型为例）
# 开始时，将每个数据点视为一个单独的簇。
# 找到最相似（距离最近）的两个簇并将它们合并。
# 重复上一步骤，直到所有数据点都合并到一个簇中或达到预定的簇数量。

# 接下来，使用 Python 的 scipy 库来实现层次聚类，并使用 matplotlib 库绘制树状图。我们将使用相同的模拟数据来展示层次聚类的结果。

from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt

# 生成层次聚类的链接矩阵
Z = linkage(X, method='ward')

# 绘制树状图
plt.figure(figsize=(10, 5))
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample index')
plt.ylabel('Distance')
dendrogram(Z)
plt.show()

# 上图展示了层次聚类的树状图，也称为树状图。在这个图中：

# 每个点代表一个数据样本。
# 水平线表示簇的合并，其长度代表合并簇之间的距离或不相似度。
# 树状图的垂直轴代表距离或不相似度，可以用来判断簇之间的距离。
# 通过这个树状图，我们可以观察数据的层次聚类结构，并根据需要选择适当的截断点来确定簇的数量。例如，通过在不同的高度水平切割树状图，可以得到不同数量的簇。

# 层次聚类特别适用于那些簇的数量不明确或数据具有自然层次结构的场景。与 K-means 等算法相比，它不需要预先指定簇的数量，但计算复杂度通常更高。