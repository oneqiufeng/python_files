# OPTICS（Ordering Points To Identify the Clustering Structure）算法是一种用于数据聚类的算法，与DBSCAN算法类似，但在处理可变密度的数据集时更为有效。其核心思想是通过分析数据点的密度-可达性来识别聚类结构。

# 简单介绍
'''
核心概念：OPTICS算法主要关注两个概念，即核心距离和可达距离。
核心距离：对于给定点，其核心距离是使其成为核心对象的最小半径。
可达距离：是一个对象到一个核心对象的最小距离。
算法流程：OPTICS算法首先根据核心距离和可达距离为数据点创建一个排序，然后基于这个排序来识别聚类。
'''

from sklearn.cluster import OPTICS
import matplotlib.pyplot as plt
import numpy as np

# 示例数据
X = np.random.rand(100, 2)

# OPTICS模型
optics_model = OPTICS(min_samples=5, xi=0.05, min_cluster_size=0.1)
optics_model.fit(X)

# 可视化
space = np.arange(len(X))
reachability = optics_model.reachability_[optics_model.ordering_]
labels = optics_model.labels_[optics_model.ordering_]

plt.figure(figsize=(10, 7))
colors = ['g.', 'r.', 'b.', 'y.', 'c.']
for klass, color in zip(range(0, 5), colors):
    Xk = space[labels == klass]
    Rk = reachability[labels == klass]
    plt.plot(Xk, Rk, color, alpha=0.3)
plt.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
plt.ylabel('Reachability (epsilon distance)')
plt.title('Reachability Plot')
plt.show()

# 在这个代码中，我们首先生成了一些随机数据点，然后使用OPTICS算法对其进行聚类，并使用matplotlib库来可视化结果。这个示例生成了一个可达性图，其中每个点的可达性距离都被绘制出来，以揭示数据中的聚类结构。