# K-medoids 用于将数据集中的数据点分成多个簇。这种算法与著名的 K-means 算法相似，但主要区别在于 K-medoids 选择数据点中的实际点作为簇的中心，而 K-means 则使用簇内数据点的均值。

'''
算法简介

初始化：随机选择  个数据点作为初始的簇中心。
分配：将每个数据点分配给最近的簇中心。
更新：计算每个簇的新中心。与 K-means 不同，这里选择簇内最能代表其他点的数据点作为中心。
重复：重复分配和更新步骤，直到簇中心不再变化或达到预设的迭代次数。

'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.datasets import make_blobs

def simple_kmedoids(data, num_clusters, max_iter=100):
    # Randomly initialize medoids
    medoids = np.random.choice(len(data), num_clusters, replace=False)
    for _ in range(max_iter):
        # Compute distances between data points and medoids
        distances = pairwise_distances(data, data[medoids, :])
        # Assign each point to the closest medoid
        clusters = np.argmin(distances, axis=1)
        # Update medoids
        new_medoids = np.array([np.argmin(np.sum(distances[clusters == k, :], axis=0)) for k in range(num_clusters)])
        # Check for convergence
        if np.array_equal(medoids, new_medoids):
            break
        medoids = new_medoids
    return clusters, medoids

# Generating synthetic data
data, _ = make_blobs(n_samples=150, centers=3, n_features=2, random_state=42)

# Applying the simple K-medoids algorithm
num_clusters = 3
clusters, medoids = simple_kmedoids(data, num_clusters)

# Plotting the results
plt.figure(figsize=(10, 6))
plt.scatter(data[:, 0], data[:, 1], c=clusters, cmap='viridis', marker='o', label='Data points')
plt.scatter(data[medoids, 0], data[medoids, 1], c='red', marker='X', s=100, label='Medoids')
plt.title('Simple K-medoids Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend()
plt.grid(True)

# Saving the plot
plot_path_simple = 'simple_kmedoids_clustering.png'
plt.savefig(plot_path_simple)

plot_path_simple

# 我已经应用了一个简化版的 K-medoids 算法来进行聚类，并生成了一个可视化图像。在这个图中，不同颜色的点代表不同的簇，而红色的“X”标记表示每个簇的中心点（即medoids）。这个图形展示了如何将数据点根据它们与中心点的距离分配到不同的簇中。