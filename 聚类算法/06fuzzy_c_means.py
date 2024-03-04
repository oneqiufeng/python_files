import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

# 生成模拟数据
X, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 模糊C-means算法实现
def fuzzy_c_means(X, C, m=2, error=0.005, maxiter=300):
    # 初始化隶属度矩阵
    n = X.shape[0]
    U = np.random.rand(n, C)
    U = U / np.sum(U, axis=1, keepdims=True)

    for iteration in range(maxiter):
        # 更新聚类中心
        centers = np.dot(U.T ** m, X) / np.sum(U.T ** m, axis=1, keepdims=True)

        # 更新隶属度
        distance = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        U_new = 1.0 / np.sum((distance[:, :, np.newaxis] / distance[:, np.newaxis, :]) ** (2 / (m - 1)), axis=2)

        # 检查是否收敛
        if np.max(np.abs(U_new - U)) < error:
            break
        U = U_new

    return centers, U

# 使用模糊C-means算法
centers, U = fuzzy_c_means(X, C=4)

# 绘制结果
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X')
plt.title('Fuzzy C-means Clustering')
plt.show()