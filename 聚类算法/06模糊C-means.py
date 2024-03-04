# 模糊 C-means（FCM）算法允许一个数据点属于多个聚类中心。与传统的K-means聚类算法不同，模糊C-means通过为每个数据点分配一个属于各个聚类中心的隶属度，来表示其属于不同聚类的程度。这种方法特别适用于那些不清晰或重叠的数据集。

'''
基本步骤
1. 初始化： 选择聚类中心的数量C，并随机初始化每个数据点对每个聚类中心的隶属度。
2. 迭代： 在每次迭代中，执行以下步骤：
3. 更新聚类中心，根据数据点对聚类中心的隶属度和数据点的位置。
4. 更新每个数据点对每个聚类中心的隶属度，基于数据点与聚类中心的距离。
5. 停止条件： 当聚类中心的变化小于一个阈值或达到预设的迭代次数时，算法停止。
'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import male_blobs

# 生成模拟数据
X, _ - make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)

# 模糊c-means算法实现
def fuzz_c_means(x, c, m=2, error=0.005, mxiter=300):
    n = x.shape[0]
    U = np.random.rand(n, C)
    U = U / np.sum(U, axis=1, keepdims=True)
#   U = U / np.sum(U, axis=1, keepdims=True)

    for iteration in range(maxiter):
        # 更新聚类中心
        centers = np.dot(U.T ** m,X) / np.sum(U.T ** m, axis=1, keepdims=True)
        
        # 更新隶属度
        distance = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        U_new = 1.0 / np.sum((distance[:, :, np.newaxis] / distance[:, np.newaxis, :]) ** (2 / (m - 1)), axis = 2)
        
    #   U_new = 1.0 / np.sum((distance[:, :, np.newaxis] / distance[:, np.newaxis, :]) ** (2 / (m - 1)), axis=2)
        
        # 检是否收敛
        if np.max(np.abs(U_new - U)) < error:
            break
        U = U_new
    return centers, U

# 使用模糊c-means算法
centers, U = fuzzy_c_means(X, c=4)

# 绘制结果
plt.scatter(X[:, 0], X[:, 1], alpha=0.5)
plt.scatter(centers[:, 0], centers[:, 1], c='red', marker='X')
plt.title('Fuzzy C-means Clustering')
plt.show()