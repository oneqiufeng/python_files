# 导入必要的库
import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# 设置随机种子以获得可重现的结果
np.random.seed(0)

# 随机生成自变量X和因变量y的数据
n_samples = 500  # 样本数量
n_features = 2    # 自变量数量

# 生成具有多重共线性的数据
X = np.random.randn(n_features, n_features)
X[:, 0] = X[:, 0] * 0.1  # 使得第一列与其他列相关性较低
y = np.dot(X, np.array([10, 20])) + np.random.randn(n_features) * 0.1  # 添加噪声

# 转换为二维数组，以适应scikit-learn的API
X = X.T  # X现在是一个n_samples x n_features的矩阵

# 创建岭回归模型实例，alpha是正则化强度的参数
ridge_model = Ridge(alpha=1.0)

# 训练模型，拟合数据
ridge_model.fit(X, y)

# 预测
predictions = ridge_model.predict(X)

# 选择X的第一个特征作为x轴数据
X1 = X[:, 0]

# 可视化结果
plt.scatter(X1, y, color='blue', label='Actual Data')  # 实际数据点
plt.plot(X1, predictions, color='red', linewidth=2, label='Predicted Line')  # 预测的线性关系
plt.title('Ridge Regression with Multiple Collinearity')
plt.xlabel('X1')
plt.ylabel('y')
plt.legend()
plt.show()