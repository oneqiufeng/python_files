# 导入必要的库
import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

# 设置随机种子以获得可重现的结果
np.random.seed(0)

# 随机生成自变量X和因变量y的数据
n_samples = 50  # 样本数量
n_features = 2    # 自变量数量

# 生成具有多重共线性的数据
X = np.random.randn(n_features, n_features)
X[:, 0] = X[:, 0] * 0.1  # 使得第一列与其他列相关性较低
y = np.dot(X, np.array([10, 20])) + np.random.randn(n_features) * 0.1  # 添加噪声

# 转换为二维数组，以适应scikit-learn的API
X = X.T  # X现在是一个n_features x n_samples的矩阵

# 创建岭回归模型实例，alpha是正则化强度的参数
ridge_model = Ridge(alpha=1.0)

# 训练模型，拟合数据
ridge_model.fit(X, y)

# 预测
predictions = ridge_model.predict(X)

# 打印模型参数
print(f'岭回归的权重（斜率）: {ridge_model.coef_}')

# 可视化结果
plt.scatter(X, y, color='blue', label='Actual Data')  # 实际数据点
plt.plot(X, predictions, color='red', linewidth=2, label='Predicted Line')  # 预测的线性关系
plt.title('Ridge Regression')
plt.xlabel('X1')
plt.xlabel('X2')
plt.legend()
plt.show()