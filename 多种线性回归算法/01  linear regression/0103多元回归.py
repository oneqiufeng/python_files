# 导入必要的库
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 设置随机种子以获得可重现的结果
np.random.seed(0)

# 随机生成自变量X和因变量y的数据
# 假设有5个自变量和一个因变量
n_samples = 30  # 样本数量
n_features = 5   # 自变量数量

# 随机生成自变量X
X = np.random.randn(n_samples, n_features)

# 随机生成模型参数（权重和截距）
np.random.seed(1)  # 为了可重现性，改变随机种子
coef = np.random.randn(n_features) * 2  # 假设权重是随机的，这里乘以2来增加变化范围
intercept = np.random.randn()  # 随机生成截距

# 计算因变量y
y = np.dot(X, coef) + intercept + np.random.randn(n_samples) * 5  # 加上截距和随机噪声

# 创建多元线性回归模型实例
model = LinearRegression()

# 训练模型，拟合数据
model.fit(X, y)

# 预测
predictions = model.predict(X)

# 打印模型参数
print(f'估计的权重（斜率）: {model.coef_}')
print(f'估计的截距: {model.intercept_}')

# 可视化结果
# 为了可视化，我们可以选择其中一个自变量X1和因变量y进行展示
plt.scatter(X[:, 0], y, color='blue', label='Actual Data')  # 实际数据点
plt.plot(X[:, 0], predictions, color='red', linewidth=2, label='Predicted Line')  # 预测的线性关系
plt.title('多元线性回归模型')
plt.xlabel('X1')
plt.ylabel('y')
plt.legend()
plt.show()