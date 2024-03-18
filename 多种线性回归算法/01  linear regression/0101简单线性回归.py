# 导入必要的库
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 示例数据：自变量X和因变量y
X = np.array([[1], [2], [3], [4]])
y = np.array([2, 3, 5, 7])

# 创建线性回归模型实例
model = LinearRegression()

# 训练模型，拟合数据
model.fit(X, y)

# 预测
predictions = model.predict(X)

# 打印模型参数
print(f'斜率（权重）: {model.coef_}')
print(f'截距: {model.intercept_}')

# 可视化结果
plt.scatter(X, y, color='blue')  # 实际数据点
plt.plot(X, predictions, color='red')  # 预测的线性关系
plt.title('线性回归模型')
plt.xlabel('X')
plt.ylabel('y')
plt.show()