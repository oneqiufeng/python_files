# 导入必要的库
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 样例数据：自变量X和因变量y
# 这里我们有三个自变量X1, X2, X3和一个因变量y
X = np.array([[1, 2, 3],  # 第一个样本，X1=1, X2=2, X3=3
              [4, 5, 6],  # 第二个样本，X1=4, X2=5, X3=6
              [7, 8, 9],  # 第三个样本，X1=7, X2=8, X3=9
              [10, 11, 12]])  # 第四个样本，X1=10, X2=11, X3=12
y = np.array([28, 53, 77, 100])

# 创建多元线性回归模型实例
model = LinearRegression()

# 训练模型，拟合数据
model.fit(X, y)

# 预测
predictions = model.predict(X)

# 打印模型参数
print(f'权重（斜率）: {model.coef_}')
print(f'截距: {model.intercept_}')

# 可视化结果
# 为了可视化，我们只取其中一个特征进行展示
plt.scatter(X[:, 0], y, color='blue', label='Actual Data')  # 实际数据点
plt.plot(X[:, 0], predictions, color='red', linewidth=2, label='Predicted Line')  # 预测的线性关系
plt.title('多元线性回归模型')
plt.xlabel('X1')
plt.ylabel('y')
plt.legend()
plt.show()