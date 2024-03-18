# 导入必要的库
import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt

from sklearn.linear_model import Ridge

# 构造数据
x = np.array([[1, 2], [2, 4], [3, 6], [4, 8], [5, 10]])
y = np.array([3, 6, 9, 12, 15])

# 岭回归
ridge = Ridge(alpha=1.0)
ridge.fit(x, y)

# 预测值
y_pred = ridge.predict(x)

# 绘制散点图和回归直线
fig, ax = plt.subplots()
ax.scatter(x[:, 1], y)
ax.plot(x[:, 1], y_pred, color='red')
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()