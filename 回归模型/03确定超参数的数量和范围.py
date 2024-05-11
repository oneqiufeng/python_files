# 我用一个简单的线性回归的例子来说明以上的内容。首先，我们需要确定超参数的范围，比如正则化参数 alpha 的范围可以设定在 0 到 1 之间。然后，我们可以使用交叉验证来评估不同超参数组合的性能。最后，选择在交叉验证中性能最佳的超参数组合作为最终模型的超参数。

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.datasets import load_boston

# 加载数据集
boston = load_boston()
X = boston.data
y = boston.target

# 定义超参数范围
param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10, 100]}

# 定义模型
ridge = Ridge()

# 网格搜索
grid_search = GridSearchCV(estimator=ridge, param_grid=param_grid, cv=5)
grid_search.fit(X, y)

# 输出最佳参数和得分
print("Best Parameters:", grid_search.best_params_)
print("Best Score:", grid_search.best_score_)

# 在这个例子中，使用了岭回归模型，并通过网格搜索来调整正则化参数 alpha。通过交叉验证，我们可以找到最佳的 alpha 值，从而得到最佳的岭回归模型。