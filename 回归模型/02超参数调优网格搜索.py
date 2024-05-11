# 网格搜索是一种通过穷举给定的超参数组合来确定最佳超参数的方法。
# 对于每个超参数组合，都会使用交叉验证来评估模型性能，最终选择具有最佳性能的超参数组合。网格搜索的缺点是计算成本高，特别是在超参数空间较大时。

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression

X, y = make_regression(n_samples=100, n_features=1, noise=0.1, random_state=42)
parameters = {'alpha': [0.1, 1, 10]}
ridge = Ridge()
grid_search = GridSearchCV(ridge, parameters)
grid_search.fit(X, y)
print(grid_search.best_params_)