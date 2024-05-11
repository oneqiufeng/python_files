# ---
# 随机搜索与网格搜索类似，但是不是穷举所有可能的超参数组合，而是在给定的超参数空间中随机选择一组组合来评估。这种方法通常比网格搜索更高效，尤其是在超参数空间较大时。

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform
parameters = {'alpha': uniform(loc=0, scale=10)}
ridge = Ridge()
random_search = RandomizedSearchCV(ridge, parameters, n_iter=100)
random_search.fit(X, y)
print(random_search.best_params_)
