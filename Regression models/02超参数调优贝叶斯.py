# ---
# 贝叶斯优化通过构建超参数的概率模型来选择下一个最有可能实现更好性能的超参数组合。

from skopt import BayesSearchCV
from skopt.space import Real
parameters = {'alpha': Real(0.1, 10)}
ridge = Ridge()
bayes_search = BayesSearchCV(ridge, parameters, n_iter=50)
bayes_search.fit(X, y)
print(bayes_search.best_params_)
