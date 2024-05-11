# ---

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# 定义参数网格
param_grid = {param1: [value1, value2], param2: [value1, value2]}

# 使用Grid Search
grid_search = GridSearchCV(estimator, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 使用Random Search
random_search = RandomizedSearchCV(estimator, param_distributions, n_iter=100, cv=5)
random_search.fit(X_train, y_train)

# ---
from hyperopt import hp, fmin, tpe, Trials

# 定义参数空间
space = {
param1 = trial.suggest_categorical(param1, [value1, value2, ...])
param2 = trial.suggest_float(param2, low, high)
...
}

# 定义优化目标函数
def objective(params):
# 使用params构建模型，并返回需要最小化的评估指标
return loss

# 使用Hyperopt进行优化
best = fmin(objective, space, algo=tpe.suggest, max_evals=100, trials=Trials())

# ---
import optuna

# 定义优化目标函数
def objective(trial):
# 定义参数搜索空间
param1 = trial.suggest_categorical(param1, [value1, value2, ...])
param2 = trial.suggest_float(param2, low, high)
...
# 使用params构建模型，并返回需要最小化的评估指标
return loss

# 使用Optuna进行优化
study = optuna.create_study(direction=minimize)
study.optimize(objective, n_trials=100)

# ---