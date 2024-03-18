# [1]: Hoerl A E, Kennard R W. Ridge regression: Biased estimation for nonorthogonal problems[J]. Technometrics, 1970, 12(1): 55-67.

import numpy as np
import pandas as pd
import os
''' 
导入数据
'''
file = os.path.abspath(os.path.join(os.getcwd(), ".."))  
data_file = os.path.join(file, 'data/train.csv')  
train = pd.read_csv(data_file)
data_file = os.path.join(file, 'data/test.csv')
test = pd.read_csv(data_file)
target_variable = train["y"].values
del train["y"]


from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import RobustScaler
''' 
建模
'''
# 定义一个交叉评估函数 Validation function
n_folds = 5
def rmsle_cv(model):
    kf = KFold(n_folds, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, target_variable, scoring="neg_mean_squared_error", cv = kf))
    return(rmse)

# 岭回归（Kernel Ridge Regression）  
KRR = make_pipeline(RobustScaler(), KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5))
score = rmsle_cv(KRR)
print("\nLasso score: {:.4f} ({:.4f})\n".format(score.mean(), score.std())) 


''' 
预测
'''
y_train = target_variable
x_train = train.values   
KRR.fit(x_train,y_train)
y = KRR.predict(test.values)
