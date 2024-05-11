import numpy as np
import matplotlib.pyplot as plt

# 生成一些随机数据
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 使用梯度下降法训练线性回归模型
def gradient_descent(X, y, learning_rate=0.01, n_iterations=1000):
    n_samples, n_features = X.shape
    weights = np.zeros((n_features, 1))
    bias = 0
    for _ in range(n_iterations):
        y_pred = np.dot(X, weights) + bias
        dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        db = (1 / n_samples) * np.sum(y_pred - y)
        weights -= learning_rate * dw
        bias -= learning_rate * db
    return weights, bias

# 训练模型
weights, bias = gradient_descent(X, y)

# 绘制数据和拟合直线
plt.scatter(X, y)
plt.plot(X, weights * X + bias, color='red')
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')
plt.show()

# 在这个例子中,weights和bias是模型的参数,通过梯度下降法学习得到;而learning_rate和n_iterations是超参数,需要手动设置。超参数的选择会影响模型的训练速度和最终的拟合效果

'''
突破最强算法模型，回归算法
突破最强算法模型，深入回归算法
突破最强算法模型，决策树 
突破最强算法模型，聚类 
突破最强算法模型，SVM
突破最强算法模型，KNN 
突破最强算法模型，随机森林 
突破最强算法模型，朴素贝叶斯
突破超强算法模型，集成学习
突破最强算法模型，XGBoost 
突破最强算法模型，LightGBM 
突破超强算法模型，逻辑回归
突破最强算法模型，相似度计算
突破机器学习核心点，特征工程
突破机器学习核心点，特征工程2 
突破机器学习核心点，特征工程3 
最强汇总，7 个特征工程核心问题
'''