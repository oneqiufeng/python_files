# 01 linear regression

## 线性回归（linear regression）

### 简单线性回归

用于建模和分析变量之间的统计方法，预测一个或者多个自变量之间的关系；简单线性回归只有一个自变量，多元线性回归则有多个自变量。

线性回归是统计学中最基础的预测模型之一，它假设输入变量（自变量）和输出变量（因变量）之间存在线性关系。在Python中，我们可以使用 `scikit-learn`库来实现线性回归模型。以下是一个简单的线性回归代码示例：

```python
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
```

这段代码首先导入了必要的库，然后创建了一些示例数据。接着，它实例化了一个 `LinearRegression`模型，并使用 `fit`方法来训练模型。之后，使用 `predict`方法来进行预测，并将结果打印出来。最后，使用 `matplotlib`库来可视化实际数据点和预测的线性关系。

请注意，为了运行这段代码，你需要确保你的Python环境中已经安装了 `scikit-learn`和 `matplotlib`库。如果没有安装，可以使用以下命令进行安装：

```bash
pip install scikit-learn matplotlib
```

这个例子是非常基础的，实际应用中可能需要对数据进行预处理，比如特征缩放、处理缺失值等，以及对模型进行评估，比如计算R-squared、均方误差等指标。

简单线性回归（Simple Linear Regression）是统计学中最基础的预测模型之一，它用于研究两个变量之间的线性关系。其中一个变量是自变量（解释变量，通常表示为 \( x \)），另一个变量是因变量（响应变量，通常表示为 \( y \)）。简单线性回归模型试图找到一条直线，这条直线最好地拟合了由自变量和因变量组成的数据点集合。

#### 数学原理

当然，以下是使用Markdown格式的LaTeX语法来展示简单线性回归的数学原理：

简单线性回归模型可以表示为：

$$
y = \beta_0 + \beta_1 x + \epsilon
$$

其中：

- y 代表因变量，也就是我们想要预测的变量的值。
- x 代表自变量，也就是用来预测因变量的变量的值。
- beta0 代表截距，这是一个常数项，它表示当自变量 x 为零时，因变量 y 的预期值。
- beta1 代表斜率，这是一个系数，它描述了自变量 x 每变化一个单位，因变量 y 预期会如何变化。
- epsilon 代表误差项，这是实际观测值与模型预测值之间差异的量度，反映了模型未能解释的随机变异。

最小二乘法的目标是最小化误差项的平方和：

$$
\min_{\beta_0, \beta_1} \sum_{i=1}^{n} (y_i - \beta_0 - \beta_1 x_i)^2
$$

参数 \( beta_1 \)（斜率）和 \( beta_0 \)（截距）的计算公式如下：

$$
\beta_1 = \frac{n \sum(x_i y_i) - \sum(x_i) \sum(y_i)}{n \sum(x_i^2) - (\sum(x_i))^2}
$$

$$
\beta_0 = \frac{\sum(y_i)}{n} - \beta_1 \frac{\sum(x_i)}{n}
$$

### 多元线性回归

多元线性回归是线性回归的一个扩展，它允许我们使用多个自变量（特征）来预测因变量。在Python中，我们同样可以使用 `scikit-learn`库中的 `LinearRegression`模型来实现多元线性回归。下面是一个多元线性回归的示例代码，包括算法的实现和样例数据的使用。

```python
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
```

在这个例子中，我们创建了一个包含三个自变量（X1, X2, X3）的多元线性回归模型。`X`是一个二维数组，其中每一行代表一个样本，每一列代表一个特征。`y`是因变量的值。

我们使用 `LinearRegression`模型的 `fit`方法来训练模型，并使用 `predict`方法来进行预测。模型的权重（斜率）和截距可以通过 `model.coef_`和 `model.intercept_`属性获得。

最后，我们使用 `matplotlib`库来可视化其中一个特征（X1）与因变量（y）的关系，以及模型预测的线性趋势。

请注意，多元线性回归的可视化通常比较复杂，因为我们不能在二维平面上直观地展示多维空间的关系。因此，上面的代码只展示了一个特征与因变量的关系。如果你想要更全面地可视化所有特征与因变量的关系，你可能需要使用降维技术（如PCA）或者为每个特征单独绘制图表。

### 随机数据进行多元回归

下面是一个使用随机生成的数据进行多元线性回归的Python代码示例。我们将使用 `numpy`库来生成随机数据，并使用 `scikit-learn`库来实现多元线性回归模型。

```python
# 导入必要的库
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 设置随机种子以获得可重现的结果
np.random.seed(0)

# 随机生成自变量X和因变量y的数据
# 假设有5个自变量和一个因变量
n_samples = 30  # 样本数量
n_features = 5   # 自变量数量

# 随机生成自变量X
X = np.random.randn(n_samples, n_features)

# 随机生成模型参数（权重和截距）
np.random.seed(1)  # 为了可重现性，改变随机种子
coef = np.random.randn(n_features) * 2  # 假设权重是随机的，这里乘以2来增加变化范围
intercept = np.random.randn()  # 随机生成截距

# 计算因变量y
y = np.dot(X, coef) + intercept + np.random.randn(n_samples) * 5  # 加上截距和随机噪声

# 创建多元线性回归模型实例
model = LinearRegression()

# 训练模型，拟合数据
model.fit(X, y)

# 预测
predictions = model.predict(X)

# 打印模型参数
print(f'估计的权重（斜率）: {model.coef_}')
print(f'估计的截距: {model.intercept_}')

# 可视化结果
# 为了可视化，我们可以选择其中一个自变量X1和因变量y进行展示
plt.scatter(X[:, 0], y, color='blue', label='Actual Data')  # 实际数据点
plt.plot(X[:, 0], predictions, color='red', linewidth=2, label='Predicted Line')  # 预测的线性关系
plt.title('多元线性回归模型')
plt.xlabel('X1')
plt.ylabel('y')
plt.legend()
plt.show()
```

在这个代码中，我们首先设置了随机种子，以确保每次运行代码时都能生成相同的随机数据。然后，我们使用 `numpy`的 `random.randn`函数生成了自变量 `X`和因变量 `y`。`y`是根据随机生成的权重、截距和噪声计算得出的。

接着，我们创建了一个 `LinearRegression`模型实例，并使用 `fit`方法来训练模型。然后，我们使用 `predict`方法来预测因变量的值，并打印出模型估计的权重和截距。

最后，我们使用 `matplotlib`库来可视化其中一个自变量 `X1`与因变量 `y`的关系，以及模型预测的线性趋势。

请注意，由于我们使用了随机生成的数据，每次运行代码时结果都会有所不同。此外，由于自变量之间可能存在多重共线性，这可能会影响模型的稳定性和预测能力。在实际应用中，可能需要进一步的数据预处理和模型评估来确保模型的有效性。
