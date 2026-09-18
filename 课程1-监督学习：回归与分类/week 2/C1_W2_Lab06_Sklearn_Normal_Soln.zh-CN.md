<!-- 本文件由课程原始 Notebook 转换并翻译。代码与公式保留原样。 -->
<!-- 原文件：1. Supervised Machine Learning - Regression and Classification/week 2/C1_W2_Lab06_Sklearn_Normal_Soln.ipynb -->

# 可选实验：使用 scikit-learn 的闭式解线性回归

[scikit-learn](https://scikit-learn.org/stable/index.html) 是一个开源、可用于商业项目的机器学习工具库，实现了本课程涉及的多种算法。

## 学习目标
在本实验中你会：
- 使用 scikit-learn 通过闭式解拟合线性回归模型。

## 所用工具
你将使用 scikit-learn、Matplotlib 和 NumPy。

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from lab_utils_multi import load_house_data
plt.style.use('./deeplearning.mplstyle')
np.set_printoptions(precision=2)
```

<a name="toc_40291_2"></a>
# 线性回归的闭式解
scikit-learn 提供了 [`LinearRegression` 模型](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression)，可通过闭式求解拟合线性回归。

使用前面实验中的数据：一套面积为 1,000 平方英尺的房屋售价为 \$300,000，另一套面积为 2,000 平方英尺的房屋售价为 \$500,000。

|大小(1 000 sqft)|价格(1 000美元)|
| ----------------| ------------------------ |
| 1               | 300                      |
| 2               | 500                      |

### 加载数据集

```python
X_train = np.array([1.0, 2.0])   #features
y_train = np.array([300, 500])   #target value
```

### 创建并拟合模型
下面使用 scikit-learn 拟合线性回归模型。 
第一步创建回归器对象。
第二步调用回归器的 `fit` 方法，根据输入数据拟合参数。scikit-learn 要求特征矩阵 `X` 为二维数组。

```python
linear_model = LinearRegression()
#X must be a 2-D Matrix
linear_model.fit(X_train.reshape(-1, 1), y_train)
```

### 查看参数
拟合后的 $\mathbf{w}$ 和 $b$ 分别保存在 scikit-learn 模型的 `coef_` 与 `intercept_` 属性中。

```python
b = linear_model.intercept_
w = linear_model.coef_
print(f"w = {w:}, b = {b:0.2f}")
print(f"'manual' prediction: f_wb = wx+b : {1200*w + b}")
```

### 进行预测

调用 `predict` 方法生成预测值。

```python
y_pred = linear_model.predict(X_train.reshape(-1, 1))

print("Prediction on training set:", y_pred)

X_test = np.array([[1200]])
print(f"Prediction for 1200 sqft house: ${linear_model.predict(X_test)[0]:0.2f}")
```

## 第二个示例
第二个示例使用前面实验中的多特征数据。最终参数和预测值与长时间运行梯度下降的结果很接近，但闭式解几乎立即得到结果。闭式解适合这类小数据集；数据规模很大时，其计算和内存开销可能较高。
> 闭式解无需为了梯度下降的收敛速度而进行特征缩放。

```python
# load the dataset
X_train, y_train = load_house_data()
X_features = ['size(sqft)','bedrooms','floors','age']
```

```python
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)
```

```python
b = linear_model.intercept_
w = linear_model.coef_
print(f"w = {w:}, b = {b:0.2f}")
```

```python
print(f"Prediction on training set:\n {linear_model.predict(X_train)[:4]}" )
print(f"prediction using w,b:\n {(X_train @ w + b)[:4]}")
print(f"Target values \n {y_train[:4]}")

x_house = np.array([1200, 3,1, 40]).reshape(-1,4)
x_house_predict = linear_model.predict(x_house)[0]
print(f" predicted price of a house with 1200 sqft, 3 bedrooms, 1 floor, 40 years old = ${x_house_predict*1000:0.2f}")
```

## 恭喜完成！
在本实验中你：
- 使用了开源机器学习工具库 scikit-learn；
- 使用 `LinearRegression` 的闭式求解方法拟合了线性回归模型。

```python

```
