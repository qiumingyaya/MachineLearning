<!-- 本文件由课程原始 Notebook 转换并翻译。代码与公式保留原样。 -->
<!-- 原文件：1. Supervised Machine Learning - Regression and Classification/week 3/C1_W3_Lab05_Cost_Function_Soln.ipynb -->

# 可选实验： 逻辑回归的代价函数（cost function for logistic regression）

## 学习目标
在本实验中，你会：
- 回顾逻辑回归代价函数（cost function for logistic regression）的实现与使用方法。

```python
import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_common import  plot_data, sigmoid, dlc
plt.style.use('./deeplearning.mplstyle')
```

## 数据集
先使用与决策边界实验相同的数据集。

```python
X_train = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])  #(m,n)
y_train = np.array([0, 0, 0, 1, 1, 1])                                           #(m,)
```

我们将使用辅助函数来绘制此数据。 标签 $y=1$ 的样本显示为红色叉号，标签 $y=0$ 的样本显示为蓝色圆点。

```python
fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X_train, y_train, ax)

# Set both axes to be from 0-4
ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.show()
```

```text
Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …
```

## 代价函数

前一个实验定义了单个样本的逻辑损失。本实验对全部样本的损失取平均，构成训练集上的**代价**。


逻辑回归的代价函数为：

$$
J(\mathbf{w},b) = \frac{1}{m} \sum_{i=0}^{m-1} \left[ loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) \right] \tag{1}
$$

其中
* $loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)})$是单个数据点的损失，即：

    $$
    loss(f_{\mathbf{w},b}(\mathbf{x}^{(i)}), y^{(i)}) = -y^{(i)} \log\left(f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) - \left( 1 - y^{(i)}\right) \log \left( 1 - f_{\mathbf{w},b}\left( \mathbf{x}^{(i)} \right) \right) \tag{2}
    $$
    
*  $m$ 是数据集中的训练样本数；模型定义为：
$$
\begin{align}
  f_{\mathbf{w},b}(\mathbf{x^{(i)}}) &= g(z^{(i)})\tag{3} \\
  z^{(i)} &= \mathbf{w} \cdot \mathbf{x}^{(i)}+ b\tag{4} \\
  g(z^{(i)}) &= \frac{1}{1+e^{-z^{(i)}}}\tag{5} 
\end{align}
$$

<a name='ex-02'></a>
#### 代码说明

`compute_cost_logistic` 循环遍历所有样本，计算并累加每个样本的逻辑损失。

请注意，`X` 和 `y` 不是标量：`X` 的形状为 $(m,n)$，`y` 的形状为 $(m,)$。其中 $n$ 是特征数，$m$ 是训练样本数。

```python
def compute_cost_logistic(X, y, w, b):
    """
    Computes cost

    Args:
      X (ndarray (m,n)): Data, m examples with n features
      y (ndarray (m,)) : target values
      w (ndarray (n,)) : model parameters  
      b (scalar)       : model parameter
      
    Returns:
      cost (scalar): cost
    """

    m = X.shape[0]
    cost = 0.0
    for i in range(m):
        z_i = np.dot(X[i],w) + b
        f_wb_i = sigmoid(z_i)
        cost +=  -y[i]*np.log(f_wb_i) - (1-y[i])*np.log(1-f_wb_i)
             
    cost = cost / m
    return cost
```

运行下面的单元格，检查代价函数的实现。

```python
w_tmp = np.array([1,1])
b_tmp = -3
print(compute_cost_logistic(X_train, y_train, w_tmp, b_tmp))
```

```text
0.36686678640551745
```

**预期输出**: 0.3668667864055175

## 示例
下面比较两组参数对应的代价值。 

* 前一个实验使用的决策边界参数为 $b=-3,w_0=1,w_1=1$，即 `b = -3, w = np.array([1,1])`.

* 比较给定的两组参数（其中一组为 $b = -4, w_0 = 1, w_1 = 1$），判断哪一个模型更好。

先绘制这两个不同 $b$ 值对应的决策边界，比较哪一个能更好地拟合数据。

* 当 $b=-3, w_0=1, w_1=1$ 时，绘制 $-3 + x_0+x_1 = 0$（蓝色）。
* 当 $b=-4, w_0=1, w_1=1$ 时，绘制 $-4 + x_0+x_1 = 0$（紫红色）。

```python
import matplotlib.pyplot as plt

# Choose values between 0 and 6
x0 = np.arange(0,6)

# Plot the two decision boundaries
x1 = 3 - x0
x1_other = 4 - x0

fig,ax = plt.subplots(1, 1, figsize=(4,4))
# Plot the decision boundary
ax.plot(x0,x1, c=dlc["dlblue"], label="$b$=-3")
ax.plot(x0,x1_other, c=dlc["dlmagenta"], label="$b$=-4")
ax.axis([0, 4, 0, 4])

# Plot the original data
plot_data(X_train,y_train,ax)
ax.axis([0, 4, 0, 4])
ax.set_ylabel('$x_1$', fontsize=12)
ax.set_xlabel('$x_0$', fontsize=12)
plt.legend(loc="upper right")
plt.title("Decision Boundary")
plt.show()
```

```text
Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …
```

从图中可以看到，这组参数对应的模型在训练数据上表现更差，代价函数的计算结果也反映了这一点。

```python
w_array1 = np.array([1,1])
b_1 = -3
w_array2 = np.array([1,1])
b_2 = -4

print("Cost for b = -3 : ", compute_cost_logistic(X_train, y_train, w_array1, b_1))
print("Cost for b = -4 : ", compute_cost_logistic(X_train, y_train, w_array2, b_2))
```

```text
Cost for b = -3 :  0.36686678640551745
Cost for b = -4 :  0.5036808636748461
```

**预期输出**

b 代价 = -3: 0.368667864055175

b 代价=-4:0.50368036748461


可以看到，`b=-4, w=np.array([1,1])` 的代价高于 `b=-3, w=np.array([1,1])`，这与图中第二条边界拟合更差的观察一致。

## 恭喜完成！
本实验回顾并使用了逻辑回归的代价函数（cost function for logistic regression）。

```python

```
