<!-- 本文件由课程原始 Notebook 转换并翻译。代码与公式保留原样。 -->
<!-- 原文件：1. Supervised Machine Learning - Regression and Classification/week 3/C1_W3_Lab03_Decision_Boundary_Soln.ipynb -->

# 可选实验：逻辑回归与决策边界（decision boundary）

## 学习目标
在本实验中，你会：
- 绘制逻辑回归模型的决策边界，以便更直观地理解模型的预测结果。

```python
import numpy as np
%matplotlib widget
import matplotlib.pyplot as plt
from lab_utils_common import plot_data, sigmoid, draw_vthresh
plt.style.use('./deeplearning.mplstyle')
```

## 数据集

假设有如下训练数据集：
- 输入变量 `X` 是一个 NumPy 数组，包含 6 个训练样本，每个样本有两个特征。
- 输出变量 `y` 也是一个包含 6 个元素的 NumPy 数组，每个元素的取值为 `0` 或 `1`。

```python
X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1]).reshape(-1,1)
```

```python
X
```

```text
array([[0.5, 1.5],
       [1. , 1. ],
       [1.5, 0.5],
       [3. , 0.5],
       [2. , 2. ],
       [1. , 2.5]])
```

```python
y
```

```text
array([[0],
       [0],
       [0],
       [1],
       [1],
       [1]])
```

### 绘制数据

下面使用辅助函数绘制数据。标签为 $y=1$ 的样本显示为红色叉号，标签为 $y=0$ 的样本显示为蓝色圆点。

```python
fig,ax = plt.subplots(1,1,figsize=(4,4))
plot_data(X, y, ax)

ax.axis([0, 4, 0, 3.5])
ax.set_ylabel('$x_1$')
ax.set_xlabel('$x_0$')
plt.show()
```

```text
Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …
```

## 逻辑回归模型


* 假设要训练一个形如下式的逻辑回归模型：

  $f(x) = g(w_0x_0+w_1x_1 + b)$
  
其中，$g(z) = \frac{1}{1+e^{-z}}$ 是 sigmoid 函数。


* 假设模型训练后得到参数 $b=-3$、$w_0=1$、$w_1=1$，则：

  $f(x) = g(x_0+x_1-3)$

（后续课程将介绍如何根据数据学习这些参数。）
  
  
下面通过绘制决策边界来理解这个训练好的模型。

### 复习：逻辑回归（logistic regression）和决策边界

* 回顾逻辑回归模型，其形式为：

  $$
  f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(\mathbf{w} \cdot \mathbf{x}^{(i)} + b) \tag{1}
  $$

其中，$g(z)$ 是 sigmoid 函数，它把任意输入映射到 0 和 1 之间：

$$
g(z) = \frac{1}{1+e^{-z}}\tag{2}
$$
$\mathbf{w}\cdot\mathbf{x}$ 表示向量点积：
  
  $$
  \mathbf{w} \cdot \mathbf{x} = w_0 x_0 + w_1 x_1
  $$
  
  
* 模型输出 $f_{\mathbf{w},b}(\mathbf{x})$ 可解释为：在给定输入 $\mathbf{x}$ 和参数 $\mathbf{w},b$ 时，$y=1$ 的概率。
* 为了把概率转换为类别 $y=0$ 或 $y=1$，可以使用以下阈值规则：

若 $f_{\mathbf{w},b}(\mathbf{x}) \ge 0.5$，则预测 $y=1$。
  
若 $f_{\mathbf{w},b}(\mathbf{x}) < 0.5$，则预测 $y=0$。
  
  
下面绘制 sigmoid 函数，观察 $g(z) \ge 0.5$ 在什么条件下成立。

```python
# Plot sigmoid(z) over a range of values from -10 to 10
z = np.arange(-10,11)

fig,ax = plt.subplots(1,1,figsize=(5,3))
# Plot z vs sigmoid(z)
ax.plot(z, sigmoid(z), c="b")

ax.set_title("Sigmoid function")
ax.set_ylabel('sigmoid(z)')
ax.set_xlabel('z')
draw_vthresh(ax,0)
```

```text
Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …
```

* 如图所示，当 $g(z) \ge 0.5$ 时，$z \ge 0$。

* 对于逻辑回归模型，$z=\mathbf{w}\cdot\mathbf{x}+b$，因此：

若 $\mathbf{w}\cdot\mathbf{x}+b \ge 0$，模型预测 $y=1$。
  
若 $\mathbf{w}\cdot\mathbf{x}+b < 0$，模型预测 $y=0$。
  
  
  
### 绘制决策边界

现在回到前面的示例，看看逻辑回归模型如何利用决策边界进行预测。

* 该逻辑回归模型的形式为：

  $f(\mathbf{x}) = g(-3 + x_0+x_1)$


* 根据上面的阈值规则，当 $-3+x_0+x_1 \ge 0$ 时，模型预测 $y=1$。

先绘制直线 $-3+x_0+x_1=0$，等价地写为 $x_1=3-x_0$。

```python
# Choose values between 0 and 6
x0 = np.arange(0,6)

x1 = 3 - x0
fig,ax = plt.subplots(1,1,figsize=(5,4))
# Plot the decision boundary
ax.plot(x0,x1, c="b")
ax.axis([0, 4, 0, 3.5])

# Fill the region below the line
ax.fill_between(x0,x1, alpha=0.2)

# Plot the original data
plot_data(X,y,ax)
ax.set_ylabel(r'$x_1$')
ax.set_xlabel(r'$x_0$')
plt.show()
```

```text
Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …
```

* 上图蓝线表示 $x_0+x_1-3=0$。令 $x_0=0$ 得到 $x_1=3$，因此它与 $x_1$ 轴交于 3；令 $x_1=0$ 得到 $x_0=3$，因此也与 $x_0$ 轴交于 3。 


* 阴影区域代表$-3 + x_0+x_1 < 0$。线上区域是$-3 + x_0+x_1 > 0$.


* 阴影区域(线下)的任何点被归类为$y=0$。线上或线上以上的任何点都被归类为$y=1$。这一行被称为 "决策边界".

如讲座所示，使用更高次的多项式（例如 $f(x) = g( x_0^2 + x_1 -1)$）可以得到更复杂的非线性决策边界。

## 恭喜完成！
你已经了解了逻辑回归中的决策边界及其与模型预测之间的关系。

```python

```

```python

```
