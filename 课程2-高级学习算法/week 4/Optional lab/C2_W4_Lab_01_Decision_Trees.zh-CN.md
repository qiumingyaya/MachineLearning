<!-- 本文件由课程原始 Notebook 转换并翻译。代码与公式保留原样。 -->
<!-- 原文件：2. Advanced Learning Algorithms/week 4/Optional lab/C2_W4_Lab_01_Decision_Trees.ipynb -->

# 非评分实验：决策树

本实验将演示决策树（decision tree）如何利用信息增益选择划分特征。

我们继续使用讲座中的猫分类数据集。

如讲座所述，决策树通过比较候选划分带来的**信息增益**，决定节点是否划分以及使用哪个特征。 (影像IG)

其中

$$
\text{Information Gain} = H(p_1^\text{node})- \left(w^{\text{left}}H\left(p_1^\text{left}\right) + w^{\text{right}}H\left(p_1^\text{right}\right)\right),
$$

$H$ 是熵（entropy），定义为：

$$
H(p_1) = -p_1 \log_2(p_1) - (1- p_1) \log_2(1- p_1)
$$

这里使用以 2 为底的对数。运行下面的代码，观察 $H(p)$ 随 $p$ 的变化。

注意，熵 $H$ 在 $p = 0.5$ 时最大，此时事件概率为 $0.5$、最难预测；在 $p = 0$ 或 $p = 1$ 时最小，此时结果完全可预测。因此，熵衡量了事件的不确定性。

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import *
```

```python
%matplotlib widget
_ = plot_entropy()
```

```text
Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …
```

```python

```

|                                                     |耳朵形状|脸型|胡须|是否为猫|
|:---------------------------------------------------:|:---------:|:-----------:|:---------:|:------:|
| <img src="images/0.png" alt="drawing" width="50"/> |尖耳|圆|有|    1   |
| <img src="images/1.png" alt="drawing" width="50"/> |垂耳|非圆|有|    1   |
| <img src="images/2.png" alt="drawing" width="50"/> |垂耳|圆|无|    0   |
| <img src="images/3.png" alt="drawing" width="50"/> |尖耳|非圆|有|    0   |
| <img src="images/4.png" alt="drawing" width="50"/> |尖耳|圆|有|    1   |
| <img src="images/5.png" alt="drawing" width="50"/> |尖耳|圆|无|    1   |
| <img src="images/6.png" alt="drawing" width="50"/> |垂耳|非圆|无|    0   |
| <img src="images/7.png" alt="drawing" width="50"/> |尖耳|圆|无|    1   |
| <img src="images/8.png" alt="drawing" width="50"/> |垂耳|圆|无|    0   |
| <img src="images/9.png" alt="drawing" width="50"/> |垂耳|圆|无|    0   |


把这些离散特征编码为二进制数值：

- 耳朵形状：尖耳 = 1，垂耳 = 0；
- 脸型：圆 = 1，非圆 = 0；
- 胡须：有 = 1，无 = 0。

因此训练数据包括：

- `X_train`:每个实例包含3个特征：
            - 耳朵形状（尖耳为 1，否则为 0。为 0）；
            - 脸型（圆脸为 1，否则为 0。为 0）；
            - 胡须（有为 1，否则为 0。为 0）。
            
- `y_train`：动物是否为猫：
            - 是猫时为 1；
            - 否则为 0。

```python

```

```python
X_train = np.array([[1, 1, 1],
[0, 0, 1],
 [0, 1, 0],
 [1, 0, 1],
 [1, 1, 1],
 [1, 1, 0],
 [0, 0, 0],
 [1, 1, 0],
 [0, 1, 0],
 [0, 1, 0]])

y_train = np.array([1, 1, 0, 0, 1, 1, 0, 1, 0, 0])
```

```python
#For instance, the first example
X_train[0]
```

```text
array([1, 1, 1])
```

因此第一个样本表示一只尖耳、圆脸且有胡须的动物。

在每个节点上，分别计算各候选特征的信息增益，并选择信息增益最大的特征。信息增益等于父节点熵减去两个子节点熵的加权和。

根节点包含数据集中的全部动物。$p_1^{node}$是根节点中正类（猫）的比例，所以

$$
p_1^{node} = \frac{5}{10} = 0.5
$$

下面实现熵函数。

```python
def entropy(p):
    if p == 0 or p == 1:
        return 0
    else:
        return -p * np.log2(p) - (1- p)*np.log2(1 - p)
    
print(entropy(0.5))
```

```text
1.0
```

接下来计算按每个特征划分节点时的信息增益。先实现几个辅助函数。

```python
def split_indices(X, index_feature):
    """Given a dataset and a index feature, return two lists for the two split nodes, the left node has the animals that have 
    that feature = 1 and the right node those that have the feature = 0 
    index feature = 0 => ear shape
    index feature = 1 => face shape
    index feature = 2 => whiskers
    """
    left_indices = []
    right_indices = []
    for i,x in enumerate(X):
        if x[index_feature] == 1:
            left_indices.append(i)
        else:
            right_indices.append(i)
    return left_indices, right_indices
```

因此，如果选择“耳朵形状”进行划分，那么我们必须在左侧节点(检查上表)有指数：

$$
0 \quad 3 \quad 4 \quad 5 \quad 7
$$

右子节点则包含其余样本。

```python
split_indices(X_train, 0)
```

```text
([0, 3, 4, 5, 7], [1, 2, 6, 8, 9])
```

下面实现另一个函数，计算划分后两个子节点的加权熵。

- $w^{\text{left}}$和$w^{\text{right}}$表示各子节点样本数占父节点样本数的比例；
- $p^{\text{left}}$和$p^{\text{right}}$表示各子节点中猫所占比例。

注意这两个定义的区别!特征指数0(耳朵形状)，然后在左边的节点， 一个有动物 0, 3, 4, 5和7, 我们有：

$$
w^{\text{left}}= \frac{5}{10} = 0.5 \text{ and } p^{\text{left}} = \frac{4}{5}
$$
$$
w^{\text{right}}= \frac{5}{10} = 0.5 \text{ and } p^{\text{right}} = \frac{1}{5}
$$

```python
def weighted_entropy(X,y,left_indices,right_indices):
    """
    This function takes the splitted dataset, the indices we chose to split and returns the weighted entropy.
    """
    w_left = len(left_indices)/len(X)
    w_right = len(right_indices)/len(X)
    p_left = sum(y[left_indices])/len(left_indices)
    p_right = sum(y[right_indices])/len(right_indices)
    
    weighted_entropy = w_left * entropy(p_left) + w_right * entropy(p_right)
    return weighted_entropy
```

```python
left_indices, right_indices = split_indices(X_train, 0)
weighted_entropy(X_train, y_train, left_indices, right_indices)
```

```text
0.7219280948873623
```

因此两个子节点的加权熵为 0.72。信息增益等于父节点（这里是根节点）的熵减去该值。

```python
def information_gain(X, y, left_indices, right_indices):
    """
    Here, X has the elements in the node and y is theirs respectives classes
    """
    p_node = sum(y)/len(y)
    h_node = entropy(p_node)
    w_entropy = weighted_entropy(X,y,left_indices,right_indices)
    return h_node - w_entropy
```

```python
information_gain(X_train, y_train, left_indices, right_indices)
```

```text
0.2780719051126377
```

现在，我们来计算信息收益，如果我们把根节点分成每个特征：

```python
for i, feature_name in enumerate(['Ear Shape', 'Face Shape', 'Whiskers']):
    left_indices, right_indices = split_indices(X_train, i)
    i_gain = information_gain(X_train, y_train, left_indices, right_indices)
    print(f"Feature: {feature_name}, information gain if we split the root node using this feature: {i_gain:.2f}")
```

```text
Feature: Ear Shape, information gain if we split the root node using this feature: 0.28
Feature: Face Shape, information gain if we split the root node using this feature: 0.03
Feature: Whiskers, information gain if we split the root node using this feature: 0.12
```

信息增益最大的特征就是当前节点的最佳划分特征。运行下面的代码查看划分结果。你不需要理解下面的代码块。

```python
tree = []
build_tree_recursive(X_train, y_train, [0,1,2,3,4,5,6,7,8,9], "Root", max_depth=1, current_depth=0, tree = tree)
generate_tree_viz([0,1,2,3,4,5,6,7,8,9], y_train, tree)
```

```text
 Depth 0, Root: Split on feature: 0
 - Left leaf node with indices [0, 3, 4, 5, 7]
 - Right leaf node with indices [1, 2, 6, 8, 9]
```

```text
Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …
```

建树过程是**递归**的，即我们必须为每个节点进行这些计算，直到我们达到停止标准：

- 树深度达到设定上限；
- 节点中的样本全部属于同一类别；
- 最佳划分的信息增益低于阈值。

最后一棵树是这样的：

```python
tree = []
build_tree_recursive(X_train, y_train, [0,1,2,3,4,5,6,7,8,9], "Root", max_depth=2, current_depth=0, tree = tree)
generate_tree_viz([0,1,2,3,4,5,6,7,8,9], y_train, tree)
```

```text
 Depth 0, Root: Split on feature: 0
- Depth 1, Left: Split on feature: 1
  -- Left leaf node with indices [0, 4, 5, 7]
  -- Right leaf node with indices [3]
- Depth 1, Right: Split on feature: 2
  -- Left leaf node with indices [1]
  -- Right leaf node with indices [2, 6, 8, 9]
```

```text
Canvas(toolbar=Toolbar(toolitems=[('Home', 'Reset original view', 'home', 'home'), ('Back', 'Back to previous …
```

恭喜完成本实验！
