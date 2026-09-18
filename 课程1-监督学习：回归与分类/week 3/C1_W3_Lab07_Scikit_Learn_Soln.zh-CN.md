<!-- 本文件由课程原始 Notebook 转换并翻译。代码与公式保留原样。 -->
<!-- 原文件：1. Supervised Machine Learning - Regression and Classification/week 3/C1_W3_Lab07_Scikit_Learn_Soln.ipynb -->

# 非评分实验：使用 scikit-learn 进行逻辑回归

## 学习目标
在本实验中你会：
- 使用 scikit-learn 训练逻辑回归模型。

## 数据集
让我们从相同的数据集开始。

```python
# Creating dataset
import numpy as np

X = np.array([[0.5, 1.5], [1,1], [1.5, 0.5], [3, 0.5], [2, 2], [1, 2.5]])
y = np.array([0, 0, 0, 1, 1, 1])
```

## 拟合模型

下面从 scikit-learn 导入 [`LogisticRegression` 模型](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression)，并调用 `fit` 训练模型。

```python
# Defining model
from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()

# Training
lr_model.fit(X, y)
```

```text
LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                   intercept_scaling=1, l1_ratio=None, max_iter=100,
                   multi_class='auto', n_jobs=None, penalty='l2',
                   random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                   warm_start=False)
```

## 进行预测

调用 `predict` 可以得到模型的类别预测。

```python
# Prediction
y_pred = lr_model.predict(X)

print("Prediction on training set:", y_pred)
```

```text
Prediction on training set: [0 0 0 1 1 1]
```

## 计算准确率

调用模型的 `score` 方法可以计算分类准确率。

```python
# Calculating accuracy
print("Accuracy on training set:", lr_model.score(X, y))
```

```text
Accuracy on training set: 1.0
```
