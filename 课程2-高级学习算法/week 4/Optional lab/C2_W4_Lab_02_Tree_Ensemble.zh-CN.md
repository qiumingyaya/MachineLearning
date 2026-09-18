<!-- 本文件由课程原始 Notebook 转换并翻译。代码与公式保留原样。 -->
<!-- 原文件：2. Advanced Learning Algorithms/week 4/Optional lab/C2_W4_Lab_02_Tree_Ensemble.ipynb -->

# 可选实验：树集成

在这个笔记本中，你会：

 - 使用 pandas 对数据集进行独热编码；
 - 使用 scikit-learn 训练决策树和随机森林，并使用 XGBoost 训练梯度提升树模型。

先导入本实验所需的库。

```python
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
!pip install xgboost --quiet
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
plt.style.use('./deeplearning.mplstyle')

RANDOM_STATE = 55 ## You will pass it to every sklearn call so we ensure reproducibility
```

# 1. 加载数据集

数据来自 [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)

背景情况
心血管疾病是全球第一大死亡原因，每年估计夺走1 790万人的生命，占全世界死亡总数的31%。 心力衰竭是心血管疾病的常见后果。本数据集包含 11 个输入特征，用于预测受试者是否患有心脏病。

患有心血管疾病或心血管风险高的人需要早期发现和管理，其中机器学习（machine learning）模型可以有很大的帮助。

你将根据这些信息训练模型，预测受试者是否患有心脏病。

#### 属性说明
- 年龄：病人年龄[岁]
- Sex：Sex（M：男，F：女）；
- ChestPainType：胸痛类型（TA：典型心绞痛，ATA：非典型心绞痛，NAP：非心绞痛性疼痛，ASY：无症状）；
- RestingBP：静息血压（mmHg）；
- Cholesterol：血清胆固醇（mg/dL）；
- FastingBS：空腹血糖是否大于 120 mg/dL（是为 1，否则为 0）；
- RestingECG：静息心电图结果（Normal：正常；ST：ST-T 波异常；LVH：按 Estes 标准判断可能或确定存在左心室肥大）；
- MaxHR：达到的最大心率（60～202）；
- ExerciseAngina：运动诱发心绞痛（Y：是，N：否）；
- Oldpeak：相对于静息状态的 ST 段压低值；
- ST_Slope：运动峰值时 ST 段斜率（Up：上升，Flat：平坦，Down：下降）；
- HeartDisease：目标标签（1：患有心脏病，0：正常）。

现在加载数据集。如上所示，变量：

- Sex
- ChestPainType
- RestingECG
- ExerciseAngina
- ST_Slope

属于分类变量，需要进行独热编码。

```python
# Load the dataset using pandas
df = pd.read_csv("heart.csv")
```

```python
df.head()
```

建模前先进行数据预处理。数据中有 5 个分类特征，下面使用 pandas 进行独热编码。

## 2. 使用单热编码pandas

二元变量无需展开成多个哑变量，因此这里只对取值数不少于 3 的分类特征进行独热编码。先统计各分类列的不同取值数。

```python
cat_variables = ['Sex',
'ChestPainType',
'RestingECG',
'ExerciseAngina',
'ST_Slope'
]
```

独热编码把一个含 $n$ 个类别的变量转换为 $n$ 个二元指示变量。

pandas 提供内置的独热编码函数 `pd.get_dummies`。本实验只使用其中几个参数。

 - `data`：待处理的 DataFrame；
 - `prefix`：新列名称使用的前缀列表；
 - `columns`：需要独热编码的列名列表。“前缀”和“列”的长度必须相同。
 
如需更多信息，可运行 `help(pd.get_dummies)` 查看完整文档。

```python
# This will replace the columns with the one-hot encoded ones and keep the columns outside 'columns' argument as it is.
df = pd.get_dummies(data = df,
                         prefix = cat_variables,
                         columns = cat_variables)
```

```python
df.head()
```

现在定义后续模型使用的最终特征矩阵和目标向量。

```python
var = [x for x in df.columns if x not in 'HeartDisease'] ## Removing our target variable
```

注意变量的数量是如何变化的。 你开始的变量有 11 个：

```python
print(len(var))
```

# 3. 分割数据集

本节把数据划分为训练集和测试集。 你将使用此函数 。`train_test_split`从scikit-learn。下面查看其主要参数。

```python
help(train_test_split)
```

```python
X_train, X_test, y_train, y_test = train_test_split(df[var], df['HeartDisease'], train_size = 0.8, random_state = RANDOM_STATE)

# We will keep the shuffle = True since our dataset has not any time dependency.
```

```python
print(f'train samples: {len(X_train)}\ntest samples: {len(X_test)}')
print(f'target proportion: {sum(y_train)/len(y_train):.4f}')
```

# 4. 构建模型

## 4.1 决策树

本节使用你已经学过的决策树（decision tree），这里使用 [scikit-learn 的实现](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html). 

scikit-learn 的决策树包含多个超参数。本实验只使用其中一部分，也不进行特征选择或系统的超参数搜索；你可以自行尝试并比较结果。


你将在此使用并调查的超参数是：

 - min samples split: 划分内部节点所需的最小样本数； 这可能会防止过拟合（overfitting）.
 - `max_depth`：树的最大深度。 这可能会防止过拟合（overfitting）.

```python
min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700] ## If the number is an integer, then it is the actual quantity of samples,
max_depth_list = [1,2, 3, 4, 8, 16, 32, 64, None] # None means that there is no depth limit.
```

```python
accuracy_list_train = []
accuracy_list_test = []
for min_samples_split in min_samples_split_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = DecisionTreeClassifier(min_samples_split = min_samples_split,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_test = model.predict(X_test) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_test = accuracy_score(predictions_test,y_test)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_test.append(accuracy_test)

plt.title('Train x Test metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_test)
plt.legend(['Train','Test'])
```

观察增大 `min_samples_split` 如何限制树的复杂度并减轻过拟合。

下面对 `max_depth` 做同样的实验。

```python
accuracy_list_train = []
accuracy_list_test = []
for max_depth in max_depth_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = DecisionTreeClassifier(max_depth = max_depth,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_test = model.predict(X_test) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_test = accuracy_score(predictions_test,y_test)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_test.append(accuracy_test)

plt.title('Train x Test metrics')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_test)
plt.legend(['Train','Test'])
```

本次划分中，`max_depth=3` 时测试准确率最高。 深度过小时，树无法充分划分正、负样本，表现为欠拟合；深度过大（例如大于 5）时，树会过度适应训练集，导致测试准确率下降，即过拟合。

- `max_depth = 3`
- `min_samples_split = 50`

```python
decision_tree_model = DecisionTreeClassifier(min_samples_split = 50,
                                             max_depth = 3,
                                             random_state = RANDOM_STATE).fit(X_train,y_train)
```

```python
print(f"Metrics train:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_train),y_train):.4f}\nMetrics test:\n\tAccuracy score: {accuracy_score(decision_tree_model.predict(X_test),y_test):.4f}")
```

训练集与测试集准确率接近，没有明显过拟合，不过整体准确率仍有提升空间。

## 4.2 随机森林

现在使用 scikit-learn 训练随机森林（random forest）。除了决策树已有的超参数，还可以通过 `n_estimators` 指定要训练的树的数量。

随机森林中的每棵树都使用训练样本和特征的随机子集进行训练。这里每次随机选择 $\sqrt{n}$ 个特征，其中 $n$ 是特征总数；该值可通过 `RandomForestClassifier` 的超参数调整；可运行 `help(RandomForestClassifier)` 查看详情。

`n_jobs` 控制并行使用的 CPU 核心数，不改变模型结果，但可以缩短训练时间。由于各棵树可独立训练，增大 `n_jobs` 会使用更多核心；若设置得接近系统总核心数，可能影响系统响应。

你会再次运行相同的脚本， 但是用另一个参数，`n_estimators`，我们将选择在10,50和100之间，默认为100。

```python
min_samples_split_list = [2,10, 30, 50, 100, 200, 300, 700]  ## If the number is an integer, then it is the actual quantity of samples,
                                             ## If it is a float, then it is the percentage of the dataset
max_depth_list = [2, 4, 8, 16, 32, 64, None]
n_estimators_list = [10,50,100,500]
```

```python
accuracy_list_train = []
accuracy_list_test = []
for min_samples_split in min_samples_split_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = RandomForestClassifier(min_samples_split = min_samples_split,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_test = model.predict(X_test) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_test = accuracy_score(predictions_test,y_test)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_test.append(accuracy_test)

plt.title('Train x Test metrics')
plt.xlabel('min_samples_split')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(min_samples_split_list )),labels=min_samples_split_list) 
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_test)
plt.legend(['Train','Test'])
```

```python
accuracy_list_train = []
accuracy_list_test = []
for max_depth in max_depth_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = RandomForestClassifier(max_depth = max_depth,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_test = model.predict(X_test) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_test = accuracy_score(predictions_test,y_test)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_test.append(accuracy_test)

plt.title('Train x Test metrics')
plt.xlabel('max_depth')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(max_depth_list )),labels=max_depth_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_test)
plt.legend(['Train','Test'])
```

```python
accuracy_list_train = []
accuracy_list_test = []
for n_estimators in n_estimators_list:
    # You can fit the model at the same time you define it, because the fit function returns the fitted estimator.
    model = RandomForestClassifier(n_estimators = n_estimators,
                                   random_state = RANDOM_STATE).fit(X_train,y_train) 
    predictions_train = model.predict(X_train) ## The predicted values for the train dataset
    predictions_test = model.predict(X_test) ## The predicted values for the test dataset
    accuracy_train = accuracy_score(predictions_train,y_train)
    accuracy_test = accuracy_score(predictions_test,y_test)
    accuracy_list_train.append(accuracy_train)
    accuracy_list_test.append(accuracy_test)

plt.title('Train x Test metrics')
plt.xlabel('n_estimators')
plt.ylabel('accuracy')
plt.xticks(ticks = range(len(n_estimators_list )),labels=n_estimators_list)
plt.plot(accuracy_list_train)
plt.plot(accuracy_list_test)
plt.legend(['Train','Test'])
```

下面使用以下参数再训练一个随机森林（random forest）：

 - `max_depth`： 8
 - `min_samples_split`： 10
 - `n_estimators`： 100

```python
random_forest_model = RandomForestClassifier(n_estimators = 100,
                                             max_depth = 8, 
                                             min_samples_split = 10).fit(X_train,y_train)
```

```python
print(f"Metrics train:\n\tAccuracy score: {accuracy_score(random_forest_model.predict(X_train),y_train):.4f}\nMetrics test:\n\tAccuracy score: {accuracy_score(random_forest_model.predict(X_test),y_test):.4f}")
```

前面每次只改变一个超参数，其余参数保持默认值，因此无法考察参数之间的组合效果。若 3 个超参数各有 4 个候选值，完整搜索需要测试 $4\times4\times4=64$ 种组合，而逐个调整只能得到 $4+4+4=12$ 组结果。scikit-learn 的 `GridSearchCV` 可以自动完成组合搜索，并可用 `refit` 参数在最佳组合上重新拟合模型。详情参见 [文档](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)。

## 4.3 XGBoost

本实验最后测试梯度提升模型 XGBoost。与随机森林中彼此独立的树不同，提升方法按顺序训练多棵树，每棵新树都用于减小已有模型的误差。

XGBoost 包含决策树的常见超参数，还包括学习率等参数。学习率控制每一步新增树对整体模型的贡献。

XGBoost 可以通过 `eval_set=[(X_val, y_val)]` 传入验证集，并在每轮迭代后计算验证指标。当指标连续若干轮不再改善时，early stopping 会终止训练，从而自动确定合适的估算器数量并降低过拟合风险。

首先从训练集中划出一个验证子集；这里不应使用测试集。

```python
n = int(len(X_train)*0.8) ## Let's use 80% to train and 20% to eval
```

```python
X_train_fit, X_train_eval, y_train_fit, y_train_eval = X_train[:n], X_train[n:], y_train[:n], y_train[n:]
```

因此可以把 `n_estimators` 设得较大，再让 early stopping 在验证指标停止改善时自动终止训练。

```python
xgb_model = XGBClassifier(n_estimators = 500, learning_rate = 0.1,verbosity = 1, random_state = RANDOM_STATE)
xgb_model.fit(X_train_fit,y_train_fit, eval_set = [(X_train_eval,y_train_eval)], early_stopping_rounds = 50)
# Here we must pass a list to the eval_set, because you can have several different tuples ov eval sets. The parameter 
# early_stopping_rounds is the number of iterations that it will wait to check if the cost function decreased or not.
# If not, it will stop and get the iteration that returned the lowest metric on the eval set.
```

虽然传入了 500 个估算器，early stopping 会在验证集指标不再改善时提前终止。日志显示训练在约 66 轮后停止，而最佳验证损失出现在第 16 棵树附近，因此最终模型采用相应数量的树。

```python
xgb_model.best_iteration
```

```python
print(f"Metrics train:\n\tAccuracy score: {accuracy_score(xgb_model.predict(X_train),y_train):.4f}\nMetrics test:\n\tAccuracy score: {accuracy_score(xgb_model.predict(X_test),y_test):.4f}")
```

可以看到，Random Forest 的准确率最高，但几个模型的结果总体接近。XGBoost 未经超参数搜索就取得了与 Random Forest 接近的测试指标，而且训练更快、可调参数更多，因此还有进一步优化空间。


恭喜，你已经学会了如何使用决策树（decision tree）, 随机森林（random forest）（均来自 scikit-learn）以及 XGBoost。
