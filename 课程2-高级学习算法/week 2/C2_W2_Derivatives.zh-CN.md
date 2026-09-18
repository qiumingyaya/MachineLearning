<!-- 本文件由课程原始 Notebook 转换并翻译。代码与公式保留原样。 -->
<!-- 原文件：2. Advanced Learning Algorithms/week 2/C2_W2_Derivatives.ipynb -->

# 可选实验：导数
本实验帮助你直观理解导数，展示计算简单导数的方法，并介绍一个能够进行符号求导的 Python 库。

```python
# Importing libraries for displaying derivatives
from sympy import symbols, diff
```

## 导数的直观定义

对导数的正式定义可能有点令人生畏，因为它涉及极限和“趋近于零” 。 这个想法实际上简单得多。

一个函数的导数描述一个函数的输出在输入变量有小变化时是如何变化的。

以代价函数 $J(w)$ 为例：$w$ 是输入变量，$J$ 是输出。
把输入的微小变化记为 epsilon，即 $\epsilon$。数学中常用 $\epsilon$ 或 $\Delta$ 表示很小的变化量，例如 0.001。

$$
\begin{equation}
\text{if } w \uparrow \epsilon \text{ causes }J(w) \uparrow \text{by }k \times \epsilon \text{ then}  \\
\frac{\partial J(w)}{\partial w} = k \tag{1}
\end{equation}
$$

式 (1) 表示：如果输入 $w$ 增加 $\epsilon$，输出 $J(w)$ 约增加 $k\epsilon$，那么 $J$ 对 $w$ 的导数就是 $k$。

下面用 $w=3$、$\epsilon=0.001$ 数值估计函数 $J(w)=w^2$ 的导数。

```python
J = (3)**2
J_epsilon = (3 + 0.001)**2
k = (J_epsilon - J)/0.001    # difference divided by epsilon
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k:0.6f} ")
```

把输入增加 $\epsilon=0.001$ 后，输出从 9 变为 9.006001，变化量约为 $6\epsilon$，因此 $\frac{\partial J}{\partial w}\approx6$。解析结果为 $2w$，在 $w=3$ 处正好等于 6；数值近似略有误差，是因为 $\epsilon$ 不是无穷小。

```python
J = (3)**2
J_epsilon = (3 + 0.000000001)**2
k = (J_epsilon - J)/0.000000001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")
```

```text
J = 9, J_epsilon = 9.000000006, dJ_dw ~= k = 6.000000496442226
```

减小 $\epsilon$ 后，数值结果更接近 6；可以继续减小它观察变化。

## 求符号导数
在反向传播中，需要知道简单函数在任意输入处的导数。这里关注符号导数，而非只在某一点计算数值。例如，$J(w)=w^2$ 的导数为 $\frac{\partial J(w)}{\partial w}=2w$，得到符号表达式后即可代入任意 $w$。

微积分给出了许多现成的[differentiation rules](https://en.wikipedia.org/wiki/Differentiation_rules#Power_laws,_polynomials,_quotients,_and_reciprocals)求导规则。符号计算软件可以自动应用这些规则；Python 中可使用 [SymPy](https://www.sympy.org/en/index.html)。下面查看其基本用法。

### $J = w^2$
定义 Python 变量及其对应的符号。

```python
J, w = symbols('J, w')
```

定义并打印表达式。SymPy 可生成 [LaTeX](https://en.wikibooks.org/wiki/LaTeX/Mathematics)字符串，用于显示易读的数学公式。

```python
J=w**2
J
```

```text
w**2
```

使用 SymPy 的 `diff` 计算 $J$ 对 $w$ 的导数。注意结果与我们先前的例子相符。

```python
# Derivative of J on W
dJ_dw = diff(J,w)
dJ_dw
```

```text
2*w
```

再用 `subs` 把 $w$ 替换为具体数值，在若干点上计算导数。

```python
# Subtituting with another value
dJ_dw.subs([(w,2)])    # derivative at the point w = 2
```

```text
4
```

```python
dJ_dw.subs([(w,3)])    # derivative at the point w = 3
```

```text
6
```

```python
dJ_dw.subs([(w,-3)])    # derivative at the point w = -3
```

```text
-6
```

## $J = 2w$

```python
w, J = symbols('w, J')
```

```python
J = 2 * w
J
```

```text
2*w
```

```python
dJ_dw = diff(J,w)
dJ_dw
```

```text
2
```

```python
dJ_dw.subs([(w,-3)])    # derivative at the point w = -3
```

```text
2
```

将结果与数值近似比较。

```python
J = 2*3
J_epsilon = 2*(3 + 0.001)
k = (J_epsilon - J)/0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")
```

```text
J = 6, J_epsilon = 6.002, dJ_dw ~= k = 1.9999999999997797
```

对于 $J=2w$，无论 $w$ 的起始值是多少，$w$ 的变化都会使 $J$ 发生 2 倍的变化，因此导数恒为 2；NumPy 的数值结果验证了这一点。

## $J = w^3$

```python
J, w = symbols('J, w')
```

```python
J=w**3
J
```

```text
w**3
```

```python
# Derivatives of J from W^3
dJ_dw = diff(J,w)
dJ_dw
```

```text
3*w**2
```

```python
# Subtituting with value w = 2
dJ_dw.subs([(w,2)])   # derivative at the point w=2
```

```text
12
```

将结果与数值近似比较。

```python
J = (2)**3
J_epsilon = (2+0.001)**3
k = (J_epsilon - J)/0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")
```

```text
J = 8, J_epsilon = 8.012006000999998, dJ_dw ~= k = 12.006000999997823
```

## $J = \frac{1}{w}$

```python
J, w = symbols('J, w')
```

```python
J= 1/w
J
```

```text
1/w
```

```python
# Derivatives of J from 1/w
dJ_dw = diff(J,w)
dJ_dw
```

```text
-1/w**2
```

```python
dJ_dw.subs([(w,2)])
```

```text
-1/4
```

将结果与数值近似比较。

```python
J = 1/2
J_epsilon = 1/(2+0.001)
k = (J_epsilon - J)/0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")
```

```text
J = 0.5, J_epsilon = 0.49975012493753124, dJ_dw ~= k = -0.2498750624687629
```

## $J = \frac{1}{w^2}$

```python
J, w = symbols('J, w')
```

如果时间允许，请对函数 $J = \frac{1}{w^2}$重复上述步骤，并在 $w=4$ 处求值。

```python
J, w = symbols('J, w')
```

```python
J= 1/w**2
J
```

```text
w**(-2)
```

```python
# Derivatives of J from 1/w^2
dJ_dw = diff(J,w)
dJ_dw
```

```text
-2/w**3
```

```python
# Subtituting with value w = 4
dJ_dw.subs([(w,4)])   # derivative at the point w=4
```

```text
-1/32
```

将结果与数值近似比较。

```python
J = 1/4**2
J_epsilon = 1/(4+0.001)**2
k = (J_epsilon - J)/0.001
print(f"J = {J}, J_epsilon = {J_epsilon}, dJ_dw ~= k = {k} ")
```

```text
J = 0.0625, J_epsilon = 0.06246876171484496, dJ_dw ~= k = -0.031238285155041345
```

<details>
  <summary><font size="3" color="darkgreen"><b>点击查看提示</b></font></summary>
    
```Python 
J= 1/w**2
dJ_dw = diff(J,w)
dJ_dw.subs([(w,4)])
```
  

</details>

## 恭喜完成！
通过上面的示例可以看到，导数描述函数输入发生微小变化时输出的变化率。也可以使用 Python 的 SymPy 计算符号导数。
