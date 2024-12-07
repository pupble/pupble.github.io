---
title: "贝叶斯VAR"
collection: readingnotes
type: "笔记"
permalink: /readingnotes/2024-12-7-talk-1
venue: "马朝阳"
date: 2024-12-7
location: "辽宁沈阳"
---

本文是基于[Joshua C. C. Chan](https://joshuachan.org/)教授的博士课程教材[Notes on Bayesian Macroeconometrics](https://joshuachan.org/notes_BayesMacro.html)所做笔记。

## **第 1 章：贝叶斯计量经济学概述**

### **1.1 贝叶斯基础**

贝叶斯定理:
对于事件 $A$ 和 $B$ ，贝叶斯定理表达为:

$$
P(A \mid B)=\frac{P(A) P(B \mid A)}{P(B)}
$$


其中:
- $P(A \mid B)$ ：给定事件 $B$ 发生后，事件 $A$ 的条件概率。
- $P(A)$ ：事件 $A$ 的边际概率（先验概率）。
- $P(B)$ : 事件 $B$ 的边际概率。
- $P(B \mid A)$ ：给定 $A$ 发生时，事件 $B$ 的条件概率。

解读：这个公式说明，在观察到 $B$ 的信息后，我们对 $A$ 的看法应如何调整。直观上，贝叶斯定理描述了"观察数据后如何修正先验概率以获得后验概率"。

### **贝叶斯定理在推断中的应用**

1. 先验分布 $p(\theta)$ :
- 体现研究者的先验知识或主观信念。
- 有时被认为是贝叶斯方法的弱点，但它可以将非数据信息正式纳入分析。
2. 后验分布 $p(\theta \mid y)$ :
- 综合先验信息和数据更新后的概率分布。
- 包含所有关于参数的关键信息。例如：
- 后验均值 $E(\theta \mid y)$ 可作为参数的点估计。
- 后验标准差 $\sqrt{\operatorname{Var}(\theta \mid y)}$ 表征参数的不确定性。

假设我们有一个模型，其特征由似然函数 $p(y \mid \theta)$ 描述，其中:
- $\theta$ ：模型参数的向量。
- $p(y \mid \theta)$ ：给定参数 $\theta$ 时数据 $y$ 的生成机制。

当获得一个观测样本 $y=\left(y_1, \ldots, y_T\right)^{\prime}$ 时，我们希望了解参数 $\theta$ 。贝叶斯方法的目标是获得后验分布 $p(\theta \mid y)$ ，它总结了在给定数据后对参数 $\theta$ 的所有信息。

根据贝叶斯定理，后验分布可以表示为:

$$
p(\theta \mid y)=\frac{p(\theta) p(y \mid \theta)}{p(y)} \propto p(\theta) p(y \mid \theta)
$$


其中:
- $p(\theta)$ ：先验分布，反映我们在观察数据之前对参数 $\theta$ 的主观信念。
- $p(y \mid \theta)$ ：似然函数，表示给定参数 $\theta$ 时数据 $y$ 的概率。
- $p(y)$ : 边际似然，用于归一化后验分布，计算公式为：

$$
p(y)=\int p(\theta) p(y \mid \theta) d \theta
$$

### **后验分布的计算与估计**

在实际中，后验分布往往无法解析地求解，因此需要借助数值方法或模拟方法。

#### **模拟方法：**

假设我们从后验分布 $p(\theta \mid y)$ 中获得 $R$ 个独立样本 $\theta^{(1)}, \ldots, \theta^{(R)}$ ，则:
1. 后验均值估计:

$$
\hat{\theta}=\frac{1}{R} \sum_{r=1}^R \theta^{(r)}
$$


根据大数定律， $\hat{\theta}$ 会随着 $R \rightarrow \infty$ 收敛到真实的后验均值 $E(\theta \mid y)$ 。
2. 其他统计量估计:

- 后验分布的高阶矩、分位数等都可以通过样本的模似值计算。

### **Markov链蒙特卡洛（MCMC）方法**

当后验分布过于复杂而无法直接采样时，可使用 **MCMC 方法**：

- **核心思想**： 构造一个马尔科夫链，其极限分布即为目标分布（如后验分布）。
- **性质**： MCMC 生成的样本是自相关的，但在弱正则条件下，样本均值等统计量依然可以收敛于真实值。

通过 MCMC 方法，我们可以近似计算参数的后验分布及其相关统计量。

### **总结**

1. **贝叶斯定理**：提供了根据数据更新先验知识的方法。
2. **后验分布**：包含所有关于参数的不确定性和置信信息。
3. **模拟方法**：通过样本近似后验分布，解决解析难题。
4. **MCMC 方法**：是应对复杂分布采样问题的重要工具。

## **1.2 正态模型（已知方差）**

### **问题设定**

假设我们对一个未知量 $\mu$ 进行 $N$ 次独立测量，测量值为 $y_1, \ldots, y_N$ 。已知测量误差的大小（方差 $\sigma^2$ )。此外，通过一个小型试验，我们得到了一个对 $\mu$ 的初步估计 $\mu_0$ 。
目标: 利用样本数据 $y=\left(y_1, \ldots, y_N\right)$ 的观测值，推导 $\mu$ 的后验分布 $p(\mu \mid y)$ 。

**贝叶斯分析的两部分:**

1. 似然函数 (Likelihood Function) :描述数据如何基于模型和参数生成。
2. 先验分布 (Prior Distribution) :表达在观察数据之前对参数的主观信念。

### **模型假设**

1. 正态分布假设：每个观测值 $y_n$ 的条件分布为：

$$
\left(y_n \mid \mu\right) \sim N\left(\mu, \sigma^2\right), \quad n=1, \ldots, N
$$


其中方差 $\sigma^2$ 已知。
2. 先验分布，假设 $\mu$ 的先验分布是正态分布:
$$
\mu \sim N\left(\mu_0, \sigma_0^2\right)
$$

其中 $\mu_0$ 是先验均值， $\sigma_0^2$ 是先验方差，均已知。

解读:

- 似然函数反映数据的信息。
- 先验分布反映了我们在数据观测之前的主观信念。

### **后验分布的推导**

根据贝叶斯定理，后验分布为:

$$
p(\mu \mid y) \propto p(\mu) p(y \mid \mu)
$$


其中:
- $p(y \mid \mu)$ ：似然函数。
- $p(\mu)$ : 先验分布。
#### 1.**似然函数：**

根据正态分布的概率密度函数：

$$
f\left(x ; a, b^2\right)=\frac{1}{\sqrt{2 \pi b^2}} \exp \left(-\frac{1}{2 b^2}(x-a)^2\right)
$$


所有 $N$ 个观测值的联合分布（似然函数）为：

$$
p(y \mid \mu)=\prod_{n-1}^N \frac{1}{\sqrt{2 \pi \sigma^2}} \exp \left(-\frac{1}{2 \sigma^2}\left(y_n-\mu\right)^2\right)
$$


简化为:

$$
p(y \mid \mu)=\left(2 \pi \sigma^2\right)^{-\frac{N}{2}} \exp \left(-\frac{1}{2 \sigma^2} \sum_{n-1}^N\left(y_n-\mu\right)^2\right) .
$$

#### 2.**先验分布:**

先验分布的概率密度函数为：

$$
p(\mu)=\frac{1}{\sqrt{2 \pi \sigma_0^2}} \exp \left(-\frac{1}{2 \sigma_0^2}\left(\mu-\mu_0\right)^2\right) .
$$

#### 3.后验分布:

结合似然函数和先验分布：

$$
p(\mu \mid y) \propto \exp \left(-\frac{1}{2 \sigma_0^2}\left(\mu-\mu_0\right)^2\right) \exp \left(-\frac{1}{2 \sigma^2} \sum_{n-1}^N\left(y_n-\mu\right)^2\right)
$$


将指数项展开并整理：

$$
p(\mu \mid y) \propto \exp \left(-\frac{1}{2}\left(\frac{1}{\sigma_0^2}+\frac{N}{\sigma^2}\right) \mu^2+\mu\left(\frac{\mu_0}{\sigma_0^2}+\frac{N \bar{y}}{\sigma^2}\right)\right)
$$


其中:

- $\bar{y}=\frac{1}{N} \sum_{n-1}^N y_n$ ：样本均值。

### **后验分布的形式**


可以看出， $\mu \mid y$ 的分布依然是正态分布:
$$
\mu \mid y \sim N\left(\hat{\mu}, D_\mu\right),
$$


其中:
1. **后验均值:**

$$
\hat{\mu}=\left(\frac{1}{\sigma_0^2}+\frac{N}{\sigma^2}\right)^{-1}\left(\frac{\mu_0}{\sigma_0^2}+\frac{N \bar{y}}{\sigma^2}\right) .
$$


这是先验均值 $\mu_0$ 和样本均值 $\bar{y}$ 的加权平均，权重分别为:
- 先验方差的倒数 $\frac{1}{\sigma_0^2}$ 。
- 样本均值的方差倒数 $\frac{N}{\sigma^2}$ 。
2. **后验方差：**

$$
D_\mu=\left(\frac{1}{\sigma_0^2}+\frac{N}{\sigma^2}\right)^{-1} .
$$


解读:
- 数据量 $N$ 增加时，样本均值的权重增加，后验分布更接近样本信息。
- 当先验方差 $\sigma_0^2$ 较小时，先验均值的权重增加，反映了我们对先验信息的更强信念。

### **数值例子**

假设：
- $\mu_0=10$ (先验均值)， $\bar{y}=20$ (样本均值)。
- $\sigma_0^2=\sigma^2=1$ (方差相同)。

两种情形：
1. $N=1$ :
- 后验均值为两者的中点 $\hat{\mu}=15$ 。
- 后验分布较分散。
2. $N=10$ :
- 后验均值更接近样本均值 $\bar{y}=20$ 。
- 后验分布更集中，表明数据提供了更多信息。

```matlab
% 使用蒙特卡罗积分估计 E[log|μ| | y]
R = 10000; % 蒙特卡罗模拟的样本数量

% 定义后验分布的参数
mu_hat = 19.09; % 后验分布的均值
Dmu = 0.09; % 后验分布的方差

% 从 N(mu_hat, Dmu) 分布生成 R 个样本
% randn(R,1) 生成 R 个服从标准正态分布 N(0, 1) 的随机数
% sqrt(Dmu)*randn(R,1) 将标准正态分布缩放至方差为 Dmu 的分布
% mu_hat + sqrt(Dmu)*randn(R,1) 将均值移动到 mu_hat
mu = mu_hat + sqrt(Dmu) * randn(R, 1);

% 对每个样本应用 log|μ| 函数
% abs(mu) 计算每个样本的绝对值
% log(abs(mu)) 计算绝对值的自然对数
% mean(log(abs(mu))) 计算 log|μ| 的样本均值，近似 E[log|μ| | y]
g_hat = mean(log(abs(mu)));

% g_hat 是 E[log|μ| | y] 的蒙特卡罗估计值
```

------

### **总结**

1. **后验均值是加权平均**： 数据量越大，样本信息占比越高；先验方差越小，先验信息占比越高。
2. **不确定性随数据量减小**： 随着 NN 增加，后验分布更加集中，反映了更小的不确定性。
3. **蒙特卡罗积分**： 对于无法解析的积分问题，可以通过模拟方法（如 MCMC）估计各种统计量。

此例子奠定了贝叶斯推断的基本框架，即结合先验与数据进行分析，是后续复杂模型的基础。