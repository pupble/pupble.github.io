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

# **第 1 章：贝叶斯计量经济学概述**

## **1.1 贝叶斯基础**

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

当分布特别复杂时，我们用 MCMC 方法来采样。MCMC 的核心思想是构造一个马尔可夫链，使其最终停留在我们想要的分布（比如后验分布）上。

简单理解 MCMC 的工作原理:
1. 假设我们从 $p(\theta \mid y)$ 开始，第一次采样得到一个 $\theta^{(1)}$ 。
2. 然后我们用一定规则决定是否接受下一个样本 $\theta^{(2)}$ 。
3. 重复这个过程，直到得到一系列样本，这些样本最终会符合目标分布。

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

## **1.3 正态模型（未知方差）**

### **主要内容**

这一节扩展了上一节的单参数模型，将测量误差的方差 $\sigma^2$ 设置为未知量，与均值 $\mu$ 一起进行推断。通过贝叶斯方法和 Gibbs 采样，估计后验分布 $p\left(\mu, \sigma^2 \mid y\right)$ 。

### 模型描述

**观测模型**
观测数据的条件分布为:
$$
\left(y_n \mid \mu, \sigma^2\right) \sim N\left(\mu, \sigma^2\right), \quad n=1, \ldots, N
$$


其中:
- $\mu$ ：均值，未知参数；
- $\sigma^2$ ：方差，未知参数。

**先验分布**

1. **均值的先验分布:**

$$
\mu \sim N\left(\mu_0, \sigma_0^2\right),
$$

$\mu_0$ 是先验均值， $\sigma_0^2$ 是先验方差。
2. **方差的先验分布：**方差 $\sigma^2$ 只能取正值，适合的先验分布是 逆伽马分布 $I G\left(\nu_0, S_0\right)$ ，其概率密度函数为：

$$
f(x ; \alpha, \beta)=\frac{\beta^\alpha}{\Gamma(\alpha)} x^{-(\alpha+1)} e^{-\beta / x}, \quad x>0 .
$$

- $\alpha>0$ : 形状参数；
- $\beta>0$ ：尺度参数。

**逆伽马分布的性质**

- 如果 $X \sim I G(\alpha, \beta)$, 则期望和方差为:

$$
\begin{gathered}
E(X)=\frac{\beta}{\alpha-1}, \quad \text { 当 } \alpha>1 ; \\
\operatorname{Var}(X)=\frac{\beta^2}{(\alpha-1)^2(\alpha-2)}, \quad \text { 当 } \alpha>2 .
\end{gathered}
$$

### 联合后验分布的推导

通过贝叶斯定理，联合后验分布为:
$$
p\left(\mu, \sigma^2 \mid y\right) \propto p(\mu) p\left(\sigma^2\right) p\left(y \mid \mu, \sigma^2\right)
$$


展开后可得：

$$
p\left(\mu, \sigma^2 \mid y\right) \propto e^{-\frac{1}{2 \sigma_0^2}\left(\mu-\mu_0\right)^2}\left(\sigma^2\right)^{-\left(\nu_0+1\right)} e^{-\frac{S_0}{\sigma^2}} \prod_{n-1}^N\left(\sigma^2\right)^{-\frac{1}{2}} e^{-\frac{1}{2 \sigma^2}\left(y_n-\mu\right)^2} .
$$


尽管我们得到了联合后验分布的显式表达式，但直接解析计算例如后验均值 $E(\mu \mid y)$ 或方差 $\operatorname{Var}\left(\sigma^2 \mid y\right)$ 是困难的。因此，我们引入数值方法，如 Gibbs 采样，来近似这些量。

### Gibbs 采样

**基本原理**
Gibbs 采样是一种 Markov 链蒙特卡罗 (MCMC) 方法，适用于从复杂的联合分布中生成样本。其关键是通过参数的条件分布逐步采样，构建一个 Markov 链，使其收敛到目标分布。

**采样步骤**
对于参数 $\left(\mu, \sigma^2\right)$ 的联合后验分布 $p\left(\mu, \sigma^2 \mid y\right)$ ，我们分别推导两个条件分布:

1. 条件分布 $p\left(\mu \mid y, \sigma^2\right)$ :

$$
p\left(\mu \mid y, \sigma^2\right) \propto p(\mu) p\left(y \mid \mu, \sigma^2\right)
$$


后验分布为正态分布:

$$
\left(\mu \mid y, \sigma^2\right) \sim N\left(\hat{\mu}, D_\mu\right)
$$


其中:

$$
D_\mu=\left(\frac{1}{\sigma_0^2}+\frac{N}{\sigma^2}\right)^{-1}, \quad \hat{\mu}=D_\mu\left(\frac{\mu_0}{\sigma_0^2}+\frac{\sum y}{\sigma^2}\right)
$$

2. 条件分布 $p\left(\sigma^2 \mid y, \mu\right)$ :

$$
p\left(\sigma^2 \mid y, \mu\right) \propto p\left(\sigma^2\right) p\left(y \mid \mu, \sigma^2\right)
$$


后验分布为逆伽马分布:

$$
\left(\sigma^2 \mid y, \mu\right) \sim I G\left(\nu_0+\frac{N}{2}, S_0+\frac{1}{2} \sum_{n-1}^N\left(y_n-\mu\right)^2\right) .
$$

**采样算法**

1. 初始化参数 $\mu^{(0)}=a_0 、 \sigma^{2(0)}=b_0$ 。
2. 重复以下步骤 $R$ 次:
- 从 $p\left(\mu \mid y, \sigma^2\right)$ 中采样 $\mu^{(r)}$ 。
- 从 $p\left(\sigma^2 \mid y, \mu\right)$ 中采样 $\sigma^{2(r)}$ 。
3. 丢弃前 $R_0$ 次样本（称为 "burn-in" 阶段），消除初始值的影响。
4. 使用剩余样本计算感兴趣的量，例如:

$$
\hat{\mu}=\frac{1}{R-R_0} \sum_{r-R_0+1}^R \mu^{(r)}
$$



### 为什么要用吉布斯采样?

在贝叶斯分析中，我们的目标是从 联合后验分布 $p\left(\mu, \sigma^2 \mid y\right)$ 中生成样本，以便近似计算均值、方差等感兴趣的量。但通常联合后验分布复杂，无法直接采样。这时，我们就需要一种 数值方法，而 Gibbs 采样 是一种高效的解决方案。

### 联合后验分布难以直接采样的原因

1. 分布复杂：联合后验分布 $p\left(\mu, \sigma^2 \mid y\right)$ 是由先验分布和似然函数共同决定的，其公式复杂，往往没有简单的解析形式。
2. 高维问题：如果参数维度较高（如 $\left(\mu, \sigma^2\right)$ 是两个参数，甚至更多），联合分布的采样需要在高维空间中操作，非常困难。
3. 依赖关系：参数之间存在统计上的相关性，无法将联合分布拆分为独立分布的乘积。

### 吉布斯采样的基本思想

Gibbs 采样 是一种 分而治之 的方法:

- 虽然联合后验分布复杂，但我们可以分别处理 条件分布 $p\left(\mu \mid y, \sigma^2\right)$ 和 $p\left(\sigma^2 \mid y, \mu\right)$ 。
- 条件分布通常比联合分布简单得多，可以从中直接采样。
- 通过交替地从条件分布中采样，我们间接生成了联合后验分布的样本。

基本流程
1. 初始化: 给定初始值 $\mu^{(0)}$ 和 $\sigma^{2(0)}$ 。
2. 交替采样：
- 给定当前的 $\sigma^2$ ，从条件分布 $p\left(\mu \mid y, \sigma^2\right)$ 中采样新的 $\mu$ 。
- 给定当前的 $\mu$ ，从条件分布 $p\left(\sigma^2 \mid y, \mu\right)$ 中采样新的 $\sigma^2$ 。
3. 重复迭代：生成的样本 $\left(\mu^{(r)}, \sigma^{2(r)}\right)$ 会逐渐逼近目标联合分布 $p\left(\mu, \sigma^2 \mid y\right)$ 。
4. 丟弃初始样本：初始几次迭代可能受到初始值的影响（称为"burn-in"阶段），因此需要丢弃。

### 吉布斯采样的优缺点

优点:

1. 高效：在联合分布无法直接采样的情况下，Gibbs 采样分解为多个简单的条件分布采样，容易实现。
2. 适用范围广：适用于很多复杂的贝叶斯模型，特别是高维参数问题。
3. 易于实现：只需要推导出每个参数的条件分布即可。

缺点:
1. 收敛速度依赖模型：如果参数之间的相关性很强，Markov 链可能需要较多的迭代才能收敛。
2. 需要条件分布可采样：如果条件分布本身也很复杂，可能需要结合其他数值方法。

### 总结

1. **模型与推断:**
- 本节讨论了一个双参数模型（均值和方差均未知）。
- 联合后验分布难以解析，因此引入了 Gibbs 采样。
2. **Gibbs 采样的核心:**
- 通过交替从条件分布中采样，生成目标后验分布的样本。
- 使用样本估计感兴趣的量，如均值或方差。
3. **逆伽马分布的引入：**
- 作为方差的先验分布，与正态似然函数构成共轭分布。
4. **实际意义:**

- 本节的框架可扩展到更复杂的模型，如多参数或高维模型。通过数值方法，贝叶斯分析中的复杂问题可以被有效解决。

### **MATLAB 代码**

以下代码生成一个包含 $N=100$ 个观测值的数据集，其中真实均值 $\mu=3$ ，真实方差 $\sigma^2=0.1$ 。随后，代码实现 Gibbs 采样算法，通过条件分布 $p\left(\mu \mid y, \sigma^2\right)$ 和 $p\left(\sigma^2 \mid y, \mu\right)$ 交替采样。

```matlab
% norm_2para.m: 使用 Gibbs 采样估计未知方差的双参数正态模型

% 设置 Gibbs 采样的模拟参数
nsim = 10000; % 模拟采样次数
burnin = 1000; % 丢弃的“burn-in”样本数量

% 生成观测数据
N = 100; % 数据样本量
mu = 3; % 真实均值
sig2 = 0.1; % 真实方差
% 生成 N 个服从 N(μ, σ^2) 的观测值
y = mu + sqrt(sig2) * randn(N, 1);

% 定义先验分布参数
mu0 = 0; % μ 的先验均值
sig20 = 100; % μ 的先验方差
nu0 = 3; % σ^2 的先验形状参数
S0 = 0.5; % σ^2 的先验尺度参数

% 初始化马尔科夫链的起始值
mu = 0; % μ 的初始值
sig2 = 1; % σ^2 的初始值
store_theta = zeros(nsim, 2); % 用于存储 Gibbs 采样的结果

% 开始 Gibbs 采样
for isim = 1:nsim + burnin
    % 第一步：从条件分布 p(μ | y, σ^2) 中采样
    % 计算后验分布的参数
    Dmu = 1 / (1/sig20 + N/sig2); % 后验方差
    mu_hat = Dmu * (mu0/sig20 + sum(y)/sig2); % 后验均值
    % 根据正态分布 N(μ_hat, Dmu) 采样
    mu = mu_hat + sqrt(Dmu) * randn;
    
    % 第二步：从条件分布 p(σ^2 | y, μ) 中采样
    % 根据逆伽马分布的参数，采样 σ^2
    sig2 = 1 / gamrnd(nu0 + N/2, 1 / (S0 + sum((y - mu).^2) / 2));
    
    % 储存采样值（跳过 burn-in 阶段的采样）
    if isim > burnin
        isave = isim - burnin; % 调整索引以匹配存储位置
        store_theta(isave, :) = [mu, sig2]; % 存储 μ 和 σ^2 的采样结果
    end
end

% 计算后验均值作为参数的估计值
theta_hat = mean(store_theta)'; % 后验均值（μ 和 σ^2）
```

### **代码的关键步骤**

1. 数据生成:
- 模拟生成 $y_n \sim N\left(\mu, \sigma^2\right)$ 的观测值，用于验证 Gibbs 采样的准确性。
2. 定义先验分布:
- 对均值 $\mu$ 使用正态先验分布 $N\left(\mu_0, \sigma_0^2\right)$ 。
- 对方差 $\sigma^2$ 使用逆伽马分布 $I G\left(\nu_0, S_0\right)$ 。
3. Gibbs 采样过程:
- 第 1 步：从条件分布 $p\left(\mu \mid y, \sigma^2\right)$ 中采样。
- 后验分布是正态分布 $N\left(\hat{\mu}, D_\mu\right)$ :

$$
D_\mu=\left(\frac{1}{\sigma_0^2}+\frac{N}{\sigma^2}\right)^{-1}, \quad \hat{\mu}=D_\mu\left(\frac{\mu_0}{\sigma_0^2}+\frac{\sum y}{\sigma^2}\right)
$$

- 第2步：从条件分布 $p\left(\sigma^2 \mid y, \mu\right)$ 中采样。
- 条件分布是逆伽马分布 $I G\left(\nu_0+N / 2, S_0+\frac{1}{2} \sum\left(y_n-\mu\right)^2\right)$ 。
4. Burn-in 阶段:
- 丟弃前 $R_0=1000$ 个样本，去除初始值的影响。
5. 后验分析:

- 通过采样值计算参数的后验均值，作为点估计值。

### **结果分析**

1. 后验分布估计:
- 使用 10000 个后验采样值计算均值，得到:
- $\mu=3.01$ (接近真实值 3 )。
- $\sigma^2=0.12$ （接近真实值 0.1 ）。
2. 伽马分布与逆伽马分布:

- MATLAB 中 gamrnd 函数生成伽马分布样本，代码通过取倒数将其转换为逆伽马分布。

### **总结**

1. **Gibbs 采样的核心**：
   - 按照参数的条件分布交替采样，从而生成目标分布的样本。
2. **优点**：
   - 易于实现，即使目标分布复杂也可逐步近似。
   - 样本量增加可提高估计精度。
3. **实际应用**：
   - 适用于贝叶斯模型中的联合后验分布，特别是多参数问题。



以下是对 **Chapter 2 Normal Linear Regression** 中涉及的公式逐一用中文详细解读：

------

# 第2章 正态线性回归

正态线性回归模型是经济计量学中的基础模型，几乎所有灵活的模型都以其为构建基础。本章详细推导了该模型的似然函数和后验采样方法。

## **2.1 线性回归的矩阵表示**

模型公式
对于 $t=1,2, \ldots, T$ 个观测值，回归模型如下:

$$
y_t=\beta_1+x_{2, t} \beta_2+\cdots+x_{k, t} \beta_k+\varepsilon_t, \quad \varepsilon_t \sim N\left(0, \sigma^2\right),
$$


其中:
- $y_t$ : 因变量;
- $x_{2, t}, \ldots, x_{k, t}$ : 自变量;
- $\beta_1, \ldots, \beta_k$ : 回归系数;
- $\varepsilon_t$ : 误差项，独立同分布，服从正态分布 $N\left(0, \sigma^2\right)$ 。

矩阵形式
将所有 $T$ 个观测值堆眼，可以写成矩阵形式:

$$
y=X \beta+\varepsilon,
$$


其中:
- $y=\left(y_1, \ldots, y_T\right)^{\prime}$ 是因变量向量;
- $X$ 是 $T \times k$ 的设计矩阵，包含常数列和自变量；
- $\beta=\left(\beta_1, \ldots, \beta_k\right)^{\prime}$ 是回归系数向量；
- $\varepsilon=\left(\varepsilon_1, \ldots, \varepsilon_T\right)^{\prime}$ 是误差向量。

误差项 $\varepsilon$ 的分布为:

$$
\varepsilon \sim N\left(0, \sigma^2 I_T\right),
$$


即均值为 0 向量，协方差矩阵为 $\sigma^2 I_T$ 。

------

## **2.2 似然函数的推导**

#### **公式：观测数据的分布**

(y∣β,σ2)∼N(Xβ,σ2IT).(y | \beta, \sigma^2) \sim N(X\beta, \sigma^2 I_T).

- 条件于参数 β\beta 和 σ2\sigma^2，观测值 yy 是服从多元正态分布的随机向量。
- 均值为 XβX\beta：反映回归模型的线性部分。
- 协方差矩阵为 σ2IT\sigma^2 I_T：误差项带来的随机性。

#### **公式 (2.4)：似然函数**

p(y∣β,σ2)=(2πσ2)−T2exp⁡(−12σ2(y−Xβ)′(y−Xβ)).p(y | \beta, \sigma^2) = (2\pi \sigma^2)^{-\frac{T}{2}} \exp\left(-\frac{1}{2\sigma^2}(y - X\beta)'(y - X\beta)\right).

- (y−Xβ)′(y−Xβ)(y - X\beta)'(y - X\beta)：残差平方和，衡量数据 yy 偏离线性预测值 XβX\beta 的程度。
- 系数 (2πσ2)−T2(2\pi\sigma^2)^{-\frac{T}{2}}：归一化常数，确保密度函数积分为 1。
- 指数项 exp⁡(−12σ2… )\exp(-\frac{1}{2\sigma^2} \dots)：描述数据 yy 在给定参数 β,σ2\beta, \sigma^2 下出现的概率。

------

## **2.3 独立先验分布**

#### **公式 (2.5)：β\beta 的先验分布**

β∼N(β0,Vβ).\beta \sim N(\beta_0, V_\beta).

- β0\beta_0：先验均值，反映我们在观测数据之前对参数的预期。
- VβV_\beta：先验协方差矩阵，反映我们对参数不确定性的信心。

具体的先验密度函数为：

p(β)=(2π)−k2∣Vβ∣−12exp⁡(−12(β−β0)′Vβ−1(β−β0)).p(\beta) = (2\pi)^{-\frac{k}{2}} |V_\beta|^{-\frac{1}{2}} \exp\left(-\frac{1}{2}(\beta - \beta_0)'V_\beta^{-1}(\beta - \beta_0)\right).

#### **公式 (2.6)：σ2\sigma^2 的先验分布**

σ2∼IG(ν0,S0).\sigma^2 \sim IG(\nu_0, S_0).

- ν0\nu_0：形状参数，决定分布的形状。
- S0S_0：尺度参数，反映对方差的初始预期。

具体的先验密度函数为：

p(σ2)=S0ν0Γ(ν0)(σ2)−(ν0+1)exp⁡(−S0σ2).p(\sigma^2) = \frac{S_0^{\nu_0}}{\Gamma(\nu_0)} (\sigma^2)^{-(\nu_0 + 1)} \exp\left(-\frac{S_0}{\sigma^2}\right).

------

## **2.4 Gibbs 采样**

#### **目标：从后验分布采样**

后验分布 p(β,σ2∣y)p(\beta, \sigma^2 | y) 难以直接采样。Gibbs 采样通过交替采样条件分布 p(β∣y,σ2)p(\beta | y, \sigma^2) 和 p(σ2∣y,β)p(\sigma^2 | y, \beta)，间接生成联合分布的样本。

#### **公式：σ2∣y,β\sigma^2 | y, \beta 的条件分布**

p(σ2∣y,β)∼IG(ν0+T2,S0+12(y−Xβ)′(y−Xβ)).p(\sigma^2 | y, \beta) \sim IG\left(\nu_0 + \frac{T}{2}, S_0 + \frac{1}{2}(y - X\beta)'(y - X\beta)\right).

- 形状参数：ν0+T2\nu_0 + \frac{T}{2}。
- 尺度参数：S0+12(y−Xβ)′(y−Xβ)S_0 + \frac{1}{2}(y - X\beta)'(y - X\beta)，结合了先验信息和观测数据的残差平方和。

#### **公式：β∣y,σ2\beta | y, \sigma^2 的条件分布**

p(β∣y,σ2)∼N(βb,Dβ),p(\beta | y, \sigma^2) \sim N(\beta_b, D_\beta),

其中：

- 协方差矩阵： Dβ=(Vβ−1+1σ2X′X)−1.D_\beta = \left(V_\beta^{-1} + \frac{1}{\sigma^2} X'X\right)^{-1}. Vβ−1V_\beta^{-1} 和 X′X/σ2X'X / \sigma^2 分别代表先验和数据贡献的不确定性。
- 均值： βb=Dβ(Vβ−1β0+1σ2X′y).\beta_b = D_\beta \left(V_\beta^{-1}\beta_0 + \frac{1}{\sigma^2}X'y\right). βb\beta_b 是先验信息和数据信息的加权平均。

------

### **总结**

1. **似然函数和先验分布**：
   - 似然函数结合了数据 yy 的信息，反映参数的可能性。
   - 先验分布反映我们对参数的初始信念。
2. **条件分布的意义**：
   - σ2∣y,β\sigma^2 | y, \beta 是残差的分布，反映数据中的噪声大小。
   - β∣y,σ2\beta | y, \sigma^2 是参数估计，结合了数据和先验信息。
3. **Gibbs 采样**：
   - 通过交替采样 β\beta 和 σ2\sigma^2，逐步逼近联合后验分布。
   - 是处理复杂贝叶斯模型的重要工具。