<script type="text/x-mathjax-config">
    MathJax.Hub.Config({
      tex2jax: {
        skipTags: ['script', 'noscript', 'style', 'textarea', 'pre'],
        inlineMath: [['$','$']]
      }
    });
</script>
<script src="https://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML" type="text/javascript"></script>

***
### 算法背景

集成树方法如随机森林（Random Forest，RF）、梯度提升树（Gradient Boosing Decision Tree, GBDT）及其变种XGBoost等由于其良好的拟合能力和模型解释性，在金融、医学统计等诸多领域有广泛的应用。

一般的RF、GBDT或XGBoost模型学习完毕后，模型会基于样本集得出各特征的重要性分数feature_importances，但是这种分数是基于样本整体得出的，模型学习完毕后各特征分数便已固定。而在实际应用中我们还关心模型对于特定个体在各个特征上的重要性评分分布，即局部解释性（local interpretation），以便于针对个体提出更精准的方案和建议。参考文献[1]中便提出了一种可以实现该目标的算法。

***
### 算法简介

<u>***决策树***</u>：

1. 初始化父节点；
2. 在所有特征中搜索最大增益对应的分裂特征和分裂值；
3. 执行分裂；
4. 重复2～3直至满足终止条件；  


<u>***GBDT***</u>：

1. 初始化:  

   $$
   s = 0
   $$

   $$
   f_0(x) = 0
   $$

2. 对于所有样本$x$，计算损失函数$r_s$：  

   $$
   r_s = L\left(y, \sum_{i=0}^{s}{f_s(x)} \right)
   $$

3. 对残差$r_s$进行拟合，得到回归树$h_s$；

4. 更新：  

   $$
   f_{s+1}(x)=f_s(x)+h_s(x)
   $$

   $$
   s = s + 1
   $$

5. 重复以上步骤2~4，直至满足终止条件；

其中，GBDT的损失函数$L$可以有以下选择：  

| Settings |                        Loss Function                         |     Negative Gradient     |
| :------: | :----------------------------------------------------------: | :-----------------------: |
|    LS    |           $\frac{1}{2}\left(y_i - f(x_i)\right)^2$         |      $y_i - f(x_i)$      |
|   LAD    |                       $\|y_i - f(x_i)\|$                      | ${\rm sign} (y_i - f(x_i))$ |
|   XGB    | $\sum_{i=1}^{n} {[l(y_i,\hat y^{t-1}) + g_if_t(x_i) + \frac{1}{2}h_if_t^2(x_i)] + \Omega(f_t)}$ |/|
|Absolute |$\vert y_i - f(x_i)\vert $|/|
|Huber|$ \frac{1}{2}(y_i - f(x_i))^2, {\rm if} \quad \vert y_i -f(x_i) \vert \leq \delta \\ \delta(\vert y_i - f(x_i) \vert - \frac{\sigma}{2}), {\rm if} \quad \vert y_i - f(x_i) \vert > \delta$  |/|

注意：
* 基于残差的GBDT在解决回归问题上不是好的选择，因为对异常值过于敏感，一般回归类损失函数会选用绝对损失函数或者huber损失函数来代替平方损失函数[2]。  

GBDT在对每个基学习器回归的过程中都希望损失函数最小化，设损失函数为：  

$$
L = \frac{1}{2}\left(y_i - f(x_i)\right)^2
$$

则1阶导数为：  

$$
\frac{\partial L}{\partial f(x_i)} = f(x_i) - y_i
$$

此时残差与负梯度一致：  

$$
y_i - f(x_i) = - \frac{\partial L}{\partial f(x_i)}
$$

<u>***GBDT的局部解释***</u>

假设我们基于样本集$X$对目标$y$进行回归得到GBDT模型$f$，其中$X$有$N$个样本，$m$个特征，而模型$f$中的$k$个基学习器表示为$\{f_0, f_1, f_2, ..., f_k\}$，则有：  

$$
\hat y = \sum_{i=0}^{k}{f_i(x)}
$$

(未完待续)


***
### 参考文献

1. W.J. Fang, J. Zhou, etc.: Unpack Local Model Interpretation for GBDT, 2018
2. https://zhuanlan.zhihu.com/p/29765582 "一文理解GBDT"



***
### 联系方式

- 我的邮箱：dreisteine262@163.com
