
### 背景

集成树方法如随机森林（Random Forest，RF）、梯度提升树（Gradient Boosing Decision Tree, GBDT）及其变种XGBoost等由于其良好的拟合能力和模型解释性，在金融、医学统计等诸多领域有广泛的应用。

一般的RF、GBDT或XGBoost模型学习完毕后，模型会基于样本集得出各特征的重要性分数feature_importances，但是这种分数是基于样本整体得出的，模型学习完毕后各特征分数便已固定。而在实际应用中我们还关心模型对于特定个体在各个特征上的重要性评分分布，即局部解释性（local interpretation），以便于针对个体提出更精准的方案和建议。参考文献[1]中便提出了一种可以实现该目标的算法。



### 算法简介

*<u>决策树</u>*：

1. 初始化父节点；
2. 在所有特征中搜索最大增益对应的分裂特征和分裂值；
3. 执行分裂；
4. 重复2～3直至满足终止条件；



<u>*GBDT*</u>：

1. 初始化：
   $$
   s = 0
   $$
   $$
   f_0(x)=0
   $$

2. 对于所有样本$x$，计算损失函数$r_s$：
   $$
   r_s = L\left(y, \sum_{i=0}^{s}{f_s(x)}\right)
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

其中，损失函数$L$可以有以下选择：

| Settings |                        Loss Function                         |     Negative Gradient      |
| :------: | :----------------------------------------------------------: | :------------------------: |
|    LS    |           $\frac{1}{2}\left(y_i - f(x_i)\right)^2$           |       $y_i - f(x_i)$       |
|   LAD    |                       $|y_i - f(x_i)|$                       | ${\rm sign}(y_i - f(x_i))$ |
|   XGB    | $\sum_{i=1}^{n}{\left[l(y_i,\hat y^{t-1}) + g_if_t(x_i) + \frac{1}{2}h_if_t^2(x_i) \right] + \Omega(f_t)}$ |             /              |





### 参考文献

1. W.J. Fang, J. Zhou, etc.: Unpack Local Model Interpretation for GBDT, 2018

   

### Jekyll主题

Your Pages site will use the layout and styles from the Jekyll theme you have selected in your [repository settings](https://github.com/Ulti-Dreisteine/local-interpretation-for-gbdt/settings). The name of this theme is saved in the Jekyll `_config.yml` configuration file.



### 联系方式

- 我的邮箱：dreisteine262@163.com


