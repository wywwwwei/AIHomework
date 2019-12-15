# 逻辑回归

## 似然估计

二元逻辑回归的一般设置

- n个观察对象（训练样本）{x<sub>i</sub> , y<sub>i</sub> }，i = 1……n
- y<sub>i</sub> ∈ { 0 , 1 }

所以
$$
p(x_{i}) = Pr(y_{i}=1|x_{i})\\
Pr(y_{i}=0|x_{i}) = 1 - p(x_{i})
$$


> 几率定义：一个事件的几率（odds）是指该事件发生的概率p与该事件不发生的概率1-p的比值$\frac{p}{1-p}$
> 对数几率即对几率求对数 $logit(p) = log\frac{p}{1-p}$
>
> 在逻辑回归中，几率表示预测因子X对一个结果发生的可能性的持续影响。

逻辑回归模型允许我们在二进制结果变量和一组预测变量之间建立关系。 它将对数转换后的概率建模为与预测变量的线性关系。

> 线性回归是对于特征的线性组合来拟合真实标记( y = wx +b )
>
> 逻辑回归是对于特征的线性组合来拟合真实标记为正例的概率的对数几率 ($ln\frac{y}{1-y}$  = wx + b)

$$
log\frac{p(x_{i})}{1-p(x_{i})} = \beta_{0}+\beta_{1}x_{i,1}+\cdots+\beta_{p}x_{i,j} = \sum_{j=0}^{p}\beta_{j}x_{i,j}
$$

> 这里我们将 x<sub>i,0</sub>  视为等于 1

写成向量表示法
$$
log\frac{p(x_{i};\beta)}{1-p(x_{i};\beta)}=x_{i}\beta\\
row-vector\;x_{i}=\begin{bmatrix}1&x_{i,1}&x_{i,2}&\cdots&x_{i,p}\end{bmatrix}=\tilde{X}_{i}\\column-vector\;\beta = \begin{bmatrix}\beta_{0}\\\beta_{1}\\\beta_{2}\\\cdots\\\beta_{p}\end{bmatrix}=\tilde{W}^{T}
$$
所以
$$
p(x_{i};\beta)=\frac{e^{x_{i}\beta}}{1+e^{x_{i}\beta}},\quad1-p(x_{i};\beta)=\frac{1}{1+e^{x_{i}\beta}}\\
$$
则第i个观察对象(样本)的似然函数为
$$
\begin{align}
L_{i}(\beta|x_{i})&=(1-y_{i})log[1-p(x_{i};\beta)]+y_{i}\log{p(x_{i};\beta)}\\&=log[1-p(x_{i};\beta)]+y_{i}\{log(p(x_{i};\beta))-log[1-p(x_{i};\beta)]\}\\&=log[1-p(x_{i};\beta)]+y_{i}\log{\frac{p(x_{i};\beta)}{1-p(x_{i};\beta)}}
\end{align}
$$

> 好比一个样本的伯努利二项分布binomial(1,p(x<sub>i</sub>))

然后，联合(joint)n个观察对象(样本)
$$
\begin{align}
L(\beta|x_{1},\cdots,x_{n}) &=\sum_{i=1}^{n}L_{i}(\beta|x_{i})\\&=\sum_{i=1}^{n}\{log[1-p({x_{i};\beta})]+y_{i}\log{\frac{p(x_{i};\beta)}{1-p(x_{i};\beta)}}\}\\&=\sum_{i=1}^{n}(y_{i}\tilde{W}^{T}\tilde{X}_{i}-log(1-p(x_{i};\beta))^{-1})\\&=\sum_{i=1}^{n}(y_{i}\tilde{W}^{T}\tilde{X}_{i}-log(1+\frac{p(x_{i};\beta)}{1-p(x_{i};\beta)})\\&=\sum_{i=1}^{n}[y_{i}\tilde{W}^{T}\tilde{X}_{i}-log(1+e^{\tilde{W}^{T}\tilde{X}_{i}})]
\end{align}
$$

> 剩下的任务是就是通过MLE解决优化问题

## 梯度回传

> 注意：在上面定义了 $\beta = \tilde{W}^{T}$ 和 $x_{i}=\tilde{X}_{i}$

为了最大化对数似然函数，我们有两个方法

1. 对对数似然函数进行梯度上升方法

   $$
   \begin{align}\frac{\partial{L(\beta)}}{\partial{\beta_{j}}}&=\sum_{i=1}^{n}\frac{\partial{[y_{i}\tilde{W}^{T}\tilde{X}_{i}-log(1+e^{\tilde{W}^{T}\tilde{X}_{i}})]}}{\partial{\beta}_{j}}\\&=\sum_{i=1}^{n}[y_{i}-\frac{e^{\tilde{W}^{T}\tilde{X}_{i}}}{1+e^{\tilde{W}^{T}\tilde{x}_{i}}}]\frac{\partial{\tilde{W}^{T}\tilde{X}_{i}}}{\partial{\tilde{\beta_{j}}}}\\&=\sum_{i=1}^{n}[(y_{i}-\frac{e^{\tilde{W}^{T}\tilde{X}_{i}}}{1+e^{\tilde{W}^{T}\tilde{X}_{i}}})\tilde{X}_{i,j}]\end{align}
   $$
   所以，重复
   $$
   \tilde{W}_{j}:=\tilde{W}_{j}+\alpha \sum_{i=1}^{n}[(y_{i}-\frac{e^{\tilde{W}^{T}\tilde{X}_{i}}}{1+e^{\tilde{W}^{T}\tilde{X}_{i}}})\tilde{X}_{i,j}]
   $$
   直至收敛

2. 对代价函数进行梯度下降方法

   对于某个观察对象（样本），损失函数
   
   > 损失函数一般使用欧几里得距离的平方，但在某些条件下（例如激活函数的选择），会造成代价函数是非凸的，不利于我们寻找全局最小值。至于为什么使用对数似然估计而不是似然估计，还有一个原因是取
   >
   > -log在[0,1]更符合“预测正确->不惩罚->损失函数小 预测越错误->损失函数越大”的特性
   
   $$
   Cost(h_{\beta}(x))=\begin{cases}-\log{(p(x;\beta))}&if\;y=1\\-\log{(1-p(x;\beta))}&if\;y=0\end{cases}
   $$
   
   多个观察对象（样本）总的代价函数
   
   > 直接将单个样本的损失函数求和， 通过除以样本数可以得到平均损失值，避免样本数量对于损失值的影响 
   
   $$
   \begin{align}J(\beta)&=\frac{1}{m}\sum_{i=1}^{n}Cost(h_{\beta}(x))\\&=-\frac{1}{m}\sum_{i=1}^{n}[(y_{i}\log{p(x_{i};\beta)+(1-y_{i})\log{(1-p(x_{i};\beta))}})]\end{align}
   $$
   
   
   
   问题转换为使得代价函数最小时的参数$\beta$的值
   $$
   \begin{align}\frac{\partial}{\partial{\beta_{j}}}J(\beta)&=-\frac{1}{m}\sum_{i=1}^{n}[y_{i}\frac{1}{p(x_{i};\beta)}\frac{\partial\,p(x_{i},\beta)}{\partial{\beta_{j}}}-(1-y_{i})\frac{1}{1-p(x_{i};\beta)}\frac{\partial\,p(x_{i},\beta)}{\partial{\beta_{j}}}]\\&=-\frac{1}{m}\sum_{i=1}^{n}[\frac{y_{i}}{p(x_{i};\beta)}-\frac{1-y_{i}}{1-p(x_{i};\beta)}]\frac{\partial{p(x_{i};\beta)}}{\partial{\beta_{j}}}\\代入并求偏导&=-\frac{1}{m}\sum_{i=1}^{n}[\frac{y_{i}-p(x_{i};\beta)}{p(x_{i};\beta)(1-p(x_{i};\beta))}]p(x_{i};\beta)(1-p(x_{i};\beta))\frac{\partial{\beta x_{i}}}{\partial{\beta_{j}}}\\&=-\frac{1}{m}\sum_{i=1}^{n}[y_{i}-p(x_{i},\beta)]\frac{\partial{\beta x_{i}}}{\partial{\beta_{j}}}\\代入p\ &=\frac{1}{m}\sum_{i=1}^{n}[(\frac{e^{\tilde{W}^{T}\tilde{X}_{i}}}{1+e^{\tilde{W}^{T}\tilde{X}_{i}}}-y_{i})\tilde{X}_{i,j}]\end{align}
   $$
   所以，重复
   $$
   \tilde{W}_{j}:=\tilde{W}_{j}-\alpha \sum_{i=1}^{n}[(\frac{e^{\tilde{W}^{T}\tilde{X}_{i}}}{1+e^{\tilde{W}^{T}\tilde{X}_{i}}}-y_{i})\tilde{X}_{i,j}]
   $$
   直至收敛