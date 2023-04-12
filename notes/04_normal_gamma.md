Normal Distribution



## Properties 

- symmetry
- additive
- A normal sampling model is appropriate for data that result from the **additive effect** of a large number of factors

e.g. human heights are heterogeneous in terms of a number of factors controlling human growth (genetics, diet, disease..). assume they are approximately additive, then each height measurement $y_i$ is a linear combination of a large number of terms. Then according to CLT, the empirical distribution of $y_i$ will look like a normal distribution.



## Inference on the mean, conditional on the variance

Suppose $y_1, y_2, \cdots, y_n \sim i.i.d. \operatorname{Normal} (\theta, \sigma^2)$, then the joint density if given by
$$
\begin{align*}
p\left(y_{1}, \ldots, y_{n} \mid \theta, \sigma^{2}\right) & = \prod_{i = 1}^{n} p\left(y_{i} \mid \theta, \sigma^{2}\right) 
\\ 
& = \prod_{i = 1}^{n} \frac{1}{\sqrt{2 \pi \sigma^{2}}} e^{-\frac{1}{2}\left(\frac{y_{i}-\theta}{\sigma}\right)^{2}} 
\\ 
& = \left(2 \pi \sigma^{2}\right)^{-n / 2} \exp \left\{-\frac{1}{2} \sum\left(\frac{y_{i}-\theta}{\sigma}\right)^{2}\right\} .
\end{align*}
$$
We can identify some sufficient statistics by expanding the squared term.
$$
\sum_{i=1}^{n}\left(\frac{y_{i}-\theta}{\sigma}\right)^{2}=\frac{1}{\sigma^{2}} \sum y_{i}^{2}-2 \frac{\theta}{\sigma^{2}} \sum y_{i}+n \frac{\theta^{2}}{\sigma^{2}}
$$
Since $\sum y_i^2, \sum_{i=1}^{n}\left(\frac{y_{i}-\theta}{\sigma}\right)^{2}=\frac{1}{\sigma^{2}} \sum y_{i}^{2}-2 \frac{\theta}{\sigma^{2}} \sum y_{i}+n \frac{\theta^{2}}{\sigma^{2}}$ only depend on the data, they make up a two-dimensional sufficient statistic.

Further,  the sample mean and variance are sufficient statistics as well.
$$
\bar{y} = \sum y_i / n \\
s^2 = \sum \frac{(y_i - \bar{y})^2}{n-1}
$$


### Derivation

Suppose 









Normal-uniform

Normal-normal

Normal-gamma



Normal 