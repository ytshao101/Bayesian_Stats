# Gibbs Sampling (Alternating Conditional Sampling)

Author: Yutong Shao

Date: Feb.21, 2023

So far, we’ve covered several sampling methods:

1. Why we need Monte Carlo sampling?

2. Monte Carlo: inverse CDF, rejection sampling, importance sampling

3. MCMC: Metropolis, Gibbs sampling



## Essence of Gibbs Sampling

1. Goal: approximate a VERY COMPLICATED joint distribution $p(x,y)$. Usually high dimensions

   Remark: for illustration purpose

2. Motivation: 
   - it is difficult to directly sample from $p(x,y)$
   - but it is easy to sample from two conditional distributions, $p(x\mid y, data),\ p(y\mid x, data)$

3. Essence: 

   In its basic version, Gibbs sampling is a special case of the Metropolis–Hastings algorithm in the sense that they both **approximate the stationary distribution** using a long Markov Chain based on **Ergodic Theorem**.

4. What do you need to do Gibbs sampling?

   Conditional distributions

   



### Two-dimension Gibbs Sampling Algorithm







### Why the algorithm works

Recall the detailed balance condition from the previous module:

> Definition 1:
>
> Let $X_0,X_1,... $ be a Markov chain with stationary distribution $\pi$ and transition probability $P$. The chain is said to be **reversible** with respect to $\pi$ or satisfy **detailed balance** with respect to $\pi$ if
> $$
> \pi_i P_{ij} = \pi_j P_{ji}  \quad \forall i,j
> $$
> Where $P_{ij} = P(X_j \mid X_i)$ and vice versa.

Gibbs sampling uses a very clever transformation to construct a detailed balance transition matrix.

Suppose we have a two-dimension target distribution $p(x,y)$. Consider two data points with the same $x$ value, $A(x_1, y_1), B(x_1, y_2)$, we can write the conditional distribution as follows.
$$
p\left(x_{1}, y_{1}\right) p\left(y_{2} \mid x_{1}\right) = p\left(x_{1}\right) p\left(y_{1} \mid x_{1}\right) p\left(y_{2} \mid x_{1}\right) \\
p\left(x_{1}, y_{2}\right) p\left(y_{1} \mid x_{1}\right) = p\left(x_{1}\right) p\left(y_{2} \mid x_{1}\right) p\left(y_{1} \mid x_{1}\right)
$$
Therefore,
$$
p\left(x_{1}, y_{1}\right) p\left(y_{2} \mid x_{1}\right) = p\left(x_{1}, y_{2}\right) p\left(y_{1} \mid x_{1}\right)
$$
i.e.,
$$
p(A) p\left(y_{2} \mid x_{1}\right) = p(B) p\left(y_{1} \mid x_{1}\right)
$$
Which looks quite similar to the detailed balance condition.

Similarly, for any two data points with the same y value, the following equation holds.
$$
p(A) p\left(x_{2} \mid y_{1}\right) = p(C) p\left(x_{1} \mid y_{1}\right)
$$
Combine the above two equations, we get
$$
p(B) p\left(y_{1} \mid x_{1}\right) = p(C) p\left(x_{1} \mid y_{1}\right)
$$
Generalize the conclusion to any two data points $X$ and $Y$ on the plane, detailed balance condition holds, i.e., 
$$
p(X) Q\left(Y \to X\right) = p(Y) Q\left(X \to  Y\right)
$$
The above two-dimensional Markov chain will converge to the stationary distribution $p(x,y)$ due to detailed balance condition.



### N-dimension Gibbs Sampling algorithm

![Screen Shot 2023-02-26 at 5.39.04 PM](/Users/shaoyutong/Library/Application Support/typora-user-images/Screen Shot 2023-02-26 at 5.39.04 PM.png)





### Properties









## Examples



### Example 1: Exponential Example



### Example 2: Truncated exponential



### Example 3: Normal with semi-conjugate prior





### Example 4: Pareto (power law distribution)







## Diagnostic analysis















## References

Caluza, Las Johansen. (2017). Deciphering West Philippine Sea: A Plutchik and VADER Algorithm Sentiment Analysis. *Indian Journal of Science and Technology*. 11. 1-12. 10.17485/ijst/2018/v11i47/130980. 

[Stationary distribution, detailed balance condition and Gibbs sampling](https://blog.csdn.net/itnerd/article/details/108969622)

 
