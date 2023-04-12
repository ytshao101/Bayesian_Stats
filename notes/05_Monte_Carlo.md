# Monte Carlo simulation

Author: Yutong Shao

Date: Feb. 5, 2023

## Motivation

The Monte Carlo technique is not a Bayesian approach (actually frequentist) but is important for Bayesian statistics. Here are the reasons:

1. Monte Carlo is commonly used when we want to simulate a **new and complicated** distribution based on a bunch of simple and known distributions.

   e.g. from a uniform distribution to normal distribution

2. When computing posterior distribution, we need to compute *marginal likelihood*, which is an *intractable integral*. We need Monte Carlo.

3. This works well for distributions with relatively <u>simple forms</u>. Like normal, gamma, and beta.

4. Generic and classic question:

   Want to evaluate 

$$
\mathbb{E}_f[h(x)] = \int_X h(x) f(x) dx
$$

If the distribution of $f(x)$ is very complicated, it is unlikely to directly compute the integral. However, we can draw $N$ i.i.d samples from $f(x)$ and use the sample mean to approximate the population expectation.

Also, under certain assumptions, the second-order moment (asymptotic variance) can also be approximated by sample variance using Monte Carlo. (By CLT)

5. In students’ IQ example, it is difficult to directly calculate

$$
P(\mu_s - \mu_c \mid \text{data})\to \frac{1}{N} \sum_i \mathbb{1}(\mu_s^{(i)} - \mu_c^{(i)})
$$

But it is easy to use Monte Carlo: draw N samples from two distributions several times and compute how many times $\mu_s - \mu_c$ so that we can use frequency to approximate the probability.

6. But when distributions take complicated forms, we cannot directly draw numbers from the distribution, because the computer can’t simulate it. There are two ways to address this: **Rejection Sampling** and **Importance Sampling**.

We will cover rejection sampling here.



## Inverse CDF

A simplest approach is using inverse CDF method. The key to this method is calculating the inverse of CDF, and draw X from uniform distribution. 

Formally,

> Settings:
>
> 1. Target distribution we want to sample from: $X\sim f(x)$.
>
> 2. CDF of the target distribution: $F(X)$.
>
> 3. The inverse of CDF: $F^{-1}(X)$.
>
> LEMMA: 
>
> if $U\sim \operatorname{Uniform}(0,1)$, then using inverse CDF, we will finally get a sample of target distribution $f(x)$, i.e.,
> $$
> F^{-1}(X)\sim X
> $$

The idea is quite straightforward. CDF is a one-to-one mapping from $x$ values to the cumulative probability. Therefore, given a strange CDF, it is natural to think about the reversal - how can we map back from CDF to X (sample)? 

Fortunately, CDF is a **one-to-one mapping**, which guarantees it has inverse function. Thus, we can use the inverse function to perform Monte Carlo sampling.

It works well with multivariate distributions, but also has limitations when the form is extremely complicated.

$\text{CDF}: x \in X \to P(X\leq x) = F(x)$,

$\text{Inv CDF}: p \to F^{-1}(p)$



## Rejection Sampling

1. Goal: Generate samples from a ***complicated*** PDF function.

2. Samples are **a series of X values** with different probabilities of being generated.

   - One need to think about what on earth is PDF. 

   - Probability Density Function gives the probability density of each $X=x$ values, that is, how likely can I draw $x$ from the population.

   - Density can be seen as the likelihood **per unit** of the entire distribution. (Like in Physics, $\rho=M/V$ is the average density of an object.)
   - Thus, the integral of PDF is the **cumulative** probability of $X\leq x$.
   - PDF does not have a point probability (the integral of  each point is 0)



3. Algorithm

- Goal: Sample from a <font color = red>complicated pdf $f(x).$</font>

Suppose that 
$$
\textcolor{red}{f(x)} = \frac{\tilde{f}(x)}{\alpha}, \alpha>0
$$

- Assumption: $f$ is difficult to evaluate, but $\tilde{f}$ is easy! 

Why? $\alpha$ may be very difficult to calculate even computationally.

*e.g.* 
$$
p(\theta \mid x) = \frac{p(x\mid \theta)p(\theta)}{\int p(x\mid \theta)p(\theta) dx}
$$
Where $p(\theta \mid x)$ is our target function $f$ (posterior) that is difficult to calculate, and the denominator $\int p(x\mid \theta)p(\theta) dx$, which can be seen as $\alpha$, is even intractable. 

But we can easily calculate the numerator $p(x\mid \theta)p(\theta)$, which is just likelihood times the prior. So, we can treat the numerator as a proposal function.



5. Steps

​	Step 1: Choose a <font color=blue>proposal distribution $q$ </font>,
$$
c \ \textcolor{blue}{q(x)} \geq \tilde{f}(x).
$$
Where $c>0$. $c$ is like *stretching* $q(x)$ so that it can *envelope* the entire $\tilde{f}(x)$. Otherwise, due to the rule of PDF, the area under PDF is always 1, we can never find a proposal function that can envelope the target function $f$.

​	Step 2: Sample $X \sim \textcolor{blue}{q}$, sample $Y \sim \text{Unif}(0, c\; \textcolor{blue}{q(X)})$ (given X)

​	Step 3: If $Y \leq \tilde{f}(X)$,  then $Z=X$; Else, we reject and return to step (2).













