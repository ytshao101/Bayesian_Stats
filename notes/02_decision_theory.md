# Intro to Decision Theory

Author: Yutong Shao

Date: Feb.12, 2023

## Decision procedure

A standard model of decision-making. 

### Components

There are three basic components of decision theory.

1. Possible states of nature: $\Theta$; (parameter space)
2. A set of possible actions: $\mathcal{A}$; (estimation, $a\in \mathcal{A}$, which may also be written as $\hat{\theta}$, representing an estimation)
3. A loss function: $\ell(\theta, a)$. (The distance between estimation and true parameter)

Together, $(\Theta, \mathcal{A}, \ell )$ defines a game where the goal is minimizing the loss function.

### Understanding

It is sometimes difficult to understand what is decision theory in a theoretical framework. Let's start with an example from everyday life. Suppose we are going to decide whether to bring an umbrella tomorrow. And nature has three states: rainy, cloudy, and sunny. The loss function is previously defined. Thus, we can write the game as follows.

1. States of nature: $\Theta = \{rainy, cloudy , sunny\}$;
2. Actions: $\mathcal{A} = \{Bring, Not \ bring\}$;
3. Loss function: $\ell(\theta, a)$ (Formula is omitted here since is not so important.)

This is a very simple example but is far from what we usually play in statistics. In statistics, 

1. the states of nature are the true distribution of parameter $\theta$;
2. The action is our estimation of $\theta$;
3. The loss function measures how bad our estimation is.







## Two kinds of risks

### Frequentist risk

Assume the true distribution of **$\theta$ is known**, conditional on $\theta$, **integrate over all data points X** to compute the expectation. It is similar to using the sample mean to estimate the population means based on LLN.

Let $\theta$ be the parameter we like to estimate, $\hat{\theta}$ be our estimation, R.V. $X$ has the conditional distribution of $p(x\mid \theta)$, then the *Frequentist risk* is 
$$
R(\theta, \hat{\theta}) 
= \mathbb{E} (\ell(\theta, \hat{\theta}) \mid \theta) 
= \int \ell(\theta, \hat{\theta}) p(x\mid \theta)dx
$$
The frequentist way looks for an optimal estimator in a prior-free way.

e.g. we want to find the probability of getting a tail when tossing a coin.

Frequentist: toss N times, and compute the frequencies. If all of the N times are all tails, then frequentists think that $P(tail) = 1$.

Bayesian: toss N times, If all of the N times are all tails, bayesian will update the belief to be between 0.3 and 0.6. 







### Bayes’ risk

Contrary to frequentist risk, Bayes’ risk (posterior risk) places a prior distribution on $\theta$, and  integrate all possible values of $\theta$ instead of $X$. 

Assume $X_{1:n}$ is known and fixed, treat $\theta$ as random,
$$
\rho(\theta, \hat{\theta}) 
= \mathbb{E} (\ell(\theta, \hat{\theta}) \mid X=x) 
= \int \ell(\theta, \hat{\theta}) p(\theta \mid x)d\theta
$$
Where $p(\theta\mid x)$ is the posterior distribution of $\theta$.

Given data, averaging over all possible values of $\theta$ to get the conditional mean.

**Essence: The conditional mean of loss function given data points.**





### Bayes rule

>  ***Definition 1.*** A *Bayes rule* is an optimal decision procedure $\tilde{\theta}(X)$  that minimizes the *posterior risk* for all possible values of $x\in X$.

The optimal decision is denoted as $\tilde{\theta}(X)$ or $\tilde{\delta}(X)$.



### The major differences and essence

The essence of Bayes’ risk or Bayes’ approach is placing a prior belief/distribution on $\theta$, treating it as a random variable. 

The key point is computing the posterior distribution of $\theta$, i.e., $p(\theta\mid X)$.

*Frequentist risk is computing the average loss given a decision rule.* 

*But Bayes risk can be used to compute an optimal decision rule. So, the two can be combined.*









## Minimizing Bayes risk

### Under squared loss

> ***Theorem 1***. Under **squared loss**, the Bayes rule is the posterior mean, i.e.
> $$
> \tilde{\theta}(X) = \mathbb{E}[\theta \mid X_{1:n}]
> $$

In other words, under squared loss, the posterior mean minimizes the posterior risk (Bayes risk)



***Proof of Theorem 1.***

It can be proved that a linear transformation of the loss function (multiply by c) does not influence theorem 1.

The posterior risk can be written as
$$
\begin{align*}
\rho(\theta, \delta(x))&=\mathbb{E}(\ell(\theta,{\delta}(x))|x_{1:n}) =
\mathbb{E}(c(\theta-{\delta}(x)))^2|x_{1:n}) \\
&= 
\mathbb{E}(c\theta^2 - 2c\theta \;{\delta}(x))  + c{\delta}^2(x)) |x_{1:n})\\
&=c\mathbb{E}(\theta^2|x_{1:n}) - 2c{\delta}(x))\mathbb{E}(\theta|x_{1:n}) + c{\delta}^2(x)).
\end{align*}
$$
Now minimize the posterior risk.
$$
\begin{align*}
\frac{\partial \rho(\theta,\delta(x))}{\partial {\delta}(x))} 
&= \frac{\partial \{c\mathbb{E}(\theta^2|x_{1:n}) - 2c{\delta}(x))\mathbb{E}(\theta|x_{1:n}) + c{\delta}^2(x))\}}{\partial {\delta{x}}} \\
&= -2 c \mathbb{E}(\theta|x_{1:n}) + 2 c {\theta}
\end{align*}
$$


Now, let 
$$
\frac{\partial \rho(\theta,\delta(x))}{\partial {\delta}(x))} = -2 c\mathbb{E}(\theta|x_{1:n}) + 2 c\delta(x) =: 0, \\
\text{which implies that } \delta(x) = \mathbb{E}(\theta|x_{1:n}).
$$


Because the loss function is convex ($\frac{\partial^2 \rho(\theta,\delta(x))}{ \partial \delta(x)^2 }=2c > 0$ ), $\delta(x) = \mathbb{E}(\theta|x_{1:n})$ is the unique solution.

==***Remark***: remember to check the uniqueness of the optimal action.==



### Under weighted squared loss

The Bayesian rule is no longer the posterior mean under weighted squared loss, but the **weighted** posterior mean.

***Proof.***

The posterior risk can be written as
$$
\begin{align*}
\rho(\theta, \delta(x))&=\mathbb{E}(\ell(\theta,{\delta}(x))|x_{1:n}) =
\mathbb{E}(w(\theta) (g(\theta)-\delta(x))^2 |x_{1:n}) \\
&= 
\mathbb{E}[ w(\theta) ( g^2(\theta) - 2 g(\theta) \delta(x) + \delta^2(x) |x_{1:n}]\\
&=\mathbb{E}( w(\theta) g^2(\theta) |x_{1:n}) - 2 {\delta}(x))\mathbb{E}(w(\theta) g(\theta) |x_{1:n}) + {\delta}^2(x)\mathbb{E}(w(\theta) |x_{1:n}) .
\end{align*}
$$
Now minimize the posterior risk.

$$
\begin{align*}
\frac{\partial \rho(\theta,\delta(x))}{\partial {\delta}(x))} 
&= -2 \mathbb{E}(w(\theta) g(\theta) |x_{1:n}) + 2 {\delta}\mathbb{E}(w(\theta) |x_{1:n})
\end{align*}
$$
Now, let 
$$
\frac{\partial \rho(\theta,\delta(x))}{\partial {\delta}(x))} = 0
$$
Which implies that
$$
\tilde{\delta}(x) = \frac{\mathbb{E}(w(\theta) g(\theta) |x_{1:n})}{\mathbb{E}(w(\theta) |x_{1:n})}
$$


Observe that the loss function is convex ($w(\theta) > 0$, and thus $\frac{\partial^2 \rho(\theta,\delta(x))}{ \partial \delta(x)^2 }=2 \mathbb{E}(w(\theta) |x_{1:n}) > 0)$, $\tilde{\delta}(x)$ is the unique solution.





### Review of Homework

See the above examples.





## Example: resource allocation



### Problem setting

Goal: find the optimal allocation of disease prevention. 

The fraction of infected individuals is $\theta$, we want to find the resource coverage rate $c$ to be as close as $\theta$ as possible

Define a loss function:
$$
\begin{align}
\ell(\theta, c) & = \left\{\begin{array}{ll}
|\theta-c| & \text { if } c \geq \theta \\
10|\theta-c| & \text { if } c<\theta
\end{array}\right.
\end{align}
$$


Define prior distribution on $\theta$. For convenience, we set $\theta \sim \operatorname{Beta}(a,b)$, where $a=0.05, b=1$.

And the disease status of individuals $X_i$ :
$$
X_1, X_2, \cdots, X_n \stackrel{iid}{\sim} \operatorname{Bernoulli}(\theta)
$$
So that the posterior distribution is an updated Beta distribution. 





### Find the argmin of posterior risk

Use matrix to calculate the integral.





### Sensitivity analysis

Try different prior hyper parameters and recompute argmin of $c$.



## Admissibility

### Admissible and inadmissible decision

1. Admissible

   A decision procedure $\delta$ is admissible if there is no $\delta’$ such that
   $$
   R(\theta, \delta)\leq R(\theta, \delta')
   $$
   $\forall \theta$ and $R(\theta, \delta)\leq R(\theta, \delta')$ for at least one $\theta$. 

   In other words, ***a decision procedure is admissible as long as it is not being dominated everywhere.***

2. Inadmissible

   A decision procedure is inadmissible is one that is **dominated everywhere**.

***Remark:***

1) Bayes procedure is admissible under general conditions;
2) An admissible decision rule is NOT necessarily good.

![Screen Shot 2023-02-15 at 11.28.08 AM](/Users/shaoyutong/Library/Application Support/typora-user-images/Screen Shot 2023-02-15 at 11.28.08 AM.png)

Here, the red line (constant decision rule) is also admissible because it is not being dominated everywhere. **But this is a bad decision rule!!!**

