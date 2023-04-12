# MCMC — — Metropolis

MCMC has two important algorithms: Metropolis-Hasting and Gibbs sampling (the latter can be regarded as a special case of the former). We will cover Metropolis in this module.

(There are other generalized MCMC, such as MC quasi-MC, Hamiltonian Monte Carlo, Pseudo-marginal M-H algorithm, etc.)

## Motivation

Pure Monte Carlo methods are just repeated sampling from probability space and have several limitations. e.g. 

1. For two-dimension variables, if we cannot get the joint distribution $p(x,y)$, and only the conditional distribution is accessible ($p(x\mid y), p(y\mid x)$), how to sample from joint distribution? Rejection sampling and inverse CDF apparently is useless here.
2. If we cannot find a good proposal distribution $q(x)$ and its corresponding normalizing constant $C$, rejction sampling is impossible.

This is where MCMC comes into play.

## MCMC

1. What is Markov Chain Monte Carlo?

   MCMC comprises a class of algorithms that use **random sampling** in probability space to estimate the **posterior distribution** of the **parameters of interest**.

   By constructing a Markov chain that has the **target distribution as its** **stationary distribution**, one can obtain a sample of the desired distribution by **recording states from the chain**. 

   The more steps that are included, the more closely the distribution of the sample matches the actual desired distribution. 

2. <font color=blue>Essence of all MCMC methods</font>:

   - Treat the target distribution as stationary distribution;

   - Figure out a transition matrix that satisfies the stationary condition;

   - Use common distributions and certain rejection/acceptance rule and the transition matrix to approximate the target distribution.

### Markov Chain sampling

1. Properties of Markov Chain

   Markov property (memorylessness)

   Where we go next only depends on the last state, i.e.,
   $$
   P(X_{n+1}=x_{n+1}\mid X_{n}=x_{n},\dots,X_{1}=x_{1})
   = P(X_{n+1}=x_{n+1}\mid X_{n}=x_{n})
   $$
   
2. Under certain conditions, MC has a stationary distribution regardless of the initial distribution. (Necessary condition)

   > ***Theorem 1:***
   >
   > An irreducible and ergodic (aperiodic) finite Markov Chain has an unique stationary distribution $\pi(x)$.
   >
   > 不可约且非周期的有限状态马尔可夫链，有唯一平稳分布存在。

Aperiodic: not recurrent （不循环）

Irreducible: the probability of transiting from any one state to any other state is greater than 0. （任意状态可达）

$\Longrightarrow$ irreducible + positive recurrent $\to$ ergodic

3. Therefore, once we know the transition matrix and the target distribution, we can use MC to do sampling.
4. **Limitation: what if we don’t know the transition matrix (probabilities)?**



## Metropolis

We want our target distribution to be the stationary distribution. And according to Ergodic Theorem, the final stationary distribution only depends on the transition matrix $\mathbf{P}$, not on the initial distribution.  But the problem now is: 

*How to find a transition matrix $\mathbf{P}$ that guarantees convergence?*

Or, in other words, *given a matrix*, how to check whether it will lead to a stationary distribution?

### Detailed Balance Condition

> Definition 1:
>
> Let $X_0,X_1,... $ be a Markov chain with stationary distribution $\pi$ and transition probability $P$. The chain is said to be **reversible** with respect to $\pi$ or to **satisfy detailed balance** with respect to $\pi$ if
> $$
> \pi_i P_{ij} = \pi_j P_{ji}  \quad \forall i,j
> $$
> Where $P_{ij} = P(X_j \mid X_i)$ and vice versa.

***Remark:*** What do equations (2) represent physically? 

Suppose we start a chain in the stationary distribution, so that $X_0 ∼ π$. Then the quantity $\pi_i P_{ij}$ represents the **“amount” of probability** that flows down edge $i \to j$ in one time step. 

If (2) holds, then the amount of probability flowing from $i$ to $j$, **equals** the amount that flows from $j$ to $i$. Therefore, there is **no net flux** of probability along the edge $i ↔ j$ during one time step, provided the chain is in the stationary distribution. 

Notice that (2) is **stronger than the condition that $\pi$ be a stationary distribution**, i.e. that it solve the system $\pi = \pi P$. (this latter system is called the *“global balance”* equations, as oppose to *detailed* balance).

**Detailed balance $\Longrightarrow$ global balance, but NOT vice versa.**

------

**There is a good example to understand this condition.**

Thinking about traffic flow in New York City and surroundings: let each borough or suburb be represented by a node of a Markov chain, and join two nodes if there is a bridge or tunnel connecting the boroughs, or if there is a road going directly from one to the other. 

Suppose that cars driving around represent little elements of probability. 

The city is in global balance, or the stationary distribution, if the number of cars in Manhattan, and in all other nodes, doesn’t change with time. As long as the number of cars per unit time leaving Manhattan across the Holland tunnel, equals the number per unit time entering Manhattan via the George Washington bridge, the number of cars in Manhattan doesn’t change, and the system can be in global balance.

The city is in detailed balance, only if the number of cars that leaves ***each** bridge or **each** tunnel* per unit time, equals the number that enter that ***same*** bridge or tunnel. For example, the flux of cars entering Manhattan through the Holland tunnel, must equal the flux of cars exiting Manhattan through the Holland tunnel, and similarly for every other bridge or tunnel. Not only does the number of cars in Manhattan remain constant with time, but the fluxes of cars across each single bridge or tunnel separately must be equal in each direction. 

[^1]: Adapted from the handout of *Applied Stochastic Analysis* by professor Miranda Holmes-Cerfon of NYU.

----

Usually, **it is hard to find a matrix that satisfies the detailed balance condition.** So, what can we do? M-H algorithm is a way to solve this problem by introducing an extra term $\alpha_{ij}$.

### Metropolis Algorithm (Included in slides)

1. Define a proposal matrix. (Proposal matrix = stochastic matrix).

Let $$Q = (Q_{ab}: a,b \in \mathcal{X}).$$ $Q_{ab} = Q(a,b)$. 

$\pi(x)$ is our target distribution.
$$
\pi(x) = \tilde{\pi}(x)/z, \ \ \ z>0.
$$
​	***Remark:*** What is known and unknown above? (Think back to rejection sampling)

​	$\pi(x)$ is unkown, and $z$ is hard to calculate (intractable integral). The only thing we know is $\tilde{\pi}(x)$.

**We then focus on $\tilde{\pi}(x)$.**

2.  <font color=blue>**The algorithm**</font>

Assume a **symmetric** proposal matrix $Q.$ So, $Q_{ab} = Q_{ba}.$ (This will be the basic version of metropolis. Asymmetric version is a generalized M-H algorithm)

If in continuous case, assume a proposal distribution $p(x)$.



Initialize a starting point $x_o \in X.$

Goal: run $n$ total iterations of the algorithm to produce $(x_0, x_1, \ldots, x_n)$ samples. 

This is useful as $$\sum_{i=1}^n f(x_i) \approx E_\pi[f(x)].$$

**For $i \in 1,2,\ldots,n$:**

- Sample a new data point $x^{*}$ from $Q(x_i, x^{*})$ if $x$ is discrete, otherwise, $p(x^{*} \mid x_i).$

- Compute acceptance rule:
  $$
  \textcolor{blue}{r=\frac{\tilde{\pi}(x^{*})}{\tilde{\pi}(x_i)}}
  $$
  $\tilde{\pi}(x^{*})$ is the probability of new data point and $\tilde{\pi}(x_i)$ is the probability of old data point. 

- Sample $\textcolor{red}{u}$ from Uniform$(0,1).$

- If $$\textcolor{red}{u} < \frac{\tilde{\pi}(x^{*})}{\tilde{\pi}(x_i)},$$ accept and $x_{i+1} = x^{*}.$ accept the new data point

- Otherwise, reject and $x_{i+1} = x_i.$ keep the current data point

Output: $x_o,x_1,\ldots,x_n$



3. The above algorithm is equivalent to 
   $$
   x_{i+1} = 
   \begin{cases}
   x^*, & \text{with prob }\min(\textcolor{blue}{r},1)\\
   x_i, & \text{o.w.}
   
   \end{cases}
   $$

Intuition behind this:

- Suppose $r>1$,

​		then the new data point has a higher probability of being sampled than the old data point. Thus in this case, we always accept $x^*$.

- Suppose $r<1$,

For every instance of $x_i$, we should only have a fraction of $x^*$. That is, we accept the new data point with a probability of $r$, or conversely, reject the new data point with a probability of $1-r$.



2. **Intuition: Why this works.**

As mentioned before, it is hard to find a matrix that satisfies the detailed balance condition. Usually,
$$
\tilde{\pi}_i P_{ij} \neq \tilde{\pi}_j P_{ji}
$$
M-H algorithm is a way to solve this problem by introducing an extra term $\alpha_{ij}$, such that
$$
\tilde{\pi}_i P_{ij} \alpha_{ij} = \tilde{\pi}_j P_{ji}\alpha_{ji}
$$


If we define $\alpha_{ij} = \tilde{\pi}_j P_{ji}$ and $\alpha_{ji} = \tilde{\pi}_i P_{ij}$, then the detailed balance condition is surely satisfied. (Notice I swapped the position of $i$ and $j$ in the subscripts.)

Given these, $\alpha_{ij}$ is called **acceptance ratio**, meaning that when transit from state $I$ to $j$ following the transition probability matrix $P$, we accept the transition with the probability of $\alpha_{ij}$, and reject the transition with the probability of $1-\alpha_{ij}$.

**Remark:**

- $P$ can be chosen arbitrarily. As long as $n\to \infty$, $\tilde{\pi}(x)$ will converge to target distribution.
- But the drawback is acceptance ratio can be really low so that most transitions will be wasted. Metropolis-Hasting is a improved version.





### Metropolis-Hasting Algorithm (not required)

According to detailed balance condition,
$$
\tilde{\pi}_i P_{ij} \alpha_{ij} = \tilde{\pi}_j P_{ji}\alpha_{ji}
$$
The equation still holds when we multiply the same number to each side. ==Increase the acceptance ratio.== Thus,

1. If $\alpha_{ij}>\alpha_{ji}$, multiply both sides by $\frac{1}{\alpha_{ji}}$, and it follows
   $$
   \tilde{\pi}_i P_{ij} \frac{\alpha_{ij}}{\alpha_{ji}} = \tilde{\pi}_j P_{ji}\cdot 1
   $$
   Let $\alpha’_{ij}=\frac{\alpha_{ij}}{\alpha_{ji}}$, and $\alpha’_{ji}=1$.

2. If $\alpha_{ij}<\alpha_{ji}$, multiply both sides by $\frac{1}{\alpha_{ij}}$.
   $$
   \tilde{\pi}_i P_{ij} \cdot 1  = \tilde{\pi}_j P_{ji}\frac{\alpha_{ji}}{\alpha_{ij}}
   $$
   let $\alpha’_{ij}=1$ and $\alpha’_{ji}=\frac{\alpha_{ij}}{\alpha_{ji}}$.

Therefore, the final acceptance ratio is 
$$
\alpha'_{ij} = \min \left\{\frac{\tilde{\pi}_j P_{ji}}{\tilde{\pi}_i P_{ij}}, 1\right\}
$$
Since the transition matrix is symmetric, $P_{ij}=P_{ji}$, 
$$
\alpha'_{ij} = \min \left\{\frac{\tilde{\pi}_j}{\tilde{\pi}_i}, 1\right\}
$$
Where $\tilde{\pi}_j$ is the probability of the new data point, and $\tilde{\pi}_i$ is the probability of old data point.



***Remark:***

For continuous case, the proposal matrix should be written as transition probability such that 
$$
\alpha'_{ij} = \min \left\{\frac{P_j}{P_i}, 1\right\}
$$
Which is called *transition kernel*.



### Code example

Calculate the integral of Gaussian distribution. ($\mu=0.5, \sigma=0.1$)

```python
import random
import numpy as np
from matplotlib import pyplot as plt

mu = 0.5
sigma = 0.1
skip = 700 # steps
num = 10000 # data points

def Gaussian(x):
  # use a symmetric proposal function
    return 1/(sigma*np.sqrt(2*np.pi)) * np.exp(-(x-mu)**2 /(2*sigma**2))
  
  
def M_H():
    x_0 = 0
    samples = []
    j = 1
    while len(samples) <= num:
        while True:
            x_1 = random.random()
            p_j = Gaussian(x_1)
            p_i = Gaussian(x_0)
            alpha = min(p_j / p_i, 1.0)
            r = random.random()
            if r <= alpha:
                x_0 = x_1
                if j >= skip:
                    samples.append(x_1)
                j += 1
                break
    return samples
norm_samples = M_H()

### calculate integral
def g(x): # constant function
    return 1.0
sum = 0
n=len(norm_samples)
for sample in norm_samples:
    sum = sum + g(sample)
integral = sum / n
print(integral) # 1
```











## References

Miranda Holmes-Cerfon, Applied Stochastic Analysis, [Spring 2019 Handout, Lecture 3.](https://cims.nyu.edu/~holmes/teaching/asa19/handout_Lecture3_2019.pdf)

[From Monte Carlo to Metropolis](https://zhuanlan.zhihu.com/p/146020807)

[MCMC](https://zhuanlan.zhihu.com/p/37121528)
