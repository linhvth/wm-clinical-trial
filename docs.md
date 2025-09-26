- [[#Notations|Notations]]
- [[#Fundamental assumptions|Fundamental assumptions]]

### Notations
- $\lambda$ the rate of patient recruitment (or arrival)
- $n$ is the number of patient that are recruited
- $N$ is the number of recruitment centers (or sites)

### Fundamental assumptions
1. Patient recruitment is modelled using Poisson processes, which means: 
	$$n \sim \mathrm{Pois}(\lambda)$$
	This is claimed as a common accepted practice [@anisimov2007modelling]. Mathematically, assume patients arrive randomly and independently at a constant average rate $\lambda_i$. Therefore, the number of patients, $n$, recruited over a period of time $t$ follows a Poisson distribution:
	$$\mathbb{P} (n) = \frac{(\lambda t)^n e^{-\lambda t}}{n!}$$
2. Different sites (or centers) have different rates. For site $i=1,2, \ldots, n$ the corresponding rate is $\lambda_i$. This rate is assumed to be constant through the time $t$.
3. Further assume that those rates draw from a Gamma distribution, that is $\lambda_i \sim \mathrm{Gamma}(\alpha, \beta)$ [@anisimov2007modelling]. There are 2 questions can be asked:
	1. Why do we want to treat $\lambda_i$ as a random variable?
		In real life, the true recruitment rate $\lambda_i$ is (1) unknown and (2) uncertain. In Bayesian framework, we treat this parameter as a random variable and assign it a probability distribution, know as the ***prior distribution***, to present the uncertainty.
	2. But why Gamma distribution?
		First of all, it is all about mathematical convenience: ***conjugate prior.*** If we believe that $\lambda \sim \mathrm{Gamma}(\alpha_0, \beta_0)$ prior distribution and we observe $n \sim \mathrm{Pois}(\lambda t)$, then the ***posterior distribution*** for $\lambda$ is also a Gamma distribution, that is:
		$$\lambda \mid n, t \sim \mathrm{Gamma}(\alpha_0 + n, \beta_0 + t)$$
		Also, the Gamma distribution is defined only for $\lambda > 0$, which aligns with the reality of a recruitment rate (we cannot recruit a negative number of patients). Explained in [@anisimov2007modelling], this approach is useful as the rates at the beginning of the trial can be very close to 0, and it definitely will change over time. One question can be asked when we derive the posterior distribution, we use $n$ and $t$, and assume that $\lambda$ is constant over $t$, which seldomly holds true in practice. Then, how should we improve the techniques? The Introduction section in [this paper]() sounds helpful.
4. Across publicly available data, the recruitment rates of trials sharing a common set of characteristics are assumed to be independent and identically distributed (i.i.d.) random variables following a single Gamma distribution, $\mathrm{Gamma}(\alpha, \beta)$

### Distribution of $T(n, N)$
Let $n_i(t)$ is the number of patients recruited by site $i$ until time $t$. We know that $n_i(t) \sim \mathrm{Pois}(\lambda_i)$ assume $\lambda_i$ is the rate of recruitment per one time period $t$. By the Poisson's property of convolution, we have:
$$
n(t) = \sum_{i=1}^N n_i(t) \sim \mathrm{Pois}(\Lambda) \quad 
\text{where} \quad \Lambda = \sum_{i=1}^N \lambda_i
$$
Clearly, $\Lambda$ is a random variable. As defined above, $\lambda_i \sim \mathrm{Gamma}(\alpha, \beta)$ and the parameters are fixed for all sites. Then, by the convolution property of the Gamma distribution, we know that:
$$
\Lambda \sim \mathrm{Gamma}(\alpha N, \beta)
$$
$T(n, N)$ is a random variable of the time needed to recruit $n$ patients from $N$ sites. By definition, we know that $T(n, N)$ follows a Gamma distribution. We now need to define the 2 parameters of shape and rate. For the rate, it's readily defined as $\Lambda$ (the overall rate of recruitment). For the shape parameter, as we aim to recruit $n$ patients, then it's set to be $n$. 
$$T(n, N) \sim \mathrm{Gamma}(n, \Lambda)$$
We now have a superposition of Gamma distributions. We can further simplify and transform this into other distribution. Let $f_T(t)$ be the PDF, we have:
$$
\begin{align*}
f_T(t) &= \int_0^\infty f_{T \mid \Lambda} (t) \, f_\Lambda (\lambda) \, d\lambda \\
&= \int_0^\infty \frac{\lambda^n t^{n-1} e^{-\lambda t}}{\Gamma(n)} \,
\frac{\beta^{\alpha N} \lambda^{\alpha N - 1} e^{-\beta \lambda}}{\Gamma(\alpha N)} \,
\, d \lambda \\
&= \frac{t^{n-1} \beta^{\alpha N}}{\Gamma(n) \Gamma(\alpha N)} \int_0^\infty \lambda^{n+\alpha N - 1} e^{-\lambda(\beta + t)} \, d \lambda \\
&= \frac{t^{n-1} \beta^{\alpha N}}{\Gamma(n) \Gamma(\alpha N)} \, 
\frac{(\beta + t)^{-1}}{(\beta + t)^{n + \alpha N - 1}} \,
\int_0^\infty [\lambda(\beta + t)]^{n+\alpha N - 1} e^{-\lambda(\beta + t)} \, d \lambda (\beta + t) \\
&= \frac{t^{n-1} \beta^{\alpha N}}{\Gamma(n) \Gamma(\alpha N)} \, \frac{\Gamma(n + \alpha N)}{(\beta + t)^{n + \alpha N}} \\
&= \frac{t^{n-1} \beta^{\alpha N} (\beta + t)^{- n - \alpha N}}{\mathrm{B}(n, \alpha N)} \\
&= \frac{(t/\beta)^{n-1} (1 + t/\beta)^{-n - \alpha N}}{\beta \, \mathrm{B}(n, \alpha N)}
\end{align*}
$$
Thus, $T(n, N)$ follows a scaled Beta prime distribution with two shape parameters $n$ and $\alpha N$, and a scale parameter $\beta$.

### Compare 2 approaches
#### Approach 1: 
### References

