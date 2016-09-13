# R package: Ensemble MCMC for Hidden Markov Models

Examples how to use the package:

### Gaussian observations

```
library(ensembleHMM)

# Data generation: specify segment lengths and their classes
segment_lengths = c(200, 10, 200, 10, 200, 20, 200, 20, 200, 30, 200, 30, 200, 50, 200, 50, 200, 70, 200, 70, 200, 100, 200, 100)
classes = c(1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4, 1, 2, 1, 3, 1, 4)

mu = c(-4, 2-0.8, 2, 2+0.8)
sigma = c(1, 1, 1, 1)
hmm_obs = generate_gaussian_obs(segment_lengths, classes, mu, sigma)

# standard Gibbs sampling
res = gibbs(type = "continuous", n_chains = 1, y = hmm_obs$y, k = 4, alpha = 0.1, max_iter = 1000, burnin = 100, thin = 1)

# parallel tempering
temperatures = seq(1, 1.5, length=8)**sqrt(2)
res = parallel_tempering(type = "continuous", swap_type = 0, swaps_freq = 5, n_chains = 8, which_chains = 1, temperatures = 1/temperatures, y = hmm_obs$y, k = 4, alpha = 0.1, max_iter = 1000, burnin = 100, thin = 1, swaps_burnin = 500)

# crossovers
res = crossovers(type = "continuous", n_chains = 8, n_crossovers = 2, swaps_freq = 2, which_chains = 1, y = hmm_obs$y, k = 4, alpha = 0.1, max_iter = 1000, burnin = 100, thin = 1, swaps_burnin = 500)

```

### Discrete observations

```
library(ensembleHMM)

# Data generation: specify segment lengths and their classes
segment_lengths = c(200, 10, 200, 10, 200, 20, 200, 20, 200, 30, 200, 30, 200, 50, 200, 50, 200, 70, 200, 70, 200, 100, 200, 100)
classes = c(1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3, 1, 2, 1, 3)

B = rbind(c(0.40, 0.40, 0.05, 0.05, 0.05, 0.05), 
          c(0.05, 0.05, 0.40-0.2, 0.40+0.2, 0.05, 0.05), 
          c(0.05, 0.05, 0.40, 0.40, 0.05, 0.05))
hmm_obs = generate_discrete_obs(segment_lengths, classes, B)

# standard Gibbs sampling
res = gibbs(type = "discrete", n_chains = 1, y = hmm_obs$y, k = 3, alpha = 0.1, max_iter = 1000, burnin = 100, thin = 1)

# parallel tempering
temperatures = seq(1, 1.5, length=8)**sqrt(2)
res = parallel_tempering(type = "discrete", swap_type = 0, swaps_freq = 5, n_chains = 8, which_chains = 1, temperatures = 1/temperatures, y = hmm_obs$y, k = 3, alpha = 0.1, max_iter = 1000, burnin = 100, thin = 1, swaps_burnin = 500)

# crossovers
res = crossovers(type = "discrete", n_chains = 8, n_crossovers = 2, swaps_freq = 2, which_chains = 1, y = hmm_obs$y, k = 3, alpha = 0.1, max_iter = 1000, burnin = 100, thin = 1, swaps_burnin = 500)

```
