#' @useDynLib ensembleHMM
#' @importFrom Rcpp sourceCpp

postprocess_chains <- function(x, ...) UseMethod("postprocess_chains")

postprocess_chains.ensembleHMM = function(res, n_chains_out, burnin, max_iter, thin){
  m = length(res$trace_x)
  seq_ind = lapply(1:n_chains_out, function(i)seq(i, m, n_chains_out))
  res$trace_x = lapply(seq_ind, function(ind) do.call("rbind", res$trace_x[ind]))
  res$trace_pi = lapply(seq_ind, function(ind) res$trace_pi[ind])
  res$trace_A = lapply(seq_ind, function(ind) res$trace_A[ind])
  if(res$type == "discrete"){
    res$trace_B = lapply(seq_ind, function(ind) res$trace_B[ind])
  }
  if(res$type == "continuous"){
    res$trace_mu = lapply(seq_ind, function(ind) do.call("rbind", res$trace_mu[ind]))
    res$trace_sigma2 = lapply(seq_ind, function(ind) do.call("rbind", res$trace_sigma2[ind]))
  }
  res$trace_alpha = lapply(seq_ind, function(ind) unlist(res$trace_alpha[ind]))
  res$switching_prob = lapply(seq_ind, function(ind) do.call("rbind", res$switching_prob[ind]))
  res$log_posterior = lapply(seq_ind, function(ind) unlist(res$log_posterior[ind]))
  res$log_posterior_cond = lapply(seq_ind, function(ind) unlist(res$log_posterior_cond[ind]))
  res$iter = as.integer(seq(burnin+1, max_iter, thin))
  return(res)
}

postprocess_chains.FHMM = function(res, n_chains_out, burnin, max_iter, thin){
  m = length(res$trace_x)
  seq_ind = lapply(1:n_chains_out, function(i)seq(i, m, n_chains_out))
  res$trace_x = lapply(seq_ind, function(ind) do.call("rbind", res$trace_x[ind]))
  res$trace_X = lapply(seq_ind, function(ind){
    res$trace_X[ind]
  })
  res$trace_pi = lapply(seq_ind, function(ind) res$trace_pi[ind])
  res$trace_A = lapply(seq_ind, function(ind) res$trace_A[ind])
  if(res$type == "discrete"){
    res$trace_B = lapply(seq_ind, function(ind) res$trace_B[ind])
  }
  if(res$type == "continuous"){
    res$trace_mu = lapply(seq_ind, function(ind) do.call("rbind", res$trace_mu[ind]))
    res$trace_sigma2 = lapply(seq_ind, function(ind) do.call("rbind", res$trace_sigma2[ind]))
  }
  res$trace_alpha = lapply(seq_ind, function(ind) unlist(res$trace_alpha[ind]))
  res$switching_prob = lapply(seq_ind, function(ind) do.call("rbind", res$switching_prob[ind]))
  res$log_posterior = lapply(seq_ind, function(ind) unlist(res$log_posterior[ind]))
  res$log_posterior_cond = lapply(seq_ind, function(ind) unlist(res$log_posterior_cond[ind]))
  res$iter = as.integer(seq(burnin+1, max_iter, thin))
  return(res)
}

#' @export
gibbs = function(type, n_chains, y, k, alpha, max_iter, burnin, which_chains = 1:n_chains, thin = 1, fixed_pars = FALSE, B = matrix(0, k, s), mu = numeric(0), sigma2 = numeric(0), subsequence = 1:length(y), estimate_marginals = TRUE, x = integer(0)){
  n = length(y)
  s = length(unique(y))
  if(type == "discrete"){
    if(!all(y %in% 1:s)) stop("y must be an integer from 1 to S")
    res = ensemble_discrete(n_chains, as.integer(y)-1, alpha, k = k, s = s, n = n, max_iter = max_iter, burnin = burnin, thin = thin, estimate_marginals = estimate_marginals, fixed_pars = fixed_pars, parallel_tempering = FALSE, crossovers = FALSE, temperatures = rep(1, n_chains), swap_type = 0, swaps_burnin = max_iter, swaps_freq = 1, B = B, which_chains = which_chains, subsequence = subsequence-1L)
  }
  if(type == "continuous"){
    res = ensemble_gaussian(n_chains, y, alpha, k = k, n = n, max_iter = max_iter, burnin = burnin, thin = thin, estimate_marginals = estimate_marginals, fixed_pars = fixed_pars, parallel_tempering = FALSE, crossovers = FALSE, temperatures = rep(1, n_chains), swap_type = 0, swaps_burnin = max_iter, swaps_freq = 1, mu = mu, sigma2 = sigma2, which_chains = which_chains, subsequence = subsequence-1L, x = x)
  }
  res$type = type
  class(res) = "ensembleHMM"
  # postprocess the traces
  n_chains_out = length(which_chains)
  return(postprocess_chains(res, n_chains_out, burnin, max_iter, thin))
}

#' @export
crossovers = function(type, n_chains, y, k, alpha, max_iter, burnin, swaps_burnin, which_chains = 1:n_chains, temperatures = rep(1, n_chains), swaps_freq = 1, thin = 1, fixed_pars = FALSE, B = matrix(0, k, s), mu = numeric(0), sigma2 = numeric(0), subsequence = 1:length(y), estimate_marginals = TRUE, x = integer(0)){
  n = length(y)
  s = length(unique(y))
  if(type == "discrete"){
    if(!all(y %in% 1:s)) stop("y must be an integer from 1 to S")
    res = ensemble_discrete(n_chains, as.integer(y)-1, alpha, k = k, s = s, n = n, max_iter = max_iter, burnin = burnin, thin = thin, estimate_marginals = estimate_marginals, fixed_pars = fixed_pars, parallel_tempering = TRUE, crossovers = TRUE, temperatures = temperatures, swap_type = -1, swaps_burnin = swaps_burnin, swaps_freq = swaps_freq, B = B, which_chains = which_chains, subsequence = subsequence-1L)
  }
  if(type == "continuous"){
    res = ensemble_gaussian(n_chains, y, alpha, k = k, n = n, max_iter = max_iter, burnin = burnin, thin = thin, estimate_marginals = estimate_marginals, fixed_pars = fixed_pars, parallel_tempering = TRUE, crossovers = TRUE, temperatures = temperatures, swap_type = -1, swaps_burnin = swaps_burnin, swaps_freq = swaps_freq, mu = mu, sigma2 = sigma2, which_chains = which_chains, subsequence = subsequence-1L, x = x)
  }
  res$type = type
  class(res) = "ensembleHMM"
  # postprocess the traces
  n_chains_out = length(which_chains)
  return(postprocess_chains(res, n_chains_out, burnin, max_iter, thin))
}

#' @export
parallel_tempering = function(type, n_chains, temperatures, y, k, alpha, max_iter, burnin, swaps_burnin, swaps_freq = 1, swap_type = 0, which_chains = 1:n_chains, thin = 1, fixed_pars = FALSE, B = matrix(0, k, s), mu = numeric(0), sigma2 = numeric(0), subsequence = 1:length(y), estimate_marginals = TRUE, x = integer(0)){
  n = length(y)
  s = length(unique(y))
  if(length(temperatures) < n_chains) stop("Specify a temperature for each chain!")
  if(type == "discrete"){
    if(!all(y %in% 1:s)) stop("y must be an integer from 1 to S")
    res = ensemble_discrete(n_chains, as.integer(y)-1, alpha, k = k, s = s, n = n, max_iter = max_iter, burnin = burnin, thin = thin, 
                             estimate_marginals = estimate_marginals, fixed_pars = fixed_pars, parallel_tempering = TRUE, crossovers = FALSE, 
                             temperatures = temperatures, swap_type = swap_type, swaps_burnin = swaps_burnin, swaps_freq = swaps_freq, B = B, which_chains = which_chains, subsequence = subsequence-1L)
  }
  if(type == "continuous"){
    res = ensemble_gaussian(n_chains, y, alpha, k = k, n = n, max_iter = max_iter, burnin = burnin, thin = thin, 
                            estimate_marginals = estimate_marginals, fixed_pars = fixed_pars, parallel_tempering = TRUE, crossovers = FALSE, 
                            temperatures = temperatures, swap_type = swap_type, swaps_burnin = swaps_burnin, swaps_freq = swaps_freq, mu = mu, sigma2 = sigma2, which_chains = which_chains, subsequence = subsequence-1L, x = x)
  }
  res$type = type
  class(res) = "ensembleHMM"
  # postprocess the traces
  n_chains_out = length(which_chains)
  return(postprocess_chains(res, n_chains_out, burnin, max_iter, thin))
}

#' @export
FHMM = function(n_chains, n, Y, K, mu, sigma, A, radius, max_iter, burnin, x_init, alpha = 0.1, swaps_burnin, which_chains = 1:n_chains, temperatures = rep(1, n_chains), swaps_freq = 1, thin = 1, 
                crossovers = FALSE, nrows_crossover = 1, 
                HB_sampling = TRUE, nrows_gibbs = 1){
  res = ensemble_FHMM(n_chains = n_chains, Y = Y, mu = mu, sigma = sigma, A = A, alpha = alpha, 
                     K = K, k = 2**K, n = n, radius = radius, 
                     max_iter = max_iter, burnin = burnin, thin = thin, 
                     estimate_marginals = FALSE, parallel_tempering = TRUE, crossovers = crossovers, 
                     temperatures = temperatures, swap_type = 0, swaps_burnin = swaps_burnin, swaps_freq = swaps_freq, 
                     which_chains = which_chains, subsequence = as.numeric(0), x = x_init-1, 
                     nrows_crossover = nrows_crossover, HB_sampling = HB_sampling, nrows_gibbs = nrows_gibbs, 
                     all_combs = combn(0:(K-1), ifelse(nrows_gibbs == K, 1, K-nrows_gibbs)))
  res$type = "factorial"
  class(res) = "FHMM"
  n_chains_out = length(which_chains)
  return(postprocess_chains(res, n_chains_out, burnin, max_iter, thin))
}
