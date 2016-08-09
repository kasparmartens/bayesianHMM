#' @useDynLib ensembleHMM
#' @importFrom Rcpp sourceCpp

postprocess_chains = function(res, n_chains_out){
  m = length(res$trace_x)
  seq_ind = lapply(1:n_chains_out, function(i)seq(i, m, n_chains_out))
  res$trace_x = lapply(seq_ind, function(ind) do.call("rbind", res$trace_x[ind]))
  res$trace_pi = lapply(seq_ind, function(ind) res$trace_pi[ind])
  res$trace_A = lapply(seq_ind, function(ind) res$trace_A[ind])
  res$trace_B = lapply(seq_ind, function(ind) res$trace_B[ind])
  res$switching_prob = lapply(seq_ind, function(ind) do.call("rbind", res$switching_prob[ind]))
  res$log_posterior = lapply(seq_ind, function(ind) unlist(res$log_posterior[ind]))
  res$log_posterior_cond = lapply(seq_ind, function(ind) unlist(res$log_posterior_cond[ind]))
  return(res)
}

#' @export
gibbs = function(n_chains, y, k, alpha, max_iter, burnin, which_chains = 1:n_chains, thin = 1, is_fixed_B = FALSE, B = matrix(0, k, s)){
  n = length(y)
  s = length(unique(y))
  if(!all(y %in% 1:s)) stop("y must be an integer from 1 to S")
  
  res = ensemble(n_chains, as.integer(y), alpha, k = k, s = s, n = n, max_iter = max_iter, burnin = burnin, thin = thin, estimate_marginals = TRUE, is_fixed_B = is_fixed_B, parallel_tempering = FALSE, crossovers = FALSE, temperatures = rep(1, n_chains), swap_type = 0, swaps_burnin = max_iter, B = B, which_chains = which_chains)
  
  # postprocess the traces
  n_chains_out = length(which_chains)
  return(postprocess_chains(res, n_chains_out))
}

#' @export
crossovers = function(n_chains, y, k, alpha, max_iter, burnin, swaps_burnin, which_chains = 1:n_chains, thin = 1, is_fixed_B = FALSE, B = matrix(0, k, s)){
  n = length(y)
  s = length(unique(y))
  if(!all(y %in% 1:s)) stop("y must be an integer from 1 to S")
  
  res = ensemble(n_chains, as.integer(y), alpha, k = k, s = s, n = n, max_iter = max_iter, burnin = burnin, thin = thin, estimate_marginals = TRUE, is_fixed_B = is_fixed_B, parallel_tempering = FALSE, crossovers = TRUE, temperatures = rep(1, n_chains), swap_type = 0, swaps_burnin = swaps_burnin, B = B, which_chains = which_chains)
  
  # postprocess the traces
  n_chains_out = length(which_chains)
  return(postprocess_chains(res, n_chains_out))
}

#' @export
parallel_tempering = function(n_chains, temperatures, y, k, alpha, max_iter, burnin, swaps_burnin, swap_type = 0, which_chains = 1:n_chains, thin = 1, is_fixed_B = FALSE, B = matrix(0, k, s)){
  n = length(y)
  s = length(unique(y))
  if(!all(y %in% 1:s)) stop("y must be an integer from 1 to S")
  if(length(temperatures) != n_chains) stop("Specify a temperature for each chain!")
  
  res = ensemble(n_chains, as.integer(y), alpha, k = k, s = s, n = n, max_iter = max_iter, burnin = burnin, thin = thin, 
                 estimate_marginals = TRUE, is_fixed_B = is_fixed_B, parallel_tempering = TRUE, crossovers = FALSE, 
                 temperatures = temperatures, swap_type = swap_type, swaps_burnin = swaps_burnin, B = B, which_chains = which_chains)
  
  # postprocess the traces
  n_chains_out = length(which_chains)
  return(postprocess_chains(res, n_chains_out))
}
