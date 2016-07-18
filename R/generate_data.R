#' @export
generate_discrete_HMM = function(n, starting_probs, hidden_transition, obs_transition){
  x = rep(NA, n)
  k_hidden = length(starting_probs)
  k_obs = ncol(obs_transition)
  if((k_hidden != nrow(hidden_transition)) | (k_hidden != ncol(hidden_transition)) | (k_hidden != nrow(obs_transition))) stop("Dimensions do not match")
  x[1] = sample(1:k_hidden, 1)
  for(i in 2:n){
    x[i] = sample(1:k_hidden, 1, prob = hidden_transition[x[i-1], ])
  }
  y = rep(NA, n)
  for(i in 1:n){
    y[i] = sample(1:k_obs, 1, prob = obs_transition[x[i], ])
  }
  return(list(x = factor(x, levels = 1:k_hidden), y = factor(y, levels = 1:k_obs)))
}

#' @export
generate_nonmarkov_seq = function(n, obs_transition, n_breakpoints = 5){
  x = rep(NA, n)
  k_hidden = nrow(obs_transition)
  breakpoints = c(sort(sample(1:(n-1), n_breakpoints)), n)
  for(i in 1:length(breakpoints)){
    if(i == 1) start = 1 else start = breakpoints[i-1]+1
    x[start:breakpoints[i]] = sample(1:k_hidden, 1)
  }
  y = rep(NA, n)
  k_obs = ncol(obs_transition)
  for(i in 1:n){
    y[i] = sample(1:k_obs, 1, prob = obs_transition[x[i], ])
  }
  return(list(x = factor(x, levels = 1:k_hidden), y = factor(y, levels = 1:k_obs)))
}
