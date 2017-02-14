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

#' @export
generate_discrete_obs = function(segment_lengths, classes, obs_transition){
  if(length(segment_lengths) != length(classes)) stop()
  n = sum(segment_lengths)
  # generate x
  x = rep(NA, n)
  cum_lengths = c(0, cumsum(segment_lengths))
  for(i in 1:length(segment_lengths)){
    ind = (cum_lengths[i]+1):(cum_lengths[i+1])
    x[ind] = classes[i]
  }
  # generate y
  y = rep(NA, n)
  k_hidden = nrow(obs_transition)
  k_obs = ncol(obs_transition)
  for(i in 1:n){
    y[i] = sample(1:k_obs, 1, prob = obs_transition[x[i], ])
  }
  return(list(x = factor(x, levels = 1:k_hidden), y = factor(y, levels = 1:k_obs)))
}

#' @export
generate_gaussian_obs = function(segment_lengths, classes, mu, sigma){
  if(length(segment_lengths) != length(classes)) stop()
  n = sum(segment_lengths)
  # generate x
  x = rep(NA, n)
  cum_lengths = c(0, cumsum(segment_lengths))
  for(i in 1:length(segment_lengths)){
    ind = (cum_lengths[i]+1):(cum_lengths[i+1])
    x[ind] = classes[i]
  }
  # generate y
  y = rnorm(length(x), mean = mu[x], sd=sigma[x])
  return(list(x = factor(x, levels = sort(unique(x))), y = y))
}

#' @export
generate_t_obs = function(segment_lengths, classes, mu, sigma, df){
  if(length(segment_lengths) != length(classes)) stop()
  n = sum(segment_lengths)
  # generate x
  x = rep(NA, n)
  cum_lengths = c(0, cumsum(segment_lengths))
  for(i in 1:length(segment_lengths)){
    ind = (cum_lengths[i]+1):(cum_lengths[i+1])
    x[ind] = classes[i]
  }
  # generate y
  y = rt(length(x), df) * sigma[x] + mu[x]
  return(list(x = factor(x, levels = sort(unique(x))), y = y))
}


generate_FHMM_transition_mat = function(K, rho){
  mapping = decimal_to_binary_mapping(K)
  m = 2**K
  A = matrix(0, m, m)
  for(i in 1:m){
    for(j in 1:m){
      x = mapping[, i]
      y = mapping[, j]
      A[i, j] = rho**(sum(x==y)) * (1-rho)**(sum(x!=y))
    }
  }
  A
}

match_col = function(mat, x) which(colSums(mat != x) == 0)

#' @export
convert_X_to_x = function(X, K, n){
  x = rep(0, n)
  for(t in 1:n){
    temp = 0
    for(i in 1:K){
      if(X[i, t] == 1) temp = temp + 2**(i-1)
    }
    x[t] = temp+1
  }
  x
}

#' @export
generate_FHMM = function(n, K, rho, weights, 
                         starting_vals = rbinom(K, 1, 0.5), 
                         sigma = 1){
  A = generate_FHMM_transition_mat(K, rho)
  mapping = decimal_to_binary_mapping(K)
  # generate X
  X = matrix(0, K, n)
  X[, 1] = starting_vals
  # keep a copy in x
  x = rep(0, n)
  for(t in 2:n){
    prev_state = match_col(mapping, X[, t-1])
    new_state = sample(1:(2**K), 1, prob = A[prev_state, ])
    X[, t] = mapping[, new_state]
    x[t-1] = prev_state
  }
  x[n] = new_state
  # generate Y
  dim_obs = ncol(weights)
  Y = matrix(0, dim_obs, n)
  if(nrow(weights) != K) stop("weight matrix must have K rows")
  for(t in 1:n){
    for(i in 1:dim_obs){
      Y[i, t] = sum(X[, t] * weights[, i]) + rnorm(1, 0, sigma)
    }
  }
  # mu values
  mu = as.numeric(weights)
  # mu = apply(mapping, 2, function(x){
  #   apply(weights, 2, function(w){
  #     sum(x * w)
  #   })
  # })
  return(list(Y = Y, X = X, x = factor(x), mu = mu, A = A))
}

#' @export
add_noise = function(X, K, n, prop = 0.5){
  # noisy X for starting vals
  noisy_X = X
  which_t = sample(1:n, floor(prop * n))
  for(t in which_t){
    how_many_rows = sample(1:K, 1)
    which_rows = sample(1:K, how_many_rows)
    for(i in which_rows){
      noisy_X[i, t] = 1 - X[i, t]
    }
  }
  x = convert_X_to_x(noisy_X, K, n)
  return(list(X = noisy_X, x = x))
}
