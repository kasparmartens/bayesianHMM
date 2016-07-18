#' @useDynLib bayesianHMM
#' @importFrom Rcpp sourceCpp

#' @export
initial_state_update = function(x){
  return(table(x[1]))
}

#' @export
transition_mat_update = function(x){
  x1 = x[-length(x)]
  x2 = x[-1]
  return(table(x1, x2))
}

#' @export
forward_backward_rcpp = function(y, pi, A, B, k, n){
  # forward
  obj = forward_backward_fast(pi, A, B, y, k, n)
  return(list(P = obj$P, x_draw = factor(obj$x_draw, levels = 1:k)))
}

#' @export
forward_backward = function(y, pi, A, B, k, n){
  # forward
  P = vector("list", n)
  temp = matrix(pi, k, k) * matrix(B[, y[1]], k, k, byrow=T)
  P[[1]] = temp / sum(temp)
  for(t in 2:n){
    temp = matrix(colSums(P[[t-1]]), k, k) * A * matrix(B[, y[t]], k, k, byrow=T)
    P[[t]] = temp / sum(temp)
  }
  # backward 1
  x_draw = rep(NA, n)
  x_draw[n] = sample(1:k, 1, prob = colSums(P[[n]]))
  for(t in (n-1):1){
    x_draw[t] = sample(1:k, 1, prob = P[[t+1]][, x_draw[t+1]])
  }
  # backward 2
  # Q = vector("list", n)
  # Q[[n]] = P[[n]]
  # for(t in (n-1):2){
    # Q[[t]] = P[[t]] * matrix(colSums(Q[[t+1]]) / colSums(P[[t]]), k, k, byrow=T)
  # }
  # most likely hidden state
  return(list(P = P, x_draw = factor(x_draw, levels = 1:k)))
}

#' @export
rdirichlet_mat = function(dirichlet_params){
  t(apply(dirichlet_params, 1, function(x)rdirichlet(1, x)))
}

#' @export
compute_marginal_distribution = function(P_list, k, n){
  marginal_distr = matrix(0, k, n)
  for(t in length(P_list):2){
    marginal_distr[, t] = colSums(P_list[[t]])
  }
  marginal_distr[, 1] = rowSums(P_list[[2]])
  return(marginal_distr)
}

#' @export
gibbs_sampling_hmm = function(y, n_hidden_states, alpha0 = 0.1, max_iter = 1000, burnin = 500){
  if(!is.factor(y)) stop("y must be a factor variable!")
  if(burnin >= max_iter) stop("burnin too large!")
  n = length(y)
  n_symbols = length(unique(y))

  # starting values for the parameters pi, A, B
  k = n_hidden_states
  pi = rep(1/k, k)
  A = 0.5 * diag(k) + 0.5*matrix(1/k, k, k)
  B = matrix(1/n_symbols, k, n_symbols)

  trace_x = matrix(0, max_iter-burnin, n)
  trace_A = list()
  trace_B = list()
  for(i in 1:max_iter){
    # update hidden states
    res = forward_backward_rcpp(y, pi, A, B, k=k, n=n)
    # update transition matrices
    pi = as.numeric(rdirichlet(1, 1 + initial_state_update(res$x_draw)))
    A = rdirichlet_mat(alpha0 + transition_mat_update(res$x_draw))
    B = rdirichlet_mat(alpha0 + table(res$x_draw, y))
    # marginal_distr = marginal_distr + 1/(max_iter)*compute_marginal_distribution(res$Q, k, n)
    if(i > burnin){
      trace_x[i-burnin, ] = res$x_draw
      trace_A[[i-burnin]] = A
      trace_B[[i-burnin]] = B
    }
    if(i %% 100 == 0) cat("iter", i, "\n")
  }
  return(list(trace_x = trace_x, trace_A = trace_A, trace_B = trace_B))
}

# likelihood p(y|x, A, B) = p(y|x, B)
loglikelihood = function(y, x, B){
  sum(log(B[cbind(x, y)]))
}
