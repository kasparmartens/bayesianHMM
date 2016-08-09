#include "Chain.h"

Chain::Chain(int K, int S, int N, double alpha_, bool is_fixed_B_){
  k = K;
  s = S;
  n = N;
  is_fixed_B = is_fixed_B_;
  is_tempered = false;
  inv_temperature = 1.0;
  x = arma::ivec(n);
  switching_prob = NumericVector(n-1);
  alpha = alpha_;
  pi = NumericVector(k);
  A = NumericMatrix(k, k);
  B = NumericMatrix(k, s);
  B_tempered = NumericMatrix(k, s);
  loglik_marginal = 0.0;
  loglik_cond = 0.0;
  possible_values = seq_len(k);
  marginal_distr = NumericMatrix(k, n);
}

void Chain::initialise_transition_matrices(){
  initialise_const_vec(pi, 1.0/k, k);
  if(!is_fixed_B) initialise_const_mat(B, 1.0/s, k, s);
  // initialise A
  initialise_const_mat(A, 0.5*1.0/k, k, k);
  for(int i=0; i<k; i++)
    A(i, i) += 0.5;
}

void Chain::initialise_transition_matrices(NumericMatrix B0){
  initialise_const_vec(pi, 1.0/k, k);
  if(is_fixed_B) B = clone(B0);
  // initialise A
  initialise_const_mat(A, 0.5*1.0/k, k, k);
  for(int i=0; i<k; i++)
    A(i, i) += 0.5;
}

void Chain::update_B_tempered(){
  for(int j=0; j<s; j++){
    for(int i=0; i<k; i++){
      B_tempered(i, j) = pow(B(i, j), inv_temperature);
    }
  }
}

void Chain::FB_step(IntegerVector& y, ListOf<NumericMatrix>& P, ListOf<NumericMatrix>& Q, bool estimate_marginals){
  // forward step (for parallel tempering, used the tempered version of B)
  if(is_tempered){
    forward_step(pi, A, B_tempered, y, P, loglik_marginal, k, n);
  }
  else{
    forward_step(pi, A, B, y, P, loglik_marginal, k, n);
  }

  // now backward sampling
  backward_sampling(x, P, possible_values, k, n);
  // and nonstochastic backward step (optional)
  if(estimate_marginals){
    backward_step(P, Q, k, n);
    switching_probabilities(Q, switching_prob, k, n);
    update_marginal_distr(Q, marginal_distr, k, n);
  }
  
  // conditional loglikelihood
  if(is_tempered){
    loglik_cond = loglikelihood(y, x, B_tempered, n);
  }
  else{
    loglik_cond = loglikelihood(y, x, B, n);
  }
}

void Chain::update_pars(IntegerVector& y){
  transition_mat_update0(pi, x, alpha, k);
  transition_mat_update1(A, x, alpha, k, n);
  if(!is_fixed_B){
    if(!is_tempered){
      transition_mat_update2(B, x, y, alpha, k, s, n);
    } 
    else{
      transition_mat_update3(B, x, y, alpha, k, s, n, inv_temperature);
    }
  }
}

void Chain::copy_values_to_trace(List& trace_x, List& trace_pi, List& trace_A, List& trace_B, List& log_posterior, List& log_posterior_cond, List& trace_switching_prob, int index){
  IntegerVector xx(x.begin(), x.end());
  trace_x[index] = clone(xx);
  trace_pi[index] = clone(pi);
  trace_A[index] = clone(A);
  trace_B[index] = clone(B);
  log_posterior[index] = loglik_marginal;
  log_posterior_cond[index] = loglik_cond;
  trace_switching_prob[index] = clone(switching_prob);
}

void Chain::scale_marginals(int max_iter, int burnin){
  arma::mat out(marginal_distr.begin(), k, n, false);
  out /= (float) (max_iter - burnin);
}
