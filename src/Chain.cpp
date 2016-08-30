#include "Chain.h"

Chain::Chain(int k_, int s_, int n_, double alpha_, bool is_fixed_B_, bool is_discrete_, bool is_gaussian_){
  k = k_;
  s = s_;
  n = n_;
  is_fixed_B = is_fixed_B_;
  is_tempered = false;
  inv_temperature = 1.0;
  x = arma::ivec(n);
  switching_prob = NumericVector(n-1);
  alpha = alpha_;
  pi = NumericVector(k);
  A = NumericMatrix(k, k);
  emission_probs = NumericMatrix(k, n);
  loglik_marginal = 0.0;
  loglik_cond = 0.0;
  possible_values = seq_len(k);
  marginal_distr = NumericMatrix(k, n);
  // discrete
  is_discrete = is_discrete_;
  B = NumericMatrix(k, k);
  // gaussian
  is_gaussian = is_gaussian_;
  mu = NumericVector(k);
  sigma = NumericVector(k);
  rho = 0.01;
}

void Chain::initialise_pars(){
  // draw pi from the prior
  NumericVector pi_pars(k);
  initialise_const_vec(pi_pars, alpha, k);
  rdirichlet_vec(pi_pars, pi, k);
  // draw A from the prior
  transition_mat_update1(A, x, alpha, k, 0);
  if(is_discrete){
    if(!is_fixed_B)
      // draw B from the prior
      transition_mat_update2(B, x, IntegerVector(1), alpha, k, s, 0);
  }
  if(is_gaussian){
    initialise_const_vec(sigma, 1.0, k);
    for(int i=0; i<k; i++)
      mu[i] = R::rnorm(0.0, 1.0/rho);
  }
}

void Chain::initialise_pars(NumericMatrix B0){
  // draw pi from the prior
  NumericVector pi_pars(k);
  initialise_const_vec(pi_pars, alpha, k);
  rdirichlet_vec(pi_pars, pi, k);
  // draw A from the prior
  transition_mat_update1(A, x, alpha, k, 0);
  // B is fixed
  if(is_discrete){
    if(is_fixed_B) B = clone(B0);
  }
}



void Chain::FB_step(NumericVector& y, ListOf<NumericMatrix>& P, ListOf<NumericMatrix>& Q, bool estimate_marginals){
  // emission_probs = emission_probs_mat_discrete(y, B, k, n);
  emission_probs = emission_probs_mat_gaussian(y, mu, sigma, k, n);
  // forward step
  if(is_tempered){
    emission_probs_tempered = temper_emission_probs(emission_probs, inv_temperature, k, n);
    forward_step(pi, A, emission_probs_tempered, P, loglik_marginal, k, n);
  }
  else{
    forward_step(pi, A, emission_probs, P, loglik_marginal, k, n);
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
    loglik_cond = loglikelihood(x, emission_probs_tempered, n);
  }
  else{
    loglik_cond = loglikelihood(x, emission_probs, n);
  }
}

void Chain::update_pars(NumericVector& y){
  transition_mat_update0(pi, x, alpha, k);
  transition_mat_update1(A, x, alpha, k, n);
  if(!is_fixed_B){
    if(!is_tempered){
      update_pars_gaussian(y, x, mu, sigma, rho, 1.0, k, n);
    } 
    else{
      update_pars_gaussian(y, x, mu, sigma, rho, inv_temperature, k, n);
    }
  }
}

// void Chain::update_pars(IntegerVector& y){
//   transition_mat_update0(pi, x, alpha, k);
//   transition_mat_update1(A, x, alpha, k, n);
//   if(!is_fixed_B){
//     if(!is_tempered){
//       transition_mat_update2(B, x, y, alpha, k, s, n);
//     } 
//     else{
//       transition_mat_update3(B, x, y, alpha, k, s, n, inv_temperature);
//     }
//   }
// }

double Chain::calculate_loglik_marginal(){
  return marginal_loglikelihood(pi, A, emission_probs, 1.0, k, s, n);
}

void Chain::copy_values_to_trace(List& trace_x, List& trace_pi, List& trace_A, List& trace_mu, List& log_posterior, List& log_posterior_cond, List& trace_switching_prob, int index){
  IntegerVector xx(x.begin(), x.end());
  trace_x[index] = clone(xx);
  trace_pi[index] = clone(pi);
  trace_A[index] = clone(A);
  trace_mu[index] = clone(mu);
  log_posterior[index] = loglik_marginal;
  log_posterior_cond[index] = loglik_cond;
  trace_switching_prob[index] = clone(switching_prob);
}
// void Chain::copy_values_to_trace(List& trace_x, List& trace_pi, List& trace_A, List& trace_B, List& log_posterior, List& log_posterior_cond, List& trace_switching_prob, int index){
//   IntegerVector xx(x.begin(), x.end());
//   trace_x[index] = clone(xx);
//   trace_pi[index] = clone(pi);
//   trace_A[index] = clone(A);
//   trace_B[index] = clone(B);
//   log_posterior[index] = loglik_marginal;
//   log_posterior_cond[index] = loglik_cond;
//   trace_switching_prob[index] = clone(switching_prob);
// }

void Chain::scale_marginals(int max_iter, int burnin){
  arma::mat out(marginal_distr.begin(), k, n, false);
  out /= (float) (max_iter - burnin);
}
