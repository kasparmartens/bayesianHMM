#include "Chain_Factorial.h"

void Chain_Factorial::update_mu(NumericMatrix Y){
  double alpha0 = 1.0/K;
  
  //fit_linear_model(X, y, n, K, mu);
  //fit_Bayesian_linear_model(X, y, n, K, mu, sigma);
  NumericVector w_i, u_i, y_i;
  for(int i=0; i<Y.nrow(); i++){
    w_i = w(i, _);
    u_i = u(i, _);
    y_i = Y(i, _);
    lambdas = calculate_mean_for_all_t(X, w_i, h, K, n);
    double logprob_current = calculate_posterior_prob(y_i, lambdas, w_i, alpha0, K, n);
    
    // propose a new alpha
    // double alpha_proposed = random_walk_log_scale(alpha0, 0.3);
    // propose a new w
    NumericVector w_unnorm_proposed = RWMH(u_i, K, 0.5);
    NumericVector w_proposed = w_unnorm_proposed / sum(w_unnorm_proposed);
    NumericVector lambdas_proposed = calculate_mean_for_all_t(X, w_proposed, h, K, n);
    double logprob_proposed = calculate_posterior_prob(y_i, lambdas_proposed, w_proposed, alpha0, K, n);
    //printf("current %f, proposed %f, accept prob: %f\n\n", logprob_current, logprob_proposed, exp(logprob_proposed - logprob_current));
    
    
    // accept or reject
    if(R::runif(0, 1) < exp(logprob_proposed - logprob_current)){
      u(i, _) = clone(w_unnorm_proposed);
      w(i, _) = clone(w_proposed);
    }
  }
  
}


void Chain_Factorial::update_emission_probs(NumericMatrix Y){
  for(int t=0; t<n; t++){
    for(int i=0; i<k; i++){
      //double loglik = R::dnorm4(y[t], mu_all[i], sigma, true);
      double loglik = 0.0;
      for(int ii=0; ii<Y.nrow(); ii++){
        loglik += R::dpois(Y(ii, t), h*mu_all(ii, i)+1e-16, true);
      }
      emission_probs(i, t) = exp(inv_temperature * loglik);
    }
  }
}


void Chain_Factorial::initialise_pars(NumericVector transition_probs_, IntegerVector x_, int nrow_Y_){
  nrow_Y = nrow_Y_;
  // draw pi from the prior
  NumericVector pi_pars(k);
  initialise_const_vec(pi_pars, alpha, k);
  rdirichlet_vec(pi_pars, pi, k);
  
  u = NumericMatrix(nrow_Y, K);
  w = NumericMatrix(nrow_Y, K);
  for(int i=0; i<nrow_Y; i++){
    for(int k=0; k<K; k++){
      // u(i, k) = w_(i, k);
      u(i, k) = R::rgamma(1.0/K, 1.0);
    }
    w(i, _) = u(i, _) / sum(u(i, _));
  }
  
  mu_all = NumericMatrix(nrow_Y, k);
  update_mu_for_all_states(nrow_Y);
  
  transition_probs = clone(transition_probs_);
  FHMM_update_A(transition_probs, A, mapping);
  for(int t=0; t<n; t++){
    x[t] = x_[t];
  }
  convert_x_to_X();
}

void Chain_Factorial::initialise_pars(NumericMatrix w_, NumericVector transition_probs_, IntegerVector x_, int nrow_Y_){
  nrow_Y = nrow_Y_;
  // draw pi from the prior
  NumericVector pi_pars(k);
  initialise_const_vec(pi_pars, alpha, k);
  rdirichlet_vec(pi_pars, pi, k);
  
  u = NumericMatrix(nrow_Y, K);
  w = NumericMatrix(nrow_Y, K);
  for(int i=0; i<nrow_Y; i++){
    for(int k=0; k<K; k++){
      u(i, k) = w_(i, k);
      // u(i, k) = R::rgamma(1.0, 1.0);
    }
    w(i, _) = u(i, _) / sum(u(i, _));
  }
  
  mu_all = NumericMatrix(nrow_Y, k);
  update_mu_for_all_states(nrow_Y);
  
  transition_probs = clone(transition_probs_);
  FHMM_update_A(transition_probs, A, mapping);
  for(int t=0; t<n; t++){
    x[t] = x_[t];
  }
  convert_x_to_X();
}

void Chain_Factorial::update_A(){
  IntegerVector counts = FHMM_count_transitions(X);
  int total = (n-1);
  for(int i=0; i<K; i++){
    transition_probs[i] = R::rbeta(1 + counts[i], 10 + total - counts[i]);
  }
  FHMM_update_A(transition_probs, A, mapping);
}

void Chain_Factorial::update_x_BlockGibbs(){
  if(nrows_gibbs == K){
    // forward step
    FHMM_forward_step(pi, A, emission_probs, P_FHMM, loglik_marginal, k_restricted, n, x, restricted_space);
    
    // now backward sampling
    FHMM_backward_sampling(x, P_FHMM, k_restricted, n, restricted_space);
    
  } else{
    for(int i=0; i<all_combinations.ncol(); i++){
      // Restrict the state space to block Gibbs updates
      IntegerVector which_rows_fixed = all_combinations(_, i);
      //IntegerVector which_rows_fixed = sample_helper(K, K-nrows_gibbs);
      restricted_space = construct_all_restricted_space(k_restricted, which_rows_fixed, mapping);
      // forward step
      FHMM_forward_step(pi, A, emission_probs, P_FHMM, loglik_marginal, k_restricted, n, x, restricted_space);
      
      // now backward sampling
      FHMM_backward_sampling(x, P_FHMM, k_restricted, n, restricted_space);
    }
    
  }
  // conditional loglikelihood
  loglik_cond = loglikelihood(x, emission_probs, n);
  
  convert_x_to_X();
}
