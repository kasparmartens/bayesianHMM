#ifndef CHAIN_FACTORIAL_H
#define CHAIN_FACTORIAL_H

#include "global.h"
#include "Chain.h"

using namespace Rcpp;

class Chain_Factorial : public Chain {
  double a_sigma, b_sigma, rho;
  NumericVector mu;
  double sigma;
  int K;
  IntegerMatrix X;
  IntegerMatrix mapping, hamming_distances;
  
public:
  Chain_Factorial(int K, int k, int n, double alpha) : Chain(k, n, alpha, true){
    //mu = NumericMatrix(k);
    a_sigma = 0.01;
    b_sigma = 0.01;
    rho = 0.01;
    mapping = decimal_to_binary_mapping(K);
    hamming_distances = calculate_hamming_dist(mapping);
    X = IntegerMatrix(K, n);
  }
  
  NumericVector& get_mu(){
    return mu;
  }
  
  double& get_sigma(){
    return sigma;
  }
  
  IntegerMatrix& get_X(){
    return X;
  }
  
  void convert_x_to_X(){
    for(int t=0; t<n; t++){
      X(_, t) = mapping(_, x[t]-1);
    }
  }
  
  void initialise_pars(NumericVector mu_, double sigma_, NumericMatrix A_){
    // draw pi from the prior
    NumericVector pi_pars(k);
    //initialise_const_vec(pi_pars, alpha, k);
    //rdirichlet_vec(pi_pars, pi, k);
    for(int i=0; i<k; i++){
      pi[i] = 1.0 / k;
    }
    // pars are fixed
    mu = clone(mu_);
    sigma = sigma_;
    A = clone(A_);
  }
  
  void initialise_pars(NumericVector mu_, double sigma_, NumericMatrix A_, IntegerVector x_){
    // draw pi from the prior
    NumericVector pi_pars(k);
    initialise_const_vec(pi_pars, alpha, k);
    rdirichlet_vec(pi_pars, pi, k);
    // pars are fixed
    mu = clone(mu_);
    sigma = sigma_;
    A = clone(A_);
    for(int t=0; t<n; t++){
      x[t] = x_[t];
    }
  }
  
  
  void update_emission_probs(NumericMatrix Y){
    emission_probs = NumericMatrix(k, n);
    for(int t=0; t<n; t++){
      for(int i=0; i<k; i++){
        double loglik = 0.0;
        for(int ii=0; ii<Y.nrow(); ii++){
          loglik += R::dnorm4(Y(ii, t), mu[i], sigma, true);
        }
        emission_probs(i, t) = exp(loglik);
      }
    }
  }
  
  void update_x(){
    bool estimate_marginals = false;
    // emission_probs = emission_probs_mat_gaussian_FHMM(Y, mu, sigma, k, n);
    
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
  
  void copy_values_to_trace(List& trace_x, List& trace_X, List& trace_pi, List& trace_A, List& trace_mu, List& trace_sigma, List& trace_alpha, List& log_posterior, List& log_posterior_cond, List& trace_switching_prob, int index, IntegerVector subsequence){
    IntegerVector xx(x.begin(), x.end());
    trace_x[index] = clone(xx);
    trace_X[index] = clone(X);
    // trace_pi[index] = clone(pi);
    // trace_A[index] = clone(A);
    // trace_mu[index] = clone(mu);
    // trace_sigma[index] = clone(sigma);
    // trace_alpha[index] = alpha;
    log_posterior[index] = loglik_marginal;
    log_posterior_cond[index] = loglik_cond;
    //IntegerVector subseq_small(subsequence.begin(), subsequence.end()-1);
    //NumericVector switching_prob_small = switching_prob[subseq_small];
    //trace_switching_prob[index] = clone(switching_prob_small);
  }
  
};

#endif