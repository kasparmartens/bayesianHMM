#ifndef CHAIN_FACTORIAL_H
#define CHAIN_FACTORIAL_H

#include "global.h"
#include "Chain.h"

using namespace Rcpp;

class Chain_Factorial : public Chain {
  double a_sigma, b_sigma, rho;
  NumericVector mu;
  double sigma;
  int K, k_hamming;
  IntegerMatrix X;
  IntegerMatrix mapping, hamming_balls;
  ListOf<NumericMatrix> P_FHMM, Q_FHMM;
  
public:
  Chain_Factorial(int K_, int k, int n, double alpha, int radius) : Chain(k, n, alpha, true){
    a_sigma = 0.01;
    b_sigma = 0.01;
    rho = 0.01;
    K = K_;
    mapping = decimal_to_binary_mapping(K);
    X = IntegerMatrix(K, n);
    hamming_balls = construct_all_hamming_balls(radius, mapping);
    k_hamming = hamming_balls.nrow();
    
    List PP(n), QQ(n);
    for(int t=0; t<n; t++){
      PP[t] = NumericMatrix(k_hamming, k_hamming);
      QQ[t] = NumericMatrix(k_hamming, k_hamming);
    }
    P_FHMM = ListOf<NumericMatrix>(PP);
    Q_FHMM = ListOf<NumericMatrix>(QQ);
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
      convert_x_to_X(t);
    }
  }
  
  void convert_x_to_X(int t){
    X(_, t) = mapping(_, x[t]);
  }
  
  void convert_X_to_x(){
    for(int t=0; t<n; t++){
      convert_X_to_x(t);
    }
  }
  
  void convert_X_to_x(int t){
    int state = 0;
    for(int i=0; i<K; i++){
      if(X(i, t) == 1){
        state += myPow(2, i);
      }
    }
    x[t] = state;
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
    convert_x_to_X();
  }
  
  
  void update_emission_probs(NumericMatrix Y){
    emission_probs = NumericMatrix(k, n);
    for(int t=0; t<n; t++){
      for(int i=0; i<k; i++){
        double loglik = 0.0;
        for(int ii=0; ii<Y.nrow(); ii++){
          loglik += R::dnorm4(Y(ii, t), mu[i], sigma, true);
        }
        emission_probs(i, t) = exp(inv_temperature * loglik);
      }
    }
  }
  
  void update_x(){
    bool estimate_marginals = false;
    
    // Hamming ball sampling: sample u_t and overwrite x_t
    sample_within_hamming_ball(x, n, hamming_balls);
    
    // forward step
    FHMM_forward_step(pi, A, emission_probs, P_FHMM, loglik_marginal, k_hamming, n, x, hamming_balls);
    
    // now backward sampling
    FHMM_backward_sampling(x, P_FHMM, k_hamming, n, hamming_balls);
    
    // conditional loglikelihood
    loglik_cond = loglikelihood(x, emission_probs, n);
    
    convert_x_to_X();
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