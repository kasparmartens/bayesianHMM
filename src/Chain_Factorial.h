#ifndef CHAIN_FACTORIAL_H
#define CHAIN_FACTORIAL_H

#include "global.h"
#include "Chain.h"

using namespace Rcpp;

class Chain_Factorial : public Chain {
  double a_sigma, b_sigma, rho;
  NumericVector mu, transition_probs;
  double sigma;
  int K, k_restricted;
  IntegerMatrix X;
  IntegerMatrix mapping, hamming_balls, restricted_space, all_combinations;
  ListOf<NumericMatrix> P_FHMM, Q_FHMM;
  int nrows_gibbs;
  double h;
  bool HB_sampling;
  //NumericVector w, w_unnorm;
  NumericVector lambdas;
  NumericMatrix w, u;
  double alpha0;
  NumericMatrix mu_all;
  int nrow_Y;
  
public:
  Chain_Factorial(int K_, int k, int n, double alpha, double h_, int radius, bool HB_sampling_, int nrows_gibbs_, IntegerMatrix all_combinations_) : Chain(k, n, alpha, true){
    a_sigma = 0.01;
    b_sigma = 0.01;
    rho = 0.01;
    K = K_;
    A = NumericMatrix(k, k);
    //mu_all = NumericVector(k);
    transition_probs = NumericVector(K);
    mapping = decimal_to_binary_mapping(K);
    X = IntegerMatrix(K, n);
    HB_sampling = HB_sampling_;
    nrows_gibbs = nrows_gibbs_;
    all_combinations = all_combinations_;
    h = h_;
    if(HB_sampling){
      // hamming ball sampling
      hamming_balls = construct_all_hamming_balls(radius, mapping);
      k_restricted = hamming_balls.nrow();
    } else{
      // block gibbs sampling
      k_restricted = myPow(2, nrows_gibbs);
      if(nrows_gibbs == K){
        restricted_space = IntegerMatrix(k_restricted, k_restricted);
        for(int i=0; i<k_restricted; i++){
          restricted_space(_, i) = seq_len(k_restricted)-1;
        }
      }
    }
    //w_unnorm = NumericMatrix(K);
    //w = NumericMatrix(K);
    lambdas = NumericVector(K);
    alpha0 = 1.0;
    
    
    List PP(n), QQ(n);
    for(int t=0; t<n; t++){
      PP[t] = NumericMatrix(k_restricted, k_restricted);
      QQ[t] = NumericMatrix(k_restricted, k_restricted);
    }
    P_FHMM = ListOf<NumericMatrix>(PP);
    Q_FHMM = ListOf<NumericMatrix>(QQ);
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
  
  void update_A();
  
  void initialise_pars(NumericVector transition_probs_, IntegerVector x_, int nrow_Y);
  void initialise_pars(NumericVector w, NumericVector transition_probs_, IntegerVector x_, int nrow_Y);
  void initialise_pars(NumericMatrix w, NumericVector transition_probs_, IntegerVector x_, int nrow_Y);
  
  void update_emission_probs(NumericVector y);
  void update_emission_probs(NumericMatrix y);
  
  void update_mu_for_all_states(int nrows){
    for(int which_row=0; which_row < nrows; which_row++){
      for(int j=0; j<k; j++){
        double temp = 0.0;
        for(int i=0; i<K; i++){
          if(mapping(i, j) == 1){
            temp += w(which_row, i);
          }
        }
        mu_all(which_row, j) = temp;
      }
    }
    
  }
  
  void update_mu(NumericVector y);
  void update_mu(NumericMatrix y);
  
  void update_x(){
    if(HB_sampling){
      update_x_HammingBall();
    } else{
      update_x_BlockGibbs();
    }
  }
  
  void update_x_BlockGibbs();
  
  void update_x_HammingBall(){
    // Hamming ball sampling: sample u_t and overwrite x_t
    sample_within_hamming_ball(x, n, hamming_balls);
    
    // forward step
    FHMM_forward_step(pi, A, emission_probs, P_FHMM, loglik_marginal, k_restricted, n, x, hamming_balls);
    
    // now backward sampling
    FHMM_backward_sampling(x, P_FHMM, k_restricted, n, hamming_balls);
    
    // conditional loglikelihood
    loglik_cond = loglikelihood(x, emission_probs, n);
    
    convert_x_to_X();
  }
  
  void copy_values_to_trace(List& trace_x, List& trace_X, List& trace_pi, List& trace_A, List& trace_mu, List& trace_sigma, List& trace_alpha, List& log_posterior, List& log_posterior_cond, List& trace_switching_prob, int index, IntegerVector subsequence){
    IntegerVector xx(x.begin(), x.end());
    trace_x[index] = clone(xx);
    trace_X[index] = clone(X);
    // trace_pi[index] = clone(pi);
    trace_A[index] = clone(transition_probs);
    trace_mu[index] = clone(w);
    // trace_sigma[index] = clone(sigma);
    // trace_alpha[index] = alpha;
    log_posterior[index] = loglik_marginal;
    log_posterior_cond[index] = loglik_cond;
    //IntegerVector subseq_small(subsequence.begin(), subsequence.end()-1);
    //NumericVector switching_prob_small = switching_prob[subseq_small];
    //trace_switching_prob[index] = clone(switching_prob_small);
  }
  
  double pointwise_loglik(int t){
    if(t == 0){
      return mylog(A(x[t], x[t+1])) + mylog(emission_probs(x[t], t));
    } else if(t == n-1){
      return mylog(A(x[t-1], x[t])) + mylog(emission_probs(x[t], t));
    } else{
      return mylog(A(x[t-1], x[t])) + mylog(A(x[t], x[t+1])) + mylog(emission_probs(x[t], t));
    }
  }
};

#endif