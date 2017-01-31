#ifndef CHAIN_GAUSSIAN_H
#define CHAIN_GAUSSIAN_H

#include "global.h"
#include "Chain.h"

using namespace Rcpp;

class Chain_Gaussian : public Chain {
  double a_sigma, b_sigma, rho;
  NumericVector mu, sigma2;
    
  public:
    Chain_Gaussian(int k, int n, double alpha, bool is_fixed_B) : Chain(k, n, alpha, is_fixed_B){
      mu = NumericVector(k);
      sigma2 = NumericVector(k);
      a_sigma = 0.01;
      b_sigma = 0.01;
      rho = 0.1;
    }
    
    NumericVector& get_mu(){
      return mu;
    }
    
    NumericVector& get_sigma2(){
      return sigma2;
    }
    
    void initialise_pars(){
      // draw pi from the prior
      NumericVector pi_pars(k);
      initialise_const_vec(pi_pars, alpha, k);
      rdirichlet_vec(pi_pars, pi, k);
      // draw A from the prior
      transition_mat_update1(A, A_pars, x, A_gamma, alpha, k, 0);
      // draw mu_k and sigma_k from the prior
      for(int i=0; i<k; i++){
        mu[i] = R::rnorm(0.0, 1.0/rho);
        sigma2[i] = 1.0 / R::rgamma(a_sigma, 1.0/b_sigma);
      }
    }
    
    void initialise_pars(NumericVector mu_, NumericVector sigma2_){
      // draw pi from the prior
      NumericVector pi_pars(k);
      initialise_const_vec(pi_pars, alpha, k);
      rdirichlet_vec(pi_pars, pi, k);
      // draw A from the prior
      transition_mat_update1(A, A_pars, x, A_gamma, alpha, k, 0);
      // pars are fixed
      mu = clone(mu_);
      sigma2 = clone(sigma2_);
    }
    
    void initialise_pars(NumericVector mu_, NumericVector sigma2_, IntegerVector x_){
      // draw pi from the prior
      // NumericVector pi_pars(k);
      // initialise_const_vec(pi_pars, alpha, k);
      // rdirichlet_vec(pi_pars, pi, k);
      for(int i=0; i<k; i++){
        pi[i] = 1.0 / k;
      }
      // draw A from the prior
      transition_mat_update1(A, A_pars, x, A_gamma, alpha, k, 0);
      // pars are fixed
      mu = clone(mu_);
      sigma2 = clone(sigma2_);
      for(int t=0; t<n; t++){
        x[t] = x_[t];
      }
      // update pi, A
      transition_mat_update0(pi, x, alpha, k);
      transition_mat_update1(A, A_pars, x, A_gamma, alpha, k, n);
    }
    
    void update_pars(NumericVector& y){
      //transition_mat_update0(pi, x, alpha, k);
      //transition_mat_update1(A, A_pars, x, A_gamma, alpha, k, n);
      update_alpha(alpha, A, A_pars, a0, b0, k);
      transition_mat_update1(A, A_pars, x, A_gamma, alpha, k, n);
      if(!is_fixed_B){
        update_pars_gaussian(y, x, mu, sigma2, rho, inv_temperature, a_sigma, b_sigma, k, n);
      }
    }
    
    void FB_step(NumericVector& y, bool estimate_marginals){

      emission_probs = emission_probs_mat_gaussian(y, mu, sigma2, k, n);
      emission_probs = temper_emission_probs(emission_probs, inv_temperature, k, n);
      
      forward_step(pi, A, emission_probs, P, loglik_marginal, k, n);
      
      // now backward sampling
      backward_sampling(x, P, possible_values, k, n);
      // and nonstochastic backward step (optional)
      if(estimate_marginals){
        backward_step(P, Q, k, n);
        switching_probabilities(Q, switching_prob, k, n);
        update_marginal_distr(Q, marginal_distr, k, n);
      }
      
      // conditional loglikelihood
      loglik_cond = loglikelihood(x, emission_probs, n);
    }
    
    void copy_values_to_trace(List& trace_x, List& trace_pi, List& trace_A, List& trace_mu, List& trace_sigma2, List& trace_alpha, List& log_posterior, List& log_posterior_cond, List& trace_switching_prob, int index, IntegerVector subsequence){
      IntegerVector xx(x.begin(), x.end());
      IntegerVector xxx = xx[subsequence];
      trace_x[index] = clone(xxx);
      trace_pi[index] = clone(pi);
      trace_A[index] = clone(A);
      trace_mu[index] = clone(mu);
      trace_sigma2[index] = clone(sigma2);
      trace_alpha[index] = alpha;
      log_posterior[index] = loglik_marginal;
      log_posterior_cond[index] = loglik_cond;
      IntegerVector subseq_small(subsequence.begin(), subsequence.end()-1);
      NumericVector switching_prob_small = switching_prob[subseq_small];
      trace_switching_prob[index] = clone(switching_prob_small);
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