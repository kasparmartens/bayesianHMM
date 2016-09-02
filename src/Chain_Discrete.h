#ifndef CHAIN_DISCRETE_H
#define CHAIN_DISCRETE_H

#include "global.h"
#include "Chain.h"

using namespace Rcpp;

class Chain_Discrete : public Chain {
  NumericMatrix B;

  public:
    Chain_Discrete(int k, int s, int n, double alpha, bool is_fixed_B) : Chain(k, s, n, alpha, is_fixed_B, true, false){}
    
    NumericMatrix& get_B(){
      return B;
    }
    
    void set_B(NumericMatrix B0){
      B = clone(B0);
    }
    
    void initialise_pars(){
      // draw pi from the prior
      NumericVector pi_pars(k);
      initialise_const_vec(pi_pars, alpha, k);
      rdirichlet_vec(pi_pars, pi, k);
      // draw A from the prior
      transition_mat_update1(A, x, alpha, k, 0);
      // draw B from the prior
      transition_mat_update2(B, x, IntegerVector(1), alpha, k, s, 0);
    }
    
    void initialise_pars(NumericMatrix B0){
      // draw pi from the prior
      NumericVector pi_pars(k);
      initialise_const_vec(pi_pars, alpha, k);
      rdirichlet_vec(pi_pars, pi, k);
      // draw A from the prior
      transition_mat_update1(A, x, alpha, k, 0);
      // B is fixed
      B = clone(B0);
    }
    
    void update_pars(IntegerVector& y){
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
    
    void FB_step(IntegerVector& y, bool estimate_marginals){
      emission_probs = emission_probs_mat_discrete(y, B, k, n);
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
    
    void copy_values_to_trace(List& trace_x, List& trace_pi, List& trace_A, List& trace_B, List& log_posterior, List& log_posterior_cond, List& trace_switching_prob, int index){
      IntegerVector xx(x.begin(), x.end());
      trace_x[index] = clone(xx);
      trace_pi[index] = clone(pi);
      trace_A[index] = clone(A);
      trace_B[index] = clone(B);
      log_posterior[index] = loglik_marginal;
      log_posterior_cond[index] = loglik_cond;
      trace_switching_prob[index] = clone(switching_prob);
    }
  
};

#endif