#ifndef CHAIN_H
#define CHAIN_H

#include "global.h"

using namespace Rcpp;

class Chain {
  int k, s, n;
  NumericVector pi, switching_prob;
  NumericMatrix A, B, B_tempered, marginal_distr;
  arma::ivec x;
  double loglik_marginal, loglik_cond, alpha, inv_temperature;
  bool estimate_marginals, is_fixed_B, is_tempered;
  IntegerVector possible_values;
  public:
    Chain(int K, int S, int N, double alpha_, bool is_fixed_B_);
    
    arma::ivec& get_x(){
      return x;
    }
    
    NumericMatrix& get_B(){
      return B;
    }
    
    NumericMatrix& get_A(){
      return A;
    }
    
    NumericVector& get_pi(){
      return pi;
    }
    
    double get_inv_temperature(){
      return inv_temperature;
    }
    
    double get_loglik(){
      return loglik_marginal;
    }
    
    void set_B(NumericMatrix B0){
      B = clone(B0);
    }
    
    void set_temperature(double a){
      is_tempered = true;
      inv_temperature = a;
    }
    
    NumericMatrix get_marginals(){
      return marginal_distr;
    }
    
    void initialise_transition_matrices();
    
    void initialise_transition_matrices(NumericMatrix B0);
    
    void update_B_tempered();
    
    void FB_step(IntegerVector& y, ListOf<NumericMatrix>& P, ListOf<NumericMatrix>& Q, bool estimate_marginals);
    
    void update_pars(IntegerVector& y);
    
    void copy_values_to_trace(List& trace_x, List& trace_pi, List& trace_A, List& trace_B, List& log_posterior, List& log_posterior_cond, List& trace_switching_prob, int index);
    
    void scale_marginals(int max_iter, int burnin);
    
    double calculate_loglik_marginal(IntegerVector& y);
};

#endif