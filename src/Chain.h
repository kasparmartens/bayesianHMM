#ifndef CHAIN_H
#define CHAIN_H

#include "global.h"

using namespace Rcpp;

class Chain {
  int k, s, n;
  NumericVector pi, switching_prob;
  NumericMatrix A, marginal_distr, emission_probs, emission_probs_tempered;
  arma::ivec x;
  double loglik_marginal, loglik_cond, alpha, inv_temperature;
  bool estimate_marginals, is_fixed_B, is_tempered;
  IntegerVector possible_values;
  
  // for discrete HMMs only
  bool is_discrete;
  NumericMatrix B;
  // for Gaussian HMMs only
  bool is_gaussian;
  NumericVector mu, sigma;
  double rho;
  
  public:
    Chain(int K, int S, int N, double alpha_, bool is_fixed_B_, bool is_discrete_, bool is_gaussian_);
    
    arma::ivec& get_x(){
      return x;
    }
    
    NumericMatrix& get_B(){
      return B;
    }
    
    NumericVector& get_mu(){
      return mu;
    }
    
    NumericVector& get_sigma(){
      return sigma;
    }
    
    NumericMatrix& get_emission_probs(){
      return emission_probs;
    }
    
    NumericMatrix& get_emission_probs_tempered(){
      return emission_probs_tempered;
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
    
    void initialise_pars();
    
    void initialise_pars(NumericMatrix B0);
    
    //void FB_step(IntegerVector& y, ListOf<NumericMatrix>& P, ListOf<NumericMatrix>& Q, bool estimate_marginals);
    void FB_step(NumericVector& y, ListOf<NumericMatrix>& P, ListOf<NumericMatrix>& Q, bool estimate_marginals);
    
    //void update_pars(IntegerVector& y);
    void update_pars(NumericVector& y);
    
    void copy_values_to_trace(List& trace_x, List& trace_pi, List& trace_A, List& trace_B, List& log_posterior, List& log_posterior_cond, List& trace_switching_prob, int index);
    
    void scale_marginals(int max_iter, int burnin);
    
    double calculate_loglik_marginal();
};

#endif