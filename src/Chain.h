#ifndef CHAIN_H
#define CHAIN_H

#include "global.h"

using namespace Rcpp;

class Chain {
  protected:
    int k, s, n;
    NumericVector pi, switching_prob;
    NumericMatrix A, marginal_distr, emission_probs, emission_probs_tempered;
    arma::ivec x;
    double loglik_marginal, loglik_cond, alpha, inv_temperature;
    bool estimate_marginals, is_fixed_B, is_tempered;
    IntegerVector possible_values;
    ListOf<NumericMatrix> P, Q;
    
  public:
    Chain(int K, int S, int N, double alpha_, bool is_fixed_B_);
    
    arma::ivec& get_x(){
      return x;
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
    
    void set_temperature(double a){
      is_tempered = true;
      inv_temperature = a;
    }
    
    NumericMatrix get_marginals(){
      return marginal_distr;
    }
    
    void scale_marginals(int max_iter, int burnin);
    
    double calculate_loglik_marginal();
};

#endif