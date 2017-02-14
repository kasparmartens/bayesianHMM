#ifndef ENSEMBLE_FACTORIAL_H
#define ENSEMBLE_FACTORIAL_H

#include "Chain_Factorial.h"

using namespace Rcpp;

class Ensemble_Factorial{
  int n_chains, K, k, n;
  bool do_parallel_tempering;
  std::vector<Chain_Factorial> chains;
  int crossover_start, crossover_end, crossover_flipped;
  int nrows_crossover;
  
public:
  Ensemble_Factorial(int n_chains_, int K_, int k_, int n_, double alpha_, double h_, int radius_, int nrows_crossover_, bool HB_sampling, int nrows_gibbs, IntegerMatrix all_combs): 
  chains(std::vector<Chain_Factorial>()) {
    do_parallel_tempering = false;
    n_chains = n_chains_;
    K = K_;
    k = k_;
    n = n_;
    nrows_crossover = nrows_crossover_;
    for(int i=0; i<n_chains; i++){
      chains.push_back(Chain_Factorial(K_, k_, n_, alpha_, h_, radius_, HB_sampling, nrows_gibbs, all_combs));
    }
  }
  
  NumericVector get_crossovers(){
    return NumericVector::create(crossover_start, crossover_end, crossover_flipped);
  }
  
  void set_temperatures(NumericVector temperatures){
    do_parallel_tempering = true;
    for(int i=0; i<n_chains; i++){
      chains[i].set_temperature(temperatures[i]);
    }
  }
  
  void initialise_pars(NumericMatrix w, NumericVector transition_probs, IntegerVector x, int nrow_Y){
    for(int i=0; i<n_chains; i++){
      chains[i].initialise_pars(w, transition_probs, x, nrow_Y);
    }
  }
  
  void initialise_pars(NumericVector transition_probs, IntegerVector x, int nrow_Y){
    for(int i=0; i<n_chains; i++){
      chains[i].initialise_pars(transition_probs, x, nrow_Y);
    }
  }
  
  void update_emission_probs(NumericMatrix Y){
    for(int i=0; i<n_chains; i++){
      chains[i].update_emission_probs(Y);
    }
  }
  
  void update_A(){
    for(int i=0; i<n_chains; i++){
      chains[i].update_A();
    }
  }
  
  void update_mu(NumericMatrix y){
    for(int i=0; i<n_chains; i++){
      chains[i].update_mu(y);
    }
  }
  
  void update_x(){
    for(int i=0; i<n_chains; i++){
      chains[i].update_x();
    }
  }
  
  void scale_marginals(int max_iter, int burnin);
  
  void uniform_crossover(int i, int j, IntegerVector which_rows);
  
  void nonuniform_crossover(NumericVector probs, int i, int j, IntegerVector which_rows);
  
  // just for FHMM
  double crossover_likelihood(int i, int j, int t, IntegerVector which_rows, int m);
  
  void do_crossover();
  
  void copy_values_to_trace(IntegerVector& which_chains, List& trace_x, List& trace_X, List& trace_pi, List& trace_A, List& trace_mu, List& trace_sigma2, List& trace_alpha, List& log_posterior, List& log_posterior_cond, List& trace_switching_prob, int index, IntegerVector subsequence){
    int m = which_chains.size();
    int base_index = index * m;
    for(int i=0; i<m; i++){
      int ii = which_chains[i]-1;
      chains[ii].copy_values_to_trace(trace_x, trace_X, trace_pi, trace_A, trace_mu, trace_sigma2, trace_alpha, log_posterior, log_posterior_cond, trace_switching_prob, base_index + i, subsequence);
    }
  }
  
  ListOf<NumericMatrix> get_copy_of_marginals(IntegerVector& which_chains);
  
};

#endif