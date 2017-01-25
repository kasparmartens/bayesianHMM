#ifndef ENSEMBLE_FACTORIAL_H
#define ENSEMBLE_FACTORIAL_H

#include "Chain_Factorial.h"

using namespace Rcpp;

class Ensemble_Factorial{
  int n_chains, K, k, n;
  bool do_parallel_tempering;
  std::vector<Chain_Factorial> chains;
  
public:
  Ensemble_Factorial(int n_chains_, int K_, int k_, int n_, double alpha_, int radius_): 
  chains(std::vector<Chain_Factorial>()) {
    do_parallel_tempering = false;
    n_chains = n_chains_;
    K = K_;
    k = k_;
    n = n_;
    for(int i=0; i<n_chains; i++){
      chains.push_back(Chain_Factorial(K_, k_, n_, alpha_, radius_));
    }
  }
  
  void set_temperatures(NumericVector temperatures){
    do_parallel_tempering = true;
    for(int i=0; i<n_chains; i++){
      chains[i].set_temperature(temperatures[i]);
    }
  }
  
  void initialise_pars(NumericVector mu, double sigma, NumericMatrix A, IntegerVector x){
    for(int i=0; i<n_chains; i++){
      chains[i].initialise_pars(mu, sigma, A, x);
    }
  }
  
  void update_emission_probs(NumericMatrix Y){
    for(int i=0; i<n_chains; i++){
      chains[i].update_emission_probs(Y);
    }
  }
  
  void update_pars(NumericVector& y);
  
  void update_x(){
    for(int i=0; i<n_chains; i++){
      chains[i].update_x();
    }
  }
  
  void scale_marginals(int max_iter, int burnin);
  
  void uniform_crossover(int i, int j, IntegerVector which_rows){
    int t0 = sample_int(n);
    // flip a coin
    if(R::runif(0, 1) < 0.5){
      crossover_mat(chains[i].get_X(), chains[j].get_X(), t0, which_rows);
    } else{
      crossover_mat(chains[j].get_X(), chains[i].get_X(), t0, which_rows);
    }
    // update x correpondingly
    chains[i].convert_X_to_x();
    chains[j].convert_X_to_x();
  }
  
  void nonuniform_crossover(NumericVector probs, int i, int j, IntegerVector which_rows){
    int t0 = sample_int(2*n, probs);
    if(t0 < n){
      crossover_mat(chains[i].get_X(), chains[j].get_X(), t0, which_rows);
    } else{
      crossover_mat(chains[j].get_X(), chains[i].get_X(), t0-n, which_rows);
    }
    // update x correpondingly
    chains[i].convert_X_to_x();
    chains[j].convert_X_to_x();
  }
  
  // just for FHMM
  double crossover_likelihood(int i, int j, int t, IntegerVector which_rows, int m){
    NumericMatrix Ai = chains[i].get_A();
    NumericMatrix Aj = chains[j].get_A();
    double beta_i = chains[i].get_inv_temperature();
    double beta_j = chains[j].get_inv_temperature();
    NumericMatrix emissions_i = chains[i].get_emission_probs();
    NumericMatrix emissions_j = chains[j].get_emission_probs();
    
    double denom_x = 1.0;
    if(t<n-1){
      denom_x = Ai(chains[i].get_x()[t], chains[i].get_x()[t+1]) * 
        Aj(chains[j].get_x()[t], chains[j].get_x()[t+1]);
    } 
    
    double log_denom_y = beta_i * log(emissions_i(chains[i].get_x()[t], t)+1.0e-16) + 
      beta_j * log(emissions_j(chains[j].get_x()[t], t)+1.0e-16);
    
    // swap X(which_rows, t) and update x[t]
    crossover_one_column(chains[i].get_X(), chains[j].get_X(), t, which_rows, m);
    chains[i].convert_X_to_x(t);
    chains[j].convert_X_to_x(t);
    // note that now x[t] has changed!
    
    double num_x = 1.0;
    if(t<n-1){
      num_x = Ai(chains[i].get_x()[t], chains[i].get_x()[t+1]) * 
        Aj(chains[j].get_x()[t], chains[j].get_x()[t+1]);
    } 
    
    double log_x = log(num_x/denom_x + 1.0e-16);
    
    double log_num_y = beta_i * log(emissions_i(chains[i].get_x()[t], t)+1.0e-16) + 
      beta_j * log(emissions_j(chains[j].get_x()[t], t)+1.0e-16);
    
    return log_x + log_num_y - log_denom_y;
  }
  
  void do_crossover(){
    // select chains [i] and [j]
    int i = sample_int(n_chains-1);
    int j = i+1;
    // which rows of X will be included in the crossover
    int m = 2;
    IntegerVector which_rows = sample_helper(K, m);
    // uniform crossover
    uniform_crossover(i, j, which_rows);
    
    // now consider all possible crossover points
    NumericVector log_probs(2*n);
    double log_cumprod = 0.0;
    for(int t=0; t<n; t++){
      log_cumprod += crossover_likelihood(i, j, t, which_rows, m);
      log_probs[t] = log_cumprod;
    }
    for(int t=0; t<n; t++){
      log_cumprod += crossover_likelihood(j, i, t, which_rows, m);
      log_probs[t+n] = log_cumprod;
    }
    NumericVector probs = exp(log_probs - max(log_probs));
    // pick one of the crossovers and accept this move
    nonuniform_crossover(probs, i, j, which_rows);
  }
  
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