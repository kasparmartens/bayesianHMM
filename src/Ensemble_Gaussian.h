#ifndef ENSEMBLE_GAUSSIAN_H
#define ENSEMBLE_GAUSSIAN_H

#include "Chain_Gaussian.h"
#include "global.h"

using namespace Rcpp;

class Ensemble_Gaussian{
  int n_chains, k, n;
  bool do_parallel_tempering;
  int n_accepts, n_total;
  std::vector<Chain_Gaussian> chains;
  int crossover_start, crossover_end, crossover_flipped;
    
  public:
    Ensemble_Gaussian(int K, int k, int n, double alpha, bool is_fixed_B);
    
    void activate_parallel_tempering(NumericVector temperatures);
    
    double get_acceptance_ratio(){
      double ratio = (double) n_accepts / (double) n_total;
      return ratio;
    }
    
    NumericVector get_crossovers(){
      return NumericVector::create(crossover_start, crossover_end, crossover_flipped);
    }
    
    void initialise_pars();
    
    void initialise_pars(NumericVector mu, NumericVector sigma2);
    
    void initialise_pars(NumericVector mu, NumericVector sigma2, IntegerVector x);
    
    void update_pars(NumericVector& y);
    
    void update_x(NumericVector& y, bool estimate_marginals);
    
    void scale_marginals(int max_iter, int burnin);
    
    void do_crossover();
    
    double crossover_likelihood(int i, int j, int t);
    
    void nonuniform_crossover(NumericVector probs, int i, int j);
    
    void uniform_crossover(int i, int j);
    
    void swap_everything();
    
    void swap_pars();
    
    void swap_x();
    
    // void copy_values_to_trace(IntegerVector& which_chains, List& trace_x, List& trace_pi, List& trace_A, List& trace_B, List& log_posterior, List& log_posterior_cond, List& trace_switching_prob, int index);
    void copy_values_to_trace(IntegerVector& which_chains, List& trace_x, List& trace_pi, List& trace_A, List& trace_mu, List& trace_sigma2, List& trace_alpha, List& log_posterior, List& log_posterior_cond, List& trace_switching_prob, int index, IntegerVector subsequence);
    
    ListOf<NumericMatrix> get_copy_of_marginals(IntegerVector& which_chains);
    
};

#endif