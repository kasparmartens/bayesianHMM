#ifndef ENSEMBLE_H
#define ENSEMBLE_H

#include "Chain.h"

using namespace Rcpp;

class Ensemble{
  int n_chains, k, s, n;
  std::vector<Chain> chains;
  bool do_parallel_tempering;
  int n_accepts, n_total;
  
public:
  Ensemble(int K, int k, int s, int n, double alpha, bool is_fixed_B, bool is_discrete, bool is_gaussian);
  
  void activate_parallel_tempering(NumericVector temperatures);
  
  double get_acceptance_ratio(){
    double ratio = (double) n_accepts / (double) n_total;
    return ratio;
  }
  
  void initialise_pars();
  
  void initialise_pars(NumericMatrix B);
  
  void update_chains(NumericVector& y, ListOf<NumericMatrix>& P, ListOf<NumericMatrix>& Q, bool estimate_marginals);
  
  void scale_marginals(int max_iter, int burnin);
  
  void do_crossover();
  
  void do_crossovers(int n_crossovers);
  
  void swap_everything();
  
  void swap_pars();
  
  void swap_x();
  
  void copy_values_to_trace(IntegerVector& which_chains, List& trace_x, List& trace_pi, List& trace_A, List& trace_B, List& log_posterior, List& log_posterior_cond, List& trace_switching_prob, int index);
  
  ListOf<NumericMatrix> get_copy_of_marginals(IntegerVector& which_chains);
  
};

#endif