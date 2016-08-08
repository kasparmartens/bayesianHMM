#ifndef ENSEMBLE_H
#define ENSEMBLE_H

#include "Chain.h"

using namespace Rcpp;

class Ensemble{
  int n_chains, n;
  std::vector<Chain> chains;
  bool do_parallel_tempering, do_crossovers;
  
public:
  Ensemble(int K, int k, int s, int n, double alpha, bool is_fixed_B);
  
  void activate_parallel_tempering(NumericVector temperatures){
    do_parallel_tempering = true;
    for(int i=0; i<n_chains; i++){
      chains[i].set_temperature(temperatures[i]);
    }
  }
  
  void activate_crossovers(){
    do_crossovers = true;
  }
  
  void initialise_transition_matrices();
  
  void initialise_transition_matrices(NumericMatrix B);
  
  void update_chains(IntegerVector& y, ListOf<NumericMatrix>& P, ListOf<NumericMatrix>& Q, bool estimate_marginals);
  
  void copy_values_to_trace(List& trace_x, List& trace_pi, List& trace_A, List& trace_B, List& log_posterior, List& trace_switching_prob, int index);
  
  void scale_marginals(int max_iter, int burnin);
  
  ListOf<NumericMatrix> get_copy_of_marginals();
  
  void do_crossover();
  
  void swap_between_chains(IntegerVector& y);
};

#endif