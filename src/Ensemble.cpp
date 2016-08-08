#include "Ensemble.h"

using namespace Rcpp;

Ensemble::Ensemble(int n_chains_, int k, int s, int n_, double alpha, bool is_fixed_B) : 
  chains(std::vector<Chain>()) {
  do_parallel_tempering = false;
  do_crossovers = false;
  n_chains = n_chains_;
  n = n_;
  for(int i=0; i<n_chains; i++){
    chains.push_back(Chain(k, s, n_, alpha, is_fixed_B));
  }
}

void Ensemble::initialise_transition_matrices(){
  for(int i=0; i<n_chains; i++){
    chains[i].initialise_transition_matrices();
  }
}

void Ensemble::initialise_transition_matrices(NumericMatrix B){
  for(int i=0; i<n_chains; i++){
    chains[i].initialise_transition_matrices(B);
  }
}

void Ensemble::update_chains(IntegerVector& y, ListOf<NumericMatrix>& P, ListOf<NumericMatrix>& Q, bool estimate_marginals){
  for(int i=0; i<n_chains; i++){
    chains[i].FB_step(y, P, Q, estimate_marginals);
    chains[i].update_pars(y);
  }
}

void Ensemble::copy_values_to_trace(List& trace_x, List& trace_pi, List& trace_A, List& trace_B, List& log_posterior, List& trace_switching_prob, int index){
  int base_index = index * n_chains;
  for(int i=0; i<n_chains; i++){
    chains[i].copy_values_to_trace(trace_x, trace_pi, trace_A, trace_B, log_posterior, trace_switching_prob, base_index + i);
  }
}

void Ensemble::scale_marginals(int max_iter, int burnin){
  for(int i=0; i<n_chains; i++){
    chains[i].scale_marginals(max_iter, burnin);
  }
}

void Ensemble::do_crossover(){
  IntegerVector selected_chains = sample_helper(n_chains, 2);
  int i = selected_chains[0]-1;
  int j = selected_chains[1]-1;
  double_crossover(chains[i].get_x(), chains[j].get_x(), n);
}

void Ensemble::swap_between_chains(IntegerVector& y){
  int j = as<int>(sample_helper(n_chains-1, 1)) - 1;
  double loglik1 = loglikelihood(y, chains[j].get_x(), chains[j].get_B(), n);
  double loglik2 = loglikelihood(y, chains[j+1].get_x(), chains[j+1].get_B(), n);
  double ratio = exp(-(chains[j].get_inv_temperature() - chains[j+1].get_inv_temperature())*(loglik1 - loglik2));
  if(R::runif(0,1) < ratio){
    std::swap(chains[j].get_B(), chains[j+1].get_B());
    std::swap(chains[j].get_A(), chains[j+1].get_A());
    std::swap(chains[j].get_pi(), chains[j+1].get_pi());
    std::swap(chains[j].get_x(), chains[j+1].get_x());
  }
}

ListOf<NumericMatrix> Ensemble::get_copy_of_marginals(){
  List trace(n_chains);
  for(int i=0; i<n_chains; i++){
    trace[i] = clone(chains[i].get_marginals());
  }
  ListOf<NumericMatrix> out(trace);
  return out;
}

