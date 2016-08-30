#include "Ensemble.h"

using namespace Rcpp;

Ensemble::Ensemble(int n_chains_, int k_, int s_, int n_, double alpha, bool is_fixed_B, bool is_discrete, bool is_gaussian) : 
  chains(std::vector<Chain>()) {
  do_parallel_tempering = false;
  n_chains = n_chains_;
  k = k_;
  s = s_;
  n = n_;
  n_accepts = 0;
  n_total = 0;
  for(int i=0; i<n_chains; i++){
    chains.push_back(Chain(k_, s_, n_, alpha, is_fixed_B, is_discrete, is_gaussian));
  }
}

void Ensemble::activate_parallel_tempering(NumericVector temperatures){
  do_parallel_tempering = true;
  for(int i=0; i<n_chains; i++){
    chains[i].set_temperature(temperatures[i]);
  }
}

void Ensemble::initialise_pars(){
  for(int i=0; i<n_chains; i++){
    chains[i].initialise_pars();
  }
}

void Ensemble::initialise_pars(NumericMatrix B){
  for(int i=0; i<n_chains; i++){
    chains[i].initialise_pars(B);
  }
}

void Ensemble::update_chains(NumericVector& y, ListOf<NumericMatrix>& P, ListOf<NumericMatrix>& Q, bool estimate_marginals){
  for(int i=0; i<n_chains; i++){
    chains[i].FB_step(y, P, Q, estimate_marginals);
    chains[i].update_pars(y);
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

void Ensemble::do_crossovers(int n_crossovers){
  for(int i=0; i<n_crossovers; i++){
    do_crossover();
  }
}

void Ensemble::swap_everything(){
  // pick chains j and j+1, and propose to swap parameters
  int j = as<int>(sample_helper(n_chains-1, 1)) - 1;
  double accept_prob = MH_acceptance_prob_swap_everything(chains[j].get_x(), chains[j].get_emission_probs_tempered(), 
                                                    chains[j+1].get_x(), chains[j+1].get_emission_probs_tempered(), 
                                                    chains[j].get_inv_temperature(), chains[j+1].get_inv_temperature(), n);
  if(R::runif(0,1) < accept_prob){
    std::swap(chains[j].get_B(), chains[j+1].get_B());
    std::swap(chains[j].get_A(), chains[j+1].get_A());
    std::swap(chains[j].get_pi(), chains[j+1].get_pi());
    std::swap(chains[j].get_x(), chains[j+1].get_x());
    n_accepts += 1;
  }
  n_total += 1;
}

void Ensemble::swap_pars(){
  // pick chains j and j+1, and propose to swap parameters
  int j = as<int>(sample_helper(n_chains-1, 1)) - 1;
  double accept_prob = MH_acceptance_prob_swap_pars(chains[j].get_pi(), chains[j].get_A(), chains[j].get_emission_probs(), 
                                                    chains[j+1].get_pi(), chains[j+1].get_A(), chains[j+1].get_emission_probs(), 
                                                    chains[j].get_inv_temperature(), chains[j+1].get_inv_temperature(), k, s, n);
  if(R::runif(0,1) < accept_prob){
    std::swap(chains[j].get_B(), chains[j+1].get_B());
    std::swap(chains[j].get_A(), chains[j+1].get_A());
    std::swap(chains[j].get_pi(), chains[j+1].get_pi());
    n_accepts += 1;
    
  }
  n_total += 1;
}

void Ensemble::swap_x(){
  // pick chains j and j+1, and propose to swap parameters
  int j = as<int>(sample_helper(n_chains-1, 1)) - 1;
  double accept_prob = MH_acceptance_prob_swap_x(chains[j].get_x(), chains[j].get_pi(), chains[j].get_A(), chains[j].get_emission_probs_tempered(), 
                                                 chains[j+1].get_x(), chains[j+1].get_pi(), chains[j+1].get_A(), chains[j+1].get_emission_probs_tempered(), 
                                                 n);
  if(R::runif(0,1) < accept_prob){
    std::swap(chains[j].get_x(), chains[j+1].get_x());
    n_accepts += 1;
  }
  n_total += 1;
}

void Ensemble::copy_values_to_trace(IntegerVector& which_chains, List& trace_x, List& trace_pi, List& trace_A, List& trace_B, List& log_posterior, List& log_posterior_cond, List& trace_switching_prob, int index){
  int m = which_chains.size();
  int base_index = index * m;
  for(int i=0; i<m; i++){
    int ii = which_chains[i]-1;
    chains[ii].copy_values_to_trace(trace_x, trace_pi, trace_A, trace_B, log_posterior, log_posterior_cond, trace_switching_prob, base_index + i);
  }
}

ListOf<NumericMatrix> Ensemble::get_copy_of_marginals(IntegerVector& which_chains){
  int m = which_chains.size();
  List trace(m);
  for(int i=0; i<m; i++){
    int ii = which_chains[i]-1;
    trace[i] = clone(chains[ii].get_marginals());
  }
  ListOf<NumericMatrix> out(trace);
  return out;
}

