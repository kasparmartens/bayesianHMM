#include "Ensemble_Discrete.h"

using namespace Rcpp;

Ensemble_Discrete::Ensemble_Discrete(int n_chains_, int k_, int s_, int n_, double alpha, bool is_fixed_B) : 
  chains(std::vector<Chain_Discrete>()) {
  do_parallel_tempering = false;
  n_chains = n_chains_;
  k = k_;
  s = s_;
  n = n_;
  n_accepts = 0;
  n_total = 0;
  for(int i=0; i<n_chains; i++){
    chains.push_back(Chain_Discrete(k_, s_, n_, alpha, is_fixed_B));
  }
}

void Ensemble_Discrete::activate_parallel_tempering(NumericVector temperatures){
  do_parallel_tempering = true;
  for(int i=0; i<n_chains; i++){
    chains[i].set_temperature(temperatures[i]);
  }
}

void Ensemble_Discrete::initialise_pars(){
  for(int i=0; i<n_chains; i++){
    chains[i].initialise_pars();
  }
}

void Ensemble_Discrete::initialise_pars(NumericMatrix B0){
  for(int i=0; i<n_chains; i++){
    chains[i].initialise_pars(B0);
  }
}

void Ensemble_Discrete::update_pars(IntegerVector& y){
  for(int i=0; i<n_chains; i++){
    chains[i].update_pars(y);
  }
}

void Ensemble_Discrete::update_x(IntegerVector& y, bool estimate_marginals){
  for(int i=0; i<n_chains; i++){
    chains[i].FB_step(y, estimate_marginals);
  }
}

void Ensemble_Discrete::scale_marginals(int max_iter, int burnin){
  for(int i=0; i<n_chains; i++){
    chains[i].scale_marginals(max_iter, burnin);
  }
}

void Ensemble_Discrete::do_crossover(){
  IntegerVector selected_chains = sample_helper(n_chains, 2);
  int i = selected_chains[0]-1;
  int j = selected_chains[1]-1;
  // create temporary vectors x and y
  arma::ivec x(chains[i].get_x_memptr(), n, true);
  arma::ivec y(chains[j].get_x_memptr(), n, true);
  // consider all crossovers
  int temp;
  NumericVector probs(n-1);
  for(int t=0; t<n-1; t++){
    // here we have (cumulatively)a crossover up to point t
    temp = x[t];
    x[t] = y[t];
    y[t] = temp;
    // compute the likelihood term
    probs[t] = crossover_likelihood(x, y, t, chains[i].get_A(), chains[j].get_A());
  }
  nonuniform_crossover(chains[i].get_x(), chains[j].get_x(), probs, n);
}


void Ensemble_Discrete::do_crossovers(int n_crossovers){
  for(int i=0; i<n_crossovers; i++){
    do_crossover();
  }
}

void Ensemble_Discrete::swap_everything(){
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

void Ensemble_Discrete::swap_pars(){
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

void Ensemble_Discrete::swap_x(){
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

void Ensemble_Discrete::copy_values_to_trace(IntegerVector& which_chains, List& trace_x, List& trace_pi, List& trace_A, List& trace_B, List& trace_alpha, List& log_posterior, List& log_posterior_cond, List& trace_switching_prob, int index, IntegerVector subsequence){
  int m = which_chains.size();
  int base_index = index * m;
  for(int i=0; i<m; i++){
    int ii = which_chains[i]-1;
    chains[ii].copy_values_to_trace(trace_x, trace_pi, trace_A, trace_B, trace_alpha, log_posterior, log_posterior_cond, trace_switching_prob, base_index + i, subsequence);
  }
}

ListOf<NumericMatrix> Ensemble_Discrete::get_copy_of_marginals(IntegerVector& which_chains){
  int m = which_chains.size();
  List trace(m);
  for(int i=0; i<m; i++){
    int ii = which_chains[i]-1;
    trace[i] = clone(chains[ii].get_marginals());
  }
  ListOf<NumericMatrix> out(trace);
  return out;
}

