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
  IntegerVector selected_chains = sample_helper(n_chains-1, 1);
  int i = selected_chains[0]-1;
  int j = i+1;
  // create shortcuts u and v such that
  // (u, v) <- uniform_crossover(...)
  arma::ivec u(chains[i].get_x_memptr(), n, false);
  arma::ivec v(chains[j].get_x_memptr(), n, false);
  if(R::runif(0, 1) < 0.5){
    uniform_crossover(u, v, n);
  } else{
    uniform_crossover(v, u, n);
  }
  // consider all crossovers of u and v
  NumericVector log_probs(2*n);
  // temporary variables
  NumericMatrix emissions_i = chains[i].get_emission_probs() + 1.0e-15;
  NumericMatrix emissions_j = chains[j].get_emission_probs() + 1.0e-15;
  double beta_i = chains[i].get_inv_temperature();
  double beta_j = chains[j].get_inv_temperature();
  double log_cumprod_x = 0.0;
  double log_cumprod_y = 0.0; 
  double tmp0, tmp1, tmp2;
  for(int t=0; t<n; t++){
    // compute the likelihood term
    tmp0 = log(crossover_likelihood(u, v, t+1, n, chains[i].get_A(), chains[j].get_A()));
    log_cumprod_x += tmp0;
    // switching u[t] and v[t]
    tmp1 = beta_i * (log(emissions_i(v[t], t)) - log(emissions_i(u[t], t))); 
    tmp2 = beta_j * (log(emissions_j(u[t], t)) - log(emissions_j(v[t], t))); 
    log_cumprod_y += tmp1 + tmp2;
    log_probs[t] = log_cumprod_x + log_cumprod_y;
    //printf("probs[%d] = %f", t, probs[t]);
  }
  for(int t=0; t<n; t++){
    // compute the likelihood term
    tmp0 = log(crossover_likelihood(v, u, t+1, n, chains[i].get_A(), chains[j].get_A()));
    log_cumprod_x += tmp0;
    // switching u[t] and v[t]
    tmp1 = beta_i * (log(emissions_i(u[t], t)) - log(emissions_i(v[t], t))); 
    tmp2 = beta_j * (log(emissions_j(v[t], t)) - log(emissions_j(u[t], t))); 
    log_cumprod_y += tmp1 + tmp2;
    log_probs[t+n] = log_cumprod_x + log_cumprod_y;
    //printf("probs[%d] = %f", t, probs[t]);
  }
  NumericVector probs = exp(log_probs - max(log_probs));
  // pick one of the crossovers and accept this move
  nonuniform_crossover2(u, v, probs, n);
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
                                                    chains[j].get_inv_temperature(), chains[j+1].get_inv_temperature(), k, n);
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

