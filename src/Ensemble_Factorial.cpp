#include "Ensemble_Factorial.h"

using namespace Rcpp;

void Ensemble_Factorial::do_crossover(){
  {
    // select chains [i] and [j]
    int i = sample_int(n_chains-1);
    int j = i+1;
    // which rows of X will be included in the crossover
    IntegerVector which_rows = sample_helper(K, nrows_crossover);
    // uniform crossover
    uniform_crossover(i, j, which_rows);
    
    // now consider all possible crossover points
    NumericVector log_probs(2*n);
    double log_cumprod = 0.0;
    for(int t=0; t<n; t++){
      log_cumprod += crossover_likelihood(i, j, t, which_rows, nrows_crossover);
      log_probs[t] = log_cumprod;
    }
    for(int t=0; t<n; t++){
      log_cumprod += crossover_likelihood(i, j, t, which_rows, nrows_crossover);
      log_probs[t+n] = log_cumprod;
    }
    NumericVector probs = exp(log_probs - max(log_probs));
    // pick one of the crossovers and accept this move
    nonuniform_crossover(probs, i, j, which_rows);
  }
}

void Ensemble_Factorial::nonuniform_crossover(NumericVector probs, int i, int j, IntegerVector which_rows){
  int t0 = sample_int(2*n, probs);
  if(t0 < n){
    crossover_end = t0;
    crossover_mat(chains[i].get_X(), chains[j].get_X(), t0, which_rows);
  } else{
    crossover_end = t0-n;
    crossover_flipped = 1 - crossover_flipped;
    crossover2_mat(chains[i].get_X(), chains[j].get_X(), t0-n, n, which_rows);
  }
  // update x correpondingly
  chains[i].convert_X_to_x();
  chains[j].convert_X_to_x();
}

double Ensemble_Factorial::crossover_likelihood(int i, int j, int t, IntegerVector which_rows, int m){
  double log_denom = chains[i].pointwise_loglik(t) + chains[j].pointwise_loglik(t);
  
  // crossover
  crossover_one_column(chains[i].get_X(), chains[j].get_X(), t, which_rows, m);
  chains[i].convert_X_to_x(t);
  chains[j].convert_X_to_x(t);
  
  double log_num = chains[i].pointwise_loglik(t) + chains[j].pointwise_loglik(t);
  
  return log_num - log_denom;
}

void Ensemble_Factorial::uniform_crossover(int i, int j, IntegerVector which_rows){
  int t0 = sample_int(n);
  crossover_start = t0;
  // flip a coin
  if(R::runif(0, 1) < 0.5){
    crossover_flipped = 0;
    crossover_mat(chains[i].get_X(), chains[j].get_X(), t0, which_rows);
  } else{
    crossover_flipped = 1;
    crossover2_mat(chains[i].get_X(), chains[j].get_X(), t0, n, which_rows);
  }
  // update x correpondingly
  chains[i].convert_X_to_x();
  chains[j].convert_X_to_x();
}
