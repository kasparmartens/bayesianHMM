// [[Rcpp::depends(RcppArmadillo)]]

#include "Ensemble.h"
#include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;
using namespace std;

void compute_P(NumericMatrix PP, double& loglik, NumericVector pi, NumericMatrix A, NumericVector b, int k){
  double temp;
  for(int s=0; s<k; s++){
    for(int r=0; r<k; r++){
      temp = pi[r] * A(r, s) * b[s];
      PP(r, s) = temp;
    }
  }
  double sum = normalise_mat(PP, k, k);
  loglik += log(sum);
}

void compute_P0(NumericMatrix PP, double& loglik, NumericVector pi, NumericVector b, int k){
  for(int s=0; s<k; s++){
    for(int r=0; r<k; r++){
      PP(r, s) = pi[r] * b[s];
    }
  }
  double sum = normalise_mat(PP, k, k);
  loglik += log(sum);
}

void compute_Q(NumericMatrix QQ, NumericMatrix PP, NumericVector pi_backward, NumericVector pi_forward, int k){
  for(int s=0; s<k; s++){
    if(pi_forward[s]>0){
      for(int r=0; r<k; r++){
        QQ(r, s) = PP(r, s) * pi_backward[s] / pi_forward[s];
      }
    }
  }
}


void update_marginal_distr(ListOf<NumericMatrix> Q, NumericMatrix res, int k, int n){
  arma::mat out(res.begin(), k, n, false);
  for(int t=1; t<=n-1; t++){
    // calculate rowsums of Q[t]
    arma::mat B(Q[t].begin(), k, k, false);
    arma::colvec rowsums = sum(B, 1);
    // assign rowsums to res(_, t-1)
    out.col(t-1) += rowsums;
  }
  // calculate colsums of Q[n-1]
  arma::mat B(Q[n-1].begin(), k, k, false);
  arma::rowvec colsums = sum(B, 0);
  arma::colvec temp = arma::vec(colsums.begin(), k, false, false);
  out.col(n-1) += temp;
}


void forward_step(NumericVector pi, NumericMatrix A, NumericMatrix B, IntegerVector y, ListOf<NumericMatrix>& P, double& loglik, int k, int n){
  NumericVector b, colsums(k);
  b = B(_, y[0]-1);
  compute_P0(P[0], loglik, pi, b, k);
  loglik = 0.0;
  for(int t=1; t<n; t++){
    colsums = calculate_colsums(P[t-1], k, k);
    b = B(_, y[t]-1);
    compute_P(P[t], loglik, colsums, A, b, k);
  }
}

void backward_sampling(arma::ivec& x, ListOf<NumericMatrix>& P, IntegerVector possible_values, int k, int n){
  NumericVector prob(k);
  NumericMatrix PP;
  prob = calculate_colsums(P[n-1], k, k);
  x[n-1] = as<int>(RcppArmadillo::sample(possible_values, 1, false, prob));
  for(int t=n-1; t>0; t--){
    prob = P[t](_, x[t]-1);
    x[t-1] = as<int>(RcppArmadillo::sample(possible_values, 1, false, prob));
  }
}

void backward_step(ListOf<NumericMatrix>& P, ListOf<NumericMatrix>& Q, int k, int n){
  NumericVector q_forward(k), q_backward(k);
  Q[n-1] = P[n-1];
  for(int t=n-2; t>=0; t--){
    q_forward = calculate_colsums(P[t], k, k);
    q_backward = calculate_rowsums(Q[t+1], k, k);
    compute_Q(Q[t], P[t], q_backward, q_forward, k);
  }
}

void switching_probabilities(ListOf<NumericMatrix>& Q, NumericVector res, int k, int n){
  for(int t=n-1; t>0; t--){
    res[t-1] = calculate_nondiagonal_sum(Q[t], k);
  }
}

void rdirichlet_vec(NumericVector a, NumericVector res, int k){
  NumericVector temp(k);
  double sum = 0;
  for(int i=0; i<k; i++){
    temp[i] = R::rgamma(a[i], 1);
    sum += temp[i];
  }
  for(int i=0; i<k; i++){
    res[i] = temp[i] / sum;
  }
}

void rdirichlet_mat(NumericMatrix A, NumericMatrix res, int k, int s){
  NumericVector temp(s);
  for(int i=0; i<k; i++){
    double sum = 0;
    for(int j=0; j<s; j++){
      temp[j] = R::rgamma(A(i, j), 1);
      sum += temp[j];
    }
    for(int j=0; j<s; j++){
      res(i, j) = temp[j] / sum;
    }
  }
}

void transition_mat_update0(NumericVector pi, const arma::ivec & x, double alpha, int k){
  NumericVector pi_pars(k);
  initialise_const_vec(pi_pars, alpha, k);
  pi_pars[x[0]-1] += 1;
  rdirichlet_vec(pi_pars, pi, k);
}

void transition_mat_update1(NumericMatrix A, const arma::ivec & x, double alpha, int k, int n){
  NumericMatrix A_pars(k, k), AA(A);
  initialise_const_mat(A_pars, alpha, k, k);
  for(int t=0; t<(n-1); t++){
    A_pars(x[t]-1, x[t+1]-1) += 1;
  }
  rdirichlet_mat(A_pars, AA, k, k);
}

void transition_mat_update2(NumericMatrix B, const arma::ivec & x, IntegerVector y, double alpha, int k, int s, int n){
  NumericMatrix B_pars(k, s);
  initialise_const_mat(B_pars, alpha, k, s);
  for(int t=0; t<n; t++){
    B_pars(x[t]-1, y[t]-1) += 1;
  }
  rdirichlet_mat(B_pars, B, k, s);
}

void transition_mat_update3(NumericMatrix B, const arma::ivec & x, IntegerVector y, double alpha, int k, int s, int n, double inv_temperature){
  NumericMatrix B_pars(k, s);
  initialise_const_mat(B_pars, alpha, k, s);
  for(int t=0; t<n; t++){
    B_pars(x[t]-1, y[t]-1) += inv_temperature;
  }
  rdirichlet_mat(B_pars, B, k, s);
}

double loglikelihood(IntegerVector& y, arma::ivec& x, NumericMatrix& B, int n){
  double loglik = 0.0;
  for(int t=0; t<n; t++){
    loglik += log(B(x[t]-1, y[t]-1));
  }
  return loglik;
}

double loglikelihood_x(arma::ivec& x, NumericVector&pi, NumericMatrix& A, int n){
  double loglik = pi[x[0]-1];
  for(int t=1; t<n; t++){
    loglik += log(A(x[t-1]-1, x[t]-1));
  }
  return loglik;
}

double MH_acceptance_prob_swap_everything(IntegerVector& y, arma::ivec& x1, NumericMatrix& B1, arma::ivec& x2, NumericMatrix& B2, 
                                     double inv_temp1, double inv_temp2, int n){
  double loglik1 = loglikelihood(y, x1, B1, n);
  double loglik2 = loglikelihood(y, x2, B2, n);
  double ratio = exp(-(inv_temp1 - inv_temp2)*(loglik1 - loglik2));
  return ratio;
}

double MH_acceptance_prob_swap_pars(double marginal_loglik1, double marginal_loglik2, double inv_temp1, double inv_temp2){
  double ratio = exp(-(inv_temp1 - inv_temp2)*(marginal_loglik1 - marginal_loglik2));
  return ratio;
}

double MH_acceptance_prob_swap_x(IntegerVector& y, 
                                 arma::ivec& x1, NumericVector& pi1, NumericMatrix& A1, NumericMatrix& B1, 
                                 arma::ivec& x2, NumericVector& pi2, NumericMatrix& A2, NumericMatrix& B2, 
                                 double inv_temp1, double inv_temp2, int n){
  double logratio_x = loglikelihood_x(x1, pi2, A2, n) + loglikelihood_x(x2, pi1, A1, n) - loglikelihood_x(x1, pi1, A1, n) - loglikelihood_x(x2, pi2, A2, n);
  double logratio_y = inv_temp2 * loglikelihood(y, x1, B2, n) + inv_temp1 * loglikelihood(y, x2, B1, n) - inv_temp2 * loglikelihood(y, x2, B2, n) - inv_temp1 * loglikelihood(y, x1, B1, n);
  double ratio = exp(logratio_x + logratio_y);
  return ratio;
}

void initialise_transition_matrices(NumericVector pi, NumericMatrix A, NumericMatrix B, int k, int s){
  initialise_const_vec(pi, 1.0/k, k);
  initialise_const_mat(B, 1.0/s, k, s);
  // initialise A
  initialise_const_mat(A, 0.5*1.0/k, k, k);
  for(int i=0; i<k; i++)
    A(i, i) += 0.5;
}

//' @export
// [[Rcpp::export]]
List forward_backward_fast(NumericVector pi, NumericMatrix A, NumericMatrix B, IntegerVector y, int k, int n, bool marginal_distr){
  List PP(n), QQ(n);
  for(int t=0; t<n; t++){
    PP[t] = NumericMatrix(k, k);
    QQ[t] = NumericMatrix(k, k);
  }
  ListOf<NumericMatrix> P(PP), Q(QQ);
  double loglik=0.0;

  forward_step(pi, A, B, y, P, loglik, k, n);
  // now backward sampling
  arma::ivec x(n);
  IntegerVector possible_values = seq_len(k);
  backward_sampling(x, P, possible_values, k, n);
  // and backward recursion to obtain marginal distributions
  if(marginal_distr) backward_step(P, Q, k, n);

  IntegerVector xx = as<IntegerVector>(wrap(x));
  xx.attr("dim") = R_NilValue;
  return List::create(Rcpp::Named("x_draw") = xx,
                      Rcpp::Named("P") = P,
                      Rcpp::Named("Q") = Q,
                      Rcpp::Named("log_posterior") = loglik);
}

void save_current_iteration(List& trace_x, List& trace_pi, List& trace_A, List& trace_B, List& log_posterior, List& trace_switching_prob,
                            arma::ivec x, NumericVector pi, NumericMatrix A, NumericMatrix B, double& loglik, NumericVector switching_prob,
                            int index){
  IntegerVector xx(x.begin(), x.end());
  trace_x[index] = clone(xx);
  trace_pi[index] = clone(pi);
  trace_A[index] = clone(A);
  trace_B[index] = clone(B);
  log_posterior[index] = clone(wrap(loglik));
  trace_switching_prob[index] = clone(switching_prob);
}

//' @export
// [[Rcpp::export]]
List gibbs_sampling_fast_with_starting_vals(NumericVector pi0, NumericMatrix A0, NumericMatrix B0, IntegerVector y, double alpha, int k, int s, int n, int max_iter, int burnin, int thin, bool marginal_distr, bool is_fixed_B){
  NumericVector pi(clone(pi0));
  NumericMatrix A(clone(A0)), B(clone(B0));
  List PP(n), QQ(n);
  for(int t=0; t<n; t++){
    PP[t] = NumericMatrix(k, k);
    QQ[t] = NumericMatrix(k, k);
  }
  ListOf<NumericMatrix> P(PP), Q(QQ);
  arma::ivec x(n);

  int trace_length, index;
  trace_length = (max_iter - burnin + (thin - 1)) / thin;
  List trace_x(trace_length), trace_pi(trace_length), trace_A(trace_length), trace_B(trace_length), trace_switching_prob(trace_length), log_posterior(trace_length);
  double loglik;
  IntegerVector possible_values = seq_len(k);
  NumericVector switching_prob(n-1);
  NumericMatrix marginal_distr_res(k, n);

  for(int iter = 1; iter <= max_iter; iter++){
    // forward step
    forward_step(pi, A, B, y, P, loglik, k, n);
    // now backward sampling and nonstochastic backward step
    backward_sampling(x, P, possible_values, k, n);
    if(marginal_distr){
      backward_step(P, Q, k, n);
      switching_probabilities(Q, switching_prob, k, n);
      update_marginal_distr(Q, marginal_distr_res, k, n);
    }

    transition_mat_update0(pi, x, alpha, k);
    transition_mat_update1(A, x, alpha, k, n);
    if(!is_fixed_B) transition_mat_update2(B, x, y, alpha, k, s, n);

    if((iter > burnin) && ((iter-1) % thin == 0)){
      index = (iter - burnin - 1)/thin;
      save_current_iteration(trace_x, trace_pi, trace_A, trace_B, log_posterior, trace_switching_prob,
                             x, pi, A, B, loglik, switching_prob, index);
    }
    if(iter % 1000 == 0) printf("iter %d\n", iter);
  }
  // scale marginal distribution estimates
  arma::mat out(marginal_distr_res.begin(), k, n, false);
  out /= (float) (max_iter - burnin);

  return List::create(Rcpp::Named("trace_x") = trace_x,
                      Rcpp::Named("trace_pi") = trace_pi,
                      Rcpp::Named("trace_A") = trace_A,
                      Rcpp::Named("trace_B") = trace_B,
                      Rcpp::Named("log_posterior") = log_posterior,
                      Rcpp::Named("switching_prob") = trace_switching_prob,
                      Rcpp::Named("marginal_distr") = marginal_distr_res);
}


//' @export
// [[Rcpp::export]]
List gibbs_sampling_fast(IntegerVector y, double alpha, int k, int s, int n, int max_iter, int burnin, int thin, bool marginal_distr, bool is_fixed_B){
  NumericVector pi(k);
  NumericMatrix A(k, k), B(k, s);
  initialise_transition_matrices(pi, A, B, k, s);
  return gibbs_sampling_fast_with_starting_vals(pi, A, B, y, alpha, k, s, n, max_iter, burnin, thin, marginal_distr, is_fixed_B);
}

void initialise_mat_list(List& mat_list, int n, int k, int s){
  for(int t=0; t<n; t++){
    mat_list[t] = NumericMatrix(k, s);
  }
}


//' @export
// [[Rcpp::export]]
void crossover(arma::ivec& x, arma::ivec& y, int n){
  IntegerVector possible_values = seq_len(n-1);
  int m = as<int>(RcppArmadillo::sample(possible_values, 1, false, NumericVector::create()));
  arma::ivec temp = y.subvec(0, m-1);
  for(int i=0; i<m; i++){
    y[i] = x[i];
    x[i] = temp[i];
  }
}

//' @export
// [[Rcpp::export]]
void double_crossover(arma::ivec& x, arma::ivec& y, int n){
  IntegerVector possible_values = seq_len(n-1);
  int start = as<int>(RcppArmadillo::sample(possible_values, 1, false, NumericVector::create()));
  int end = as<int>(RcppArmadillo::sample(possible_values, 1, false, NumericVector::create()));
  if(start == end){
    return void();
  }
  if(start > end){
    int a = start;
    start = end;
    end = a;
  }
  arma::ivec temp = x.subvec(start, end-1);
  x.subvec(start, end-1) = y.subvec(start, end-1);
  y.subvec(start, end-1) = temp;
}

IntegerVector sample_helper(int n_chains, int n){
  IntegerVector possible_values = seq_len(n_chains);
  IntegerVector out = RcppArmadillo::sample(possible_values, n, false, NumericVector::create());
  return out;
}




// void initialise_trace_lists(List& tr_x, List& tr_pi, List& tr_A, List& tr_B,
//                             List& tr_switching_prob, List& tr_loglik, List& tr_marginal_distr,
//                             int k, int n, int trace_length, int n_chains){
//   for(int i=0; i<n_chains; i++){
//     tr_x[i] = List(trace_length);
//     tr_pi[i] = List(trace_length);
//     tr_A[i] = List(trace_length);
//     tr_B[i] = List(trace_length);
//     tr_switching_prob[i] = List(trace_length);
//     tr_loglik[i] = List(trace_length);
//     tr_marginal_distr[i] = NumericMatrix(k, n);
//   }
// }

void scale_marginal_distr(NumericMatrix marginal_distr_res, int k, int n, int max_iter, int burnin){
  arma::mat out(marginal_distr_res.begin(), k, n, false);
  out /= (float) (max_iter - burnin);
}

//' @export
// [[Rcpp::export]]
List ensemble(int n_chains, IntegerVector y, double alpha, int k, int s, int n, 
              int max_iter, int burnin, int thin, 
              bool estimate_marginals, bool is_fixed_B, bool parallel_tempering, bool crossovers, 
              NumericVector temperatures, int swap_type, int swaps_burnin, NumericMatrix B, IntegerVector which_chains){

  // initialise ensemble of n_chains
  Ensemble ensemble(n_chains, k, s, n, alpha, is_fixed_B);
  
  // initialise transition matrices for all chains in the ensemble
  if(is_fixed_B){
    ensemble.initialise_transition_matrices(B);
  } else{
    ensemble.initialise_transition_matrices();
  }
  
  // parallel tempering initilisation
  if(parallel_tempering){
    ensemble.activate_parallel_tempering(temperatures);
  }

  List PP(n), QQ(n);
  for(int t=0; t<n; t++){
    PP[t] = NumericMatrix(k, k);
    QQ[t] = NumericMatrix(k, k);
  }
  ListOf<NumericMatrix> P(PP), Q(QQ);

  int index;
  int n_chains_out = which_chains.size();
  int trace_length = (max_iter - burnin + (thin - 1)) / thin;
  int list_length = n_chains_out * trace_length;
  List tr_x(list_length), tr_pi(list_length), tr_A(list_length), tr_B(list_length), tr_switching_prob(list_length), tr_loglik(list_length), tr_loglik_cond(list_length);

  for(int iter = 1; iter <= max_iter; iter++){
    ensemble.update_chains(y, P, Q, estimate_marginals && (iter > burnin));

    if(crossovers && (iter > swaps_burnin)){
      ensemble.do_crossover();
    }
    if(parallel_tempering && (iter > swaps_burnin)){
      if(swap_type == 0) ensemble.swap_everything(y);
      if(swap_type == 1) ensemble.swap_pars(y);
      if(swap_type == 2) ensemble.swap_x(y);
    }
    
    if((iter > burnin) && ((iter-1) % thin == 0)){
      index = (iter - burnin - 1)/thin;
      ensemble.copy_values_to_trace(which_chains, tr_x, tr_pi, tr_A, tr_B, tr_loglik, tr_loglik_cond, tr_switching_prob, index);
    }
    if(iter % 1000 == 0) printf("iter %d\n", iter);
  }

  ensemble.scale_marginals(max_iter, burnin);
  ListOf<NumericMatrix> tr_marginal_distr = ensemble.get_copy_of_marginals(which_chains);

  return List::create(Rcpp::Named("trace_x") = tr_x,
                      Rcpp::Named("trace_pi") = tr_pi,
                      Rcpp::Named("trace_A") = tr_A,
                      Rcpp::Named("trace_B") = tr_B,
                      Rcpp::Named("log_posterior") = tr_loglik,
                      Rcpp::Named("log_posterior_cond") = tr_loglik_cond,
                      Rcpp::Named("switching_prob") = tr_switching_prob,
                      Rcpp::Named("marginal_distr") = tr_marginal_distr, 
                      Rcpp::Named("acceptance_ratio") = ensemble.get_acceptance_ratio());

}
