// [[Rcpp::depends(RcppArmadillo)]]

#include "Ensemble_Gaussian.h"
#include "Ensemble_Discrete.h"
#include "Ensemble_Factorial.h"
#include <RcppArmadilloExtensions/sample.h>
#include <Rcpp/Benchmark/Timer.h>
using namespace Rcpp;
using namespace std;

void compute_P(NumericMatrix PP, double& loglik, NumericVector pi, NumericMatrix A, NumericVector b, int k){
  for(int s=0; s<k; s++){
    for(int r=0; r<k; r++){
      PP(r, s) = pi[r] * A(r, s) * b[s];
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

void update_mu(NumericVector mu, NumericVector sigma2, NumericVector n_k, NumericVector cluster_sums, double rho, double inv_temp, int k){
  double var, mean;
  for(int i=0; i<k; i++){
    var = 1.0 / (rho + inv_temp * n_k[i] / sigma2[i]);
    mean = inv_temp * var / sigma2[i] * cluster_sums[i];
    mu[i] = R::rnorm(mean, sqrt(var));
  }
}

void update_sigma(NumericVector sigma2, NumericVector n_k, NumericVector ss, double a0, double b0, double inv_temp, int k){
  double sigma2inv, a, b;
  for(int i=0; i<k; i++){
    a = a0 + 0.5 * inv_temp * n_k[i];
    b = b0 + 0.5 * inv_temp * ss[i];
    sigma2inv = R::rgamma(a, 1.0 / b);
    sigma2[i] = 1.0 / sigma2inv;
  }
}

void update_pars_gaussian(NumericVector& y, arma::ivec& x, NumericVector& mu, NumericVector& sigma2, double rho, double inv_temp, double a0, double b0, int k, int n){
  NumericVector n_k(k), cluster_sums(k), cluster_means(k);
  int index;
  // the number of elements in each component and their sums
  for(int t=0; t<n; t++){
    index = x[t];
    n_k[index] += 1;
    cluster_sums[index] += y[t];
  }
  // mean for each component
  for(int i=0; i<k; i++){
    cluster_means[i] = cluster_sums[i] / n_k[i];
  }
  // sum of squares for each component
  NumericVector ss(k);
  for(int t=0; t<n; t++){
    index = x[t];
    ss[index] += pow(y[t] - cluster_means[index], 2);
  }
  // draw sigma from its posterior
  update_sigma(sigma2, n_k, ss, a0, b0, inv_temp, k);
  // draw mu from its posterior mu|sigma2
  update_mu(mu, sigma2, n_k, cluster_sums, rho, inv_temp, k);
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


void forward_step(NumericVector pi, NumericMatrix A, NumericMatrix emission_probs, ListOf<NumericMatrix>& P, double& loglik, int k, int n){
  NumericVector b, colsums(k);
  b = emission_probs(_, 0);
  compute_P0(P[0], loglik, pi, b, k);
  loglik = 0.0;
  for(int t=1; t<n; t++){
    colsums = calculate_colsums(P[t-1], k, k);
    b = emission_probs(_, t);
    compute_P(P[t], loglik, colsums, A, b, k);
  }
}

// FHMM functions

void FHMM_compute_P(NumericMatrix PP, double& loglik, NumericVector pi, NumericMatrix A, NumericVector b, int k, 
                    IntegerVector which_states1, IntegerVector which_states2){
  // here k is length(which_states) or equivalently the ncol/nrow of PP
  int i, j;
  for(int s=0; s<k; s++){
    j = which_states2[s];
    for(int r=0; r<k; r++){
      i = which_states1[r];
      //printf("pi[%d] * A(%d, %d) * b[%d]\n", r, i, j, j);
      //printf("pi %e * A %e * b %e \n", pi[r], A(i, j), b[j]);
      PP(r, s) = pi[r] * A(i, j) * b[j] + 1.0e-16;
    }
  }
  double sum = normalise_mat(PP, k, k);
  loglik += log(sum);
}

void FHMM_compute_P0(NumericMatrix PP, double& loglik, NumericVector pi, NumericVector b, int k, 
                    IntegerVector which_states){
  // here k is length(which_states) or equivalently the ncol/nrow of PP
  int j;
  for(int s=0; s<k; s++){
    j = which_states[s];
    for(int r=0; r<k; r++){
      PP(r, s) = pi[r] * b[j] + 1.0e-16;
    }
  }
  double sum = normalise_mat(PP, k, k);
  loglik += log(sum);
}

void FHMM_forward_step(NumericVector pi, NumericMatrix A, NumericMatrix emission_probs, ListOf<NumericMatrix>& P, double& loglik, int k, int n, 
                       arma::ivec& x, IntegerMatrix all_hamming_balls){
  // here k == nrow(all_hamming_balls)
  NumericVector b, colsums(k);
  b = emission_probs(_, 0);
  FHMM_compute_P0(P[0], loglik, pi, b, k, all_hamming_balls(_, x[0]));
  loglik = 0.0;
  for(int t=1; t<n; t++){
    colsums = calculate_colsums(P[t-1], k, k);
    b = emission_probs(_, t);
    FHMM_compute_P(P[t], loglik, colsums, A, b, k, all_hamming_balls(_, x[t-1]), all_hamming_balls(_, x[t]));
  }
}

void FHMM_backward_sampling(arma::ivec& x, ListOf<NumericMatrix>& P, int k, int n, IntegerMatrix all_hamming_balls){
  NumericVector prob(k);
  NumericMatrix PP;
  IntegerVector x_temp(n);
  IntegerVector possible_values = seq_len(k)-1;
  prob = calculate_colsums(P[n-1], k, k);
  x_temp[n-1] = as<int>(RcppArmadillo::sample(possible_values, 1, false, prob));
  for(int t=n-1; t>0; t--){
    prob = P[t](_, x_temp[t]);
    x_temp[t-1] = as<int>(RcppArmadillo::sample(possible_values, 1, false, prob));
  }
  for(int t=0; t<n; t++){
    x[t] = all_hamming_balls(x_temp[t], x[t]);
  }
}

void sample_within_hamming_ball(arma::ivec& x, int n, IntegerMatrix hamming_balls){
  IntegerVector possible_values;
  for(int t=0; t<n; t++){
    // select u[t] uniformly within the hamming ball centered at x[t] (and overwrite it)
    possible_values = hamming_balls(_, x[t]);
    x[t] = as<int>(RcppArmadillo::sample(possible_values, 1, false, NumericVector::create()));
  }
}

NumericMatrix emission_probs_mat_gaussian(NumericVector y, NumericVector mu, NumericVector sigma2, int k, int n){
  NumericMatrix out(k, n);
  for(int t=0; t<n; t++){
    for(int i=0; i<k; i++){
      out(i, t) = R::dnorm4(y[t], mu[i], sqrt(sigma2[i]), false);
    }
  }
  out = out / max(out);
  return out;
}

NumericMatrix emission_probs_mat_discrete(IntegerVector y, NumericMatrix B, int k, int n){
  NumericMatrix out(k, n);
  for(int t=0; t<n; t++){
    out(_, t) = B(_, y[t]);
  }
  return out;
}

NumericMatrix temper_emission_probs(NumericMatrix mat, double inv_temperature, int k, int n){
  NumericMatrix out(k, n);
  for(int t=0; t<n; t++){
    for(int i=0; i<k; i++){
      out(i, t) = pow(mat(i, t), inv_temperature);
    }
  }
  out = out / max(out);
  return out;
}

// NumericVector gaussian_emission_probs(double y, NumericVector mu, NumericVector sigma, int k){
//   NumericVector out(k);
//   for(int i=0; i<k; i++)
//     out[i] = R::dnorm4(y, mu[i], sigma[i], false);
//   return out;
// }

void backward_sampling(arma::ivec& x, ListOf<NumericMatrix>& P, IntegerVector possible_values, int k, int n){
  NumericVector prob(k);
  NumericMatrix PP;
  prob = calculate_colsums(P[n-1], k, k);
  x[n-1] = as<int>(RcppArmadillo::sample(possible_values, 1, false, prob));
  for(int t=n-1; t>0; t--){
    prob = P[t](_, x[t]);
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
  double sum = 0.0;
  for(int i=0; i<k; i++){
    temp[i] = R::rgamma(a[i], 1.0) + 1.0e-16;
    sum += temp[i];
  }
  for(int i=0; i<k; i++){
    res[i] = temp[i] / sum;
  }
}

void rdirichlet_mat(NumericMatrix A, NumericMatrix res, int k, int s){
  NumericVector temp(s);
  for(int i=0; i<k; i++){
    double sum = 0.0;
    for(int j=0; j<s; j++){
      temp[j] = R::rgamma(A(i, j), 1.0) + 1.0e-16;
      sum += temp[j];
    }
    for(int j=0; j<s; j++){
      res(i, j) = temp[j] / sum;
    }
  }
}

void rdirichlet_mat(NumericMatrix A, NumericMatrix res, NumericMatrix Y, double alpha, int k, int s){
  NumericVector temp(s);
  for(int i=0; i<k; i++){
    double sum = 0.0;
    for(int j=0; j<s; j++){
      Y(i, j) = R::rgamma(A(i, j) + alpha, 1.0) + 1.0e-16;
      sum += Y(i, j);
    }
    for(int j=0; j<s; j++){
      res(i, j) = Y(i, j) / sum;
    }
  }
}

void transition_mat_update0(NumericVector pi, const arma::ivec & x, double alpha, int k){
  NumericVector pi_pars(k);
  initialise_const_vec(pi_pars, alpha, k);
  pi_pars[x[0]-1] += 1;
  rdirichlet_vec(pi_pars, pi, k);
}

double random_walk_log_scale(double current_value, double sd){
  double proposal = log(current_value) + R::rnorm(0, sd);
  return exp(proposal);
}

double calculate_logprob(double alpha, NumericMatrix A, NumericMatrix A_pars, double a0, double b0, int k, int s){
  double logprob = 0.0;
  // for each row i of transition matrix
  logprob = R::dgamma(alpha, a0, 1.0/b0, true) + log(alpha);
  for(int i=0; i<k; i++){
    logprob += Rf_lgammafn(s*alpha) - s*Rf_lgammafn(alpha);
    for(int j=0; j<s; j++){
      //logprob += R::dgamma(Y(i, j), alpha + A_pars(i, j), 1.0, true);
      logprob += (alpha-1)*log(A(i, j));
    }
  }
  return logprob;
}


void update_alpha(double& alpha, NumericMatrix Y, NumericMatrix A_pars, double a0, double b0, int k){
  double logprob_current = calculate_logprob(alpha, Y, A_pars, a0, b0, k, k);
  // propose new alpha
  double alpha_proposed = random_walk_log_scale(alpha, 0.3);
  double logprob_proposed = calculate_logprob(alpha_proposed, Y, A_pars, a0, b0, k, k);
  // accept or reject
  if(R::runif(0, 1) < exp(logprob_proposed - logprob_current)){
    alpha = alpha_proposed;
    //printf("new alpha: %f\n", alpha);
  }
}

void gamma_mat_to_dirichlet(NumericMatrix out, NumericMatrix& Y, int k, int s){
  for(int i=0; i<k; i++){
    double sum = 0;
    for(int j=0; j<s; j++){
      sum += Y(i, j);
    }
    for(int j=0; j<s; j++){
      out(i, j) = Y(i, j) / sum;
    }
  }
}


// void transition_mat_update1(NumericMatrix A, const arma::ivec & x, double alpha, int k, int n){
//   NumericMatrix A_pars(k, k), AA(A);
//   initialise_const_mat(A_pars, alpha, k, k);
//   // add 1 to diagonal
//   for(int i=0; i<k; i++)
//     A_pars(i, i) += 1.0;
//   // add transition counts
//   for(int t=0; t<(n-1); t++){
//     A_pars(x[t], x[t+1]) += 1;
//   }
//   rdirichlet_mat(A_pars, AA, k, k);
// }

void transition_mat_update1(NumericMatrix A, NumericMatrix A_pars, const arma::ivec & x, NumericMatrix Y, double alpha, int k, int n){
  initialise_const_mat(A_pars, 0.0, k, k);
  // add 1 to diagonal
  for(int i=0; i<k; i++)
    A_pars(i, i) += 1.0;
  // add transition counts
  for(int t=0; t<(n-1); t++){
    A_pars(x[t], x[t+1]) += 1.0;
  }
  rdirichlet_mat(A_pars, A, Y, alpha, k, k);
}

void transition_mat_update2(NumericMatrix B, const arma::ivec & x, IntegerVector y, double alpha, int k, int s, int n){
  NumericMatrix B_pars(k, s);
  initialise_const_mat(B_pars, alpha, k, s);
  for(int t=0; t<n; t++){
    B_pars(x[t], y[t]) += 1.0;
  }
  rdirichlet_mat(B_pars, B, k, s);
}

void transition_mat_update3(NumericMatrix B, const arma::ivec & x, IntegerVector y, double alpha, int k, int s, int n, double inv_temperature){
  NumericMatrix B_pars(k, s);
  initialise_const_mat(B_pars, alpha, k, s);
  for(int t=0; t<n; t++){
    B_pars(x[t], y[t]) += inv_temperature;
  }
  rdirichlet_mat(B_pars, B, k, s);
}

double loglikelihood(arma::ivec& x, NumericMatrix& emission_probs, int n){
  double loglik = 0.0;
  for(int t=0; t<n; t++){
    loglik += log(emission_probs(x[t], t));
  }
  return loglik;
}

// double loglikelihood(IntegerVector& y, arma::ivec& x, NumericMatrix& B, int n){
//   double loglik = 0.0;
//   for(int t=0; t<n; t++){
//     loglik += log(B(x[t], y[t]));
//   }
//   return loglik;
// }

double loglikelihood_x(arma::ivec& x, NumericVector&pi, NumericMatrix& A, int n){
  double loglik = pi[x[0]-1];
  for(int t=1; t<n; t++){
    loglik += log(A(x[t-1], x[t]));
  }
  return loglik;
}

double marginal_loglikelihood(NumericVector pi, NumericMatrix A, NumericMatrix emission_probs, double inv_temp, int k, int n){
  double loglik = 0.0;
  NumericMatrix PP(k, k);
  NumericVector b;
  
  NumericMatrix emission_probs_tempered = temper_emission_probs(emission_probs, inv_temp, k, n);
  
  for(int t=0; t<n; t++){
    b = emission_probs_tempered(_, t);
    for(int s=0; s<k; s++){
      for(int r=0; r<k; r++){
        if(t==0){
          PP(r, s) = pi[r] * b[s];
        } 
        else{
          PP(r, s) = pi[r] * A(r, s) * b[s];
        }
      }
    }
    loglik += log(normalise_mat(PP, k, k));
  }
  return loglik;
}

double MH_acceptance_prob_swap_everything(arma::ivec& x1, NumericMatrix& emission_probs1, arma::ivec& x2, NumericMatrix& emission_probs2, 
                                          double inv_temp1, double inv_temp2, int n){
  // here, emission_probs are already tempered. Need to "untemper" first
  double loglik1 = 1.0/inv_temp1 * loglikelihood(x1, emission_probs1, n);
  double loglik2 = 1.0/inv_temp2 * loglikelihood(x2, emission_probs2, n);
  double ratio = exp(-(inv_temp1 - inv_temp2)*(loglik1 - loglik2));
  return ratio;
}

double MH_acceptance_prob_swap_pars(NumericVector& pi1, NumericMatrix& A1, NumericMatrix& emission_probs1, 
                                    NumericVector& pi2, NumericMatrix& A2, NumericMatrix& emission_probs2, 
                                    double inv_temp1, double inv_temp2, int k, int n){
  double ll_12 = marginal_loglikelihood(pi1, A1, emission_probs1, inv_temp2, k, n);
  double ll_21 = marginal_loglikelihood(pi2, A2, emission_probs2, inv_temp1, k, n);
  double ll_11 = marginal_loglikelihood(pi1, A1, emission_probs1, inv_temp1, k, n);
  double ll_22 = marginal_loglikelihood(pi2, A2, emission_probs2, inv_temp2, k, n);
  double ratio = exp(ll_12 + ll_21 - ll_11 - ll_22);
  return ratio;
}

// double MH_acceptance_prob_swap_pars(double marginal_loglik1, double marginal_loglik2, double inv_temp1, double inv_temp2){
//   double ratio = exp(-(inv_temp1 - inv_temp2)*(marginal_loglik1 - marginal_loglik2));
//   return ratio;
// }

double MH_acceptance_prob_swap_x(arma::ivec& x1, NumericVector& pi1, NumericMatrix& A1, NumericMatrix& emission_probs1, 
                                 arma::ivec& x2, NumericVector& pi2, NumericMatrix& A2, NumericMatrix& emission_probs2, 
                                 int n){
  double logratio_x = loglikelihood_x(x1, pi2, A2, n) + loglikelihood_x(x2, pi1, A1, n) - loglikelihood_x(x1, pi1, A1, n) - loglikelihood_x(x2, pi2, A2, n);
  double logratio_y = loglikelihood(x1, emission_probs2, n) + loglikelihood(x2, emission_probs1, n) - loglikelihood(x2, emission_probs2, n) - loglikelihood(x1, emission_probs1, n);
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
  
  NumericMatrix emission_probs = emission_probs_mat_discrete(y, B, k, n);
  forward_step(pi, A, emission_probs, P, loglik, k, n);
  // now backward sampling
  arma::ivec x(n);
  IntegerVector possible_values = seq_len(k)-1;
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

// List gibbs_sampling_fast_with_starting_vals(NumericVector pi0, NumericMatrix A0, NumericMatrix B0, IntegerVector y, double alpha, int k, int s, int n, int max_iter, int burnin, int thin, bool marginal_distr, bool is_fixed_B){
//   NumericVector pi(clone(pi0));
//   NumericMatrix A(clone(A0)), B(clone(B0));
//   List PP(n), QQ(n);
//   for(int t=0; t<n; t++){
//     PP[t] = NumericMatrix(k, k);
//     QQ[t] = NumericMatrix(k, k);
//   }
//   ListOf<NumericMatrix> P(PP), Q(QQ);
//   arma::ivec x(n);
//   
//   int trace_length, index;
//   trace_length = (max_iter - burnin + (thin - 1)) / thin;
//   List trace_x(trace_length), trace_pi(trace_length), trace_A(trace_length), trace_B(trace_length), trace_switching_prob(trace_length), log_posterior(trace_length);
//   double loglik;
//   IntegerVector possible_values = seq_len(k)-1;
//   NumericVector switching_prob(n-1);
//   NumericMatrix marginal_distr_res(k, n);
//   NumericMatrix emission_probs(k, n);
//   
//   for(int iter = 1; iter <= max_iter; iter++){
//     // forward step
//     emission_probs = emission_probs_mat_discrete(y, B, k, n);
//     forward_step(pi, A, emission_probs, P, loglik, k, n);
//     // now backward sampling and nonstochastic backward step
//     backward_sampling(x, P, possible_values, k, n);
//     if(marginal_distr){
//       backward_step(P, Q, k, n);
//       switching_probabilities(Q, switching_prob, k, n);
//       update_marginal_distr(Q, marginal_distr_res, k, n);
//     }
//     
//     transition_mat_update0(pi, x, alpha, k);
//     transition_mat_update1(A, x, alpha, k, n);
//     if(!is_fixed_B) transition_mat_update2(B, x, y, alpha, k, s, n);
//     
//     if((iter > burnin) && ((iter-1) % thin == 0)){
//       index = (iter - burnin - 1)/thin;
//       save_current_iteration(trace_x, trace_pi, trace_A, trace_B, log_posterior, trace_switching_prob,
//                              x, pi, A, B, loglik, switching_prob, index);
//     }
//     if(iter % 1000 == 0) printf("iter %d\n", iter);
//   }
//   // scale marginal distribution estimates
//   arma::mat out(marginal_distr_res.begin(), k, n, false);
//   out /= (float) (max_iter - burnin);
//   
//   return List::create(Rcpp::Named("trace_x") = trace_x,
//                       Rcpp::Named("trace_pi") = trace_pi,
//                       Rcpp::Named("trace_A") = trace_A,
//                       Rcpp::Named("trace_B") = trace_B,
//                       Rcpp::Named("log_posterior") = log_posterior,
//                       Rcpp::Named("switching_prob") = trace_switching_prob,
//                       Rcpp::Named("marginal_distr") = marginal_distr_res);
// }

// 
// List gibbs_sampling_fast(IntegerVector y, double alpha, int k, int s, int n, int max_iter, int burnin, int thin, bool marginal_distr, bool is_fixed_B){
//   NumericVector pi(k);
//   NumericMatrix A(k, k), B(k, s);
//   initialise_transition_matrices(pi, A, B, k, s);
//   return gibbs_sampling_fast_with_starting_vals(pi, A, B, y, alpha, k, s, n, max_iter, burnin, thin, marginal_distr, is_fixed_B);
// }

void initialise_mat_list(List& mat_list, int n, int k, int s){
  for(int t=0; t<n; t++){
    mat_list[t] = NumericMatrix(k, s);
  }
}

// crossover of (x, y) at point t, resulting in subsequences
// (Cpp-indexing) 0:t and (t+1):(n) 

//' @export
// [[Rcpp::export]]
void crossover(arma::ivec& x, arma::ivec& y, int t){
  int temp;
  for(int i=0; i<=t; i++){
    temp = y[i];
    y[i] = x[i];
    x[i] = temp;
  }
}

void crossover2(arma::ivec& x, arma::ivec& y, int t, int n){
  int temp;
  for(int i=t+1; i<n; i++){
    temp = y[i];
    y[i] = x[i];
    x[i] = temp;
  }
}

void crossover_one_element(arma::ivec& x, arma::ivec& y, int t){
  int temp = y[t];
  y[t] = x[t];
  x[t] = temp;
}

//' @export
// [[Rcpp::export]]
void crossover_mat(IntegerMatrix X, IntegerMatrix Y, int t, IntegerVector which_rows){
  int m = which_rows.size();
  for(int i=0; i<=t; i++){
    crossover_one_column(X, Y, i, which_rows, m);
  }
}

void crossover2_mat(IntegerMatrix X, IntegerMatrix Y, int t, int n, IntegerVector which_rows){
  int m = which_rows.size();
  for(int i=t+1; i<n; i++){
    crossover_one_column(X, Y, i, which_rows, m);
  }
}

void crossover_one_column(IntegerMatrix X, IntegerMatrix Y, int t, IntegerVector which_rows, int m){
  int index, temp;
  for(int k=0; k<m; k++){
    index = which_rows[k];
    temp = Y(index, t);
    Y(index, t) = X(index, t);
    X(index, t) = temp;
  }
}

double crossover_likelihood(const arma::ivec& x, const arma::ivec& y, int t, int n, NumericMatrix Ax, NumericMatrix Ay){
  if((t == 0) || (t==n)){
    return 1.0;
  } else{
    double num = Ax(y[t-1], x[t]) * Ay(x[t-1], y[t]);
    double denom = Ax(x[t-1], x[t]) * Ay(y[t-1], y[t]) + 1.0e-15;
    return num / denom;
  }
}



// void uniform_crossover(arma::ivec& x, arma::ivec& y, int n){
//   IntegerVector possible_values = seq_len(n);
//   int m = as<int>(RcppArmadillo::sample(possible_values, 1, false, NumericVector::create()));
//   crossover(x, y, m);
// }

// void nonuniform_crossover(arma::ivec& x, arma::ivec& y, NumericVector& probs, int n){
//   IntegerVector possible_values = seq_len(n);
//   int m = as<int>(RcppArmadillo::sample(possible_values, 1, false, probs));
//   crossover(x, y, m);
// }
// 
// void nonuniform_crossover2(arma::ivec& x, arma::ivec& y, NumericVector& probs, int n){
//   IntegerVector possible_values = seq_len(2*n);
//   int m = as<int>(RcppArmadillo::sample(possible_values, 1, false, probs));
//   if(m <= n){
//     //printf("normal crossover, m = %d", m);
//     crossover(x, y, m);
//   } else{
//     //printf("flipped crossover, m-n = %d", m-n);
//     crossover(y, x, m-n);
//   }
// }


IntegerVector sample_helper(int n_chains, int n){
  IntegerVector possible_values = seq_len(n_chains)-1;
  IntegerVector out = RcppArmadillo::sample(possible_values, n, false, NumericVector::create());
  return out;
}

int sample_int(int n){
  IntegerVector possible_values = seq_len(n)-1;
  int out = as<int>(RcppArmadillo::sample(possible_values, 1, false, NumericVector::create()));
  return out;
}

int sample_int(int n, NumericVector probs){
  IntegerVector possible_values = seq_len(n)-1;
  int out = as<int>(RcppArmadillo::sample(possible_values, 1, false, probs));
  return out;
}

void scale_marginal_distr(NumericMatrix marginal_distr_res, int k, int n, int max_iter, int burnin){
  arma::mat out(marginal_distr_res.begin(), k, n, false);
  out /= (float) (max_iter - burnin);
}

void helper_binary_matrix(IntegerMatrix& A, int nrows, int ncols, int i, int start_col, int end_col){
  int middle_col = start_col + (end_col - start_col)/2;
  for(int j=start_col; j<middle_col; j++){
    A(i, j) = 0;
  }
  for(int j=middle_col; j<end_col; j++){
    A(i, j) = 1;
  }
  if((i+1) < nrows){
    helper_binary_matrix(A, nrows, ncols, i+1, start_col, middle_col);
    helper_binary_matrix(A, nrows, ncols, i+1, middle_col, end_col);
  }
}

//' @export
// [[Rcpp::export]]
IntegerMatrix decimal_to_binary_mapping(int K){
  int ncols = pow(2, K);
  int nrows = K;
  IntegerMatrix out(nrows, ncols);
  helper_binary_matrix(out, nrows, ncols, 0, 0, ncols);
  return out;
}

// IntegerMatrix calculate_hamming_dist(IntegerMatrix& mapping){
//   int K = mapping.nrow();
//   int ncols = mapping.ncol();
//   IntegerMatrix dist(ncols, ncols);
//   // for each pair of configurations calculate hamming distance
//   for(int j=0; j<ncols; j++){
//     for(int jj=0; jj<ncols; jj++){
//       for(int i=0; i<K; i++){
//         if(mapping(i, j) != mapping(i, jj)){
//           dist(j, jj) += 1;
//         }
//       }
//     }
//   }
//   return dist;
// }

int hamming_distance(IntegerVector x, IntegerVector y){
  int dist = 0;
  for(int i=0; i<x.size(); i++){
    if(x[i] != y[i]) dist += 1;
  }
  return dist;
}

IntegerVector logical_to_ind(LogicalVector x, int n){
  int length = sum(x);
  IntegerVector out(length);
  int counter = 0;
  for(int i=0; i<n; i++){
    if(x[i]){
      out[counter] = i;
      counter += 1;
    }
  }
  return out;
}

//' @export
// [[Rcpp::export]]
IntegerVector hamming_ball(int index, int radius, IntegerMatrix& mapping){
  int n_states = mapping.ncol();
  LogicalVector boolean(n_states);
  // center of the ball
  IntegerVector x = mapping(_, index);
  // find all elements witihn B(x, r)
  for(int i=0; i<n_states; i++){
    if(hamming_distance(x, mapping(_, i)) <= radius){
      boolean[i] = true;
    }
  }
  IntegerVector out = logical_to_ind(boolean, n_states);
  return out;
}

//' @export
// [[Rcpp::export]]
IntegerMatrix construct_all_hamming_balls(int radius, IntegerMatrix& mapping){
  IntegerVector x = hamming_ball(0, radius, mapping);
  int n_elements_inside_ball = x.size();
  int n_states = mapping.ncol();
  IntegerMatrix out(n_elements_inside_ball, n_states);
  for(int i=0; i<n_states; i++){
    out(_, i) = hamming_ball(i, radius, mapping);
  }
  return out;
}


// helpers for block gibbs sampling

bool subset_rows_match(IntegerVector x, IntegerVector y, IntegerVector which_rows){
  bool match = true;
  for(int i=0; i<which_rows.size(); i++){
    if(x[which_rows[i]] != y[which_rows[i]]){
      match = false;
    }
  }
  return match;
}

//' @export
// [[Rcpp::export]]
IntegerVector construct_restricted_space(int x_t, IntegerVector which_rows_fixed, IntegerMatrix mapping){
  int n_states = mapping.ncol();
  LogicalVector boolean(n_states);
  for(int i=0; i<n_states; i++){
    // check whether mapping(which_rows_fixed, i) matches mapping(which_rows_fixed, x_t)
    boolean[i] = subset_rows_match(mapping(_, i), mapping(_, x_t), which_rows_fixed);
  }
  IntegerVector out = logical_to_ind(boolean, n_states);
  return out;
}

//' @export
// [[Rcpp::export]]
IntegerMatrix construct_all_restricted_space(int k_restricted, IntegerVector which_rows_fixed, IntegerMatrix mapping){
  int n_states = mapping.ncol();
  IntegerMatrix out(k_restricted, n_states);
  for(int i=0; i<n_states; i++){
    out(_, i) = construct_restricted_space(i, which_rows_fixed, mapping);
  }
  return out;
}


//' @export
// [[Rcpp::export]]
List ensemble_gaussian(int n_chains, NumericVector y, double alpha, int k, int n, 
                       int max_iter, int burnin, int thin, 
                       bool estimate_marginals, bool fixed_pars, bool parallel_tempering, bool crossovers, 
                       NumericVector temperatures, int swap_type, int swaps_burnin, int swaps_freq, NumericVector mu, NumericVector sigma2, 
                       IntegerVector which_chains, IntegerVector subsequence, IntegerVector x){
  
  // initialise ensemble of n_chains
  Ensemble_Gaussian ensemble(n_chains, k, n, alpha, fixed_pars);
  
  // initialise transition matrices and x latent sequences for all chains
  if((mu.size() != 0) & (x.size() != 0)){
    ensemble.initialise_pars(mu, sigma2, x);
  } else if(mu.size() != 0){
    ensemble.initialise_pars(mu, sigma2);
    // initialise x
    ensemble.update_x(y, false);
  } else{
    ensemble.initialise_pars();
    // initialise x
    ensemble.update_x(y, false);
  }
  
  // parallel tempering initilisation
  if(parallel_tempering){
    ensemble.activate_parallel_tempering(temperatures);
  }
  
  int index;
  int n_chains_out = which_chains.size();
  int trace_length = (max_iter - burnin + (thin - 1)) / thin;
  int list_length = n_chains_out * trace_length;
  List tr_x(list_length), tr_pi(list_length), tr_A(list_length), tr_mu(list_length), tr_sigma2(list_length), tr_alpha(list_length), tr_switching_prob(list_length), tr_loglik(list_length), tr_loglik_cond(list_length);
  List tr_crossovers(trace_length);
  
  Timer timer;
  nanotime_t t0, t1, t2, t3;
  NumericVector comp_times(3);
  for(int iter = 1; iter <= max_iter; iter++){
    t0 = timer.now();
    ensemble.update_pars(y);
    t1 = timer.now();
    ensemble.update_x(y, estimate_marginals && (iter > burnin));
    t2 = timer.now();
    
    if(crossovers && (iter > swaps_burnin) && ((iter-1) % swaps_freq == 0)){
      ensemble.do_crossover();
    }
    if(parallel_tempering && (iter > swaps_burnin) && ((iter-1) % swaps_freq == 0)){
      if(swap_type == 0) ensemble.swap_everything();
      if(swap_type == 1) ensemble.swap_pars();
      if(swap_type == 2) ensemble.swap_x();
    }
    t3 = timer.now();
    
    if((iter > burnin) && ((iter-1) % thin == 0)){
      index = (iter - burnin - 1)/thin;
      ensemble.copy_values_to_trace(which_chains, tr_x, tr_pi, tr_A, tr_mu, tr_sigma2, tr_alpha, tr_loglik, tr_loglik_cond, tr_switching_prob, index, subsequence);
      if(crossovers && (iter > swaps_burnin) && ((iter-1) % swaps_freq == 0)){
        tr_crossovers[index] = ensemble.get_crossovers();  
      }
      comp_times += 1.0/trace_length * NumericVector::create(t1-t0, t2-t1, t3-t2);
      comp_times[0] += 1.0/trace_length * (t1 - t0);
      comp_times[1] += 1.0/trace_length * (t2 - t1);
      if((iter-1) % swaps_freq == 0){
        comp_times[2] += 1.0/trace_length * swaps_freq * (t3 - t2);
      }
    }
    if(iter % 1000 == 0) printf("iter %d\n", iter);
  }
  comp_times.attr("names") = CharacterVector::create("update pars", "update x", "swap/crossover");
  
  ensemble.scale_marginals(max_iter, burnin);
  ListOf<NumericMatrix> tr_marginal_distr = ensemble.get_copy_of_marginals(which_chains);
  
  return List::create(Rcpp::Named("trace_x") = tr_x,
                      Rcpp::Named("trace_pi") = tr_pi,
                      Rcpp::Named("trace_A") = tr_A,
                      Rcpp::Named("trace_mu") = tr_mu,
                      Rcpp::Named("trace_sigma2") = tr_sigma2,
                      Rcpp::Named("trace_alpha") = tr_alpha,
                      Rcpp::Named("log_posterior") = tr_loglik,
                      Rcpp::Named("log_posterior_cond") = tr_loglik_cond,
                      Rcpp::Named("switching_prob") = tr_switching_prob,
                      Rcpp::Named("marginal_distr") = tr_marginal_distr, 
                      Rcpp::Named("acceptance_ratio") = ensemble.get_acceptance_ratio(), 
                      Rcpp::Named("timer") = comp_times, 
                      Rcpp::Named("crossovers") = tr_crossovers);
  
}

//' @export
// [[Rcpp::export]]
List ensemble_discrete(int n_chains, IntegerVector y, double alpha, int k, int s, int n, 
                       int max_iter, int burnin, int thin, 
                       bool estimate_marginals, bool fixed_pars, bool parallel_tempering, bool crossovers, 
                       NumericVector temperatures, int swap_type, int swaps_burnin, int swaps_freq, NumericMatrix B, 
                       IntegerVector which_chains, IntegerVector subsequence){
  
  // initialise ensemble of n_chains
  Ensemble_Discrete ensemble(n_chains, k, s, n, alpha, fixed_pars);
  
  // initialise transition matrices for all chains in the ensemble
  ensemble.initialise_pars();
  if(fixed_pars){
    ensemble.initialise_pars(B);
  }
  
  // parallel tempering initilisation
  if(parallel_tempering){
    ensemble.activate_parallel_tempering(temperatures);
  }
  
  // initialise x
  ensemble.update_x(y, false);
  
  int index;
  int n_chains_out = which_chains.size();
  int trace_length = (max_iter - burnin + (thin - 1)) / thin;
  int list_length = n_chains_out * trace_length;
  List tr_x(list_length), tr_pi(list_length), tr_A(list_length), tr_B(list_length), tr_switching_prob(list_length), tr_loglik(list_length), tr_loglik_cond(list_length), tr_alpha(list_length);
  
  Timer timer;
  nanotime_t t0, t1, t2, t3;
  NumericVector comp_times(3);
  for(int iter = 1; iter <= max_iter; iter++){
    t0 = timer.now();
    ensemble.update_pars(y);
    t1 = timer.now();
    ensemble.update_x(y, estimate_marginals && (iter > burnin));
    t2 = timer.now();
    
    if(crossovers && (iter > swaps_burnin) && ((iter-1) % swaps_freq == 0)){
      ensemble.do_crossover();
    }
    if(parallel_tempering && (iter > swaps_burnin) && ((iter-1) % swaps_freq == 0)){
      if(swap_type == 0) ensemble.swap_everything();
      if(swap_type == 1) ensemble.swap_pars();
      if(swap_type == 2) ensemble.swap_x();
    }
    t3 = timer.now();
    
    if((iter > burnin) && ((iter-1) % thin == 0)){
      index = (iter - burnin - 1)/thin;
      ensemble.copy_values_to_trace(which_chains, tr_x, tr_pi, tr_A, tr_B, tr_alpha, tr_loglik, tr_loglik_cond, tr_switching_prob, index, subsequence);
      comp_times += 1.0/trace_length * NumericVector::create(t1-t0, t2-t1, t3-t2);
      comp_times[0] += 1.0/trace_length * (t1 - t0);
      comp_times[1] += 1.0/trace_length * (t2 - t1);
      if((iter-1) % swaps_freq == 0){
        comp_times[2] += 1.0/trace_length * swaps_freq * (t3 - t2);
      }
    }
    if(iter % 1000 == 0) printf("iter %d\n", iter);
  }
  comp_times.attr("names") = CharacterVector::create("update pars", "update x", "swap/crossover");
  
  ensemble.scale_marginals(max_iter, burnin);
  ListOf<NumericMatrix> tr_marginal_distr = ensemble.get_copy_of_marginals(which_chains);
  
  return List::create(Rcpp::Named("trace_x") = tr_x,
                      Rcpp::Named("trace_pi") = tr_pi,
                      Rcpp::Named("trace_A") = tr_A,
                      Rcpp::Named("trace_B") = tr_B,
                      Rcpp::Named("trace_alpha") = tr_alpha,
                      Rcpp::Named("log_posterior") = tr_loglik,
                      Rcpp::Named("log_posterior_cond") = tr_loglik_cond,
                      Rcpp::Named("switching_prob") = tr_switching_prob,
                      Rcpp::Named("marginal_distr") = tr_marginal_distr, 
                      Rcpp::Named("acceptance_ratio") = ensemble.get_acceptance_ratio(), 
                      Rcpp::Named("timer") = comp_times);
  
}

//' @export
// [[Rcpp::export]]
List ensemble_FHMM(int n_chains, NumericMatrix Y, NumericMatrix mu, double sigma, NumericMatrix A, double alpha, 
          int K, int k, int n, int radius, 
                       int max_iter, int burnin, int thin, 
                       bool estimate_marginals, bool parallel_tempering, bool crossovers, 
                       NumericVector temperatures, int swap_type, int swaps_burnin, int swaps_freq, 
                       IntegerVector which_chains, IntegerVector subsequence, IntegerVector x, 
                       int nrows_crossover, bool HB_sampling, int nrows_gibbs, IntegerMatrix all_combs){
  
  // initialise ensemble of n_chains
  Ensemble_Factorial ensemble(n_chains, K, k, n, alpha, radius, nrows_crossover, HB_sampling, nrows_gibbs, all_combs);
  
  ensemble.set_temperatures(temperatures);
  
  // all parameters must be fixed, given as inputs
  ensemble.initialise_pars(mu, sigma, A, x);
  ensemble.update_emission_probs(Y);
  
  int index;
  int n_chains_out = which_chains.size();
  int trace_length = (max_iter - burnin + (thin - 1)) / thin;
  int list_length = n_chains_out * trace_length;
  List tr_x(list_length), tr_X(list_length), tr_pi(list_length), tr_A(list_length), tr_mu(list_length), tr_sigma2(list_length), tr_alpha(list_length), tr_switching_prob(list_length), tr_loglik(list_length), tr_loglik_cond(list_length);
  List tr_crossovers(trace_length);
  
  Timer timer;
  nanotime_t t0, t1;
  t0 = timer.now();
  for(int iter = 1; iter <= max_iter; iter++){
    
    ensemble.update_x();
    
    if(crossovers && (iter > swaps_burnin) && ((iter-1) % swaps_freq == 0)){
      ensemble.do_crossover();
    }

    if((iter > burnin) && ((iter-1) % thin == 0)){
      index = (iter - burnin - 1)/thin;
      ensemble.copy_values_to_trace(which_chains, tr_x, tr_X, tr_pi, tr_A, tr_mu, tr_sigma2, tr_alpha, tr_loglik, tr_loglik_cond, tr_switching_prob, index, subsequence);
      if(crossovers && (iter > swaps_burnin) && ((iter-1) % swaps_freq == 0)){
        tr_crossovers[index] = ensemble.get_crossovers();  
      }
    }
    if(iter % 1000 == 0) printf("iter %d\n", iter);
  }

  //ensemble.scale_marginals(max_iter, burnin);
  //ListOf<NumericMatrix> tr_marginal_distr = ensemble.get_copy_of_marginals(which_chains);
  
  t1 = timer.now();
  return List::create(Rcpp::Named("trace_x") = tr_x, 
                      Rcpp::Named("trace_X") = tr_X,
                      Rcpp::Named("trace_pi") = tr_pi,
                      Rcpp::Named("trace_A") = tr_A,
                      Rcpp::Named("trace_mu") = tr_mu,
                      Rcpp::Named("trace_sigma2") = tr_sigma2,
                      Rcpp::Named("trace_alpha") = tr_alpha,
                      Rcpp::Named("log_posterior") = tr_loglik,
                      Rcpp::Named("log_posterior_cond") = tr_loglik_cond,
                      Rcpp::Named("switching_prob") = tr_switching_prob,
                      //Rcpp::Named("marginal_distr") = tr_marginal_distr, 
                      //Rcpp::Named("acceptance_ratio") = ensemble.get_acceptance_ratio(), 
                      Rcpp::Named("timer") = t1-t0, 
                      Rcpp::Named("crossovers") = tr_crossovers);
  
}
