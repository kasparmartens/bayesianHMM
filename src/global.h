#include <RcppArmadillo.h>

using namespace Rcpp;

double normalise_mat(NumericMatrix A, int m, int n);

void compute_P(NumericMatrix PP, double& loglik, NumericVector pi, NumericMatrix A, NumericVector b, int k);

void compute_P0(NumericMatrix PP, double& loglik, NumericVector pi, NumericVector b, int k);


void compute_Q(NumericMatrix QQ, NumericMatrix PP, NumericVector pi_backward, NumericVector pi_forward, int k);

double calculate_nondiagonal_sum(NumericMatrix mat, int k);

NumericVector calculate_colsums(NumericMatrix A, int m, int n);

NumericVector calculate_rowsums(NumericMatrix A, int m, int n);

void update_marginal_distr(ListOf<NumericMatrix> Q, NumericMatrix res, int k, int n);

void initialise_const_vec(NumericVector pi, double alpha, int length);

void initialise_const_mat(NumericMatrix A, double alpha, int nrow, int ncol);

void forward_step(NumericVector pi, NumericMatrix A, NumericMatrix emission_probs, ListOf<NumericMatrix>& P, double& loglik, int k, int n);

void backward_sampling(arma::ivec& x, ListOf<NumericMatrix>& P, IntegerVector possible_values, int k, int n);

void backward_step(ListOf<NumericMatrix>& P, ListOf<NumericMatrix>& Q, int k, int n);

void switching_probabilities(ListOf<NumericMatrix>& Q, NumericVector res, int k, int n);

void rdirichlet_vec(NumericVector a, NumericVector res, int k);

void rdirichlet_mat(NumericMatrix A, NumericMatrix res, int k, int s);

void transition_mat_update0(NumericVector pi, const arma::ivec & x, double alpha, int k);

void transition_mat_update1(NumericMatrix A, const arma::ivec & x, double alpha, int k, int n);

void transition_mat_update2(NumericMatrix B, const arma::ivec & x, IntegerVector y, double alpha, int k, int s, int n);

void initialise_transition_matrices(NumericVector pi, NumericMatrix A, NumericMatrix B, int k, int s);

void crossover(arma::ivec& x, arma::ivec& y, int n);

void double_crossover(arma::ivec& x, arma::ivec& y, int n);

IntegerVector sample_helper(int n_chains, int n);

void transition_mat_update3(NumericMatrix B, const arma::ivec & x, IntegerVector y, double alpha, int k, int s, int n, double inv_temperature);

double loglikelihood(arma::ivec& x, NumericMatrix& B, int n);

double loglikelihood_x(arma::ivec& x, NumericVector&pi, NumericMatrix& A, int n);

double marginal_loglikelihood(NumericVector pi, NumericMatrix A, NumericMatrix emission_probs, double inv_temp, int k, int s, int n);
double MH_acceptance_prob_swap_everything(arma::ivec& x1, NumericMatrix& emission_probs1, arma::ivec& x2, NumericMatrix& emission_probs2, 
                                          double inv_temp1, double inv_temp2, int n);

//double MH_acceptance_prob_swap_pars(double marginal_loglik1, double marginal_loglik2, double inv_temp1, double inv_temp2);
double MH_acceptance_prob_swap_pars(NumericVector& pi1, NumericMatrix& A1, NumericMatrix& emission_probs1, 
                                    NumericVector& pi2, NumericMatrix& A2, NumericMatrix& emission_probs2, 
                                    double inv_temp1, double inv_temp2, int k, int s, int n);

double MH_acceptance_prob_swap_x(arma::ivec& x1, NumericVector& pi1, NumericMatrix& A1, NumericMatrix& emission_probs1, 
                                 arma::ivec& x2, NumericVector& pi2, NumericMatrix& A2, NumericMatrix& emission_probs2, 
                                 int n);

NumericMatrix emission_probs_mat_discrete(IntegerVector y, NumericMatrix B, int k, int n);

NumericMatrix emission_probs_mat_gaussian(NumericVector y, NumericVector mu, NumericVector sigma, int k, int n);

NumericMatrix temper_emission_probs(NumericMatrix mat, double inv_temperature, int k, int n);

void update_pars_gaussian(NumericVector& y, arma::ivec& x, NumericVector& mu, NumericVector& sigma, double rho, double inv_temp, int k, int n);
