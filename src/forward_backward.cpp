#include <RcppArmadilloExtensions/sample.h>
using namespace Rcpp;
using namespace std;

void normalise_mat(NumericMatrix mat, int m, int n, double sum){
  for(int s=0; s<n; s++){
    for(int r=0; r<m; r++){
      mat(r, s) /= sum;
    }
  }
}

void compute_P(ListOf<NumericMatrix> P, int t, NumericVector pi, NumericMatrix A, NumericVector b, int k){
  NumericMatrix PP(k, k);
  double sum = 0;
  double temp;
  for(int s=0; s<k; s++){
    for(int r=0; r<k; r++){
      temp = pi[r] * A(r, s) * b[s];
      sum += temp;
      PP(r, s) = temp;
    }
  }
  normalise_mat(PP, k, k, sum);
  P[t] = clone(PP);
}

void compute_P0(ListOf<NumericMatrix> P, NumericVector pi, NumericVector b, int k){
  NumericMatrix PP(k, k);
  double sum = 0;
  for(int s=0; s<k; s++){
    for(int r=0; r<k; r++){
      PP(r, s) = pi[r] * b[s];
      sum += PP(r, s);
    }
  }
  normalise_mat(PP, k, k, sum);
  P[0] = clone(PP);
}

void compute_Q(ListOf<NumericMatrix> Q, ListOf<NumericMatrix> P, int t, NumericVector pi_backward, NumericVector pi_forward, int k){
  //NumericMatrix Q(k, k);
  NumericMatrix PP(P[t]), QQ(Q[t]);
  for(int s=0; s<k; s++){
    if(pi_forward[s]>0){
      for(int r=0; r<k; r++){
        QQ(r, s) = PP(r, s) * pi_backward[s] / pi_forward[s];
      }
    }
  }
  Q[t] = clone(QQ);
}

void calculate_colsums(NumericMatrix mat, NumericVector res, int m, int n){
  double temp;
  for(int j=0; j<n; j++){
    temp = 0;
    for(int i=0; i<m; i++){
      temp += mat(i, j);
    }
    res[j] = temp;
  }
}

void calculate_rowsums(NumericMatrix mat, NumericVector res, int m, int n){
  double temp;
  for(int i=0; i<m; i++){
    temp = 0;
    for(int j=0; j<n; j++){
      temp += mat(i, j);
    }
    res[i] = temp;
  }
}

void initialise_const_vec(NumericVector pi, double alpha, int length){
  for(int i=0; i<length; i++){
    pi[i] = alpha;
  }
}

void initialise_const_mat(NumericMatrix A, double alpha, int nrow, int ncol){
  for(int i=0; i<nrow; i++){
    for(int j=0; j<ncol; j++){
      A(i, j) = alpha;
    }
  }
}

void forward_step(NumericVector pi, NumericMatrix A, NumericMatrix B, IntegerVector y, ListOf<NumericMatrix> P, int k, int n){
  NumericVector b, colsums(k);
  b = B(_, y[0]-1);
  compute_P0(P, pi, b, k);
  for(int t=1; t<n; t++){
    calculate_colsums(P[t-1], colsums, k, k);
    //loglikelihood += std::accumulate(colsums.begin(), colsums.end(), 0.0);
    b = B(_, y[t]-1);
    compute_P(P, t, colsums, A, b, k);
  }
}

void forward_step(NumericVector pi, NumericMatrix A, NumericMatrix B, IntegerVector y, ListOf<NumericMatrix> P, double& loglikelihood, int k, int n){
  NumericVector b, colsums(k);
  b = B(_, y[0]-1);
  compute_P0(P, pi, b, k);
  for(int t=1; t<n; t++){
    calculate_colsums(P[t-1], colsums, k, k);
    loglikelihood += std::accumulate(colsums.begin(), colsums.end(), 0.0);
    b = B(_, y[t]-1);
    compute_P(P, t, colsums, A, b, k);
  }
}

void backward_sampling(arma::ivec& x, ListOf<NumericMatrix> P, IntegerVector possible_values, int k, int n){
  NumericVector prob(k);
  NumericMatrix PP;
  calculate_colsums(P[n-1], prob, k, k);
  x[n-1] = as<int>(RcppArmadillo::sample(possible_values, 1, false, prob));
  for(int t=n-1; t>0; t--){
    prob = P[t](_, x[t]-1);
    x[t-1] = as<int>(RcppArmadillo::sample(possible_values, 1, false, prob));
  }
}

void backward_step(ListOf<NumericMatrix> P, ListOf<NumericMatrix> Q, int k, int n){
  NumericVector q_forward(k), q_backward(k);
  Q[n-1] = P[n-1];
  for(int t=n-2; t>=0; t--){
    calculate_colsums(P[t], q_forward, k, k);
    calculate_rowsums(Q[t+1], q_backward, k, k);
    compute_Q(Q, P, t, q_backward, q_forward, k);
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

void transition_mat_update0(NumericVector pi, NumericVector pi_pars, const arma::ivec & x, double alpha, int k){
  initialise_const_vec(pi_pars, alpha, k);
  pi_pars[x[0]-1] += 1;
  rdirichlet_vec(pi_pars, pi, k);
}

void transition_mat_update1(NumericMatrix A, NumericMatrix A_pars, const arma::ivec & x, double alpha, int k, int n){
  initialise_const_mat(A_pars, alpha, k, k);
  for(int t=0; t<(n-1); t++){
    A_pars(x[t]-1, x[t+1]-1) += 1;
  }
  rdirichlet_mat(A_pars, A, k, k);
}

void transition_mat_update2(NumericMatrix B, NumericMatrix B_pars, const arma::ivec & x, IntegerVector y, double alpha, int k, int s, int n){
  initialise_const_mat(B_pars, alpha, k, s);
  for(int t=0; t<n; t++){
    B_pars(x[t]-1, y[t]-1) += 1;
  }
  rdirichlet_mat(B_pars, B, k, s);
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
  NumericMatrix temp(k, k);
  for(int t=0; t<n; t++){
    PP[t] = temp;
    QQ[t] = temp;
  }
  ListOf<NumericMatrix> P(PP), Q(QQ);
  forward_step(pi, A, B, y, P, k, n);
  // now backward sampling
  arma::ivec x(n);
  IntegerVector possible_values = seq_len(k);
  backward_sampling(x, P, possible_values, k, n);
  // and backward recursion to obtain marginal distributions
  if(marginal_distr) backward_step(P, Q, k, n);

  IntegerVector xx = as<IntegerVector>(wrap(x));
  xx.attr("dim") = R_NilValue;
  return List::create(Rcpp::Named("x_draw") = clone(xx),
                      Rcpp::Named("P") = P,
                      Rcpp::Named("Q") = Q);;
}

//' @export
// [[Rcpp::export]]
List gibbs_sampling_fast(IntegerVector y, double alpha, int k, int s, int n, int max_iter, int burnin, int thin, bool marginal_distr){
  NumericVector pi(k), pi_pars(k);
  NumericMatrix A(k, k), B(k, s), A_pars(k, k), B_pars(k, s);
  List PP(n), QQ(n);
  NumericMatrix temp(k, k);
  for(int t=0; t<n; t++){
    PP[t] = temp;
    QQ[t] = temp;
  }
  ListOf<NumericMatrix> P(PP), Q(QQ);
  arma::ivec x(n);
  IntegerVector xx(n);

  int trace_length, index;
  trace_length = (max_iter - burnin + (thin - 1)) / thin;
  List trace_x(trace_length), trace_pi(trace_length), trace_A(trace_length), trace_B(trace_length);
  IntegerVector possible_values = seq_len(k);

  initialise_transition_matrices(pi, A, B, k, s);

  for(int iter = 1; iter <= max_iter; iter++){
    // forward step
    forward_step(pi, A, B, y, P, k, n);
    // now backward sampling and nonstochastic backward step
    backward_sampling(x, P, possible_values, k, n);
    if(marginal_distr) backward_step(P, Q, k, n);

    transition_mat_update0(pi, pi_pars, x, alpha, k);
    transition_mat_update1(A, A_pars, x, alpha, k, n);
    transition_mat_update2(B, B_pars, x, y, alpha, k, s, n);

    if(iter > burnin){
      index = (iter - burnin - 1)/thin;
      xx = as<IntegerVector>(wrap(x));
      xx.attr("dim") = R_NilValue;
      trace_x[index] = clone(xx);
      trace_pi[index] = clone(pi);
      trace_A[index] = clone(A);
      trace_B[index] = clone(B);
    }
    if(iter % 100 == 0) printf("iter %d\n", iter);
  }

  return List::create(Rcpp::Named("trace_x") = trace_x,
                      Rcpp::Named("trace_pi") = trace_pi,
                      Rcpp::Named("trace_A") = trace_A,
                      Rcpp::Named("trace_B") = trace_B);;
}

//' @export
// [[Rcpp::export]]
List gibbs_sampling_fast_with_starting_vals(NumericVector pi0, NumericMatrix A0, NumericMatrix B0, IntegerVector y, double alpha, int k, int s, int n, int max_iter, int burnin, int thin, bool marginal_distr){
  NumericVector pi_pars(k), pi(clone(pi0));
  NumericMatrix A_pars(k, k), B_pars(k, s), A(clone(A0)), B(clone(B0));
  List PP(n), QQ(n);
  NumericMatrix temp(k, k);
  for(int t=0; t<n; t++){
    PP[t] = temp;
    QQ[t] = temp;
  }
  ListOf<NumericMatrix> P(PP), Q(QQ);
  arma::ivec x(n);
  IntegerVector xx(n);

  int trace_length, index;
  trace_length = (max_iter - burnin + (thin - 1)) / thin;
  List trace_x(trace_length), trace_pi(trace_length), trace_A(trace_length), trace_B(trace_length);
  IntegerVector possible_values = seq_len(k);

  //initialise_transition_matrices(pi, A, B, k, s);

  for(int iter = 1; iter <= max_iter; iter++){
    // forward step
    forward_step(pi, A, B, y, P, k, n);
    // now backward sampling and nonstochastic backward step
    backward_sampling(x, P, possible_values, k, n);
    if(marginal_distr) backward_step(P, Q, k, n);

    transition_mat_update0(pi, pi_pars, x, alpha, k);
    transition_mat_update1(A, A_pars, x, alpha, k, n);
    transition_mat_update2(B, B_pars, x, y, alpha, k, s, n);

    if((iter > burnin) && ((iter-1) % thin == 0)){
      index = (iter - burnin - 1)/thin;
      xx = as<IntegerVector>(wrap(x));
      xx.attr("dim") = R_NilValue;
      trace_x[index] = clone(xx);
      trace_pi[index] = clone(pi);
      trace_A[index] = clone(A);
      trace_B[index] = clone(B);
    }
    if(iter % 100 == 0) printf("iter %d\n", iter);
  }

  return List::create(Rcpp::Named("trace_x") = trace_x,
                      Rcpp::Named("trace_pi") = trace_pi,
                      Rcpp::Named("trace_A") = trace_A,
                      Rcpp::Named("trace_B") = trace_B);;
}

//' @export
// [[Rcpp::export]]
void swap_matrices(List x, int i, int j){
  NumericMatrix temp = x[i-1];
  x[i-1] = x[j-1];
  x[j-1] = temp;
}

//' @export
// [[Rcpp::export]]
void swap_vectors(List x, int i, int j){
  NumericVector temp = x[i-1];
  x[i-1] = x[j-1];
  x[j-1] = temp;
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

