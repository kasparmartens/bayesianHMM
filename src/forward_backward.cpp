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

NumericMatrix compute_P(NumericVector pi, NumericMatrix A, NumericVector b, int k){
  NumericMatrix P(k, k);
  double sum = 0;
  for(int s=0; s<k; s++){
    for(int r=0; r<k; r++){
      P(r, s) = pi[r] * A(r, s) * b[s];
      sum += P(r, s);
    }
  }
  normalise_mat(P, k, k, sum);
  return P;
}

NumericMatrix compute_P0(NumericVector pi, NumericVector b, int k){
  NumericMatrix P(k, k);
  double sum = 0;
  for(int s=0; s<k; s++){
    for(int r=0; r<k; r++){
      P(r, s) = pi[r] * b[s];
      sum += P(r, s);
    }
  }
  normalise_mat(P, k, k, sum);
  return P;
}

NumericMatrix compute_Q(NumericMatrix P, NumericVector pi_backward, NumericVector pi_forward, int k){
  NumericMatrix Q(k, k);
  for(int s=0; s<k; s++){
    if(pi_forward[s]>0){
      for(int r=0; r<k; r++){
        Q(r, s) = P(r, s) * pi_backward[s] / pi_forward[s];
      }
    }
  }
  return Q;
}


NumericVector calculate_colsums(NumericMatrix mat, int m, int n){
  double temp;
  NumericVector x(n);
  for(int j=0; j<n; j++){
    temp = 0;
    for(int i=0; i<m; i++){
      temp += mat(i, j);
    }
    x[j] = temp;
  }
  return x;
}

NumericVector calculate_rowsums(NumericMatrix mat, int m, int n){
  double temp;
  NumericVector x(m);
  for(int i=0; i<m; i++){
    temp = 0;
    for(int j=0; j<n; j++){
      temp += mat(i, j);
    }
    x[i] = temp;
  }
  return x;
}

// [[Rcpp::export]]
List forward_backward_fast(NumericVector pi, NumericMatrix A, NumericMatrix B, NumericVector y, int k, int n){
  List P(n), Q(n);
  NumericVector prob, b, q_forward, q_backward;
  b = B(_, y[0]-1);
  P[0] = compute_P0(pi, b, k);
  for(int t=1; t<n; t++){
    NumericMatrix Pt = P[t-1];
    NumericVector colsums = calculate_colsums(Pt, k, k);
    b = B(_, y[t]-1);
    P[t] = compute_P(colsums, A, b, k);
  }
  // now backward sampling
  IntegerVector x_draw(n);
  IntegerVector possible_values = seq_len(k);
  prob = calculate_colsums(P[n-1], k, k);
  x_draw[n-1] = as<int>(RcppArmadillo::sample(possible_values, 1, false, prob));
  for(int t=n-1; t>0; t--){
    NumericMatrix temp = P[t];
    prob = temp(_, x_draw[t]-1);
    x_draw[t-1] = as<int>(RcppArmadillo::sample(possible_values, 1, false, prob));
  }
  // and backward recursion to obtain marginal distributions
  Q[n-1] = P[n-1];
  for(int t=n-2; t>=0; t--){
    q_forward = calculate_colsums(P[t], k, k);
    q_backward = calculate_rowsums(Q[t+1], k, k);
    Q[t] = compute_Q(P[t], q_backward, q_forward, k);
  }
  return List::create(Rcpp::Named("x_draw") = x_draw,
                      Rcpp::Named("P") = P,
                      Rcpp::Named("Q") = Q);;
}

