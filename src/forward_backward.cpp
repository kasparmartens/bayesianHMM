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

// [[Rcpp::export]]
List forward_backward_fast(NumericVector pi, NumericMatrix A, NumericMatrix B, NumericVector y, int k, int n){
  List L(n);
  NumericVector prob, b;
  b = B(_, y[0]-1);
  L[0] = compute_P0(pi, b, k);
  for(int t=1; t<n; t++){
    NumericMatrix P = L[t-1];
    NumericVector colsums = calculate_colsums(P, k, k);
    b = B(_, y[t]-1);
    L[t] = compute_P(colsums, A, b, k);
  }
  // now backward sampling
  IntegerVector x_draw(n);
  IntegerVector possible_values = seq_len(k);
  prob = calculate_colsums(L[n-1], k, k);
  x_draw[n-1] = as<int>(RcppArmadillo::sample(possible_values, 1, false, prob));
  for(int t=n-1; t>0; t--){
    NumericMatrix temp = L[t];
    prob = temp(_, x_draw[t]-1);
    x_draw[t-1] = as<int>(RcppArmadillo::sample(possible_values, 1, false, prob));
  }
  return List::create(Rcpp::Named("x_draw") = x_draw,
                      Rcpp::Named("P") = L);;
}

