#include "global.h"

using namespace Rcpp;

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

double calculate_nondiagonal_sum(NumericMatrix mat, int k){
  double sum=0;
  for(int j=0; j<k; j++){
    for(int i=0; i<k; i++){
      if(i != j) sum += mat(i, j);
    }
  }
  return sum;
}

NumericVector calculate_colsums(NumericMatrix A, int m, int n){
  arma::mat B(A.begin(), m, n, false);
  arma::rowvec colsums = sum(B, 0);
  NumericVector out(colsums.begin(), colsums.end());
  return out;
}

NumericVector calculate_rowsums(NumericMatrix A, int m, int n){
  arma::mat B(A.begin(), m, n, false);
  arma::colvec rowsums = sum(B, 1);
  NumericVector out(rowsums.begin(), rowsums.end());
  return out;
}

double normalise_mat(NumericMatrix A, int m, int n){
  // divide all elements of A by the sum of A
  arma::mat B(A.begin(), m, n, false);
  double sum = accu(B);
  B /= sum;
  return sum;
}


IntegerMatrix hamming_distance(NumericMatrix X, int n, int m){
  IntegerMatrix dist(n, n);
  int temp;
  for(int j=0; j<(n-1); j++){
    for(int i=j+1; i<n; i++){
      temp = 0;
      for(int t=0; t<m; t++){
        if(X(i, t) != X(j, t)){
          temp += 1;
        }
      }
      dist(i, j) = temp;
    }
  }
  return dist;
}

int myPow(int x, int p) {
  if (p == 0) return 1;
  if (p == 1) return x;
  return x * myPow(x, p-1);
}

double mylog(double x){
  return log(x + 1.0e-16);
}
