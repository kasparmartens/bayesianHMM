// This file was generated by Rcpp::compileAttributes
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// forward_backward_fast
List forward_backward_fast(NumericVector pi, NumericMatrix A, NumericMatrix B, NumericVector y, int k, int n);
RcppExport SEXP bayesianHMM_forward_backward_fast(SEXP piSEXP, SEXP ASEXP, SEXP BSEXP, SEXP ySEXP, SEXP kSEXP, SEXP nSEXP) {
BEGIN_RCPP
    Rcpp::RObject __result;
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< NumericVector >::type pi(piSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type A(ASEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type B(BSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    __result = Rcpp::wrap(forward_backward_fast(pi, A, B, y, k, n));
    return __result;
END_RCPP
}
// swap_matrices
void swap_matrices(List x, int i, int j);
RcppExport SEXP bayesianHMM_swap_matrices(SEXP xSEXP, SEXP iSEXP, SEXP jSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< List >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type i(iSEXP);
    Rcpp::traits::input_parameter< int >::type j(jSEXP);
    swap_matrices(x, i, j);
    return R_NilValue;
END_RCPP
}
// swap_vectors
void swap_vectors(List x, int i, int j);
RcppExport SEXP bayesianHMM_swap_vectors(SEXP xSEXP, SEXP iSEXP, SEXP jSEXP) {
BEGIN_RCPP
    Rcpp::RNGScope __rngScope;
    Rcpp::traits::input_parameter< List >::type x(xSEXP);
    Rcpp::traits::input_parameter< int >::type i(iSEXP);
    Rcpp::traits::input_parameter< int >::type j(jSEXP);
    swap_vectors(x, i, j);
    return R_NilValue;
END_RCPP
}
