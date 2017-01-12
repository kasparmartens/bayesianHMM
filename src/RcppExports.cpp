// Generated by using Rcpp::compileAttributes() -> do not edit by hand
// Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

#include <RcppArmadillo.h>
#include <Rcpp.h>

using namespace Rcpp;

// forward_backward_fast
List forward_backward_fast(NumericVector pi, NumericMatrix A, NumericMatrix B, IntegerVector y, int k, int n, bool marginal_distr);
RcppExport SEXP ensembleHMM_forward_backward_fast(SEXP piSEXP, SEXP ASEXP, SEXP BSEXP, SEXP ySEXP, SEXP kSEXP, SEXP nSEXP, SEXP marginal_distrSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type pi(piSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type A(ASEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type B(BSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< bool >::type marginal_distr(marginal_distrSEXP);
    rcpp_result_gen = Rcpp::wrap(forward_backward_fast(pi, A, B, y, k, n, marginal_distr));
    return rcpp_result_gen;
END_RCPP
}
// gibbs_sampling_fast_with_starting_vals
List gibbs_sampling_fast_with_starting_vals(NumericVector pi0, NumericMatrix A0, NumericMatrix B0, IntegerVector y, double alpha, int k, int s, int n, int max_iter, int burnin, int thin, bool marginal_distr, bool is_fixed_B);
RcppExport SEXP ensembleHMM_gibbs_sampling_fast_with_starting_vals(SEXP pi0SEXP, SEXP A0SEXP, SEXP B0SEXP, SEXP ySEXP, SEXP alphaSEXP, SEXP kSEXP, SEXP sSEXP, SEXP nSEXP, SEXP max_iterSEXP, SEXP burninSEXP, SEXP thinSEXP, SEXP marginal_distrSEXP, SEXP is_fixed_BSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< NumericVector >::type pi0(pi0SEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type A0(A0SEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type B0(B0SEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< int >::type s(sSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< int >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< int >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< bool >::type marginal_distr(marginal_distrSEXP);
    Rcpp::traits::input_parameter< bool >::type is_fixed_B(is_fixed_BSEXP);
    rcpp_result_gen = Rcpp::wrap(gibbs_sampling_fast_with_starting_vals(pi0, A0, B0, y, alpha, k, s, n, max_iter, burnin, thin, marginal_distr, is_fixed_B));
    return rcpp_result_gen;
END_RCPP
}
// gibbs_sampling_fast
List gibbs_sampling_fast(IntegerVector y, double alpha, int k, int s, int n, int max_iter, int burnin, int thin, bool marginal_distr, bool is_fixed_B);
RcppExport SEXP ensembleHMM_gibbs_sampling_fast(SEXP ySEXP, SEXP alphaSEXP, SEXP kSEXP, SEXP sSEXP, SEXP nSEXP, SEXP max_iterSEXP, SEXP burninSEXP, SEXP thinSEXP, SEXP marginal_distrSEXP, SEXP is_fixed_BSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< IntegerVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< int >::type s(sSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< int >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< int >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< bool >::type marginal_distr(marginal_distrSEXP);
    Rcpp::traits::input_parameter< bool >::type is_fixed_B(is_fixed_BSEXP);
    rcpp_result_gen = Rcpp::wrap(gibbs_sampling_fast(y, alpha, k, s, n, max_iter, burnin, thin, marginal_distr, is_fixed_B));
    return rcpp_result_gen;
END_RCPP
}
// ensemble_gaussian
List ensemble_gaussian(int n_chains, NumericVector y, double alpha, int k, int s, int n, int max_iter, int burnin, int thin, bool estimate_marginals, bool fixed_pars, bool parallel_tempering, bool crossovers, NumericVector temperatures, int swap_type, int swaps_burnin, int swaps_freq, NumericVector mu, NumericVector sigma2, IntegerVector which_chains, IntegerVector subsequence);
RcppExport SEXP ensembleHMM_ensemble_gaussian(SEXP n_chainsSEXP, SEXP ySEXP, SEXP alphaSEXP, SEXP kSEXP, SEXP sSEXP, SEXP nSEXP, SEXP max_iterSEXP, SEXP burninSEXP, SEXP thinSEXP, SEXP estimate_marginalsSEXP, SEXP fixed_parsSEXP, SEXP parallel_temperingSEXP, SEXP crossoversSEXP, SEXP temperaturesSEXP, SEXP swap_typeSEXP, SEXP swaps_burninSEXP, SEXP swaps_freqSEXP, SEXP muSEXP, SEXP sigma2SEXP, SEXP which_chainsSEXP, SEXP subsequenceSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n_chains(n_chainsSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< int >::type s(sSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< int >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< int >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< bool >::type estimate_marginals(estimate_marginalsSEXP);
    Rcpp::traits::input_parameter< bool >::type fixed_pars(fixed_parsSEXP);
    Rcpp::traits::input_parameter< bool >::type parallel_tempering(parallel_temperingSEXP);
    Rcpp::traits::input_parameter< bool >::type crossovers(crossoversSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type temperatures(temperaturesSEXP);
    Rcpp::traits::input_parameter< int >::type swap_type(swap_typeSEXP);
    Rcpp::traits::input_parameter< int >::type swaps_burnin(swaps_burninSEXP);
    Rcpp::traits::input_parameter< int >::type swaps_freq(swaps_freqSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type mu(muSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type sigma2(sigma2SEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type which_chains(which_chainsSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type subsequence(subsequenceSEXP);
    rcpp_result_gen = Rcpp::wrap(ensemble_gaussian(n_chains, y, alpha, k, s, n, max_iter, burnin, thin, estimate_marginals, fixed_pars, parallel_tempering, crossovers, temperatures, swap_type, swaps_burnin, swaps_freq, mu, sigma2, which_chains, subsequence));
    return rcpp_result_gen;
END_RCPP
}
// ensemble_discrete
List ensemble_discrete(int n_chains, IntegerVector y, double alpha, int k, int s, int n, int max_iter, int burnin, int thin, bool estimate_marginals, bool fixed_pars, bool parallel_tempering, bool crossovers, NumericVector temperatures, int swap_type, int swaps_burnin, int swaps_freq, NumericMatrix B, IntegerVector which_chains, IntegerVector subsequence);
RcppExport SEXP ensembleHMM_ensemble_discrete(SEXP n_chainsSEXP, SEXP ySEXP, SEXP alphaSEXP, SEXP kSEXP, SEXP sSEXP, SEXP nSEXP, SEXP max_iterSEXP, SEXP burninSEXP, SEXP thinSEXP, SEXP estimate_marginalsSEXP, SEXP fixed_parsSEXP, SEXP parallel_temperingSEXP, SEXP crossoversSEXP, SEXP temperaturesSEXP, SEXP swap_typeSEXP, SEXP swaps_burninSEXP, SEXP swaps_freqSEXP, SEXP BSEXP, SEXP which_chainsSEXP, SEXP subsequenceSEXP) {
BEGIN_RCPP
    Rcpp::RObject rcpp_result_gen;
    Rcpp::RNGScope rcpp_rngScope_gen;
    Rcpp::traits::input_parameter< int >::type n_chains(n_chainsSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type y(ySEXP);
    Rcpp::traits::input_parameter< double >::type alpha(alphaSEXP);
    Rcpp::traits::input_parameter< int >::type k(kSEXP);
    Rcpp::traits::input_parameter< int >::type s(sSEXP);
    Rcpp::traits::input_parameter< int >::type n(nSEXP);
    Rcpp::traits::input_parameter< int >::type max_iter(max_iterSEXP);
    Rcpp::traits::input_parameter< int >::type burnin(burninSEXP);
    Rcpp::traits::input_parameter< int >::type thin(thinSEXP);
    Rcpp::traits::input_parameter< bool >::type estimate_marginals(estimate_marginalsSEXP);
    Rcpp::traits::input_parameter< bool >::type fixed_pars(fixed_parsSEXP);
    Rcpp::traits::input_parameter< bool >::type parallel_tempering(parallel_temperingSEXP);
    Rcpp::traits::input_parameter< bool >::type crossovers(crossoversSEXP);
    Rcpp::traits::input_parameter< NumericVector >::type temperatures(temperaturesSEXP);
    Rcpp::traits::input_parameter< int >::type swap_type(swap_typeSEXP);
    Rcpp::traits::input_parameter< int >::type swaps_burnin(swaps_burninSEXP);
    Rcpp::traits::input_parameter< int >::type swaps_freq(swaps_freqSEXP);
    Rcpp::traits::input_parameter< NumericMatrix >::type B(BSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type which_chains(which_chainsSEXP);
    Rcpp::traits::input_parameter< IntegerVector >::type subsequence(subsequenceSEXP);
    rcpp_result_gen = Rcpp::wrap(ensemble_discrete(n_chains, y, alpha, k, s, n, max_iter, burnin, thin, estimate_marginals, fixed_pars, parallel_tempering, crossovers, temperatures, swap_type, swaps_burnin, swaps_freq, B, which_chains, subsequence));
    return rcpp_result_gen;
END_RCPP
}
