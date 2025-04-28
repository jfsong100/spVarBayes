#include <RcppEigen.h>
#include <RcppParallel.h>

using namespace Rcpp;
using namespace Eigen;

extern "C" SEXP nearest_psd_single(SEXP A_) {
  try {
    const Map<MatrixXd> A(as<Map<MatrixXd>>(A_));
    if (A.rows() != A.cols()) {
      stop("Matrix must be square.");
    }
    
    MatrixXd symA = 0.5 * (A + A.transpose());
    
    SelfAdjointEigenSolver<MatrixXd> eig(symA);
    if (eig.info() != Success) {
      stop("Eigen decomposition failed.");
    }
    
    VectorXd eigvals_clipped = eig.eigenvalues().cwiseMax(0.0);
    MatrixXd Q = eig.eigenvectors();
    MatrixXd A_psd = Q * eigvals_clipped.asDiagonal() * Q.transpose();
    return wrap(A_psd);
    
  } catch (std::exception &ex) {
    forward_exception_to_r(ex);
  } catch (...) {
    ::Rf_error("C++ exception");
  }
  return R_NilValue;
}


extern "C" SEXP fix_negative_eigvals_diag(SEXP A_) {
  try {
    const Map<MatrixXd> A(as<Map<MatrixXd>>(A_));
    int n = A.rows();
    if (n != A.cols()) stop("Matrix must be square.");
    
    MatrixXd symA = 0.5 * (A + A.transpose());
    
    SelfAdjointEigenSolver<MatrixXd> eig(symA);
    if (eig.info() != Success) stop("Eigen decomposition failed.");
    
    VectorXd eigvals = eig.eigenvalues();
    MatrixXd Q = eig.eigenvectors();
    
    bool updated = false;
    for (int i = 0; i < n; ++i) {
      if (eigvals(i) < 0.0) {
        eigvals(i) = 0.0;
        updated = true;
      }
    }
    
    if (!updated) return wrap(symA);
    
    MatrixXd corrected = Q * eigvals.asDiagonal() * Q.transpose();
    return wrap(corrected);
    
  } catch (std::exception &ex) {
    forward_exception_to_r(ex);
  } catch (...) {
    ::Rf_error("C++ exception (unknown reason)");
  }
  return R_NilValue;
}